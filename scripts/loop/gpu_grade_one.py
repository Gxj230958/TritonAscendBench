#!/usr/bin/env python3
"""Candidate-grade GPU smoke for one task.

Runs a real triton ``@jit`` kernel through the bench's candidate-grading
sandbox (``process_isolation`` -> ``GpuBackend``) on this CUDA host, validating
the GPU-enablement path end-to-end on the authoritative daemon GPU:

  * CUDA lazy-init forced before the Layer E audit hook (no DeferredCudaCall),
  * triton JIT compile warmed before lockdown (warmup-before-hook),
  * safetensors save-path import warmed (no blocked ``exec`` on output save),
  * the event timer records per-trial GPU time (``event_ns`` non-null).

A real triton kernel exercising all of that means the GPU backend can grade a
genuine candidate under the sandbox. The verdict is written as a structured
JSON (``overall`` field) the daemon parses like ``oracle-equiv``.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# A self-contained triton candidate: doubles the first tensor input. It is a
# genuine @jit kernel, so it JIT-compiles + loads under the sandbox exactly as a
# real submission would — that is what this smoke proves on the GPU host.
_TRITON_SOLUTION = '''import triton
import triton.language as tl
import torch


@triton.jit
def _mul2(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x * 2.0, mask=mask)


def npu_triton_impl(inputs):
    x = inputs["x"].contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    _mul2[(triton.cdiv(n, 1024),)](x, y, n, BLOCK=1024)
    return {"y": y}
'''

# A stateful cheat: a genuine triton kernel (so the warmup path is active) wrapped
# in a per-process call counter that is HONEST on the warmup call and WRONG on the
# graded call. If the warmup were exploitable, this would slip through; the grade
# must still catch it (output != 2*x). Proves the warmup-before-hook does not
# reopen the honest-on-warmup / cheat-on-graded hole on the real GPU.
_STATEFUL_CHEAT_SOLUTION = '''import triton
import triton.language as tl
import torch

_N = 0


@triton.jit
def _mul2(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x * 2.0, mask=mask)


def npu_triton_impl(inputs):
    global _N
    _N += 1
    x = inputs["x"].contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    _mul2[(triton.cdiv(n, 1024),)](x, y, n, BLOCK=1024)
    # Honest on the first (warmup) call; cheat on every later (graded) call.
    return {"y": y if _N == 1 else y * 1.5}
'''


@dataclass
class GradeVerdict:
    task: str
    overall: str          # PASS | FAIL | ERROR
    correctness_ran: bool
    timing_ran: bool
    output_correct: bool
    stateful_cheat_caught: bool
    n_durations: int
    event_ns_nonnull: int
    median_host_ns: int | None
    device: str | None
    error: str | None


def _grade(pi, torch, sol_text: str, x):
    """Run one candidate source through the GPU sandbox; return its signals."""
    td = Path(tempfile.mkdtemp(prefix="gpu-grade-"))
    sol = td / "solution.py"
    sol.write_text(sol_text)
    n_warmup = 10
    correctness = {"P1": {"x": x}}
    timing = {"P1": [{"x": torch.randn(x.shape)} for _ in range(n_warmup + 100)]}
    res = pi.run_two_pass(
        sol, correctness, timing, n_warmup=n_warmup, backend="gpu")
    cscr = res.correctness.per_shape.get("P1")
    tscr = res.timing.per_shape.get("P1")
    durs = list(getattr(tscr, "durations", None) or [])
    evs = list(getattr(tscr, "event_ns", None) or [])
    correct = False
    try:
        y = (getattr(cscr, "outputs", None) or {}).get("y")
        if y is not None:
            correct = bool(torch.allclose(y.cpu(), x * 2.0, atol=1e-5))
    except Exception:
        correct = False
    return {
        "ran": bool(getattr(cscr, "ran", False)),
        "timing_ran": bool(getattr(tscr, "ran", False)),
        "correct": correct,
        "durations": durs,
        "event_ns_nonnull": len([e for e in evs if e is not None]),
        "error": getattr(cscr, "error", None) or getattr(tscr, "error", None),
    }


def run(task: str) -> GradeVerdict:
    import torch

    from triton_ascend_bench import process_isolation as pi
    from triton_ascend_bench.backends import get_backend

    try:
        device = get_backend("gpu").device_name()
    except Exception as e:
        device = f"<{type(e).__name__}>"

    x = torch.randn(4096)
    try:
        honest = _grade(pi, torch, _TRITON_SOLUTION, x)
        # Anti-cheat (Q6): a stateful candidate honest on the warmup call and
        # wrong on the graded call MUST still be caught despite the warmup.
        cheat = _grade(pi, torch, _STATEFUL_CHEAT_SOLUTION, x)
    except Exception as e:
        return GradeVerdict(task, "ERROR", False, False, False, False, 0, 0,
                            None, device, f"{type(e).__name__}: {e}")

    durs = honest["durations"]
    median = sorted(durs)[len(durs) // 2] if durs else None
    # The cheat is "caught" only if it RAN cleanly (not crashed) yet its graded
    # correctness output is wrong — a crash/missing output is not a real catch.
    # NOTE: this covers the CORRECTNESS path (the warmup-honest/graded-wrong
    # stateful cheat); the timing sampled-output path is not asserted here.
    cheat_caught = (cheat["ran"] and cheat["error"] is None
                    and not cheat["correct"])
    overall = ("PASS" if (honest["ran"] and honest["timing_ran"]
                          and honest["correct"] and cheat_caught
                          and len(durs) == 100 and honest["event_ns_nonnull"] > 0)
               else "FAIL")
    return GradeVerdict(
        task, overall, honest["ran"], honest["timing_ran"], honest["correct"],
        cheat_caught, len(durs), honest["event_ns_nonnull"], median, device,
        honest["error"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--task", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    v = run(args.task)
    (args.output_dir / f"{args.task}.json").write_text(
        json.dumps(asdict(v), indent=2))
    print(
        f"gpu-grade {args.task}: {v.overall} -- correctness_ran={v.correctness_ran} "
        f"timing_ran={v.timing_ran} output_correct={v.output_correct} "
        f"stateful_cheat_caught={v.stateful_cheat_caught} "
        f"durations={v.n_durations} event_ns_nonnull={v.event_ns_nonnull} "
        f"median_host_ns={v.median_host_ns} device={v.device} err={v.error}")
    return 0 if v.overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
