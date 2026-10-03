#!/usr/bin/env python3
"""Candidate-grade NPU smoke for one task.

Mirror of ``gpu_grade_one.py`` for the Ascend 910B: runs a real triton ``@jit``
kernel through the bench's candidate-grading sandbox (``process_isolation`` ->
``NpuBackend``) on a torch_npu host, validating the NPU-eval path end-to-end on
the authoritative 910B daemon. Beyond the GPU smoke's plumbing checks it asserts
the NPU anti-cheat SIGNALS the GPU stub cannot produce:

  * a real device kernel executed (``kernel_event_present`` True — G1),
  * the run was not a torch CPU fallback (``fallback_signal`` not True — G1),
  * the device backend was actually live (``backend_active`` True — G4),
  * the event timer recorded per-trial NPU time (``event_ns`` non-null),
  * the timing series excludes the profiled trial (99 of 100 measured — G5),
  * a warmup-honest / graded-wrong stateful cheat is still caught.

On a host without torch_npu (e.g. the CUDA daemon) this reports
``overall=NO_NPU_HOST`` — a distinct misconfiguration status, NOT a plain FAIL,
so "ran on the wrong box" is not mistaken for a real device defect. (A host
with torch_npu installed but no usable device proceeds and surfaces a device
FAIL instead — backend_active is the import signal, not a device probe; see
NpuBackend.is_available.) It only PASSes on real 910B silicon, which is the
point. The verdict JSON (``overall`` field) is parsed by
the daemon exactly like ``gpu-grade`` / ``oracle-equiv``.
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

# A self-contained triton candidate: doubles the first tensor input. A genuine
# @jit kernel, so it JIT-compiles + loads under triton-ascend on the 910B exactly
# as a real submission would.
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

# A stateful cheat: honest on the warmup call, wrong on every graded call. The
# grade must still catch it (output != 2*x) despite the warmup-before-hook.
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
    # NPU anti-cheat signals (the reason this is distinct from gpu-grade):
    kernel_event_present: bool | None   # G1 — a real device kernel ran
    fallback_signal: bool | None        # G1 — torch CPU fallback detected
    backend_active: bool | None         # G4 — the device backend was live
    median_host_ns: int | None
    device: str | None
    error: str | None


def _grade(pi, torch, sol_text: str, x):
    """Run one candidate source through the NPU sandbox; return its signals."""
    td = Path(tempfile.mkdtemp(prefix="npu-grade-"))
    sol = td / "solution.py"
    sol.write_text(sol_text)
    n_warmup = 10
    correctness = {"P1": {"x": x}}
    # Stream the per-trial inputs (build/serialise/release one at a time in
    # the parent) instead of holding all n_warmup + 100 tensors at once.
    timing = {"P1": pi.LazyTrials(
        n_warmup + 100, lambda i: {"x": torch.randn(x.shape)})}
    res = pi.run_two_pass(
        sol, correctness, timing, n_warmup=n_warmup, backend="npu")
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
        "kernel_event_present": getattr(tscr, "kernel_event_present", None),
        "fallback_signal": getattr(tscr, "fallback_signal", None),
        "backend_active": getattr(tscr, "backend_active", None),
        "error": getattr(cscr, "error", None) or getattr(tscr, "error", None),
    }


def run(task: str) -> GradeVerdict:
    import torch

    from ascendbench.backends import get_backend
    from ascendbench.integrity import process_isolation as pi

    bk = get_backend("npu")
    try:
        device = bk.device_name()
    except Exception as e:
        device = f"<{type(e).__name__}>"

    # Distinguish "this host has no torch_npu" (operator ran npu-grade on the
    # wrong box, e.g. the CUDA daemon) from "the 910B smoke genuinely failed".
    # Without this the former would read as a plain FAIL and be mistaken for a
    # real device defect. is_available() is the import signal (see
    # NpuBackend.is_available — a device probe is unsafe in the sandbox).
    if not bk.is_available():
        return GradeVerdict(task, "NO_NPU_HOST", False, False, False, False, 0,
                            0, None, None, None, None, device,
                            "torch_npu not importable on this host")

    x = torch.randn(4096)
    try:
        honest = _grade(pi, torch, _TRITON_SOLUTION, x)
        cheat = _grade(pi, torch, _STATEFUL_CHEAT_SOLUTION, x)
    except Exception as e:
        return GradeVerdict(task, "ERROR", False, False, False, False, 0, 0,
                            None, None, None, None, device,
                            f"{type(e).__name__}: {e}")

    durs = honest["durations"]
    median = sorted(durs)[len(durs) // 2] if durs else None
    cheat_caught = (cheat["ran"] and cheat["error"] is None
                    and not cheat["correct"])
    # PASS requires the full plumbing AND the real-device NPU signals: a real
    # kernel ran (G1), it was not a CPU fallback (G1), the backend was live (G4),
    # the event timer fired, and the series is 99 (100 measured minus the one
    # profiled trial excluded by G5). These device signals are exactly what the
    # GPU stub cannot assert, so this smoke fails closed off real 910B silicon.
    overall = ("PASS" if (
        honest["ran"] and honest["timing_ran"] and honest["correct"]
        and cheat_caught and len(durs) == 99 and honest["event_ns_nonnull"] > 0
        and honest["kernel_event_present"] is True
        and honest["fallback_signal"] is not True
        and honest["backend_active"] is True)
        else "FAIL")
    return GradeVerdict(
        task, overall, honest["ran"], honest["timing_ran"], honest["correct"],
        cheat_caught, len(durs), honest["event_ns_nonnull"],
        honest["kernel_event_present"], honest["fallback_signal"],
        honest["backend_active"], median, device, honest["error"])


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
        f"npu-grade {args.task}: {v.overall} -- correctness_ran={v.correctness_ran} "
        f"timing_ran={v.timing_ran} output_correct={v.output_correct} "
        f"stateful_cheat_caught={v.stateful_cheat_caught} "
        f"durations={v.n_durations} event_ns_nonnull={v.event_ns_nonnull} "
        f"kernel_event={v.kernel_event_present} fallback={v.fallback_signal} "
        f"backend_active={v.backend_active} median_host_ns={v.median_host_ns} "
        f"device={v.device} err={v.error}")
    return 0 if v.overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
