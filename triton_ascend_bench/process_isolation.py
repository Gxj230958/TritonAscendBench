"""process_isolation.py — run a candidate solution in an isolated process.

Implements the dev-machine-runnable subset of specs/process_isolation.md.
The full three-process Docker/seccomp/cgroup/NPU-device model belongs on
the GPU/NPU training machine; here we enforce the security-critical,
host-independent properties:

  * **Process isolation** — the candidate runs in a separate Python
    subprocess (no shared interpreter state / module cache with the
    trusted oracle), killable via timeout + SIGKILL of the process group.
  * **No golden-output leak (R11-P0-1)** — the child is given ONLY the
    task dir + solution path; it rebuilds inputs itself with the same
    seeded `inputs.build_inputs`, so it sees inputs but never the
    oracle's expected outputs.
  * **safetensors-only IPC (R12-P0-1)** — the child serialises its
    output dict with `safetensors.torch.save_file`; the parent reads it
    with `load_file` (data-only, no pickle → no RCE on load).
  * **Timeout** — a hung/over-long candidate is terminated.

NOT enforced here (GPU-machine layer, documented in the spec):
seccomp/cgroup/Docker, NPU device allowlist, two-process correctness vs
timing split, trusted-shim alias/data_ptr checks. Timing is therefore
not measured on this machine.

Candidate entrypoint convention (per spec §Trusted Shim): the solution
exposes `npu_triton_impl(inputs: dict) -> dict[str, Tensor]`, matching
the oracle's `run(inputs) -> dict`.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ShapeCandidateResult:
    shape_id: str
    ran: bool
    outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    error: str | None = None


@dataclass
class CandidateRunResult:
    ran_any: bool
    timed_out: bool
    load_error: str | None
    per_shape: dict[str, ShapeCandidateResult] = field(default_factory=dict)


# Standalone child script. Imports the candidate, rebuilds inputs with the
# SAME seed the trusted oracle uses, calls npu_triton_impl per shape, and
# serialises each output dict to safetensors. Never imports the oracle.
_CHILD = r'''
import json, sys, importlib.util
from pathlib import Path
import torch
from safetensors.torch import save_file

args = json.loads(sys.argv[1])
repo_root = args["repo_root"]
sys.path.insert(0, repo_root)
from triton_ascend_bench.inputs import build_inputs  # noqa: E402

import yaml  # noqa: E402
shapes_doc = yaml.safe_load(Path(args["shapes_yaml"]).read_text())

spec = importlib.util.spec_from_file_location("_candidate", args["solution"])
status = {"shapes": {}, "load_error": None}
try:
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "npu_triton_impl", None)
    if not callable(fn):
        status["load_error"] = "solution.py exposes no npu_triton_impl(inputs)"
        fn = None
except Exception as e:  # noqa: BLE001
    status["load_error"] = f"{type(e).__name__}: {e}"
    fn = None

out_dir = Path(args["out_dir"])
for sid in args["shape_ids"]:
    rec = {"ran": False, "error": None, "keys": []}
    if fn is None:
        rec["error"] = "no entrypoint"
        status["shapes"][sid] = rec
        continue
    try:
        inp = build_inputs(shapes_doc, sid, seed=args["seed"])
        out = fn(inp)
        if not isinstance(out, dict):
            rec["error"] = f"npu_triton_impl returned {type(out).__name__}, expected dict"
            status["shapes"][sid] = rec
            continue
        flat = {}
        for k, v in out.items():
            if not torch.is_tensor(v):
                rec["error"] = f"output[{k!r}] is {type(v).__name__}, not a Tensor"
                break
            flat[k] = v.detach().contiguous().cpu()
        else:
            save_file(flat, str(out_dir / f"{sid}.safetensors"))
            rec["ran"] = True
            rec["keys"] = list(flat.keys())
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    status["shapes"][sid] = rec

Path(args["status_json"]).write_text(json.dumps(status))
'''


def run_candidate_isolated(
    task_dir: Path,
    solution_path: Path,
    shape_ids: list[str],
    seed: int = 0,
    timeout_s: int = 90,
) -> CandidateRunResult:
    """Spawn the candidate in an isolated subprocess; return its per-shape
    outputs (loaded from safetensors). Never executes candidate code in
    this (trusted) process."""
    # Absolute paths — the child runs with cwd set to a scratch tmpdir.
    task_dir = Path(task_dir).resolve()
    solution_path = Path(solution_path).resolve()
    shapes_yaml = task_dir / "shapes.yaml"
    if not shapes_yaml.exists():
        return CandidateRunResult(False, False, "no shapes.yaml")

    with tempfile.TemporaryDirectory(prefix="tab-cand-") as td:
        tdp = Path(td)
        child_py = tdp / "_child.py"
        child_py.write_text(_CHILD)
        status_json = tdp / "status.json"
        args = {
            "repo_root": str(REPO_ROOT),
            "shapes_yaml": str(shapes_yaml),
            "solution": str(solution_path),
            "out_dir": str(tdp),
            "status_json": str(status_json),
            "shape_ids": shape_ids,
            "seed": seed,
        }
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            # No network creds; minimal env. (Full seccomp/cgroup is the
            # GPU-machine layer — see module docstring.)
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        timed_out = False
        try:
            subprocess.run(
                [sys.executable, str(child_py), json.dumps(args)],
                timeout=timeout_s,
                capture_output=True,
                cwd=td,
                env=env,
                start_new_session=True,  # own process group → killable
            )
        except subprocess.TimeoutExpired:
            timed_out = True

        if not status_json.exists():
            return CandidateRunResult(
                False, timed_out,
                "candidate produced no status (crash or timeout)")

        status = json.loads(status_json.read_text())
        res = CandidateRunResult(
            ran_any=False, timed_out=timed_out,
            load_error=status.get("load_error"))
        for sid in shape_ids:
            rec = status.get("shapes", {}).get(sid, {})
            scr = ShapeCandidateResult(
                shape_id=sid, ran=bool(rec.get("ran")),
                error=rec.get("error"))
            if scr.ran:
                st_file = tdp / f"{sid}.safetensors"
                if st_file.exists():
                    scr.outputs = load_file(str(st_file))
                    res.ran_any = True
                else:
                    scr.ran = False
                    scr.error = "status.ran but safetensors missing"
            res.per_shape[sid] = scr
        return res


def kill_process_group(pgid: int) -> None:
    """Best-effort SIGKILL of a candidate process group (spec §SIGKILL)."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
