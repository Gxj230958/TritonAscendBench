#!/usr/bin/env python3
"""GPU-side R10 stage-2: run kernel.py on each shape and compare against
the pre-computed PyTorch reference produced by oracle_pyref_precompute.py.

For each task with provenance.golden.oracle_pyref_status == "ALIGNED_READY":
  1. Load tasks/<id>/r10_pyref/inputs_<sid>.safetensors  (CPU view)
  2. Load tasks/<id>/r10_pyref/ref_<sid>.safetensors      (CPU view)
  3. Move inputs to GPU, run kernel.py upstream entrypoint
  4. Move kernel output to CPU, compare against ref via tolerance.compare
  5. Emit /tmp/r10_verdict/<task_id>.json with per-shape PASS/FAIL +
     numeric diff (max_abs, max_rel) per output tensor.

Manual operator workflow (V9 M2.1b):
  ssh <gpu-host>
  cd /path/to/triton-ascend-bench && git pull
  bash scripts/gpu_pipeline/oracle_equivalence_verify.sh --task-list <task1,task2,...>
  # OR: bash scripts/gpu_pipeline/oracle_equivalence_verify.sh --all
  # then rsync /tmp/r10_verdict/ back to dev machine, run apply script.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
from safetensors.torch import load_file
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from triton_ascend_bench.tolerance import ToleranceClass, compare  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("oracle_equivalence_gpu_runner")


@dataclass
class ShapeVerdict:
    shape_id: str
    status: str = ""  # VERIFIED | FAILED | LAUNCH_BLOCKED | TOLERANCE_BLOCKED
    max_abs_diff: dict[str, float] = field(default_factory=dict)
    max_rel_diff: dict[str, float] = field(default_factory=dict)
    blocker_reason: str = ""
    output_keys_ref: list[str] = field(default_factory=list)
    output_keys_kernel: list[str] = field(default_factory=list)


@dataclass
class TaskVerdict:
    task_id: str
    overall: str = ""  # VERIFIED | FAILED | MIXED | LAUNCH_BLOCKED | SKIPPED
    shapes: list[ShapeVerdict] = field(default_factory=list)
    elapsed_s: float = 0.0


def _resolve_kernel_entrypoint(meta: dict, kernel_path: Path, task_id: str) -> str | None:
    """Mirror logic in oracle_pyref_precompute.py for consistency."""
    gpu_triton = (meta.get("oracle") or {}).get("gpu_triton", "") or ""
    if isinstance(gpu_triton, str) and "::" in gpu_triton:
        _, _, fn = gpu_triton.partition("::")
        if fn and fn != "placeholder":
            return fn
    # Load resolve cache
    rj = REPO_ROOT / "results" / "resolve_all.json"
    if rj.exists():
        try:
            d = json.loads(rj.read_text())
            for v in d.get("verdicts", []):
                if v.get("task_id") == task_id:
                    tried = v.get("tried") or []
                    if tried:
                        return tried[0]
        except Exception:
            pass
    meta_name = meta.get("name", "")
    if meta_name.endswith("_kernel"):
        meta_name = meta_name[:-len("_kernel")].lstrip("_")
    return meta_name or None


def _import_kernel_function(kernel_path: Path, fn_name: str, infer_root: Path):
    """Import kernel.py and return the host-wrapper function fn_name.

    The kernel.py may import from vllm/sglang;ensure infer/ is on sys.path
    via the existing setup_infer_paths shim.
    """
    # Make infer/{vllm,sglang} importable
    if (infer_root / "vllm").exists():
        sys.path.insert(0, str(infer_root / "vllm"))
    if (infer_root / "sglang" / "python").exists():
        sys.path.insert(0, str(infer_root / "sglang" / "python"))

    import importlib.util as iu

    spec = iu.spec_from_file_location(f"kernel_{kernel_path.parent.name}", str(kernel_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import kernel.py at {kernel_path}")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise AttributeError(f"function `{fn_name}` not found in kernel.py")
    return fn


def _materialise_kernel_optional_params(
    kernel_fn, provided_keys: set[str]
) -> dict[str, object]:
    """For kernel optional params not in inputs, return their defaults."""
    import inspect

    sig = inspect.signature(kernel_fn)
    extras: dict[str, object] = {}
    for name, param in sig.parameters.items():
        if name in provided_keys:
            continue
        if param.default is not inspect.Parameter.empty:
            extras[name] = param.default
    return extras


def _verify_one_task(
    task_dir: Path, infer_root: Path, gpu_device: str = "cuda"
) -> TaskVerdict:
    task_id = task_dir.name
    t0 = time.time()
    verdict = TaskVerdict(task_id=task_id)

    ref_dir = task_dir / "r10_pyref"
    if not ref_dir.is_dir() or not list(ref_dir.glob("ref_*.safetensors")):
        verdict.overall = "SKIPPED"
        verdict.elapsed_s = time.time() - t0
        log.info("  %s: SKIPPED (no r10_pyref artifacts)", task_id)
        return verdict

    meta = yaml.safe_load((task_dir / "meta.yaml").read_text())
    kernel_path = task_dir / "kernel.py"
    fn_name = _resolve_kernel_entrypoint(meta, kernel_path, task_id)
    if not fn_name:
        verdict.overall = "LAUNCH_BLOCKED"
        log.info("  %s: cannot resolve kernel entrypoint", task_id)
        return verdict

    try:
        kernel_fn = _import_kernel_function(kernel_path, fn_name, infer_root)
    except Exception as e:
        verdict.overall = "LAUNCH_BLOCKED"
        v = ShapeVerdict(shape_id="*", status="LAUNCH_BLOCKED", blocker_reason=str(e))
        verdict.shapes.append(v)
        log.warning("  %s: import fail: %s", task_id, e)
        return verdict

    tolerance_class_name = meta.get("tolerance_class", "vector_reduce")
    tol = ToleranceClass.from_name(tolerance_class_name)

    n_verified = 0
    n_failed = 0
    n_blocked = 0
    for ref_file in sorted(ref_dir.glob("ref_*.safetensors")):
        sid = ref_file.stem.replace("ref_", "")
        inputs_file = ref_dir / f"inputs_{sid}.safetensors"
        sv = ShapeVerdict(shape_id=sid)

        try:
            ref_tensors = load_file(str(ref_file))
            input_tensors_cpu = load_file(str(inputs_file)) if inputs_file.exists() else {}
        except Exception as e:
            sv.status = "LAUNCH_BLOCKED"
            sv.blocker_reason = f"safetensors load: {e}"
            n_blocked += 1
            verdict.shapes.append(sv)
            continue

        try:
            # Move inputs to GPU; keep scalars as-is
            inputs_gpu: dict[str, object] = {}
            for k, v in input_tensors_cpu.items():
                if isinstance(v, torch.Tensor):
                    inputs_gpu[k] = v.to(gpu_device)
                else:
                    inputs_gpu[k] = v

            # Provide kernel optional defaults if not in inputs
            extras = _materialise_kernel_optional_params(kernel_fn, set(inputs_gpu.keys()))
            for k, v in extras.items():
                inputs_gpu.setdefault(k, v)

            kernel_out = kernel_fn(**inputs_gpu)
        except Exception as e:
            sv.status = "LAUNCH_BLOCKED"
            sv.blocker_reason = f"{type(e).__name__}: {e}"
            n_blocked += 1
            verdict.shapes.append(sv)
            log.warning("  %s/%s: kernel launch fail: %s", task_id, sid, e)
            continue

        # Normalise kernel_out to dict
        if isinstance(kernel_out, dict):
            out_dict = kernel_out
        elif isinstance(kernel_out, torch.Tensor):
            # Single tensor → name it by first ref key
            if len(ref_tensors) == 1:
                key = list(ref_tensors.keys())[0]
                out_dict = {key: kernel_out}
            else:
                out_dict = {"output": kernel_out}
        elif isinstance(kernel_out, (tuple, list)):
            ref_keys = list(ref_tensors.keys())
            out_dict = {
                ref_keys[i] if i < len(ref_keys) else f"out_{i}": v
                for i, v in enumerate(kernel_out)
            }
        else:
            sv.status = "FAILED"
            sv.blocker_reason = f"kernel returned non-tensor type: {type(kernel_out)}"
            n_failed += 1
            verdict.shapes.append(sv)
            continue

        # Move kernel outputs to CPU for compare
        out_dict_cpu = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in out_dict.items()
        }
        sv.output_keys_kernel = sorted(out_dict_cpu.keys())
        sv.output_keys_ref = sorted(ref_tensors.keys())

        # Compare each matched key
        all_pass = True
        for key in sorted(set(sv.output_keys_ref) & set(sv.output_keys_kernel)):
            ref_t = ref_tensors[key].float()
            ker_t = out_dict_cpu[key].float() if isinstance(out_dict_cpu[key], torch.Tensor) else None
            if ker_t is None:
                continue
            try:
                diff = (ref_t - ker_t).abs()
                max_abs = float(diff.max().item())
                denom = ref_t.abs().clamp_min(1e-12)
                max_rel = float((diff / denom).max().item())
                sv.max_abs_diff[key] = max_abs
                sv.max_rel_diff[key] = max_rel
                if not compare(ref_t, ker_t, tol):
                    all_pass = False
            except Exception as e:
                sv.blocker_reason = f"compare error on {key}: {e}"
                all_pass = False

        # Verdict
        if sv.output_keys_ref != sv.output_keys_kernel:
            sv.status = "FAILED"
            sv.blocker_reason = (
                f"structure mismatch: ref={sv.output_keys_ref} "
                f"kernel={sv.output_keys_kernel}"
            )
            n_failed += 1
        elif all_pass:
            sv.status = "VERIFIED"
            n_verified += 1
        else:
            sv.status = "FAILED"
            n_failed += 1

        verdict.shapes.append(sv)
        log.info(
            "  %s/%s: %s (max_abs=%.3e max_rel=%.3e)",
            task_id,
            sid,
            sv.status,
            max(sv.max_abs_diff.values()) if sv.max_abs_diff else 0.0,
            max(sv.max_rel_diff.values()) if sv.max_rel_diff else 0.0,
        )

    if n_blocked == len(verdict.shapes):
        verdict.overall = "LAUNCH_BLOCKED"
    elif n_failed == 0 and n_verified > 0:
        verdict.overall = "VERIFIED"
    elif n_verified == 0 and n_failed > 0:
        verdict.overall = "FAILED"
    else:
        verdict.overall = "MIXED"

    verdict.elapsed_s = time.time() - t0
    return verdict


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--task", help="single task id")
    p.add_argument("--task-list", help="comma-separated list (or 'all')")
    p.add_argument("--all", action="store_true", help="all tasks with r10_pyref artifacts")
    p.add_argument("--gpu", default="cuda", help="torch device for kernel run")
    p.add_argument(
        "--infer-root",
        type=Path,
        default=REPO_ROOT.parent / "infer",
        help="path to infer/{vllm,sglang} sibling directory",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/r10_verdict"),
        help="where to write per-task verdict JSON",
    )
    args = p.parse_args(argv)

    # Resolve task list
    task_dirs: list[Path] = []
    if args.task:
        for d in (REPO_ROOT / "tasks").glob(f"level*/{args.task}"):
            task_dirs.append(d)
    elif args.task_list and args.task_list != "all":
        wanted = set(args.task_list.split(","))
        for level_dir in (REPO_ROOT / "tasks").glob("level*"):
            for td in level_dir.iterdir():
                if td.is_dir() and td.name in wanted:
                    task_dirs.append(td)
    else:
        # All tasks with r10_pyref artifacts
        for level_dir in (REPO_ROOT / "tasks").glob("level*"):
            for td in level_dir.iterdir():
                if td.is_dir() and (td / "r10_pyref").is_dir():
                    if list((td / "r10_pyref").glob("ref_*.safetensors")):
                        task_dirs.append(td)

    if not task_dirs:
        log.error("no task with r10_pyref artifacts found")
        return 2

    if not args.infer_root.is_dir():
        log.error("infer root does not exist: %s", args.infer_root)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "verifying %d task(s) on %s (infer=%s, output=%s)",
        len(task_dirs),
        args.gpu,
        args.infer_root,
        args.output_dir,
    )

    summary = {"verified": 0, "failed": 0, "mixed": 0, "blocked": 0, "skipped": 0}
    for td in sorted(task_dirs):
        v = _verify_one_task(td, args.infer_root, args.gpu)
        verdict_path = args.output_dir / f"{td.name}.json"
        verdict_path.write_text(json.dumps(asdict(v), indent=2, default=str))

        key = {
            "VERIFIED": "verified",
            "FAILED": "failed",
            "MIXED": "mixed",
            "LAUNCH_BLOCKED": "blocked",
            "SKIPPED": "skipped",
        }.get(v.overall, "blocked")
        summary[key] += 1

    log.info("--- summary ---")
    for k, n in summary.items():
        log.info("  %s: %d", k, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
