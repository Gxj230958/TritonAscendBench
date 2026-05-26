#!/usr/bin/env python3
"""Dev-side R10 stage-1: oracle PyTorch reference precompute.

For each task:
  1. Static signature audit: compare oracle_pytorch.py `inputs[...]` keys
     against kernel.py upstream entrypoint required params AND
     shapes.yaml declared inputs.
  2. If ALIGNED: build_inputs on CPU (deterministic seed) + run
     oracle_pytorch.run(inputs) on CPU + save reference tensors as
     `tasks/<id>/r10_pyref/{inputs_<sid>.safetensors, ref_<sid>.safetensors}`.
  3. Emit per-task oracle_pyref_status into a report json + audit md.

This script is V9 architectural-shift unit M2.1a (dev / CPU portion of
R10 GPU equivalence verification). It produces artifacts the GPU-side
operator script (gpu_pipeline/25_oracle_equivalence_verify.sh) will
consume in stage 2 (M2.1b, manual operator run).

Status enum (closed):
  ALIGNED_READY        — oracle ran OK on CPU, artifacts written
  SIGNATURE_MISMATCH   — oracle / kernel / shapes inputs disagree on keys
  NEEDS_BRIDGE         — oracle and kernel have different input layouts
                         (e.g. oracle takes splits, kernel takes packed)
                         and require a build_inputs derivation bridge
  NEEDS_GPU_FOR_ORACLE — oracle itself imports libdevice / Triton ops
                         that cannot run under CPU; defer to GPU stage
  RUNTIME_ERROR        — oracle import or call raised an exception
  KERNEL_PARSE_ERROR   — kernel.py AST parse / entrypoint detection failed
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from triton_ascend_bench.inputs import build_inputs  # noqa: E402

GOLDEN_SEED = 0
DEFAULT_SHAPES = ["P1", "P2"]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("oracle_pyref_precompute")


@dataclass
class TaskAudit:
    task_id: str
    level: str
    status: str = ""
    oracle_input_keys: list[str] = field(default_factory=list)
    oracle_output_keys: list[str] = field(default_factory=list)
    kernel_required_params: list[str] = field(default_factory=list)
    kernel_optional_params: list[str] = field(default_factory=list)
    shapes_input_keys: list[str] = field(default_factory=list)
    missing_in_shapes: list[str] = field(default_factory=list)
    missing_in_oracle: list[str] = field(default_factory=list)
    extra_in_oracle: list[str] = field(default_factory=list)
    suggested_fix: str = ""
    ref_artifacts: dict[str, str] = field(default_factory=dict)  # shape_id -> sha256
    runtime_error: str = ""


def _oracle_input_output_keys(oracle_path: Path) -> tuple[list[str], list[str]]:
    """Static AST: find `inputs["X"]` accesses + `run()` return dict keys."""
    if not oracle_path.exists():
        return [], []
    src = oracle_path.read_text()
    input_keys = sorted(
        set(re.findall(r'inputs\[["\']([^"\']+)["\']\]', src))
        | set(re.findall(r'inputs\.get\(["\']([^"\']+)["\']', src))
    )
    output_keys: list[str] = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return input_keys, []
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run"
        ):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Return) and isinstance(inner.value, ast.Dict):
                    for k in inner.value.keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            output_keys.append(k.value)
            break
    return input_keys, sorted(set(output_keys))


def _kernel_entrypoint_signature(
    kernel_path: Path, fn_name: str
) -> tuple[list[str], list[str]] | None:
    """AST parse kernel.py; find function fn_name; return (required, optional)."""
    if not kernel_path.exists():
        return None
    try:
        tree = ast.parse(kernel_path.read_text())
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == fn_name
        ):
            args = [a.arg for a in node.args.args]
            defaults = node.args.defaults
            n_req = len(args) - len(defaults)
            req = args[:n_req]
            opt = args[n_req:]
            if req and req[0] == "self":
                req = req[1:]
            return (req, opt)
    return None


def _shapes_input_keys(shapes_path: Path) -> tuple[list[str], list[str]]:
    """Read shapes.yaml; return (input_keys, size_knob_keys)."""
    if not shapes_path.exists():
        return ([], [])
    try:
        d = yaml.safe_load(shapes_path.read_text())
    except yaml.YAMLError:
        return ([], [])
    inputs = list((d or {}).get("inputs", {}).keys())
    knobs = list((d or {}).get("size_knobs", {}).keys())
    return (sorted(inputs), sorted(knobs))


_RESOLVE_CACHE: dict[str, list[str]] | None = None


def _load_resolve_tried_map() -> dict[str, list[str]]:
    """Load results/resolve_all.json once;return {task_id: [tried...]}."""
    global _RESOLVE_CACHE
    if _RESOLVE_CACHE is not None:
        return _RESOLVE_CACHE
    rj = REPO_ROOT / "results" / "resolve_all.json"
    if not rj.exists():
        _RESOLVE_CACHE = {}
        return _RESOLVE_CACHE
    try:
        d = json.loads(rj.read_text())
        _RESOLVE_CACHE = {
            v["task_id"]: list(v.get("tried") or [])
            for v in d.get("verdicts", [])
        }
    except Exception:
        _RESOLVE_CACHE = {}
    return _RESOLVE_CACHE


def _resolve_kernel_entrypoint(
    meta: dict, kernel_path: Path, task_id: str
) -> str | None:
    """Find host wrapper. Preference order:
      1) meta.oracle.gpu_triton parsed as `kernel.py::<name>` if not 'placeholder'
      2) resolver's tried[0] from results/resolve_all.json
      3) meta.name stripped of `_kernel` suffix or `_` prefix
      4) first non-underscored module-level def in kernel.py
    """
    gpu_triton = (meta.get("oracle") or {}).get("gpu_triton", "") or ""
    if isinstance(gpu_triton, str) and "::" in gpu_triton:
        _, _, fn = gpu_triton.partition("::")
        if fn and fn != "placeholder":
            return fn
    tried_map = _load_resolve_tried_map()
    if task_id in tried_map and tried_map[task_id]:
        return tried_map[task_id][0]
    meta_name = meta.get("name")
    if isinstance(meta_name, str) and meta_name:
        candidate = meta_name
        if candidate.endswith("_kernel"):
            candidate = candidate[: -len("_kernel")]
        candidate = candidate.lstrip("_")
        if candidate and candidate != meta_name:
            # Verify the stripped name actually exists in kernel.py
            sig = _kernel_entrypoint_signature(kernel_path, candidate)
            if sig is not None:
                return candidate
        return meta_name
    return None


def _audit_task(task_dir: Path) -> TaskAudit:
    task_id = task_dir.name
    level = task_dir.parent.name
    audit = TaskAudit(task_id=task_id, level=level)

    meta_path = task_dir / "meta.yaml"
    oracle_path = task_dir / "oracle_pytorch.py"
    kernel_path = task_dir / "kernel.py"
    shapes_path = task_dir / "shapes.yaml"

    try:
        meta = yaml.safe_load(meta_path.read_text()) or {}
    except Exception as e:
        audit.status = "RUNTIME_ERROR"
        audit.runtime_error = f"meta.yaml parse: {e}"
        return audit

    audit.oracle_input_keys, audit.oracle_output_keys = _oracle_input_output_keys(
        oracle_path
    )

    fn_name = _resolve_kernel_entrypoint(meta, kernel_path, task_id)
    if not fn_name:
        audit.status = "KERNEL_PARSE_ERROR"
        audit.runtime_error = "could not resolve kernel entrypoint from meta"
        return audit

    sig = _kernel_entrypoint_signature(kernel_path, fn_name)
    if sig is None:
        audit.status = "KERNEL_PARSE_ERROR"
        audit.runtime_error = (
            f"kernel entrypoint `{fn_name}` not found in kernel.py "
            "(may be a class method or placeholder)"
        )
        return audit
    audit.kernel_required_params, audit.kernel_optional_params = sig

    shapes_inputs, _ = _shapes_input_keys(shapes_path)
    audit.shapes_input_keys = shapes_inputs

    kernel_required = set(audit.kernel_required_params)
    oracle_inputs = set(audit.oracle_input_keys)
    shapes_inputs_set = set(audit.shapes_input_keys)

    # Cross-check: what kernel needs that shapes doesn't provide
    missing_in_shapes = sorted(kernel_required - shapes_inputs_set)
    # What oracle reads that's not in shapes
    missing_in_oracle_view = sorted(oracle_inputs - shapes_inputs_set)
    # What oracle reads that's not in kernel signature
    extra_oracle = sorted(oracle_inputs - kernel_required)

    audit.missing_in_shapes = missing_in_shapes
    audit.missing_in_oracle = missing_in_oracle_view
    audit.extra_in_oracle = extra_oracle

    if missing_in_shapes:
        audit.status = "SIGNATURE_MISMATCH"
        audit.suggested_fix = (
            f"shapes.yaml.inputs is missing {missing_in_shapes} "
            f"that kernel `{fn_name}` requires; add or rename keys."
        )
        return audit

    kernel_all_params = kernel_required | set(audit.kernel_optional_params)
    # oracle reads must be subset of (kernel required + optional)
    # AND kernel required must all be readable (from shapes — already checked above)
    oracle_extra = oracle_inputs - kernel_all_params
    if oracle_extra:
        # Oracle reads keys that kernel doesn't have AT ALL (not even optional).
        # Could be aliased shapes input or oracle reads truly different layout.
        audit.status = "NEEDS_BRIDGE"
        audit.suggested_fix = (
            f"oracle reads {sorted(oracle_inputs)} including "
            f"{sorted(oracle_extra)} which kernel `{fn_name}` does not have "
            f"(kernel required={sorted(kernel_required)}, "
            f"optional={sorted(audit.kernel_optional_params)}); "
            "bridge build_inputs to derive kernel inputs from oracle inputs "
            "OR rewrite oracle."
        )
        return audit

    # ALIGNED — attempt CPU run
    try:
        ref_dir = task_dir / "r10_pyref"
        ref_dir.mkdir(parents=True, exist_ok=True)

        # build_inputs takes shapes_doc dict directly
        shapes_doc = yaml.safe_load(shapes_path.read_text())

        for shape_id in DEFAULT_SHAPES:
            try:
                inputs = build_inputs(shapes_doc, shape_id, seed=GOLDEN_SEED)
            except Exception as e:
                audit.status = "RUNTIME_ERROR"
                audit.runtime_error = f"build_inputs({shape_id}): {e}"
                return audit

            # CPU device
            inputs_cpu = {
                k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
            }

            # Import oracle module dynamically
            import importlib.util as iu

            spec_o = iu.spec_from_file_location(
                f"oracle_{task_id}", str(oracle_path)
            )
            if spec_o is None or spec_o.loader is None:
                audit.status = "RUNTIME_ERROR"
                audit.runtime_error = "oracle spec_from_file_location failed"
                return audit
            mod = iu.module_from_spec(spec_o)
            try:
                spec_o.loader.exec_module(mod)
            except Exception as e:
                msg = str(e)
                if "libdevice" in msg or "triton" in msg:
                    audit.status = "NEEDS_GPU_FOR_ORACLE"
                    audit.suggested_fix = (
                        "oracle imports GPU-only ops (libdevice/triton); "
                        "skip dev pre-compute, run on GPU stage instead."
                    )
                else:
                    audit.status = "RUNTIME_ERROR"
                    audit.runtime_error = f"oracle import: {e}"
                return audit

            try:
                ref_out = mod.run(inputs_cpu)
            except Exception as e:
                audit.status = "RUNTIME_ERROR"
                audit.runtime_error = (
                    f"oracle.run({shape_id}): {type(e).__name__}: {e}"
                )
                return audit

            # Save inputs + ref_out as safetensors
            from safetensors.torch import save_file
            import hashlib

            inputs_tensors = {
                k: (
                    v.detach().cpu()
                    if isinstance(v, torch.Tensor)
                    else torch.tensor(v)
                )
                for k, v in inputs_cpu.items()
                if isinstance(v, torch.Tensor) or isinstance(v, (int, float))
            }
            ref_tensors = {
                k: (
                    v.detach().cpu()
                    if isinstance(v, torch.Tensor)
                    else torch.tensor(v)
                )
                for k, v in ref_out.items()
                if isinstance(v, torch.Tensor) or isinstance(v, (int, float))
            }

            inputs_path = ref_dir / f"inputs_{shape_id}.safetensors"
            ref_path = ref_dir / f"ref_{shape_id}.safetensors"

            save_file(inputs_tensors, str(inputs_path))
            save_file(ref_tensors, str(ref_path))

            audit.ref_artifacts[shape_id] = hashlib.sha256(
                ref_path.read_bytes()
            ).hexdigest()

        audit.status = "ALIGNED_READY"
        return audit
    except Exception as e:
        audit.status = "RUNTIME_ERROR"
        audit.runtime_error = (
            f"unhandled: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        )
        return audit


def _load_task_for_inputs(task_dir: Path):
    """Build a minimal task spec (compatible with build_inputs)."""
    meta = yaml.safe_load((task_dir / "meta.yaml").read_text())
    shapes = yaml.safe_load((task_dir / "shapes.yaml").read_text())

    # build_inputs expects task spec with shapes.yaml fields
    class _Spec:
        pass

    spec = _Spec()
    spec.meta = meta
    spec.shapes = shapes
    spec.public_shapes = {
        sh["id"]: sh.get("knobs", {}) for sh in shapes.get("public_shapes", [])
    }
    spec.hidden_shapes = {
        sh["id"]: sh.get("knobs", {}) for sh in shapes.get("hidden_shapes", [])
    }
    spec.inputs = shapes.get("inputs", {})
    spec.size_knobs = shapes.get("size_knobs", {})
    return spec


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--task", help="single task id (e.g. L1_12_rms_norm)")
    p.add_argument("--all", action="store_true", help="all 100 tasks")
    p.add_argument(
        "--workers", type=int, default=8, help="thread pool size (default 8)"
    )
    p.add_argument(
        "--report",
        default="docs/triage_backlog/2026-05-26-oracle-pyref-audit.md",
        help="audit report md path",
    )
    p.add_argument(
        "--json-out",
        default="/tmp/oracle_pyref_audit.json",
        help="per-task audit raw json",
    )
    args = p.parse_args(argv)

    if not args.all and not args.task:
        p.error("either --task or --all required")

    if args.task:
        # find that task dir
        for d in (REPO_ROOT / "tasks").glob(f"level*/{args.task}"):
            task_dirs = [d]
            break
        else:
            log.error("task %s not found", args.task)
            return 2
    else:
        task_dirs = sorted(
            d
            for level_dir in (REPO_ROOT / "tasks").glob("level*")
            if level_dir.is_dir()
            for d in level_dir.iterdir()
            if d.is_dir() and (d / "meta.yaml").exists()
        )

    log.info("auditing %d task(s) with %d workers", len(task_dirs), args.workers)
    audits: dict[str, TaskAudit] = {}

    if len(task_dirs) == 1 or args.workers == 1:
        for td in task_dirs:
            audits[td.name] = _audit_task(td)
            log.info("  %s: %s", td.name, audits[td.name].status)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_audit_task, td): td for td in task_dirs}
            for fut in as_completed(futs):
                td = futs[fut]
                try:
                    audits[td.name] = fut.result()
                except Exception as e:
                    log.error("  %s: worker failed: %s", td.name, e)
                    audits[td.name] = TaskAudit(
                        task_id=td.name,
                        level=td.parent.name,
                        status="RUNTIME_ERROR",
                        runtime_error=str(e),
                    )
                log.info("  %s: %s", td.name, audits[td.name].status)

    # Status tally
    from collections import Counter

    tally = Counter(a.status for a in audits.values())
    log.info("--- tally ---")
    for s, n in tally.most_common():
        log.info("  %s: %d", s, n)

    # Write raw json
    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(
            {tid: asdict(a) for tid, a in audits.items()},
            indent=2,
            default=str,
        )
    )
    log.info("raw audit json -> %s", json_out)

    # Write md report
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "---",
        "date: 2026-05-26",
        "tool: scripts/oracle_pyref_precompute.py",
        "purpose: V9 M2.1a (dev) — R10 stage-1 oracle PyTorch reference precompute",
        "---",
        "",
        "# Oracle PyRef Precompute Audit",
        "",
        f"Tasks audited: **{len(audits)}**  ",
        "",
        "## Status distribution",
        "",
        "| status | count |",
        "|---|---|",
    ]
    for s, n in tally.most_common():
        lines.append(f"| `{s}` | {n} |")
    lines.append("")

    for status_label in (
        "ALIGNED_READY",
        "SIGNATURE_MISMATCH",
        "NEEDS_BRIDGE",
        "NEEDS_GPU_FOR_ORACLE",
        "RUNTIME_ERROR",
        "KERNEL_PARSE_ERROR",
    ):
        sub = [a for a in audits.values() if a.status == status_label]
        if not sub:
            continue
        lines.append(f"## {status_label} ({len(sub)})")
        lines.append("")
        for a in sub:
            lines.append(f"### `{a.task_id}` ({a.level})")
            lines.append("")
            if a.kernel_required_params:
                lines.append(
                    f"- kernel required ({len(a.kernel_required_params)}): "
                    f"`{', '.join(a.kernel_required_params)}`"
                )
            if a.kernel_optional_params:
                lines.append(
                    f"- kernel optional ({len(a.kernel_optional_params)}): "
                    f"`{', '.join(a.kernel_optional_params)}`"
                )
            if a.oracle_input_keys:
                lines.append(
                    f"- oracle inputs[..] ({len(a.oracle_input_keys)}): "
                    f"`{', '.join(a.oracle_input_keys)}`"
                )
            if a.oracle_output_keys:
                lines.append(
                    f"- oracle outputs ({len(a.oracle_output_keys)}): "
                    f"`{', '.join(a.oracle_output_keys)}`"
                )
            if a.shapes_input_keys:
                lines.append(
                    f"- shapes inputs ({len(a.shapes_input_keys)}): "
                    f"`{', '.join(a.shapes_input_keys)}`"
                )
            if a.missing_in_shapes:
                lines.append(f"- **missing in shapes**: `{a.missing_in_shapes}`")
            if a.missing_in_oracle:
                lines.append(f"- missing-in-oracle (oracle reads non-shapes key): `{a.missing_in_oracle}`")
            if a.extra_in_oracle:
                lines.append(f"- extra-in-oracle (oracle reads non-kernel param): `{a.extra_in_oracle}`")
            if a.suggested_fix:
                lines.append(f"- **suggested fix**: {a.suggested_fix}")
            if a.runtime_error:
                lines.append(f"- runtime error: `{a.runtime_error[:200]}`")
            if a.ref_artifacts:
                lines.append("- saved ref artifacts:")
                for sid, sha in a.ref_artifacts.items():
                    lines.append(f"  - `r10_pyref/ref_{sid}.safetensors` sha256=`{sha[:16]}...`")
            lines.append("")

    report.write_text("\n".join(lines))
    log.info("md report -> %s", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
