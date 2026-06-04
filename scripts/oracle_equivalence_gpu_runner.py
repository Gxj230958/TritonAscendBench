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
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from triton_ascend_bench.inputs import build_inputs  # noqa: E402
from triton_ascend_bench.tolerance import (  # noqa: E402
    ToleranceClass,
    compare,
    load_policy,
)

GOLDEN_SEED = 0  # mirror oracle_pyref_precompute.py's seed for determinism


_TOLERANCE_CACHE: dict[str, ToleranceClass] | None = None
_CEILING_CFG: dict | None = None


def _ceiling_config() -> dict:
    """Load the two-mode dtype-ceiling acceptance allowlist (cached). See
    triton_ascend_bench/ceiling_acceptance.yaml. Allowlist-gated: L2_23/24 and every other
    task stay strict by construction."""
    global _CEILING_CFG
    if _CEILING_CFG is None:
        p = REPO_ROOT / "triton_ascend_bench" / "ceiling_acceptance.yaml"
        try:
            _CEILING_CFG = yaml.safe_load(p.read_text()) or {}
        except Exception:
            _CEILING_CFG = {}
    return _CEILING_CFG


def _ceiling_verdict(task_id: str, sv: "ShapeVerdict") -> tuple[bool, str, str]:
    """Decide whether a STRICT-failing shape of an allowlisted task is an accepted
    dtype-ULP ceiling. Returns (accepted, mode, reason). FAIL-CLOSED: any missing config,
    unknown mode, missing/zero/non-finite scale, or non-allowlisted task -> (False, ...).

    Mode A (tol_units_tasks): accept iff max(max_tol_units) <= tol_units_cap.
    Mode B (physical_tasks): accept iff EVERY strict-failing output key satisfies
      max_abs_diff[key] <= physical_factor * max|ref[key]| * ulp_rel  (max|ref|>0, finite)."""
    import math

    cfg = _ceiling_config()
    if not cfg:
        return False, "none", "no ceiling config (fail-closed)"
    # Universal fail-closed gates (both modes): an accepted shape MUST have at least one
    # recorded tolerance-failing key (never accept some other all_pass=False path, e.g. a
    # compare exception), no compare blocker, and no float-dtype mismatch on any key.
    if not sv.failing_keys:
        return False, "none", "no failing keys recorded (fail-closed)"
    if getattr(sv, "compare_error", False):
        # A compare() RAISED (not an ordinary tolerance fail, whose detail also lands in
        # blocker_reason). Only the genuine exception fails-closed -- otherwise every normal
        # ceiling failure would be rejected (its fail-detail is copied into blocker_reason).
        return False, "none", "compare exception (fail-closed)"
    if sv.bad_dtype_keys:
        return False, "none", f"dtype mismatch on {sv.bad_dtype_keys} (fail-closed)"
    # Mode A -- co-located tol_units cap.
    if task_id in set(cfg.get("tol_units_tasks") or []):
        cap = cfg.get("tol_units_cap")
        if not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0:
            return False, "tol_units_cap", f"invalid cap {cap!r} (fail-closed)"
        vals = list(sv.max_tol_units.values())
        if not vals or not all(math.isfinite(v) for v in vals):
            return False, "tol_units_cap", "non-finite/empty tol_units (fail-closed)"
        worst = max(vals)
        return (worst <= float(cap)), "tol_units_cap", f"max_tol_units={worst:.4g} cap={cap}"
    # Mode B -- per-output-key physical scale.
    phys = cfg.get("physical_tasks") or {}
    if task_id in phys:
        factor = cfg.get("physical_factor")
        ulp = (phys.get(task_id) or {}).get("ulp_rel")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0
                   for x in (factor, ulp)):
            return False, "physical_scale", f"invalid factor/ulp ({factor!r},{ulp!r}) fail-closed"
        factor = float(factor); ulp = float(ulp)
        for key in sv.failing_keys:
            mref = sv.max_ref_abs.get(key)
            mabs = sv.max_abs_diff.get(key)
            if mref is None or mabs is None or not (math.isfinite(mref) and math.isfinite(mabs)) or mref <= 0.0:
                return False, "physical_scale", f"{key}: missing/zero/non-finite scale (fail-closed)"
            bound = factor * mref * ulp
            if mabs > bound:
                return (
                    False, "physical_scale",
                    f"{key}: max_abs={mabs:.4g} > {factor}*{mref:.4g}*{ulp:.3g}={bound:.4g}",
                )
        return True, "physical_scale", f"all {len(sv.failing_keys)} failing key(s) within {factor}xULP scale"
    return False, "none", "not on ceiling allowlist"


def _tolerance_for(name: str) -> ToleranceClass:
    """Lookup ToleranceClass by name from the global policy file."""
    global _TOLERANCE_CACHE
    if _TOLERANCE_CACHE is None:
        _TOLERANCE_CACHE = load_policy()
    if name in _TOLERANCE_CACHE:
        return _TOLERANCE_CACHE[name]
    # fallback: use a permissive default if the task's tolerance_class is
    # missing from the policy (shouldn't happen but fail-safe)
    log.warning(
        "tolerance_class %r not in tolerance_policy.yaml; using vector_reduce",
        name,
    )
    return _TOLERANCE_CACHE.get(
        "vector_reduce",
        ToleranceClass(name="fallback", atol=1e-3, rtol=1e-3),
    )


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
    # Co-located normalized error max(|actual-ref| / (atol + rtol*|ref|)) -- the
    # ONLY metric that separates a structural divergence (>> 1) from a 1-ULP
    # dtype ceiling (~1). max_abs_diff and max_rel_diff are independent per-tensor
    # maxima at DIFFERENT elements, so combining them mis-classifies (a large-abs
    # bf16 ULP on a large-magnitude output co-occurs with a large-rel near-zero
    # element). Diagnostic only -- the verdict is still compare()'s .passed.
    max_tol_units: dict[str, float] = field(default_factory=dict)
    # Per-output-key max|ref| (the physical output scale) + which keys strict-failed +
    # acceptance evidence stats (fail density, signed-mean ratio, rms ratio). Used ONLY by
    # the dtype-ceiling acceptance path (CEILING_ACCEPTED); diagnostic for the rest.
    max_ref_abs: dict[str, float] = field(default_factory=dict)
    failing_keys: list[str] = field(default_factory=list)
    bad_dtype_keys: list[str] = field(default_factory=list)  # ker dtype != ref dtype
    compare_error: bool = False  # a compare() raised (NOT an ordinary tolerance fail)
    accept_stats: dict[str, dict] = field(default_factory=dict)
    accept_reason: str = ""
    blocker_reason: str = ""
    output_keys_ref: list[str] = field(default_factory=list)
    output_keys_kernel: list[str] = field(default_factory=list)


@dataclass
class TaskVerdict:
    task_id: str
    overall: str = ""  # VERIFIED | FAILED | MIXED | LAUNCH_BLOCKED | SKIPPED
    shapes: list[ShapeVerdict] = field(default_factory=list)
    elapsed_s: float = 0.0


def _kernel_has_function(kernel_path: Path, fn_name: str) -> bool:
    """Quick AST check: does kernel.py have a module-level def for fn_name?"""
    try:
        tree = ast.parse(kernel_path.read_text())
    except (OSError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == fn_name
        ):
            return True
    return False


def _enumerate_non_jit_module_defs(kernel_path: Path) -> list[tuple[str, list[str]]]:
    """Mirror of oracle_pyref_precompute.py helper: non-jit module-level
    defs with their required params, for AST-based wrapper selection."""
    if not kernel_path.exists():
        return []
    try:
        tree = ast.parse(kernel_path.read_text())
    except SyntaxError:
        return []
    out: list[tuple[str, list[str]]] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        is_jit = False
        for dec in node.decorator_list:
            src = ast.unparse(dec)
            if (
                "triton.jit" in src
                or "triton.heuristics" in src
                or "triton.autotune" in src
            ):
                is_jit = True
                break
        if is_jit:
            continue
        args = [a.arg for a in node.args.args]
        defaults = node.args.defaults
        n_req = len(args) - len(defaults)
        req = args[:n_req]
        if req and req[0] == "self":
            req = req[1:]
        out.append((node.name, req))
    return out


def _resolve_kernel_entrypoint(
    meta: dict,
    kernel_path: Path,
    task_id: str,
    oracle_input_keys: list[str] | None = None,
) -> str | None:
    """Shim over the shared resolver in
    :mod:`triton_ascend_bench.kernel_resolver`. The R10.a / R10.b / auto-
    patch scripts all delegate here so they can never drift on jaccard
    thresholds, jit gating, or tried-cache trust rules.
    """
    from triton_ascend_bench.kernel_resolver import (  # noqa: PLC0415
        resolve_kernel_entrypoint,
    )

    return resolve_kernel_entrypoint(
        meta,
        kernel_path,
        task_id,
        oracle_input_keys,
        resolve_cache_path=REPO_ROOT / "results" / "resolve_all.json",
    )


def _import_kernel_function(kernel_path: Path, fn_name: str, infer_root: Path):
    """Import kernel.py and return the host-wrapper function fn_name.

    Vendored kernel.py files use `from .X import Y` relative imports that
    resolve into the upstream sibling directory recorded in their
    `Source path:` header. A bare `spec_from_file_location` leaves
    `__package__` empty and these imports fail with
    `attempted relative import with no known parent package`.

    Delegate to the existing `_pa_oracle_check_runner._load()` helper
    which builds a synthetic package chain pointing at the upstream
    sibling so the relative imports resolve correctly. This is the same
    loader the existing dual-anchor oracle check uses, so behavior is
    consistent with phase_a_prime semantics.
    """
    # Make infer/{vllm,sglang} importable so absolute upstream imports
    # (e.g. `from vllm._custom_ops import ...`) also resolve.
    if (infer_root / "vllm").exists():
        sys.path.insert(0, str(infer_root / "vllm"))
    if (infer_root / "sglang" / "python").exists():
        sys.path.insert(0, str(infer_root / "sglang" / "python"))

    # Lazy-import the loader so a stale gpu host without _pa_oracle_check_runner
    # still parses this file (the import would fail at call time, not module load).
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from _pa_oracle_check_runner import _load as _load_kernel

    upstream_roots: list[Path] = []
    if (infer_root / "vllm").is_dir():
        upstream_roots.append(infer_root / "vllm")
    if (infer_root / "sglang" / "python").is_dir():
        upstream_roots.append(infer_root / "sglang" / "python")

    # Un-severed kernels import vllm num_compute_units (grid sizing), which raises
    # on the daemon. Wrap it before _load execs the kernel so the `from ... import`
    # binds the stub. The C1 path patches it elsewhere; R10.b loads via _load here.
    for _r in upstream_roots:
        if str(_r) not in sys.path:
            sys.path.insert(0, str(_r))
    try:
        from triton_ascend_bench.upstream_shims.vllm_runtime_hygiene import (
            install_vllm_num_compute_units_stub,
        )
        install_vllm_num_compute_units_stub()
    except Exception:
        pass

    mod = _load_kernel(
        kernel_path,
        f"kernel_{kernel_path.parent.name}",
        upstream_roots=upstream_roots or None,
    )
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


def _mutated_input_keys(meta: dict, oracle_src: str) -> set[str]:
    """Input keys an in-place task mutates. These are graded outputs and must be
    compared, not just the returned dict. Sources: interface.mutability
    ('state: inplace'), inplace_outputs / return_plus_inplace, and the oracle's
    inputs['X'].copy_() targets."""
    import re as _re

    iface = meta.get("interface") or {}
    keys: set[str] = set()
    mut = iface.get("mutability") or ""
    if isinstance(mut, str) and ":" in mut and "inplace" in mut.split(":", 1)[1]:
        keys |= {n.strip() for n in mut.split(":", 1)[0].split(",") if n.strip()}
    for fld in ("inplace_outputs", "return_plus_inplace"):
        v = iface.get(fld)
        if isinstance(v, dict):  # return_plus_inplace: {return_name, inplace: [...]}
            v = v.get("inplace")
        if isinstance(v, str):
            keys |= {n.strip() for n in v.split(",") if n.strip()}
        elif isinstance(v, (list, tuple)):
            keys |= {str(n).strip() for n in v}
    keys |= set(_re.findall(r'inputs\[["\']([^"\']+)["\']\]\.copy_\(', oracle_src or ""))
    return keys


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

    # Read oracle's `inputs[...]` keys for AST-based wrapper auto-detect
    oracle_input_keys: list[str] = []
    oracle_path = task_dir / "oracle_pytorch.py"
    oracle_src_full = oracle_path.read_text() if oracle_path.exists() else ""
    if oracle_src_full:
        import re as _re
        oracle_input_keys = sorted(
            set(_re.findall(r'inputs\[["\']([^"\']+)["\']\]', oracle_src_full))
            | set(_re.findall(r'inputs\.get\(["\']([^"\']+)["\']', oracle_src_full))
        )
    # In-place mutated keys (graded outputs captured into the ref by R10.a);
    # resolved here while `meta` is still the task meta (the loop shadows it).
    mutated_keys_b = _mutated_input_keys(meta, oracle_src_full)

    # output_projection: a kernel may return a SUPERSET tuple (e.g. layer_norm
    # returns (out, mean, rstd)); grade only the mapped indices. Mirrors C1
    # (_pa_oracle_check_runner) so both gates project the same surface.
    oproj_b = (meta.get("interface") or {}).get("output_projection") or {}

    # Launcher tasks: the graded entrypoint is _pa_launcher.py::run(inputs),
    # which _load execs into the kernel module (so mod.run resolves) and which
    # takes the whole input dict, not by-name kwargs. Some metas record
    # kernel.py::run while the real run lives in the launcher (e.g. L1_27), so
    # treat any ::run as a launcher when a _pa_launcher.py defines run.
    gpu_triton = str((meta.get("oracle", {}) or {}).get("gpu_triton", "") or "").strip()
    launcher_file = task_dir / "_pa_launcher.py"
    has_launcher_run = launcher_file.is_file() and "def run" in launcher_file.read_text()
    is_launcher = has_launcher_run and gpu_triton.endswith("::run")
    if is_launcher:
        fn_name = "run"
    else:
        fn_name = _resolve_kernel_entrypoint(
            meta, kernel_path, task_id, oracle_input_keys=oracle_input_keys
        )
    if not fn_name:
        verdict.overall = "LAUNCH_BLOCKED"
        log.info("  %s: cannot resolve kernel entrypoint", task_id)
        return verdict

    # V9 codex review 2026-05-27: static wrapper-rank check BEFORE GPU
    # import / kernel call. Many tasks previously fell through to
    # `_import_kernel_function`, then LAUNCH_BLOCKED with stale
    # `not enough values to unpack` tracebacks. Catching the rank
    # mismatch statically gives an actionable verdict (shapes-patch
    # candidate vs source-2 reroute) instead of a misleading runtime
    # trace; AND when ALL shapes are static-blocked we skip kernel
    # import entirely, so import-time failures cannot mask a clean
    # static verdict (codex Medium #2).
    from triton_ascend_bench.wrapper_contract import (  # noqa: PLC0415
        check_against_inputs,
        extract_oracle_helper_contract,
        extract_wrapper_contract,
    )

    wrapper_contract = extract_wrapper_contract(kernel_path, fn_name)
    # Shared oracle-helper extractor (codex Medium #3 -- de-duplicates
    # the near-identical R10.a / R10.b inline walks).
    oracle_helper_contract = (
        extract_oracle_helper_contract(oracle_path)
        if oracle_path.exists() else None
    )

    # Re-load shapes.yaml so we can rebuild inputs deterministically (the
    # precompute step's safetensors only stores torch.Tensor values; string
    # scalars like 'kv_cache_dtype' get dropped on save and would surface
    # later as "missing 1 required positional argument").
    shapes_doc = yaml.safe_load((task_dir / "shapes.yaml").read_text())

    # Phase 1 (PRE-IMPORT): scan every shape's rebuilt inputs for
    # wrapper-contract violations. Collect static verdicts + reusable
    # inputs so we don't pay build_inputs twice.
    all_shape_ids = [
        f.stem.replace("ref_", "")
        for f in sorted(ref_dir.glob("ref_*.safetensors"))
    ]
    per_shape_inputs: dict[str, dict] = {}
    per_shape_static_verdict: dict[str, ShapeVerdict] = {}
    if wrapper_contract is not None and (
        wrapper_contract.accepted_ranks or wrapper_contract.min_ranks
    ):
        for sid in all_shape_ids:
            try:
                inputs_full = build_inputs(shapes_doc, sid, seed=GOLDEN_SEED)
            except Exception as e:
                sv = ShapeVerdict(shape_id=sid, status="LAUNCH_BLOCKED")
                sv.blocker_reason = f"build_inputs: {type(e).__name__}: {e}"
                per_shape_static_verdict[sid] = sv
                continue
            per_shape_inputs[sid] = inputs_full
            cpu_ranks = {
                k: v.ndim
                for k, v in inputs_full.items()
                if isinstance(v, torch.Tensor)
            }
            violations = check_against_inputs(wrapper_contract, cpu_ranks)
            if not violations:
                continue
            # Classify by whether oracle agrees with wrapper.
            conflict_keys: list[str] = []
            if oracle_helper_contract is not None:
                for v in violations:
                    oranks = (
                        oracle_helper_contract.accepted_ranks.get(v.input_name)
                        or set()
                    )
                    if oranks and not (oranks & v.expected_ranks):
                        conflict_keys.append(v.input_name)
            details = "; ".join(
                f"{v.input_name}: wrapper expects rank "
                f"{sorted(v.expected_ranks) or f'>= {v.expected_min_rank}'}, "
                f"build_inputs gave {v.actual_rank}"
                for v in violations
            )
            sv = ShapeVerdict(shape_id=sid)
            if conflict_keys:
                sv.status = "ORACLE_KERNEL_LAYOUT_MISMATCH"
                sv.blocker_reason = (
                    f"oracle helper and wrapper disagree on rank for "
                    f"{conflict_keys}; route to source (2). "
                    f"Details: {details}"
                )
            else:
                sv.status = "WRAPPER_CONTRACT_MISMATCH"
                sv.blocker_reason = (
                    f"shapes.yaml.inputs.dims rank disagrees with kernel "
                    f"host wrapper `{fn_name}` unpack. Details: {details}"
                )
            per_shape_static_verdict[sid] = sv

    # If EVERY shape has a static verdict (any mix of WRAPPER_CONTRACT_
    # MISMATCH / ORACLE_KERNEL_LAYOUT_MISMATCH / build-error
    # LAUNCH_BLOCKED), skip kernel import entirely. This is the
    # codex Medium #2 fix: an unrelated import-time crash can no
    # longer mask a clean static verdict.
    if (
        per_shape_static_verdict
        and len(per_shape_static_verdict) == len(all_shape_ids)
    ):
        for sid in all_shape_ids:
            sv = per_shape_static_verdict[sid]
            verdict.shapes.append(sv)
            log.warning("  %s/%s: %s", task_id, sid, sv.status)
        statuses = {s.status for s in verdict.shapes}
        if statuses == {"WRAPPER_CONTRACT_MISMATCH"}:
            verdict.overall = "WRAPPER_CONTRACT_MISMATCH"
        elif statuses == {"ORACLE_KERNEL_LAYOUT_MISMATCH"}:
            verdict.overall = "ORACLE_KERNEL_LAYOUT_MISMATCH"
        elif statuses <= {
            "WRAPPER_CONTRACT_MISMATCH",
            "ORACLE_KERNEL_LAYOUT_MISMATCH",
        }:
            verdict.overall = "ORACLE_KERNEL_LAYOUT_MISMATCH"
        else:
            verdict.overall = "LAUNCH_BLOCKED"
        verdict.elapsed_s = time.time() - t0
        log.info(
            "  %s: %s (all %d shape(s) statically blocked; kernel import skipped)",
            task_id, verdict.overall, len(all_shape_ids),
        )
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
    tol = _tolerance_for(tolerance_class_name)

    import inspect
    sig = inspect.signature(kernel_fn)
    sig_param_names = set(sig.parameters.keys())

    n_verified = 0
    n_failed = 0
    n_ceiling_accepted = 0
    n_blocked = 0
    n_input_drift = 0
    # Ref output keys seen across shapes; the in-place backstop below uses them.
    ref_keys_seen: set[str] = set()
    for ref_file in sorted(ref_dir.glob("ref_*.safetensors")):
        sid = ref_file.stem.replace("ref_", "")

        # Honor the Phase 1 static verdict for this shape if one exists
        # (partial static block -- some shapes statically failed, others
        # passed and need the GPU path).
        if sid in per_shape_static_verdict:
            sv = per_shape_static_verdict[sid]
            n_blocked += 1
            verdict.shapes.append(sv)
            log.warning("  %s/%s: %s", task_id, sid, sv.status)
            continue

        sv = ShapeVerdict(shape_id=sid)

        try:
            ref_tensors = load_file(str(ref_file))
        except Exception as e:
            sv.status = "LAUNCH_BLOCKED"
            sv.blocker_reason = f"safetensors load: {e}"
            n_blocked += 1
            verdict.shapes.append(sv)
            continue
        ref_keys_seen |= set(ref_tensors.keys())

        # V9 codex review 2026-05-27: safetensors load_file returns
        # alphabetical keys, so a multi-output oracle whose dict
        # insertion order didn't happen to be alphabetical can no
        # longer be re-aligned with the kernel's positional tuple
        # return. R10.a now stamps the original ``output_key_order``
        # into the safetensors metadata; read it back here so the
        # tuple-to-dict mapping below honours the oracle's intent.
        ref_output_order: list[str] = []
        try:
            with safe_open(str(ref_file), framework="pt") as f:
                meta = f.metadata() or {}
            if "output_key_order" in meta:
                ref_output_order = json.loads(meta["output_key_order"])
        except Exception:
            # Missing / malformed metadata is fine on pre-2026-05-27
            # ref artifacts -- fall through to the alphabetical
            # mapping path below.
            ref_output_order = []

        # Reuse Phase 1 build_inputs result when available; rebuild as a
        # safety net for paths that didn't populate the cache (the
        # wrapper had no rank constraints so Phase 1 short-circuited).
        inputs_full = per_shape_inputs.get(sid)
        if inputs_full is None:
            try:
                inputs_full = build_inputs(shapes_doc, sid, seed=GOLDEN_SEED)
            except Exception as e:
                sv.status = "LAUNCH_BLOCKED"
                sv.blocker_reason = f"build_inputs: {type(e).__name__}: {e}"
                n_blocked += 1
                verdict.shapes.append(sv)
                log.warning(
                    "  %s/%s: build_inputs fail: %s", task_id, sid, e
                )
                continue

        # H2 input-faithfulness guard. R10.b rebuilds inputs independently of
        # R10.a's captured ref; any divergence (RNG/shapes/build_inputs drift)
        # makes a correct kernel falsely FAIL. Fail-closed: artifacts must
        # exist+load and the full payload (tensor bytes + non-tensor values)
        # must match R10.a, else INPUT_DRIFT (never a silent precision FAILED).
        from triton_ascend_bench.inputs import split_inputs  # noqa: PLC0415

        inputs_file = ref_dir / f"inputs_{sid}.safetensors"
        sidecar_file = ref_dir / f"inputs_{sid}.sidecar.json"
        drift_reason: str | None = None
        saved_tensors: dict | None = None
        saved_sidecar: dict | None = None
        rb_tensors: dict | None = None
        rb_sidecar: dict | None = None
        if not inputs_file.is_file():
            drift_reason = f"R10.a inputs artifact missing ({inputs_file.name})"
        elif not sidecar_file.is_file():
            drift_reason = (
                f"R10.a inputs sidecar missing ({sidecar_file.name}); "
                f"re-run R10.a to capture the full input payload"
            )
        else:
            try:
                saved_tensors = load_file(str(inputs_file))
            except Exception as e:
                drift_reason = f"R10.a inputs load failed: {e}"
            if drift_reason is None:
                try:
                    saved_sidecar = json.loads(sidecar_file.read_text())
                except Exception as e:
                    drift_reason = f"R10.a sidecar load failed: {e}"
        if drift_reason is None:
            try:
                rb_tensors, rb_sidecar = split_inputs(inputs_full)
            except Exception as e:
                # A non-JSON-able input (e.g. a struct) R10.a could not
                # capture faithfully -> cannot prove input parity -> block.
                drift_reason = f"rebuilt inputs not splittable for compare: {e}"
        if drift_reason is None:
            # Type-strict: Python `==` treats 1 == 1.0 == True, so compare
            # type and value recursively to catch scalar type drift.
            def _strict_eq(a: object, b: object) -> bool:
                if type(a) is not type(b):
                    return False
                if isinstance(a, list):
                    return len(a) == len(b) and all(
                        _strict_eq(x, y) for x, y in zip(a, b)
                    )
                if isinstance(a, dict):
                    return a.keys() == b.keys() and all(
                        _strict_eq(a[k], b[k]) for k in a
                    )
                return a == b

            if set(rb_tensors.keys()) != set(saved_tensors.keys()):
                drift_reason = (
                    f"tensor key-set drift: rebuilt {sorted(rb_tensors)} "
                    f"vs saved {sorted(saved_tensors)}"
                )
            elif not _strict_eq(rb_sidecar, saved_sidecar):
                drift_reason = (
                    f"non-tensor input drift (type-strict): rebuilt "
                    f"{rb_sidecar} != saved {saved_sidecar}"
                )
        if drift_reason is None:
            for k, saved_t in saved_tensors.items():
                rb_t = rb_tensors[k]  # split_inputs already cpu + contiguous
                if (
                    tuple(rb_t.shape) != tuple(saved_t.shape)
                    or rb_t.dtype != saved_t.dtype
                    or not torch.equal(rb_t, saved_t)
                ):
                    drift_reason = f"tensor input '{k}' value/shape/dtype drift"
                    break
        if drift_reason is not None:
            sv.status = "INPUT_DRIFT"
            sv.blocker_reason = (
                f"INPUT_DRIFT: {drift_reason} -- ref captured on different "
                f"inputs; re-run R10.a on this host (not a precision FAILED)."
            )
            n_input_drift += 1
            verdict.shapes.append(sv)
            log.warning(
                "  %s/%s: INPUT_DRIFT: %s", task_id, sid, drift_reason
            )
            continue

        try:
            # Move tensor inputs to GPU; keep non-tensor inputs (str, int,
            # float, etc.) as-is.
            inputs_gpu: dict[str, object] = {}
            for k, v in inputs_full.items():
                if isinstance(v, torch.Tensor):
                    inputs_gpu[k] = v.to(gpu_device)
                else:
                    inputs_gpu[k] = v
            if is_launcher:
                # launcher: run(inputs) -> dict; pass the whole input dict.
                kernel_out = kernel_fn(inputs_gpu)
            else:
                # Filter to only the keys the kernel signature actually accepts —
                # shapes.yaml may include extras (e.g. BLOCK_SIZE_M, scalars meant
                # for the underlying @triton.jit kernel, not the host wrapper)
                # that the host wrapper rejects as unexpected kwargs.
                inputs_for_kernel = {
                    k: v for k, v in inputs_gpu.items() if k in sig_param_names
                }
                # Provide kernel optional defaults for any sig param the rebuilt
                # inputs do not cover.
                extras = _materialise_kernel_optional_params(
                    kernel_fn, set(inputs_for_kernel.keys())
                )
                for k, v in extras.items():
                    inputs_for_kernel.setdefault(k, v)

                kernel_out = kernel_fn(**inputs_for_kernel)
        except Exception as e:
            sv.status = "LAUNCH_BLOCKED"
            sv.blocker_reason = f"{type(e).__name__}: {e}"
            n_blocked += 1
            verdict.shapes.append(sv)
            log.warning("  %s/%s: kernel launch fail: %s", task_id, sid, e)
            continue

        # Project a superset tuple return to the graded indices (fail-closed:
        # tuple/list, pinned arity, in-range unique indices) before normalising.
        if oproj_b and isinstance(kernel_out, (tuple, list)):
            _arity = oproj_b.get("arity")
            _omap = oproj_b.get("map") or {}
            _idxs = list(_omap.values())
            if (
                isinstance(_arity, int)
                and len(kernel_out) == _arity
                and _omap
                and len(set(_idxs)) == len(_idxs)
                and all(isinstance(i, int) and 0 <= i < _arity for i in _idxs)
            ):
                kernel_out = {k: kernel_out[i] for k, i in _omap.items()}
            else:
                sv.status = "FAILED"
                sv.blocker_reason = (
                    f"output_projection invalid (declared={oproj_b}, "
                    f"actual return len={len(kernel_out)})"
                )
                n_failed += 1
                verdict.shapes.append(sv)
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
        elif kernel_out is None:
            # Fully in-place kernel (returns None, writes pre-allocated input
            # buffers). Require a DECLARED in-place output present in the ref --
            # else a no-op echoing read-only inputs would trivially pass.
            declared_inplace = mutated_keys_b & set(ref_tensors.keys())
            if not declared_inplace:
                sv.status = "FAILED"
                sv.blocker_reason = (
                    "kernel returned None but no declared inplace_outputs cover "
                    "the ref keys; cannot establish the graded in-place write"
                )
                n_failed += 1
                verdict.shapes.append(sv)
                continue
            out_dict = {
                k: inputs_gpu[k]
                for k in ref_tensors.keys()
                if isinstance(inputs_gpu.get(k), torch.Tensor)
            }
        elif isinstance(kernel_out, (tuple, list)):
            # Multi-output kernels MUST carry the oracle's dict insertion
            # order in safetensors metadata. Silently falling back to
            # alphabetical ``ref_tensors.keys()`` is a mispair landmine
            # (codex r2 review 2026-05-27): for the L1_36 case it would
            # quietly swap seg_indptr and src2dst and report inf again.
            # Validate the metadata: it must be a list of strings whose
            # set EXACTLY matches the safetensors keys and whose length
            # equals the kernel tuple length.
            alphabetical_keys = list(ref_tensors.keys())
            if len(ref_tensors) <= 1:
                # 1-output case can use alphabetical safely (single key).
                ref_keys = ref_output_order or alphabetical_keys
            else:
                if not ref_output_order:
                    sv.status = "LAUNCH_BLOCKED"
                    sv.blocker_reason = (
                        f"multi-output kernel returned {len(kernel_out)}-tuple "
                        f"but ref safetensors has no `output_key_order` "
                        f"metadata; the alphabetical-load fallback would "
                        f"mis-pair outputs. Re-run R10.a so the artifact is "
                        f"stamped with the oracle's dict insertion order."
                    )
                    n_blocked += 1
                    verdict.shapes.append(sv)
                    log.warning("  %s/%s: %s", task_id, sid, sv.blocker_reason)
                    continue
                if (
                    not isinstance(ref_output_order, list)
                    or not all(isinstance(k, str) for k in ref_output_order)
                    or len(set(ref_output_order)) != len(ref_output_order)
                    or set(ref_output_order) != set(alphabetical_keys)
                    or len(ref_output_order) != len(kernel_out)
                ):
                    sv.status = "LAUNCH_BLOCKED"
                    sv.blocker_reason = (
                        f"output_key_order metadata invalid: got "
                        f"{ref_output_order!r}, alphabetical ref keys "
                        f"{alphabetical_keys!r}, kernel returned "
                        f"{len(kernel_out)}-tuple. Re-run R10.a to refresh."
                    )
                    n_blocked += 1
                    verdict.shapes.append(sv)
                    log.warning("  %s/%s: %s", task_id, sid, sv.blocker_reason)
                    continue
                ref_keys = ref_output_order
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

        # In-place graded outputs: compare the kernel's post-launch mutated
        # inputs, overriding any returned copy -- a candidate that returns a
        # correct tensor without mutating the input must still fail.
        for mk in mutated_keys_b:
            mv = inputs_gpu.get(mk)
            if isinstance(mv, torch.Tensor):
                out_dict[mk] = mv

        # Move kernel outputs to CPU for compare
        out_dict_cpu = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in out_dict.items()
        }
        sv.output_keys_kernel = sorted(out_dict_cpu.keys())
        sv.output_keys_ref = sorted(ref_tensors.keys())

        # Optional debug dump of the kernel's actual output (gated by env or a
        # task-level `.dump_actual` sentinel; kept outside r10_pyref which R10.a
        # wipes) so a contract divergence can be diffed against ref offline.
        if os.environ.get("TAB_DUMP_ACTUAL") or (task_dir / ".dump_actual").exists():
            from safetensors.torch import save_file as _save_actual

            dump = {
                k: v.contiguous()
                for k, v in out_dict_cpu.items()
                if isinstance(v, torch.Tensor)
            }
            if dump:
                _save_actual(dump, str(ref_dir / f"actual_{sid}.safetensors"))

        # Compare each matched key
        all_pass = True
        fail_reasons: list[str] = []
        for key in sorted(set(sv.output_keys_ref) & set(sv.output_keys_kernel)):
            ref_t = ref_tensors[key]
            ker_raw = out_dict_cpu[key]
            if not isinstance(ker_raw, torch.Tensor):
                continue
            ker_t = ker_raw
            # dtype-equality gate: a ceiling exception must never accept a wrong-float-dtype
            # output (compare() casts both to fp32 and would not catch it). Record any
            # mismatch so the ceiling verdict can fail-closed on it.
            if getattr(ker_raw, "dtype", None) != getattr(ref_t, "dtype", None):
                sv.bad_dtype_keys.append(key)
            try:
                # Numeric diff stats for logging (works on float view)
                ref_f = ref_t.detach().to(torch.float32).cpu()
                ker_f = ker_t.detach().to(torch.float32).cpu()
                if ref_f.shape == ker_f.shape:
                    diff = (ref_f - ker_f).abs()
                    sv.max_abs_diff[key] = float(diff.max().item())
                    denom = ref_f.abs().clamp_min(1e-12)
                    sv.max_rel_diff[key] = float((diff / denom).max().item())
                    # Co-located tolerance units (same element as the worst
                    # normalized error): <=1 passes the allclose band, ~1 is a
                    # dtype ULP just over, >>1 is a structural divergence.
                    tol_denom = tol.atol + tol.rtol * ref_f.abs()
                    sv.max_tol_units[key] = float(
                        (diff / tol_denom.clamp_min(1e-12)).max().item()
                    )
                    # Physical output scale (for the Mode-B dtype-ceiling bound).
                    sv.max_ref_abs[key] = float(ref_f.abs().max().item())
                else:
                    sv.max_abs_diff[key] = float("inf")
                    sv.max_rel_diff[key] = float("inf")
                    sv.max_tol_units[key] = float("inf")
                    sv.max_ref_abs[key] = float("inf")
                # Tolerance-aware verdict — compare returns ComparisonResult,
                # NOT bool. Check the .passed field; earlier code treated the
                # whole object as truthy which produced false-positive VERIFIED.
                result = compare(ker_t, ref_t, tol)
                if not result.passed:
                    all_pass = False
                    sv.failing_keys.append(key)
                    # Acceptance evidence: sparse + unbiased = clean ceiling;
                    # high density AND same-sign mean = a masked systematic error.
                    if ref_f.shape == ker_f.shape and diff.numel() > 0:
                        band = (tol.atol + tol.rtol * ref_f.abs())
                        mean_abs = float(diff.mean().item()) or 1e-30
                        rms = float(diff.pow(2).mean().sqrt().item())
                        mx = sv.max_abs_diff.get(key, 0.0) or 1e-30
                        sv.accept_stats[key] = {
                            "fail_density": float((diff > band).float().mean().item()),
                            "signed_mean_ratio": abs(float((ker_f - ref_f).mean().item())) / mean_abs,
                            "rms_over_max": rms / mx,
                            "max_abs": sv.max_abs_diff.get(key),
                            "max_ref_abs": sv.max_ref_abs.get(key),
                        }
                    fail_reasons.append(
                        f"{key}: {result.reason or 'tolerance fail'} "
                        f"(max_abs={sv.max_abs_diff.get(key, '?')}, "
                        f"max_rel={sv.max_rel_diff.get(key, '?')}, "
                        f"tol={tol.name} atol={tol.atol} rtol={tol.rtol}"
                        f"{' bit_exact' if tol.bit_exact else ''})"
                    )
            except Exception as e:
                sv.blocker_reason = f"compare error on {key}: {e}"
                sv.compare_error = True  # a real exception -> ceiling must fail-closed
                all_pass = False
        if fail_reasons and not sv.blocker_reason:
            sv.blocker_reason = "; ".join(fail_reasons[:3])

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
            _accepted, _mode, _reason = _ceiling_verdict(task_id, sv)
            if _accepted:
                # Accepted dtype-ULP ceiling: strict-fails but an allowlisted task's
                # mode predicate holds (Mode-A tol_units cap / Mode-B per-key physical
                # scale). DISTINCT from VERIFIED -- reported + counted separately.
                sv.status = "CEILING_ACCEPTED"
                sv.accept_reason = f"{_mode}: {_reason}"
                n_ceiling_accepted += 1
            else:
                sv.status = "FAILED"
                n_failed += 1

        verdict.shapes.append(sv)
        # Diagnostic only (NOT a gate): worst backward/physical ULP =
        # max_abs / (max|ref| * bf16_ULP). Shows how a "<= N ULP of the output
        # scale" bound would grade each shape, while tol_units stays the actual
        # gate (a pure backward bound mis-rejects L2_03's independent-maxima
        # clean ceiling and stops protecting small-reference elements).
        _BF16_ULP = 0.00390625
        _back_ulp = 0.0
        for _bk, _bma in sv.max_abs_diff.items():
            _bmr = sv.max_ref_abs.get(_bk, 0.0)
            if isinstance(_bmr, (int, float)) and _bmr > 0.0:
                _back_ulp = max(_back_ulp, _bma / (_bmr * _BF16_ULP))
        _extra = ""
        if sv.status == "CEILING_ACCEPTED":
            _st = next(iter(sv.accept_stats.values()), {})
            _extra = (
                f" [{sv.accept_reason}"
                + (f" fail_density={_st.get('fail_density'):.2g}"
                   f" signed_mean_ratio={_st.get('signed_mean_ratio'):.2g}"
                   if _st else "")
                + "]"
            )
        log.info(
            "  %s/%s: %s (max_abs=%.3e max_rel=%.3e max_tol_units=%.3g back_ulp=%.2f)%s",
            task_id,
            sid,
            sv.status,
            max(sv.max_abs_diff.values()) if sv.max_abs_diff else 0.0,
            max(sv.max_rel_diff.values()) if sv.max_rel_diff else 0.0,
            max(sv.max_tol_units.values()) if sv.max_tol_units else 0.0,
            _back_ulp,
            _extra,
        )

    if n_input_drift > 0:
        # Any drifted shape makes the task's precision verdict untrustworthy;
        # never report VERIFIED/FAILED on partial drift.
        verdict.overall = "INPUT_DRIFT"
    elif n_blocked == len(verdict.shapes):
        # All shapes blocked. Distinguish wrapper-rank mismatches
        # from real GPU launch failures so the operator sees the
        # actionable verdict.
        shape_statuses = {s.status for s in verdict.shapes}
        if shape_statuses == {"WRAPPER_CONTRACT_MISMATCH"}:
            verdict.overall = "WRAPPER_CONTRACT_MISMATCH"
        elif shape_statuses == {"ORACLE_KERNEL_LAYOUT_MISMATCH"}:
            verdict.overall = "ORACLE_KERNEL_LAYOUT_MISMATCH"
        elif shape_statuses <= {
            "WRAPPER_CONTRACT_MISMATCH",
            "ORACLE_KERNEL_LAYOUT_MISMATCH",
        }:
            # Mixed pre-check failures (different shapes hit different
            # static-check buckets). Surface the stronger signal --
            # any ORACLE_KERNEL_LAYOUT_MISMATCH means the task needs
            # source ② regardless of how the other shape classified.
            verdict.overall = "ORACLE_KERNEL_LAYOUT_MISMATCH"
        else:
            verdict.overall = "LAUNCH_BLOCKED"
    elif n_failed == 0 and n_blocked == 0 and n_ceiling_accepted > 0:
        # No strict failure AND no blocked shape; one or more shapes accepted at the
        # dtype-ULP ceiling. DISTINCT from VERIFIED so the report never conflates
        # strict-pass with ceiling. A partial LAUNCH_BLOCKED falls through to MIXED.
        verdict.overall = "CEILING_ACCEPTED"
    elif n_failed == 0 and n_blocked == 0 and n_verified > 0:
        verdict.overall = "VERIFIED"
    elif n_verified == 0 and n_ceiling_accepted == 0 and n_blocked == 0 and n_failed > 0:
        verdict.overall = "FAILED"
    else:
        verdict.overall = "MIXED"

    # Backstop: refuse VERIFIED if an in-place mutated key was never compared
    # (e.g. a stale ref captured before mutated-output support). An honest
    # block beats a false VERIFIED.
    if verdict.overall == "VERIFIED":
        try:
            _task_meta = yaml.safe_load((task_dir / "meta.yaml").read_text()) or {}
        except Exception:
            _task_meta = {}
        _op = task_dir / "oracle_pytorch.py"
        _oracle_src = _op.read_text() if _op.exists() else ""
        _uncompared = sorted(
            _mutated_input_keys(_task_meta, _oracle_src) - ref_keys_seen
        )
        if _uncompared:
            verdict.overall = "INPLACE_UNCOMPARED"
            verdict.shapes.append(ShapeVerdict(
                shape_id="*",
                status="INPLACE_UNCOMPARED",
                blocker_reason=(
                    f"in-place task mutates {_uncompared} but the ref omits them "
                    f"(ref keys={sorted(ref_keys_seen)}); re-run R10.a to capture "
                    f"mutated outputs."
                ),
            ))
            log.warning(
                "  %s: VERIFIED -> INPLACE_UNCOMPARED (%s not in ref)",
                task_id, _uncompared,
            )

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

    summary = {
        "verified": 0,
        "ceiling_accepted": 0,
        "failed": 0,
        "mixed": 0,
        "blocked": 0,
        "skipped": 0,
        "wrapper_contract_mismatch": 0,
        "oracle_kernel_layout_mismatch": 0,
    }
    for td in sorted(task_dirs):
        v = _verify_one_task(td, args.infer_root, args.gpu)
        verdict_path = args.output_dir / f"{td.name}.json"
        verdict_path.write_text(json.dumps(asdict(v), indent=2, default=str))

        key = {
            "VERIFIED": "verified",
            "CEILING_ACCEPTED": "ceiling_accepted",
            "FAILED": "failed",
            "MIXED": "mixed",
            "LAUNCH_BLOCKED": "blocked",
            "SKIPPED": "skipped",
            "WRAPPER_CONTRACT_MISMATCH": "wrapper_contract_mismatch",
            "ORACLE_KERNEL_LAYOUT_MISMATCH": "oracle_kernel_layout_mismatch",
        }.get(v.overall, "blocked")
        summary[key] += 1

    log.info("--- summary ---")
    for k, n in summary.items():
        log.info("  %s: %d", k, n)
    # Aggregate "needs human action" tally for the operator at a glance.
    actionable_block = (
        summary.get("wrapper_contract_mismatch", 0)
        + summary.get("oracle_kernel_layout_mismatch", 0)
    )
    if actionable_block:
        log.info(
            "  (note: %d task(s) flagged statically -- shapes.yaml patch "
            "or source-2 reroute, NOT a GPU launch failure)",
            actionable_block,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
