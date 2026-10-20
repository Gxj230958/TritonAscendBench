"""Per-task root-cause diagnostic for the 100-task corpus.

Read-only analyzer. For each BLOCKED/UNRUN task, cross-references:
  - resolver verdict (status + per-entrypoint segment) from results/resolve_all.json
  - meta.yaml (intended kernel name + oracle config + upstream path)
  - kernel.py (intended function signature + parameters + asserts)
  - shapes.yaml (declared input rows: param names + dtypes + dims)
  - oracle_pytorch.py (expected input contract)

Outputs a per-task structured report with:
  - resolver surface error (verbatim segment for the INTENDED entrypoint)
  - intended kernel signature (parsed param list with defaults)
  - shapes.yaml declared inputs (param -> dtype/dims/dist or scalar)
  - DIFF: kernel-required params not in shapes.yaml
  - DIFF: kernel asserts that look violated by current shapes
  - root cause category (one of: missing-shape-row / wrong-dtype / wrong-shape /
    assertion-violated / import-error / kernel-assert-on-knob / oracle-divergence /
    interpreter-numeric / triton-version-incompat / sigsegv / multi-entrypoint /
    other)
  - fix recipe: concrete shapes.yaml / meta.yaml / kernel.py / oracle_pytorch.py
    diff sketch
  - effort estimate: S (≤30min) / M (30-120min) / L (≥120min)

Usage::

    python3 scripts/internal/diagnose_blocked_tasks.py                # stdout summary
    python3 scripts/internal/diagnose_blocked_tasks.py --md out.md    # full markdown
    python3 scripts/internal/diagnose_blocked_tasks.py --task L1_30   # one task
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
CACHE = REPO / "results" / "resolve_all.json"


@dataclass
class TaskDiagnosis:
    task_id: str
    family: str
    aggregate_status: str
    intended_kernel_name: str = ""
    intended_segment: str = ""
    intended_substatus: str = ""
    kernel_required_params: list[str] = field(default_factory=list)
    kernel_optional_params: list[str] = field(default_factory=list)
    oracle_input_keys: set[str] = field(default_factory=set)
    shapes_inputs: dict[str, dict] = field(default_factory=dict)
    missing_in_shapes: list[str] = field(default_factory=list)
    abstraction_mismatch_params: list[str] = field(default_factory=list)
    case_collisions: list[tuple[str, str]] = field(default_factory=list)
    upstream_exception: str = ""
    upstream_exception_detail: str = ""
    root_cause_category: str = ""
    root_cause_detail: str = ""
    fix_recipe: list[str] = field(default_factory=list)
    effort: str = ""  # S/M/L


def _task_dir(task_id: str) -> Path:
    lvl = task_id.split("_", 1)[0].lower()
    return REPO / "tasks" / f"level{lvl[1:]}" / task_id


def _split_segments(reason: str) -> list[tuple[str, str, str]]:
    """Return list of ``(entrypoint_name, substatus, segment_text)``."""
    after_tried = re.split(r";\s+", reason, maxsplit=1)
    body = after_tried[1] if len(after_tried) > 1 else reason
    parts = re.split(r"\s\|\s", body)
    out: list[tuple[str, str, str]] = []
    for part in parts:
        m = re.match(r"\s*([A-Za-z_][\w\-]*):\s*([a-z][a-z_\-]+)\b", part)
        if m:
            out.append((m.group(1), m.group(2), part))
    return out


def _parse_oracle_inputs(oracle_src: str) -> set[str]:
    """Return the set of param names oracle_pytorch.py reads from
    ``inputs`` (e.g. ``inputs["X"]`` or ``inputs.get("X")``).

    The bench convention is a ``def run(inputs: dict) -> dict`` whose
    body indexes ``inputs[...]``. Extracting this set lets us detect
    oracle-vs-kernel abstraction mismatches: when kernel signature has
    `state_source` but oracle reads `state`, adding `state_source` to
    shapes.yaml does not restore comparability.
    """
    if not oracle_src:
        return set()
    keys = set(re.findall(r"""inputs\[(?:["']([^"']+)["'])\]""", oracle_src))
    keys |= set(re.findall(r"""inputs\.get\((?:["']([^"']+)["'])""", oracle_src))
    return keys


def _meta_intended_kernel_name(meta: dict) -> str:
    """Return the raw meta.name (no wrapper-vs-jit resolution)."""
    name = (meta or {}).get("name")
    if isinstance(name, str) and name:
        return name
    gt = ((meta or {}).get("oracle") or {}).get("gpu_triton") or ""
    m = re.search(r"::([A-Za-z_][\w]*)", gt)
    if m and m.group(1) != "placeholder":
        return m.group(1)
    return ""


def _is_triton_jit_decorated(kernel_src: str, fn_name: str) -> bool:
    """True if `def fn_name(...)` is preceded by a @triton.jit (or
    .autotune / .heuristics chain ending in jit) decorator."""
    if not kernel_src or not fn_name:
        return False
    try:
        tree = ast.parse(kernel_src)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            for dec in node.decorator_list:
                # @triton.jit -> Attribute(value=Name("triton"), attr="jit")
                # @triton.autotune(...) -> Call(func=Attribute(value=Name("triton"), attr="autotune"))
                # @triton.heuristics(...) -> Call(...)
                target = dec.func if isinstance(dec, ast.Call) else dec
                if isinstance(target, ast.Attribute):
                    if isinstance(target.value, ast.Name) and target.value.id == "triton":
                        if target.attr in ("jit", "autotune", "heuristics"):
                            return True
            return False
    return False


def _find_user_callable_for(
    kernel_src: str,
    meta_name: str,
    resolver_tried: list[str] | None = None,
) -> tuple[str, str]:
    """Return ``(user_callable_name, kind)``.

    Resolution order (most authoritative first):
      1. If any name in ``resolver_tried`` exists as a non-jit ``def`` in
         kernel.py, return it. The resolver's candidate-search already
         identified what to call; trust it over our heuristic.
      2. If meta.name itself is a non-jit ``def``, use it.
      3. If meta.name is a jit ``def``, look for a non-jit wrapper whose
         name overlaps. Prefer the LONGEST overlap (most-specific wrapper)
         over the shortest -- shorter names often catch a sibling
         (`dequantize_k_cache` instead of `dequantize_k_cache_paged`).
      4. Fall back to meta.name itself.

    A kernel.py user-callable is a python ``def`` not decorated with
    @triton.jit / .autotune / .heuristics.
    """
    if not kernel_src:
        return ("", "meta-name-missing")
    try:
        tree = ast.parse(kernel_src)
    except SyntaxError:
        return ("", "meta-name-missing")
    defs: list[tuple[str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defs.append((node.name, _is_triton_jit_decorated(kernel_src, node.name)))

    def_names_nonjit = {n for n, is_jit in defs if not is_jit}

    # 1. Resolver-tried wins, BUT only if the name has meaningful overlap
    # with meta.name. Without overlap, the resolver's candidate-search
    # picked an unrelated sibling (e.g. L1_04: meta is _copy_page_indices_kernel,
    # resolver tried trtllm_prefill_attn_kvfp8_dequant -- zero overlap,
    # don't trust it).
    if resolver_tried and meta_name:
        base = meta_name.lstrip("_")
        if base.endswith("_kernel"):
            base = base[: -len("_kernel")]
        for cand in resolver_tried:
            if cand in def_names_nonjit and (
                base in cand or cand in meta_name or
                # Allow short tokens to match when they appear at word boundary.
                any(part for part in base.split("_") if len(part) >= 4 and part in cand)
            ):
                return (cand, "resolver-tried-wrapper")

    if not meta_name:
        return ("", "meta-name-missing")

    # 2 + 3 + 4.
    for n, is_jit in defs:
        if n == meta_name:
            if not is_jit:
                return (n, "meta-name-is-wrapper")
            base = meta_name.lstrip("_")
            if base.endswith("_kernel"):
                base = base[: -len("_kernel")]
            candidates = [
                n2 for n2, is_jit2 in defs
                if not is_jit2 and (base in n2 or n2 in meta_name)
                and n2 != meta_name
            ]
            if candidates:
                # Prefer LONGEST overlap (most-specific wrapper) over shortest;
                # shortest often catches a sibling.
                candidates.sort(key=lambda c: -len(c))
                return (candidates[0], "meta-name-is-jit-found-wrapper")
            return (meta_name, "meta-name-is-jit-no-wrapper")
    return ("", "meta-name-missing")


def _pick_intended_segment(
    task_id: str,
    segments: list[tuple[str, str, str]],
    intended_canonical: str = "",
) -> tuple[str, str, str]:
    if not segments:
        return ("", "", "")
    # Prefer the canonical name from meta.yaml when supplied; fall back to
    # task_id-stripped form. Without normalization the matcher drifted to
    # sibling wrappers when the task_id name and the resolver's chosen
    # entrypoint name only loosely overlap (e.g. L1_04 task_id derives
    # ``copy_page_indices`` but the resolver tried ``trtllm_prefill_*``).
    intended = intended_canonical or re.sub(r"^L\d+_\d+_", "", task_id)
    def norm(s):
        return s.replace("-", "_")
    intended_n = norm(intended)
    for name, sub, seg in segments:
        if norm(name) == intended_n:
            return (name, sub, seg)
    # Bidirectional substring match only when the canonical name has
    # meaningful overlap (>= 6 chars to avoid matching everything).
    if len(intended_n) >= 6:
        for name, sub, seg in segments:
            name_n = norm(name)
            if intended_n in name_n or (len(name_n) >= 6 and name_n in intended_n):
                return (name, sub, seg)
    # No match -- intended kernel name is not in the resolver's tried set.
    # Returning empty signals the caller to fall back to ``meta.name``
    # for kernel.py signature inspection rather than trusting a wrong
    # sibling.
    return ("", "", "")


# Heuristic filter for @triton.jit kernel-internal params that are NEVER
# user-callable inputs (the host wrapper allocates / derives these and
# passes them down). DO NOT add single-letter filters here -- single
# letters are commonly real user tensors (q, k, v in attention; a, b in
# gemm; b for batch as scalar). Filtering them silently drops genuine
# missing-input findings.
_INTERNAL_PARAM_PATTERNS = (
    re.compile(r"_ptr$"),                 # raw pointer args (Q_ptr, output_ptr, etc.)
    re.compile(r"^BLOCK(?:_|$)"),         # BLOCK_SIZE, BLOCK_M, BLOCK_DMODEL, BLOCK
    re.compile(r"^NUM_(?:BLOCK|FBLOCK|CBLOCK)"),
    re.compile(r"^stride_"),               # stride_x_row, stride_y_row
    re.compile(r"^(?:HAS|IS|USE|STORE|NORM|DOT)_"),  # boolean tl.constexpr flags
    # ALL_CAPS at least 2 chars = tl.constexpr meta-param. The 2+ floor
    # avoids stripping single-letter `M` / `N` / `K`; when ambiguous,
    # prefer surfacing in `missing` so the operator can decide.
    re.compile(r"^[A-Z][A-Z0-9_]+$"),
)


def _is_internal_kernel_param(name: str) -> bool:
    return any(p.search(name) for p in _INTERNAL_PARAM_PATTERNS)


def _parse_kernel_signature(kernel_src: str, fn_name: str) -> tuple[list[str], list[str]]:
    """Return ``(user_required_params, optional_params)`` for ``def fn_name(...)``.

    Filters @triton.jit kernel-internal params (pointers, BLOCK sizes,
    strides, tl.constexpr flags) which the host wrapper supplies; user
    inputs in shapes.yaml only need to satisfy the host-wrapper signature
    or the @triton.jit signature's user-facing tensor/scalar params.
    """
    if not kernel_src or not fn_name:
        return ([], [])
    try:
        tree = ast.parse(kernel_src)
    except SyntaxError:
        return ([], [])
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
            args = node.args
            n_defaults = len(args.defaults)
            n_args = len(args.args)
            required_raw = [a.arg for a in args.args[: n_args - n_defaults]]
            optional_raw = [a.arg for a in args.args[n_args - n_defaults :]]
            # Strip method receivers (self/cls) and kernel-internals.
            required = [
                a for a in required_raw
                if a not in ("self", "cls") and not _is_internal_kernel_param(a)
            ]
            optional = [
                a for a in optional_raw
                if a not in ("self", "cls") and not _is_internal_kernel_param(a)
            ]
            return (required, optional)
    return ([], [])


def _parse_shapes_inputs(shapes_src: str) -> dict[str, dict]:
    """Return ``{name: {dtype, dims, dist, scalar, ...}}`` from shapes.yaml."""
    try:
        doc = yaml.safe_load(shapes_src) or {}
    except yaml.YAMLError:
        return {}
    inputs = doc.get("inputs") or {}
    if not isinstance(inputs, dict):
        return {}
    return {k: v if isinstance(v, dict) else {"value": v} for k, v in inputs.items()}


def _extract_upstream_exception(segment: str) -> tuple[str, str]:
    """Return ``(exception_type, short_detail)`` from a segment like
    ``foo: upstream-not-invocable-here [SomeError: details...]``."""
    m = re.search(r"\[([A-Za-z][A-Za-z0-9]*(?:Error|Exception)?):?\s*([^\]]{0,200})", segment)
    if m:
        return (m.group(1), m.group(2).strip().rstrip(" |"))
    # No bracketed; look for 'missing param' (shapes-incomplete) phrasing
    mp = re.search(r"missing param '([^']+)'", segment)
    if mp:
        return ("ShapesIncompleteError", f"missing param '{mp.group(1)}'")
    return ("", "")


def _diagnose_one(verdict: dict) -> TaskDiagnosis:
    tid = verdict["task_id"]
    td = _task_dir(tid)
    diag = TaskDiagnosis(
        task_id=tid,
        family="?",
        aggregate_status=verdict.get("status", ""),
    )

    meta_path = td / "meta.yaml"
    try:
        meta = yaml.safe_load(meta_path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        meta = {}
    diag.family = (meta.get("category") or "?")

    # Always parse kernel.py and resolver text first so UNRUN short-
    # circuit can still report the verified entrypoint name.
    try:
        kernel_src = (td / "kernel.py").read_text()
    except OSError:
        kernel_src = ""
    reason = verdict.get("reason", "") or ""
    segments = _split_segments(reason)
    meta_kernel_name = _meta_intended_kernel_name(meta)
    # Parse the resolver-tried list from reason like ``tried ['a','b']``.
    tried_match = re.search(r"tried \[([^\]]*)\]", reason)
    resolver_tried: list[str] = []
    if tried_match:
        resolver_tried = re.findall(r"'([^']+)'", tried_match.group(1))

    # If resolver already verified an entrypoint (PASS / UNRUN / oracle-
    # consistent), use that as the canonical user-callable. The
    # ``resolved kernel.py::<name>`` form is higher-truth than our
    # heuristic wrapper search.
    resolved_match = re.search(r"resolved kernel\.py::([A-Za-z_][\w]*)", reason)
    if resolved_match:
        user_callable = resolved_match.group(1)
        kind = "resolver-verified"
    else:
        user_callable, kind = _find_user_callable_for(
            kernel_src, meta_kernel_name, resolver_tried=resolver_tried
        )

    sig_target_for_short_circuit = user_callable or meta_kernel_name

    # PASS / UNRUN short-circuit -- still populate intended_kernel_name.
    if "PASS" in diag.aggregate_status:
        diag.intended_kernel_name = sig_target_for_short_circuit
        diag.root_cause_category = "pass"
        diag.fix_recipe = []
        diag.effort = ""
        return diag
    if "UNRUN" in diag.aggregate_status:
        diag.intended_kernel_name = sig_target_for_short_circuit
        diag.root_cause_category = "needs-gpu"
        diag.root_cause_detail = (
            f"Resolver verified `{sig_target_for_short_circuit}` as "
            "oracle-consistent on dev; Phase A' on GPU/NPU is the only "
            "missing step before PASS."
        )
        diag.fix_recipe = [
            "On GPU host: bash scripts/automation/runners/tab_quickstart.sh verify-all",
            f"Or per-task: python3 scripts/verify/standalone_freeze_verifier.py --run --task {tid}",
            "Expected outcome: PASS after Phase A' completes",
        ]
        diag.effort = "S"
        return diag

    # Intended-kernel-name for downstream use: the USER CALLABLE
    # (wrapper if found, else meta.name itself).
    sig_target = user_callable or meta_kernel_name

    # Match resolver segment on EITHER the user callable OR meta.name
    # (resolver may have tried either).
    name, sub, seg = _pick_intended_segment(tid, segments, intended_canonical=sig_target)
    if not name and sig_target != meta_kernel_name:
        name, sub, seg = _pick_intended_segment(tid, segments, intended_canonical=meta_kernel_name)

    diag.intended_kernel_name = sig_target
    diag.intended_substatus = sub
    diag.intended_segment = seg

    # Parse signature on user_callable (host wrapper preferred). When
    # the wrapper is found, its param list IS the user-facing input
    # contract -- no internal _ptr / BLOCK_ / strides leak through.
    # Skip signature parsing when sig_target is the @triton.jit kernel
    # itself with no wrapper -- its params are kernel-internal and
    # cannot be authored in shapes.yaml; the real fix is upstream meta
    # or wrapper-add, not shape rows.
    sig_target_is_jit_no_wrapper = (
        kind == "meta-name-is-jit-no-wrapper"
        or (sig_target == meta_kernel_name and _is_triton_jit_decorated(kernel_src, sig_target))
    )
    if sig_target and kernel_src and not sig_target_is_jit_no_wrapper:
        required, optional = _parse_kernel_signature(kernel_src, sig_target)
        if not required and not optional and meta_kernel_name and meta_kernel_name != sig_target:
            required, optional = _parse_kernel_signature(kernel_src, meta_kernel_name)
        diag.kernel_required_params = required
        diag.kernel_optional_params = optional

    # Parse shapes.yaml declared inputs + oracle_pytorch.py inputs.
    try:
        shapes_src = (td / "shapes.yaml").read_text()
    except OSError:
        shapes_src = ""
    try:
        oracle_src = (td / "oracle_pytorch.py").read_text()
    except OSError:
        oracle_src = ""
    diag.shapes_inputs = _parse_shapes_inputs(shapes_src)
    diag.oracle_input_keys = _parse_oracle_inputs(oracle_src)
    declared_in_shapes = set(diag.shapes_inputs.keys())

    # Detect case-insensitive name collisions between shapes and oracle
    # (e.g. shapes declares `BLOCK_SIZE` but oracle reads `block_size`).
    # This is a contract bug that surfaces as ``shapes-incomplete`` or
    # oracle-mismatch but the real fix is renaming one or the other.
    if diag.oracle_input_keys:
        shapes_lower = {k.lower(): k for k in declared_in_shapes}
        for ok in diag.oracle_input_keys:
            if ok not in declared_in_shapes and ok.lower() in shapes_lower:
                actual = shapes_lower[ok.lower()]
                if actual != ok:
                    diag.case_collisions.append((actual, ok))

    # Detect oracle-kernel abstraction mismatch: kernel signature has
    # params that oracle does NOT read (oracle and kernel are written
    # at different representation layers). When this fires, adding the
    # missing param to shapes.yaml alone will not restore comparability
    # -- the oracle or the wrapper needs realignment.
    if diag.kernel_required_params and diag.oracle_input_keys:
        oracle_lower = {k.lower() for k in diag.oracle_input_keys}
        diag.abstraction_mismatch_params = [
            p for p in diag.kernel_required_params
            if p not in diag.oracle_input_keys
            and p.lower() not in oracle_lower
        ]

    # Compute missing params: required not in shapes.
    if diag.kernel_required_params:
        diag.missing_in_shapes = [
            p for p in diag.kernel_required_params if p not in declared_in_shapes
        ]

    # Extract upstream exception detail.
    exc, exc_detail = _extract_upstream_exception(seg or reason)
    diag.upstream_exception = exc
    diag.upstream_exception_detail = exc_detail

    # Root-cause categorization + fix recipe.
    _classify_root_cause(diag, kernel_src, meta, reason)
    return diag


def _classify_root_cause(
    diag: TaskDiagnosis, kernel_src: str, meta: dict, full_reason: str
) -> None:
    """Populate diag.root_cause_category + .root_cause_detail + .fix_recipe + .effort."""
    sub = diag.intended_substatus
    exc = diag.upstream_exception
    exc_detail = diag.upstream_exception_detail
    name = diag.intended_kernel_name

    # Priority: case-collision and abstraction-mismatch are deeper bugs
    # than the resolver's surface "missing param" claim. Detect them
    # first; fall through to the substatus branches only if neither
    # applies.
    if diag.case_collisions:
        pairs = ", ".join(f"`{a}`/`{b}`" for a, b in diag.case_collisions[:3])
        diag.root_cause_category = "case-collision"
        diag.root_cause_detail = (
            f"Contract name drift: shapes.yaml declares "
            f"{[a for a,_ in diag.case_collisions]} but oracle_pytorch.py "
            f"reads {[b for _,b in diag.case_collisions]} (case-only difference). "
            f"Resolver surface error masks the real contract bug."
        )
        diag.fix_recipe = [
            "Pick ONE casing across all three: kernel.py / shapes.yaml / oracle_pytorch.py",
            "Rename in the minority file (usually the oracle or kernel) so all match",
            f"Collisions: {pairs}",
        ]
        diag.effort = "S"
        return

    # Abstraction-mismatch only when kernel needs params oracle doesn't
    # AND those params represent a different state representation
    # (heuristic: include 2+ kernel-required params absent from oracle's
    # input-key set). A single-param difference is often just optional /
    # constexpr; we tolerate that to avoid false positives.
    if (
        sub == "shapes-incomplete"
        and len(diag.abstraction_mismatch_params) >= 2
        and diag.oracle_input_keys
    ):
        kernel_only = diag.abstraction_mismatch_params
        oracle_only = sorted(
            k for k in diag.oracle_input_keys
            if k not in diag.kernel_required_params
            and k not in diag.kernel_optional_params
        )
        diag.root_cause_category = "oracle-kernel-abstraction-mismatch"
        diag.root_cause_detail = (
            f"Kernel `{name}` requires params {kernel_only} that "
            f"oracle_pytorch.py does not read; oracle reads "
            f"{oracle_only} not in the kernel signature. Adding shape "
            f"rows for the kernel-only params will not restore "
            f"comparability -- they encode a different state "
            f"representation than the oracle is authored against."
        )
        # Enumerate shape rows still missing AFTER the abstraction
        # disagreement is resolved (params in BOTH kernel and oracle
        # but not in shapes.yaml). Operator must add these too.
        declared_now = set(diag.shapes_inputs.keys())
        shared_missing = sorted(
            p for p in diag.kernel_required_params
            if p in diag.oracle_input_keys
            and p not in declared_now
        )
        recipe = [
            "Decide on the canonical contract layer: kernel or oracle?",
            f"Option A: rewrite oracle_pytorch.py to accept "
            f"{kernel_only} and compute the same math (often via a "
            f"reference implementation that mimics the wrapper's data layout)",
            f"Option B: rewrite the kernel-side wrapper to accept "
            f"{oracle_only} (matching oracle's representation)",
        ]
        if shared_missing:
            recipe.append(
                f"After the contract layer is decided, also add shape rows "
                f"for {shared_missing} (these appear in BOTH kernel and "
                f"oracle but not in shapes.yaml)."
            )
        else:
            recipe.append("Add shapes.yaml entries only AFTER the contract layer is decided")
        diag.fix_recipe = recipe
        diag.effort = "L"
        return

    if sub == "shapes-incomplete":
        # The resolver's "missing param 'X'" inside the intended segment.
        # Cross-check against CURRENT shapes.yaml (cache may be stale).
        declared = set(diag.shapes_inputs.keys())
        mp = re.search(r"missing param '([^']+)'", diag.intended_segment)
        param = mp.group(1) if mp else (diag.missing_in_shapes[0] if diag.missing_in_shapes else "?")
        is_known_to_kernel = (param in diag.kernel_required_params) or (param in diag.kernel_optional_params)
        if param in declared:
            # Stale cache: resolver complained about this param but
            # shapes.yaml already provides it. Don't trust the cache;
            # diagnose what's TRULY missing from kernel signature vs
            # shapes.yaml right now.
            still_missing = [p for p in diag.kernel_required_params if p not in declared]
            if still_missing:
                param = still_missing[0]
                diag.missing_in_shapes = still_missing
                diag.root_cause_category = "missing-shape-row"
                diag.root_cause_detail = (
                    f"Resolver cache is STALE: it claims `{mp.group(1) if mp else '?'}` "
                    f"missing but shapes.yaml already declares it. "
                    f"Current real gap: kernel `{name}` requires params "
                    f"{still_missing} that shapes.yaml does not declare."
                )
                diag.fix_recipe = [
                    f"First refresh cache for this task: "
                    f"python3 scripts/verify/resolve_upstream_entrypoints.py --task {diag.task_id} --json /tmp/r_{diag.task_id}.json",
                    "Then merge the fresh verdict into results/resolve_all.json",
                    f"Then add the still-missing params to shapes.yaml: {still_missing}",
                ]
                diag.effort = "M"
            else:
                diag.root_cause_category = "stale-cache-resolved"
                diag.root_cause_detail = (
                    f"Resolver cache is STALE: claims `{param}` missing but shapes.yaml "
                    f"already provides every kernel-required param. This task likely "
                    f"PASSES after a cache refresh."
                )
                diag.fix_recipe = [
                    f"Refresh cache for this task: "
                    f"python3 scripts/verify/resolve_upstream_entrypoints.py --task {diag.task_id} --json /tmp/r_{diag.task_id}.json",
                    "Merge into results/resolve_all.json, then re-run this diagnostic to confirm new status",
                ]
                diag.effort = "S"
        elif is_known_to_kernel:
            # Enumerate ALL kernel-required params not in shapes (multi-blocker).
            all_missing = [p for p in diag.kernel_required_params if p not in declared]
            diag.missing_in_shapes = all_missing
            diag.root_cause_category = "missing-shape-row"
            diag.root_cause_detail = (
                f"Kernel function `{name}` requires {len(all_missing)} param(s) "
                f"that shapes.yaml does not declare: {all_missing}. "
                f"Resolver reports `{param}` first; subsequent params surface in cascade."
            )
            diag.fix_recipe = [
                f"Open tasks/{_task_dir(diag.task_id).name}/shapes.yaml",
                f"Add ALL {len(all_missing)} missing entries: {all_missing}",
                "For each, refer to kernel.py signature for dtype + dims + dist semantics",
                "Re-run resolver after each batch to confirm progress",
            ]
            diag.effort = "S" if (len(all_missing) == 1 and param in ("chunk_size", "max_seq_len")) or all(p.startswith("BLOCK_SIZE") for p in all_missing) else ("M" if len(all_missing) <= 5 else "L")
        else:
            diag.root_cause_category = "shapes-incomplete-sibling"
            diag.root_cause_detail = (
                f"Resolver's missing-param mention (`{param}`) belongs to a sibling "
                f"candidate, not the intended kernel `{name}`. The aggregate "
                f"shapes-incomplete masks the intended kernel's true status."
            )
            diag.fix_recipe = [
                f"Verify intended kernel `{name}` actually exists in kernel.py",
                f"If `{name}` is correct, the resolver may be trying wrong siblings -- "
                "fix meta.upstream.path or meta.oracle.gpu_triton to point at the right symbol",
            ]
            diag.effort = "M"

    elif sub == "upstream-not-invocable-here" or sub == "upstream-not-invocable":
        if "No module named" in diag.intended_segment:
            mm = re.search(r"No module named '([^']+)'", diag.intended_segment)
            mod = mm.group(1) if mm else "?"
            diag.root_cause_category = "import-error"
            diag.root_cause_detail = f"Upstream import chain fails: ModuleNotFoundError for `{mod}`."
            diag.fix_recipe = [
                f"Option A: `pip install {mod}` (if pure-Python, low risk)",
                "Option B: oracle_pytorch.py implement a standalone reference (no upstream import)",
                "Option C: GPU-only -- defer to Phase B if module is CUDA-bound",
            ]
            diag.effort = "M"
        elif exc == "KeyError":
            key = re.search(r"KeyError: '?([^'\)]+)'?", diag.intended_segment)
            ks = key.group(1) if key else "?"
            diag.root_cause_category = "kernel-config-keyerror"
            diag.root_cause_detail = (
                f"Upstream raises KeyError(`{ks}`). The missing dict key is provided by an "
                f"upstream config layer / model_executor caller that does not initialise on dev."
            )
            diag.fix_recipe = [
                f"Read the call site to identify which dict expected `{ks}`",
                "Option A: shim the dict in oracle_pytorch.py with a known-good default",
                "Option B: rewrite oracle_pytorch.py as a standalone reference (no upstream import)",
            ]
            diag.effort = "M"
        elif "num_compute_units" in diag.intended_segment:
            diag.root_cause_category = "platform-stub-noimpl"
            diag.root_cause_detail = (
                "Upstream calls vllm.platforms.current_platform.num_compute_units() which "
                "raises NotImplementedError on the dev host's stub platform."
            )
            diag.fix_recipe = [
                "Option A: mock num_compute_units in tests/conftest.py or shim layer",
                "Option B: rewrite oracle_pytorch.py to not depend on platform helpers",
            ]
            diag.effort = "M"
        elif "AssertionError" in diag.intended_segment:
            assertion = re.search(r"AssertionError:?\s*([^|\]]{0,200})", diag.intended_segment)
            astxt = assertion.group(1).strip() if assertion else ""
            diag.root_cause_category = "kernel-assertion"
            diag.root_cause_detail = (
                f"Kernel function `{name}` hits an AssertionError when called with the "
                f"current shapes: `{astxt}`"
            )
            diag.fix_recipe = [
                "Read the asserted condition in kernel.py",
                "Adjust shapes.yaml so the input tensors satisfy the assertion",
                "(e.g. paired dtype, divisibility, shape compatibility)",
            ]
            diag.effort = "M"
        elif "Tried to register an operator" in diag.intended_segment or "Duplicate registration" in diag.intended_segment:
            diag.root_cause_category = "duplicate-op-registration"
            diag.root_cause_detail = (
                "Upstream import triggers torch.library duplicate operator registration -- "
                "another vendored copy registered the same op first."
            )
            diag.fix_recipe = [
                "Identify the duplicate registration path (usually two vllm copies)",
                "Option A: dedupe sys.path so only one vllm import wins",
                "Option B: oracle_pytorch.py rewrite to not import vllm",
            ]
            diag.effort = "L"
        elif "ZeroDivisionError" in diag.intended_segment:
            diag.root_cause_category = "kernel-zerodivision"
            diag.root_cause_detail = "Shape sweep produces a zero-valued dim/stride the kernel does not guard."
            diag.fix_recipe = [
                "Identify which shape row triggers the zero (usually H2/H_singleton with extreme small knobs)",
                "Add a guard in shapes.yaml hidden_shapes or in oracle_pytorch.py for that row",
            ]
            diag.effort = "S"
        elif "AttributeError" in diag.intended_segment and "_AutotunedKernel" in diag.intended_segment:
            diag.root_cause_category = "triton-autotune-api-mismatch"
            diag.root_cause_detail = (
                "Kernel applies @triton.heuristics on top of a wrapper that lacks .arg_names. "
                "Current Triton API expects different ordering."
            )
            recipe = [
                "Pin Triton version OR rewrite kernel.py decorator ordering",
                "If oracle_pytorch.py is independent, defer kernel.py fix to Phase B",
            ]
            # Enumerate downstream blockers the operator will still hit.
            declared = set(diag.shapes_inputs.keys())
            also_missing = [p for p in diag.kernel_required_params if p not in declared]
            if also_missing:
                recipe.append(
                    f"After the Triton API issue is resolved, this task will still "
                    f"need shape rows added for {also_missing}."
                )
            diag.fix_recipe = recipe
            diag.effort = "L"
        elif "rc=-11" in diag.intended_segment or "SIGSEGV" in diag.intended_segment:
            diag.root_cause_category = "sigsegv"
            diag.root_cause_detail = "Subprocess SIGSEGV under CPU interpreter (likely Triton interpreter bug)."
            diag.fix_recipe = [
                "Phase B golden capture on GPU/NPU is the only authoritative path",
                "On dev: try TRITON_INTERPRET=0 to skip interpreter path",
            ]
            diag.effort = "L"
        elif exc:
            diag.root_cause_category = f"upstream-{exc.lower()}"
            diag.root_cause_detail = f"Upstream raises {exc}: {exc_detail[:200]}"
            diag.fix_recipe = [
                "Inspect the exception's call site in upstream",
                "Decide: shim, mock, or rewrite oracle_pytorch.py standalone",
            ]
            diag.effort = "M"
        else:
            diag.root_cause_category = "upstream-unclassified"
            diag.root_cause_detail = "Upstream raises an exception not matched by known patterns; see intended_segment."
            diag.fix_recipe = ["Read intended_segment in full and classify manually"]
            diag.effort = "M"

    elif sub == "oracle-mismatch":
        if "STRUCTURE MISMATCH" in diag.intended_segment or "wrong operator" in diag.intended_segment:
            diag.root_cause_category = "oracle-structure-mismatch"
            diag.root_cause_detail = (
                "Kernel output and oracle disagree on output structure (shape, dtype, or "
                "operator semantics)."
            )
            diag.fix_recipe = [
                "Read kernel.py + oracle_pytorch.py side by side",
                "Identify which output dim/dtype the oracle expects vs what kernel produces",
                "Fix oracle_pytorch.py to match the kernel's true output contract",
            ]
            diag.effort = "M"
        elif "dtype mismatch" in diag.intended_segment.lower():
            diag.root_cause_category = "oracle-dtype-mismatch"
            diag.root_cause_detail = "Output dtype divergence between kernel and oracle."
            diag.fix_recipe = [
                "Align oracle_pytorch.py output dtype with kernel.py's emitted dtype",
            ]
            diag.effort = "S"
        else:
            diag.root_cause_category = "oracle-numeric-tolerance"
            diag.root_cause_detail = (
                "Numeric divergence between kernel and oracle under the task's tolerance class. "
                "Likely CPU interpreter precision drift but may be real kernel bug."
            )
            diag.fix_recipe = [
                "Re-run on GPU to distinguish interpreter drift from real divergence",
                "If real, audit oracle_pytorch.py against kernel.py mathematically",
            ]
            diag.effort = "M"

    elif sub == "smoke-fail":
        diag.root_cause_category = "smoke-fail-cpu-budget"
        diag.root_cause_detail = (
            "S2 smoke (TRITON_INTERPRET) fails / times out on CPU. Often a compute-density "
            "issue (large GEMMs, atomic ops) where the interpreter is too slow."
        )
        diag.fix_recipe = [
            "Reduce smoke shape coverage to only H2 (smallest hidden shape)",
            "OR: skip smoke pre-filter for this task family if interpreter cannot run",
            "Phase B GPU run will verify",
        ]
        diag.effort = "S"

    elif not sub and name:
        # meta.name is set but the resolver did not even try this kernel
        # (its candidate-search picked unrelated wrappers). This is a
        # meta-vs-resolver mismatch.
        diag.root_cause_category = "meta-resolver-mismatch"
        is_jit_only = _is_triton_jit_decorated(kernel_src, name)
        diag.root_cause_detail = (
            f"Resolver candidate-search did not try the meta-declared kernel "
            f"`{name}`. Likely meta.upstream.path / meta.oracle.gpu_triton "
            f"points at a stale location, OR the kernel.py vendored function "
            f"is named differently than meta records. Resolver tried instead: "
            f"{[s[0] for s in _split_segments(full_reason)][:4]}"
        )
        if is_jit_only:
            diag.fix_recipe = [
                f"`{name}` is a bare @triton.jit kernel -- shapes.yaml "
                f"cannot author its launch-time pointer / BLOCK args directly",
                f"Write a host wrapper in kernel.py around `{name}` that "
                f"accepts user-level tensors and computes pointers / sizes",
                f"OR: locate the existing wrapper elsewhere in upstream "
                f"(grep `infer/{{vllm,sglang}}/` for callers of `{name}`) "
                f"and copy it into kernel.py",
                "Update meta.oracle.gpu_triton to point at the wrapper, "
                "then add shapes.yaml inputs matching the wrapper signature",
            ]
            diag.effort = "L"
        else:
            diag.fix_recipe = [
                f"Verify `{name}` exists in kernel.py (grep `def {name}`)",
                "If not, find the actual entrypoint name in kernel.py and update meta.name",
                "If yes, audit scripts/verify/resolve_upstream_entrypoints.py candidate-search "
                "to understand why this kernel was not tried",
                f"Then add shapes.yaml inputs matching `{name}`'s signature",
            ]
            diag.effort = "M"

    else:
        # Catch-all for special resolver outputs that don't follow the
        # `<entrypoint>: <substatus>` structure (SIGSEGV preambles,
        # severed-side-no-run reports, A-own deferred tasks with custom
        # phase_a_prime SKIP messages, etc.).
        if "rc=-11" in full_reason or "SIGSEGV" in full_reason:
            diag.root_cause_category = "sigsegv"
            diag.root_cause_detail = (
                "Resolver subprocess SIGSEGV under CPU interpreter. Crash is in "
                "third-party C (Triton interpreter / numpy backend); not reproducible "
                "on GPU/NPU."
            )
            diag.fix_recipe = [
                "Phase B golden capture on GPU/NPU is the authoritative path",
                "Optional: rewrite oracle_pytorch.py to avoid interpreter-only codepath",
            ]
            diag.effort = "L"
        elif "S2-smoke MISMATCH" in full_reason or "severed-no-run" in full_reason or "original-no-run" in full_reason:
            diag.root_cause_category = "smoke-fail-severed-run"
            diag.root_cause_detail = (
                "S2-smoke side process did not run (severed or original side rc!=0). "
                "Usually CPU interpreter cannot exec the kernel within budget."
            )
            diag.fix_recipe = [
                "Reduce smoke shape coverage to H2 (smallest hidden shape)",
                "Or skip CPU pre-filter for this task family",
            ]
            diag.effort = "M"
        elif "smoke-fail" in full_reason or "smoke fail" in full_reason.lower():
            diag.root_cause_category = "smoke-fail-cpu-budget"
            diag.root_cause_detail = (
                "Resolver records the kernel as oracle-consistent but S2-smoke "
                "(CPU interpreter pre-filter) fails or exceeds budget on the dev host. "
                "The kernel runs correctly on GPU/NPU."
            )
            diag.fix_recipe = [
                "Phase B golden capture on GPU/NPU is the authoritative path",
                "Optional: reduce smoke shape coverage to H2 to skip the worst row",
            ]
            diag.effort = "S"
        elif "phase B golden capture" in full_reason.lower() or "SKIPPED_BY_DESIGN" in full_reason:
            diag.root_cause_category = "phase-b-deferred"
            diag.root_cause_detail = (
                "Task is explicitly marked SKIPPED_BY_DESIGN in meta.phase_a_prime "
                "(A-owned deferral). Phase B golden capture is the authoritative path."
            )
            diag.fix_recipe = [
                "Run phase_b_golden_capture --task <id> on GPU host",
            ]
            diag.effort = "M"
        else:
            diag.root_cause_category = "other"
            diag.root_cause_detail = f"Unclassified substatus: {sub or '(none extracted)'}; full_reason head: {full_reason[:200]}"
            diag.fix_recipe = ["Inspect resolver verdict manually"]
            diag.effort = "M"


def _render_markdown(diags: list[TaskDiagnosis]) -> str:
    """Render full markdown report grouped by family + root-cause."""
    by_family: dict[str, list[TaskDiagnosis]] = {}
    for d in diags:
        by_family.setdefault(d.family, []).append(d)

    out: list[str] = []
    out.append("---")
    out.append("date: 2026-05-26")
    out.append("status: per-task root-cause diagnostic for 100-task corpus")
    out.append("based-on: results/resolve_all.json + per-task source files")
    out.append("scope: read-only diagnosis (no code/yaml/meta changes)")
    out.append("---\n")
    out.append("# Per-Task Root-Cause Diagnostic\n")

    # Top-level summary.
    cat_counter = Counter(d.root_cause_category for d in diags if d.root_cause_category)
    effort_counter = Counter(d.effort for d in diags if d.effort)
    out.append("## Summary\n")
    out.append(f"Total tasks analysed: {len(diags)}\n")
    out.append("### Root-cause categories\n")
    out.append("| category | count |")
    out.append("|---|---|")
    for k, n in cat_counter.most_common():
        out.append(f"| `{k}` | {n} |")
    out.append("")
    out.append("### Effort estimate distribution\n")
    out.append("| effort | count | meaning |")
    out.append("|---|---|---|")
    out.append(f"| S | {effort_counter.get('S', 0)} | ≤30 min per task |")
    out.append(f"| M | {effort_counter.get('M', 0)} | 30-120 min per task |")
    out.append(f"| L | {effort_counter.get('L', 0)} | ≥120 min per task |")
    out.append("")

    # Per-family per-task detail.
    out.append("## Per-Task Detail\n")
    for family in sorted(by_family.keys()):
        tasks = by_family[family]
        non_pass = [t for t in tasks if t.root_cause_category not in ("pass", "")]
        if not non_pass:
            continue
        out.append(f"### family: `{family}` (n={len(non_pass)})\n")
        for d in sorted(non_pass, key=lambda t: t.task_id):
            out.append(f"#### `{d.task_id}`\n")
            out.append(f"- **aggregate status**: `{d.aggregate_status}`")
            if d.intended_kernel_name:
                out.append(f"- **intended kernel**: `{d.intended_kernel_name}`")
            if d.intended_substatus:
                out.append(f"- **intended substatus**: `{d.intended_substatus}`")
            if d.upstream_exception:
                out.append(
                    f"- **upstream exception**: `{d.upstream_exception}` "
                    f"({d.upstream_exception_detail[:120]})"
                )
            out.append(f"- **root cause category**: `{d.root_cause_category}`")
            if d.root_cause_detail:
                out.append(f"- **root cause detail**: {d.root_cause_detail}")
            if d.kernel_required_params:
                out.append(
                    f"- **kernel required params** ({len(d.kernel_required_params)}): "
                    f"`{', '.join(d.kernel_required_params)}`"
                )
            if d.missing_in_shapes:
                out.append(
                    f"- **missing in shapes.yaml** ({len(d.missing_in_shapes)}): "
                    f"`{', '.join(d.missing_in_shapes)}`"
                )
            if d.shapes_inputs:
                declared = sorted(d.shapes_inputs.keys())
                out.append(
                    f"- **shapes.yaml declared inputs** ({len(declared)}): "
                    f"`{', '.join(declared)}`"
                )
            if d.fix_recipe:
                out.append("- **fix recipe**:")
                for step in d.fix_recipe:
                    out.append(f"  - {step}")
            out.append(f"- **effort**: `{d.effort}`")
            out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=str(CACHE), help="resolver cache path")
    ap.add_argument("--task", help="restrict to one task_id")
    ap.add_argument("--md", metavar="PATH", help="write markdown report to PATH")
    ap.add_argument("--json", metavar="PATH", help="write JSON dump to PATH")
    args = ap.parse_args()

    cache = json.loads(Path(args.cache).read_text())
    verdicts = cache["verdicts"]
    if args.task:
        verdicts = [v for v in verdicts if v["task_id"] == args.task]
        if not verdicts:
            print(f"task_id not found in cache: {args.task}", file=sys.stderr)
            return 1

    diags = [_diagnose_one(v) for v in verdicts]

    if args.md:
        Path(args.md).write_text(_render_markdown(diags))
        print(f"wrote markdown: {args.md} ({len(diags)} tasks)")

    if args.json:
        payload = [
            {
                "task_id": d.task_id,
                "family": d.family,
                "aggregate_status": d.aggregate_status,
                "intended_kernel_name": d.intended_kernel_name,
                "intended_substatus": d.intended_substatus,
                "intended_segment": d.intended_segment[:500],
                "kernel_required_params": d.kernel_required_params,
                "kernel_optional_params": d.kernel_optional_params,
                "shapes_inputs": list(d.shapes_inputs.keys()),
                "missing_in_shapes": d.missing_in_shapes,
                "oracle_input_keys": sorted(d.oracle_input_keys),
                "abstraction_mismatch_params": d.abstraction_mismatch_params,
                "case_collisions": d.case_collisions,
                "upstream_exception": d.upstream_exception,
                "upstream_exception_detail": d.upstream_exception_detail,
                "root_cause_category": d.root_cause_category,
                "root_cause_detail": d.root_cause_detail,
                "fix_recipe": d.fix_recipe,
                "effort": d.effort,
            }
            for d in diags
        ]
        Path(args.json).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"wrote json: {args.json}")

    # Stdout summary.
    cat_counter = Counter(d.root_cause_category for d in diags)
    effort_counter = Counter(d.effort for d in diags)
    print()
    print(f"=== diagnosed {len(diags)} tasks ===")
    print("--- root cause categories ---")
    for k, n in cat_counter.most_common():
        print(f"  {n:3d} {k}")
    print("--- effort ---")
    for k in ("S", "M", "L"):
        print(f"  {effort_counter.get(k, 0):3d} {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
