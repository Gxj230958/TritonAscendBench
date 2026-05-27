"""kernel_resolver.py -- shared helpers for picking the right host wrapper.

V9 plan M2.1a/b run two near-identical resolvers in
``scripts/oracle_pyref_precompute.py`` (dev / CPU audit) and
``scripts/oracle_equivalence_gpu_runner.py`` (GPU verify). They were
silently divergent: when ``results/resolve_all.json`` cached a
``@triton.jit`` raw kernel as ``tried[0]`` (the common case for tasks
where the original phase-A resolver could not invoke the host wrapper
on the dev box) BOTH resolvers returned that JIT-decorated name
without ever asking "is this actually a host wrapper?". The GPU
verify then tried to call a Triton kernel with bare tensor args,
which falls over inside the kernel with a stale unrelated traceback
(``ValueError: not enough values to unpack`` or a kernel-launch
``AttributeError``) and gets misclassified as ``LAUNCH_BLOCKED``.

The helpers here are pure-stdlib + AST-only -- no torch import, no
file execution -- so both scripts can call into them without dragging
in GPU deps. The only invariant they encode is the V9 ground rule:

    a function decorated with ``@triton.jit`` / ``@triton.heuristics``
    / ``@triton.autotune`` is a kernel, NOT a host wrapper. The R10
    pipeline always wants the host wrapper.
"""
from __future__ import annotations

import ast
from pathlib import Path


_JIT_DECORATORS: tuple[str, ...] = (
    "triton.jit",
    "triton.heuristics",
    "triton.autotune",
)


def is_jit_decorated(kernel_path: Path, fn_name: str) -> bool:
    """True iff ``fn_name`` in ``kernel_path`` carries a ``@triton.jit``
    (or ``@triton.heuristics`` / ``@triton.autotune``) decorator.

    Returns ``False`` when the function is absent / the file does not
    parse -- the caller's own existence check handles that path.
    """
    if not kernel_path.exists():
        return False
    try:
        tree = ast.parse(kernel_path.read_text())
    except (OSError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == fn_name
        ):
            for dec in node.decorator_list:
                try:
                    src = ast.unparse(dec)
                except Exception:
                    continue
                if any(d in src for d in _JIT_DECORATORS):
                    return True
            return False
    return False


def enumerate_non_jit_module_defs(
    kernel_path: Path,
) -> list[tuple[str, list[str]]]:
    """List ``(name, required_params)`` for every module-level ``def``
    in ``kernel_path`` that is NOT ``@triton.jit``-decorated.

    JIT kernels carry pointer / stride args and never align with the
    oracle's tensor-level inputs; filtering them here removes the
    most common false-positive in resolver Jaccard scoring.
    """
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
            try:
                src = ast.unparse(dec)
            except Exception:
                continue
            if any(d in src for d in _JIT_DECORATORS):
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


def kernel_function_exists(kernel_path: Path, fn_name: str) -> bool:
    """Quick AST check: does ``kernel_path`` have a module-level def for
    ``fn_name`` (whether JIT-decorated or not)?"""
    if not kernel_path.exists():
        return False
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


def resolve_kernel_entrypoint(
    meta: dict,
    kernel_path: Path,
    task_id: str,
    oracle_input_keys: list[str] | None,
    *,
    resolve_cache_path: Path | None = None,
    min_tried_jaccard: float = 0.2,
) -> str | None:
    """V9 P5.2: single shared resolver for R10.a / R10.b / auto-patch.

    Preference (each step verifies the candidate is a real host wrapper
    via :func:`is_jit_decorated` / :func:`kernel_function_exists`):

    1. ``meta.oracle.gpu_triton`` parsed as ``kernel.py::<fn>``, unless
       ``<fn>`` is the ``placeholder`` sentinel.
    2. ``results/resolve_all.json::tried[0]`` cache, **if** the
       candidate's required-param set has at least ``min_tried_jaccard``
       overlap with the oracle's input keys. The cache historically
       picked the wrong host wrapper for tasks where the original phase-A
       resolver mismatched (e.g. L3_07 cached ``pack_seq_triton`` even
       though the oracle works on ``correct_attn_out``); without the
       jaccard floor we'd silently auto-patch shapes against the wrong
       wrapper.
    3. AST-based wrapper auto-detection: pick the non-jit def with the
       highest jaccard overlap (>= 0.3).
    4. Pattern variants from ``meta.name``.
    5. ``meta.name`` itself if it resolves to a non-jit def.

    Returns ``None`` rather than a jit-decorated raw kernel -- emitting
    a clean LAUNCH_BLOCKED with that reason beats a misleading runtime
    trace from invoking a Triton kernel with bare tensor args.
    """
    import json as _json  # noqa: PLC0415

    def _accept(cand: str) -> bool:
        if not cand:
            return False
        if is_jit_decorated(kernel_path, cand):
            return False
        return kernel_function_exists(kernel_path, cand)

    def _jaccard_with_oracle(cand: str) -> float:
        """Cheap sanity gate: cand's required params vs oracle keys.

        Returns ``0`` when we cannot extract cand's signature -- the
        caller treats that the same as a low jaccard and falls through.
        """
        if not oracle_input_keys:
            return 1.0  # no oracle to compare; trust the candidate
        for name, req in enumerate_non_jit_module_defs(kernel_path):
            if name != cand:
                continue
            if not req:
                return 0.0
            req_set = set(req)
            oracle_set = set(oracle_input_keys)
            inter = len(req_set & oracle_set)
            union = len(req_set | oracle_set)
            return inter / union if union else 0.0
        return 0.0

    gpu_triton = (meta.get("oracle") or {}).get("gpu_triton", "") or ""
    if isinstance(gpu_triton, str) and "::" in gpu_triton:
        _, _, fn = gpu_triton.partition("::")
        if fn and fn != "placeholder" and _accept(fn):
            return fn

    if resolve_cache_path and resolve_cache_path.exists():
        try:
            d = _json.loads(resolve_cache_path.read_text())
            for v in d.get("verdicts", []):
                if v.get("task_id") == task_id:
                    tried = v.get("tried") or []
                    if (
                        tried
                        and _accept(tried[0])
                        and _jaccard_with_oracle(tried[0]) >= min_tried_jaccard
                    ):
                        return tried[0]
                    break
        except Exception:
            pass

    if oracle_input_keys:
        oracle_set = set(oracle_input_keys)
        non_jit = enumerate_non_jit_module_defs(kernel_path)
        scored: list[tuple[float, str]] = []
        for name, req in non_jit:
            if not req:
                continue
            req_set = set(req)
            inter = len(oracle_set & req_set)
            union = len(oracle_set | req_set)
            if union == 0:
                continue
            jaccard = inter / union
            subset_bonus = 0.2 if req_set.issubset(oracle_set) else 0.0
            scored.append((jaccard + subset_bonus, name))
        scored.sort(reverse=True)
        if scored and scored[0][0] >= 0.3:
            return scored[0][1]

    meta_name = meta.get("name", "") or ""
    if not meta_name:
        return None
    base = meta_name
    if base.endswith("_kernel"):
        base = base[: -len("_kernel")]
    base = base.lstrip("_")
    pattern_candidates = [
        base,
        f"triton_{base}",
        f"{base}_triton",
        f"{base}_fast",
        f"{base}_paged",
        meta_name,
    ]
    seen: set[str] = set()
    for cand in pattern_candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        if _accept(cand):
            return cand
    if _accept(meta_name):
        return meta_name
    return None


__all__ = [
    "is_jit_decorated",
    "enumerate_non_jit_module_defs",
    "kernel_function_exists",
    "resolve_kernel_entrypoint",
]
