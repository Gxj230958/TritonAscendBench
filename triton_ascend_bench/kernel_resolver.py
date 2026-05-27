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


__all__ = [
    "is_jit_decorated",
    "enumerate_non_jit_module_defs",
    "kernel_function_exists",
]
