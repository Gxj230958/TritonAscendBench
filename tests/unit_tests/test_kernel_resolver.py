"""Unit tests for triton_ascend_bench.kernel_resolver.

Codex 2026-05-27 review caught that R10.a/R10.b resolvers were
returning ``@triton.jit``-decorated raw kernels when
``results/resolve_all.json::tried[0]`` was a JIT entry, then GPU
verify would fail with stale unrelated tracebacks. The fix is to
reject any JIT-decorated candidate at every return point.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from triton_ascend_bench.kernel_resolver import (
    enumerate_non_jit_module_defs,
    is_jit_decorated,
    kernel_function_exists,
)


def _write_kernel(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "kernel.py"
    p.write_text(body)
    return p


def test_jit_decorated_recognised(tmp_path):
    src = (
        "import triton\n"
        "@triton.jit\n"
        "def my_kernel(x_ptr, y_ptr): pass\n"
        "def my_wrapper(x): return None\n"
    )
    kp = _write_kernel(tmp_path, src)
    assert is_jit_decorated(kp, "my_kernel") is True
    assert is_jit_decorated(kp, "my_wrapper") is False


def test_heuristics_decorator_also_flagged(tmp_path):
    src = (
        "import triton\n"
        "@triton.heuristics({'X': lambda: 1})\n"
        "def k(): pass\n"
    )
    kp = _write_kernel(tmp_path, src)
    assert is_jit_decorated(kp, "k") is True


def test_autotune_decorator_also_flagged(tmp_path):
    src = (
        "import triton\n"
        "@triton.autotune(configs=[], key=['M'])\n"
        "def k(): pass\n"
    )
    kp = _write_kernel(tmp_path, src)
    assert is_jit_decorated(kp, "k") is True


def test_chained_decorators(tmp_path):
    """Real kernels often stack @triton.autotune + @triton.heuristics +
    @triton.jit -- all three must independently flag."""
    src = (
        "import triton\n"
        "@triton.autotune(configs=[], key=['M'])\n"
        "@triton.heuristics({'X': lambda: 1})\n"
        "@triton.jit\n"
        "def k(): pass\n"
    )
    kp = _write_kernel(tmp_path, src)
    assert is_jit_decorated(kp, "k") is True


def test_undecorated_function(tmp_path):
    src = "def f(): pass\n"
    kp = _write_kernel(tmp_path, src)
    assert is_jit_decorated(kp, "f") is False


def test_missing_function_returns_false(tmp_path):
    kp = _write_kernel(tmp_path, "def f(): pass\n")
    assert is_jit_decorated(kp, "g") is False


def test_missing_file_returns_false(tmp_path):
    kp = tmp_path / "does_not_exist.py"
    assert is_jit_decorated(kp, "anything") is False


def test_syntax_error_returns_false(tmp_path):
    kp = _write_kernel(tmp_path, "def not_valid(((\n")
    assert is_jit_decorated(kp, "anything") is False


def test_enumerate_skips_jit_defs(tmp_path):
    src = (
        "import triton\n"
        "@triton.jit\n"
        "def kernel_a(x_ptr): pass\n"
        "def wrapper_a(x): return None\n"
        "@triton.autotune(configs=[], key=['M'])\n"
        "@triton.jit\n"
        "def kernel_b(y_ptr): pass\n"
        "def wrapper_b(y, scale): return None\n"
    )
    kp = _write_kernel(tmp_path, src)
    defs = enumerate_non_jit_module_defs(kp)
    names = {n for n, _ in defs}
    assert names == {"wrapper_a", "wrapper_b"}


def test_enumerate_extracts_required_params(tmp_path):
    src = (
        "def f(a, b, c=1): return a + b + c\n"
        "def g(self, x): return x\n"
    )
    kp = _write_kernel(tmp_path, src)
    defs = dict(enumerate_non_jit_module_defs(kp))
    assert defs["f"] == ["a", "b"]  # c has default
    assert defs["g"] == ["x"]  # self stripped


# ─────────────────────────────────────────────────────────────────────
# Real corpus integration: L3_23 must resolve to solve_tril, not raw JIT
# ─────────────────────────────────────────────────────────────────────


def test_l3_23_kernel_has_jit_and_host_wrapper():
    """The L3_23 kernel.py carries both the raw JIT entry AND a host
    wrapper; we depend on the resolver choosing the wrapper."""
    kp = Path("tasks/level3/L3_23_merge_16x16_to_32x32_inverse/kernel.py")
    assert is_jit_decorated(kp, "merge_16x16_to_32x32_inverse_kernel") is True
    assert is_jit_decorated(kp, "solve_tril") is False
    names = {n for n, _ in enumerate_non_jit_module_defs(kp)}
    assert "solve_tril" in names
    assert "merge_16x16_to_32x32_inverse_kernel" not in names


def test_l3_23_resolver_picks_host_wrapper():
    """End-to-end: both R10.a and R10.b resolvers prefer solve_tril."""
    import sys

    sys.path.insert(0, "scripts")
    import yaml
    from oracle_equivalence_gpu_runner import _resolve_kernel_entrypoint as r1
    from oracle_pyref_precompute import _resolve_kernel_entrypoint as r2

    td = Path("tasks/level3/L3_23_merge_16x16_to_32x32_inverse")
    meta = yaml.safe_load((td / "meta.yaml").read_text())
    keys = ["A"]
    assert r1(meta, td / "kernel.py", td.name, oracle_input_keys=keys) == "solve_tril"
    assert r2(meta, td / "kernel.py", td.name, oracle_input_keys=keys) == "solve_tril"
