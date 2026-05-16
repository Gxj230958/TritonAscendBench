"""Unit tests for triton_ascend_bench.tolerance."""
from __future__ import annotations

import torch

from triton_ascend_bench.tolerance import compare, load_policy


def test_load_policy_has_core_classes():
    pol = load_policy()
    for name in [
        "elementwise",
        "vector_reduce",
        "int8_w8a8",
        "attention_flash",
        "elementwise_exact_or_inf_safe",
    ]:
        assert name in pol, f"missing class {name}"


def test_identical_tensors_pass():
    pol = load_policy()
    a = torch.arange(100, dtype=torch.float32)
    b = a.clone()
    res = compare(a, b, pol["elementwise"])
    assert res.passed


def test_within_tolerance_pass():
    pol = load_policy()
    a = torch.arange(100, dtype=torch.float32)
    b = a + 5.0e-4   # within atol 1e-3
    res = compare(a, b, pol["elementwise"])
    assert res.passed


def test_outside_tolerance_fail():
    pol = load_policy()
    a = torch.arange(100, dtype=torch.float32)
    b = a + 0.5      # way over any sensible atol
    res = compare(a, b, pol["elementwise"])
    assert not res.passed
    assert "atol" in (res.reason or "").lower() or "rtol" in (res.reason or "").lower()


def test_shape_mismatch_fail():
    pol = load_policy()
    a = torch.zeros(10)
    b = torch.zeros(11)
    res = compare(a, b, pol["elementwise"])
    assert not res.passed
    assert "shape" in (res.reason or "").lower()


def test_inf_positions_enforced():
    pol = load_policy()
    a = torch.tensor([1.0, float("inf"), 3.0])
    b = torch.tensor([1.0, 100.0, 3.0])   # b has no inf where a does
    res = compare(a, b, pol["elementwise_exact_or_inf_safe"])
    assert not res.passed
    assert "inf" in (res.reason or "").lower()


def test_matching_inf_positions_ok():
    pol = load_policy()
    a = torch.tensor([1.0, float("inf"), 3.0])
    b = torch.tensor([1.0, float("inf"), 3.0])
    res = compare(a, b, pol["elementwise_exact_or_inf_safe"])
    assert res.passed


def test_int64_dtype_strict_equality():
    """codex round-2 P1: integer dtypes must not go through float32 allclose.

    Reproduces the codex case: int64(16777216) vs int64(16777217) differ
    by 1, but float32 can only represent 16777216 (24-bit mantissa). With
    a float-cast allclose they would erroneously be 'equal'.
    """
    from triton_ascend_bench.tolerance import ToleranceClass, compare

    cls = ToleranceClass(
        name="sampling_rng", atol=0.0, rtol=0.0, bit_exact=True
    )
    a = torch.tensor([16777216], dtype=torch.int64)
    e = torch.tensor([16777217], dtype=torch.int64)
    res = compare(a, e, cls)
    assert not res.passed
    assert "bit_exact" in (res.reason or "").lower()


def test_int_dtype_pass_when_equal():
    from triton_ascend_bench.tolerance import ToleranceClass, compare
    cls = ToleranceClass(
        name="sampling_rng", atol=0.0, rtol=0.0, bit_exact=True
    )
    a = torch.tensor([42, 100, -7], dtype=torch.int64)
    e = a.clone()
    assert compare(a, e, cls).passed


def test_bool_dtype_strict_equality():
    from triton_ascend_bench.tolerance import ToleranceClass, compare
    cls = ToleranceClass(name="atomic_scatter", atol=0.0, rtol=0.0)
    a = torch.tensor([True, False, True])
    e = torch.tensor([True, True, True])
    res = compare(a, e, cls)
    assert not res.passed


def test_sampled_path_still_runs_full_allclose():
    """codex P1-3: sampled fast-fail must NOT be the final verdict.

    Construct a 2000-element tensor with a single NaN in the last
    position; force `sample_fraction` so small that statistically the
    sample likely misses the NaN. The full check must still fail.
    """
    from triton_ascend_bench.tolerance import ToleranceClass, compare

    cls = ToleranceClass(
        name="sampled", atol=1e-3, rtol=1e-3,
        sample_fraction=0.005,        # 10 samples out of 2000
        inf_must_match=True, nan_must_match=True,
    )
    a = torch.zeros(2000)
    e = torch.zeros(2000)
    a[-1] = float("nan")                                     # one outlier

    # Seed so the sample deterministically misses index 1999.
    torch.manual_seed(0)
    res = compare(a, e, cls)
    assert not res.passed, (
        "tolerance.compare reported pass on a sampled-fraction window "
        "even though the full tensor contains a NaN — codex P1-3 attack."
    )
    assert "nan" in (res.reason or "").lower()
