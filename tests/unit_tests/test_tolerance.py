"""Unit tests for ascendbench.correctness.tolerance."""
from __future__ import annotations

import torch

from ascendbench.correctness.tolerance import compare, load_policy


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
    """integer dtypes must not go through float32 allclose.

    Reproduces the case: int64(16777216) vs int64(16777217) differ
    by 1, but float32 can only represent 16777216 (24-bit mantissa). With
    a float-cast allclose they would erroneously be 'equal'.
    """
    from ascendbench.correctness.tolerance import ToleranceClass, compare

    cls = ToleranceClass(
        name="sampling_rng", atol=0.0, rtol=0.0, bit_exact=True
    )
    a = torch.tensor([16777216], dtype=torch.int64)
    e = torch.tensor([16777217], dtype=torch.int64)
    res = compare(a, e, cls)
    assert not res.passed
    assert "bit_exact" in (res.reason or "").lower()


def test_int_dtype_pass_when_equal():
    from ascendbench.correctness.tolerance import ToleranceClass, compare
    cls = ToleranceClass(
        name="sampling_rng", atol=0.0, rtol=0.0, bit_exact=True
    )
    a = torch.tensor([42, 100, -7], dtype=torch.int64)
    e = a.clone()
    assert compare(a, e, cls).passed


def test_bool_dtype_strict_equality():
    from ascendbench.correctness.tolerance import ToleranceClass, compare
    cls = ToleranceClass(name="atomic_scatter", atol=0.0, rtol=0.0)
    a = torch.tensor([True, False, True])
    e = torch.tensor([True, True, True])
    res = compare(a, e, cls)
    assert not res.passed


def test_sampled_path_still_runs_full_allclose():
    """P1-3: sampled fast-fail must NOT be the final verdict.

    Construct a 2000-element tensor with a single NaN in the last
    position; force `sample_fraction` so small that statistically the
    sample likely misses the NaN. The full check must still fail.
    """
    from ascendbench.correctness.tolerance import ToleranceClass, compare

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
        "even though the full tensor contains a NaN — P1-3 attack."
    )
    assert "nan" in (res.reason or "").lower()


# ---------------------------------------------------------------------------
# G6 — migration-mode output-dtype-equality gate (require_output_dtype_match).
# ---------------------------------------------------------------------------
from ascendbench.correctness.tolerance import ToleranceClass  # noqa: E402

_FLOAT_CLS = ToleranceClass(name="g6", atol=1e-3, rtol=1e-3)


def test_dtype_gate_off_by_default_allows_float_widening():
    """Default (off): a fp32 candidate vs a bf16 reference still passes via the
    historical fp32-cast allclose — GPU-C2 callers are byte-unchanged."""
    a = torch.zeros(8, dtype=torch.float32)
    e = torch.zeros(8, dtype=torch.bfloat16)
    assert compare(a, e, _FLOAT_CLS).passed
    assert compare(a, e, _FLOAT_CLS, require_output_dtype_match=False).passed


def test_dtype_gate_on_fails_float_dtype_mismatch():
    """On (NPU migration): output dtype must equal the reference dtype."""
    a = torch.zeros(8, dtype=torch.float32)
    e = torch.zeros(8, dtype=torch.bfloat16)
    res = compare(a, e, _FLOAT_CLS, require_output_dtype_match=True)
    assert not res.passed
    assert "dtype mismatch (migration)" in (res.reason or "")


def test_dtype_gate_on_passes_same_dtype():
    """On + identical dtype + identical values still passes (gate is dtype-only,
    it does not bypass the numeric allclose)."""
    a = torch.zeros(8, dtype=torch.bfloat16)
    e = torch.zeros(8, dtype=torch.bfloat16)
    assert compare(a, e, _FLOAT_CLS, require_output_dtype_match=True).passed


def test_dtype_gate_on_does_not_bypass_numeric_check():
    """On + same dtype but a large value drift must STILL fail — the gate adds a
    dtype precondition, it never short-circuits the value comparison to a pass."""
    a = torch.full((8,), 5.0, dtype=torch.float32)
    e = torch.zeros(8, dtype=torch.float32)
    assert not compare(a, e, _FLOAT_CLS, require_output_dtype_match=True).passed


def test_dtype_gate_does_not_touch_integer_path():
    """Integer outputs already enforce dtype in their own branch; the gate flag
    is irrelevant there (int32 vs int32 passes regardless of the flag)."""
    a = torch.zeros(8, dtype=torch.int32)
    e = torch.zeros(8, dtype=torch.int32)
    int_cls = ToleranceClass(name="g6i", atol=0, rtol=0, int_atol=0)
    assert compare(a, e, int_cls, require_output_dtype_match=True).passed
    assert compare(a, e, int_cls, require_output_dtype_match=False).passed
