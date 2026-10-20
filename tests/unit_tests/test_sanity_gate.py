"""Unit tests for V9 plan M3.0 -- R16 sanity bounds gate.

Covers:

* central sanity_bounds.yaml loads with every corpus category present
* apply_bounds rejects NaN, Inf, magnitude, all-zero, rank mismatch
* unknown category fails closed (forces reviewer to register it)
* clean tensors within bounds pass
"""
from __future__ import annotations

import pytest
import torch
import yaml

from ascendbench.release_gate.sanity_gate import (
    apply_bounds,
    load_bounds,
)

_BOUNDS = load_bounds()


# ─────────────────────────────────────────────────────────────────────
# load_bounds
# ─────────────────────────────────────────────────────────────────────


def test_load_bounds_returns_known_categories():
    """The central YAML covers every category seen in the corpus."""
    # Categories observed in the 100-task census:
    EXPECTED = {  # noqa: N806 - fixed schema key set
        "norm", "normalization", "elementwise", "attention", "softmax",
        "paged-attention", "gemm", "moe", "ssm", "scan", "quantization",
        "sampling", "rng", "convolution", "spec_decode", "other",
    }
    missing = EXPECTED - _BOUNDS.keys()
    assert not missing, f"sanity_bounds.yaml missing categories: {missing}"


def test_load_bounds_softmax_is_tightly_bounded():
    """Softmax output magnitude is bounded by [0, 1]."""
    assert _BOUNDS["softmax"].max_abs <= 1.0


def test_load_bounds_quantization_int8_ceiling():
    """Quantized int8 tensors cannot exceed |256|."""
    assert _BOUNDS["quantization"].max_abs <= 256.0


# ─────────────────────────────────────────────────────────────────────
# apply_bounds -- happy path
# ─────────────────────────────────────────────────────────────────────


def test_clean_tensor_passes():
    t = torch.tensor([0.1, -0.2, 0.5, 0.3])
    res = apply_bounds(t, category="elementwise", bounds=_BOUNDS)
    assert res.passed is True
    assert res.max_abs_value == pytest.approx(0.5)
    assert res.nan_rate == 0.0
    assert res.inf_rate == 0.0


def test_clean_softmax_output_passes():
    t = torch.tensor([0.1, 0.2, 0.3, 0.4])
    res = apply_bounds(t, category="softmax", bounds=_BOUNDS)
    assert res.passed is True


# ─────────────────────────────────────────────────────────────────────
# apply_bounds -- failure paths
# ─────────────────────────────────────────────────────────────────────


def test_nan_present_rejects():
    t = torch.tensor([0.1, float("nan"), 0.3])
    res = apply_bounds(t, category="elementwise", bounds=_BOUNDS)
    assert res.passed is False
    assert "NaN" in res.reason
    assert res.nan_rate > 0


def test_inf_present_rejects():
    t = torch.tensor([0.1, float("inf"), 0.3])
    res = apply_bounds(t, category="elementwise", bounds=_BOUNDS)
    assert res.passed is False
    assert "Inf" in res.reason


def test_all_zero_output_rejected_when_forbidden():
    """elementwise forbids all-zero output."""
    t = torch.zeros(8)
    res = apply_bounds(t, category="elementwise", bounds=_BOUNDS)
    assert res.passed is False
    assert "all-zero" in res.reason


def test_all_zero_softmax_output_allowed():
    """softmax permits all-zero (padding mask path)."""
    t = torch.zeros(8)
    res = apply_bounds(t, category="softmax", bounds=_BOUNDS)
    assert res.passed is True


def test_max_abs_violation_rejected():
    """Softmax bound is 1.0; output of 2.0 must reject."""
    t = torch.tensor([2.0, 0.5])
    res = apply_bounds(t, category="softmax", bounds=_BOUNDS)
    assert res.passed is False
    assert "max_abs" in res.reason


def test_quantization_int8_ceiling_enforced():
    t = torch.tensor([300.0, 50.0])
    res = apply_bounds(t, category="quantization", bounds=_BOUNDS)
    assert res.passed is False
    assert "max_abs" in res.reason
    # Bound prints in scientific notation -- check for the bound's magnitude.
    assert "2.560e+02" in res.reason or "256" in res.reason


def test_rank_mismatch_rejected_when_required():
    """elementwise requires output rank == input rank."""
    t = torch.zeros(2, 4)
    t[0, 0] = 0.5  # non-zero so forbid_all_zero doesn't trip first
    res = apply_bounds(
        t, category="elementwise", bounds=_BOUNDS, input_rank=3
    )
    assert res.passed is False
    assert "rank" in res.reason


def test_rank_match_allows_pass():
    t = torch.tensor([0.5, -0.3, 0.7])
    res = apply_bounds(
        t, category="elementwise", bounds=_BOUNDS, input_rank=1
    )
    assert res.passed is True


def test_unknown_category_fails_closed():
    """A category not in sanity_bounds.yaml is a release-blocking
    error (forces the reviewer to register it)."""
    t = torch.tensor([0.5])
    res = apply_bounds(
        t, category="weird_made_up_category", bounds=_BOUNDS
    )
    assert res.passed is False
    assert "no sanity bounds entry" in res.reason


def test_non_tensor_input_rejected():
    res = apply_bounds(
        [0.1, 0.2, 0.3], category="elementwise", bounds=_BOUNDS
    )
    assert res.passed is False
    assert "not a torch.Tensor" in res.reason


def test_empty_tensor_passes():
    """An empty tensor is a degenerate but not corrupting case."""
    t = torch.tensor([])
    res = apply_bounds(t, category="elementwise", bounds=_BOUNDS)
    assert res.passed is True


# ─────────────────────────────────────────────────────────────────────
# Corpus integration
# ─────────────────────────────────────────────────────────────────────


def test_every_corpus_category_has_a_bounds_entry():
    """Every category present in tasks/level{1,2,3}/*/meta.yaml MUST
    have a corresponding entry in sanity_bounds.yaml (fail-closed
    contract -- M3.0 ships with full coverage)."""
    import pathlib

    seen: set[str] = set()
    for p in pathlib.Path("tasks").rglob("meta.yaml"):
        meta = yaml.safe_load(p.read_text()) or {}
        c = meta.get("category")
        if c:
            seen.add(c)
    missing = seen - _BOUNDS.keys()
    assert not missing, (
        f"sanity_bounds.yaml is missing entries for corpus categories: "
        f"{missing}"
    )
