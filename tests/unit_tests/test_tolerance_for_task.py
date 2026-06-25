"""Unit tests for V9 plan M0.0 -- tolerance.for_task() authority."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ascendbench.correctness.tolerance import (
    ToleranceSpec,
    for_task,
    load_policy,
)

_POLICY = load_policy()


def _task(tolerance_class: str, **meta_extras):
    """Build a duck-typed task object for the spec computation."""
    meta = {"tolerance_class": tolerance_class, **meta_extras}
    return SimpleNamespace(
        task_id="L_synth",
        tolerance_class=tolerance_class,
        meta=meta,
    )


def test_for_task_default_no_golden_block():
    """No provenance.golden -> source is upstream_test_default, base
    atol/rtol from tolerance_policy.yaml."""
    t = _task("vector_reduce")
    spec = for_task(t, policy=_POLICY)
    assert isinstance(spec, ToleranceSpec)
    assert spec.tolerance_class_name == "vector_reduce"
    assert spec.source == "upstream_test_default"
    assert spec.pinned_from_upstream is False
    assert spec.atol == _POLICY["vector_reduce"].atol
    assert spec.rtol == _POLICY["vector_reduce"].rtol


def test_for_task_pinned_from_upstream_explicit():
    """provenance.golden.tolerance_source=upstream_test_explicit with
    atol/rtol values -> spec is pinned from upstream."""
    t = _task(
        "vector_reduce",
        provenance={
            "golden": {
                "tolerance_source": "upstream_test_explicit",
                "atol": 5e-4,
                "rtol": 5e-4,
            }
        },
    )
    spec = for_task(t, policy=_POLICY)
    assert spec.source == "upstream_test_explicit"
    assert spec.pinned_from_upstream is True
    assert spec.atol == pytest.approx(5e-4)
    assert spec.rtol == pytest.approx(5e-4)
    assert spec.cls.atol == pytest.approx(5e-4)
    assert spec.cls.rtol == pytest.approx(5e-4)


def test_for_task_explicit_without_values_falls_back_to_default():
    """explicit source but no atol/rtol -> falls back to default."""
    t = _task(
        "vector_reduce",
        provenance={
            "golden": {"tolerance_source": "upstream_test_explicit"}
        },
    )
    spec = for_task(t, policy=_POLICY)
    # falls through to default (no values to pin)
    assert spec.source == "upstream_test_default"
    assert spec.pinned_from_upstream is False


def test_for_task_migration_drift_budget_is_recognised_but_not_pinned():
    """migration_drift_budget source -> spec returned with that label;
    is_provenance_eligible() handles the release-ready exclusion."""
    t = _task(
        "elementwise",
        provenance={
            "golden": {"tolerance_source": "migration_drift_budget"}
        },
    )
    spec = for_task(t, policy=_POLICY)
    assert spec.source == "migration_drift_budget"
    assert spec.pinned_from_upstream is False
    # Uses tolerance class defaults (NOT pinned upstream values).
    assert spec.atol == _POLICY["elementwise"].atol


def test_for_task_propagates_bit_exact_flag():
    """bit_exact tolerance classes (sampling_rng) keep that flag in the
    pinned-from-upstream variant -- pinning atol/rtol must not silently
    flip bit_exact off."""
    # sampling_rng is bit_exact=True in tolerance_policy.yaml
    if "sampling_rng" not in _POLICY:
        pytest.skip("policy missing sampling_rng class")
    t = _task(
        "sampling_rng",
        provenance={
            "golden": {
                "tolerance_source": "upstream_test_explicit",
                "atol": 0.0,
                "rtol": 0.0,
            }
        },
    )
    spec = for_task(t, policy=_POLICY)
    assert spec.cls.bit_exact is True  # NOT lost by the pinned construction
    assert spec.pinned_from_upstream is True


def test_for_task_raises_on_missing_tolerance_class():
    """Unknown tolerance class -> KeyError (loader-strict, refuses to
    silently substitute a default)."""
    t = SimpleNamespace(
        task_id="L_bad", tolerance_class="not_a_real_class",
        meta={"tolerance_class": "not_a_real_class"}
    )
    with pytest.raises(KeyError):
        for_task(t, policy=_POLICY)


def test_for_task_accepts_raw_meta_dict():
    """The duck-typing contract: meta.tolerance_class is checked when
    the object lacks the .tolerance_class attribute."""
    t = SimpleNamespace(meta={"tolerance_class": "elementwise"})
    spec = for_task(t, policy=_POLICY)
    assert spec.tolerance_class_name == "elementwise"
