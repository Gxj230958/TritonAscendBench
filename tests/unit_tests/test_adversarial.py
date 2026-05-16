"""Unit tests for triton_ascend_bench.adversarial."""
from __future__ import annotations

from triton_ascend_bench.adversarial import (
    MetamorphicResult,
    check_canaries,
    run_metamorphic_checks,
)


def test_run_metamorphic_checks_no_transformations():
    out = run_metamorphic_checks(
        [{"name": "zero_v_yields_zero", "description": "v=0 → out=0"}],
        transformations={},
    )
    assert len(out) == 1
    assert not out[0].passed
    assert "no transformation registered" in (out[0].detail or "")


def test_run_metamorphic_checks_with_passing_transformation():
    def passing_check() -> MetamorphicResult:
        return MetamorphicResult("zero_v_yields_zero", True)

    out = run_metamorphic_checks(
        [{"name": "zero_v_yields_zero"}],
        transformations={"zero_v_yields_zero": passing_check},
    )
    assert out[0].passed


def test_run_metamorphic_checks_exception_caught():
    def boom() -> MetamorphicResult:
        raise RuntimeError("kernel exploded")

    out = run_metamorphic_checks(
        [{"name": "k"}], transformations={"k": boom}
    )
    assert not out[0].passed
    assert "raised: RuntimeError" in (out[0].detail or "")


def test_check_canaries_misclassification_is_hard():
    """C1-prefixed canaries are hard failures (likely cheating)."""
    findings = check_canaries(
        submitted_results={"C1_oracle_wrong": True},
        canary_truth={"C1_oracle_wrong": False},
    )
    triggered = [f for f in findings if f.triggered]
    assert len(triggered) == 1
    assert triggered[0].severity == "hard"


def test_check_canaries_match_truth_no_trigger():
    findings = check_canaries(
        submitted_results={"C1_ok": True},
        canary_truth={"C1_ok": True},
    )
    assert not any(f.triggered for f in findings)
