"""Unit tests for ascendbench.integrity.adversarial."""
from __future__ import annotations

import torch

from ascendbench.integrity.adversarial import (
    KNOWN_METAMORPHIC_CHECKS,
    CanaryFinding,
    MetamorphicContext,
    MetamorphicResult,
    aggregate_canary_findings,
    build_metamorphic_transformations,
    canary_finding_json,
    check_canaries,
    is_known_metamorphic_check,
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

    # Use a registered check name so the exception path is exercised
    # (an unregistered name is rejected before the transform runs).
    out = run_metamorphic_checks(
        [{"name": "row_independence"}],
        transformations={"row_independence": boom},
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


# ─────────────────────────────────────────────────────────────────────────────
# W2.4 — metamorphic registry
# ─────────────────────────────────────────────────────────────────────────────
def test_metamorphic_registry_covers_corpus_check_names():
    """The registry must contain every check name a task may declare —
    including the 5 names the live corpus actually uses and the design
    plan's generic baseline check."""
    corpus = {
        "empty_allowed_is_noop_stage1",
        "qk_scale_inverse",
        "scale_homogeneity_a",
        "weight_one_identity_eps_zero",
        "zero_x1_yields_zero",
    }
    assert corpus <= KNOWN_METAMORPHIC_CHECKS
    assert "differential_perturbation_general" in KNOWN_METAMORPHIC_CHECKS


def test_is_known_metamorphic_check():
    assert is_known_metamorphic_check("scale_homogeneity_a")
    assert not is_known_metamorphic_check("totally_made_up_check")


def test_run_metamorphic_checks_rejects_unknown_name():
    """An unknown check name is rejected at runtime as a backstop."""
    out = run_metamorphic_checks(
        [{"name": "totally_made_up_check"}],
        transformations={"totally_made_up_check": lambda: MetamorphicResult(
            "totally_made_up_check", True)},
    )
    assert len(out) == 1
    assert not out[0].passed
    assert "unknown metamorphic check" in (out[0].detail or "")


def test_validate_tasks_imports_the_registry():
    """W2.3 → W2.4 hand-off: validate_tasks must use the adversarial
    registry as the single source of truth, not a local literal copy."""
    import scripts.validate_tasks as vt

    assert vt.KNOWN_METAMORPHIC_CHECKS is KNOWN_METAMORPHIC_CHECKS


# ─────────────────────────────────────────────────────────────────────────────
# W2.4 — canary escalation chain (separate from metamorphic path)
# ─────────────────────────────────────────────────────────────────────────────
def test_aggregate_canary_findings_no_catch_not_suspicious():
    agg = aggregate_canary_findings([
        CanaryFinding("C1_ok", False, "soft", "canary C1_ok: ok"),
    ])
    assert agg.canary_caught == 0
    assert agg.suspicious is False


def test_aggregate_canary_findings_catch_marks_suspicious():
    """canary_caught > 0 ⇒ submission flagged suspicious (DESIGN §6.8)."""
    agg = aggregate_canary_findings([
        CanaryFinding("C1_ok", False, "soft", "ok"),
        CanaryFinding("C3_identity", True, "hard", "passed an identity trap"),
    ])
    assert agg.canary_caught == 1
    assert agg.suspicious is True


def test_aggregate_canary_findings_empty_not_suspicious():
    agg = aggregate_canary_findings([])
    assert agg.canary_caught == 0
    assert agg.suspicious is False


def test_canary_finding_json_schema_is_distinct_from_metamorphic():
    """Canary findings carry their own keys — canary_id / severity —
    which a MetamorphicResult does not have."""
    cj = canary_finding_json(
        CanaryFinding("C3_identity", True, "hard", "caught"))
    assert set(cj) == {"canary_id", "triggered", "severity", "message"}
    # A metamorphic result is keyed by name/passed/detail — disjoint.
    assert "name" not in cj
    assert "passed" not in cj


# ─────────────────────────────────────────────────────────────────────────────
# R7-3 — real metamorphic transformations
# ─────────────────────────────────────────────────────────────────────────────
def _ctx(base_inputs, kernel):
    """Build a MetamorphicContext whose run_candidate is an in-process
    `kernel(inputs) -> outputs` (no subprocess — pure transform test)."""
    def run(inp):
        return kernel(inp)

    def close(a, b):
        return (torch.is_tensor(a) and torch.is_tensor(b)
                and a.shape == b.shape
                and torch.allclose(a, b, atol=1e-5, rtol=1e-4))

    return MetamorphicContext(
        base_inputs=base_inputs, run_candidate=run, tensors_close=close)


def _run_one(check_name, ctx):
    reg = build_metamorphic_transformations([{"name": check_name}], ctx)
    out = run_metamorphic_checks([{"name": check_name}], reg)
    assert len(out) == 1
    return out[0]


def test_metamorphic_annihilation_pass_and_fail():
    """zero_x1_yields_zero: zeroing x1 must yield an all-zero output."""
    base = {"x1": torch.randn(4, 8), "x3": torch.randn(4, 8)}

    # honest kernel: out = silu(x1) * x3 — silu(0)=0 ⇒ zeroing x1 ⇒ 0.
    honest = lambda i: {"out": torch.nn.functional.silu(i["x1"]) * i["x3"]}
    assert _run_one("zero_x1_yields_zero", _ctx(base, honest)).passed

    # cheating kernel: returns a constant, ignores x1.
    cheat = lambda i: {"out": torch.ones(4, 8)}
    res = _run_one("zero_x1_yields_zero", _ctx(base, cheat))
    assert not res.passed
    assert "all-zero" in (res.detail or "")


def test_metamorphic_homogeneity_pass_and_fail():
    """linear_x3_scaling: scaling x3 by k scales the output by k."""
    base = {"x1": torch.randn(4, 8), "x3": torch.randn(4, 8)}

    honest = lambda i: {"out": torch.nn.functional.silu(i["x1"]) * i["x3"]}
    assert _run_one("linear_x3_scaling", _ctx(base, honest)).passed

    # non-linear in x3 (squares it) ⇒ scaling does not scale linearly.
    cheat = lambda i: {"out": i["x3"] * i["x3"]}
    assert not _run_one("linear_x3_scaling", _ctx(base, cheat)).passed


def test_metamorphic_compensating_pair_pass_and_fail():
    """qk_scale_inverse: (Q*α, K/α) leaves Q·K unchanged."""
    base = {"q": torch.randn(4, 8), "k": torch.randn(4, 8)}

    honest = lambda i: {"out": i["q"] * i["k"]}    # bilinear product
    assert _run_one("qk_scale_inverse", _ctx(base, honest)).passed

    # additive kernel: q+k is NOT invariant to the compensating pair.
    cheat = lambda i: {"out": i["q"] + i["k"]}
    assert not _run_one("qk_scale_inverse", _ctx(base, cheat)).passed


def test_metamorphic_row_independence_pass_and_fail():
    """row_independence: permuting input rows permutes output rows."""
    base = {"input": torch.randn(6, 8)}

    honest = lambda i: {"out": i["input"] * 2.0}      # per-row, no mixing
    assert _run_one("row_independence", _ctx(base, honest)).passed

    # cross-row contamination: output row r depends on input row r-1
    # (a circular shift) — permuting input rows does NOT permute output
    # rows identically, so the invariant is violated.
    def cheat(i):
        return {"out": torch.roll(i["input"], shifts=1, dims=0)}
    assert not _run_one("row_independence", _ctx(base, cheat)).passed


def test_metamorphic_differential_pass_and_fail():
    """differential_perturbation_general: perturbing input changes output."""
    base = {"x": torch.randn(4, 8)}

    honest = lambda i: {"out": i["x"] * 3.0}
    assert _run_one("differential_perturbation_general",
                    _ctx(base, honest)).passed

    # constant kernel: ignores its input ⇒ caught.
    cheat = lambda i: {"out": torch.zeros(4, 8)}
    res = _run_one("differential_perturbation_general", _ctx(base, cheat))
    assert not res.passed
    assert "ignores its input" in (res.detail or "")


def test_metamorphic_operator_specific_reports_explicit_reason():
    """An operator-specific check has no generic transform — it reports an
    explicit `not generically implementable`, never the misleading
    `no transformation registered`."""
    base = {"input": torch.randn(4, 8), "weight": torch.ones(8)}
    reg = build_metamorphic_transformations(
        [{"name": "weight_one_identity_eps_zero"}],
        _ctx(base, lambda i: {"out": i["input"]}))
    out = run_metamorphic_checks(
        [{"name": "weight_one_identity_eps_zero"}], reg)
    assert not out[0].passed
    assert "not generically implementable" in (out[0].detail or "")


def test_metamorphic_candidate_did_not_run_is_non_pass():
    """A candidate that fails to run yields a non-pass, never a crash."""
    base = {"x1": torch.randn(4, 8)}
    reg = build_metamorphic_transformations(
        [{"name": "zero_x1_yields_zero"}],
        _ctx(base, lambda i: None))      # run_candidate returns None
    out = run_metamorphic_checks([{"name": "zero_x1_yields_zero"}], reg)
    assert not out[0].passed
    assert "did not run" in (out[0].detail or "")


def test_build_registry_skips_unknown_names():
    """An unknown check name is left out of the registry — run_metamorphic
    _checks then emits its own `unknown metamorphic check` verdict."""
    base = {"x": torch.randn(2, 2)}
    reg = build_metamorphic_transformations(
        [{"name": "made_up_check"}], _ctx(base, lambda i: {"out": i["x"]}))
    assert "made_up_check" not in reg
