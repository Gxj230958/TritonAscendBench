"""W2.4 — CLI emit + canary escalation tests.

`tab run` round_result.json must serialise per-shape `metamorphic_results`
and a submission-level `canary_findings` array (distinct schemas);
`run-suite` / `analyze` must do submission-level canary aggregation so
`canary_caught > 0` flips the submission to `suspicious`.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ascendbench.engine import cli as _cli
from ascendbench.integrity.adversarial import MetamorphicResult
from ascendbench.engine.eval import ShapeRunResult, TaskEvalResult

REPO = Path(__file__).resolve().parent.parent.parent
_TASK = "L1_01_bias"
_SOLUTION = REPO / "solutions" / _TASK / "solution.py"


def _run_tab(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ascendbench.engine.cli", *args],
        cwd=REPO, capture_output=True, text=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# round_result emit — metamorphic_results + canary_findings
# ─────────────────────────────────────────────────────────────────────────────
def test_round_result_json_serialises_metamorphic_results():
    """Each shape carries its metamorphic_results array."""
    result = TaskEvalResult(task_id="T", static_checks_passed=True)
    sr = ShapeRunResult("P1", True, None, None, None)
    sr.metamorphic_results = [
        MetamorphicResult("scale_homogeneity_a", False, "no transformation"),
    ]
    result.shape_results.append(sr)

    doc = _cli._round_result_json(result)
    mr = doc["shape_results"][0]["metamorphic_results"]
    assert len(mr) == 1
    assert mr[0]["name"] == "scale_homogeneity_a"
    assert mr[0]["passed"] is False
    # metamorphic schema is name/passed/detail — no canary keys.
    assert "canary_id" not in mr[0]


def test_round_result_json_has_canary_findings_array():
    """A real-task round_result carries an (empty) canary_findings array
    — a distinct slot from per-shape metamorphic_results."""
    result = TaskEvalResult(task_id="T", static_checks_passed=True)
    doc = _cli._round_result_json(result)
    assert "canary_findings" in doc
    assert doc["canary_findings"] == []


def test_canary_findings_distinct_from_metamorphic_in_emit():
    """Canary findings serialise at the submission level with their own
    schema, never folded into a shape's metamorphic_results."""
    from ascendbench.integrity.adversarial import CanaryFinding

    result = TaskEvalResult(task_id="T", static_checks_passed=True)
    result.shape_results.append(ShapeRunResult("P1", True, None, None, None))
    findings = [CanaryFinding("C3_identity", True, "hard", "caught")]
    doc = _cli._round_result_json(result, canary_findings=findings)

    assert len(doc["canary_findings"]) == 1
    assert doc["canary_findings"][0]["canary_id"] == "C3_identity"
    # shape-level metamorphic_results stays its own (empty) list.
    assert doc["shape_results"][0]["metamorphic_results"] == []


# ─────────────────────────────────────────────────────────────────────────────
# canary escalation chain — _aggregate_suite_canaries
# ─────────────────────────────────────────────────────────────────────────────
def test_aggregate_suite_canaries_no_catch():
    per_task = [
        {"task_id": "T1", "canary_findings": []},
        {"task_id": "T2", "canary_findings": [
            {"canary_id": "C1_ok", "triggered": False,
             "severity": "soft", "message": "ok"}]},
    ]
    agg = _cli._aggregate_suite_canaries(per_task)
    assert agg["canary_caught"] == 0
    assert agg["suspicious"] is False


def test_aggregate_suite_canaries_catch_marks_suspicious():
    per_task = [
        {"task_id": "T1", "canary_findings": [
            {"canary_id": "C3_identity", "triggered": True,
             "severity": "hard", "message": "passed an identity trap"}]},
    ]
    agg = _cli._aggregate_suite_canaries(per_task)
    assert agg["canary_caught"] == 1
    assert agg["suspicious"] is True


# ─────────────────────────────────────────────────────────────────────────────
# analyze — canary_caught > 0 ⇒ suspicious
# ─────────────────────────────────────────────────────────────────────────────
def test_analyze_marks_suspicious_when_canary_caught(tmp_path):
    """A verified suite JSON with a triggered canary ⇒ score_report
    carries suspicious=True and canary_caught>0."""
    verified = tmp_path / "suite.json"
    verified.write_text(json.dumps({
        "mode": "verified",
        "results": [
            {"task_id": "ZZZ_not_in_catalogue",
             "correctness_status": "ok", "runtime_status": "ok",
             "performance_status": "ok",
             "canary_findings": [
                 {"canary_id": "C3_identity", "triggered": True,
                  "severity": "hard", "message": "caught"}]},
        ],
    }))
    out = tmp_path / "score.json"
    rc = _run_tab("analyze", "--verified-results", str(verified),
                  "--out", str(out))
    assert rc.returncode == 0, rc.stderr
    doc = json.loads(out.read_text())
    assert doc["canary_caught"] == 1
    assert doc["suspicious"] is True


def test_analyze_not_suspicious_without_canary(tmp_path):
    verified = tmp_path / "suite.json"
    verified.write_text(json.dumps({
        "mode": "verified",
        "results": [
            {"task_id": "ZZZ_not_in_catalogue",
             "correctness_status": "ok", "runtime_status": "ok",
             "performance_status": "ok", "canary_findings": []},
        ],
    }))
    out = tmp_path / "score.json"
    rc = _run_tab("analyze", "--verified-results", str(verified),
                  "--out", str(out))
    assert rc.returncode == 0, rc.stderr
    doc = json.loads(out.read_text())
    assert doc["canary_caught"] == 0
    assert doc["suspicious"] is False


def test_analyze_reads_suite_level_canary_findings(tmp_path):
    """A run-suite doc's top-level `canary_findings` is also aggregated."""
    verified = tmp_path / "suite.json"
    verified.write_text(json.dumps({
        "mode": "verified",
        "results": [],
        "canary_findings": [
            {"canary_id": "C7_torch_fallback", "triggered": True,
             "severity": "hard", "message": "fell back to torch"}],
    }))
    out = tmp_path / "score.json"
    rc = _run_tab("analyze", "--verified-results", str(verified),
                  "--out", str(out))
    assert rc.returncode == 0, rc.stderr
    doc = json.loads(out.read_text())
    assert doc["canary_caught"] == 1
    assert doc["suspicious"] is True


# ─────────────────────────────────────────────────────────────────────────────
# validate-result accepts the W2.4 fields
# ─────────────────────────────────────────────────────────────────────────────
def test_validate_result_accepts_w2_4_fields(tmp_path):
    """`tab validate-result` accepts metamorphic_results + canary_findings."""
    rr = tmp_path / "round_result.json"
    rr.write_text(json.dumps({
        "task_id": "T1",
        "static_checks_passed": True,
        "aggregate_score": 0.5,
        "correctness_status": "ok",
        "runtime_status": "ok",
        "performance_status": "ok",
        "performance_flags": [],
        "fallback_detected": False,
        "shape_results": [
            {"shape_id": "P1", "correctness_passed": True,
             "metamorphic_results": [
                 {"name": "scale_homogeneity_a", "passed": False,
                  "detail": "no transformation"}]},
        ],
        "canary_findings": [],
        "canary_caught": 0,
        "suspicious": False,
    }))
    rc = _run_tab("validate-result", str(rr))
    assert rc.returncode == 0, rc.stderr
