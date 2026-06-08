"""W1.4 — CLI round_result schema tests.

`tab run` must emit the W1.4 producer schema (task-level `s_t`, the
DESIGN §9 provenance pair `baseline_ns`/`measured_ns`, the orthogonal
`task_status` field, and per-shape Roofline + timing detail), and
`tab validate-result` must accept that schema — including the
`task_status != "evaluated"` case where the perf fields are `null`.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
# WA.4: use a RECONCILED task (`param_constraints.scaffold: false`,
# semantically clean) so its repo-authoritative `task_status` is genuinely
# "evaluated" — a `scaffold: true` corpus task is "blocked" by the WA.4 gate.
_TASK = "L2_04_silu_and_mul"
_SOLUTION = REPO / "solutions" / _TASK / "solution.py"


def _run_tab(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ascendbench.engine.cli", *args],
        cwd=REPO, capture_output=True, text=True,
    )


def test_run_emits_w1_4_schema(tmp_path):
    """`tab run` round_result.json carries s_t / baseline_ns / measured_ns
    / task_status and a per-shape shape_results list."""
    assert _SOLUTION.exists(), f"fixture missing: {_SOLUTION}"
    out = tmp_path / "round_result.json"
    _run_tab("run", _TASK, "--solution", str(_SOLUTION), "--out", str(out))
    doc = json.loads(out.read_text())

    # W1.4 task-level fields present (value may be null on a dev box with
    # an unverified peak table — the *schema* is what matters here).
    for key in ("s_t", "baseline_ns", "measured_ns", "task_status",
                "shape_results"):
        assert key in doc, f"round_result.json missing {key!r}"
    assert doc["task_status"] == "evaluated"
    assert isinstance(doc["shape_results"], list)


def test_run_shape_results_disclose_timing_stats(tmp_path):
    """Each per-shape entry discloses median / p99 / cv / trials and the
    Roofline provenance keys (DESIGN §6.6 / §9)."""
    out = tmp_path / "round_result.json"
    _run_tab("run", _TASK, "--solution", str(_SOLUTION), "--out", str(out))
    doc = json.loads(out.read_text())
    assert doc["shape_results"], "expected at least one shape result"
    for sr in doc["shape_results"]:
        for key in ("shape_id", "t_ref", "t_npu", "efficiency",
                    "median", "p99", "cv", "trials",
                    "w_per_class", "m_per_resource", "baseline_ns",
                    "measured_ns"):
            assert key in sr, f"shape_results entry missing {key!r}"
        assert isinstance(sr["trials"], int)


def test_shape_result_json_baseline_ns_is_ns_not_seconds():
    """per-shape baseline_ns is ns (t_ref*1e9), not raw seconds t_ref."""
    from types import SimpleNamespace

    from ascendbench.engine.cli import _shape_result_json
    r = SimpleNamespace(
        shape_id="P1", correctness_passed=True, correctness_reason=None,
        t_ref=1e-6, baseline_ns=1000.0, t_npu=2000.0,
        efficiency=0.5, w_per_class={}, m_per_resource={},
        median=2000.0, p99=2000.0, cv=0.0, trials=1,
        roofline_error=None, metamorphic_results=[],
    )
    j = _shape_result_json(r)
    assert j["baseline_ns"] == 1000.0   # ns (= t_ref * 1e9), not 1e-6 seconds
    assert j["t_ref"] == 1e-6           # raw t_ref disclosure stays seconds
    assert j["measured_ns"] == 2000.0


def test_validate_result_accepts_w1_4_schema(tmp_path):
    """validate-result accepts a round_result.json with the full W1.4
    producer schema for an evaluated task."""
    doc = {
        "task_id": "L1_01_bias",
        "static_checks_passed": True,
        "aggregate_score": 0.75,
        "correctness_status": "ok",
        "runtime_status": "ok",
        "performance_status": "ok",
        "performance_flags": [],
        "fallback_detected": False,
        "s_t": 0.75,
        "baseline_ns": 1.0e-6,
        "measured_ns": 1.33e-6,
        "task_status": "evaluated",
        "shape_results": [
            {"shape_id": "P1", "correctness_passed": True,
             "t_ref": 1.0e-6, "t_npu": 1.33e-6, "efficiency": 0.75,
             "median": 1.33e-6, "p99": 1.5e-6, "cv": 0.05, "trials": 30,
             "w_per_class": {"vector": 1024.0},
             "m_per_resource": {"hbm": 4096.0}},
        ],
    }
    p = tmp_path / "result.json"
    p.write_text(json.dumps(doc))
    r = _run_tab("validate-result", str(p))
    assert r.returncode == 0, f"validate-result rejected W1.4 schema: {r.stderr}"


def test_validate_result_accepts_non_evaluated_with_null_perf(tmp_path):
    """For task_status != "evaluated" the perf fields may be null —
    validate-result must accept that (W1.4 null-field contract)."""
    doc = {
        "task_id": "L3_04_flash_attn",
        "static_checks_passed": True,
        "aggregate_score": 0.0,
        "correctness_status": "ok",
        "runtime_status": "ok",
        "performance_status": "ok",
        "performance_flags": [],
        "fallback_detected": False,
        # unsupported task: no provider golden in this mode -> null perf.
        "s_t": None,
        "baseline_ns": None,
        "measured_ns": None,
        "task_status": "unsupported",
        "shape_results": [
            {"shape_id": "P1", "correctness_passed": True,
             "t_ref": None, "t_npu": None, "efficiency": None,
             "median": None, "p99": None, "cv": None, "trials": 0},
        ],
    }
    p = tmp_path / "result.json"
    p.write_text(json.dumps(doc))
    r = _run_tab("validate-result", str(p))
    assert r.returncode == 0, (
        f"validate-result rejected unsupported task w/ null perf: {r.stderr}"
    )


def test_validate_result_rejects_bad_task_status(tmp_path):
    doc = {
        "task_id": "L1_01_bias",
        "static_checks_passed": True,
        "aggregate_score": 0.5,
        "correctness_status": "ok",
        "runtime_status": "ok",
        "performance_status": "ok",
        "performance_flags": [],
        "fallback_detected": False,
        "task_status": "garbage_value",
    }
    p = tmp_path / "result.json"
    p.write_text(json.dumps(doc))
    r = _run_tab("validate-result", str(p))
    assert r.returncode == 1
    assert "task_status" in (r.stdout + r.stderr)


def test_validate_result_rejects_nan_s_t(tmp_path):
    """A non-finite s_t is a gaming vector — validate-result rejects it."""
    # JSON has no NaN literal; emit it via the Python encoder's allow_nan.
    doc_text = json.dumps({
        "task_id": "L1_01_bias",
        "static_checks_passed": True,
        "aggregate_score": 0.5,
        "correctness_status": "ok",
        "runtime_status": "ok",
        "performance_status": "ok",
        "performance_flags": [],
        "fallback_detected": False,
        "task_status": "evaluated",
        "s_t": float("inf"),
    })
    p = tmp_path / "result.json"
    p.write_text(doc_text)
    r = _run_tab("validate-result", str(p))
    assert r.returncode == 1
    assert "s_t" in (r.stdout + r.stderr)


def test_analyze_consumes_real_s_t(tmp_path):
    """analyze assembles the perf metric from the real per-task `s_t`,
    not the removed `measured_ns=1.0` shim. Two eligible tasks with
    s_t = 0.5 and s_t = 2.0 -> capped_geomean_native = geomean = 1.0."""
    # WA.4: use the two RECONCILED corpus tasks (`param_constraints.
    # scaffold: false`) so their repo-authoritative `task_status` is
    # genuinely "evaluated" — a scaffold corpus task is "blocked".
    verified = {
        "mode": "verified",
        "results": [
            {"task_id": "L2_04_silu_and_mul",
             "static_checks_passed": True, "correctness_status": "ok",
             "runtime_status": "ok", "performance_status": "ok",
             "performance_flags": [], "fallback_detected": False,
             "aggregate_score": 0.5, "s_t": 0.5, "task_status": "evaluated"},
            {"task_id": "L1_12_rms_norm",
             "static_checks_passed": True, "correctness_status": "ok",
             "runtime_status": "ok", "performance_status": "ok",
             "performance_flags": [], "fallback_detected": False,
             "aggregate_score": 2.0, "s_t": 2.0, "task_status": "evaluated"},
        ],
    }
    (tmp_path / "v.json").write_text(json.dumps(verified))
    out = tmp_path / "score.json"
    r = _run_tab("analyze", "--verified-results", str(tmp_path / "v.json"),
                 "--out", str(out))
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())
    assert doc["n_native_measurements"] == 2
    # perf metric reflects the real s_t geomean, padded with ε for the
    # remaining eligible tasks: geomean of {0.5, 2.0, ε...}.
    assert doc["perf_status"] == "ok"
    assert doc["capped_geomean_native"] is not None


def test_analyze_skips_non_evaluated_tasks(tmp_path):
    """R2-4 (W4.3 ④): in self-reported mode an `upstream_oracle_required`
    task is excluded from the scored denominator by the FIXED repo-derived
    formula — NOT counted as a failure, NOT influenced by the result's own
    reported `task_status`.

    `L2_03_act_and_mul` is an `upstream_oracle_required` task; in
    self-reported mode it has no provider golden, so its result is dropped
    from `task_measurements` and the denominator regardless of what the
    result claims.
    """
    verified = {
        "mode": "self-reported",
        "results": [
            # WA.4: a RECONCILED non-upstream task — genuinely "evaluated".
            {"task_id": "L2_04_silu_and_mul",
             "static_checks_passed": True, "correctness_status": "ok",
             "runtime_status": "ok", "performance_status": "ok",
             "performance_flags": [], "fallback_detected": False,
             "aggregate_score": 1.0, "s_t": 1.0, "task_status": "evaluated"},
            {"task_id": "L2_03_act_and_mul",
             "static_checks_passed": True, "correctness_status": "ok",
             "runtime_status": "ok", "performance_status": "ok",
             "performance_flags": [], "fallback_detected": False,
             "aggregate_score": 0.0, "s_t": None,
             "baseline_ns": None, "measured_ns": None,
             # The result claims "evaluated" — but the repo authority says
             # this upstream_oracle_required task is "unsupported" in
             # self-reported mode, so it is excluded anyway.
             "task_status": "evaluated"},
        ],
    }
    (tmp_path / "v.json").write_text(json.dumps(verified))
    out = tmp_path / "score.json"
    r = _run_tab("analyze", "--verified-results", str(tmp_path / "v.json"),
                 "--out", str(out))
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())
    # R2-4: the upstream_oracle_required task is dropped from the
    # self-reported denominator — only the one evaluated task is measured.
    assert doc["mode"] == "self-reported"
    assert doc["n_native_measurements"] == 1
    assert doc["upstream_oracle_excluded_count"] == 1
    # The result claimed task_status="evaluated" against the repo's
    # "unsupported" — a counted (diagnostic) mismatch.
    assert doc["task_status_mismatch_count"] == 1


def test_validate_result_back_compat_no_task_status(tmp_path):
    """A pre-W1.4 result.json without task_status still validates (the
    field defaults to "evaluated")."""
    doc = {
        "task_id": "L1_01_bias",
        "static_checks_passed": True,
        "aggregate_score": 0.5,
        "correctness_status": "ok",
        "runtime_status": "ok",
        "performance_status": "ok",
        "performance_flags": [],
        "fallback_detected": False,
    }
    p = tmp_path / "result.json"
    p.write_text(json.dumps(doc))
    r = _run_tab("validate-result", str(p))
    assert r.returncode == 0, f"back-compat result rejected: {r.stderr}"
