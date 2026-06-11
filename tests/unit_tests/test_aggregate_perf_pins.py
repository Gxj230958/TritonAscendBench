"""Pinned regressions: aggregate_perf fixed-denominator semantics.

Pins the two R9 MUST-FIX findings so a regression re-surfaces as a test
failure:

  * R9-1 — ``scoring.aggregate_perf`` no longer short-circuits to
    ``perf_status="unavailable"`` when no task is ``perf_measurable``.
    DESIGN §9 / P6 fixed-denominator semantics: a real candidate failure
    (``blocked`` / a non-ok status / a detected torch fallback) stays in
    the fixed ``|S|`` denominator and is scored as ``ε`` (R8-1: no
    subset-escape). Only a *benchmark-data* gap — a task whose Roofline
    data is incomplete (no cost model / a referenced peak still an
    ``[unverified]`` placeholder) — is honestly excluded from the perf
    geomean rather than ε-collapsed against the candidate's score; when
    EVERY in-scored-set task is such a gap the status is
    ``"roofline_data_incomplete"``, not ``"unavailable"``.
  * R9-2 — ``eval._apply_roofline`` no longer loads the WHOLE peak table
    with ``require_verified=True`` (which let any one unrelated
    ``[unverified]`` entry disable Roofline scoring for every task). It
    loads the table unverified and checks π/β only for the
    compute-class / resource keys THIS task's normalised ``cost_model``
    actually references — a task that only touches verified keys
    (``fp16_cube`` / ``int8`` / ``hbm``) scores fine even while
    ``vector`` / ``interconnect`` are still placeholders.

Dev machine has no NPU; these verify the scoring / Roofline plumbing
with synthetic measurements and peak tables.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ascendbench.engine import eval as _eval
from ascendbench.perf import peak_table
from ascendbench.perf.cost_model import CostModel
from ascendbench.engine.eval import ShapeRunResult, TaskEvalResult
from ascendbench.perf.scoring import (
    EPSILON,
    TaskMeasurement,
    aggregate_perf,
)


# ──────────────────────────────────────────────────────────────────────────
# R9-1 — aggregate_perf: blocked-ε vs Roofline-data-incomplete.
# ──────────────────────────────────────────────────────────────────────────
class TestR9AggregatePerf:
    def test_all_failures_score_as_epsilon_not_unavailable(self):
        """A corpus of nothing-but-failures (a correctness fail + a
        ``blocked`` task) is an HONEST ε-floored geomean, NOT a
        short-circuit to ``perf_status="unavailable"``. The failures stay
        in the fixed ``|S|`` denominator (R8-1 no-subset-escape)."""
        ms = [
            TaskMeasurement(task_id="a", correctness_status="fail"),
            TaskMeasurement(task_id="b", task_status="blocked",
                            correctness_status="error"),
        ]
        out = aggregate_perf(ms, eligible_size=2)
        assert out["perf_status"] == "ok"
        # geomean(ε, ε) == ε — every task failed, none escaped.
        assert out["capped_geomean_native"] == pytest.approx(EPSILON)
        assert out["fastp_native_at_1_0"] == pytest.approx(0.0)
        assert out["roofline_data_incomplete_count"] == 0

    def test_blocked_stays_in_perf_denominator(self):
        """A ``blocked`` task must stay in ``|S|`` and pull the geomean
        down with ε — it must NOT silently shrink the denominator."""
        ms = [
            TaskMeasurement(task_id="a", s_t=1.0),
            TaskMeasurement(task_id="b", task_status="blocked",
                            correctness_status="error"),
        ]
        out = aggregate_perf(ms, eligible_size=2)
        assert out["perf_status"] == "ok"
        # geomean(1.0, ε) = sqrt(ε) — blocked is an ε term in |S|=2.
        assert out["capped_geomean_native"] == pytest.approx(EPSILON ** 0.5)

    def test_all_roofline_data_incomplete_is_distinct_status(self):
        """When EVERY in-scored-set task is uncomputable purely because
        its Roofline data is incomplete, the status is
        ``roofline_data_incomplete`` with null metrics — the benchmark
        data gap is reported honestly, never as a candidate score."""
        ms = [
            TaskMeasurement(task_id="a", roofline_data_complete=False),
            TaskMeasurement(task_id="b", roofline_data_complete=False),
        ]
        out = aggregate_perf(ms, eligible_size=2)
        assert out["perf_status"] == "roofline_data_incomplete"
        assert out["capped_geomean_native"] is None
        assert out["fastp_native_at_1_0"] is None
        assert out["roofline_data_incomplete_count"] == 2

    def test_data_incomplete_task_excluded_not_epsilon_pressed(self):
        """A Roofline-data-incomplete task is NOT the candidate's fault —
        it is dropped from the perf geomean's numerator AND denominator,
        not ε-collapsed. With one scorable task (s_t=2.0) and one
        data-gap task over |S|=2, the geomean is 2.0 (the data-gap task
        gone), not sqrt(2.0*ε)."""
        ms = [
            TaskMeasurement(task_id="a", s_t=2.0),
            TaskMeasurement(task_id="b", roofline_data_complete=False),
        ]
        out = aggregate_perf(ms, eligible_size=2)
        assert out["perf_status"] == "ok"
        # denom = |S|(2) - data_gap(1) = 1 ; geomean({2.0}) = 2.0.
        assert out["capped_geomean_native"] == pytest.approx(2.0)
        assert out["roofline_data_incomplete_count"] == 1

    def test_real_failure_kept_even_when_data_incomplete_flag_false(self):
        """A genuine candidate failure must score ε even if its
        ``roofline_data_complete`` flag is False — a failure is a failure;
        the data-gap exemption is ONLY for status-ok tasks."""
        ms = [
            TaskMeasurement(task_id="a", s_t=1.0),
            TaskMeasurement(task_id="b", correctness_status="fail",
                            roofline_data_complete=False),
        ]
        out = aggregate_perf(ms, eligible_size=2)
        assert out["perf_status"] == "ok"
        # the failure stays an ε term — geomean(1.0, ε) = sqrt(ε).
        assert out["capped_geomean_native"] == pytest.approx(EPSILON ** 0.5)
        # it is a real failure, NOT counted as a data gap.
        assert out["roofline_data_incomplete_count"] == 0

    def test_mixed_scorable_failure_and_data_gap(self):
        """1 scorable + 1 real failure + 1 data-gap over |S|=3: the
        data-gap leaves the denominator (3-1=2); the failure stays as ε."""
        ms = [
            TaskMeasurement(task_id="a", s_t=2.0),
            TaskMeasurement(task_id="b", correctness_status="fail"),
            TaskMeasurement(task_id="c", roofline_data_complete=False),
        ]
        out = aggregate_perf(ms, eligible_size=3)
        assert out["perf_status"] == "ok"
        # geomean(2.0, ε) over denom 2 = sqrt(2.0 * ε).
        assert out["capped_geomean_native"] == pytest.approx(
            (2.0 * EPSILON) ** 0.5)
        assert out["roofline_data_incomplete_count"] == 1

    def test_unsupported_dropped_before_partition(self):
        """An ``unsupported`` task is out of the scored set entirely — it
        must not be counted as a data gap nor as a failure."""
        ms = [
            TaskMeasurement(task_id="a", s_t=1.0),
            TaskMeasurement(task_id="b", task_status="unsupported"),
        ]
        out = aggregate_perf(ms, eligible_size=2)
        assert out["perf_status"] == "ok"
        # denom = |S|(2) - unsupported(1) = 1 ; geomean({1.0}) = 1.0.
        assert out["capped_geomean_native"] == pytest.approx(1.0)
        assert out["roofline_data_incomplete_count"] == 0


# ──────────────────────────────────────────────────────────────────────────
# R9-2 — _apply_roofline: per-task π/β verification, not whole-table.
# ──────────────────────────────────────────────────────────────────────────
# A peak table with ONE unverified placeholder (`vector` π = 0.0) and the
# rest verified — mirrors the real `ascendbench/policies/peak_table_910b.toml` state.
def _mixed_peaks(_path=None, *, require_verified=False):
    compute = {"fp16_cube": 256e12, "bf16_cube": 0.0, "int8": 512e12,
               "vector": 0.0}
    resource = {"hbm": 1.6e12, "interconnect": 0.0}
    return compute, resource


_SHAPES_DOC = {
    "size_knobs": {"B": 2, "N": 4},
    "public_shapes": [{"id": "P1", "knobs": {"B": 2, "N": 4}}],
}


def _make_result(durations: list[float]) -> TaskEvalResult:
    from ascendbench.perf.scoring import duration_stats
    r = TaskEvalResult(task_id="T", static_checks_passed=True)
    st = duration_stats(durations)
    r.shape_results.append(ShapeRunResult(
        shape_id="P1", correctness_passed=True, correctness_reason=None,
        measured_ns=None, baseline_ns=None, durations=list(durations),
        t_npu=st.median, median=st.median, p99=st.p99, cv=st.cv,
        trials=st.trials))
    return r


def test_r9_2_fp16_task_unaffected_by_unrelated_placeholder(monkeypatch):
    """A task whose cost model only references VERIFIED keys (`fp16_cube`
    + `hbm`) scores its Roofline `s_t` even though the SAME peak table
    holds `vector` / `interconnect` as `[unverified]` placeholders. The
    old whole-table `require_verified=True` load would have raised on
    those and disabled Roofline for this task too."""
    monkeypatch.setattr(peak_table, "load_peak_table", _mixed_peaks)
    task = SimpleNamespace(cost_model=CostModel(
        flops={"fp16_cube": "B*N"}, bytes={"hbm": "B*N"}))
    result = _make_result([16.0])
    _eval._apply_roofline(result, task, _SHAPES_DOC)
    # the task scored — no roofline_error, s_t present, data complete.
    assert result.shape_results[0].roofline_error is None
    assert result.shape_results[0].efficiency is not None
    assert result.s_t is not None
    assert result.roofline_data_complete is True


def test_r9_2_vector_task_flagged_data_incomplete(monkeypatch):
    """A task referencing a 0.0 (unfilled) `vector` π is a Roofline-data gap —
    `roofline_error`, no `s_t`, `roofline_data_complete=False`. Not a candidate
    failure. (The default table now fills vector; this mocks a 0.0 vector.)"""
    monkeypatch.setattr(peak_table, "load_peak_table", _mixed_peaks)
    task = SimpleNamespace(cost_model=CostModel(
        flops={"vector": "B*N"}, bytes={"hbm": "B*N"}))
    result = _make_result([16.0])
    _eval._apply_roofline(result, task, _SHAPES_DOC)
    assert result.shape_results[0].roofline_error is not None
    assert "no positive value" in result.shape_results[0].roofline_error
    assert result.s_t is None
    assert result.roofline_data_complete is False


def test_r9_2_no_cost_model_flags_data_incomplete(monkeypatch):
    """A task with no cost model is a benchmark-data gap (cost-model
    backfill not done) — `roofline_data_complete=False`, so scoring
    excludes it from the perf geomean rather than ε-collapsing it."""
    monkeypatch.setattr(peak_table, "load_peak_table", _mixed_peaks)
    result = _make_result([16.0])
    _eval._apply_roofline(result, SimpleNamespace(cost_model=None),
                          _SHAPES_DOC)
    assert result.s_t is None
    assert result.roofline_data_complete is False


def test_r9_2_unverified_resource_flags_the_task(monkeypatch):
    """A task referencing the unverified `interconnect` β is flagged the
    same way — per-resource verification, not just per-compute-class."""
    monkeypatch.setattr(peak_table, "load_peak_table", _mixed_peaks)
    task = SimpleNamespace(cost_model=CostModel(
        flops={"fp16_cube": "B*N"}, bytes={"interconnect": "B*N"}))
    result = _make_result([16.0])
    _eval._apply_roofline(result, task, _SHAPES_DOC)
    assert result.shape_results[0].roofline_error is not None
    assert result.roofline_data_complete is False


def test_r9_2_task_measurement_carries_data_complete_flag(monkeypatch):
    """The `roofline_data_complete` flag propagates onto the
    `TaskMeasurement` so `aggregate_perf` can act on it."""
    monkeypatch.setattr(peak_table, "load_peak_table", _mixed_peaks)
    task = SimpleNamespace(cost_model=CostModel(
        flops={"vector": "B*N"}, bytes={"hbm": "B*N"}))
    result = _make_result([16.0])
    result.correctness_status = "ok"
    result.runtime_status = "ok"
    result.performance_status = "ok"
    _eval._apply_roofline(result, task, _SHAPES_DOC)
    tm = result.task_measurement()
    assert tm.roofline_data_complete is False
