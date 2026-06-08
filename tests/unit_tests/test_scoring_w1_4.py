"""W1.4 — Roofline scoring unit tests.

The dev machine has no NPU; these tests validate the *scoring logic*
(given W / M / π / β / t_npu, compute the right per-shape efficiency `e`
and the right task-level geomean `s_t`) against synthetic cost models and
peak tables — never the realness of `t_npu`. See plan W1.4 "约束".
"""
from __future__ import annotations

import pytest

from ascendbench.perf.cost_model import CostModel
from ascendbench.perf.scoring import (
    P_CAP,
    TASK_STATUS_EVALUATED,
    VALID_TASK_STATUS,
    DurationStats,
    RooflineError,
    TaskMeasurement,
    capped_geomean_native,
    compute_t_ref,
    duration_stats,
    geomean,
    native_coverage,
    perf_measurable,
    s_t_value,
    shape_efficiency,
)


# ---------------------------------------------------------------------------
# compute_t_ref — Roofline theoretical-latency floor.
# ---------------------------------------------------------------------------
class TestComputeTRef:
    def _cm(self, flops, bytes_):
        return CostModel(flops=flops, bytes=bytes_)

    def test_compute_bound_picks_max_flops_term(self):
        # W_c = 2e12 FLOP at π = 1e12 FLOP/s  -> 2.0 s.
        # M_r = 1e9 byte at β = 1e12 byte/s   -> 0.001 s.
        # t_ref = max(2.0, 0.001) = 2.0 (compute-bound).
        cm = self._cm({"fp16_cube": "2e12"}, {"hbm": "1e9"})
        t_ref = compute_t_ref(
            cm, knobs={}, compute_peaks={"fp16_cube": 1e12},
            resource_peaks={"hbm": 1e12})
        assert t_ref == pytest.approx(2.0)

    def test_memory_bound_picks_max_bytes_term(self):
        # M_r = 4e12 byte at β = 1e12 byte/s -> 4.0 s dominates.
        cm = self._cm({"fp16_cube": "1e12"}, {"hbm": "4e12"})
        t_ref = compute_t_ref(
            cm, knobs={}, compute_peaks={"fp16_cube": 1e12},
            resource_peaks={"hbm": 1e12})
        assert t_ref == pytest.approx(4.0)

    def test_formula_uses_size_knobs(self):
        # W = 2*B*N FLOPs; with B=4 N=8 -> 64. π = 8 -> 8.0 s.
        cm = self._cm({"vector": "2*B*N"}, {"hbm": "1"})
        t_ref = compute_t_ref(
            cm, knobs={"B": 4, "N": 8}, compute_peaks={"vector": 8.0},
            resource_peaks={"hbm": 1e12})
        assert t_ref == pytest.approx(8.0)

    def test_max_over_multiple_classes(self):
        cm = self._cm(
            {"fp16_cube": "1e12", "vector": "3e12"}, {"hbm": "1"})
        t_ref = compute_t_ref(
            cm, knobs={}, compute_peaks={"fp16_cube": 1e12, "vector": 1e12},
            resource_peaks={"hbm": 1e12})
        # max(1e12/1e12, 3e12/1e12, ~0) = 3.0
        assert t_ref == pytest.approx(3.0)

    def test_unknown_class_fail_closed(self):
        cm = self._cm({"nonsense_class": "1"}, {"hbm": "1"})
        with pytest.raises(RooflineError, match="nonsense_class"):
            compute_t_ref(cm, knobs={}, compute_peaks={"fp16_cube": 1e12},
                          resource_peaks={"hbm": 1e12})

    def test_unverified_peak_fail_closed(self):
        # A non-positive peak (an [unverified] placeholder) must raise,
        # never silently distort t_ref.
        cm = self._cm({"vector": "1"}, {"hbm": "1"})
        with pytest.raises(RooflineError, match="non-positive"):
            compute_t_ref(cm, knobs={}, compute_peaks={"vector": 0.0},
                          resource_peaks={"hbm": 1e12})

    def test_dtype_size_derived_symbol(self):
        # bytes formula uses the derived symbol dtype_size.
        cm = self._cm({"vector": "1"}, {"hbm": "B*dtype_size"})
        t_ref = compute_t_ref(
            cm, knobs={"B": 10}, compute_peaks={"vector": 1e30},
            resource_peaks={"hbm": 1.0}, dtype_size=4.0)
        # bytes term = 10*4 / 1.0 = 40.0; flops term ~0.
        assert t_ref == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# shape_efficiency — e = min(P_CAP, t_ref / t_npu).
# ---------------------------------------------------------------------------
class TestShapeEfficiency:
    def test_exact_roofline_is_one(self):
        assert shape_efficiency(t_ref=1.0, t_npu=1.0) == 1.0

    def test_slower_than_roofline(self):
        # t_npu twice the floor -> e = 0.5.
        assert shape_efficiency(t_ref=1.0, t_npu=2.0) == pytest.approx(0.5)

    def test_capped_at_p_cap(self):
        # t_npu far below the floor (cost-model under-estimate) -> clamped.
        assert shape_efficiency(t_ref=100.0, t_npu=1.0) == P_CAP

    def test_nan_t_npu_rejected(self):
        with pytest.raises(ValueError):
            shape_efficiency(t_ref=1.0, t_npu=float("nan"))

    def test_zero_t_npu_rejected(self):
        with pytest.raises(ValueError):
            shape_efficiency(t_ref=1.0, t_npu=0.0)


# ---------------------------------------------------------------------------
# geomean — the per-shape e -> task-level s_t reducer (DESIGN §9).
# ---------------------------------------------------------------------------
class TestGeomean:
    def test_geomean_two_values(self):
        # geomean(0.5, 2.0) = 1.0
        assert geomean([0.5, 2.0]) == pytest.approx(1.0)

    def test_geomean_single(self):
        assert geomean([0.7]) == pytest.approx(0.7)

    def test_geomean_empty_raises(self):
        with pytest.raises(ValueError):
            geomean([])

    def test_geomean_nonpositive_raises(self):
        with pytest.raises(ValueError):
            geomean([1.0, 0.0])


# ---------------------------------------------------------------------------
# duration_stats — DESIGN §6.6 median / p99 / cv / trials.
# ---------------------------------------------------------------------------
class TestDurationStats:
    def test_basic_stats(self):
        st = duration_stats([10.0, 20.0, 30.0])
        assert st.trials == 3
        assert st.median == pytest.approx(20.0)
        assert st.p99 == pytest.approx(30.0)  # nearest-rank top sample
        assert st.cv is not None and st.cv > 0

    def test_empty_series_all_none(self):
        st = duration_stats([])
        assert st == DurationStats(trials=0, median=None, p99=None, cv=None)

    def test_drops_nonfinite_and_nonpositive(self):
        st = duration_stats([5.0, 0.0, -1.0, float("inf"), 15.0])
        assert st.trials == 2
        assert st.median == pytest.approx(10.0)

    def test_single_sample_cv_zero(self):
        st = duration_stats([42.0])
        assert st.trials == 1
        assert st.cv == 0.0


# ---------------------------------------------------------------------------
# Roofline end-to-end: W / M / π / β / t_npu  ->  e.
# ---------------------------------------------------------------------------
def test_roofline_efficiency_from_w_m_pi_beta_t_npu():
    """Given W_c / M_r / π_c / β_r and a measured t_npu, the per-shape
    efficiency is min(P_CAP, t_ref/t_npu) — the core W1.4 scoring path."""
    cm = CostModel(flops={"vector": "2*B*N"}, bytes={"hbm": "4*B*N*dtype_size"})
    knobs = {"B": 8, "N": 64}
    # W = 2*8*64 = 1024 FLOPs;  π = 1024 FLOP/s  -> 1.0 s
    # M = 4*8*64*2 = 4096 byte; β = 8192 byte/s  -> 0.5 s
    # t_ref = max(1.0, 0.5) = 1.0 s
    t_ref = compute_t_ref(
        cm, knobs, compute_peaks={"vector": 1024.0},
        resource_peaks={"hbm": 8192.0}, dtype_size=2.0)
    assert t_ref == pytest.approx(1.0)
    # measured t_npu = 4.0 s  -> e = min(5.0, 1.0/4.0) = 0.25
    e = shape_efficiency(t_ref, t_npu=4.0)
    assert e == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# TaskMeasurement.s_t consumed by s_t_value / capped_geomean_native.
# ---------------------------------------------------------------------------
class TestSTConsumption:
    def test_s_t_value_uses_pre_aggregated_s_t(self):
        m = TaskMeasurement(task_id="t", s_t=0.8)
        assert s_t_value(m) == pytest.approx(0.8)

    def test_s_t_value_clamps_to_p_cap(self):
        m = TaskMeasurement(task_id="t", s_t=999.0)
        assert s_t_value(m) == P_CAP

    def test_s_t_value_fail_status_collapses_to_eps(self):
        m = TaskMeasurement(task_id="t", s_t=0.8, correctness_status="fail")
        assert s_t_value(m) == pytest.approx(0.01)

    def test_capped_geomean_consumes_task_level_s_t(self):
        ms = [
            TaskMeasurement(task_id="a", s_t=0.5),
            TaskMeasurement(task_id="b", s_t=2.0),
        ]
        # geomean(0.5, 2.0) = 1.0
        assert capped_geomean_native(ms, s_native_size=2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# task_status — the new orthogonal evaluability field.
# ---------------------------------------------------------------------------
class TestTaskStatus:
    def test_default_is_evaluated(self):
        m = TaskMeasurement(task_id="t")
        assert m.task_status == TASK_STATUS_EVALUATED
        assert m.is_evaluated

    def test_enum_members(self):
        assert VALID_TASK_STATUS == {"evaluated", "unsupported", "blocked"}

    def test_unsupported_dropped_from_geomean(self):
        # W4.3 ④ / R8-1: an `unsupported` task is OUT of the scored set —
        # dropped from numerator AND denominator (no provider golden for
        # this mode). NOT ε-collapsed (that would unfairly depress score).
        ms = [
            TaskMeasurement(task_id="a", s_t=1.0),
            TaskMeasurement(task_id="b", s_t=1.0, task_status="unsupported"),
        ]
        # Only task a counts; geomean of {1.0} over a denom of 2-1=1 = 1.0.
        assert capped_geomean_native(ms, s_native_size=2) == pytest.approx(1.0)

    def test_blocked_stays_in_denominator_as_eps(self):
        # W4.3 ④ / R8-1: a `blocked` task MUST stay in the fixed |S|
        # denominator and be scored as a failure (ε) — it must NOT escape
        # the denominator (no-subset-escape; BLOCKED is the publish-gate's
        # target). With s_native_size=2, geomean(1.0, ε) = sqrt(1.0*0.01)
        # = 0.1 — NOT 1.0 (which would mean blocked silently shrank |S|).
        ms = [
            TaskMeasurement(task_id="a", s_t=1.0),
            TaskMeasurement(task_id="b", s_t=1.0, task_status="blocked"),
        ]
        score = capped_geomean_native(ms, s_native_size=2)
        assert score == pytest.approx(0.1)  # sqrt(1.0 * ε), ε in denom

    def test_blocked_fastp_threshold_not_met(self):
        # W4.3 ④ / R8-1: a `blocked` task counts as threshold-not-met and
        # stays in the |S| denominator — FastP@1.0 = 1/2 = 0.5, not 1/1.
        from ascendbench.perf.scoring import fastp_at_threshold
        ms = [
            TaskMeasurement(task_id="a", s_t=1.0),
            TaskMeasurement(task_id="b", s_t=1.0, task_status="blocked"),
        ]
        assert fastp_at_threshold(
            ms, threshold=1.0, s_native_size=2) == pytest.approx(0.5)

    def test_unsupported_excluded_from_coverage(self):
        ms = [
            TaskMeasurement(task_id="a", correctness_status="ok"),
            TaskMeasurement(task_id="b", correctness_status="ok",
                            task_status="unsupported"),
        ]
        # b out of the scored set: coverage = 1 / (2 - 1) = 1.0.
        assert native_coverage(ms, s_native_size=2) == pytest.approx(1.0)

    def test_blocked_counts_against_coverage(self):
        # W4.3 ④ / R8-1: a `blocked` task stays in the |S| denominator and
        # is NOT an `ok` — even with correctness_status="ok" on the carrier
        # it cannot clear the gate (no usable golden). coverage = 1/2 = 0.5.
        ms = [
            TaskMeasurement(task_id="a", correctness_status="ok"),
            TaskMeasurement(task_id="b", correctness_status="ok",
                            task_status="blocked"),
        ]
        assert native_coverage(ms, s_native_size=2) == pytest.approx(0.5)

    def test_non_evaluated_not_perf_measurable(self):
        m = TaskMeasurement(task_id="t", s_t=1.0, task_status="blocked")
        assert not perf_measurable(m)
