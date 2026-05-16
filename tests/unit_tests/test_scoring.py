"""Unit tests for triton_ascend_bench.scoring per specs/scoring_spec.md.

Post-codex P1-2 fix:
- P_CAP = 5 (not 10)
- Failed task contributes ε=0.01 to geomean (NOT excluded)
- All failures → geomean = ε, not 0
- native_coverage / is_rank_eligible use |S_native| as denominator (fixed)
- FastP@1.0 and FastP@1.5 are headline indicators
"""
from __future__ import annotations

import math

import pytest

from triton_ascend_bench.scoring import (
    EPSILON,
    MIN_COVERAGE_FOR_RANKING,
    P_CAP,
    ShapeMeasurement,
    TaskMeasurement,
    capped_geomean_native,
    coverage_for_ranking,
    fastp_at_threshold,
    fastp_native,
    is_rank_eligible,
    native_coverage,
    s_t_value,
)


class TestFastPLegacy:
    def test_parity_is_one(self):
        assert fastp_native(1000, 1000) == 1.0

    def test_twice_as_fast(self):
        assert fastp_native(500, 1000) == 2.0

    def test_half_speed(self):
        assert fastp_native(2000, 1000) == 0.5

    def test_clamped_at_p_cap(self):
        assert P_CAP == 5.0
        assert fastp_native(10, 1000) == P_CAP

    def test_zero_measured_raises(self):
        with pytest.raises(ValueError):
            fastp_native(0, 1000)

    def test_negative_baseline_raises(self):
        with pytest.raises(ValueError):
            fastp_native(1000, -1)


class TestSTValue:
    def test_all_ok_passes_through_speedup(self):
        m = TaskMeasurement(
            task_id="t1", measured_ns=500, baseline_ns=1000
        )
        assert s_t_value(m) == 2.0

    def test_cap_applied(self):
        m = TaskMeasurement(task_id="t1", measured_ns=10, baseline_ns=1000)
        assert s_t_value(m) == P_CAP

    def test_correctness_fail_yields_epsilon(self):
        m = TaskMeasurement(
            task_id="t1", correctness_status="fail",
            measured_ns=500, baseline_ns=1000,
        )
        assert s_t_value(m) == EPSILON

    def test_runtime_invalid_yields_epsilon(self):
        m = TaskMeasurement(
            task_id="t1", runtime_status="invalid_timing",
            measured_ns=500, baseline_ns=1000,
        )
        assert s_t_value(m) == EPSILON

    def test_fallback_detected_yields_epsilon(self):
        m = TaskMeasurement(
            task_id="t1", fallback_detected=True,
            measured_ns=500, baseline_ns=1000,
        )
        assert s_t_value(m) == EPSILON

    def test_missing_baseline_yields_epsilon(self):
        m = TaskMeasurement(task_id="t1", measured_ns=500, baseline_ns=None)
        assert s_t_value(m) == EPSILON


class TestGeomean:
    def test_two_passing_tasks(self):
        m = [
            TaskMeasurement(task_id="t1", measured_ns=500, baseline_ns=1000),
            TaskMeasurement(task_id="t2", measured_ns=400, baseline_ns=1000),
        ]
        # geomean(2.0, 2.5) = sqrt(5)
        assert math.isclose(
            capped_geomean_native(m, s_native_size=2),
            math.sqrt(5.0),
            rel_tol=1e-9,
        )

    def test_failed_task_contributes_epsilon(self):
        """Per spec, failed task → s_t=ε, NOT excluded."""
        m = [
            TaskMeasurement(task_id="t1", measured_ns=500, baseline_ns=1000),
            TaskMeasurement(
                task_id="t2", correctness_status="fail",
                measured_ns=None, baseline_ns=None,
            ),
        ]
        # geomean(2.0, ε) = sqrt(2.0 * ε) ≈ 0.1414
        expected = math.sqrt(2.0 * EPSILON)
        assert math.isclose(
            capped_geomean_native(m, s_native_size=2),
            expected,
            rel_tol=1e-9,
        )

    def test_all_failures_yield_epsilon_not_zero(self):
        m = [
            TaskMeasurement(task_id="t1", correctness_status="fail"),
            TaskMeasurement(task_id="t2", correctness_status="fail"),
        ]
        assert math.isclose(
            capped_geomean_native(m, s_native_size=2),
            EPSILON,
            rel_tol=1e-9,
        )

    def test_missing_task_pads_with_epsilon(self):
        """|S_native|=5 but only 1 attempted → other 4 are ε."""
        m = [TaskMeasurement(task_id="t1", measured_ns=500, baseline_ns=1000)]
        expected = math.exp(
            (math.log(2.0) + 4 * math.log(EPSILON)) / 5
        )
        assert math.isclose(
            capped_geomean_native(m, s_native_size=5),
            expected,
            rel_tol=1e-9,
        )


class TestFastPThresholds:
    def test_fastp_at_1_0(self):
        m = [
            TaskMeasurement(task_id="t1", measured_ns=500, baseline_ns=1000),  # 2.0
            TaskMeasurement(task_id="t2", measured_ns=1500, baseline_ns=1000),  # 0.67
            TaskMeasurement(task_id="t3", measured_ns=900, baseline_ns=1000),  # 1.11
        ]
        # t1 (2.0) and t3 (1.11) ≥ 1.0 → 2/3
        assert math.isclose(
            fastp_at_threshold(m, threshold=1.0, s_native_size=3),
            2 / 3,
            rel_tol=1e-9,
        )

    def test_fastp_at_1_5(self):
        m = [
            TaskMeasurement(task_id="t1", measured_ns=500, baseline_ns=1000),
            TaskMeasurement(task_id="t2", measured_ns=700, baseline_ns=1000),  # 1.43
            TaskMeasurement(task_id="t3", measured_ns=900, baseline_ns=1000),  # 1.11
        ]
        # Only t1 (2.0) >= 1.5 → 1/3
        assert math.isclose(
            fastp_at_threshold(m, threshold=1.5, s_native_size=3),
            1 / 3,
            rel_tol=1e-9,
        )


class TestNativeCoverage:
    def test_all_pass_full_coverage(self):
        m = [
            TaskMeasurement(task_id="t1"),
            TaskMeasurement(task_id="t2"),
        ]
        assert native_coverage(m, s_native_size=2) == 1.0
        assert is_rank_eligible(m, s_native_size=2)

    def test_below_threshold_not_eligible(self):
        m = [
            TaskMeasurement(task_id="t1"),
            TaskMeasurement(task_id="t2", correctness_status="fail"),
            TaskMeasurement(task_id="t3", correctness_status="fail"),
        ]
        assert native_coverage(m, s_native_size=3) < MIN_COVERAGE_FOR_RANKING
        assert not is_rank_eligible(m, s_native_size=3)

    def test_coverage_uses_fixed_denominator(self):
        """Even if only 1 task attempted, denominator stays |S_native|."""
        m = [TaskMeasurement(task_id="t1")]
        # 1 ok out of 5 native tasks → 20%, well below 70%
        assert native_coverage(m, s_native_size=5) == 0.2
        assert not is_rank_eligible(m, s_native_size=5)


class TestDeduplication:
    """codex round-2 P1: duplicate task_id must error, not silently inflate."""

    def test_duplicate_task_id_raises_in_geomean(self):
        m = [
            TaskMeasurement(task_id="t1", measured_ns=500, baseline_ns=1000),
            TaskMeasurement(task_id="t1", measured_ns=500, baseline_ns=1000),
        ]
        with pytest.raises(ValueError, match="duplicate task_id"):
            capped_geomean_native(m, s_native_size=1)

    def test_duplicate_task_id_raises_in_coverage(self):
        from triton_ascend_bench.scoring import native_coverage
        m = [
            TaskMeasurement(task_id="t1"),
            TaskMeasurement(task_id="t1"),
        ]
        with pytest.raises(ValueError, match="duplicate task_id"):
            native_coverage(m, s_native_size=1)

    def test_coverage_clamped_to_one(self):
        """Even if caller passes more correct tasks than s_native_size, coverage ≤ 1.0."""
        from triton_ascend_bench.scoring import native_coverage
        m = [
            TaskMeasurement(task_id="t1"),
            TaskMeasurement(task_id="t2"),
            TaskMeasurement(task_id="t3"),
        ]
        assert native_coverage(m, s_native_size=2) == 1.0


class TestLegacyShapeAPI:
    """Back-compat tests for the per-shape API used by older eval paths."""
    def test_coverage_for_ranking_shape_api(self):
        m = [
            ShapeMeasurement("t1", "P1", True, 1, 1),
            ShapeMeasurement("t1", "P2", False, None, None),
        ]
        assert coverage_for_ranking(m) == 0.5
