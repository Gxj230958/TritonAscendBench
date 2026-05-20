"""Unit tests for triton_ascend_bench.timing (host-side authoritative timer).

Covers the W1.3 reshape: `measure` now drives a per-trial input factory
(fresh input per measured trial) and `arbitrate_timing` applies the
specs/timing_arbitration.md rule table.
"""
from __future__ import annotations

import math
import time

from triton_ascend_bench.timing import (
    QUARANTINE_NS,
    arbitrate_timing,
    measure,
    time_one_run,
)


def test_time_one_run_returns_positive_host_ns():
    sample = time_one_run(lambda: time.sleep(0.001))
    assert sample.host_ns > 0
    # At least 0.5ms since we slept 1ms (with margin).
    assert sample.host_ns > 500_000
    # No event hook on the dev machine -> host-only sample.
    assert sample.event_ns is None
    assert sample.drift_status == "single_source"


def test_measure_runs_n_repeat_times_with_fresh_inputs():
    """`measure` calls the run fn once per warmup + measured trial, each
    on the input the factory produced for that trial index."""
    seen_indices: list[int] = []
    calls: list[int] = []

    def factory(t: int) -> int:
        seen_indices.append(t)
        return t

    def run(inp: int) -> None:
        calls.append(inp)

    samples = measure(factory, run, n_warmup=2, n_repeat=5)

    assert len(samples) == 5
    # 2 warmup (indices -1,-2) + 5 measured (0..4) = 7 factory/run calls.
    assert len(calls) == 7
    # Warmup uses negative indices, measured uses 0..n_repeat-1 — warmup
    # and measured inputs never coincide (R16-P1 fresh-input guarantee).
    assert sorted(seen_indices) == [-2, -1, 0, 1, 2, 3, 4]


def test_measure_durations_positive():
    samples = measure(lambda t: t, lambda inp: time.sleep(0.0005),
                      n_warmup=1, n_repeat=3)
    for s in samples:
        assert s.host_ns > 0


def test_arbitrate_timing_ok_path():
    """A normal millisecond-scale series yields no status overrides."""
    arb = arbitrate_timing([5_000_000, 5_100_000, 4_900_000])
    assert arb.runtime_status is None
    assert arb.performance_status is None
    assert arb.performance_flags == []


def test_arbitrate_timing_invalid_on_nonfinite():
    """A non-finite / non-positive host duration ⇒ invalid_timing."""
    arb = arbitrate_timing([5_000_000, float("nan"), 5_000_000])
    assert arb.runtime_status == "invalid_timing"
    assert "nonfinite_host_ns" in arb.performance_flags

    arb0 = arbitrate_timing([5_000_000, 0, 5_000_000])
    assert arb0.runtime_status == "invalid_timing"


def test_arbitrate_timing_empty_series_invalid():
    arb = arbitrate_timing([])
    assert arb.runtime_status == "invalid_timing"
    assert "no_timing_samples" in arb.performance_flags


def test_arbitrate_timing_quarantined_sub_microsecond():
    """A sub-µs median ⇒ performance_status=quarantined (likely no run)."""
    arb = arbitrate_timing([100, 120, 90])  # all < 1 µs
    assert arb.performance_status == "quarantined"
    assert QUARANTINE_NS == 1_000


def test_arbitrate_timing_invalid_stream_on_drift_plus_nondefault():
    """Host/event drift > 10% AND a profiler-detected non-default stream
    ⇒ invalid_timing + performance_flags has 'invalid_stream'."""
    host = [10_000_000, 10_000_000, 10_000_000]
    event = [2_000_000, 2_000_000, 2_000_000]  # ~80% drift
    arb = arbitrate_timing(host, event_ns=event, stream_nondefault=True)
    assert arb.runtime_status == "invalid_timing"
    assert "invalid_stream" in arb.performance_flags
    assert arb.drift_ratio is not None and arb.drift_ratio > 0.10


def test_arbitrate_timing_drift_without_nondefault_is_soft():
    """Drift > 10% but no non-default stream ⇒ soft 'host_event_drift'
    flag only, timing not invalidated."""
    host = [10_000_000, 10_000_000, 10_000_000]
    event = [2_000_000, 2_000_000, 2_000_000]
    arb = arbitrate_timing(host, event_ns=event, stream_nondefault=False)
    assert arb.runtime_status is None
    assert "host_event_drift" in arb.performance_flags
    assert "invalid_stream" not in arb.performance_flags


def test_arbitrate_timing_matched_event_no_flags():
    """Host and event within 10% ⇒ no drift flag."""
    host = [10_000_000, 10_000_000, 10_000_000]
    event = [9_500_000, 9_500_000, 9_500_000]  # 5% drift
    arb = arbitrate_timing(host, event_ns=event, stream_nondefault=True)
    assert arb.runtime_status is None
    assert "invalid_stream" not in arb.performance_flags
    assert math.isclose(arb.drift_ratio, 0.05, rel_tol=1e-6)
