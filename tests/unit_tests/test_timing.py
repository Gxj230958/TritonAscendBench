"""Unit tests for triton_ascend_bench.timing (host-side authoritative timer)."""
from __future__ import annotations

import time

from triton_ascend_bench.timing import measure, time_one_run


def test_time_one_run_returns_positive_host_ns():
    sample = time_one_run(lambda: time.sleep(0.001))
    assert sample.host_ns > 0
    # At least 0.5ms since we slept 1ms (with margin).
    assert sample.host_ns > 500_000


def test_measure_runs_n_repeat_times():
    counter = []
    durations = measure(lambda: counter.append(1), n_warmup=2, n_repeat=5)
    assert len(durations) == 5
    # Counter was called 2 (warmup) + 5 = 7 times.
    assert len(counter) == 7


def test_measure_durations_positive():
    durations = measure(lambda: time.sleep(0.0005), n_warmup=1, n_repeat=3)
    for d in durations:
        assert d > 0
