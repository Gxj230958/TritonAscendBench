"""EventTimer is a sync-free record-only bracket with a deferred elapsed read.

The timer must NOT synchronize inside the bracket (that redundant in-bracket
sync inflated host_ns and biased small-kernel latency). It records start/end
only; elapsed_ns() reads start.elapsed_time(end) lazily AFTER the caller's
post-bracket device sync. An exception in the bracket leaves it incomplete and
elapsed_ns() returns None.
"""
from __future__ import annotations

from unittest import mock

import pytest

from triton_ascend_bench.backends.npu import EventTimer, NpuBackend


def _mock_npu():
    rec_order: list[str] = []
    elapsed_calls: list[tuple[str, str]] = []

    def make_event(name):
        ev = mock.MagicMock(name=name)
        ev._tab = name
        ev.record.side_effect = lambda n=name: rec_order.append(n)
        ev.elapsed_time.side_effect = lambda other, n=name: (
            elapsed_calls.append((n, other._tab)) or 2.0)
        return ev

    events = iter([make_event("start"), make_event("end")])
    fake = mock.MagicMock()
    fake.Event.side_effect = lambda *a, **kw: next(events)
    return fake, rec_order, elapsed_calls


def test_event_timer_defers_elapsed_read_past_exit():
    """elapsed_time is NOT read in __exit__; only when elapsed_ns() is called
    (after the caller's post-bracket sync)."""
    fake, rec_order, elapsed_calls = _mock_npu()
    with mock.patch("triton_ascend_bench.backends.npu._NPU_AVAILABLE", True), \
            mock.patch("torch.npu", fake, create=True):
        timer = EventTimer()
        with timer:
            pass
        # After __exit__: end recorded, but elapsed_time NOT yet read.
        assert rec_order == ["start", "end"]
        assert elapsed_calls == []
        ns = timer.elapsed_ns()
        # The deferred read fires now: start.elapsed_time(end), 2 ms -> 2e6 ns.
        assert elapsed_calls == [("start", "end")]
        assert ns == 2_000_000


def test_event_timer_exception_leaves_elapsed_none():
    """An exception in the bracket → end not recorded, elapsed_ns() is None."""
    fake, rec_order, elapsed_calls = _mock_npu()
    with mock.patch("triton_ascend_bench.backends.npu._NPU_AVAILABLE", True), \
            mock.patch("torch.npu", fake, create=True):
        timer = EventTimer()
        with pytest.raises(ValueError):
            with timer:
                raise ValueError("boom")
        assert rec_order == ["start"]          # end never recorded
        assert timer.elapsed_ns() is None      # incomplete bracket
        assert elapsed_calls == []


def test_event_timer_takes_no_sync_arg():
    """The timer no longer accepts a sync callable (the caller owns syncs)."""
    fake, _, _ = _mock_npu()
    with mock.patch("triton_ascend_bench.backends.npu._NPU_AVAILABLE", True), \
            mock.patch("torch.npu", fake, create=True):
        assert NpuBackend().event_timer() is not None
        with pytest.raises(TypeError):
            EventTimer(sync=lambda: None)
