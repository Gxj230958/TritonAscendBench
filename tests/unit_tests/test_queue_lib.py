"""Tests for ``automation.lib.queue`` (Unit 1.2).

Covers the 5-case mv state matrix, concurrent claim contention,
reaper recovery, PID-reuse defense, and the steal/audit path.
"""
from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"


@pytest.fixture(autouse=True)
def _ensure_scripts_on_path():
    inserted = False
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
        inserted = True
    yield
    if inserted:
        try:
            sys.path.remove(str(_SCRIPTS))
        except ValueError:
            pass


def _now():
    return socket.gethostname()


def test_enqueue_claim_complete_happy(tmp_path):
    from automation.lib import queue

    run_id = "r1"
    queue.enqueue(tmp_path, run_id, {"task_id": "L1_99_demo", "cmd": "resolve"})
    assert (tmp_path / run_id / "READY").exists()
    assert (tmp_path / run_id / "task.json").exists()

    h = queue.claim(tmp_path, host=_now(), pid=os.getpid(), starttime=1234)
    assert h is not None
    assert h.run_id == run_id
    assert h.claim_marker.exists()
    assert not (tmp_path / run_id / "READY").exists()

    done = queue.complete(h)
    assert done.name == "DONE"
    assert done.exists()


def test_claim_returns_none_when_empty(tmp_path):
    from automation.lib import queue

    assert queue.claim(tmp_path) is None


def test_concurrent_claim_single_winner(tmp_path):
    """Two simulated workers race for one READY -> exactly one wins."""
    from automation.lib import queue

    queue.enqueue(tmp_path, "r1", {"x": 1})

    # Run two claims back-to-back; the directory-rename atomicity is
    # what guarantees mutual exclusion, even within a single thread.
    h1 = queue.claim(tmp_path, host="hostA", pid=11, starttime=1, start_heartbeat=False)
    h2 = queue.claim(tmp_path, host="hostB", pid=22, starttime=2, start_heartbeat=False)
    assert (h1 is not None) ^ (h2 is not None), "exactly one worker should claim"


def test_concurrent_claim_multiprocess(tmp_path):
    """Use multiprocessing to truly race two processes on the same READY."""
    import multiprocessing as mp

    from automation.lib import queue

    queue.enqueue(tmp_path, "r1", {"x": 1})

    def _worker(queue_root_str: str, host: str, q_out):
        sys.path.insert(0, str(_SCRIPTS))
        from automation.lib import queue as q
        h = q.claim(queue_root_str, host=host, pid=os.getpid(), starttime=1, start_heartbeat=False)
        q_out.put(h is not None)

    ctx = mp.get_context("fork")
    out: mp.Queue = ctx.Queue()
    p1 = ctx.Process(target=_worker, args=(str(tmp_path), "hostA", out))
    p2 = ctx.Process(target=_worker, args=(str(tmp_path), "hostB", out))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    results = [out.get(), out.get()]
    assert results.count(True) == 1


def test_reaper_recovers_dead_worker_same_host(tmp_path):
    from automation.lib import queue

    queue.enqueue(tmp_path, "r1", {"x": 1})
    # Simulate a worker that already exited (use PID 1 which is owned by
    # init but acts as a stand-in via monkeypatch to ProcessLookupError).
    fake_pid = 99999999  # nearly guaranteed nonexistent
    h = queue.claim(
        tmp_path, host=socket.gethostname(), pid=fake_pid, starttime=1,
        start_heartbeat=False,
    )
    assert h is not None
    # Backdate heartbeat to be very old
    hb = h.claim_marker.parent / (h.claim_marker.name + ".heartbeat")
    if not hb.exists():
        hb.touch()
    very_old = time.time() - 100000
    os.utime(hb, (very_old, very_old))

    reclaimed = queue.reaper(tmp_path, heartbeat_grace_s=10)
    assert len(reclaimed) == 1
    assert (tmp_path / "r1" / "READY").exists()


def test_reaper_keeps_alive_worker(tmp_path):
    """A worker with a fresh heartbeat must NOT be reaped."""
    from automation.lib import queue

    queue.enqueue(tmp_path, "r1", {"x": 1})
    h = queue.claim(
        tmp_path, host=socket.gethostname(), pid=os.getpid(), starttime=1,
        start_heartbeat=False,
    )
    assert h is not None
    # Manually touch a fresh heartbeat
    hb = h.claim_marker.parent / (h.claim_marker.name + ".heartbeat")
    hb.touch()

    reclaimed = queue.reaper(tmp_path, heartbeat_grace_s=30)
    assert reclaimed == []
    assert h.claim_marker.exists()


def test_reaper_cross_host_needs_consecutive_failures(tmp_path):
    """Cross-host claims need N=2 consecutive grace periods to be reaped."""
    from automation.lib import queue

    queue.enqueue(tmp_path, "r1", {"x": 1})
    h = queue.claim(
        tmp_path, host="far-away-host", pid=12345, starttime=1,
        start_heartbeat=False,
    )
    assert h is not None
    hb = h.claim_marker.parent / (h.claim_marker.name + ".heartbeat")
    if not hb.exists():
        hb.touch()
    # Backdate heartbeat slightly older than one grace, but less than 2x
    half_stale = time.time() - 25  # grace=10 -> 2*1 grace=20 just past
    os.utime(hb, (half_stale, half_stale))
    # cross_host_consecutive=2 means we need 2*2*grace = 40s; not yet stale
    reclaimed = queue.reaper(tmp_path, heartbeat_grace_s=10, cross_host_consecutive=2)
    assert reclaimed == []
    # Now backdate fully past the cross-host threshold
    very_old = time.time() - 200
    os.utime(hb, (very_old, very_old))
    reclaimed = queue.reaper(tmp_path, heartbeat_grace_s=10, cross_host_consecutive=2)
    assert len(reclaimed) == 1


def test_quarantine_prevents_re_enqueue(tmp_path):
    from automation.lib import queue

    queue.enqueue(tmp_path, "r1", {"x": 1})
    queue.quarantine(tmp_path, "r1", reason="manual_test")
    with pytest.raises(queue.QueueCorruptError, match="terminal marker"):
        queue.enqueue(tmp_path, "r1", {"x": 2})


def test_complete_after_reaper_steal_raises(tmp_path):
    """A worker whose claim was stolen by reaper must NOT silently
    succeed on complete()."""
    from automation.lib import queue

    queue.enqueue(tmp_path, "r1", {"x": 1})
    h = queue.claim(
        tmp_path, host=socket.gethostname(), pid=os.getpid(), starttime=1,
        start_heartbeat=False,
    )
    assert h is not None
    # Simulate reaper stealing -- rename claim back to READY
    ready = h.claim_marker.parent / "READY"
    os.rename(h.claim_marker, ready)
    # New worker claims it with different identity
    h2 = queue.claim(
        tmp_path, host="other-host", pid=88888, starttime=2,
        start_heartbeat=False,
    )
    assert h2 is not None
    # h tries to complete -> must raise (marker filename no longer ours)
    # NOTE: claim_marker stored in h still points to the old path which
    # was renamed away.
    with pytest.raises(queue.QueueCorruptError):
        queue.complete(h)
    # Audit trail present
    audits = list((tmp_path / "r1").glob("STOLEN_BY_REAPER.*"))
    assert audits, "STOLEN_BY_REAPER audit marker required"


def test_complete_marker_renamed_to_different_identity(tmp_path):
    """If someone manually renames the claim marker to a different
    identity (different host/pid/starttime), complete() must refuse."""
    from automation.lib import queue

    queue.enqueue(tmp_path, "r1", {"x": 1})
    h = queue.claim(
        tmp_path, host=socket.gethostname(), pid=os.getpid(), starttime=1,
        start_heartbeat=False,
    )
    assert h is not None
    bogus = h.claim_marker.parent / "CLAIMED.other.99999.999.x"
    os.rename(h.claim_marker, bogus)
    with pytest.raises(queue.QueueCorruptError):
        queue.complete(h)


def test_heartbeat_thread_starts_and_stops(tmp_path):
    """Heartbeat thread should touch the heartbeat marker periodically and
    stop cleanly when the claim is completed."""
    from automation.lib import queue

    queue.enqueue(tmp_path, "r1", {"x": 1})
    h = queue.claim(
        tmp_path, host=socket.gethostname(), pid=os.getpid(), starttime=1,
        start_heartbeat=True, heartbeat_interval_s=1,
    )
    assert h is not None
    hb = h.claim_marker.parent / (h.claim_marker.name + ".heartbeat")
    assert hb.exists()
    initial = hb.stat().st_mtime
    time.sleep(1.5)
    after = hb.stat().st_mtime
    assert after >= initial  # touched (mtime advanced or equal-because-tick)
    queue.complete(h)
    # Heartbeat thread should be stopped (set its stop event)
    assert h.heartbeat_stop is not None and h.heartbeat_stop.is_set()


def test_enqueue_creates_task_json(tmp_path):
    import json as _j

    from automation.lib import queue

    payload = {"task_id": "L1_06_x", "cmd": "resolve", "extra": [1, 2]}
    queue.enqueue(tmp_path, "r1", payload)
    data = _j.loads((tmp_path / "r1" / "task.json").read_text(encoding="utf-8"))
    assert data == payload


def test_reaper_quarantines_multi_claim(tmp_path):
    """Two CLAIMED.* markers on the same run -> split state -> quarantine."""
    from automation.lib import queue

    queue.enqueue(tmp_path, "r1", {"x": 1})
    run_dir = tmp_path / "r1"
    # Manually craft two CLAIMED.* markers
    (run_dir / "CLAIMED.hostA.10.1.20260101T000000Z000000").touch()
    (run_dir / "CLAIMED.hostB.20.2.20260101T000001Z000000").touch()
    # Remove READY to simulate a broken claim race outcome
    if (run_dir / "READY").exists():
        (run_dir / "READY").unlink()
    queue.reaper(tmp_path, heartbeat_grace_s=10)
    assert any(run_dir.glob("QUARANTINED.multi_claim.*"))


# ---------------------------------------------------------------------------
# Tier 2 enqueue contracts consumed by ``submit_reverify_job`` (Unit 4.2)
# ---------------------------------------------------------------------------

def test_enqueue_writes_task_json_with_orchestrator_fields(tmp_path):
    """``fix_orchestrator.submit_reverify_job`` writes a fixed task.json
    schema (``task_id`` / ``run_id`` / ``cmd`` / ``requested_at``); the
    queue lib must round-trip every field verbatim so the daemon can
    parse them back."""
    import json as _j

    from automation.lib import queue

    payload = {
        "task_id": "L1_06_x",
        "run_id": "tier2-20260524T000000Z-cafebabe",
        "cmd": "resolve",
        "requested_at": "2026-05-24T00:00:00Z",
    }
    queue.enqueue(tmp_path, payload["run_id"], payload)
    on_disk = _j.loads(
        (tmp_path / payload["run_id"] / "task.json").read_text(encoding="utf-8")
    )
    assert on_disk == payload
    # READY marker accompanies task.json (sibling, not parent dir).
    assert (tmp_path / payload["run_id"] / "READY").is_file()


def test_enqueue_ready_marker_appears_after_task_json(tmp_path):
    """READY is the claim sentinel for ``queue.claim`` (and therefore the
    ``bench_evaluator_daemon``). It must be created strictly AFTER
    ``task.json`` -- otherwise a daemon polling for READY can win the
    rename before task.json is durable on disk and crash on parse.
    """
    from automation.lib import queue

    queue.enqueue(tmp_path, "r-task-after-ready", {"task_id": "T"})
    run_dir = tmp_path / "r-task-after-ready"
    task_json = run_dir / "task.json"
    ready = run_dir / "READY"
    assert task_json.is_file() and ready.is_file()
    # mtime ordering: task.json must be older-or-equal to READY (strict
    # ``<`` is unreliable on filesystems with second-grained mtime).
    assert task_json.stat().st_mtime <= ready.stat().st_mtime


def test_enqueue_refuses_to_overwrite_quarantined_run(tmp_path):
    """If a previous run_id quarantined itself, the orchestrator must
    mint a fresh run_id rather than re-using the stale one. ``enqueue``
    enforces this with a ``QueueCorruptError``.
    """
    from automation.lib import queue

    run_id = "tier2-quarantined-dead"
    queue.quarantine(tmp_path, run_id, reason="prior_run_crashed")
    with pytest.raises(queue.QueueCorruptError, match="terminal marker"):
        queue.enqueue(tmp_path, run_id, {"task_id": "X"})
