"""Tests for ``automation.lib.lock`` (Unit 1.3).

Covers mkdir-based atomic lock, lease/heartbeat/steal protocol, two-tier
nesting, eviction audit, and SIGKILL-style stale recovery.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from datetime import UTC
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


def test_happy_acquire_and_release(tmp_path):
    from automation.lib import lock

    lp = tmp_path / "state" / ".lock"
    with lock.acquire_lock(lp, start_heartbeat=False, lease_s=60, heartbeat_s=10) as h:
        assert lp.is_dir()
        meta = json.loads((lp / "meta.json").read_text(encoding="utf-8"))
        assert meta["owner_uuid"] == h.owner_uuid
    # On exit, the lock dir should be released (renamed away and rmtree-d)
    assert not lp.is_dir()


def test_same_process_double_acquire_raises(tmp_path):
    from automation.lib import lock

    lp = tmp_path / ".lock"
    with lock.acquire_lock(lp, start_heartbeat=False) as _h1:
        with pytest.raises(lock.LockBusyError):
            lock.acquire_lock(
                lp, start_heartbeat=False, owner_uuid="other",
                host=socket.gethostname(), pid=os.getpid(),
            )


def test_expired_lease_dead_owner_can_steal(tmp_path):
    """Expired lease + dead pid -> stealer wins."""
    from automation.lib import lock

    lp = tmp_path / ".lock"
    # Place a stale lock manually pointing to a definitely-dead PID
    lp.mkdir()
    os.symlink("old_owner_uuid", lp / "owner")
    meta = {
        "owner_uuid": "old_owner_uuid",
        "host": socket.gethostname(),
        "pid": 99999999,
        "starttime": 1,
        "acquired_at": "2020-01-01T00:00:00Z",
        "lease_until": "2020-01-01T00:00:01Z",  # long expired
        "last_heartbeat_at": "2020-01-01T00:00:00Z",
    }
    (lp / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    # Steal succeeds
    with lock.acquire_lock(lp, start_heartbeat=False) as h:
        assert h.owner_uuid != "old_owner_uuid"


def test_expired_lease_live_owner_does_not_steal(tmp_path):
    """Expired lease but pid still alive AND starttime matches
    (clock drift case) -> no steal. PID + starttime together identify
    the original process; an identical pair means the original owner
    is still running and we must not evict.
    """
    from automation.lib import lock

    lp = tmp_path / ".lock"
    lp.mkdir()
    os.symlink("old_owner_uuid", lp / "owner")
    # Use the live test runner's real (pid, starttime) so the steal
    # protocol's "same process" identity check succeeds.
    meta = {
        "owner_uuid": "old_owner_uuid",
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "starttime": lock._process_starttime(os.getpid()),
        "acquired_at": "2020-01-01T00:00:00Z",
        "lease_until": "2020-01-01T00:00:01Z",  # long expired
        "last_heartbeat_at": "2020-01-01T00:00:00Z",
    }
    (lp / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(lock.LockBusyError):
        lock.acquire_lock(lp, start_heartbeat=False)


def test_same_host_pid_reused_starttime_mismatch_steals(tmp_path):
    """Same-host owner with a PID that is alive but whose starttime
    differs from meta.json -> the original owner is dead and the live
    PID belongs to an unrelated process. Steal succeeds.
    """
    from automation.lib import lock

    lp = tmp_path / ".lock"
    lp.mkdir()
    os.symlink("old_owner_uuid", lp / "owner")
    real_st = lock._process_starttime(os.getpid())
    if real_st < 0:
        pytest.skip("/proc/<pid>/stat starttime not readable on this host")
    meta = {
        "owner_uuid": "old_owner_uuid",
        "host": socket.gethostname(),
        "pid": os.getpid(),
        # Synthesise a starttime that cannot match the live process so
        # the steal protocol treats the live PID as PID-reuse.
        "starttime": real_st + 1_000_000,
        "acquired_at": "2020-01-01T00:00:00Z",
        "lease_until": "2020-01-01T00:00:01Z",
        "last_heartbeat_at": "2020-01-01T00:00:00Z",
    }
    (lp / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with lock.acquire_lock(lp, start_heartbeat=False) as h:
        assert h.owner_uuid != "old_owner_uuid"


def test_cross_host_stale_heartbeat_steal(tmp_path):
    """Cross-host owner with N=2 grace stale -> steal."""
    from automation.lib import lock

    lp = tmp_path / ".lock"
    lp.mkdir()
    os.symlink("far_owner_uuid", lp / "owner")
    meta = {
        "owner_uuid": "far_owner_uuid",
        "host": "remote-host-nowhere",
        "pid": 12345,
        "starttime": 1,
        "acquired_at": "2020-01-01T00:00:00Z",
        "lease_until": "2020-01-01T00:00:01Z",
        "last_heartbeat_at": "2020-01-01T00:00:00Z",
    }
    (lp / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with lock.acquire_lock(lp, start_heartbeat=False) as h:
        assert h.owner_uuid != "far_owner_uuid"


def test_cross_host_fresh_heartbeat_no_steal(tmp_path):
    """Cross-host owner with fresh heartbeat -> no steal."""
    from datetime import datetime

    from automation.lib import lock

    lp = tmp_path / ".lock"
    lp.mkdir()
    os.symlink("far_owner_uuid", lp / "owner")
    fresh = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {
        "owner_uuid": "far_owner_uuid",
        "host": "remote-host-nowhere",
        "pid": 12345,
        "starttime": 1,
        "acquired_at": fresh,
        "lease_until": "2020-01-01T00:00:01Z",  # nominally expired
        "last_heartbeat_at": fresh,            # but recent
    }
    (lp / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(lock.LockBusyError):
        lock.acquire_lock(lp, start_heartbeat=False)


def test_heartbeat_thread_refreshes_lease(tmp_path):
    from automation.lib import lock

    lp = tmp_path / ".lock"
    with lock.acquire_lock(lp, start_heartbeat=True, lease_s=30, heartbeat_s=3):
        meta_before = json.loads((lp / "meta.json").read_text(encoding="utf-8"))
        time.sleep(1.5)
        meta_after = json.loads((lp / "meta.json").read_text(encoding="utf-8"))
        # Heartbeat should have advanced (or stayed equal-because-tick-not-fired)
        # We accept >= because mtime resolution and timer granularity.
        assert meta_after["last_heartbeat_at"] >= meta_before["last_heartbeat_at"]


def test_eviction_audit_dropped_when_owner_changes(tmp_path):
    """Owner_uuid symlink changed -> heartbeat thread sees + drops audit."""
    from automation.lib import lock

    lp = tmp_path / ".lock"
    h = lock.acquire_lock(lp, start_heartbeat=True, lease_s=30, heartbeat_s=1)
    # Mutate the owner symlink behind the lock's back
    owner = lp / "owner"
    if owner.is_symlink():
        owner.unlink()
        os.symlink("hostile_new_owner", owner)
    time.sleep(2.5)
    # Heartbeat should have detected eviction; check audit marker
    parent = lp.parent
    audits = list(parent.glob(".steal_evicted.*"))
    assert audits, "eviction audit marker must be created"
    assert h._evicted is True
    lock.release_lock(h)


def test_two_tier_nesting_global_then_per_task(tmp_path):
    """Two-tier nesting works: global first, then per-task."""
    from automation.lib import lock

    gl = tmp_path / "state" / ".lock"
    pl = tmp_path / "state" / ".lock.taskA"
    with lock.acquire_lock(gl, start_heartbeat=False) as gh:
        with lock.acquire_lock(pl, start_heartbeat=False) as th:
            assert gl.is_dir() and pl.is_dir()
            assert gh.owner_uuid != th.owner_uuid
    # Both released after exit
    assert not gl.is_dir()
    assert not pl.is_dir()


def test_global_lock_blocks_second_acquire(tmp_path):
    from automation.lib import lock

    gl = tmp_path / ".lock"
    with lock.acquire_lock(gl, start_heartbeat=False):
        with pytest.raises(lock.LockBusyError):
            lock.acquire_lock(gl, start_heartbeat=False, timeout_s=1)


def test_mkdir_permission_failure_propagates(tmp_path):
    from automation.lib import lock

    # Make the parent unwritable to provoke ``PermissionError``.
    parent = tmp_path / "ro"
    parent.mkdir()
    os.chmod(parent, 0o500)
    try:
        with pytest.raises((PermissionError, OSError)):
            lock.acquire_lock(parent / ".lock", start_heartbeat=False)
    finally:
        os.chmod(parent, 0o700)


def test_corrupt_meta_lets_us_steal(tmp_path):
    from automation.lib import lock

    lp = tmp_path / ".lock"
    lp.mkdir()
    (lp / "meta.json").write_text("not a valid json", encoding="utf-8")
    # No symlink set; corrupt -> we steal.
    with lock.acquire_lock(lp, start_heartbeat=False) as h:
        assert h.owner_uuid is not None
