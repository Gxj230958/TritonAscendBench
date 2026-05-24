"""mkdir-based atomic lock with symlink owner, lease, heartbeat, and
three-stage steal protocol (Unit 1.3).

Two lock tiers (caller's responsibility to enforce ordering):

* Global lock      ``state/.lock``                 -- triage_state.json SoT
* Per-task lock    ``state/.lock.<task_id>``       -- per-task isolation

The lib does NOT enforce nesting in its signature; orchestrator wraps
its writes with ``with acquire_lock(global) as gh: with acquire_lock(per_task) as th: ...``
(LIFO release via contextmanager ``__exit__``).

Three-stage steal (when ``os.mkdir`` collides):
  1. read ``meta.json`` -- if ``lease_until`` not expired, owner is
     still legitimately running -> caller raises (no steal).
  2. liveness check:
       - same host: ``os.kill(old_pid, 0)`` -> alive? abort steal.
       - cross host: ``last_heartbeat_at`` stale >= 2 * heartbeat_s?
                     proceed; else abort.
  3. atomic ownership transfer: ``os.rename(lock_path, .lock.stale.<old_uuid>)``
     then ``rmtree`` + fresh ``mkdir`` + new symlink/meta.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import signal
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("automation.lib.lock")

DEFAULT_LEASE_S = 900
DEFAULT_HEARTBEAT_S = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso_utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_ts_filename() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ%f")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# LockHandle
# ---------------------------------------------------------------------------

@dataclass
class LockHandle:
    lock_path: Path
    owner_uuid: str
    host: str
    pid: int
    lease_s: int
    heartbeat_s: int
    acquired_at: str
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _released: bool = False
    _evicted: bool = False

    # context manager
    def __enter__(self) -> LockHandle:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._released:
            release_lock(self)


class LockBusyError(RuntimeError):
    """Raised when ``acquire_lock`` cannot get the lock (live owner)."""


class LockEvictedError(RuntimeError):
    """Raised by heartbeat thread when it detects ownership change."""


# ---------------------------------------------------------------------------
# Acquire
# ---------------------------------------------------------------------------

def acquire_lock(
    lock_path: os.PathLike[str] | str,
    *,
    owner_uuid: str | None = None,
    host: str | None = None,
    pid: int | None = None,
    lease_s: int = DEFAULT_LEASE_S,
    heartbeat_s: int = DEFAULT_HEARTBEAT_S,
    start_heartbeat: bool = True,
    steal_consecutive: int = 2,
    timeout_s: int | None = None,
) -> LockHandle:
    """Acquire ``lock_path``. Returns a ``LockHandle`` context manager.

    On collision, attempts the three-stage steal protocol. If the owner
    is alive (kill -0 passes / heartbeat fresh), raises ``LockBusyError``.

    ``timeout_s``: if set, retries every 1s until acquire or timeout
    (used by tests that don't want to block forever).
    """
    lock_path = Path(lock_path)
    owner_uuid = owner_uuid or uuid.uuid4().hex
    host = host or socket.gethostname()
    pid = pid or os.getpid()

    deadline = time.time() + timeout_s if timeout_s else None
    while True:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            os.mkdir(lock_path)
        except FileExistsError:
            # Attempt steal
            if _try_steal(lock_path, owner_uuid, host, pid,
                          heartbeat_s=heartbeat_s,
                          steal_consecutive=steal_consecutive):
                # Steal succeeded -- the directory is fresh now; loop
                # to write our meta/owner.
                continue
            # Cannot steal -- live owner.
            if deadline is None:
                raise LockBusyError(
                    f"lock busy: {lock_path}; owner is live (no steal)"
                ) from None
            if time.time() >= deadline:
                raise LockBusyError(
                    f"lock busy: {lock_path}; timeout after {timeout_s}s"
                ) from None
            time.sleep(1.0)
            continue
        except OSError:
            raise
        # mkdir succeeded -- write owner symlink + meta.json
        try:
            owner_link = lock_path / "owner"
            try:
                os.symlink(owner_uuid, owner_link)
            except OSError:  # pragma: no cover -- symlink unsupported FS
                # Fallback: write owner uuid to a plain file
                (lock_path / "owner.txt").write_text(owner_uuid)
            now = _iso_utc_now()
            meta = {
                "owner_uuid": owner_uuid,
                "host": host,
                "pid": pid,
                "starttime": _process_starttime(pid),
                "acquired_at": now,
                "lease_until": _iso_offset(time.time() + lease_s),
                "last_heartbeat_at": now,
            }
            (lock_path / "meta.json").write_text(
                json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8",
            )
        except Exception:
            shutil.rmtree(lock_path, ignore_errors=True)
            raise

        handle = LockHandle(
            lock_path=lock_path,
            owner_uuid=owner_uuid,
            host=host,
            pid=pid,
            lease_s=lease_s,
            heartbeat_s=heartbeat_s,
            acquired_at=meta["acquired_at"],
        )
        if start_heartbeat:
            _start_heartbeat(handle)
        # atexit + SIGTERM cleanup
        _register_cleanup(handle)
        return handle


def _process_starttime(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            raw = f.read()
        close = raw.rfind(b")")
        rest = raw[close + 2:].split()
        return int(rest[19])
    except Exception:
        return -1


def _iso_offset(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s, tz=UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _parse_iso(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=UTC
    ).timestamp()


# ---------------------------------------------------------------------------
# Steal
# ---------------------------------------------------------------------------

def _try_steal(
    lock_path: Path,
    my_uuid: str,
    my_host: str,
    my_pid: int,
    *,
    heartbeat_s: int,
    steal_consecutive: int,
) -> bool:
    """Three-stage steal. Returns True if lock_path is now free for the
    caller to ``os.mkdir`` again. Returns False if the owner is live."""
    meta_path = lock_path / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        # Stale lock-dir with no meta -> safe to reclaim (corrupted state)
        return _atomic_evict(lock_path, "no_meta")

    # Stage 1: lease expiry check
    try:
        lease_until = _parse_iso(meta.get("lease_until", ""))
    except (ValueError, TypeError):
        lease_until = 0
    if time.time() < lease_until:
        # Lease still valid; we don't steal yet.
        return False

    # Stage 2: liveness check
    owner_host = meta.get("host")
    owner_pid = meta.get("pid", -1)
    if owner_host == my_host:
        # PID + ``/proc/<pid>/stat`` starttime defeats PID reuse. Linux
        # recycles PIDs (32k cap) so ``kill -0`` alone can match a new,
        # unrelated process that inherited the dead owner's pid. If
        # starttime in /proc differs from the value persisted in
        # meta.json at lock-acquire time, the original owner is dead
        # and we can safely steal -- the live process answering kill -0
        # is somebody else.
        try:
            owner_pid_int = int(owner_pid)
        except (TypeError, ValueError):
            owner_pid_int = -1
        meta_starttime = meta.get("starttime", -1)
        try:
            meta_starttime_int = int(meta_starttime)
        except (TypeError, ValueError):
            meta_starttime_int = -1
        cur_starttime = _process_starttime(owner_pid_int)
        if _pid_alive(owner_pid_int):
            # PID alive, but verify it is the SAME process as recorded.
            # Both must be >= 0 (an unknown starttime means we can't
            # prove identity either way; fall back to old behaviour and
            # treat as live).
            if (
                meta_starttime_int < 0
                or cur_starttime < 0
                or cur_starttime == meta_starttime_int
            ):
                return False
            # PID alive but starttime differs: original owner is dead,
            # the live PID is a re-used unrelated process. Continue to
            # stage 3 steal.
    else:
        # Cross-host -- rely on heartbeat staleness
        try:
            last_hb = _parse_iso(meta.get("last_heartbeat_at", ""))
        except (ValueError, TypeError):
            last_hb = 0
        threshold = heartbeat_s * steal_consecutive
        if time.time() - last_hb < threshold:
            return False

    # Stage 3: atomic ownership transfer
    return _atomic_evict(lock_path, meta.get("owner_uuid", "unknown"))


def _atomic_evict(lock_path: Path, old_owner_uuid: str) -> bool:
    """Rename lock_path -> .lock.stale.<old_uuid>.<ts>, then rmtree."""
    parent = lock_path.parent
    stale = parent / f".lock.stale.{old_owner_uuid}.{_now_ts_filename()}"
    try:
        os.rename(lock_path, stale)
    except FileNotFoundError:
        # Another stealer beat us to it; safe to retry mkdir.
        return True
    except OSError:
        return False
    # Best-effort cleanup of the stale dir; keep it briefly as audit.
    try:
        shutil.rmtree(stale)
    except OSError:
        pass
    return True


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

def _start_heartbeat(handle: LockHandle) -> None:
    interval = max(1, handle.heartbeat_s // 3)

    def _loop() -> None:
        while not handle._stop_event.wait(interval):
            try:
                # Verify ownership: read owner symlink
                owner_link = handle.lock_path / "owner"
                current_owner: str | None = None
                if owner_link.is_symlink():
                    current_owner = os.readlink(owner_link)
                else:
                    txt = handle.lock_path / "owner.txt"
                    if txt.is_file():
                        current_owner = txt.read_text(encoding="utf-8").strip()
                if current_owner is not None and current_owner != handle.owner_uuid:
                    # Stolen out from under us -- stop touching, drop
                    # audit marker, flag eviction.
                    handle._evicted = True
                    _drop_eviction_audit(handle, current_owner)
                    return
                # Refresh meta.json with new heartbeat + lease
                meta_path = handle.lock_path / "meta.json"
                if not meta_path.is_file():
                    handle._evicted = True
                    return
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    handle._evicted = True
                    return
                meta["last_heartbeat_at"] = _iso_utc_now()
                meta["lease_until"] = _iso_offset(time.time() + handle.lease_s)
                meta_path.write_text(
                    json.dumps(meta, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            except FileNotFoundError:
                handle._evicted = True
                return
            except OSError as exc:  # pragma: no cover
                logger.warning("heartbeat failed: %s", exc)
                return

    t = threading.Thread(
        target=_loop,
        name=f"lock-heartbeat-{handle.owner_uuid[:8]}",
        daemon=True,
    )
    t.start()
    handle._thread = t


def _drop_eviction_audit(handle: LockHandle, new_owner_uuid: str) -> None:
    parent = handle.lock_path.parent
    audit = parent / (
        f".steal_evicted.{handle.owner_uuid}.{new_owner_uuid}.{_now_ts_filename()}"
    )
    try:
        audit.write_text(_iso_utc_now() + "\n", encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------

def release_lock(handle: LockHandle) -> None:
    if handle._released:
        return
    handle._released = True
    handle._stop_event.set()
    parent = handle.lock_path.parent
    released_marker = parent / (
        f".lock.released.{handle.owner_uuid}.{_now_ts_filename()}"
    )
    try:
        os.rename(handle.lock_path, released_marker)
    except FileNotFoundError:
        # Already gone (stolen?) -- not fatal.
        return
    except OSError as exc:
        logger.warning("release_lock rename failed: %s", exc)
        return
    try:
        shutil.rmtree(released_marker)
    except OSError:
        pass


_REGISTERED: list[LockHandle] = []
_SIGTERM_HANDLER_INSTALLED = False


def _register_cleanup(handle: LockHandle) -> None:
    global _SIGTERM_HANDLER_INSTALLED
    _REGISTERED.append(handle)

    def _atexit_cleanup() -> None:
        try:
            release_lock(handle)
        except Exception:  # pragma: no cover
            pass
    atexit.register(_atexit_cleanup)

    if not _SIGTERM_HANDLER_INSTALLED:
        # Independent SIGTERM handler -- atexit alone does not fire on
        # SIGTERM, so a parallel signal hook is required.
        previous = signal.getsignal(signal.SIGTERM)

        def _sigterm(signo, frame):
            for h in list(_REGISTERED):
                try:
                    release_lock(h)
                except Exception:  # pragma: no cover
                    pass
            if callable(previous):
                try:
                    previous(signo, frame)
                except Exception:  # pragma: no cover
                    pass

        try:
            signal.signal(signal.SIGTERM, _sigterm)
            _SIGTERM_HANDLER_INSTALLED = True
        except (ValueError, OSError):
            # Not in main thread; ignore.
            pass


__all__ = [
    "DEFAULT_HEARTBEAT_S",
    "DEFAULT_LEASE_S",
    "LockBusyError",
    "LockEvictedError",
    "LockHandle",
    "acquire_lock",
    "release_lock",
]
