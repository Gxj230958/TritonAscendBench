"""Directory-rename queue with 5-case mv state machine (Unit 1.2).

State markers
-------------
Each ``<run_id>`` lives under ``<queue_root>/`` and uses marker files to
track its state:

* ``<run_id>/task.json``        -- serialized work item payload.
* ``<run_id>/READY``            -- enqueue-time marker.
* ``<run_id>/CLAIMED.<host>.<pid>.<starttime>.<ts>``
                                -- atomic claim marker, the rename of
                                   ``READY`` to this filename IS the
                                   claim operation.
* ``<run_id>/CLAIMED.<...>.heartbeat`` (touched by worker thread)
* ``<run_id>/DONE``             -- success terminal.
* ``<run_id>/QUARANTINED.<reason>.<ts>`` -- bad-state terminal.
* ``<run_id>/STOLEN_BY_REAPER.<host>.<pid>.<ts>``
                                -- audit trail when reaper steals a
                                   stuck claim from a different worker.

5-case mv matrix (atomic claim semantics)
-----------------------------------------
On ``os.rename(ready_path, claim_path)``:

  case A -- rename rc=0, ready missing, claim present  -> we won.
  case B -- FileNotFoundError or ready missing         -> other worker won.
  case C -- OSError + ready present + claim absent     -> real I/O error.
  case D -- ready present + claim present              -> race split,
                                                          quarantine.
  case E -- any other exception                        -> fail loud.

A worker about to ``complete()`` re-derives ``<host>.<pid>.<starttime>``
from its claim_marker filename and verifies it still matches; if it
doesn't, the reaper has reclaimed and the worker writes a
``STOLEN_BY_REAPER.*`` audit marker + raises.
"""
from __future__ import annotations

import atexit
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from automation.lib import atomic

logger = logging.getLogger("automation.lib.queue")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso_utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_ts() -> str:
    """Filesystem-safe UTC timestamp suffix used in marker filenames."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ%f")


def _process_starttime(pid: int) -> int:
    """Read ``/proc/<pid>/stat`` field 22 (process start time in jiffies).

    Used together with PID to defeat PID reuse on Linux (32k PID cap).
    Returns -1 on any platform that lacks ``/proc`` (tests should mock).
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            raw = f.read()
        # Field 2 is "(comm)" possibly with spaces -> find closing ')'
        close = raw.rfind(b")")
        rest = raw[close + 2:].split()
        # After ")" the next field is state (idx 0 in `rest`); field 22
        # of the original /proc/<pid>/stat is `rest[19]` (= start time
        # in jiffies since boot).
        return int(rest[19])
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError, OSError):
        return -1


def _pid_alive(pid: int) -> bool:
    """``kill -0 <pid>`` -- same-host liveness check."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Different uid but pid exists.
        return True
    except OSError:
        return False


def _parse_claim_filename(name: str) -> tuple[str, int, int, str] | None:
    """Parse ``CLAIMED.<host>.<pid>.<starttime>.<ts>`` into a tuple.

    Returns ``(host, pid, starttime, ts)`` or None if name does not
    match the expected schema.
    """
    if not name.startswith("CLAIMED."):
        return None
    parts = name.split(".")
    # Expect at least 5 parts: CLAIMED, host, pid, starttime, ts(+possible suffix)
    if len(parts) < 5:
        return None
    try:
        host = parts[1]
        pid = int(parts[2])
        starttime = int(parts[3])
        ts = ".".join(parts[4:])
        return host, pid, starttime, ts
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ClaimHandle:
    """Returned by ``claim()``; holds enough state to ``complete()``
    later and identifies the heartbeat thread that should stop on
    close."""
    run_id: str
    queue_root: Path
    claim_marker: Path
    host: str
    pid: int
    starttime: int
    heartbeat_thread: threading.Thread | None = None
    heartbeat_stop: threading.Event | None = None

    @property
    def run_dir(self) -> Path:
        return self.queue_root / self.run_id


class QueueError(RuntimeError):
    """Base exception for queue-state violations."""


class QueueCorruptError(QueueError):
    """Raised when an enqueue collides with a terminal-state marker
    (``QUARANTINED.*`` or ``DONE``) that should have prevented re-use of
    the run_id."""


# ---------------------------------------------------------------------------
# Enqueue
# ---------------------------------------------------------------------------

def enqueue(
    queue_root: os.PathLike[str] | str,
    run_id: str,
    task_json: dict[str, Any],
) -> Path:
    """Create ``<queue_root>/<run_id>/{task.json, READY}`` atomically.

    Raises ``QueueCorruptError`` if a terminal marker already exists for
    this ``run_id`` (re-enqueue of a quarantined / done id is a bug;
    callers should mint a fresh ``run_id``).
    """
    queue_root = Path(queue_root)
    run_dir = queue_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Fail-loud on terminal markers
    for terminal_glob in ("QUARANTINED.*", "DONE", "STOLEN_BY_REAPER.*"):
        if any(run_dir.glob(terminal_glob)):
            raise QueueCorruptError(
                f"refusing to enqueue {run_id!r}: terminal marker present "
                f"(glob={terminal_glob}); mint a fresh run_id"
            )

    atomic.atomic_write_json(run_dir / "task.json", task_json)
    ready = run_dir / "READY"
    # Use an atomic create marker -- ``touch`` semantics; we want the
    # READY marker to appear in one shot so scanners cannot read a
    # partial file.
    with open(ready, "w", encoding="utf-8") as f:
        f.write(_iso_utc_now() + "\n")
    return ready


# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------

def claim(
    queue_root: os.PathLike[str] | str,
    host: str | None = None,
    pid: int | None = None,
    starttime: int | None = None,
    *,
    start_heartbeat: bool = True,
    heartbeat_interval_s: int = 60,
) -> ClaimHandle | None:
    """Try to claim one READY run_id atomically.

    Scans ``<queue_root>/*/READY`` and attempts to rename each to
    ``CLAIMED.<host>.<pid>.<starttime>.<ts>``. Returns a ``ClaimHandle``
    on success, ``None`` if no READY entries.
    """
    queue_root = Path(queue_root)
    host = host or socket.gethostname()
    pid = pid or os.getpid()
    if starttime is None:
        starttime = _process_starttime(pid)

    ready_markers = sorted(queue_root.glob("*/READY"))
    for ready in ready_markers:
        run_dir = ready.parent
        ts = _now_ts()
        claim_name = f"CLAIMED.{host}.{pid}.{starttime}.{ts}"
        claim_path = run_dir / claim_name
        try:
            os.rename(ready, claim_path)
        except FileNotFoundError:
            # case B -- other worker grabbed it; try next.
            continue
        except OSError:
            ready_exists = ready.exists()
            claim_exists = claim_path.exists()
            if ready_exists and not claim_exists:
                # case C -- real I/O error.
                raise
            if ready_exists and claim_exists:
                # case D -- split state.
                quarantine(queue_root, run_dir.name, reason="race")
                raise QueueCorruptError(
                    f"claim race: ready+claim both present for {run_dir.name!r}"
                ) from None
            # case B (raced and lost)
            continue
        # Verify case A invariant
        if ready.exists() or not claim_path.exists():
            if claim_path.exists() and ready.exists():
                quarantine(queue_root, run_dir.name, reason="race")
                raise QueueCorruptError(
                    f"claim race: ready+claim both present for {run_dir.name!r}"
                )
            # Lost the race silently
            continue

        # Post-claim integrity check: task.json MUST exist and parse.
        # An orphan READY (no task.json beside it, or a corrupt one)
        # would otherwise drive the worker into an unrecoverable state.
        # Quarantine on failure so reaper / operator can see it.
        task_json_path = run_dir / "task.json"
        try:
            text = task_json_path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError) as exc:
            quarantine(queue_root, run_dir.name, reason="task_json_missing")
            raise QueueCorruptError(
                f"claim integrity: task.json missing for {run_dir.name!r}: {exc}"
            ) from exc
        try:
            import json as _j
            _j.loads(text)
        except (ValueError, TypeError) as exc:
            quarantine(queue_root, run_dir.name, reason="task_json_corrupt")
            raise QueueCorruptError(
                f"claim integrity: task.json unparseable for "
                f"{run_dir.name!r}: {exc}"
            ) from exc

        # Start heartbeat thread
        stop_event = None
        thread = None
        if start_heartbeat:
            stop_event = threading.Event()
            thread = _start_heartbeat_thread(
                claim_path, stop_event, interval_s=heartbeat_interval_s,
            )
        return ClaimHandle(
            run_id=run_dir.name,
            queue_root=queue_root,
            claim_marker=claim_path,
            host=host,
            pid=pid,
            starttime=starttime,
            heartbeat_thread=thread,
            heartbeat_stop=stop_event,
        )
    return None


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

def _start_heartbeat_thread(
    claim_marker: Path, stop_event: threading.Event, *, interval_s: int,
) -> threading.Thread:
    heartbeat_path = claim_marker.parent / (claim_marker.name + ".heartbeat")
    heartbeat_path.touch()

    def _loop() -> None:
        while not stop_event.wait(interval_s):
            try:
                # touch -- update mtime
                os.utime(heartbeat_path, None)
            except FileNotFoundError:
                # Claim marker gone (e.g. reaper rolled back); stop.
                return
            except OSError as exc:  # pragma: no cover -- defensive
                logger.warning("heartbeat failed: %s", exc)
                return

    t = threading.Thread(
        target=_loop,
        name=f"queue-heartbeat-{claim_marker.name}",
        daemon=True,
    )
    t.start()
    # Make sure the thread stops on interpreter exit.
    atexit.register(stop_event.set)
    return t


# ---------------------------------------------------------------------------
# Complete
# ---------------------------------------------------------------------------

def complete(handle: ClaimHandle) -> Path:
    """Verify claim is still ours and atomically rename to ``DONE``.

    If the claim_marker filename no longer encodes our
    ``(host, pid, starttime)``, the reaper has stolen the lock: we
    audit-log via ``STOLEN_BY_REAPER.*`` and raise ``QueueCorruptError``
    so the caller can drop its result.
    """
    if handle.heartbeat_stop is not None:
        handle.heartbeat_stop.set()

    if not handle.claim_marker.exists():
        _audit_stolen(handle)
        raise QueueCorruptError(
            f"claim marker missing for {handle.run_id!r} -- reaper stole?"
        )

    parsed = _parse_claim_filename(handle.claim_marker.name)
    if parsed is None:
        raise QueueCorruptError(
            f"unparseable claim marker {handle.claim_marker.name!r}"
        )
    host, pid, starttime, _ts = parsed
    if (host, pid, starttime) != (handle.host, handle.pid, handle.starttime):
        _audit_stolen(handle)
        raise QueueCorruptError(
            f"claim marker reassigned for {handle.run_id!r}"
        )

    done_path = handle.run_dir / "DONE"
    os.replace(handle.claim_marker, done_path)
    # Clean up the heartbeat marker (best effort)
    hb = handle.claim_marker.parent / (handle.claim_marker.name + ".heartbeat")
    try:
        if hb.exists():
            hb.unlink()
    except OSError:
        pass
    return done_path


def _audit_stolen(handle: ClaimHandle) -> None:
    audit = handle.run_dir / (
        f"STOLEN_BY_REAPER.{handle.host}.{handle.pid}.{_now_ts()}"
    )
    try:
        audit.write_text(_iso_utc_now() + "\n", encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------

def quarantine(
    queue_root: os.PathLike[str] | str, run_id: str, reason: str,
) -> Path:
    """Drop a ``QUARANTINED.<reason>.<ts>`` terminal marker on a run."""
    queue_root = Path(queue_root)
    run_dir = queue_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    safe_reason = reason.replace(".", "_").replace("/", "_")
    marker = run_dir / f"QUARANTINED.{safe_reason}.{_now_ts()}"
    marker.write_text(_iso_utc_now() + "\n", encoding="utf-8")
    return marker


# ---------------------------------------------------------------------------
# Reaper
# ---------------------------------------------------------------------------

def reaper(
    queue_root: os.PathLike[str] | str,
    *,
    heartbeat_grace_s: int = 600,
    cross_host_consecutive: int = 2,
) -> list[Path]:
    """Scan ``CLAIMED.*`` markers; reclaim stuck claims.

    A claim is stuck when:

      * heartbeat mtime is older than ``heartbeat_grace_s * 2``, AND
      * (same host: ``kill -0 pid`` returns False)
        OR (cross host: heartbeat has been stale for at least
        ``cross_host_consecutive`` consecutive sweeps, indicated by
        ``heartbeat_grace_s * 2 * cross_host_consecutive`` stale).

    Reclaim = rename ``CLAIMED.*`` back to ``READY`` (atomic).
    Returns list of reclaimed run_dirs.
    """
    queue_root = Path(queue_root)
    reclaimed: list[Path] = []
    own_host = socket.gethostname()
    now = time.time()
    same_host_thresh = heartbeat_grace_s * 2
    cross_host_thresh = heartbeat_grace_s * 2 * cross_host_consecutive

    for run_dir in sorted(p for p in queue_root.iterdir() if p.is_dir()):
        if any(run_dir.glob("DONE")):
            continue
        if any(run_dir.glob("QUARANTINED.*")):
            continue
        claim_markers = list(run_dir.glob("CLAIMED.*"))
        # exclude heartbeat markers
        claim_markers = [m for m in claim_markers if not m.name.endswith(".heartbeat")]
        if not claim_markers:
            continue
        if len(claim_markers) > 1:
            # Multiple CLAIMED.* on the same run -- always a split state.
            quarantine(queue_root, run_dir.name, reason="multi_claim")
            continue
        marker = claim_markers[0]
        parsed = _parse_claim_filename(marker.name)
        if parsed is None:
            quarantine(queue_root, run_dir.name, reason="bad_claim_name")
            continue
        host, pid, claim_starttime, _ts = parsed
        hb_path = run_dir / (marker.name + ".heartbeat")
        if hb_path.exists():
            hb_age = now - hb_path.stat().st_mtime
        else:
            hb_age = now - marker.stat().st_mtime
        if host == own_host:
            # PID + starttime defeats PID reuse: if /proc says the PID
            # is alive but its starttime no longer matches the value
            # baked into the claim filename, the original worker is
            # dead and the live PID is an unrelated process. Treat as
            # dead and reclaim.
            cur_starttime = _process_starttime(pid)
            owner_alive = (
                _pid_alive(pid)
                and (
                    cur_starttime < 0
                    or claim_starttime < 0
                    or cur_starttime == claim_starttime
                )
            )
            if hb_age >= same_host_thresh and not owner_alive:
                ready = run_dir / "READY"
                try:
                    os.rename(marker, ready)
                except FileExistsError:
                    quarantine(queue_root, run_dir.name, reason="reaper_split")
                    continue
                if hb_path.exists():
                    try:
                        hb_path.unlink()
                    except OSError:
                        pass
                reclaimed.append(run_dir)
        else:
            # Cross-host: we cannot kill -0, rely on consecutive grace
            if hb_age >= cross_host_thresh:
                ready = run_dir / "READY"
                try:
                    os.rename(marker, ready)
                except FileExistsError:
                    quarantine(queue_root, run_dir.name, reason="reaper_split")
                    continue
                if hb_path.exists():
                    try:
                        hb_path.unlink()
                    except OSError:
                        pass
                reclaimed.append(run_dir)
    return reclaimed


__all__ = [
    "ClaimHandle",
    "QueueCorruptError",
    "QueueError",
    "claim",
    "complete",
    "enqueue",
    "quarantine",
    "reaper",
]
