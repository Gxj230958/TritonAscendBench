"""Bench evaluator daemon (Unit 4.1).

A long-running Python daemon that lives on the GPU machine. It polls a
GPFS-shared directory-rename queue (``<gpfs>/queue/READY/*``), claims one
``run_id`` at a time, runs the requested bench sub-command locally
(``scripts/verify/standalone_freeze_verifier.py --verify`` or
``scripts/verify/resolve_upstream_entrypoints.py``), and writes the result back
to ``<gpfs>/results/<run_id>/`` as the 9-field ``result.json`` +
``_provenance.json`` sidecar + ``DONE`` read-barrier marker.

CLI entry-point. Operator starts it via::

    nohup setsid python3 scripts/automation/bench_evaluator_daemon.py \\
        --gpfs-root /gpfs/.../tab_automation \\
        --bench-root /local/triton-ascend-bench \\
        > /gpfs/.../tab_automation/logs/daemon-<host>-<ts>.log 2>&1 &

The daemon is NOT shipped in the release wheel (``scripts/automation/``
is ``export-ignore`` in ``.gitattributes``).

Key invariants
--------------
* Atomic claim via ``automation.lib.queue.claim`` (directory rename).
* Per-daemon global mkdir-lock so two daemons cannot run on the same
  GPU concurrently.
* Result write order: ``_provenance.json`` -> ``result.json`` -> ``DONE``.
  ``DONE`` is the barrier; readers MUST wait for it before parsing.
* Strict 9-field ``result.json`` schema enforced via
  ``extract_verdict.validate_result_schema``.
* Graceful shutdown: ``SIGTERM`` / ``SIGINT`` let the current job finish
  before the main loop exits.
"""
from __future__ import annotations

# ---- KTD-11 sys.path bootstrap (CLI entry-point) -------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
# --------------------------------------------------------------------------

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Any  # noqa: E402

from automation import extract_verdict  # noqa: E402
from automation.lib import atomic, lock, queue  # noqa: E402

logger = logging.getLogger("automation.bench_evaluator_daemon")


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class DaemonContext:
    """Resolved CLI arguments + derived paths for a daemon instance."""

    gpfs_root: Path
    host_name: str
    bench_root: Path
    results_root_rel: str
    queue_root_rel: str
    poll_interval: float
    max_jobs: int
    lease_seconds: int
    heartbeat_seconds: int
    once: bool
    dry_run: bool

    @property
    def queue_root(self) -> Path:
        return self.gpfs_root / self.queue_root_rel

    @property
    def results_root(self) -> Path:
        return self.gpfs_root / self.results_root_rel

    @property
    def daemon_lock_path(self) -> Path:
        return self.gpfs_root / "state" / f".lock.daemon.{self.host_name}"


@dataclass
class JobClaim:
    """A claimed work item ready for execution."""

    task_id: str
    run_id: str
    cmd: str
    claim_handle: queue.ClaimHandle
    task_json_path: Path


@dataclass
class JobResult:
    """Outcome of running a single bench sub-command."""

    verdict: str
    verdict_class: str | None
    reason: str
    raw_subclass: str = ""
    status: str = "completed"  # completed | timeout | error | mock
    error: str | None = None
    stdout_log: Path | None = None
    duration_ms: int = 0
    started_at: str = ""
    finished_at: str = ""
    extra_provenance: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_CMDS = ("resolve", "phase_a_verify")


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_task_json(task_json_path: Path) -> dict[str, Any]:
    text = task_json_path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"task.json is not a JSON object: {task_json_path}")
    return data


# ---------------------------------------------------------------------------
# BenchEvaluatorDaemon
# ---------------------------------------------------------------------------

class BenchEvaluatorDaemon:
    """Daemon main loop + job execution.

    Constructor validates ``gpfs_root`` exists. Use ``run()`` to enter
    the main loop; ``claim_one_job`` / ``execute_job`` / ``write_result``
    are exposed for unit-test introspection.
    """

    def __init__(self, ctx: DaemonContext) -> None:
        if not ctx.gpfs_root.is_dir():
            raise FileNotFoundError(
                f"gpfs_root does not exist or is not a directory: {ctx.gpfs_root}"
            )
        if not ctx.bench_root.is_dir():
            raise FileNotFoundError(
                f"bench_root does not exist or is not a directory: {ctx.bench_root}"
            )
        self.ctx = ctx
        self.ctx.queue_root.mkdir(parents=True, exist_ok=True)
        self.ctx.results_root.mkdir(parents=True, exist_ok=True)
        (self.ctx.gpfs_root / "state").mkdir(parents=True, exist_ok=True)
        self._stop_requested = False
        self._jobs_done = 0
        self._install_signal_handlers()

    # -- Signal handling --------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Install SIGTERM / SIGINT handlers that request graceful stop.

        Each handler is registered independently; combining them into one
        ``signal.signal`` call would break the convention that SIGTERM
        must always trigger the stop path.
        """
        def _stop_handler(signo, frame):
            logger.info(
                "daemon received signal %d; finishing current job then exiting",
                signo,
            )
            self._stop_requested = True

        try:
            signal.signal(signal.SIGTERM, _stop_handler)
            signal.signal(signal.SIGINT, _stop_handler)
        except (ValueError, OSError):
            # Not in main thread (e.g. running inside a test) -- ignore.
            logger.debug("signal handler install skipped (not main thread)")

    def request_stop(self) -> None:
        """Programmatic stop request (used by tests / supervisor)."""
        self._stop_requested = True

    # -- Main loop --------------------------------------------------------

    def run(self) -> int:
        """Enter the main poll-claim-execute-write loop.

        Holds a global daemon lock under ``<gpfs>/state/.lock.daemon.<host>``
        to prevent two daemons on the same host from claiming the same
        queue. Exits when:

        * ``--max-jobs N`` reached, or
        * ``--once`` and at least one job has been attempted, or
        * SIGTERM / SIGINT received (after current job finishes).
        """
        logger.info(
            "bench_evaluator_daemon banner: host=%s pid=%d gpfs_root=%s "
            "queue_root=%s results_root=%s dry_run=%s",
            self.ctx.host_name, os.getpid(), self.ctx.gpfs_root,
            self.ctx.queue_root, self.ctx.results_root, self.ctx.dry_run,
        )

        try:
            with lock.acquire_lock(
                self.ctx.daemon_lock_path,
                host=self.ctx.host_name,
                lease_s=self.ctx.lease_seconds,
                heartbeat_s=self.ctx.heartbeat_seconds,
            ):
                self._main_loop()
        except lock.LockBusyError as exc:
            logger.error("daemon lock busy -- another daemon already running: %s", exc)
            return 2
        return 0

    def _main_loop(self) -> None:
        attempted = 0
        while not self._stop_requested:
            claim = self.claim_one_job()
            if claim is None:
                if self.ctx.once and attempted > 0:
                    logger.info("--once: no more READY jobs; exiting")
                    break
                if self.ctx.once:
                    # Nothing to do this iteration, but --once means try
                    # at most one full cycle.
                    logger.info("--once: queue empty; exiting")
                    break
                time.sleep(self.ctx.poll_interval)
                continue

            attempted += 1
            try:
                job_result = self.execute_job(claim)
                self.write_result(claim, job_result)
                self.complete_queue(claim)
                self._jobs_done += 1
            except Exception as exc:
                logger.exception(
                    "job execution failed for run_id=%s task_id=%s: %s",
                    claim.run_id, claim.task_id, exc,
                )
                self.quarantine_on_failure(claim, exc)

            if self.ctx.max_jobs > 0 and self._jobs_done >= self.ctx.max_jobs:
                logger.info(
                    "--max-jobs=%d reached; exiting", self.ctx.max_jobs,
                )
                break
            if self.ctx.once:
                logger.info("--once: finished one job; exiting")
                break

    # -- Claim ------------------------------------------------------------

    def claim_one_job(self) -> JobClaim | None:
        """Claim one READY work item from the queue, or return None."""
        handle = queue.claim(
            self.ctx.queue_root,
            host=self.ctx.host_name,
            heartbeat_interval_s=max(1, self.ctx.heartbeat_seconds),
        )
        if handle is None:
            return None

        task_json_path = handle.run_dir / "task.json"
        try:
            payload = _read_task_json(task_json_path)
        except (OSError, ValueError) as exc:
            logger.error(
                "task.json unreadable for run_id=%s: %s", handle.run_id, exc,
            )
            queue.quarantine(
                self.ctx.queue_root, handle.run_id, reason="task_json_read_fail",
            )
            raise

        cmd = payload.get("cmd")
        task_id = payload.get("task_id")
        if cmd not in _VALID_CMDS:
            queue.quarantine(
                self.ctx.queue_root, handle.run_id, reason="bad_cmd",
            )
            raise ValueError(
                f"task.json has invalid cmd={cmd!r}; expected one of {_VALID_CMDS}"
            )
        if not isinstance(task_id, str) or not task_id:
            queue.quarantine(
                self.ctx.queue_root, handle.run_id, reason="missing_task_id",
            )
            raise ValueError(f"task.json missing task_id for run_id={handle.run_id!r}")

        return JobClaim(
            task_id=task_id,
            run_id=handle.run_id,
            cmd=cmd,
            claim_handle=handle,
            task_json_path=task_json_path,
        )

    # -- Execute ----------------------------------------------------------

    def execute_job(self, claim: JobClaim) -> JobResult:
        """Run the bench sub-command for ``claim`` and parse a verdict.

        Sub-process timeout is ``lease_seconds * 0.8`` so the daemon has
        head-room to write the result + complete the queue transition
        before the queue-lib heartbeat lease expires.

        ``--dry-run`` short-circuits the subprocess and emits a synthetic
        ``verdict_class="unexpected-blocked"`` result tagged
        ``status="mock"``; the strict 9-field schema is still honoured.
        """
        started_at = _iso_utc_now()
        t0 = time.monotonic()

        if self.ctx.dry_run:
            duration_ms = int((time.monotonic() - t0) * 1000)
            return JobResult(
                verdict="BLOCKED",
                verdict_class="unexpected-blocked",
                reason="dry-run: subprocess not invoked",
                raw_subclass="dry_run",
                status="mock",
                stdout_log=None,
                duration_ms=duration_ms,
                started_at=started_at,
                finished_at=_iso_utc_now(),
                extra_provenance={"dry_run": True},
            )

        result_dir = self.ctx.results_root / claim.run_id
        result_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = result_dir / "log.txt"

        sub_cmd = self._build_subcommand(claim)
        timeout_s = max(1, int(self.ctx.lease_seconds * 0.8))
        timed_out = False
        exit_code: int | None = None
        try:
            with open(stdout_log, "w", encoding="utf-8") as log_fh:
                cp = subprocess.run(
                    sub_cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_s,
                    check=False,
                )
                exit_code = cp.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            with open(stdout_log, "a", encoding="utf-8") as log_fh:
                log_fh.write(f"\n*** TIMEOUT after {timeout_s}s\n")
        except FileNotFoundError as exc:
            with open(stdout_log, "a", encoding="utf-8") as log_fh:
                log_fh.write(f"\n*** sub-command not found: {exc}\n")
            exit_code = -1

        duration_ms = int((time.monotonic() - t0) * 1000)

        if timed_out:
            return JobResult(
                verdict="BLOCKED",
                verdict_class="unexpected-blocked",
                reason=f"daemon subprocess timeout after {timeout_s}s",
                raw_subclass="timeout",
                status="timeout",
                error=f"TimeoutExpired after {timeout_s}s",
                stdout_log=stdout_log,
                duration_ms=duration_ms,
                started_at=started_at,
                finished_at=_iso_utc_now(),
                extra_provenance={
                    "timed_out": True, "timeout_s": timeout_s,
                    "exit_code": None, "sub_cmd": sub_cmd,
                },
            )

        verdict, vclass, reason, raw_sub = self._parse_subcommand_verdict(
            claim, stdout_log,
        )
        return JobResult(
            verdict=verdict,
            verdict_class=vclass,
            reason=reason,
            raw_subclass=raw_sub,
            status="completed",
            error=None,
            stdout_log=stdout_log,
            duration_ms=duration_ms,
            started_at=started_at,
            finished_at=_iso_utc_now(),
            extra_provenance={
                "exit_code": exit_code, "sub_cmd": sub_cmd,
                "timeout_s": timeout_s,
            },
        )

    def _build_subcommand(self, claim: JobClaim) -> list[str]:
        if claim.cmd == "resolve":
            return [
                sys.executable,
                str(self.ctx.bench_root / "scripts" / "verify" / "resolve_upstream_entrypoints.py"),
                "--task", claim.task_id,
                "--apply",
            ]
        # phase_a_verify
        return [
            sys.executable,
            str(self.ctx.bench_root / "scripts" / "verify" / "standalone_freeze_verifier.py"),
            "--verify", claim.task_id,
        ]

    def _parse_subcommand_verdict(
        self, claim: JobClaim, stdout_log: Path,
    ) -> tuple[str, str | None, str, str]:
        if claim.cmd == "resolve":
            return extract_verdict.parse_resolve_log(stdout_log, claim.task_id)
        return extract_verdict.parse_phase_a_verify(
            self.ctx.bench_root, claim.task_id, stdout_log=stdout_log,
        )

    # -- Write result -----------------------------------------------------

    def write_result(self, claim: JobClaim, job_result: JobResult) -> Path:
        """Persist ``_provenance.json``, ``result.json``, and ``DONE``.

        Order matters: ``DONE`` is touched STRICTLY AFTER both JSON files
        are durably written (each via the M1 atomic_write_json fsync
        sequence). Readers must wait for ``DONE`` to appear.
        """
        run_dir = self.ctx.results_root / claim.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        extra = dict(job_result.extra_provenance)
        extra.setdefault("status", job_result.status)
        extra.setdefault("duration_ms", job_result.duration_ms)
        if job_result.error is not None:
            extra["error"] = job_result.error
        if job_result.stdout_log is not None:
            extra["stdout_log"] = str(job_result.stdout_log)

        # write_verdict() encapsulates the strict 9-field schema check,
        # provenance sidecar, and DONE barrier ordering.
        result_path = extract_verdict.write_verdict(
            task_id=claim.task_id,
            cmd=claim.cmd,
            run_id=claim.run_id,
            producer="daemon",
            out_dir=self.ctx.results_root,
            verdict=job_result.verdict,
            verdict_class=job_result.verdict_class,
            reason=job_result.reason,
            raw_subclass=job_result.raw_subclass,
            started_at=job_result.started_at,
            extra_provenance=extra,
        )

        # Verify DONE barrier ordering is intact (it is touched inside
        # write_verdict, but only after both JSON files exist).
        done = run_dir / "DONE"
        if not done.is_file():  # pragma: no cover -- defensive
            raise RuntimeError(
                f"DONE marker missing after write_verdict for {claim.run_id!r}"
            )
        return result_path

    # -- Complete / quarantine -------------------------------------------

    def complete_queue(self, claim: JobClaim) -> Path:
        """Move CLAIMED.* marker -> DONE in the queue dir.

        Pre-flight: ensure the result-side DONE barrier is present. This
        prevents a corner case where the queue advances past a job whose
        artifacts were never durably written.
        """
        done_marker = self.ctx.results_root / claim.run_id / "DONE"
        if not done_marker.is_file():
            raise RuntimeError(
                f"refusing to complete queue: DONE missing at {done_marker}"
            )
        return queue.complete(claim.claim_handle)

    def quarantine_on_failure(
        self, claim: JobClaim, exc: BaseException,
    ) -> None:
        """Tag a corrupt / failed job with a quarantine marker."""
        reason = type(exc).__name__.lower()
        # Stop the heartbeat thread so the queue lib doesn't keep
        # touching the now-quarantined claim file.
        try:
            if claim.claim_handle.heartbeat_stop is not None:
                claim.claim_handle.heartbeat_stop.set()
        except Exception:  # pragma: no cover -- defensive
            pass
        queue.quarantine(self.ctx.queue_root, claim.run_id, reason=reason)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "",
    )
    parser.add_argument(
        "--gpfs-root", required=True, type=Path,
        help="Shared GPFS root containing queue/ and results/ subdirs.",
    )
    parser.add_argument(
        "--host-name", default=socket.gethostname(),
        help="Daemon instance identifier (default: socket.gethostname()).",
    )
    parser.add_argument(
        "--bench-root", required=True, type=Path,
        help="Local bench repo root containing scripts/.",
    )
    parser.add_argument(
        "--results-root-rel", default="results",
        help="Results dir under gpfs-root (default: results).",
    )
    parser.add_argument(
        "--queue-root-rel", default="queue",
        help="Queue dir under gpfs-root (default: queue).",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=5.0,
        help="Seconds to sleep between empty-queue polls (default: 5).",
    )
    parser.add_argument(
        "--max-jobs", type=int, default=0,
        help="Exit after this many jobs (0 = unlimited; default: 0).",
    )
    parser.add_argument(
        "--lease-seconds", type=int, default=600,
        help="Daemon-lock lease in seconds (default: 600).",
    )
    parser.add_argument(
        "--heartbeat-seconds", type=int, default=30,
        help="Heartbeat interval in seconds (default: 30).",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single claim-exec-done cycle and exit (test mode).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Claim jobs but skip subprocess; emit mock results "
             "(verdict_class=unexpected-blocked, status=mock).",
    )
    return parser


def _ctx_from_args(args: argparse.Namespace) -> DaemonContext:
    return DaemonContext(
        gpfs_root=args.gpfs_root.resolve(),
        host_name=args.host_name,
        bench_root=args.bench_root.resolve(),
        results_root_rel=args.results_root_rel,
        queue_root_rel=args.queue_root_rel,
        poll_interval=args.poll_interval,
        max_jobs=args.max_jobs,
        lease_seconds=args.lease_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
        once=args.once,
        dry_run=args.dry_run,
    )


def _r19_startup_self_check() -> tuple[bool, str]:
    """V9 plan M2.0 (R19 verified-runner boundary, fail-closed):
    refuse to start unless the daemon is running as the provider uid
    OR has CAP_SETUID.

    The daemon brokers verified-mode evaluation -- it materialises
    sealed-golden tensors in process memory and runs candidate code
    against them. Running it under the same uid as the submitter
    would let a candidate process trivially read those tensors via
    ``/proc/<pid>/mem`` or shared CUDA mappings. v1 ships the
    *minimum-viable* defence: refuse-to-start when the runtime cannot
    enforce uid isolation, rather than silently degrading to a no-
    isolation mode (which is the prior behaviour).

    Returns ``(ok, reason)`` where ``ok=False`` means "refuse to
    start" and ``reason`` is a single-line human-readable error.

    The check is bypassed when ``$TAB_R19_SELF_CHECK_DISABLE=1`` is
    set, which is the explicit opt-out for tooling that runs the
    daemon in an unprivileged sandbox (CI / local dev) and accepts
    the weakened isolation. The opt-out is loud (logged at WARNING).
    """
    import os as _os

    if _os.environ.get("TAB_R19_SELF_CHECK_DISABLE") == "1":
        return True, "R19 self-check disabled via TAB_R19_SELF_CHECK_DISABLE=1"

    provider_uid_str = _os.environ.get("TAB_PROVIDER_UID")
    current_uid = _os.geteuid() if hasattr(_os, "geteuid") else None
    if current_uid is None:
        return False, (
            "R19: os.geteuid() unavailable on this platform; the daemon "
            "cannot verify it is running as the provider uid. Refusing "
            "to start."
        )

    if provider_uid_str:
        try:
            provider_uid = int(provider_uid_str)
        except ValueError:
            return False, (
                f"R19: $TAB_PROVIDER_UID={provider_uid_str!r} is not an "
                "integer; refusing to start"
            )
        if current_uid == provider_uid:
            return True, f"R19: running as provider uid {current_uid}"

    # uid mismatch (or $TAB_PROVIDER_UID unset): fall back to checking
    # CAP_SETUID -- the daemon needs that capability to switch to a
    # submitter uid for candidate sub-process isolation (the security
    # follow-up plan ships the actual switching). Without either, no
    # isolation is possible and we refuse to start.
    try:
        with open("/proc/self/status") as fh:
            caps_line = next(
                (line for line in fh if line.startswith("CapEff:")),
                None,
            )
    except OSError as exc:
        return False, (
            f"R19: cannot read /proc/self/status ({exc}); cannot verify "
            "CAP_SETUID. Refusing to start."
        )

    if caps_line is None:
        return False, (
            "R19: /proc/self/status missing CapEff line; cannot verify "
            "CAP_SETUID. Refusing to start."
        )

    try:
        caps_mask = int(caps_line.split()[1], 16)
    except (IndexError, ValueError) as exc:
        return False, (
            f"R19: malformed CapEff line {caps_line!r}: {exc}; refusing "
            "to start."
        )

    # CAP_SETUID is bit 7 (Linux capabilities, see capabilities(7)).
    _CAP_SETUID_BIT = 1 << 7  # noqa: N806 - capability bit constant
    if caps_mask & _CAP_SETUID_BIT:
        return True, f"R19: uid {current_uid} has CAP_SETUID"

    return False, (
        f"R19: daemon uid {current_uid} is not the provider uid "
        f"(${'TAB_PROVIDER_UID'}={provider_uid_str or '<unset>'}) "
        "AND lacks CAP_SETUID; cannot enforce candidate isolation. "
        "Refusing to start. Set $TAB_R19_SELF_CHECK_DISABLE=1 to opt "
        "out and accept the weakened isolation."
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log = logging.getLogger("bench_evaluator_daemon")
    # Parse args first so --help / argparse errors exit cleanly without
    # tripping the R19 self-check (which would block --help with a
    # confusing security error). The self-check runs ONLY before the
    # daemon actually starts brokering jobs.
    parser = _build_parser()
    args = parser.parse_args(argv)
    ok, reason = _r19_startup_self_check()
    if not ok:
        log.error(reason)
        return 2
    if "disabled" in reason.lower():
        log.warning(reason)
    else:
        log.info(reason)
    ctx = _ctx_from_args(args)
    daemon = BenchEvaluatorDaemon(ctx)
    return daemon.run()


# Re-export commonly used names so tests can patch ``daemon.queue`` etc.
__all__ = [
    "BenchEvaluatorDaemon",
    "DaemonContext",
    "JobClaim",
    "JobResult",
    "atomic",
    "extract_verdict",
    "lock",
    "main",
    "queue",
    "uuid",
]


if __name__ == "__main__":
    raise SystemExit(main())
