"""Tests for ``automation.bench_evaluator_daemon`` (Unit 4.1).

dev-machine unit tests: mock GPFS via tmpdir, mock ``subprocess.run`` so
the daemon never invokes real bench binaries, mock queue / lock primitives
where convenient. Verifies CLI plumbing, the 5 happy-/edge-path scenarios
named in the impl plan, and the strict 9-field ``result.json`` schema.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(
    tmp_path: Path,
    *,
    bench_root: Path | None = None,
    max_jobs: int = 1,
    once: bool = True,
    dry_run: bool = False,
    poll_interval: float = 0.01,
    lease_seconds: int = 60,
):
    from automation.bench_evaluator_daemon import DaemonContext

    gpfs = tmp_path / "gpfs"
    gpfs.mkdir(parents=True, exist_ok=True)
    if bench_root is None:
        bench_root = tmp_path / "bench"
        bench_root.mkdir(parents=True, exist_ok=True)
        (bench_root / "scripts").mkdir(parents=True, exist_ok=True)
    return DaemonContext(
        gpfs_root=gpfs,
        host_name="hostA",
        bench_root=bench_root,
        results_root_rel="results",
        queue_root_rel="queue",
        poll_interval=poll_interval,
        max_jobs=max_jobs,
        lease_seconds=lease_seconds,
        heartbeat_seconds=1,
        once=once,
        dry_run=dry_run,
    )


def _enqueue(ctx, run_id: str, task_id: str, cmd: str) -> Path:
    from automation.lib import queue as queue_lib

    return queue_lib.enqueue(
        ctx.queue_root, run_id,
        {"task_id": task_id, "cmd": cmd},
    )


def _fake_subprocess_run(stdout_text: str, returncode: int = 0):
    """Build a ``subprocess.run`` stand-in that writes ``stdout_text``."""

    def _run(cmd, stdout=None, stderr=None, timeout=None, check=False, **kw):
        if hasattr(stdout, "write"):
            stdout.write(stdout_text)
            stdout.flush()
        cp = MagicMock()
        cp.returncode = returncode
        cp.args = cmd
        return cp

    return _run


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_daemon_init_validates_gpfs_root(tmp_path):
    """``BenchEvaluatorDaemon.__init__`` rejects a missing gpfs_root."""
    from automation.bench_evaluator_daemon import (
        BenchEvaluatorDaemon,
        DaemonContext,
    )

    bench = tmp_path / "bench"
    bench.mkdir()
    bad_ctx = DaemonContext(
        gpfs_root=tmp_path / "does-not-exist",
        host_name="hostA",
        bench_root=bench,
        results_root_rel="results",
        queue_root_rel="queue",
        poll_interval=1.0,
        max_jobs=1,
        lease_seconds=60,
        heartbeat_seconds=10,
        once=True,
        dry_run=False,
    )
    with pytest.raises(FileNotFoundError, match="gpfs_root"):
        BenchEvaluatorDaemon(bad_ctx)


def test_claim_one_job_empty_queue(tmp_path):
    """``claim_one_job`` returns None when no READY entries exist."""
    from automation.bench_evaluator_daemon import BenchEvaluatorDaemon

    ctx = _make_ctx(tmp_path)
    daemon = BenchEvaluatorDaemon(ctx)
    assert daemon.claim_one_job() is None


def test_claim_one_job_ready_to_claimed(tmp_path):
    """READY -> CLAIMED.<host>.<pid>.<starttime>.<ts> via lib.queue.claim."""
    from automation.bench_evaluator_daemon import BenchEvaluatorDaemon

    ctx = _make_ctx(tmp_path)
    _enqueue(ctx, "r1", "L1_99_demo", "resolve")
    daemon = BenchEvaluatorDaemon(ctx)
    claim = daemon.claim_one_job()
    assert claim is not None
    assert claim.run_id == "r1"
    assert claim.task_id == "L1_99_demo"
    assert claim.cmd == "resolve"
    assert claim.claim_handle.claim_marker.name.startswith("CLAIMED.hostA.")
    assert not (ctx.queue_root / "r1" / "READY").exists()


def test_execute_job_resolve_cmd(tmp_path):
    """resolve cmd: subprocess output is parsed via parse_resolve_log."""
    from automation.bench_evaluator_daemon import BenchEvaluatorDaemon

    ctx = _make_ctx(tmp_path)
    _enqueue(ctx, "r1", "L1_99_demo", "resolve")
    daemon = BenchEvaluatorDaemon(ctx)
    claim = daemon.claim_one_job()
    assert claim is not None

    fake = _fake_subprocess_run(
        "[L1_99_demo] RESOLVED demo_fn :: PASS\n", returncode=0,
    )
    with patch.object(subprocess, "run", side_effect=fake):
        result = daemon.execute_job(claim)

    assert result.verdict == "PASS"
    assert result.verdict_class is None
    assert result.status == "completed"
    # Verify the sub-cmd was built for resolve
    log = (ctx.results_root / "r1" / "log.txt").read_text()
    assert "RESOLVED" in log


def test_execute_job_phase_a_verify_cmd(tmp_path):
    """phase_a_verify cmd: meta.yaml drives the verdict."""
    from automation.bench_evaluator_daemon import BenchEvaluatorDaemon

    ctx = _make_ctx(tmp_path)
    # phase_a_prime --verify writes status into tasks/<level>/<task>/meta.yaml.
    # We seed meta.yaml inside bench_root so parse_phase_a_verify sees PASS.
    td = ctx.bench_root / "tasks" / "L2" / "L2_77_demo"
    td.mkdir(parents=True)
    (td / "meta.yaml").write_text(
        yaml.safe_dump({"phase_a_prime": {"status": "PASS", "reason": "ok"}}),
        encoding="utf-8",
    )
    _enqueue(ctx, "r2", "L2_77_demo", "phase_a_verify")
    daemon = BenchEvaluatorDaemon(ctx)
    claim = daemon.claim_one_job()
    assert claim is not None

    fake = _fake_subprocess_run("verify ok\n", returncode=0)
    with patch.object(subprocess, "run", side_effect=fake):
        result = daemon.execute_job(claim)

    assert result.verdict == "PASS"
    assert result.status == "completed"


def test_execute_job_timeout_quarantine(tmp_path):
    """subprocess timeout -> JobResult(status=timeout, verdict=BLOCKED).

    The daemon main loop catches the resulting failure path and calls
    ``quarantine_on_failure``; we verify the JobResult shape directly and
    also exercise the quarantine helper.
    """
    from automation.bench_evaluator_daemon import BenchEvaluatorDaemon

    ctx = _make_ctx(tmp_path, lease_seconds=2)
    _enqueue(ctx, "r1", "L1_99_demo", "resolve")
    daemon = BenchEvaluatorDaemon(ctx)
    claim = daemon.claim_one_job()
    assert claim is not None

    def _raise_timeout(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kw.get("timeout", 0))

    with patch.object(subprocess, "run", side_effect=_raise_timeout):
        result = daemon.execute_job(claim)

    assert result.status == "timeout"
    assert result.verdict == "BLOCKED"
    assert result.verdict_class == "unexpected-blocked"
    assert "timeout" in result.reason.lower()

    # Now exercise the quarantine helper, simulating a main-loop catch.
    daemon.quarantine_on_failure(claim, subprocess.TimeoutExpired(cmd="x", timeout=1))
    quarantine_markers = list(
        (ctx.queue_root / "r1").glob("QUARANTINED.*"),
    )
    assert any("timeoutexpired" in m.name.lower() for m in quarantine_markers)


def test_write_result_atomic_and_strict_9_fields(tmp_path):
    """result.json honours the strict 9-field schema + sidecar provenance."""
    from automation.bench_evaluator_daemon import (
        BenchEvaluatorDaemon,
        JobResult,
    )
    from automation.extract_verdict import validate_result_schema

    ctx = _make_ctx(tmp_path)
    _enqueue(ctx, "r1", "L1_99_demo", "resolve")
    daemon = BenchEvaluatorDaemon(ctx)
    claim = daemon.claim_one_job()
    assert claim is not None

    job_result = JobResult(
        verdict="PASS",
        verdict_class=None,
        reason="resolver PASS",
        raw_subclass="",
        status="completed",
        duration_ms=42,
    )
    daemon.write_result(claim, job_result)

    run_dir = ctx.results_root / "r1"
    result = json.loads((run_dir / "result.json").read_text())
    prov = json.loads((run_dir / "_provenance.json").read_text())
    assert (run_dir / "DONE").is_file()

    # Strict schema (will raise if any field is wrong)
    validate_result_schema(result)
    expected_keys = {
        "schema_version", "run_id", "task_id", "cmd", "verdict",
        "verdict_class", "reason", "produced_at", "producer",
    }
    assert set(result.keys()) == expected_keys
    assert result["producer"] == "daemon"
    assert result["cmd"] == "resolve"
    # Provenance carries our duration / status
    assert prov["extra"]["duration_ms"] == 42
    assert prov["extra"]["status"] == "completed"


def test_write_result_done_barrier_after_files(tmp_path):
    """DONE mtime must be >= result.json + _provenance.json mtimes."""
    from automation.bench_evaluator_daemon import (
        BenchEvaluatorDaemon,
        JobResult,
    )

    ctx = _make_ctx(tmp_path)
    _enqueue(ctx, "r1", "L1_99_demo", "resolve")
    daemon = BenchEvaluatorDaemon(ctx)
    claim = daemon.claim_one_job()
    assert claim is not None

    job_result = JobResult(
        verdict="PASS", verdict_class=None, reason="ok",
        status="completed",
    )
    daemon.write_result(claim, job_result)

    run_dir = ctx.results_root / "r1"
    t_result = os.path.getmtime(run_dir / "result.json")
    t_prov = os.path.getmtime(run_dir / "_provenance.json")
    t_done = os.path.getmtime(run_dir / "DONE")
    # DONE was touched STRICTLY after both JSON files were written.
    # Use >= because filesystems may quantise mtime to whole seconds.
    assert t_done >= t_result
    assert t_done >= t_prov


def test_complete_queue_claimed_to_done(tmp_path):
    """CLAIMED.<…> renamed to DONE in the queue dir after write_result."""
    from automation.bench_evaluator_daemon import (
        BenchEvaluatorDaemon,
        JobResult,
    )

    ctx = _make_ctx(tmp_path)
    _enqueue(ctx, "r1", "L1_99_demo", "resolve")
    daemon = BenchEvaluatorDaemon(ctx)
    claim = daemon.claim_one_job()
    assert claim is not None

    job_result = JobResult(
        verdict="PASS", verdict_class=None, reason="ok",
        status="completed",
    )
    daemon.write_result(claim, job_result)
    done_path = daemon.complete_queue(claim)

    assert done_path.name == "DONE"
    assert done_path.is_file()
    # And the CLAIMED.* marker is gone.
    claim_markers = list((ctx.queue_root / "r1").glob("CLAIMED.*"))
    # Filter out heartbeat markers (already cleaned up by lib.queue.complete)
    claim_markers = [m for m in claim_markers if not m.name.endswith(".heartbeat")]
    assert claim_markers == []


def test_signal_term_graceful_shutdown(tmp_path):
    """SIGTERM mid-loop: current job finishes, then main loop exits.

    Simulate this by triggering ``request_stop`` from a thread while a
    long-running execute_job is in-flight; assert the loop terminates
    only after the in-flight job is recorded.
    """
    from automation.bench_evaluator_daemon import (
        BenchEvaluatorDaemon,
        JobResult,
    )

    ctx = _make_ctx(tmp_path, max_jobs=10, once=False, poll_interval=0.01)
    _enqueue(ctx, "r1", "L1_99_demo", "resolve")
    daemon = BenchEvaluatorDaemon(ctx)

    call_count = {"execute": 0}
    job_started = threading.Event()
    job_finish = threading.Event()

    def _slow_execute(claim):
        call_count["execute"] += 1
        job_started.set()
        # Block briefly while the "signal" arrives, then complete normally.
        job_finish.wait(timeout=2.0)
        return JobResult(
            verdict="PASS", verdict_class=None, reason="ok",
            status="completed",
        )

    # Patch lock.acquire_lock so the daemon does not need a real lock dir.
    class _DummyLock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _send_term():
        job_started.wait(timeout=2.0)
        daemon.request_stop()
        job_finish.set()

    sender = threading.Thread(target=_send_term)
    sender.start()
    try:
        with patch.object(daemon, "execute_job", side_effect=_slow_execute), \
             patch("automation.bench_evaluator_daemon.lock.acquire_lock",
                   return_value=_DummyLock()):
            rc = daemon.run()
        sender.join(timeout=2.0)
    finally:
        # Defensive: ensure no thread is left blocked
        job_finish.set()
        sender.join(timeout=1.0)

    assert rc == 0
    # The one in-flight job ran to completion.
    assert call_count["execute"] == 1
    # And the loop stopped (no second claim attempt).
    assert daemon._jobs_done == 1


def test_max_jobs_terminates_loop(tmp_path):
    """max_jobs=2 -> loop exits after exactly 2 jobs."""
    from automation.bench_evaluator_daemon import (
        BenchEvaluatorDaemon,
        JobResult,
    )

    ctx = _make_ctx(tmp_path, max_jobs=2, once=False, poll_interval=0.01)
    _enqueue(ctx, "r1", "L1_99_demo", "resolve")
    _enqueue(ctx, "r2", "L1_98_demo", "resolve")
    _enqueue(ctx, "r3", "L1_97_demo", "resolve")
    daemon = BenchEvaluatorDaemon(ctx)

    def _fast_execute(claim):
        return JobResult(
            verdict="PASS", verdict_class=None, reason="ok",
            status="completed",
        )

    class _DummyLock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch.object(daemon, "execute_job", side_effect=_fast_execute), \
         patch("automation.bench_evaluator_daemon.lock.acquire_lock",
               return_value=_DummyLock()):
        rc = daemon.run()

    assert rc == 0
    assert daemon._jobs_done == 2
    # r3 is still READY (or CLAIMED-but-not-finished) in the queue
    r3_done = (ctx.queue_root / "r3" / "DONE").exists()
    assert not r3_done, "third job should not have completed"


def test_dry_run_no_subprocess(tmp_path):
    """--dry-run: subprocess.run is never called; result.json has 9 fields."""
    from automation.bench_evaluator_daemon import BenchEvaluatorDaemon
    from automation.extract_verdict import validate_result_schema

    ctx = _make_ctx(tmp_path, dry_run=True)
    _enqueue(ctx, "r1", "L1_99_demo", "resolve")
    daemon = BenchEvaluatorDaemon(ctx)
    claim = daemon.claim_one_job()
    assert claim is not None

    def _explode(*a, **kw):
        raise AssertionError("subprocess.run must NOT be called in --dry-run")

    with patch.object(subprocess, "run", side_effect=_explode):
        result = daemon.execute_job(claim)

    assert result.status == "mock"
    assert result.verdict_class == "unexpected-blocked"
    daemon.write_result(claim, result)
    daemon.complete_queue(claim)

    result_json = json.loads(
        (ctx.results_root / "r1" / "result.json").read_text(),
    )
    validate_result_schema(result_json)
    # Strict 9-field schema -- no extra keys, no missing keys.
    assert set(result_json.keys()) == {
        "schema_version", "run_id", "task_id", "cmd", "verdict",
        "verdict_class", "reason", "produced_at", "producer",
    }
    prov = json.loads(
        (ctx.results_root / "r1" / "_provenance.json").read_text(),
    )
    assert prov["extra"]["dry_run"] is True
    assert prov["extra"]["status"] == "mock"


def test_cli_help_does_not_crash(capsys):
    """`--help` parses and exits 0 (smoke test for argparse plumbing)."""
    from automation.bench_evaluator_daemon import main

    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--gpfs-root" in out
    assert "--bench-root" in out
    assert "--dry-run" in out
    assert "--max-jobs" in out
