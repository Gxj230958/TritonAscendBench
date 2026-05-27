"""GPU A side: poll <bench>/.loop/jobs/ for batch specs, run phase_a_prime --verify per task.

Deployment (one-time on GPU A):

    cd /inspire/hdd/project/qianghuaxuexi/liuliming-liuliming/kernel/TritonAscendBench
    tmux new-session -d -s loop_daemon \
        'python3 scripts/loop/daemon_a.py --workers 2 2>&1 | tee -a .loop/daemon.log'

The daemon polls .loop/jobs/<batch_id>.json, claims via atomic rename to
.taken, runs each task in the spec (concurrent up to --workers), writes
per-task JSON verdicts to .loop/results/<batch_id>/<task_id>.json, refreshes
.loop/heartbeat/<batch_id>.txt every 10s, then writes .loop/done/<batch_id>.flag
when the batch is fully drained. Crashes are captured to .loop/dead/<id>.err.

The daemon does NOT git-pull: it runs whatever is currently in the on-disk
working tree (dev side rsyncs new code into gpfs before submitting a batch).
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[2]
LOOP_DIR = BENCH_ROOT / ".loop"
JOBS = LOOP_DIR / "jobs"
RESULTS = LOOP_DIR / "results"
HEARTBEAT = LOOP_DIR / "heartbeat"
DONE = LOOP_DIR / "done"
DEAD = LOOP_DIR / "dead"
TAKEN_SUFFIX = ".taken"


def _utc_now() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _atomic_write(dest: Path, content: str) -> None:
    """Write atomically via temp file in the same dir, then rename."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(dest)


def _claim(spec_path: Path) -> Path | None:
    """Atomic claim: rename jobs/<id>.json -> jobs/<id>.json.taken.
    Returns the new path if we won the race, None otherwise."""
    target = spec_path.with_suffix(spec_path.suffix + TAKEN_SUFFIX)
    try:
        spec_path.rename(target)
        return target
    except OSError:
        return None


def _heartbeat(batch_id: str) -> None:
    _atomic_write(HEARTBEAT / f"{batch_id}.txt", _utc_now() + "\n")


def _run_verify_one(
    task_id: str,
    batch_id: str,
    accept_stale_cache: bool,
    skip_smoke: bool,
    timeout_s: int,
) -> dict:
    """Run phase_a_prime --verify <task_id> as a subprocess; capture verdict."""
    started = _utc_now()
    cmd = [
        "python3",
        "scripts/phase_a_prime.py",
        "--verify",
        task_id,
    ]
    if accept_stale_cache:
        cmd.append("--accept-stale-cache")
    if skip_smoke:
        cmd.append("--skip-smoke")
    rc = None
    stdout = ""
    stderr = ""
    err = None
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(BENCH_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        rc = proc.returncode
        stdout = proc.stdout[-8000:]
        stderr = proc.stderr[-8000:]
    except subprocess.TimeoutExpired:
        err = f"timeout after {timeout_s}s"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    # Parse the "verify <task>: <STATUS> -- <reason>" line.
    verdict_status = ""
    verdict_line = ""
    for line in stdout.splitlines():
        if line.startswith(f"verify {task_id}:"):
            verdict_line = line
            parts = line.split(":", 2)
            if len(parts) >= 3:
                verdict_status = parts[2].strip().split(" ", 1)[0]
            break

    return {
        "task_id": task_id,
        "batch_id": batch_id,
        "started": started,
        "completed": _utc_now(),
        "rc": rc,
        "verdict_status": verdict_status,
        "verdict_line": verdict_line,
        "stdout_tail": stdout,
        "stderr_tail": stderr,
        "error": err,
    }


def _handle_batch(spec_path: Path, workers: int, timeout_s: int) -> None:
    """Run every task in a claimed spec; write results + heartbeat + done flag."""
    spec = json.loads(spec_path.read_text())
    batch_id = spec["batch_id"]
    tasks = spec["tasks"]
    accept_stale = bool(spec.get("accept_stale_cache", True))
    skip_smoke = bool(spec.get("skip_smoke", False))
    results_dir = RESULTS / batch_id
    results_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[{batch_id}] start n={len(tasks)} workers={workers} "
        f"accept_stale={accept_stale} skip_smoke={skip_smoke}",
        flush=True,
    )
    _heartbeat(batch_id)

    hb_stop = threading.Event()

    def _hb_pulse() -> None:
        while not hb_stop.is_set():
            _heartbeat(batch_id)
            time.sleep(10)

    hb_thread = threading.Thread(target=_hb_pulse, daemon=True)
    hb_thread.start()

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    _run_verify_one, t, batch_id, accept_stale, skip_smoke, timeout_s
                ): t
                for t in tasks
            }
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = {
                        "task_id": t,
                        "batch_id": batch_id,
                        "error": f"unhandled: {type(e).__name__}: {e}",
                    }
                _atomic_write(
                    results_dir / f"{t}.json",
                    json.dumps(result, indent=2, ensure_ascii=False),
                )
                line = result.get("verdict_line") or result.get("error") or "(no output)"
                print(f"[{batch_id}] {t} -> {line[:200]}", flush=True)
    finally:
        hb_stop.set()
        hb_thread.join(timeout=2)

    _atomic_write(DONE / f"{batch_id}.flag", _utc_now() + "\n")
    print(f"[{batch_id}] DONE", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--workers",
        type=int,
        default=2,
        help="number of concurrent tasks per batch (default 2 = 2x GPU A)",
    )
    ap.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="seconds between jobs/ scans",
    )
    ap.add_argument(
        "--per-task-timeout",
        type=int,
        default=1200,
        help="kill a single phase_a_prime --verify after this many seconds",
    )
    args = ap.parse_args()

    for d in (LOOP_DIR, JOBS, RESULTS, HEARTBEAT, DONE, DEAD):
        d.mkdir(parents=True, exist_ok=True)

    print(
        f"daemon_a watching {JOBS} (poll={args.poll_interval}s, "
        f"workers={args.workers}, per-task-timeout={args.per_task_timeout}s)",
        flush=True,
    )
    while True:
        # Sweep jobs/. We sort by name so the dev side can imply ordering
        # via batch_id (timestamp prefix).
        for spec_path in sorted(JOBS.glob("*.json")):
            taken = _claim(spec_path)
            if taken is None:
                continue
            try:
                _handle_batch(taken, args.workers, args.per_task_timeout)
            except Exception as e:
                err_file = DEAD / f"{spec_path.stem}.err"
                _atomic_write(
                    err_file,
                    f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                )
                print(f"[{spec_path.stem}] CRASH: {e}", flush=True)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main() or 0)
