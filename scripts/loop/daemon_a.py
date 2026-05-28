"""GPU A side: poll <loop_mailbox>/jobs/ for batch specs, run phase_a_prime --verify per task.

The loop mailbox lives OUTSIDE the bench repo at <bench_parent>/loop_mailbox/
(sibling of TritonAscendBench/). On gpfs that resolves to
/inspire/hdd/.../kernel/loop_mailbox/. Override via env var LOOP_MAILBOX_DIR.

Deployment (one-time on GPU A):

    cd /inspire/hdd/project/qianghuaxuexi/liuliming-liuliming/kernel/TritonAscendBench
    mkdir -p ../loop_mailbox/{jobs,results,heartbeat,done,dead}
    tmux new-session -d -s loop_daemon \
        'python3 scripts/loop/daemon_a.py --workers 2 2>&1 | tee -a ../loop_mailbox/daemon.log'

The daemon polls <loop_mailbox>/jobs/<batch_id>.json, claims via atomic rename
to .taken, runs each task in the spec (concurrent up to --workers), writes
per-task JSON verdicts to <loop_mailbox>/results/<batch_id>/<task_id>.json,
refreshes <loop_mailbox>/heartbeat/<batch_id>.txt every 10s, then writes
<loop_mailbox>/done/<batch_id>.flag when the batch is fully drained. Crashes
are captured to <loop_mailbox>/dead/<id>.err.

The daemon does NOT git-pull: it runs whatever is currently in the on-disk
working tree (dev side rsyncs new code into gpfs before submitting a batch).
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Strict batch_id whitelist: starts with 'b', alphanumeric / dash / underscore,
# max 64 chars. Rejects path traversal ('../...'), shell metacharacters,
# absolute paths, and excessively long ids. Both dev and A enforce this.
_BATCH_ID_RE = re.compile(r"^b[A-Za-z0-9_\-]{1,63}$")
# Strict task_id whitelist matches the bench naming convention L<level>_<idx>_<name>.
_TASK_ID_RE = re.compile(r"^L[1-3]_[0-9]{1,3}_[A-Za-z0-9_]+$")


def _validate_batch_id(batch_id: str) -> None:
    if not isinstance(batch_id, str) or not _BATCH_ID_RE.match(batch_id):
        raise ValueError(
            f"invalid batch_id {batch_id!r} -- must match {_BATCH_ID_RE.pattern}"
        )


def _validate_task_id(task_id: str) -> None:
    if not isinstance(task_id, str) or not _TASK_ID_RE.match(task_id):
        raise ValueError(
            f"invalid task_id {task_id!r} -- must match {_TASK_ID_RE.pattern}"
        )

import os

BENCH_ROOT = Path(__file__).resolve().parents[2]
# Loop mailbox lives OUTSIDE the bench repo so runtime artifacts cannot
# pollute git status / get clobbered by repo cleanup. Default location:
# the bench's parent directory (sibling of TritonAscendBench/), which is
# the kernel/ root on gpfs and on dev. Overridable via env var for
# alternate deployments without touching the script.
LOOP_DIR = Path(os.environ.get("LOOP_MAILBOX_DIR") or (BENCH_ROOT.parent / "loop_mailbox"))
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

    # Parse the "verify <task>: <STATUS> -- <reason>" line. cmd_verify
    # prints exactly one colon between task_id and STATUS (codex review
    # caught the prior 3-field split bug that left verdict_status empty
    # for normal PASS / BLOCKED runs).
    verdict_status = ""
    verdict_line = ""
    prefix = f"verify {task_id}: "
    for line in stdout.splitlines():
        if line.startswith(prefix):
            verdict_line = line
            rest = line[len(prefix):].strip()
            verdict_status = rest.split(" ", 1)[0]
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
    # Authoritative batch_id = the claimed FILENAME stem, not the JSON
    # payload (which an attacker could set to '../etc/passwd' to redirect
    # heartbeat/results outside the mailbox).
    batch_id = spec_path.stem
    if batch_id.endswith(TAKEN_SUFFIX.lstrip(".")):
        batch_id = batch_id[: -len(TAKEN_SUFFIX.lstrip("."))]
    _validate_batch_id(batch_id)
    payload_batch = spec.get("batch_id")
    if payload_batch != batch_id:
        print(
            f"[{batch_id}] payload.batch_id={payload_batch!r} != filename stem; "
            f"using filename stem as authoritative.",
            flush=True,
        )
    tasks = spec.get("tasks") or []
    for t in tasks:
        _validate_task_id(t)
    accept_stale = bool(spec.get("accept_stale_cache", True))
    skip_smoke = bool(spec.get("skip_smoke", False))
    results_dir = RESULTS / batch_id
    done_flag = DONE / f"{batch_id}.flag"

    # Idempotency / re-submit guard (Codex FAIL #2). Refuse to re-process
    # a batch whose done flag already exists -- dev side must use a new
    # batch_id, NOT recycle a stale one. Without this an old success
    # marker could mask a fresh failure when the same batch_id is sent
    # twice (also the per-task results would be silently overwritten).
    if done_flag.exists():
        print(
            f"[{batch_id}] REFUSED: done.flag already exists; "
            f"submit with a fresh batch_id instead. spec moved to dead/.",
            flush=True,
        )
        _atomic_write(
            DEAD / f"{batch_id}.duplicate.err",
            f"duplicate batch_id (done.flag exists): {spec_path}\n",
        )
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[{batch_id}] start n={len(tasks)} workers={workers} "
        f"accept_stale={accept_stale} skip_smoke={skip_smoke}",
        flush=True,
    )
    _heartbeat(batch_id)

    hb_stop = threading.Event()
    hb_fail = threading.Event()

    def _hb_pulse() -> None:
        # Codex REFINE: catch heartbeat write failures so a silently
        # dying pulse thread does not leave wait() guessing whether the
        # daemon is dead vs healthy. Any exception flips hb_fail so the
        # batch is marked dead at completion.
        while not hb_stop.is_set():
            try:
                _heartbeat(batch_id)
            except Exception as e:
                print(
                    f"[{batch_id}] heartbeat write failed: {type(e).__name__}: {e}",
                    flush=True,
                )
                hb_fail.set()
                return
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

    if hb_fail.is_set():
        _atomic_write(
            DEAD / f"{batch_id}.heartbeat.err",
            "heartbeat thread failed mid-batch; results may be incomplete.\n",
        )
        print(f"[{batch_id}] HEARTBEAT FAILED (see dead/)", flush=True)
        return
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
