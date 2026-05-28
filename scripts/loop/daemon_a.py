"""GPU A side: poll <loop_mailbox>/jobs/ for batch specs, run phase_a_prime --verify per task.

The loop mailbox lives OUTSIDE the bench repo at <bench_parent>/loop_mailbox/
(sibling of TritonAscendBench/). On gpfs that resolves to
/inspire/hdd/.../kernel/loop_mailbox/. Override via env var LOOP_MAILBOX_DIR.

Deployment (one-time on GPU A, then zero further intervention):

    cd /inspire/hdd/project/qianghuaxuexi/liuliming-liuliming/kernel/TritonAscendBench
    mkdir -p ../loop_mailbox/{jobs,results,heartbeat,done,dead}
    tmux new-session -d -s loop_daemon \
        'scripts/loop/supervise_daemon.sh --workers 4 --gpus 4 2>&1 | tee -a ../loop_mailbox/daemon.log'

Pass --gpus to fan out across multiple GPUs (one CUDA_VISIBLE_DEVICES per
worker). Without --gpus all workers share whatever CUDA env the daemon
inherited, which crowds device 0.

Self-healing:
  - daemon_a.py edits are picked up by an in-process re-exec at the next
    idle moment (see ``_maybe_restart_self``); the supervisor's pid slot
    is preserved.
  - hard crashes / OOM kills are restarted by ``supervise_daemon.sh`` with
    a 5s back-off.
  - The only event that requires re-running the tmux line is a host reboot.

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
import os
import queue
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

# Self-update plumbing. The daemon is long-running, so changes pushed to
# daemon_a.py do NOT propagate until the process restarts. Recording the
# file's mtime at startup lets us detect a real edit and re-exec in place
# (preserving the supervisor's pid slot -- it never sees an exit).
SELF_PATH = Path(__file__).resolve()
SELF_MTIME_AT_START = SELF_PATH.stat().st_mtime


def _self_changed_since_start() -> bool:
    """True if daemon_a.py on disk has been edited since this process started.

    Stat errors are swallowed (treated as "no change") so a transient FS hiccup
    cannot trigger a restart loop.
    """
    try:
        return SELF_PATH.stat().st_mtime > SELF_MTIME_AT_START
    except OSError:
        return False


def _maybe_restart_self() -> None:
    """Re-exec the daemon if daemon_a.py was edited since startup.

    Called between batches in the main poll loop (before each claim and
    before each sleep). ``os.execv`` REPLACES the current process in place,
    so the outer supervisor's pid slot is preserved and a fresh interpreter
    loads the new daemon_a.py with the original argv. Any still-live
    heartbeat threads from completed batches are also killed by the execv
    (they are ``daemon=True`` and were ``join(timeout=2)``-ed, so they may
    not always be fully stopped; execv collapses the whole process anyway).
    """
    if not _self_changed_since_start():
        return
    new_mtime = SELF_PATH.stat().st_mtime
    print(
        f"daemon_a: detected daemon_a.py edit (mtime {SELF_MTIME_AT_START:.0f}"
        f" -> {new_mtime:.0f}); re-execing self to pick up new code.",
        flush=True,
    )
    os.execv(sys.executable, [sys.executable] + sys.argv)


def _utc_now() -> str:
    # `isoformat() + "Z"` would emit invalid `...+00:00Z`; use strftime for `...Z`.
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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


_LIVE_HEARTBEAT_WINDOW_S = 60


def _quarantine_stranded_claims() -> None:
    """Mark `*.json.taken` files left by a crashed prior daemon as DEAD.

    A claim that finished cleanly leaves a ``done/<id>.flag`` next to its
    ``.taken``; a SIGKILL / OOM / host-kill mid-batch leaves only the
    ``.taken``. Without this sweep the orchestrator's wait() would hang on
    a stale heartbeat until the overall timeout. We do NOT re-claim or
    re-process: partial per-task results may already exist under
    ``results/<id>/`` and re-running would clobber them; the operator should
    submit a fresh batch_id to retry if needed.

    A second running daemon (overlap during a supervisor restart, two
    supervisors against the same mailbox) would otherwise misclassify a
    LIVE batch as stranded -- skip when ``heartbeat/<id>.txt`` was touched
    within ``_LIVE_HEARTBEAT_WINDOW_S`` seconds (the live daemon's pulse
    interval is 10s, so 60s is well above the noise floor).
    """
    now = time.time()
    for taken in JOBS.glob("*.json" + TAKEN_SUFFIX):
        name = taken.name
        if not name.endswith(".json" + TAKEN_SUFFIX):
            continue
        batch_id = name[: -len(".json" + TAKEN_SUFFIX)]
        try:
            _validate_batch_id(batch_id)
        except ValueError:
            continue
        if (DONE / f"{batch_id}.flag").exists():
            continue
        hb = HEARTBEAT / f"{batch_id}.txt"
        if hb.exists():
            try:
                age = now - hb.stat().st_mtime
            except OSError:
                age = float("inf")
            if age < _LIVE_HEARTBEAT_WINDOW_S:
                # Another daemon is actively pulsing for this batch; do not
                # falsely quarantine it.
                continue
        dead_marker = DEAD / f"{batch_id}.stranded.err"
        if dead_marker.exists():
            continue
        _atomic_write(
            dead_marker,
            "previous daemon exited while this batch was claimed "
            f"(orphan {taken.name} found with no fresh heartbeat). "
            f"Partial results may exist in results/{batch_id}/. Submit a "
            "fresh batch_id to retry.\n",
        )
        print(
            f"[{batch_id}] quarantined stranded claim -> {dead_marker.name}",
            flush=True,
        )


def _heartbeat(batch_id: str) -> None:
    _atomic_write(HEARTBEAT / f"{batch_id}.txt", _utc_now() + "\n")


def _run_verify_one(
    task_id: str,
    batch_id: str,
    accept_stale_cache: bool,
    require_smoke: bool,
    timeout_s: int,
    gpu_pool: "queue.Queue[int] | None",
) -> dict:
    """Run phase_a_prime --verify <task_id> as a subprocess; capture verdict.

    When ``gpu_pool`` is non-None each task acquires one GPU id from the pool
    and exports ``CUDA_VISIBLE_DEVICES=<id>`` to its subprocess, so N concurrent
    workers actually fan out across N GPUs instead of all crowding device 0.
    The id is returned to the pool when the subprocess exits (success, fail,
    timeout, or unexpected exception alike) so a single bad task never leaks
    the slot.
    """
    started = _utc_now()
    cmd = [
        "python3",
        "scripts/phase_a_prime.py",
        "--verify",
        task_id,
    ]
    if accept_stale_cache:
        cmd.append("--accept-stale-cache")
    if require_smoke:
        cmd.append("--require-smoke")
    rc = None
    stdout = ""
    stderr = ""
    err = None
    gpu_id: int | None = None
    env = None
    if gpu_pool is not None:
        # Blocks if all GPUs are busy -- correct backpressure when
        # workers > gpus. No timeout because per-task-timeout already
        # caps how long a slot can be held.
        gpu_id = gpu_pool.get()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(BENCH_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        rc = proc.returncode
        stdout = proc.stdout[-8000:]
        stderr = proc.stderr[-8000:]
    except subprocess.TimeoutExpired:
        err = f"timeout after {timeout_s}s"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    finally:
        if gpu_id is not None and gpu_pool is not None:
            gpu_pool.put(gpu_id)

    # Parse the "verify <task>: <STATUS> -- <reason>" line. cmd_verify
    # prints exactly one colon between task_id and STATUS, so split on
    # the first ":" only -- a 3-field split leaves verdict_status empty
    # for normal PASS / BLOCKED runs.
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
        "gpu_id": gpu_id,
        "verdict_status": verdict_status,
        "verdict_line": verdict_line,
        "stdout_tail": stdout,
        "stderr_tail": stderr,
        "error": err,
    }


def _handle_batch(
    spec_path: Path,
    workers: int,
    timeout_s: int,
    gpu_pool: "queue.Queue[int] | None",
) -> None:
    """Run every task in a claimed spec; write results + heartbeat + done flag."""
    spec = json.loads(spec_path.read_text())
    # Authoritative batch_id = claimed filename with `.json[.taken]` stripped,
    # not the JSON payload (which an attacker could set to '../etc/passwd' to
    # redirect heartbeat/results outside the mailbox). Path.stem only strips
    # the LAST suffix, so a ``<id>.json.taken`` spec needs both removed.
    name = spec_path.name
    if name.endswith(".json" + TAKEN_SUFFIX):
        batch_id = name[: -len(".json" + TAKEN_SUFFIX)]
    elif name.endswith(".json"):
        batch_id = name[: -len(".json")]
    else:
        raise ValueError(
            f"unexpected spec filename {name!r}; expected *.json or *.json{TAKEN_SUFFIX}"
        )
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
    require_smoke = bool(spec.get("require_smoke", False))
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
        f"accept_stale={accept_stale} require_smoke={require_smoke}",
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
                    _run_verify_one,
                    t,
                    batch_id,
                    accept_stale,
                    require_smoke,
                    timeout_s,
                    gpu_pool,
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
        help="number of concurrent tasks per batch (default 2)",
    )
    ap.add_argument(
        "--gpus",
        type=str,
        default="",
        help=(
            "GPU pool for per-worker CUDA_VISIBLE_DEVICES pinning. "
            "Either an integer N (use device ids 0..N-1) or a comma-separated "
            "list like '0,2,5'. Empty (default) disables pinning -- every "
            "subprocess inherits the daemon's CUDA env, which is correct only "
            "when --workers=1 or the workload genuinely fits on one device."
        ),
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

    gpu_ids = _parse_gpu_spec(args.gpus)
    gpu_pool: "queue.Queue[int] | None" = None
    if gpu_ids:
        if args.workers > len(gpu_ids):
            # Allowed but warn: extra workers will block on get() until a
            # GPU frees up. That's correct backpressure, just wasteful if
            # the operator meant for --workers to mean "true parallelism".
            print(
                f"WARNING: --workers={args.workers} exceeds --gpus pool "
                f"size {len(gpu_ids)}; extra workers will queue.",
                flush=True,
            )
        gpu_pool = queue.Queue()
        for g in gpu_ids:
            gpu_pool.put(g)

    for d in (LOOP_DIR, JOBS, RESULTS, HEARTBEAT, DONE, DEAD):
        d.mkdir(parents=True, exist_ok=True)

    gpu_desc = f"gpus={gpu_ids}" if gpu_ids else "gpus=inherit"
    print(
        f"daemon_a watching {JOBS} (poll={args.poll_interval}s, "
        f"workers={args.workers}, {gpu_desc}, "
        f"per-task-timeout={args.per_task_timeout}s)",
        flush=True,
    )
    while True:
        # Re-run on EVERY poll cycle (not just startup). A daemon that crashes
        # right after a heartbeat write leaves an inherited-but-soon-stale
        # heartbeat behind; checking only once at startup would skip the
        # quarantine while the heartbeat was still fresh, then never re-check
        # after it aged past the window. The work is cheap (stat over the
        # few .taken files in flight) and safe (live batches stay skipped by
        # the fresh-heartbeat guard inside the function).
        _quarantine_stranded_claims()
        # Sweep jobs/. We sort by name so the dev side can imply ordering
        # via batch_id (timestamp prefix). The mtime self-check fires BEFORE
        # every claim (not just per outer loop) so an edit landing mid-sweep
        # re-execs us before we pick up the next batch under stale code.
        for spec_path in sorted(JOBS.glob("*.json")):
            _maybe_restart_self()
            taken = _claim(spec_path)
            if taken is None:
                continue
            try:
                _handle_batch(
                    taken, args.workers, args.per_task_timeout, gpu_pool
                )
            except Exception as e:
                err_file = DEAD / f"{spec_path.stem}.err"
                _atomic_write(
                    err_file,
                    f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                )
                print(f"[{spec_path.stem}] CRASH: {e}", flush=True)
        # Also check before sleeping: an edit during a poll-wait should be
        # picked up at the next wake without forcing one more dead sweep.
        _maybe_restart_self()
        time.sleep(args.poll_interval)


def _parse_gpu_spec(spec: str) -> list[int]:
    """Parse the --gpus flag into a list of device ids.

    Accepts either an integer count ("8" -> [0,1,...,7]) or an explicit
    comma-separated list ("0,2,5"). Empty string means "no pinning".
    Duplicates and negatives are rejected so a typo can't silently route
    two workers to the same device or to an invalid id.
    """
    spec = spec.strip()
    if not spec:
        return []
    if "," in spec:
        # Validate every segment explicitly -- silently dropping blanks
        # would turn typos like '0,,2' or '0,' into a smaller (or empty)
        # pool, falling back to inherited-CUDA-env behavior without
        # warning the operator.
        ids: list[int] = []
        for part in spec.split(","):
            s = part.strip()
            if not s:
                raise ValueError(
                    f"--gpus has empty segment in {spec!r}; expected ids"
                )
            ids.append(int(s))
    else:
        n = int(spec)
        if n <= 0:
            raise ValueError(f"--gpus must be positive, got {n}")
        ids = list(range(n))
    if any(x < 0 for x in ids):
        raise ValueError(f"--gpus contains negative device id: {ids}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"--gpus contains duplicate device ids: {ids}")
    return ids


if __name__ == "__main__":
    sys.exit(main() or 0)
