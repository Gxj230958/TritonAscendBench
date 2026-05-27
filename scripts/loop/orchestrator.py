"""dev side: submit / wait / collect batches against the A-side daemon.

The dev box cannot see gpfs directly. All file movement goes through
the reverse SSH tunnel on 127.0.0.1:2222 -> root@<B> which has gpfs
mounted at the same path as A.

Subcommands:
  sync-code        rsync local working tree (tasks/ scripts/ pyproject.toml +
                   tasks/_frozen_shim) to gpfs so the daemon sees latest code
  submit           write a batch spec to gpfs <bench>/.loop/jobs/<id>.json
  wait             poll <bench>/.loop/done/<id>.flag (heartbeat-aware)
  collect          rsync <bench>/.loop/results/<id>/ back to local
  run              sync-code + submit + wait + collect in one shot
  status           list jobs / running / done on gpfs

Convention:
  REMOTE_BENCH = /inspire/hdd/project/qianghuaxuexi/liuliming-liuliming/kernel/TritonAscendBench
  SSH = ssh -p 2222 root@127.0.0.1
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

LOCAL_BENCH = Path(__file__).resolve().parents[2]
LOCAL_LOOP = LOCAL_BENCH / ".loop"
REMOTE_BENCH = "/inspire/hdd/project/qianghuaxuexi/liuliming-liuliming/kernel/TritonAscendBench"
SSH_PORT = "2222"
SSH_HOST = "root@127.0.0.1"
SSH_RSH = f"ssh -p {SSH_PORT} -o BatchMode=yes -o StrictHostKeyChecking=no"


def _ssh(cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-p", SSH_PORT, "-o", "BatchMode=yes",
         "-o", "StrictHostKeyChecking=no", SSH_HOST, cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def _rsync(src: str, dst: str, *, delete: bool = False, exclude: list[str] | None = None) -> int:
    args = ["rsync", "-az", "--info=stats1,progress0", "-e", SSH_RSH]
    if delete:
        args.append("--delete")
    for pat in exclude or []:
        args.extend(["--exclude", pat])
    args.extend([src, dst])
    proc = subprocess.run(args, text=True)
    return proc.returncode


def cmd_sync_code(args: argparse.Namespace) -> int:
    """rsync local bench code dirs into the remote gpfs bench root."""
    # Push selected paths to keep traffic small. Bench infra under scripts/,
    # task data under tasks/, dep deps in pyproject.toml.
    paths = [
        "scripts/",
        "tasks/",
        "pyproject.toml",
        "triton_ascend_bench/",
    ]
    rc_total = 0
    for p in paths:
        src = str(LOCAL_BENCH / p)
        if (LOCAL_BENCH / p).is_dir():
            src = src.rstrip("/") + "/"
            dst = f"{SSH_HOST}:{REMOTE_BENCH}/{p.rstrip('/')}/"
        else:
            dst = f"{SSH_HOST}:{REMOTE_BENCH}/{p}"
        print(f"rsync {src} -> {dst}")
        rc = _rsync(src, dst, exclude=[".loop/", "__pycache__/", "*.pyc"])
        rc_total |= rc
    return rc_total


def cmd_submit(args: argparse.Namespace) -> int:
    """Write a batch spec into local .loop/jobs/ + rsync to gpfs."""
    batch_id = args.batch_id or _new_batch_id()
    spec = {
        "batch_id": batch_id,
        "submitted_at": _utc_now(),
        "tasks": args.tasks,
        "accept_stale_cache": True,
        "skip_smoke": bool(args.skip_smoke),
    }
    (LOCAL_LOOP / "jobs").mkdir(parents=True, exist_ok=True)
    spec_path = LOCAL_LOOP / "jobs" / f"{batch_id}.json"
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    print(f"local spec: {spec_path}")

    # Push to gpfs.
    rc = _rsync(
        str(spec_path),
        f"{SSH_HOST}:{REMOTE_BENCH}/.loop/jobs/{batch_id}.json",
    )
    if rc != 0:
        print(f"rsync FAILED rc={rc}", file=sys.stderr)
        return rc
    print(f"submitted batch_id={batch_id} tasks={args.tasks}")
    print(batch_id)  # last line is just the batch_id so callers can capture
    return 0


def cmd_wait(args: argparse.Namespace) -> int:
    """Poll <gpfs>/.loop/done/<batch>.flag with heartbeat-staleness guard."""
    batch_id = args.batch_id
    timeout = args.timeout
    interval = args.interval
    stale_after = args.stale_after
    start = time.time()
    last_print = 0.0
    while time.time() - start < timeout:
        r = _ssh(
            f"test -f {REMOTE_BENCH}/.loop/done/{batch_id}.flag && echo DONE || "
            f"stat -c '%Y' {REMOTE_BENCH}/.loop/heartbeat/{batch_id}.txt 2>/dev/null || echo MISSING"
        )
        out = r.stdout.strip()
        if out == "DONE":
            print(f"{batch_id}: DONE")
            return 0
        if out.isdigit():
            hb_age = time.time() - int(out)
            elapsed = int(time.time() - start)
            if time.time() - last_print > 30:
                print(f"  [{elapsed:5d}s] heartbeat age {hb_age:5.0f}s", flush=True)
                last_print = time.time()
            if hb_age > stale_after:
                print(
                    f"{batch_id}: HEARTBEAT STALE ({hb_age:.0f}s > {stale_after}s) -- "
                    f"daemon may be dead",
                    file=sys.stderr,
                )
                return 2
        elif out == "MISSING":
            elapsed = int(time.time() - start)
            if elapsed > 60:
                print(
                    f"{batch_id}: no heartbeat file after {elapsed}s -- "
                    f"daemon never picked it up?",
                    file=sys.stderr,
                )
                return 2
        time.sleep(interval)
    print(f"{batch_id}: timeout after {timeout}s", file=sys.stderr)
    return 2


def cmd_collect(args: argparse.Namespace) -> int:
    """rsync <gpfs>/.loop/results/<batch>/ back to local; print verdict summary."""
    batch_id = args.batch_id
    local_dir = LOCAL_LOOP / "results" / batch_id
    local_dir.mkdir(parents=True, exist_ok=True)
    rc = _rsync(
        f"{SSH_HOST}:{REMOTE_BENCH}/.loop/results/{batch_id}/",
        str(local_dir) + "/",
    )
    if rc != 0:
        return rc
    print(f"=== {batch_id} verdicts ===")
    counts: dict[str, int] = {}
    for jf in sorted(local_dir.glob("*.json")):
        d = json.loads(jf.read_text())
        tid = d["task_id"]
        status = d.get("verdict_status") or ("ERR" if d.get("error") else "?")
        line = d.get("verdict_line") or d.get("error") or "(no verdict line)"
        counts[status] = counts.get(status, 0) + 1
        print(f"  {tid:50s} {status:25s} {line[:120]}")
    print("---")
    for k, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:3d} {k}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """sync-code + submit + wait + collect, one shot."""
    rc = cmd_sync_code(args)
    if rc != 0:
        return rc
    batch_id = args.batch_id or _new_batch_id()
    args.batch_id = batch_id
    rc = cmd_submit(args)
    if rc != 0:
        return rc
    rc = cmd_wait(args)
    if rc != 0:
        return rc
    return cmd_collect(args)


def cmd_status(args: argparse.Namespace) -> int:
    r = _ssh(
        f"echo === jobs ===; ls -la {REMOTE_BENCH}/.loop/jobs/ 2>/dev/null; "
        f"echo === done ===; ls -la {REMOTE_BENCH}/.loop/done/ 2>/dev/null; "
        f"echo === results ===; ls -la {REMOTE_BENCH}/.loop/results/ 2>/dev/null; "
        f"echo === heartbeat ===; ls -la {REMOTE_BENCH}/.loop/heartbeat/ 2>/dev/null; "
        f"echo === dead ===; ls -la {REMOTE_BENCH}/.loop/dead/ 2>/dev/null"
    )
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode


def _utc_now() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _new_batch_id() -> str:
    return "b" + datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sync-code", help="rsync local bench code to gpfs")
    s.set_defaults(func=cmd_sync_code)

    s = sub.add_parser("submit", help="submit a batch spec")
    s.add_argument("--batch-id", default=None)
    s.add_argument("--tasks", nargs="+", required=True)
    s.add_argument("--skip-smoke", action="store_true")
    s.set_defaults(func=cmd_submit)

    s = sub.add_parser("wait", help="poll for done flag")
    s.add_argument("batch_id")
    s.add_argument("--timeout", type=int, default=7200, help="overall timeout seconds")
    s.add_argument("--interval", type=int, default=10)
    s.add_argument("--stale-after", type=int, default=300, help="heartbeat staleness limit")
    s.set_defaults(func=cmd_wait)

    s = sub.add_parser("collect", help="rsync results back + print summary")
    s.add_argument("batch_id")
    s.set_defaults(func=cmd_collect)

    s = sub.add_parser("run", help="sync-code + submit + wait + collect")
    s.add_argument("--batch-id", default=None)
    s.add_argument("--tasks", nargs="+", required=True)
    s.add_argument("--skip-smoke", action="store_true")
    s.add_argument("--timeout", type=int, default=7200)
    s.add_argument("--interval", type=int, default=10)
    s.add_argument("--stale-after", type=int, default=300)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("status", help="list jobs/done/results on gpfs")
    s.set_defaults(func=cmd_status)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
