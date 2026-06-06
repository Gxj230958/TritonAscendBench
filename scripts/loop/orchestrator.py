"""dev side: submit / wait / collect batches against the A-side daemon.

The dev box cannot see gpfs directly. All file movement goes through
the reverse SSH tunnel on 127.0.0.1:2222 -> root@<B> which has gpfs
mounted at the same path as A.

Subcommands:
  sync-code        rsync local working tree (tasks/ scripts/ pyproject.toml +
                   tasks/_frozen_shim) to gpfs so the daemon sees latest code
  submit           write a batch spec to <loop_mailbox>/jobs/<id>.json
  wait             poll <loop_mailbox>/done/<id>.flag (heartbeat-aware)
  collect          rsync <loop_mailbox>/results/<id>/ back to local
  run              sync-code + submit + wait + collect in one shot
  status           list jobs / running / done on gpfs

Convention:
  REMOTE_BENCH = /inspire/hdd/.../kernel/TritonAscendBench   (overridable env: LOOP_REMOTE_BENCH)
  REMOTE_LOOP  = /inspire/hdd/.../kernel/loop_mailbox        (overridable env: LOOP_MAILBOX_REMOTE)
  LOCAL_LOOP   = /data1/code/.../kernel/loop_mailbox         (overridable env: LOOP_MAILBOX_LOCAL)
  SSH = ssh -p 2222 root@127.0.0.1

The loop mailbox lives OUTSIDE the bench repo (sibling of TritonAscendBench/)
so runtime artifacts cannot pollute git status or get clobbered by
`git clean`. Both A's gpfs view and dev's local view default to the
same relative location.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import time
from pathlib import Path

# Strict whitelist (kept in sync with daemon_a.py).
_BATCH_ID_RE = re.compile(r"^b[A-Za-z0-9_\-]{1,63}$")
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

LOCAL_BENCH = Path(__file__).resolve().parents[2]
# Loop mailbox lives OUTSIDE the bench repo (decouple runtime state).
# Both sides default to <kernel>/loop_mailbox/ -- on dev that's
# /data1/code/agentic-rl/kernel/loop_mailbox/, on A's gpfs view that's
# /inspire/hdd/.../kernel/loop_mailbox/. Override via env var without
# editing the script.
LOCAL_LOOP = Path(
    os.environ.get("LOOP_MAILBOX_LOCAL") or (LOCAL_BENCH.parent / "loop_mailbox")
)
REMOTE_BENCH = os.environ.get(
    "LOOP_REMOTE_BENCH",
    "/inspire/hdd/project/qianghuaxuexi/liuliming-liuliming/kernel/TritonAscendBench",
)
REMOTE_LOOP = os.environ.get(
    "LOOP_MAILBOX_REMOTE",
    "/inspire/hdd/project/qianghuaxuexi/liuliming-liuliming/kernel/loop_mailbox",
)
SSH_PORT = "2222"
SSH_HOST = "root@127.0.0.1"
SSH_RSH = f"ssh -p {SSH_PORT} -o BatchMode=yes -o StrictHostKeyChecking=no"


def _ssh(cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-p", SSH_PORT, "-o", "BatchMode=yes",
         "-o", "StrictHostKeyChecking=no", SSH_HOST, cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def _rsync(
    src: str,
    dst: str,
    *,
    delete: bool = False,
    exclude: list[str] | None = None,
    include_only_meta: bool = False,
    update: bool = False,
) -> int:
    # ``--update`` skips files whose destination mtime is newer than the
    # source. Use this for the resolver round-trip ONLY (sync-code +
    # pull-meta), where a freshly-resolved meta.yaml on either side must
    # not be clobbered by an older counterpart. Other transfers (job spec
    # push at submit, results pull at collect) intentionally keep the
    # default overwrite-receiver behavior so a stale recycled job spec
    # or a stale local result JSON does NOT silently win.
    args = ["rsync", "-az", "--info=stats1,progress0", "-e", SSH_RSH]
    if update:
        args.append("--update")
    if delete:
        args.append("--delete")
    if include_only_meta:
        # Pull-meta narrows rsync to ONLY tasks/level*/<task>/meta.yaml. This
        # is the cmd_pull_meta path (gpfs -> dev) where we want resolver edits
        # but absolutely nothing else from the daemon's working tree -- no
        # stale shaped artifacts, no cache directories, no __pycache__.
        args.extend([
            "--include=*/",
            "--include=meta.yaml",
            "--exclude=*",
        ])
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
        "infra/",          # peak tables (cost-model Roofline needs these on GPU A)
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
        # .loop/ excluded as defense-in-depth: mailbox now lives at
        # <bench_parent>/loop_mailbox/ outside bench, but if anyone
        # accidentally creates a local .loop/ in bench it stays local.
        # update=True so a freshly-resolved meta.yaml on gpfs is NOT
        # clobbered by an older dev-side copy during a sync-code call
        # that races with an in-flight resolve batch.
        rc = _rsync(src, dst, exclude=[".loop/", "__pycache__/", "*.pyc"],
                    update=True)
        rc_total |= rc
    return rc_total


def cmd_submit(args: argparse.Namespace) -> int:
    """Write a batch spec into local loop_mailbox/jobs/ + rsync to gpfs."""
    batch_id = args.batch_id or _new_batch_id()
    _validate_batch_id(batch_id)
    for t in args.tasks:
        _validate_task_id(t)

    # Codex FAIL #2: refuse to reuse a batch_id whose artifacts already
    # exist on either side. Otherwise an old done.flag can mask a fresh
    # failure or silently overwrite prior results.
    if not args.allow_recycle:
        local_collisions: list[str] = []
        # Static paths.
        for name, sub in (
            (f"{batch_id}.json", "jobs"),
            (f"{batch_id}.json.taken", "jobs"),
            (f"{batch_id}.flag", "done"),
            (batch_id, "results"),
        ):
            p = LOCAL_LOOP / sub / name
            if p.exists():
                local_collisions.append(str(p))
        # dead/<id>.err  AND  dead/<id>.<reason>.err  variants (daemon
        # writes e.g. .duplicate.err / .heartbeat.err / bare .err).
        for p in (LOCAL_LOOP / "dead").glob(f"{batch_id}.err"):
            local_collisions.append(str(p))
        for p in (LOCAL_LOOP / "dead").glob(f"{batch_id}.*.err"):
            local_collisions.append(str(p))
        if local_collisions:
            print(
                f"batch_id {batch_id} already used locally: {local_collisions}. "
                f"Use a fresh id, or pass --allow-recycle (dangerous).",
                file=sys.stderr,
            )
            return 1
        # Also probe remote, including the same dead/<id>{,.*}.err variants.
        # Each test line ends with `|| true` so a single failed test does NOT
        # abort the probe; the script collects echoes from successful checks.
        # NOTE: do NOT use `ls dead/<id>.*.err | head -1 && echo` to detect
        # the suffixed variant -- `head` exits 0 on empty stdin, so the &&
        # echo always fires and reports every fresh batch_id as a collision.
        # POSIX `find -print -quit` gives "match exists?" without that trap.
        bid = shlex.quote(batch_id)
        loop = shlex.quote(REMOTE_LOOP)
        bid_inner = batch_id  # already validated by regex; safe for -name glob
        probe = _ssh(
            f"test -f {loop}/jobs/{bid}.json && echo JOB || true; "
            f"test -f {loop}/jobs/{bid}.json.taken && echo TAKEN || true; "
            f"test -f {loop}/done/{bid}.flag && echo DONE || true; "
            f"test -d {loop}/results/{bid} && echo RESULTS || true; "
            f"test -f {loop}/dead/{bid}.err && echo DEAD_BARE || true; "
            f"[ -n \"$(find {loop}/dead -maxdepth 1 -name '{bid_inner}.*.err' -print -quit 2>/dev/null)\" ] "
            f"&& echo DEAD_REASONED || true"
        )
        remote_collision = probe.stdout.strip()
        if remote_collision:
            print(
                f"batch_id {batch_id} already exists remote: {remote_collision}. "
                f"Use a fresh id.",
                file=sys.stderr,
            )
            return 1

    spec = {
        "batch_id": batch_id,
        "submitted_at": _utc_now(),
        "tasks": args.tasks,
        "accept_stale_cache": True,
        "require_smoke": bool(args.require_smoke),
        # action defaults to "verify"; resolve subcommand sets it to "resolve"
        # to dispatch phase_a_prime --resolve-entrypoint instead.
        "action": getattr(args, "action", "verify"),
    }
    (LOCAL_LOOP / "jobs").mkdir(parents=True, exist_ok=True)
    spec_path = LOCAL_LOOP / "jobs" / f"{batch_id}.json"
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    print(f"local spec: {spec_path}")

    # Push to gpfs.
    rc = _rsync(
        str(spec_path),
        f"{SSH_HOST}:{REMOTE_LOOP}/jobs/{batch_id}.json",
    )
    if rc != 0:
        print(f"rsync FAILED rc={rc}", file=sys.stderr)
        return rc
    print(f"submitted batch_id={batch_id} tasks={args.tasks}")
    print(batch_id)  # last line is just the batch_id so callers can capture
    return 0


def cmd_wait(args: argparse.Namespace) -> int:
    """Poll <gpfs>/loop_mailbox/done/<batch>.flag with heartbeat-staleness guard.

    Exits:
      0 = batch completed (done.flag present)
      2 = heartbeat stale / never appeared / timeout
      3 = dead/<batch>.err present (daemon crashed mid-batch)
      4 = ssh tunnel failure (transport error)
    """
    batch_id = args.batch_id
    _validate_batch_id(batch_id)
    timeout = args.timeout
    interval = args.interval
    stale_after = args.stale_after
    start = time.time()
    last_print = 0.0
    bid = shlex.quote(batch_id)
    loop = shlex.quote(REMOTE_LOOP)
    probe_cmd = (
        f"set -e; "
        f"if ls {loop}/dead/{bid}.*.err 2>/dev/null; then exit 0; fi; "
        f"if test -f {loop}/dead/{bid}.err; then echo DEAD; exit 0; fi; "
        f"if test -f {loop}/done/{bid}.flag; then echo DONE; exit 0; fi; "
        f"stat -c '%Y' {loop}/heartbeat/{bid}.txt 2>/dev/null || echo MISSING"
    )
    while time.time() - start < timeout:
        r = _ssh(probe_cmd)
        if r.returncode != 0:
            print(
                f"{batch_id}: ssh probe rc={r.returncode}: "
                f"{(r.stderr or '').strip()[:200]}",
                file=sys.stderr,
            )
            return 4
        out = r.stdout.strip()
        # dead/<id>.*.err lines are listed first; if any present, fail fast.
        if ".err" in out:
            print(
                f"{batch_id}: DEAD (see {out.splitlines()[0]})",
                file=sys.stderr,
            )
            return 3
        if "DEAD" in out:
            print(f"{batch_id}: DEAD (dead/{batch_id}.err present)", file=sys.stderr)
            return 3
        if "DONE" in out:
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
    """rsync <gpfs>/loop_mailbox/results/<batch>/ back to local; print verdict summary."""
    batch_id = args.batch_id
    _validate_batch_id(batch_id)
    local_dir = LOCAL_LOOP / "results" / batch_id
    local_dir.mkdir(parents=True, exist_ok=True)
    rc = _rsync(
        f"{SSH_HOST}:{REMOTE_LOOP}/results/{batch_id}/",
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
    """sync-code + submit + wait + collect, one shot.

    Exit-code mapping (from cmd_wait):
      0 → success path: collect normally
      3 → DEAD: still collect partial results so the operator sees
          what made it before the crash, then propagate exit 3
      2 (stale/timeout) / 4 (ssh transport) → skip collect (no
          confidence in remote state), propagate as-is
    """
    rc = cmd_sync_code(args)
    if rc != 0:
        return rc
    batch_id = args.batch_id or _new_batch_id()
    args.batch_id = batch_id
    rc = cmd_submit(args)
    if rc != 0:
        return rc
    wait_rc = cmd_wait(args)
    if wait_rc == 0:
        rc = cmd_collect(args)
        # P1.2 drift-kill: the verify (S2) action writes phase_a_prime.status +
        # standalone_sha256 into the daemon-side (gpfs) meta. Auto pull-meta so
        # the dev source of truth never goes stale (the 23-vs-101 bug). Run the
        # C2 apply AFTER this (it writes provenance.golden on top of the fresh
        # S2 status); pull-meta uses --update so a locally-newer meta is kept.
        cmd_pull_meta(args)
        return rc
    if wait_rc == 3:
        # Partial results may still be on gpfs; pull them so the operator
        # can diagnose, then propagate the DEAD signal.
        print(f"{batch_id}: collecting partial results before exit (DEAD)")
        cmd_collect(args)
        return 3
    return wait_rc


def cmd_resolve(args: argparse.Namespace) -> int:
    """submit a batch with spec.action='resolve'.

    The daemon dispatches each task to ``phase_a_prime --resolve-entrypoint
    --task <id> --apply`` instead of ``--verify``. The resolver writes the
    discovered upstream entrypoint to meta.yaml on the daemon's working tree
    (= gpfs). Use ``pull-meta`` afterwards to rsync those edits back to dev
    so the dev-side source of truth picks them up.
    """
    args.action = "resolve"
    # Resolver doesn't take --require-smoke, but cmd_submit reads it via
    # getattr; provide a stable default so the spec field stays consistent.
    if not hasattr(args, "require_smoke"):
        args.require_smoke = False
    return cmd_submit(args)


def cmd_pull_meta(args: argparse.Namespace) -> int:
    """Rsync resolved tasks/level*/<task>/meta.yaml files gpfs -> dev.

    Mirror direction of sync-code: that pushes dev -> gpfs and is the normal
    workflow direction. Resolver writes on the gpfs side (the daemon's
    cwd), so a one-time pull is needed to surface those edits on dev.
    Only meta.yaml files are pulled to keep the blast radius minimal --
    no other gpfs-side artifact can sneak into the dev tree this way.
    """
    bench_root = Path(__file__).resolve().parents[2]
    dest = bench_root / "tasks/"
    src = f"{SSH_HOST}:{REMOTE_BENCH}/tasks/"
    # update=True so a meta.yaml the operator just edited locally is NOT
    # clobbered by a stale gpfs copy from before the resolver ran.
    rc = _rsync(
        src,
        str(dest),
        delete=False,
        exclude=None,
        include_only_meta=True,
        update=True,
    )
    if rc == 0:
        print(f"pulled resolved meta.yaml files into {dest}")
    return rc


def cmd_run_resolve(args: argparse.Namespace) -> int:
    """sync-code + resolve + wait + collect + pull-meta, one shot.

    Same exit-code mapping as cmd_run; on DEAD/wait failure we still try to
    pull whatever meta.yaml the resolver managed to apply before the crash.
    """
    rc = cmd_sync_code(args)
    if rc != 0:
        return rc
    batch_id = args.batch_id or _new_batch_id()
    args.batch_id = batch_id
    rc = cmd_resolve(args)
    if rc != 0:
        return rc
    wait_rc = cmd_wait(args)
    if wait_rc == 0:
        rc = cmd_collect(args)
        if rc != 0:
            return rc
        return cmd_pull_meta(args)
    if wait_rc == 3:
        print(f"{batch_id}: collecting partial results before exit (DEAD)")
        cmd_collect(args)
        cmd_pull_meta(args)
        return 3
    return wait_rc


def cmd_oracle(args: argparse.Namespace) -> int:
    """submit a batch with spec.action='oracle-equiv' (C2 precision align).

    The daemon runs R10.a precompute then R10.b GPU compare per task on the
    same host (H2 input parity), writing TaskVerdict JSON to
    results/<batch>/r10/<task>.json. No pull-meta: the R10.c writeback to
    meta.yaml is a separate dev-side apply step, not a gpfs edit.
    """
    args.action = "oracle-equiv"
    if not hasattr(args, "require_smoke"):
        args.require_smoke = False
    return cmd_submit(args)


def cmd_run_oracle(args: argparse.Namespace) -> int:
    """sync-code + oracle-equiv + wait + collect, one shot."""
    rc = cmd_sync_code(args)
    if rc != 0:
        return rc
    batch_id = args.batch_id or _new_batch_id()
    args.batch_id = batch_id
    rc = cmd_oracle(args)
    if rc != 0:
        return rc
    wait_rc = cmd_wait(args)
    if wait_rc == 0:
        return cmd_collect(args)
    if wait_rc == 3:
        print(f"{batch_id}: collecting partial results before exit (DEAD)")
        cmd_collect(args)
        return 3
    return wait_rc


def cmd_gpu_grade(args: argparse.Namespace) -> int:
    """submit a batch with spec.action='gpu-grade' (candidate-grade GPU smoke).

    The daemon runs a real triton kernel through the process_isolation sandbox
    per task on the CUDA host, writing a verdict JSON to
    results/<batch>/gpu_grade/<task>.json. Validates the GPU-backend sandbox
    path (CUDA pre-init, warmup-before-hook JIT, save-path warm, event timer).
    """
    args.action = "gpu-grade"
    if not hasattr(args, "require_smoke"):
        args.require_smoke = False
    return cmd_submit(args)


def cmd_run_gpu_grade(args: argparse.Namespace) -> int:
    """sync-code + gpu-grade + wait + collect, one shot."""
    rc = cmd_sync_code(args)
    if rc != 0:
        return rc
    batch_id = args.batch_id or _new_batch_id()
    args.batch_id = batch_id
    rc = cmd_gpu_grade(args)
    if rc != 0:
        return rc
    wait_rc = cmd_wait(args)
    if wait_rc == 0:
        return cmd_collect(args)
    if wait_rc == 3:
        print(f"{batch_id}: collecting partial results before exit (DEAD)")
        cmd_collect(args)
        return 3
    return wait_rc


def cmd_gpu_rank(args: argparse.Namespace) -> int:
    """submit a batch with spec.action='gpu-rank' (full GPU ranking per task).

    The daemon constructs a clean triton candidate from each task's severed
    kernel.py and grades it through the GPU sandbox + cost-model, writing a
    verdict JSON (correctness + s_t + per-shape e) to
    results/<batch>/gpu_rank/<task>.json — i.e. whether each task RANKS on H200.
    """
    args.action = "gpu-rank"
    if not hasattr(args, "require_smoke"):
        args.require_smoke = False
    return cmd_submit(args)


def cmd_run_gpu_rank(args: argparse.Namespace) -> int:
    """sync-code + gpu-rank + wait + collect, one shot."""
    rc = cmd_sync_code(args)
    if rc != 0:
        return rc
    batch_id = args.batch_id or _new_batch_id()
    args.batch_id = batch_id
    rc = cmd_gpu_rank(args)
    if rc != 0:
        return rc
    wait_rc = cmd_wait(args)
    if wait_rc == 0:
        return cmd_collect(args)
    if wait_rc == 3:
        print(f"{batch_id}: collecting partial results before exit (DEAD)")
        cmd_collect(args)
        return 3
    return wait_rc


def cmd_status(args: argparse.Namespace) -> int:
    loop = shlex.quote(REMOTE_LOOP)
    # Single-quote the section markers: the remote login shell may be zsh,
    # whose EQUALS expansion treats a bare `===` word as a `=cmd` lookup and
    # fails with "== not found". Quoting is safe under both bash and zsh.
    r = _ssh(
        f"echo '=== jobs ==='; ls -la {loop}/jobs/ 2>/dev/null; "
        f"echo '=== done ==='; ls -la {loop}/done/ 2>/dev/null; "
        f"echo '=== results ==='; ls -la {loop}/results/ 2>/dev/null; "
        f"echo '=== heartbeat ==='; ls -la {loop}/heartbeat/ 2>/dev/null; "
        f"echo '=== dead ==='; ls -la {loop}/dead/ 2>/dev/null"
    )
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode


def _utc_now() -> str:
    # `isoformat() + "Z"` would emit invalid `...+00:00Z`; use strftime for `...Z`.
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_batch_id() -> str:
    # Higher-entropy id: timestamp + 6 random hex chars (Codex FAIL #2 fix).
    # Reduces accidental collisions when two submits land within the same
    # second on busy days.
    return (
        "b"
        + datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S")
        + "-"
        + secrets.token_hex(3)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sync-code", help="rsync local bench code to gpfs")
    s.set_defaults(func=cmd_sync_code)

    s = sub.add_parser("submit", help="submit a batch spec")
    s.add_argument("--batch-id", default=None)
    s.add_argument("--tasks", nargs="+", required=True)
    s.add_argument(
        "--require-smoke",
        action="store_true",
        help="Force S2-smoke (CPU/TRITON_INTERPRET) pre-filter to run on "
        "the GPU host. Default skips it because the GPU run is the "
        "authoritative verdict and uses the same compare path.",
    )
    s.add_argument(
        "--allow-recycle",
        action="store_true",
        help="DANGEROUS: allow reusing a batch_id whose artifacts already "
        "exist (may overwrite results / mask success-flag race).",
    )
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
    s.add_argument("--require-smoke", action="store_true")
    s.add_argument("--allow-recycle", action="store_true")
    s.add_argument("--timeout", type=int, default=7200)
    s.add_argument("--interval", type=int, default=10)
    s.add_argument("--stale-after", type=int, default=300)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("status", help="list jobs/done/results on gpfs")
    s.set_defaults(func=cmd_status)

    # ---- resolver subcommands ---------------------------------------
    # Resolver path mirrors the verify path (submit + wait + collect) but
    # tells the daemon to run phase_a_prime --resolve-entrypoint instead of
    # --verify, then rsyncs the resolved meta.yaml files BACK from gpfs to
    # dev so the dev-side source-of-truth picks up the daemon's changes.
    s = sub.add_parser("resolve",
                       help="submit a batch that runs --resolve-entrypoint")
    s.add_argument("--batch-id", default=None)
    s.add_argument("--tasks", nargs="+", required=True)
    s.add_argument("--allow-recycle", action="store_true")
    s.set_defaults(func=cmd_resolve)

    s = sub.add_parser("pull-meta",
                       help="rsync resolved meta.yaml files gpfs -> dev")
    s.set_defaults(func=cmd_pull_meta)

    s = sub.add_parser("run-resolve",
                       help="sync-code + resolve + wait + collect + pull-meta")
    s.add_argument("--batch-id", default=None)
    s.add_argument("--tasks", nargs="+", required=True)
    s.add_argument("--allow-recycle", action="store_true")
    s.add_argument("--timeout", type=int, default=7200)
    s.add_argument("--interval", type=int, default=10)
    s.add_argument("--stale-after", type=int, default=300)
    s.set_defaults(func=cmd_run_resolve)

    s = sub.add_parser("oracle",
                       help="submit a batch that runs R10 oracle-equiv (C2)")
    s.add_argument("--batch-id", default=None)
    s.add_argument("--tasks", nargs="+", required=True)
    s.add_argument("--allow-recycle", action="store_true")
    s.set_defaults(func=cmd_oracle)

    s = sub.add_parser("run-oracle",
                       help="sync-code + oracle-equiv + wait + collect (C2)")
    s.add_argument("--batch-id", default=None)
    s.add_argument("--tasks", nargs="+", required=True)
    s.add_argument("--allow-recycle", action="store_true")
    s.add_argument("--timeout", type=int, default=7200)
    s.add_argument("--interval", type=int, default=10)
    s.add_argument("--stale-after", type=int, default=300)
    s.set_defaults(func=cmd_run_oracle)

    s = sub.add_parser("gpu-grade",
                       help="submit a batch that runs the candidate-grade GPU "
                            "sandbox smoke per task")
    s.add_argument("--batch-id", default=None)
    s.add_argument("--tasks", nargs="+", required=True)
    s.add_argument("--allow-recycle", action="store_true")
    s.set_defaults(func=cmd_gpu_grade)

    s = sub.add_parser("run-gpu-grade",
                       help="sync-code + gpu-grade + wait + collect")
    s.add_argument("--batch-id", default=None)
    s.add_argument("--tasks", nargs="+", required=True)
    s.add_argument("--allow-recycle", action="store_true")
    s.add_argument("--timeout", type=int, default=3600)
    s.add_argument("--interval", type=int, default=10)
    s.add_argument("--stale-after", type=int, default=300)
    s.set_defaults(func=cmd_run_gpu_grade)

    s = sub.add_parser("gpu-rank",
                       help="submit a batch that runs each task's reference "
                            "triton kernel with device-only CUDA-event timing "
                            "and folds it into the cost-model ranking on H200")
    s.add_argument("--batch-id", default=None)
    s.add_argument("--tasks", nargs="+", required=True)
    s.add_argument("--allow-recycle", action="store_true")
    s.set_defaults(func=cmd_gpu_rank)

    s = sub.add_parser("run-gpu-rank",
                       help="sync-code + gpu-rank + wait + collect")
    s.add_argument("--batch-id", default=None)
    s.add_argument("--tasks", nargs="+", required=True)
    s.add_argument("--allow-recycle", action="store_true")
    s.add_argument("--timeout", type=int, default=3600)
    s.add_argument("--interval", type=int, default=10)
    s.add_argument("--stale-after", type=int, default=300)
    s.set_defaults(func=cmd_run_gpu_rank)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
