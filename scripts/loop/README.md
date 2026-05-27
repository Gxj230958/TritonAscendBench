# Closed-loop autonomous verification (dev ↔ B-tunnel ↔ gpfs ↔ GPU A)

## Topology

```
dev (no gpfs visibility) ──ssh -p 2222 root@127.0.0.1──> GPU B ──gpfs shared FS──> GPU A
```

- dev: my workstation. Cannot mount gpfs. All file movement goes through B.
- B: jumphost on `127.0.0.1:2222` (reverse tunnel). gpfs mounted at the same
  bench path as A. dev rsyncs into B's local path = lands on gpfs = A sees it.
- A: GPU host where `phase_a_prime --verify` actually executes. No GitHub
  auth on A; we never `git pull` on A. Code arrives via dev → B rsync.

## File-based mailbox

Everything under `<bench>/.loop/` on gpfs (which is `.gitignore`'d):

```
.loop/
  jobs/<batch_id>.json       dev writes; A claims via .taken rename
  jobs/<batch_id>.json.taken A's claim marker
  heartbeat/<batch_id>.txt   A writes UTC timestamp every 10s
  results/<batch_id>/<task>.json  A writes per-task verdict
  done/<batch_id>.flag       A writes when batch is fully drained
  dead/<batch_id>.err        A writes on crash
  daemon.log                 daemon stdout tail (optional, via tee)
```

## One-time deploy on GPU A

**Prereq**: bench tree must exist at the gpfs path
`/inspire/hdd/project/qianghuaxuexi/liuliming-liuliming/kernel/TritonAscendBench/`.
If it doesn't (first ever deploy), bootstrap from dev. The rsync command
below honors the local `.gitignore` via `--filter=dir-merge` so it does
NOT push `.venv/`, `.pytest_cache/`, `.cache/`, `external/`, build
artifacts, or `.loop/` runtime state to gpfs.

```bash
# from dev, one-time bootstrap (push bench tree to gpfs via B tunnel)
cd /data1/code/agentic-rl/kernel/triton-ascend-bench
rsync -az \
    --filter='dir-merge,- .gitignore' \
    --exclude '.git/' \
    --exclude '.loop/' \
    -e 'ssh -p 2222' \
    ./ \
    root@127.0.0.1:/inspire/hdd/project/qianghuaxuexi/liuliming-liuliming/kernel/TritonAscendBench/
```

(`.git/` is excluded explicitly because `.gitignore` itself doesn't
ignore it; `.loop/` is excluded because we don't want to clobber the
A-side runtime mailbox during bootstrap.)

Then on A:

```bash
# (on A) ssh to A from wherever you have credentials
cd /inspire/hdd/project/qianghuaxuexi/liuliming-liuliming/kernel/TritonAscendBench
mkdir -p .loop/{jobs,results,heartbeat,done,dead}
tmux new-session -d -s loop_daemon \
  'python3 scripts/loop/daemon_a.py --workers 2 2>&1 | tee -a .loop/daemon.log'
```

To stop: `tmux kill-session -t loop_daemon`.

The daemon runs whatever's currently in the on-disk working tree -- dev
syncs new code in *before* submitting a batch.

## dev side: one-shot run

```bash
# from dev
cd /data1/code/agentic-rl/kernel/triton-ascend-bench
python3 scripts/loop/orchestrator.py run \
    --tasks L1_22_min_p L1_06_reshape_and_cache_kernel_flash L1_03_convert_req_index_to_global_index
# sync-code + submit + wait + collect; prints verdict summary
```

Individual steps:

```bash
python3 scripts/loop/orchestrator.py sync-code                 # push latest scripts/ tasks/ to gpfs
python3 scripts/loop/orchestrator.py submit --tasks L1_22_min_p   # writes spec + rsyncs to gpfs
python3 scripts/loop/orchestrator.py wait b20260527T030245     # poll done flag + heartbeat
python3 scripts/loop/orchestrator.py collect b20260527T030245  # rsync results back, print summary
python3 scripts/loop/orchestrator.py status                    # ls everything on gpfs
```

## What a result JSON looks like

```json
{
  "task_id": "L1_22_min_p",
  "batch_id": "b20260527T030245",
  "started": "2026-05-27T03:02:51Z",
  "completed": "2026-05-27T03:03:18Z",
  "rc": 0,
  "verdict_status": "PASS",
  "verdict_line": "verify L1_22_min_p: PASS (s2-L1_22_min_p-...) -- bit-exact (values + alias + mutated-input)",
  "stdout_tail": "...",
  "stderr_tail": "...",
  "error": null
}
```

Verdict parsing follows the existing `verify <task>: <STATUS> -- <reason>`
format from `cmd_verify`. Status is one of: PASS, BLOCKED, UNRUN,
SKIPPED_BY_DESIGN.

## Failure modes

- **rsync auth fails**: tunnel down or key broken. Reopen `ssh -R 2222:localhost:22`
  from the remote.
- **heartbeat goes stale (>5min)**: daemon crashed or A is unresponsive.
  Restart the tmux session.
- **A daemon crashed mid-batch**: check `.loop/dead/<batch>.err`. The batch's
  partial results still in `.loop/results/<batch>/`.
- **A's git working tree drifts** (someone manually edited on A): dev's
  next `sync-code` overwrites the affected files. Manual A-side edits are
  destructive; treat A as a read-only execution target.

## Iteration pattern (Claude-in-the-loop)

The orchestrator does NOT auto-fix tasks. It moves files. Diagnosis and
patch creation are still my job. Each iteration:

1. `orchestrator.py run --tasks ...`
2. read the verdict summary that prints
3. for any FAILED / BLOCKED task: open its result JSON tail + meta + source
4. propose + apply fix (edit shapes / oracle / meta / scripts)
5. `git commit -m "..."` (history) + `git push` only after codex review pass
6. goto 1

For batched same-category passes (e.g. all `missing-shape-row` at once):
list the affected task ids in `--tasks`. The daemon runs them concurrent
(workers=2), so the wall time for N tasks is ~N/2 * avg_task_seconds.
