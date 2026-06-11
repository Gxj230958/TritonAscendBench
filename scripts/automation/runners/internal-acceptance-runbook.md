# INTERNAL: Three-machine acceptance runbook

> **This document is INTERNAL.** It lives under `scripts/automation/`,
> which is export-ignored in `.gitattributes`, so it is automatically
> excluded from the bench release tarball and the Docker build context.

This runbook is the operator's checklist for bringing the three-machine
automation stack online and running the acceptance gate. It is the
operational complement to the three-machine automation design (see git
history for the original plan document).

## 1. Roles and network topology

| Machine | Role | Bench code? | Key binary |
|---------|------|-------------|------------|
| A       | GPU bench host | Yes (full checkout) | `bench_evaluator_daemon.py` |
| B       | CPU orchestrator host | Yes (full checkout) | `fix_orchestrator.py` |
| C       | LLM serving host | No (user-owned) | OpenAI-compatible HTTP service |

Network flow (plain text):

```
                       SSH key-based auth
   +---------+        (BatchMode, no pwd)        +---------+
   | B (CPU) | -------------------------------> | A (GPU) |
   |  orch   |    via shared GPFS queue dir     |  daemon |
   +---------+                                  +---------+
        |                                            |
        |  ssh -N -L 8080:localhost:8000             |
        v                                            v
   localhost:8080  ====>  C (LLM HTTP)         /gpfs/.../queue
                                                /gpfs/.../results
                                                /gpfs/.../state
                                                /gpfs/.../logs
```

GPFS is the canonical rendezvous between A and B; B never SSHes into A
to launch the daemon, and A never opens a network socket to B. The SSH
tunnel from B to C is **the only persistent network connection**.

## 2. One-time setup checklist

Run each script once per machine, while logged in as the operator user.

### 2.1 Machine A (GPU host)

```bash
ssh <a-host>
cd /local/triton-ascend-bench   # full bench checkout
bash scripts/automation/runners/setup_machine_a_gpu.sh \
    --gpfs-root /gpfs/shared/tab_automation
```

Verifies torch >= 2.9.1, triton >= 3.5.1, CUDA visible, GPFS writable,
`pip install -e ".[automation,dev,gpu]"`, and that the daemon CLI
imports cleanly. Exits non-zero on the first failed check.

### 2.2 Machine B (CPU orchestrator host)

```bash
ssh <b-host>
cd /local/triton-ascend-bench   # full bench checkout
bash scripts/automation/runners/setup_machine_b_cpu.sh \
    --gpfs-root /gpfs/shared/tab_automation \
    --a-host <a-host> \
    --c-host <c-host>
```

Verifies Python >= 3.11, git, SSH BatchMode reachability of both A and
C, GPFS writability, `pip install -e ".[automation,dev]"`, and that
`fix_orchestrator.py` plus `automation.llm_client.OpenAIChatClient`
import cleanly.

### 2.3 Machine C (LLM host)

Operator-owned. Out of scope for this repo: bench does not ship the
LLM. The only contract C must honour is exposing an OpenAI-compatible
HTTP API on the port supplied via `--c-port` -- in particular both
`/v1/models` (used as the liveness probe) and `/v1/chat/completions`
(used by `automation.llm_client.LLMClient`).

## 3. Startup order

Bring the stack up in this order; tearing it down runs the steps in
reverse.

1. **C (LLM)** -- operator starts the LLM HTTP service. Verify with
   `curl http://localhost:<c-port>/v1/models` from machine C; any
   2xx / 3xx / 4xx response indicates the server is live (5xx and
   connection refused / timeout count as down).
2. **A (daemon)** -- on A, start the long-running daemon:
   ```bash
   bash scripts/automation/runners/run_machine_a_daemon.sh \
       --gpfs-root /gpfs/shared/tab_automation
   ```
   The runner acquires an exclusive `flock` on
   `<gpfs-root>/.daemons/<hostname>.lock` (single-daemon-per-host
   invariant) and writes the daemon PID to
   `<gpfs-root>/.daemons/<hostname>.pid`. A second concurrent
   invocation on the same host fails fast with rc=6 ("another daemon
   already holds the host lock"). The PID file is only removed by the
   instance that wrote it. Logs go to
   `<gpfs-root>/logs/daemon-<host>-<ts>.log`.

3. **B (orchestrator + tunnel)** -- on B, start the orchestrator:
   ```bash
   bash scripts/automation/runners/run_machine_b_orchestrator.sh \
       --gpfs-root /gpfs/shared/tab_automation \
       --c-host <c-host> --c-port 8000 \
       --resolve-log build/reports/resolver-run.jsonl \
       --apply
   ```

   ### 3.3 Orchestrator runner behaviour

   The runner enforces these guarantees:

   * **Resolve-log preflight.** `--resolve-log` must point to a
     non-empty file *before* the tunnel opens. Empty / missing log
     forces immediate exit rc=2 (avoids the silent no-op where the
     orchestrator's `parse_resolve_log()` returns `[]`).
   * **SSH tunnel probe.** Opens an SSH local-port-forward to C and
     probes `http://localhost:<local-port><health-path>` ten times,
     one second apart. The default `--health-path` is `/v1/models`
     (OpenAI-compatible). Any HTTP response code in `[200, 500)`
     marks the tunnel alive; 5xx and connection refused / timeout
     count as down. If no successful probe arrives, the runner exits
     rc=7 without launching the orchestrator.
   * **Supervised children.** Both the SSH tunnel and the orchestrator
     run as backgrounded children of the runner. The runner loops
     `while both alive; sleep 5; done`; the moment either dies, the
     runner sends `SIGTERM` to the other and exits with the
     orchestrator's exit code (or rc=7 when the tunnel died first
     and the orchestrator exited 0 -- forcing a non-zero rc so
     external automation reacts).
   * **Clean shutdown.** `SIGTERM` / `SIGINT` / `EXIT` traps signal
     both children and `wait` for them, no `SIGKILL`.

## 4. Acceptance gate

The full four-phase gate is run from B:

```bash
bash scripts/automation/runners/run_acceptance_gate.sh \
    --gpfs-root /gpfs/shared/tab_automation \
    --c-host <c-host> --c-port 8000 \
    --resolve-log build/reports/resolver-run.jsonl \
    --phase all
```

The runner writes a dated report to
`build/reports/internal-acceptance-report-YYYYMMDD.md` with one stanza
per phase, plus a final `Overall: PASS|FAIL` line and an LLM
cost/token stanza per phase (cost stanza degrades to
"not tracked" when `BENCH_AUTOMATION_BUDGET_JSON` was not honoured by
the orchestrator run).

* **Phase 1** -- 3 `shapes-incomplete` tasks, single-machine Tier 1;
  expect at least one task to reach `DONE`. State lives at
  `<gpfs-root>/state-phase1/`.
* **Phase 2** -- stratified 5 + 5 + 5 mixed corpus (5 `shapes-incomplete`
  + 5 `oracle-mismatch` + up to 5 `upstream-not-invocable-here`; the
  third stratum is honestly truncated to the real corpus count when
  fewer than 5 are available); expect at least one `DONE` plus at
  least one `BACKLOG_PENDING_GOLDEN` transition (the defer-to-golden
  path). State lives at `<gpfs-root>/state-phase2/`.
* **Phase 3** -- full 101-task corpus, then `backlog_report.py` +
  `summary_metrics.py`; expect all four dated outputs
  (`backlog-report-<ISO_DATE>.{md,json}` and
  `summary-metrics-<ISO_DATE>.{md,json}`) to be non-empty. State
  lives at `<gpfs-root>/state-phase3/`.
* **Phase 4** -- 5-task Tier-1 vs Tier-2 spike. Tier 1 runs in the
  primary checkout; Tier 2 runs inside a throw-away `git worktree`
  at `<gpfs-root>/.phase4-worktree` so the two tiers cannot
  cross-contaminate the `tasks/<level>/<task>/meta.yaml` tree.
  Equivalence is asserted on canonical `state` plus `verdict_class`
  agreement for tasks shared by both runs. Per-tier state lives at
  `<gpfs-root>/state-phase4-tier1/` and `<gpfs-root>/state-phase4-tier2/`.

Run individual phases via `--phase 1|2|3|4` when iterating. Each
phase still gets its own dedicated state root, so a `--phase 3` run
never observes `DONE` counts from a prior `--phase 1` run.

The acceptance gate opens **one** SSH tunnel for the whole run --
phase 2/3/4 reuse it. The EXIT / SIGTERM / SIGINT trap closes the
tunnel and, if Phase 4 ran, removes the worktree at
`<gpfs-root>/.phase4-worktree`.

## 5. Troubleshooting

| Symptom | Where to look | Recovery |
|---------|---------------|----------|
| Daemon lease expired (queue entries stuck) | `<gpfs>/logs/daemon-*.log` | Orchestrator's main loop detects stale leases and re-queues the job; no operator action needed. If repeated, restart daemon. |
| SSH tunnel dropped mid-run | `[warn] SSH tunnel died` in orchestrator runner log | `run_machine_b_orchestrator.sh` watches the tunnel and the orchestrator in a single supervisor loop; either process exiting triggers a SIGTERM to the other. Restart the runner; it reopens the tunnel and re-probes `/v1/models`. |
| LLM 5xx burst | orchestrator log shows `llm_client` retry messages | `automation.llm_client` applies exponential backoff up to its retry budget; only after the budget is exhausted does the task transition to `ATTEMPTING_NEXT_ROUND`. |
| LLM 4xx on health probe (e.g. 401 unauthorized) | runner log line `[ok] LLM endpoint reachable (http=...)` | Expected behaviour: `/v1/models` returning 4xx still proves the tunnel forwarded to a live server. Adjust `--health-path` if the server lacks `/v1/models`. |
| `GPFS mount disappeared` | both daemon and orchestrator fail loud at startup | Re-mount, then re-run the appropriate `setup_machine_*.sh` to re-probe writability, then re-launch. |
| Daemon refuses to start (`flock` busy) | `[fail] another daemon already holds the host lock` | Another invocation on the same host already holds `<gpfs>/.daemons/<host>.lock`. Identify with `lsof <lockfile>` (or `fuser`), terminate cleanly, then re-launch. Manual cleanup of the lock file is unnecessary -- `flock` is process-owned and releases on exit. |
| Daemon refuses to start (no `flock`) | `[fail] flock(1) is required` | Install `util-linux` (provides `flock(1)`). The single-daemon-per-host invariant is enforced by `flock`, not by the legacy stale-PID check. |
| Orchestrator can't find resolver log | `[fail] --resolve-log ... is missing or empty` | Resolver log must be produced ahead of time via `scripts/resolve_upstream_entrypoints.py`; this is a prerequisite, not produced by the orchestrator itself. The runner now preflight-rejects empty / missing logs *before* opening the tunnel. |
| Phase 4 fails with `git worktree add failed` | `<gpfs>/logs/phase4-tier2-*.log` | `git worktree` requires a clean parent checkout. Check for stale worktrees with `git worktree list`; if `<gpfs-root>/.phase4-worktree` exists from a previous aborted run, `git worktree remove --force <path>`. |

## 6. Shutdown procedure

Tear down in the reverse of startup. Every signal is `SIGTERM`, never
`SIGKILL`; the orchestrator and daemon do best-effort termination.

1. On B, send `SIGTERM` to the orchestrator runner (Ctrl-C in the
   foreground shell, or `kill -TERM <pid>`). The runner trap then:
   * Sends `SIGTERM` to the orchestrator process. The Python
     orchestrator currently has **no in-process graceful-drain signal
     handler**, so it terminates on best-effort -- an in-flight
     reverify enqueue or per-task transition may be lost. State
     corruption is prevented by the orchestrator's atomic
     `write_state()` path: every state mutation goes through
     `atomic.atomic_write_json()` (write-temp + fsync + rename), so the
     on-disk `triage_state.json` is never half-written. If the
     terminated transition left a task in an in-progress canonical
     state, the next launch's resilient-resume path (per-phase state
     root + corrupt-state quarantine) reconciles cleanly.
   * Sends `SIGTERM` to the SSH tunnel.
   * Sends `SIGTERM` to the operator-side `tail -f` log follower.
   * Waits for all three to exit.
   * Returns the orchestrator's rc as the script's exit code.
2. On A, send `SIGTERM` to the daemon runner. The trap forwards the
   signal to the Python daemon, which finishes its current claim
   before exiting. The PID file is removed only if it still records
   *this* invocation's daemon PID; the host `flock` is released
   implicitly on process exit.
3. On C, stop the LLM service (operator-owned).
4. (Optional) On any machine, inspect `<gpfs>/logs/` for the
   `daemon-*.log` and `orchestrator-*.log` files from the run, plus
   `<gpfs>/budget/phase*-<DATE>.json` for per-phase LLM cost
   summaries (when the orchestrator produced them).

> *Note (out of scope).* A true graceful-drain handler inside
> `fix_orchestrator.py` -- one that traps `SIGTERM`, lets the current
> task-transition finish, then flushes state -- would be a follow-up
> in its own right. Today's contract is "atomic per-write durability
> plus idempotent resume", which is sufficient for the acceptance gate
> but does not guarantee zero work loss on signal.

## 7. Acceptance criteria

The acceptance gate is considered PASSED only when every phase that
ran reports `PASS`. The per-phase pass conditions are:

1. **Phase 1** -- `PASS` iff orchestrator rc=0 AND >= 1 DONE on 3
   `shapes-incomplete` tasks.
2. **Phase 2** -- `PASS` iff orchestrator rc=0 AND >= 1 DONE AND >= 1
   `BACKLOG_PENDING_GOLDEN` over the stratified 5+5+(up to 5) sample.
   "Up to 5" reflects the real corpus distribution: the
   `upstream-not-invocable-here` bucket can be smaller than 5 -- the
   runner does not back-fill from a different class.
3. **Phase 3** -- `PASS` iff orchestrator rc=0 AND backlog-report rc=0
   AND summary-metrics rc=0 AND all four dated outputs non-empty:
   `backlog-report-<ISO_DATE>.{md,json}` +
   `summary-metrics-<ISO_DATE>.{md,json}`.
4. **Phase 4** -- `PASS` iff tier-1 rc=0 AND tier-2 rc=0 AND
   `>= 5` tasks shared by both runs agree on canonical `state` +
   `verdict_class`. Stdout / stderr differences are allowed; only the
   state-file equivalence is checked.

The dated report
(`build/reports/internal-acceptance-report-YYYYMMDD.md`) plus the
per-phase budget JSONs under `<gpfs-root>/budget/` are the
artefact-of-record for handoff to the bench team operator.
