#!/usr/bin/env bash
# Acceptance gate driver for the three-machine automation stack.
#
# Runs the four acceptance phases in order (or one selected phase) and
# produces a dated INTERNAL acceptance report under docs/release/.
#
# Phase 1: 3 shapes-incomplete tasks, single-machine Tier 1.
#          PASS when orchestrator rc=0 AND at least one task reaches DONE.
# Phase 2: stratified 5+5+5 mixed corpus (shapes / oracle-mismatch /
#          upstream-not-invocable-here -- the third bucket may be
#          truncated to the real corpus count when fewer than 5 are
#          available). PASS when at least one DONE AND at least one
#          BACKLOG_PENDING_GOLDEN transition occur.
# Phase 3: full 101-task corpus + backlog-report + summary-metrics.
#          PASS when all four dated outputs are non-empty.
# Phase 4: 5-task Tier-1 vs Tier-2 verdict equivalence spike inside an
#          isolated worktree, so Tier 1 + Tier 2 cannot pollute the
#          shared tasks/<level>/<task>/meta.yaml tree.
#
# Each phase appends a markdown stanza to:
#   docs/release/internal-acceptance-report-YYYYMMDD.md
#
# Design choices (vs. earlier drafts):
#   * Single long-lived SSH tunnel for the whole gate (one start, one
#     stop in the EXIT trap; phase 2/3/4 reuse it).
#   * Per-phase state roots ($GPFS_ROOT/state-phase{1,2,3}, plus
#     state-tier1/state-tier2 under a Phase 4 worktree). Late phases
#     never observe DONE counts from earlier phases.
#   * Phase 3 + 4 verify dated artefact filenames
#     (backlog-report-<ISO_DATE>.{md,json}, summary-metrics-<ISO_DATE>.{md,json}).
#   * Phase 4 runs Tier 2 inside an isolated `git worktree`, so the
#     orchestrator's meta.yaml mutations do not cross-contaminate.
#   * Health probe uses the OpenAI-compatible /v1/models endpoint by
#     default (configurable via --health-path) and treats any HTTP
#     response in [200, 500) (2xx / 3xx / 4xx) as "tunnel alive" --
#     matching run_machine_b_orchestrator.sh and the runbook (5xx and
#     connection refused / timeout count as down).
#
# Usage:
#   bash scripts/automation/runners/run_acceptance_gate.sh \
#     --gpfs-root /gpfs/shared/tab_automation \
#     --c-host llm-svc-c01 --c-port 8000 \
#     --resolve-log docs/triage_backlog/resolver-run.jsonl \
#     --phase all
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_acceptance_gate.sh --gpfs-root PATH --resolve-log PATH
                              [http: --c-host SSH ...]
                              [ssh-cli: --llm-ssh-host SSH
                                        --llm-cli-cmd TPL ...]
                              [options]

Required:
  --gpfs-root PATH        Shared GPFS root.
  --resolve-log PATH      Resolver classification log (jsonl).

Transport selection:
  --llm-transport MODE    'http' (default) or 'ssh-cli'.

Required when --llm-transport http:
  --c-host SSH            SSH target for the LLM machine.

Required when --llm-transport ssh-cli:
  --llm-ssh-host SSH      SSH target for the remote CLI host.
  --llm-cli-cmd TPL       CLI command template with literal '{prompt}'.

Options (http mode):
  --c-port INT            LLM remote port (default 8000).
  --local-port INT        Local tunnel port (default 8080).
  --health-path PATH      HTTP path to probe for tunnel liveness
                          (default: /v1/models -- OpenAI-compatible).

Options (both modes):
  --state-path PATH       Override the report-side triage state file
                          path (the per-phase state roots are always
                          derived from --gpfs-root).
  --llm-timeout-seconds N Per-call transport timeout (default 600).
  --phase MODE            1|2|3|4|all  (default: all).
  -h, --help              Show this help and exit 0.
EOF
}

GPFS_ROOT=""
C_HOST=""
C_PORT="8000"
LOCAL_PORT="8080"
RESOLVE_LOG=""
STATE_PATH=""
HEALTH_PATH="/v1/models"
PHASE="all"
LLM_TRANSPORT="http"
LLM_SSH_HOST=""
LLM_CLI_CMD=""
LLM_TIMEOUT_SECONDS="600"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpfs-root)   GPFS_ROOT="${2:-}";   shift 2 ;;
    --c-host)      C_HOST="${2:-}";      shift 2 ;;
    --c-port)      C_PORT="${2:-}";      shift 2 ;;
    --local-port)  LOCAL_PORT="${2:-}";  shift 2 ;;
    --resolve-log) RESOLVE_LOG="${2:-}"; shift 2 ;;
    --state-path)  STATE_PATH="${2:-}";  shift 2 ;;
    --health-path) HEALTH_PATH="${2:-}"; shift 2 ;;
    --phase)       PHASE="${2:-}";       shift 2 ;;
    --llm-transport)        LLM_TRANSPORT="${2:-}";        shift 2 ;;
    --llm-ssh-host)         LLM_SSH_HOST="${2:-}";         shift 2 ;;
    --llm-cli-cmd)          LLM_CLI_CMD="${2:-}";          shift 2 ;;
    --llm-timeout-seconds)  LLM_TIMEOUT_SECONDS="${2:-}";  shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *)
      echo "[fail] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# Universal required args.
for var in GPFS_ROOT RESOLVE_LOG; do
  if [[ -z "${!var}" ]]; then
    echo "[fail] --${var//_/-} (lowercase) is required" >&2
    usage >&2
    exit 2
  fi
done

# Transport-specific required args.
case "$LLM_TRANSPORT" in
  http)
    if [[ -z "$C_HOST" ]]; then
      echo "[fail] --c-host is required for --llm-transport http" >&2
      usage >&2
      exit 2
    fi
    ;;
  ssh-cli)
    if [[ -z "$LLM_SSH_HOST" ]]; then
      echo "[fail] --llm-ssh-host is required for --llm-transport ssh-cli" >&2
      usage >&2
      exit 2
    fi
    if [[ -z "$LLM_CLI_CMD" ]]; then
      echo "[fail] --llm-cli-cmd is required for --llm-transport ssh-cli" >&2
      usage >&2
      exit 2
    fi
    if [[ "$LLM_CLI_CMD" != *"{prompt}"* ]]; then
      echo "[fail] --llm-cli-cmd must contain literal '{prompt}' placeholder" >&2
      exit 2
    fi
    ;;
  *)
    echo "[fail] --llm-transport must be 'http' or 'ssh-cli'" >&2
    exit 2
    ;;
esac

case "$PHASE" in
  1|2|3|4|all) ;;
  *) echo "[fail] --phase must be 1|2|3|4|all" >&2; exit 2 ;;
esac

# Preflight: resolve log must exist and be non-empty before opening
# tunnel + spinning workers. (Mirrors the orchestrator runner.)
if [[ ! -s "$RESOLVE_LOG" ]]; then
  echo "[fail] --resolve-log $RESOLVE_LOG is missing or empty" >&2
  exit 2
fi

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

DATE_STAMP="$(date +%Y%m%d)"
EXPECT_DATE="$(date -u +%Y-%m-%d)"
REPORT="docs/release/internal-acceptance-report-$DATE_STAMP.md"
DEFAULT_STATE_PATH="$GPFS_ROOT/state/triage_state.json"
STATE_PATH="${STATE_PATH:-$DEFAULT_STATE_PATH}"
QUEUE_ROOT="$GPFS_ROOT/queue"
RESULTS_ROOT="$GPFS_ROOT/results"
LOGS_DIR="$GPFS_ROOT/logs"
BUDGET_DIR="$GPFS_ROOT/budget"
TRIAGE_OUT="docs/triage_backlog"
LLM_ENDPOINT="http://localhost:$LOCAL_PORT"

mkdir -p "$QUEUE_ROOT" "$RESULTS_ROOT" "$LOGS_DIR" "$BUDGET_DIR" "$TRIAGE_OUT" docs/release

# Initialize the report (clobber any prior same-day partial run).
{
  echo "# INTERNAL Acceptance Report -- $DATE_STAMP"
  echo
  echo "Generated by \`scripts/automation/runners/run_acceptance_gate.sh\`."
  echo
  echo "Selected phase: \`$PHASE\`"
  echo
} > "$REPORT"

# ---------------------------------------------------------------------------
# SSH tunnel: one tunnel per gate run, supervised in main().
# ---------------------------------------------------------------------------
TUNNEL_PID=""
PHASE4_WORKTREE=""

cleanup() {
  if [[ -n "$TUNNEL_PID" ]] && kill -0 "$TUNNEL_PID" 2>/dev/null; then
    kill -TERM "$TUNNEL_PID" 2>/dev/null || true
    wait "$TUNNEL_PID" 2>/dev/null || true
  fi
  # Remove the Phase 4 worktree if we created one; never touch the
  # operator's own worktrees.
  if [[ -n "$PHASE4_WORKTREE" && -d "$PHASE4_WORKTREE" ]]; then
    git worktree remove --force "$PHASE4_WORKTREE" 2>/dev/null || true
  fi
}
trap cleanup SIGTERM SIGINT EXIT

start_tunnel() {
  # Transport-aware liveness setup.
  #
  # http mode: open one long-lived SSH local-port-forward and probe it
  # with curl against --health-path (default /v1/models). Any HTTP
  # response in [200, 500) counts as alive (4xx like 401/404 still
  # proves the tunnel forwarded to a live HTTP server). 5xx and
  # connection refused / timeout (code 000) count as down.
  #
  # ssh-cli mode: no tunnel is opened (TUNNEL_PID stays empty and the
  # EXIT trap simply skips the kill). We confirm SSH reachability with
  # a single `echo CLI_OK` non-interactive ssh, mirroring
  # setup_machine_b_cpu.sh's reachability probe.
  if [[ "$LLM_TRANSPORT" == "ssh-cli" ]]; then
    echo "[step] probing SSH to LLM CLI host $LLM_SSH_HOST (no tunnel)"
    if ! ssh -o BatchMode=yes -o ConnectTimeout=5 \
           "$LLM_SSH_HOST" 'echo CLI_OK' 2>/dev/null \
           | grep -q '^CLI_OK$'; then
      echo "[fail] ssh $LLM_SSH_HOST not reachable (BatchMode key auth required)" >&2
      return 7
    fi
    echo "[ok] ssh $LLM_SSH_HOST reachable"
    return 0
  fi
  echo "[step] opening SSH tunnel: localhost:$LOCAL_PORT -> $C_HOST:$C_PORT"
  ssh -N -L "$LOCAL_PORT:localhost:$C_PORT" \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes \
      "$C_HOST" &
  TUNNEL_PID=$!
  for i in $(seq 1 10); do
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
              "$LLM_ENDPOINT$HEALTH_PATH" 2>/dev/null || echo 000)"
    if [[ "$code" != "000" ]] && (( code >= 200 && code < 500 )); then
      echo "[ok] LLM endpoint reachable (attempt $i, http=$code)"
      return 0
    fi
    sleep 1
  done
  echo "[fail] LLM endpoint not reachable at $LLM_ENDPOINT$HEALTH_PATH" >&2
  return 7
}

# Append the transport-specific LLM args to an orchestrator argv array.
# Usage: append_llm_args ARRAY_NAME
# In http mode this adds --llm-endpoint pointing at the local forward
# (LLM_ENDPOINT). In ssh-cli mode it adds --llm-ssh-host + --llm-cli-cmd
# so the orchestrator skips the http endpoint requirement.
append_llm_args() {
  local -n _arr=$1
  _arr+=(--llm-transport "$LLM_TRANSPORT"
         --llm-timeout-seconds "$LLM_TIMEOUT_SECONDS")
  if [[ "$LLM_TRANSPORT" == "http" ]]; then
    _arr+=(--llm-endpoint "$LLM_ENDPOINT")
  else
    _arr+=(--llm-ssh-host "$LLM_SSH_HOST"
           --llm-cli-cmd "$LLM_CLI_CMD")
  fi
}

record_phase() {
  local phase_name="$1"
  local verdict="$2"
  local note="$3"
  {
    echo "## $phase_name"
    echo
    echo "Verdict: **$verdict**"
    echo
    echo "$note"
    echo
  } >> "$REPORT"
}

# Append an LLM cost stanza for the given phase (uses budget JSON when
# present; gracefully reports "not tracked" when the orchestrator was
# launched without budget reporting wired up).
append_budget_stanza() {
  local phase_name="$1"
  local budget_json="$2"
  if [[ ! -s "$budget_json" ]]; then
    {
      echo "LLM cost / tokens: not tracked (\`$budget_json\` missing)."
      echo
    } >> "$REPORT"
    return 0
  fi
  python3 - "$budget_json" "$REPORT" <<'PY' || true
import json, pathlib, sys
src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
try:
    data = json.loads(src.read_text(encoding="utf-8"))
except Exception as exc:
    with dst.open("a", encoding="utf-8") as f:
        f.write(f"LLM cost / tokens: unreadable budget file ({exc}).\n\n")
    sys.exit(0)
calls = int(data.get("total_calls", 0) or 0)
pt = int(data.get("prompt_tokens", 0) or 0)
ct = int(data.get("completion_tokens", 0) or 0)
cost = float(data.get("estimated_cost_usd", 0.0) or 0.0)
lines = [
    f"LLM calls: {calls}",
    f"Tokens: prompt={pt} completion={ct}",
    f"Estimated cost: ${cost:.4f}",
]
with dst.open("a", encoding="utf-8") as f:
    f.write("\n".join(f"- {ln}" for ln in lines))
    f.write("\n\n")
PY
}

# Count entries in a triage_state.json by canonical ``state`` field.
# The state file canonical field is ``state`` (see
# scripts/automation/state_machine.py:137 -- COMMON_FIELDS). The legacy
# ``status`` field was never produced by any writer in this tree.
count_state() {
  local state_file="$1"
  local target_state="$2"
  python3 - "$state_file" "$target_state" <<'PY' 2>/dev/null || echo 0
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
target = sys.argv[2]
if not path.is_file():
    print(0); sys.exit(0)
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print(0); sys.exit(0)
tasks = data.get("tasks", {}) or {}
print(sum(1 for t in tasks.values() if t.get("state") == target))
PY
}

ALL_PASS=1

# ---------------------------------------------------------------------------
# Phase 1 -- 3 shapes-incomplete tasks, Tier 1, isolated state root.
# ---------------------------------------------------------------------------
phase1() {
  echo "[phase1] real e2e -- 3 shapes-incomplete tasks (Tier 1)"
  local state_root="$GPFS_ROOT/state-phase1"
  local state_file="$state_root/triage_state.json"
  local budget_json="$BUDGET_DIR/phase1-$DATE_STAMP.json"
  rm -rf "$state_root"
  mkdir -p "$state_root"
  local log="$LOGS_DIR/phase1-$DATE_STAMP.log"
  local cmd=(
    python3 scripts/automation/fix_orchestrator.py
    --tier 1
    --resolve-log "$RESOLVE_LOG"
    --reverify-results "$RESULTS_ROOT/phase1"
    --state-root "$state_root"
    --verdict-class shapes-incomplete
    --init-empty-state
    --max-tasks 3
    --apply
  )
  append_llm_args cmd
  local rc=0
  BENCH_AUTOMATION_BUDGET_JSON="$budget_json" \
    "${cmd[@]}" > "$log" 2>&1 || rc=$?
  local done_count
  done_count="$(count_state "$state_file" DONE)"
  if [[ "$rc" -eq 0 && "$done_count" -ge 1 ]]; then
    record_phase "Phase 1 -- Real e2e (3 shapes-incomplete)" "PASS" \
      "Orchestrator rc=$rc; DONE count=$done_count (>=1 required). Log: \`$log\`."
  else
    record_phase "Phase 1 -- Real e2e (3 shapes-incomplete)" "FAIL" \
      "Orchestrator rc=$rc; DONE count=$done_count (>=1 required). Log: \`$log\`."
    ALL_PASS=0
  fi
  append_budget_stanza "Phase 1" "$budget_json"
}

# ---------------------------------------------------------------------------
# Phase 2 -- stratified 5+5+5 by verdict class, Tier 1, isolated state root.
#
# The corpus contains 2 upstream-not-invocable-here tasks in total (per
# the resolver report). We honestly take what's available rather than
# back-fill from a different class.
# ---------------------------------------------------------------------------
phase2() {
  echo "[phase2] stratified mixed corpus (5 shapes / 5 oracle / up to 5 upstream)"
  local state_root="$GPFS_ROOT/state-phase2"
  local state_file="$state_root/triage_state.json"
  local budget_json="$BUDGET_DIR/phase2-$DATE_STAMP.json"
  rm -rf "$state_root"
  mkdir -p "$state_root"
  local log="$LOGS_DIR/phase2-$DATE_STAMP.log"

  # Resolve the per-class task IDs once, *before* the orchestrator
  # enqueues anything, so we can guarantee the 5+5+<=5 split.
  local task_ids
  task_ids="$(python3 - "$RESOLVE_LOG" <<'PY'
import json, re, sys, pathlib
log = pathlib.Path(sys.argv[1])
text = log.read_text(encoding="utf-8", errors="replace")
buckets: dict[str, list[str]] = {
    "shapes-incomplete": [],
    "oracle-mismatch": [],
    "upstream-not-invocable-here": [],
}
# Recognise both jsonl and the legacy ``task=X BLOCKED reason=Y`` form.
for ln in text.splitlines():
    ln = ln.strip()
    if not ln:
        continue
    if ln.startswith("{"):
        try:
            row = json.loads(ln)
        except Exception:
            continue
        tid = row.get("task_id") or row.get("task")
        reason = (row.get("verdict_class")
                  or row.get("reason")
                  or row.get("subclass") or "")
        if tid and reason:
            for k in buckets:
                if k in reason and tid not in buckets[k]:
                    buckets[k].append(tid)
        continue
    m = re.match(r".*\btask=(\S+).*BLOCKED.*reason=(\S+)", ln)
    if m:
        tid = m.group(1); reason = m.group(2)
        for k in buckets:
            if k in reason and tid not in buckets[k]:
                buckets[k].append(tid)
selected: list[str] = []
selected += buckets["shapes-incomplete"][:5]
selected += buckets["oracle-mismatch"][:5]
selected += buckets["upstream-not-invocable-here"][:5]
print(",".join(selected))
PY
)"
  if [[ -z "$task_ids" ]]; then
    record_phase "Phase 2 -- Stratified mixed corpus" "FAIL" \
      "Could not derive a stratified task list from \`$RESOLVE_LOG\`."
    ALL_PASS=0
    return
  fi
  # Resolved id count tells the operator how the strata broke down.
  local id_count=0
  id_count=$(awk -F, '{print NF}' <<<"$task_ids")

  # Pass the stratified id list as an explicit --task-ids allowlist so
  # the orchestrator does NOT silently fall back to --max-tasks slicing
  # on a class it does not expect. The orchestrator's --task-ids branch
  # bypasses both --max-tasks and the verdict-class enqueue filter (the
  # per-task dispatch still gates on the supported class set).
  local cmd=(
    python3 scripts/automation/fix_orchestrator.py
    --tier 1
    --resolve-log "$RESOLVE_LOG"
    --reverify-results "$RESULTS_ROOT/phase2"
    --state-root "$state_root"
    --verdict-class all
    --init-empty-state
    --max-tasks "$id_count"
    --task-ids "$task_ids"
    --apply
  )
  append_llm_args cmd
  local rc=0
  BENCH_AUTOMATION_BUDGET_JSON="$budget_json" \
    "${cmd[@]}" > "$log" 2>&1 || rc=$?
  local backlog_count done_count
  backlog_count="$(count_state "$state_file" BACKLOG_PENDING_GOLDEN)"
  done_count="$(count_state "$state_file" DONE)"
  if [[ "$rc" -eq 0 && "$backlog_count" -ge 1 && "$done_count" -ge 1 ]]; then
    record_phase "Phase 2 -- Three-verdict closure" "PASS" \
      "rc=$rc; tasks=$id_count; BACKLOG_PENDING_GOLDEN=$backlog_count (>=1); DONE=$done_count (>=1). Log: \`$log\`."
  else
    record_phase "Phase 2 -- Three-verdict closure" "FAIL" \
      "rc=$rc; tasks=$id_count; BACKLOG_PENDING_GOLDEN=$backlog_count; DONE=$done_count. Log: \`$log\`."
    ALL_PASS=0
  fi
  append_budget_stanza "Phase 2" "$budget_json"
}

# ---------------------------------------------------------------------------
# Phase 3 -- full corpus + dated backlog-report + dated summary-metrics.
# ---------------------------------------------------------------------------
phase3() {
  echo "[phase3] full corpus + backlog-report + summary-metrics"
  local state_root="$GPFS_ROOT/state-phase3"
  local state_file="$state_root/triage_state.json"
  local budget_json="$BUDGET_DIR/phase3-$DATE_STAMP.json"
  rm -rf "$state_root"
  mkdir -p "$state_root"
  local log="$LOGS_DIR/phase3-$DATE_STAMP.log"
  local cmd=(
    python3 scripts/automation/fix_orchestrator.py
    --tier 1
    --resolve-log "$RESOLVE_LOG"
    --reverify-results "$RESULTS_ROOT/phase3"
    --state-root "$state_root"
    --verdict-class all
    --init-empty-state
    --max-tasks 1000
    --apply
  )
  append_llm_args cmd
  local rc=0
  BENCH_AUTOMATION_BUDGET_JSON="$budget_json" \
    "${cmd[@]}" > "$log" 2>&1 || rc=$?

  local report_rc=0
  python3 scripts/automation/backlog_report.py \
    --state-path "$state_file" \
    --tasks-root tasks/ \
    --out-dir "$TRIAGE_OUT" \
    --date-iso "$EXPECT_DATE" \
    --format both >> "$log" 2>&1 || report_rc=$?

  local metrics_rc=0
  python3 scripts/automation/summary_metrics.py \
    --state-path "$state_file" \
    --tasks-root tasks/ \
    --out-dir "$TRIAGE_OUT" \
    --date-iso "$EXPECT_DATE" \
    --format both >> "$log" 2>&1 || metrics_rc=$?

  # Writers always include the date in the filename; assert the dated
  # artefacts exist non-empty.
  local outputs_ok=1
  for f in \
      "$TRIAGE_OUT/backlog-report-$EXPECT_DATE.md" \
      "$TRIAGE_OUT/backlog-report-$EXPECT_DATE.json" \
      "$TRIAGE_OUT/summary-metrics-$EXPECT_DATE.md" \
      "$TRIAGE_OUT/summary-metrics-$EXPECT_DATE.json"; do
    if [[ ! -s "$f" ]]; then
      outputs_ok=0
      echo "[phase3] missing or empty: $f"
      break
    fi
  done

  if [[ "$rc" -eq 0 && "$report_rc" -eq 0 && "$metrics_rc" -eq 0 && "$outputs_ok" -eq 1 ]]; then
    record_phase "Phase 3 -- Full corpus + backlog/metrics" "PASS" \
      "orchestrator rc=$rc; backlog_report rc=$report_rc; summary_metrics rc=$metrics_rc; all 4 dated outputs non-empty. Log: \`$log\`."
  else
    record_phase "Phase 3 -- Full corpus + backlog/metrics" "FAIL" \
      "orchestrator rc=$rc; backlog_report rc=$report_rc; summary_metrics rc=$metrics_rc; outputs_ok=$outputs_ok. Log: \`$log\`."
    ALL_PASS=0
  fi
  append_budget_stanza "Phase 3" "$budget_json"
}

# ---------------------------------------------------------------------------
# Phase 4 -- Tier 1 vs Tier 2 verdict equivalence in an isolated worktree.
#
# Tier 1 runs in the primary checkout (its meta.yaml mutations write the
# phase_a_prime / phase_b_capture blocks). Tier 2 runs inside a
# throw-away `git worktree` so the two tiers cannot stomp on each
# other's task tree. After both runs complete we diff the two state
# files by canonical ``state`` field plus verdict_class.
# ---------------------------------------------------------------------------
phase4() {
  echo "[phase4] Tier-1 vs Tier-2 verdict equivalence (isolated worktrees)"
  local state_root_t1="$GPFS_ROOT/state-phase4-tier1"
  local state_root_t2="$GPFS_ROOT/state-phase4-tier2"
  local state_file_t1="$state_root_t1/triage_state.json"
  local state_file_t2="$state_root_t2/triage_state.json"
  local results_t1="$RESULTS_ROOT/phase4-tier1"
  local results_t2="$RESULTS_ROOT/phase4-tier2"
  local budget_t1="$BUDGET_DIR/phase4-tier1-$DATE_STAMP.json"
  local budget_t2="$BUDGET_DIR/phase4-tier2-$DATE_STAMP.json"
  rm -rf "$state_root_t1" "$state_root_t2"
  mkdir -p "$state_root_t1" "$state_root_t2" "$results_t1" "$results_t2"

  local tier1_log="$LOGS_DIR/phase4-tier1-$DATE_STAMP.log"
  local tier2_log="$LOGS_DIR/phase4-tier2-$DATE_STAMP.log"

  local t1_cmd=(
    python3 scripts/automation/fix_orchestrator.py
    --tier 1
    --resolve-log "$RESOLVE_LOG"
    --reverify-results "$results_t1"
    --state-root "$state_root_t1"
    --verdict-class all
    --init-empty-state
    --max-tasks 5
    --apply
  )
  append_llm_args t1_cmd
  local t1_rc=0
  BENCH_AUTOMATION_BUDGET_JSON="$budget_t1" \
    "${t1_cmd[@]}" > "$tier1_log" 2>&1 || t1_rc=$?

  # Set up the Phase 4 Tier-2 worktree. The worktree is a detached
  # checkout of HEAD: the orchestrator can mutate meta.yaml freely
  # without ever touching the primary tree. EXIT trap removes it.
  PHASE4_WORKTREE="$GPFS_ROOT/.phase4-worktree"
  rm -rf "$PHASE4_WORKTREE"
  if ! git worktree add --detach "$PHASE4_WORKTREE" HEAD > "$tier2_log" 2>&1; then
    record_phase "Phase 4 -- Tier-1/Tier-2 equivalence" "FAIL" \
      "git worktree add failed; see \`$tier2_log\`. Phase 4 requires an isolated tree to keep Tier 1 + Tier 2 task mutations disjoint."
    ALL_PASS=0
    append_budget_stanza "Phase 4" "$budget_t1"
    return
  fi

  local t2_cmd=(
    python3 scripts/automation/fix_orchestrator.py
    --tier 2
    --resolve-log "$RESOLVE_LOG"
    --reverify-results "$results_t2"
    --state-root "$state_root_t2"
    --queue-root "$QUEUE_ROOT"
    --repo-root "$PHASE4_WORKTREE"
    --verdict-class all
    --init-empty-state
    --max-tasks 5
    --apply
  )
  append_llm_args t2_cmd
  local t2_rc=0
  BENCH_AUTOMATION_BUDGET_JSON="$budget_t2" \
    "${t2_cmd[@]}" >> "$tier2_log" 2>&1 || t2_rc=$?

  # Equivalence: tasks shared by both runs must agree on canonical
  # ``state`` + ``verdict_class``.
  local match_count
  match_count="$(python3 - "$state_file_t1" "$state_file_t2" <<'PY' 2>/dev/null || echo 0
import json, pathlib, sys
t1_path, t2_path = map(pathlib.Path, sys.argv[1:3])
if not (t1_path.is_file() and t2_path.is_file()):
    print(0); sys.exit(0)
try:
    t1 = json.loads(t1_path.read_text(encoding="utf-8")).get("tasks", {}) or {}
    t2 = json.loads(t2_path.read_text(encoding="utf-8")).get("tasks", {}) or {}
except Exception:
    print(0); sys.exit(0)
shared = set(t1) & set(t2)
match = 0
for k in shared:
    a, b = t1[k], t2[k]
    if a.get("state") == b.get("state") and a.get("verdict_class") == b.get("verdict_class"):
        match += 1
print(match)
PY
)"

  if [[ "$t1_rc" -eq 0 && "$t2_rc" -eq 0 && "$match_count" -ge 5 ]]; then
    record_phase "Phase 4 -- Tier-1/Tier-2 equivalence" "PASS" \
      "tier1 rc=$t1_rc; tier2 rc=$t2_rc; agreeing tasks (state + verdict_class)=$match_count (>=5). Logs: \`$tier1_log\`, \`$tier2_log\`."
  else
    record_phase "Phase 4 -- Tier-1/Tier-2 equivalence" "FAIL" \
      "tier1 rc=$t1_rc; tier2 rc=$t2_rc; agreeing=$match_count. Logs: \`$tier1_log\`, \`$tier2_log\`."
    ALL_PASS=0
  fi
  append_budget_stanza "Phase 4 (Tier 1)" "$budget_t1"
  append_budget_stanza "Phase 4 (Tier 2)" "$budget_t2"
}

# ---------------------------------------------------------------------------
# main: one tunnel, then run the requested phase(s).
# ---------------------------------------------------------------------------
main() {
  start_tunnel
  case "$PHASE" in
    1)   phase1 ;;
    2)   phase2 ;;
    3)   phase3 ;;
    4)   phase4 ;;
    all) phase1; phase2; phase3; phase4 ;;
  esac
}

main

{
  echo "## Summary"
  echo
  if [[ "$ALL_PASS" -eq 1 ]]; then
    echo "Overall: **PASS**"
  else
    echo "Overall: **FAIL**"
  fi
  echo
  echo "Report file: \`$REPORT\`"
  echo
  echo "Per-phase state roots: \`$GPFS_ROOT/state-phase{1,2,3,4-tier1,4-tier2}\`."
  echo
  echo "Per-phase budget files: \`$BUDGET_DIR/phase*-$DATE_STAMP.json\` (when produced)."
  echo
  echo "Compatibility note: the legacy single state file \`$STATE_PATH\` is"
  echo "not written by this gate; per-phase isolation is the canonical layout."
} >> "$REPORT"

echo "[done] acceptance report: $REPORT"
if [[ "$ALL_PASS" -ne 1 ]]; then
  exit 8
fi
