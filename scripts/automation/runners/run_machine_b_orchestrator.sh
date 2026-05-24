#!/usr/bin/env bash
# Machine B runner: launch fix_orchestrator (Tier 2) with an SSH local-port
# forward to the LLM machine (C).
#
# The tunnel maps B's localhost:<local-port> -> C's localhost:<c-port>, so
# the orchestrator can hit the LLM with a plain http://localhost:<port> URL
# while the actual HTTP traffic rides the SSH connection.
#
# Operational guarantees:
#   * --resolve-log must exist and be non-empty before we open the tunnel.
#     This avoids the silent no-op behaviour where fix_orchestrator's
#     parse_resolve_log() returns [] when the file is missing.
#   * The HTTP health probe accepts any 2xx / 3xx / 4xx response code on
#     the configured --health-path (default /v1/models -- the standard
#     OpenAI-compatible model listing endpoint). 5xx and connection
#     failures count as "tunnel down". Matching
#     scripts/automation/lib/ssh_tunnel.py semantics.
#   * Orchestrator runs in the background so the runner can watch the
#     tunnel + the orchestrator together. Either process dying triggers
#     a clean SIGTERM to the other and the runner exits with that
#     process's rc.
#
# Usage:
#   bash scripts/automation/runners/run_machine_b_orchestrator.sh \
#     --gpfs-root /gpfs/shared/tab_automation \
#     --c-host llm-svc-c01 --c-port 8000 \
#     --resolve-log docs/triage_backlog/resolver-run.jsonl \
#     --apply
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_machine_b_orchestrator.sh --gpfs-root PATH --c-host SSH
                                     --resolve-log PATH [options]

Required:
  --gpfs-root PATH        Shared GPFS root (queue/, results/, logs/, state/).
  --c-host SSH            SSH target for machine C (LLM host).
  --resolve-log PATH      Path to resolver classification log (jsonl).

Options:
  --c-port INT            Remote LLM port on machine C (default 8000).
  --local-port INT        Local forward port on machine B (default 8080).
  --queue-root PATH       Default: <gpfs-root>/queue
  --reverify-results PATH Default: <gpfs-root>/results
  --state-root PATH       Default: <gpfs-root>/state
  --health-path PATH      HTTP path to probe for tunnel liveness
                          (default: /v1/models -- OpenAI-compatible).
  --tier INT              1 or 2 (default 2 for production three-machine).
  --verdict-class CLASS   all | shapes-incomplete | oracle-mismatch |
                          upstream-not-invocable-here (default: all)
  --max-tasks INT         Default 30.
  --max-attempts INT      Default 3.
  --apply                 Required to actually mutate task files. Default
                          (omitted) is preview-only.
  -h, --help              Show this help and exit 0.
EOF
}

GPFS_ROOT=""
C_HOST=""
C_PORT="8000"
LOCAL_PORT="8080"
QUEUE_ROOT=""
REVERIFY_RESULTS=""
STATE_ROOT=""
RESOLVE_LOG=""
HEALTH_PATH="/v1/models"
TIER="2"
VERDICT_CLASS="all"
MAX_TASKS="30"
MAX_ATTEMPTS="3"
APPLY_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpfs-root)        GPFS_ROOT="${2:-}";        shift 2 ;;
    --c-host)           C_HOST="${2:-}";           shift 2 ;;
    --c-port)           C_PORT="${2:-}";           shift 2 ;;
    --local-port)       LOCAL_PORT="${2:-}";       shift 2 ;;
    --queue-root)       QUEUE_ROOT="${2:-}";       shift 2 ;;
    --reverify-results) REVERIFY_RESULTS="${2:-}"; shift 2 ;;
    --state-root)       STATE_ROOT="${2:-}";       shift 2 ;;
    --resolve-log)      RESOLVE_LOG="${2:-}";      shift 2 ;;
    --health-path)      HEALTH_PATH="${2:-}";      shift 2 ;;
    --tier)             TIER="${2:-}";             shift 2 ;;
    --verdict-class)    VERDICT_CLASS="${2:-}";    shift 2 ;;
    --max-tasks)        MAX_TASKS="${2:-}";        shift 2 ;;
    --max-attempts)     MAX_ATTEMPTS="${2:-}";     shift 2 ;;
    --apply)            APPLY_FLAG="--apply";      shift ;;
    --dry-run)          APPLY_FLAG="--dry-run";    shift ;;
    -h|--help)          usage; exit 0 ;;
    *)
      echo "[fail] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for var in GPFS_ROOT C_HOST RESOLVE_LOG; do
  if [[ -z "${!var}" ]]; then
    echo "[fail] --${var//_/-} (lowercase) is required" >&2
    usage >&2
    exit 2
  fi
done

if [[ ! -d "$GPFS_ROOT" ]]; then
  echo "[fail] gpfs-root does not exist: $GPFS_ROOT" >&2
  exit 3
fi

# Preflight: resolve-log must exist and be non-empty. Without this,
# fix_orchestrator.parse_resolve_log() returns [] and the orchestrator
# silently no-ops (see scripts/automation/fix_orchestrator.py:514-515).
if [[ ! -s "$RESOLVE_LOG" ]]; then
  echo "[fail] --resolve-log $RESOLVE_LOG is missing or empty" >&2
  exit 2
fi

QUEUE_ROOT="${QUEUE_ROOT:-$GPFS_ROOT/queue}"
REVERIFY_RESULTS="${REVERIFY_RESULTS:-$GPFS_ROOT/results}"
STATE_ROOT="${STATE_ROOT:-$GPFS_ROOT/state}"
LOGS_DIR="$GPFS_ROOT/logs"
mkdir -p "$QUEUE_ROOT" "$REVERIFY_RESULTS" "$STATE_ROOT" "$LOGS_DIR"

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# ---- 1. start SSH local-port-forward --------------------------------------
TUNNEL_PID=""
ORCH_PID=""
TAIL_PID=""
LLM_ENDPOINT="http://localhost:$LOCAL_PORT"

start_tunnel() {
  echo "[step] opening SSH tunnel: localhost:$LOCAL_PORT -> $C_HOST:$C_PORT"
  ssh -N -L "$LOCAL_PORT:localhost:$C_PORT" \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      -o ControlMaster=no \
      -o ExitOnForwardFailure=yes \
      "$C_HOST" &
  TUNNEL_PID=$!
}

# Tunnel + orchestrator + tail are all children of this runner. The trap
# terminates them in the canonical order (orchestrator -> tunnel -> tail)
# in case one of them dies (the others are killed on EXIT) or the
# operator sends SIGTERM / Ctrl-C. The tail process must be killed
# explicitly: without this step it survives as a stray child of init
# after the parent runner exits.
cleanup() {
  local rc=$?
  if [[ -n "$ORCH_PID" ]] && kill -0 "$ORCH_PID" 2>/dev/null; then
    echo "[stop] sending SIGTERM to orchestrator pid=$ORCH_PID"
    kill -TERM "$ORCH_PID" 2>/dev/null || true
  fi
  if [[ -n "$TUNNEL_PID" ]] && kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo "[stop] closing SSH tunnel pid=$TUNNEL_PID"
    kill -TERM "$TUNNEL_PID" 2>/dev/null || true
  fi
  if [[ -n "$TAIL_PID" ]] && kill -0 "$TAIL_PID" 2>/dev/null; then
    echo "[stop] closing operator tail pid=$TAIL_PID"
    kill -TERM "$TAIL_PID" 2>/dev/null || true
  fi
  # Best-effort wait; never block forever. Order matches the kill order
  # so the log shows orchestrator drain before tunnel teardown.
  for pid in "$ORCH_PID" "$TUNNEL_PID" "$TAIL_PID"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
  return "$rc"
}
trap cleanup SIGTERM SIGINT EXIT

start_tunnel

# ---- 2. wait for tunnel health -------------------------------------------
# OpenAI-compatible APIs do not expose /health; the standard model
# listing endpoint /v1/models is a robust liveness check (200 + JSON).
# We accept any HTTP response code in [200, 500) as "tunnel alive":
# 4xx (e.g. 401 unauthorized, 404 wrong path) still proves the tunnel
# forwarded the request to a live HTTP server, only 5xx / connection
# refused / timeout mean the server side is genuinely down.
echo "[step] probing LLM tunnel at $LLM_ENDPOINT$HEALTH_PATH"
HEALTHY=0
for i in $(seq 1 10); do
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "$LLM_ENDPOINT$HEALTH_PATH" 2>/dev/null || echo 000)"
  if [[ "$code" != "000" ]] && (( code >= 200 && code < 500 )); then
    HEALTHY=1
    echo "[ok] LLM endpoint reachable on attempt $i (http=$code)"
    break
  fi
  sleep 1
done
if [[ "$HEALTHY" -eq 0 ]]; then
  echo "[fail] LLM endpoint not reachable at $LLM_ENDPOINT$HEALTH_PATH after 10 probes" >&2
  exit 7
fi

# ---- 3. launch orchestrator in background ---------------------------------
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOGS_DIR/orchestrator-$TIMESTAMP.log"

ORCHESTRATOR_CMD=(
  python3 scripts/automation/fix_orchestrator.py
  --tier "$TIER"
  --resolve-log "$RESOLVE_LOG"
  --queue-root "$QUEUE_ROOT"
  --reverify-results "$REVERIFY_RESULTS"
  --state-root "$STATE_ROOT"
  --llm-endpoint "$LLM_ENDPOINT"
  --verdict-class "$VERDICT_CLASS"
  --max-tasks "$MAX_TASKS"
  --max-attempts "$MAX_ATTEMPTS"
)
if [[ -n "$APPLY_FLAG" ]]; then
  ORCHESTRATOR_CMD+=("$APPLY_FLAG")
fi

echo "[start] orchestrator log=$LOG_FILE"
echo "[cmd] ${ORCHESTRATOR_CMD[*]}"
# Run in background so the runner can supervise both children. tee
# duplicates stdout for the operator on the console.
"${ORCHESTRATOR_CMD[@]}" > "$LOG_FILE" 2>&1 &
ORCH_PID=$!

# Tail the log in the background for the operator (best-effort; doesn't
# affect supervision).
tail -f "$LOG_FILE" &
TAIL_PID=$!

# ---- 4. supervise: either process dying tears the whole thing down -------
while kill -0 "$ORCH_PID" 2>/dev/null && kill -0 "$TUNNEL_PID" 2>/dev/null; do
  sleep 5
done

# At least one died; record which.
ORCH_DEAD=0
TUNNEL_DEAD=0
kill -0 "$ORCH_PID" 2>/dev/null || ORCH_DEAD=1
kill -0 "$TUNNEL_PID" 2>/dev/null || TUNNEL_DEAD=1
if [[ "$TUNNEL_DEAD" -eq 1 && "$ORCH_DEAD" -eq 0 ]]; then
  echo "[warn] SSH tunnel died (pid=$TUNNEL_PID); terminating orchestrator" >&2
elif [[ "$ORCH_DEAD" -eq 1 && "$TUNNEL_DEAD" -eq 0 ]]; then
  echo "[info] orchestrator exited (pid=$ORCH_PID); closing SSH tunnel"
fi

# Stop the operator tail; cleanup() will kill any remaining children.
kill -TERM "$TAIL_PID" 2>/dev/null || true

# Collect orchestrator exit code if it has already finished; otherwise
# the trap will signal+reap it and we default to "tunnel-driven exit".
ORCH_EXIT=0
if [[ "$ORCH_DEAD" -eq 1 ]]; then
  wait "$ORCH_PID" 2>/dev/null
  ORCH_EXIT=$?
else
  # Tunnel died first: signal orchestrator and reap.
  kill -TERM "$ORCH_PID" 2>/dev/null || true
  wait "$ORCH_PID" 2>/dev/null
  ORCH_EXIT=$?
  # Non-zero rc forced by tunnel loss -- ensure the caller sees a
  # non-zero exit so external automation reacts.
  if [[ "$ORCH_EXIT" -eq 0 ]]; then
    ORCH_EXIT=7
  fi
fi

echo "[exit] orchestrator exit_code=$ORCH_EXIT"
exit "$ORCH_EXIT"
