#!/usr/bin/env bash
# Machine A runner: launch bench_evaluator_daemon in long-running mode.
#
# Single-daemon-per-host is enforced by an exclusive ``flock`` on a
# per-host lock file at <gpfs-root>/.daemons/<hostname>.lock. The lock
# is held by file descriptor 9 for the lifetime of this script -- two
# concurrent invocations on the same host cannot both acquire it, so
# the second instance fails fast with rc=6.
#
# Side files:
#   * <gpfs-root>/.daemons/<host>.lock  -- the flock target.
#   * <gpfs-root>/.daemons/<host>.pid   -- the daemon's PID (for
#     operators inspecting the running pid). Only removed by this
#     script if the PID file still records *our* PID, so a kill -9 of
#     the script never destroys another process's PID file.
#
# Usage:
#   bash scripts/automation/runners/run_machine_a_daemon.sh \
#     --gpfs-root /gpfs/shared/tab_automation
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_machine_a_daemon.sh --gpfs-root PATH [options]

Required:
  --gpfs-root PATH         Shared GPFS root (queue/, results/, logs/, .daemons/).

Options:
  --bench-root PATH        Local bench repo root (default: git rev-parse --show-toplevel).
  --max-jobs INT           Exit after N jobs (0 = unlimited; default 0).
  --lease-seconds INT      Daemon-lock lease (default 600).
  --heartbeat-seconds INT  Heartbeat cadence (default 30).
  -h, --help               Show this help and exit 0.
EOF
}

GPFS_ROOT=""
BENCH_ROOT=""
MAX_JOBS="0"
LEASE_SECONDS="600"
HEARTBEAT_SECONDS="30"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpfs-root)         GPFS_ROOT="${2:-}";         shift 2 ;;
    --bench-root)        BENCH_ROOT="${2:-}";        shift 2 ;;
    --max-jobs)          MAX_JOBS="${2:-}";          shift 2 ;;
    --lease-seconds)     LEASE_SECONDS="${2:-}";     shift 2 ;;
    --heartbeat-seconds) HEARTBEAT_SECONDS="${2:-}"; shift 2 ;;
    -h|--help)           usage; exit 0 ;;
    *)
      echo "[fail] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$GPFS_ROOT" ]]; then
  echo "[fail] --gpfs-root is required" >&2
  usage >&2
  exit 2
fi

if [[ -z "$BENCH_ROOT" ]]; then
  BENCH_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
fi

if [[ ! -d "$GPFS_ROOT" ]]; then
  echo "[fail] gpfs-root does not exist: $GPFS_ROOT" >&2
  exit 3
fi
if [[ ! -d "$BENCH_ROOT/scripts/automation" ]]; then
  echo "[fail] bench-root does not look like a bench checkout: $BENCH_ROOT" >&2
  exit 3
fi

HOSTNAME_CANON="$(hostname)"
DAEMONS_DIR="$GPFS_ROOT/.daemons"
LOGS_DIR="$GPFS_ROOT/logs"
HOST_LOCK="$DAEMONS_DIR/$HOSTNAME_CANON.lock"
PID_FILE="$DAEMONS_DIR/$HOSTNAME_CANON.pid"
PID_TMP="$DAEMONS_DIR/.$HOSTNAME_CANON.pid.tmp.$$"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOGS_DIR/daemon-$HOSTNAME_CANON-$TIMESTAMP.log"

mkdir -p "$DAEMONS_DIR" "$LOGS_DIR"

# ---- Acquire exclusive lock for this host (single-daemon invariant) -------
#
# Two concurrent invocations on the same host both passing the old
# "kill -0 on stale pid" check is a real race: between the check and
# the PID-file write, another instance can squeeze in. ``flock -n``
# rejects the second caller deterministically.
if ! command -v flock > /dev/null 2>&1; then
  echo "[fail] flock(1) is required for single-daemon-per-host enforcement" >&2
  exit 4
fi

exec 9>"$HOST_LOCK"
if ! flock -n 9; then
  EXISTING_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  echo "[fail] another daemon already holds the host lock at $HOST_LOCK" \
       "(pid=${EXISTING_PID:-unknown})" >&2
  exit 6
fi

cd "$BENCH_ROOT"

# Launch daemon in the background; capture PID.
python3 scripts/automation/bench_evaluator_daemon.py \
  --gpfs-root "$GPFS_ROOT" \
  --bench-root "$BENCH_ROOT" \
  --poll-interval 5 \
  --lease-seconds "$LEASE_SECONDS" \
  --heartbeat-seconds "$HEARTBEAT_SECONDS" \
  --max-jobs "$MAX_JOBS" \
  > "$LOG_FILE" 2>&1 &
DAEMON_PID=$!

# Write PID file atomically (write-tmp + rename).
echo "$DAEMON_PID" > "$PID_TMP"
mv "$PID_TMP" "$PID_FILE"
# Track *our* PID-file owner for cleanup: only this script's PID matches.
PID_OWNER="$$"

cleanup() {
  if [[ -n "${DAEMON_PID:-}" ]] && kill -0 "$DAEMON_PID" 2>/dev/null; then
    echo "[stop] forwarding signal to daemon pid=$DAEMON_PID"
    kill -TERM "$DAEMON_PID" 2>/dev/null || true
    wait "$DAEMON_PID" 2>/dev/null || true
  fi
  # Only delete the PID file if it still records our daemon's PID;
  # otherwise another instance has taken over and we must not touch it.
  if [[ -f "$PID_FILE" ]]; then
    on_disk="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ "$on_disk" == "$DAEMON_PID" ]]; then
      rm -f "$PID_FILE"
    fi
  fi
  # Releasing the flock is implicit on process exit (fd 9 closes).
  echo "[stop] daemon exited; pid file cleared if owned"
}
trap cleanup SIGTERM SIGINT EXIT

echo "[start] daemon pid=$DAEMON_PID log=$LOG_FILE host_lock=$HOST_LOCK owner=$PID_OWNER"

# Block until daemon exits (or signal arrives).
wait "$DAEMON_PID"
EXIT_CODE=$?
echo "[exit] daemon exit_code=$EXIT_CODE"
exit "$EXIT_CODE"
