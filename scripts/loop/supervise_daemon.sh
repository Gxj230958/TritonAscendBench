#!/usr/bin/env bash
# Outer supervisor for scripts/loop/daemon_a.py.
#
# The daemon itself self-re-execs on daemon_a.py edits (see
# _maybe_restart_self in daemon_a.py), so routine code updates do NOT exit
# the process. This wrapper exists for the cases the daemon CANNOT recover
# from in-process: hard crashes, OOM kills, segfaults, host-induced
# terminations. A back-off keeps a fast crash loop from spamming the log
# and burning GPU time.
#
# Deploy ONCE on GPU A (inside tmux/screen for SSH persistence):
#
#   tmux new-session -d -s loop_daemon \
#     "scripts/loop/supervise_daemon.sh --workers 4 --gpus 4"
#
# All arguments are forwarded to daemon_a.py. After the initial install
# Claude needs zero manual intervention: code updates are picked up by the
# daemon's self-execv, crashes are restarted here. Only a host reboot (which
# terminates tmux) requires re-running the tmux line above.

set -u

# Resolve the bench root from this script's location so the supervisor works
# regardless of caller pwd. scripts/loop/supervise_daemon.sh -> bench root is
# two levels up.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BENCH_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
DAEMON="$BENCH_ROOT/scripts/loop/daemon_a.py"

if [ ! -f "$DAEMON" ]; then
    echo "[supervisor] FATAL: $DAEMON not found" >&2
    exit 2
fi

# Short-circuit --help / -h ANYWHERE in argv so the supervisor's `while true`
# never respawns a help-and-exit run forever. Scan all args, not just the
# first, so `supervise_daemon.sh --workers 4 --help` is handled too.
for arg in "$@"; do
    case "$arg" in
        -h|--help)
            cat <<EOF
Usage: $(basename "$0") <args-forwarded-to-daemon_a.py>

Outer supervisor with a 5s back-off ``while true`` loop around daemon_a.py.
Typical deployment (one-time on GPU A, inside tmux):

    tmux new-session -d -s loop_daemon \\
      "$(basename "$0") --workers 4 --gpus 4"

Forwarded ``daemon_a.py --help``:

EOF
            exec python3 "$DAEMON" --help
            ;;
    esac
done

cd "$BENCH_ROOT"

BACKOFF=5
while true; do
    ts=$(date -u +%FT%TZ)
    echo "[supervisor $ts] starting daemon: python3 $DAEMON $*"
    python3 "$DAEMON" "$@"
    rc=$?
    ts=$(date -u +%FT%TZ)
    echo "[supervisor $ts] daemon exited rc=$rc; restart in ${BACKOFF}s"
    sleep "$BACKOFF"
done
