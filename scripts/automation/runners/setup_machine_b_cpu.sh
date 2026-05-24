#!/usr/bin/env bash
# Machine B (CPU orchestrator host) one-time setup.
#
# Machine B does NOT run bench scripts. It runs:
#   * scripts/automation/fix_orchestrator.py (main loop)
#   * an SSH local-port-forward tunnel to the LLM machine (C)
#
# Verifies SSH reachability of A and C, GPFS mount, and python+automation
# install.
#
# Usage:
#   bash scripts/automation/runners/setup_machine_b_cpu.sh \
#     --gpfs-root /gpfs/shared/tab_automation \
#     --a-host bench-gpu-a01 \
#     --c-host llm-svc-c01
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: setup_machine_b_cpu.sh --gpfs-root PATH --a-host SSH --c-host SSH
                              [--skip-install]

Options:
  --gpfs-root PATH   Required. Shared GPFS root (same path as machine A).
  --a-host SSH       Required. SSH target for machine A (GPU host).
  --c-host SSH       Required. SSH target for machine C (LLM host).
  --skip-install     Skip "pip install -e .[automation,dev]".
  -h, --help         Show this help text and exit 0.
EOF
}

GPFS_ROOT=""
A_HOST=""
C_HOST=""
SKIP_INSTALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpfs-root) GPFS_ROOT="${2:-}"; shift 2 ;;
    --a-host)    A_HOST="${2:-}";    shift 2 ;;
    --c-host)    C_HOST="${2:-}";    shift 2 ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    *)
      echo "[fail] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for var in GPFS_ROOT A_HOST C_HOST; do
  if [[ -z "${!var}" ]]; then
    echo "[fail] --${var,,} is required" >&2
    usage >&2
    exit 2
  fi
done

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

echo "[step] repo root: $REPO_ROOT"

# ---- 1. python + git checks -----------------------------------------------
echo "[step] checking python>=3.11 and git"
python3 - <<'PY'
import sys
assert sys.version_info >= (3, 11), f"python>=3.11 required, got {sys.version_info}"
PY
command -v git > /dev/null || { echo "[fail] git not found in PATH" >&2; exit 4; }
echo "[ok] python>=3.11 and git present"

# ---- 2. pip install (NO gpu extras) ---------------------------------------
if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  echo "[step] pip install -e .[automation,dev]"
  pip install -e ".[automation,dev]"
else
  echo "[step] skipping pip install (--skip-install)"
fi

# ---- 3. SSH reachability for A and C --------------------------------------
A_REACHABLE="no"
C_REACHABLE="no"
echo "[step] SSH probe: $A_HOST"
if ssh -o BatchMode=yes -o ConnectTimeout=5 "$A_HOST" 'echo OK' 2>/dev/null | grep -q '^OK$'; then
  A_REACHABLE="yes"
  echo "[ok] ssh $A_HOST reachable"
else
  echo "[fail] ssh $A_HOST not reachable (BatchMode key-based auth required)" >&2
  exit 5
fi
echo "[step] SSH probe: $C_HOST"
if ssh -o BatchMode=yes -o ConnectTimeout=5 "$C_HOST" 'echo OK' 2>/dev/null | grep -q '^OK$'; then
  C_REACHABLE="yes"
  echo "[ok] ssh $C_HOST reachable"
else
  echo "[fail] ssh $C_HOST not reachable (BatchMode key-based auth required)" >&2
  exit 5
fi

# ---- 4. GPFS mount writability --------------------------------------------
echo "[step] checking GPFS mount writability at $GPFS_ROOT"
if [[ ! -d "$GPFS_ROOT" ]]; then
  echo "[fail] gpfs-root does not exist: $GPFS_ROOT" >&2
  exit 3
fi
GPFS_PROBE="$GPFS_ROOT/.gpfs_test"
if ! ( : > "$GPFS_PROBE" ) 2>/dev/null; then
  echo "[fail] gpfs-root not writable: $GPFS_PROBE" >&2
  exit 3
fi
rm -f "$GPFS_PROBE"
echo "[ok] gpfs writable"

# ---- 5. orchestrator + llm_client smoke -----------------------------------
echo "[step] orchestrator --help smoke test"
python3 scripts/automation/fix_orchestrator.py --help > /dev/null
echo "[ok] fix_orchestrator importable"

echo "[step] llm_client import smoke test"
PYTHONPATH="$REPO_ROOT/scripts" python3 -c "from automation.llm_client import OpenAIChatClient; print('OK')" > /dev/null
echo "[ok] OpenAIChatClient importable"

# ---- 6. final health-check line -------------------------------------------
echo "[OK] machine B ready: GPFS=$GPFS_ROOT A-host=$A_REACHABLE C-host=$C_REACHABLE"
