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
                              [--llm-transport MODE] [--llm-cli-cmd TPL]
                              [--skip-install]

Options:
  --gpfs-root PATH    Required. Shared GPFS root (same path as machine A).
  --a-host SSH        Required. SSH target for machine A (GPU host).
  --c-host SSH        Required. SSH target for machine C (LLM host).
  --llm-transport MODE 'http' (default; setup just confirms SSH to C
                       reaches a live host) or 'ssh-cli' (in addition,
                       if --llm-cli-cmd is supplied, runs the template
                       with '{prompt}' substituted for '--version' as a
                       best-effort probe). The bench never hardcodes a
                       specific CLI name.
  --llm-cli-cmd TPL   Optional CLI command template for the ssh-cli
                       transport probe; must contain literal '{prompt}'.
                       Example: '<your-cli> exec {prompt}'.
  --skip-install      Skip "pip install -e .[automation,dev]".
  -h, --help          Show this help text and exit 0.
EOF
}

GPFS_ROOT=""
A_HOST=""
C_HOST=""
SKIP_INSTALL=0
LLM_TRANSPORT="http"
LLM_CLI_CMD=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpfs-root) GPFS_ROOT="${2:-}"; shift 2 ;;
    --a-host)    A_HOST="${2:-}";    shift 2 ;;
    --c-host)    C_HOST="${2:-}";    shift 2 ;;
    --llm-transport) LLM_TRANSPORT="${2:-}"; shift 2 ;;
    --llm-cli-cmd)   LLM_CLI_CMD="${2:-}";   shift 2 ;;
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

case "$LLM_TRANSPORT" in
  http|ssh-cli) ;;
  *) echo "[fail] --llm-transport must be 'http' or 'ssh-cli'" >&2; exit 2 ;;
esac

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
PYTHONPATH="$REPO_ROOT/scripts" python3 -c "from automation.llm_client import LLMClient, SSHCLIClient, create_llm_client; print('OK')" > /dev/null
echo "[ok] llm_client transports importable"

# ---- 6. (optional) ssh-cli mode CLI version probe -------------------------
# When --llm-transport ssh-cli is selected AND the operator supplied a
# --llm-cli-cmd template, run the template once on C with '{prompt}'
# substituted for '--version'. This is a best-effort probe: a non-zero
# rc is reported as a warning, not a fatal error, because the operator
# may have a CLI that does not implement --version.
CLI_PROBE="not-attempted"
if [[ "$LLM_TRANSPORT" == "ssh-cli" ]]; then
  if [[ -n "$LLM_CLI_CMD" ]]; then
    if [[ "$LLM_CLI_CMD" != *"{prompt}"* ]]; then
      echo "[fail] --llm-cli-cmd must contain literal '{prompt}' placeholder" >&2
      exit 2
    fi
    PROBE_CMD="${LLM_CLI_CMD//\{prompt\}/--version}"
    echo "[step] probing remote CLI on $C_HOST: $PROBE_CMD"
    if probe_out="$(ssh -o BatchMode=yes -o ConnectTimeout=5 \
                       "$C_HOST" "$PROBE_CMD" 2>&1 | head -3)"; then
      echo "[ok] remote CLI probe output:"
      printf '  %s\n' "$probe_out"
      CLI_PROBE="ok"
    else
      echo "[warn] remote CLI --version probe failed (rc=$?); operator may need a different probe command" >&2
      CLI_PROBE="warn"
    fi
  else
    echo "[hint] --llm-transport ssh-cli selected but --llm-cli-cmd not supplied; skipping CLI probe"
    CLI_PROBE="skipped-no-template"
  fi
fi

# ---- 7. final health-check line -------------------------------------------
echo "[OK] machine B ready: GPFS=$GPFS_ROOT A-host=$A_REACHABLE C-host=$C_REACHABLE transport=$LLM_TRANSPORT cli-probe=$CLI_PROBE"
