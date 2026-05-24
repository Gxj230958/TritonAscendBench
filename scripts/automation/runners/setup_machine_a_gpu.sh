#!/usr/bin/env bash
# Runs on: A-machine (GPU host). One-time environment validation + automation extras install.
#
# Machine A (GPU host) one-time setup.
#
# Verifies the local GPU host can run:
#   * bench scripts: scripts/phase_a_prime.py --verify,
#     scripts/phase_b_golden_capture.py, scripts/resolve_upstream_entrypoints.py
#   * the automation daemon: scripts/automation/bench_evaluator_daemon.py
#
# Side effects (when --skip-install is NOT set):
#   * pip install -e ".[automation,dev,gpu]" inside the active env.
#
# Usage:
#   bash scripts/automation/runners/setup_machine_a_gpu.sh \
#     --gpfs-root /gpfs/shared/tab_automation
#
# Exit non-zero on the first check that fails (set -euo pipefail).
set -euo pipefail

usage() {
  cat <<'EOF'
Runs on: A-machine (GPU host).
Usage: setup_machine_a_gpu.sh --gpfs-root PATH [--skip-install]

Options:
  --gpfs-root PATH   Required. Shared GPFS root that holds queue/, results/,
                     logs/ for the orchestrator <-> daemon protocol.
  --skip-install     Skip "pip install -e .[automation,dev,gpu]".
  -h, --help         Show this help text and exit 0.
EOF
}

GPFS_ROOT=""
SKIP_INSTALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpfs-root)
      GPFS_ROOT="${2:-}"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
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

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

echo "[step] repo root: $REPO_ROOT"

# ---- 1. python toolchain checks -------------------------------------------
echo "[step] checking python / torch / triton toolchain"
python3 - <<'PY'
import sys
assert sys.version_info >= (3, 11), f"python>=3.11 required, got {sys.version_info}"
import torch
v = torch.__version__.split("+", 1)[0]
parts = [int(p) for p in v.split(".")[:3]]
assert parts >= [2, 9, 1], f"torch>=2.9.1 required, got {torch.__version__}"
import triton
tv = [int(p) for p in triton.__version__.split(".")[:3]]
assert tv >= [3, 5, 1], f"triton>=3.5.1 required, got {triton.__version__}"
print(f"[ok] torch={torch.__version__} triton={triton.__version__}")
PY

# ---- 2. GPU visibility -----------------------------------------------------
echo "[step] checking CUDA availability"
GPU_INFO="$(python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() returned False"
i = 0
name = torch.cuda.get_device_name(i)
mem_gb = round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 1)
print(f"{name}|{mem_gb}")
PY
)"
GPU_NAME="${GPU_INFO%%|*}"
GPU_MEM_GB="${GPU_INFO##*|}"
echo "[ok] gpu: $GPU_NAME mem=${GPU_MEM_GB}GB"

# ---- 3. GPFS mount writability ---------------------------------------------
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

# ---- 4. pip install --------------------------------------------------------
if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  echo "[step] pip install -e .[automation,dev,gpu]"
  pip install -e ".[automation,dev,gpu]"
else
  echo "[step] skipping pip install (--skip-install)"
fi

# ---- 5. daemon CLI smoke test ----------------------------------------------
echo "[step] daemon --help smoke test"
python3 scripts/automation/bench_evaluator_daemon.py --help > /dev/null
echo "[ok] bench_evaluator_daemon importable"

# ---- 6. bench CLI smoke tests ----------------------------------------------
echo "[step] bench scripts --help smoke test"
python3 scripts/phase_a_prime.py --help > /dev/null
echo "[ok] phase_a_prime.py importable"

# ---- 7. final health-check line --------------------------------------------
echo "[OK] machine A ready: GPU=$GPU_NAME mem=${GPU_MEM_GB}GB GPFS=$GPFS_ROOT"
