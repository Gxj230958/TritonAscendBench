#!/usr/bin/env bash
# Runs on: A-machine (GPU host). One-time environment validation + automation extras install.
#
# Machine A (GPU host) one-time setup.
#
# Verifies the local GPU host can run:
#   * bench scripts: scripts/verify/standalone_freeze_verifier.py --verify,
#     scripts/provenance/capture_golden_baselines.py, scripts/verify/resolve_upstream_entrypoints.py
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
# Note: the [infer] extra carries the leaf packages that vllm/sglang's
# source trees under infer/ import (pybase64, pyzmq, cbor2, msgspec).
# Without these, the oracle-consistency subprocess hits ModuleNotFoundError
# before reaching the per-task verdict logic and resolve mis-classifies
# everything as upstream-not-invocable-here. Step 5 below probes the
# import chain to catch any new leaf that isn't yet listed in [infer].
if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  echo "[step] pip install -e .[automation,dev,gpu,infer]"
  pip install -e ".[automation,dev,gpu,infer]"
else
  echo "[step] skipping pip install (--skip-install)"
fi

# ---- 5. vllm/sglang import probe -------------------------------------------
# Defence in depth: even with [infer] installed, vllm/sglang upgrades may
# add a new transitive import we have not listed yet. Probe the actual
# import chain the resolver uses (vllm.config triggers vllm.utils.hashing
# -> cbor2; sglang.srt.utils is the most-used sub-module across tasks)
# and surface the exact missing module + a pip install hint -- so setup
# fails LOUDLY when the env can't really invoke upstream, instead of
# letting resolve silently bucket every task as upstream-not-invocable-here.
INFER_ROOT="$(cd "$REPO_ROOT/../infer" 2>/dev/null && pwd || true)"
if [[ -z "$INFER_ROOT" || ! -d "$INFER_ROOT/vllm" || ! -d "$INFER_ROOT/sglang/python" ]]; then
  echo "[fail] infer/ tree missing (expected REPO_ROOT/../infer with vllm/ and sglang/python/)" >&2
  echo "[fail] looked at: ${INFER_ROOT:-<REPO_ROOT/../infer unresolved>}" >&2
  exit 4
fi
echo "[step] probing vllm/sglang import chain (infer/ at $INFER_ROOT)"
INFER_ROOT="$INFER_ROOT" python3 - <<'PY'
import importlib, os, sys
infer = os.environ["INFER_ROOT"]
for p in (os.path.join(infer, "vllm"), os.path.join(infer, "sglang", "python")):
    if p not in sys.path:
        sys.path.insert(0, p)

probes = ["vllm.config", "sglang.srt.utils"]
failures = []
for name in probes:
    try:
        importlib.import_module(name)
    except Exception as e:
        failures.append((name, f"{type(e).__name__}: {e}"))

if failures:
    print("[fail] vllm/sglang import chain broken:", file=sys.stderr)
    for name, err in failures:
        print(f"  {name}: {err}", file=sys.stderr)
    print(
        "\nResolver / S2-smoke will mis-classify every affected task as\n"
        "upstream-not-invocable-here. Install the missing leaf packages\n"
        "and re-run setup:\n"
        "  pip install -e \".[infer]\"\n"
        "or pin the known leaves individually:\n"
        "  pip install pybase64 pyzmq cbor2 msgspec\n"
        "If a NEW leaf appears in the error above, add it to the [infer]\n"
        "extra in pyproject.toml so future setups install it automatically.",
        file=sys.stderr,
    )
    sys.exit(4)
print("[ok] vllm.config and sglang.srt.utils import cleanly")
PY

# ---- 6. daemon CLI smoke test ----------------------------------------------
echo "[step] daemon --help smoke test"
python3 scripts/automation/bench_evaluator_daemon.py --help > /dev/null
echo "[ok] bench_evaluator_daemon importable"

# ---- 7. bench CLI smoke tests ----------------------------------------------
echo "[step] bench scripts --help smoke test"
python3 scripts/verify/standalone_freeze_verifier.py --help > /dev/null
echo "[ok] phase_a_prime.py importable"

# ---- 8. final health-check line --------------------------------------------
echo "[OK] machine A ready: GPU=$GPU_NAME mem=${GPU_MEM_GB}GB GPFS=$GPFS_ROOT"
