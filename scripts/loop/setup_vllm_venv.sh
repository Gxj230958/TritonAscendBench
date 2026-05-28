#!/usr/bin/env bash
# One-time bootstrap for the daemon-side vllm venv on GPU A.
#
# The bench's vendored ../infer/vllm/ hard-pins torch==2.11.0 (see
# infer/vllm/pyproject.toml + requirements/{build/cuda,cuda,cpu}.txt) and
# does NOT ship the compiled vllm._C / vllm._C_stable_libtorch extensions.
# Whatever python runs phase_a_prime must therefore have torch 2.11.0
# AND a built-in-tree vllm._C, or every task that transitively imports
# vllm.platforms.cuda fails at module load with ModuleNotFoundError.
#
# This script builds a SEPARATE venv (does NOT touch the daemon's own
# env) so the system python stays at whatever it was. Daemon picks it
# up via:
#
#   supervise_daemon.sh --workers 4 --gpus 4 \
#     --phase-python <BENCH_PARENT>/vllm_venv/bin/python3
#
# Run once on GPU A. ~10-30 min build (vllm._C is a real C++/CUDA tree).
#
# Paths default to those derived from this script's location:
#   BENCH_PARENT = <repo>/../   (i.e. kernel/ on gpfs)
#   VENV_DIR     = $BENCH_PARENT/vllm_venv
#   VLLM_SRC     = $BENCH_PARENT/infer/vllm
# Override any with an env var if your layout differs.
#
# Re-run: requires the venv NOT to already exist (refuses to clobber).
# To rebuild, `rm -rf <BENCH_PARENT>/vllm_venv` first.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BENCH_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
BENCH_PARENT="${BENCH_PARENT:-$(cd "$BENCH_ROOT/.." && pwd)}"
VENV_DIR="${VENV_DIR:-$BENCH_PARENT/vllm_venv}"
VLLM_SRC="${VLLM_SRC:-$BENCH_PARENT/infer/vllm}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
MAX_JOBS="${MAX_JOBS:-8}"
BUILD_LOG="${BUILD_LOG:-/tmp/vllm_build.log}"
# Default torch install uses pip's normal resolution, i.e. whatever
# PIP_INDEX_URL / pip.conf the operator has configured (typically an
# internal mirror like Nexus on closed networks). The fallback below
# only fires if that install yields a CPU-only or wrong-version wheel;
# if you have a CUDA-tagged wheel index, set TORCH_INDEX_URL to it.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

echo "=== layout ==="
echo "  BENCH_ROOT   = $BENCH_ROOT"
echo "  BENCH_PARENT = $BENCH_PARENT"
echo "  VENV_DIR     = $VENV_DIR"
echo "  VLLM_SRC     = $VLLM_SRC"
echo "  TORCH        = $TORCH_VERSION  (MAX_JOBS=$MAX_JOBS)"

# ===== 0. sanity: nvcc + CUDA + system python + vllm source layout =====
echo ""
echo "=== sanity ==="
if ! command -v nvidia-smi >/dev/null; then
    echo "FATAL: nvidia-smi not on PATH; this script must run on a GPU host" >&2
    exit 2
fi
# `head -N` on a long nvidia-smi listing trips set -o pipefail via SIGPIPE
# from the closed reader; `awk 'NR<=6'` consumes to EOF and avoids that.
nvidia-smi --query-gpu=index,name,memory.total --format=csv | awk 'NR<=6'
if ! command -v nvcc >/dev/null; then
    echo "FATAL: nvcc not on PATH; cuda-toolkit-dev required to build vllm._C" >&2
    exit 2
fi
nvcc --version | head -4
# Detect CUDA major.minor for the optional pytorch.org wheel index fallback.
CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1 | tr -d .)
if [ -z "$CUDA_VER" ]; then
    echo "FATAL: could not parse CUDA version from nvcc --version" >&2
    exit 2
fi
echo "  detected CUDA: cu${CUDA_VER}"

if [ ! -f "$VLLM_SRC/setup.py" ]; then
    echo "FATAL: $VLLM_SRC/setup.py not found; check VLLM_SRC=$VLLM_SRC" >&2
    exit 2
fi

# Show what system python the daemon currently sees (informational; the
# venv we build below is intentionally independent of this).
python3 -c "import sys, torch; print('  system python:', sys.executable);"\
" print('  system torch:', torch.__version__, '(cuda:', torch.version.cuda, ')')" || true

# ===== 1. create venv =====
echo ""
echo "=== create venv ==="
if [ -d "$VENV_DIR" ]; then
    echo "FATAL: $VENV_DIR already exists. To rebuild, remove it first:" >&2
    # printf %q so a VENV_DIR containing spaces / metacharacters renders
    # as a safely-pasteable command.
    printf '  rm -rf %q\n' "$VENV_DIR" >&2
    exit 2
fi
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip wheel setuptools

# ===== 2. torch == 2.11.0 (with PyTorch index fallback) =====
echo ""
echo "=== install torch==$TORCH_VERSION ==="

# A torch wheel that "pip-installs cleanly" can still be CPU-only -- then
# torch.cuda.is_available() comes back False and every task would later
# UNRUN(needs-gpu). Treat "install + cuda visible" as a single atomic
# attempt; on failure of EITHER side, wipe the broken install and retry
# from the pytorch.org cu<X> wheel index.
_install_and_validate_torch() {
    local extra_args=()
    if [ "$#" -gt 0 ]; then
        # --force-reinstall so a stale CPU wheel from the previous attempt
        # is replaced by the CUDA-tagged wheel from the index URL.
        extra_args=(--index-url "$1" --force-reinstall)
    fi
    "$VENV_DIR/bin/pip" install "torch==$TORCH_VERSION" "${extra_args[@]}" || return 1
    "$VENV_DIR/bin/python3" - <<PY || return 2
import torch
assert torch.__version__.startswith("$TORCH_VERSION"), \
    f"wrong torch version: {torch.__version__}"
assert torch.cuda.is_available(), \
    "torch installed but torch.cuda.is_available()==False (CPU-only wheel)"
print("  torch", torch.__version__, "cuda", torch.version.cuda,
      "devices", torch.cuda.device_count())
PY
}

if ! _install_and_validate_torch; then
    if [ -z "$TORCH_INDEX_URL" ]; then
        cat >&2 <<EOF
FATAL: default pip install produced a CPU-only or wrong-version torch.
Set TORCH_INDEX_URL to a CUDA-tagged wheel index and re-run, e.g.:
    export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu${CUDA_VER}
or point at an internal mirror that serves the cu${CUDA_VER} build.
EOF
        exit 2
    fi
    echo "  retrying via TORCH_INDEX_URL=$TORCH_INDEX_URL"
    "$VENV_DIR/bin/pip" uninstall -y torch || true
    _install_and_validate_torch "$TORCH_INDEX_URL"
fi

# ===== 3. vllm build + runtime requirements =====
echo ""
echo "=== install vllm requirements ==="
for req in \
    "$VLLM_SRC/requirements/build/cuda.txt" \
    "$VLLM_SRC/requirements/cuda.txt"
do
    if [ ! -f "$req" ]; then
        echo "FATAL: missing $req (vllm source tree incomplete?)" >&2
        exit 2
    fi
    "$VENV_DIR/bin/pip" install -r "$req"
done

# ===== 4. build vllm in-place (compiles _C + _C_stable_libtorch) =====
echo ""
echo "=== build vllm in-place (10-30 min; log -> $BUILD_LOG) ==="
cd "$VLLM_SRC"
VLLM_TARGET_DEVICE=cuda MAX_JOBS="$MAX_JOBS" \
    "$VENV_DIR/bin/pip" install -e . --no-build-isolation 2>&1 | tee "$BUILD_LOG"
cd - >/dev/null

# ===== 5. verify (all three imports must pass) =====
echo ""
echo "=== verify ==="
"$VENV_DIR/bin/python3" -c "
import vllm._C, vllm._C_stable_libtorch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
print('  vllm._C:                 ', vllm._C.__file__)
print('  vllm._C_stable_libtorch: ', vllm._C_stable_libtorch.__file__)
print('  PAD_SLOT_ID:             ', PAD_SLOT_ID)
"

# Render the operator-facing commands with printf %q so spaces or shell
# metacharacters in VENV_DIR survive copy-paste intact.
PHASE_PYTHON_Q=$(printf '%q' "$VENV_DIR/bin/python3")
echo ""
echo "=== DONE ==="
echo ""
echo "Daemon flag to use this venv:"
echo "  --phase-python $PHASE_PYTHON_Q"
echo ""
echo "Restart supervisor:"
echo "  tmux kill-session -t loop_daemon 2>/dev/null"
printf '  tmux new-session -d -s loop_daemon "scripts/loop/supervise_daemon.sh \\\n'
printf '    --workers 4 --gpus 4 \\\n'
printf '    --phase-python %s \\\n' "$PHASE_PYTHON_Q"
printf '    2>&1 | tee -a ../loop_mailbox/daemon.log"\n'
