#!/usr/bin/env bash
# 00_setup_env.sh -- GPU-host environment setup for triton-ascend-bench.
#
# Run this FIRST, right after `git clone`, on the NVIDIA CUDA GPU host.
# It is idempotent: re-running only does what is missing.
#
# What it does:
#   1. Verifies python3 >= 3.11 and a CUDA driver / nvidia-smi.
#   2. Installs the COMPLETE frozen dependency set pinned in
#      infra/version_matrix.toml + infra/Dockerfile.gpu: torch / triton /
#      numpy from [gpu], pyyaml / safetensors / click / rich from
#      [python_deps], and einops from [gpu_capture].extra_pins.
#   3. Installs the bench itself in editable mode with the [dev] extra
#      (the only extra pyproject.toml declares -- the GPU numerics stack
#      is pinned explicitly in step 2, not via a pyproject extra).
#   4. Checks for the frozen upstream vllm/sglang checkouts that the
#      Phase A' resolver needs at <repo>/../infer/{vllm,sglang}; if they
#      are missing it prints exactly what to clone (it does NOT guess a
#      commit -- the per-task commit SHAs live in tasks/*/meta.yaml and
#      the checkout must be pinned to a clean state by the operator).
#   5. Runs a CUDA smoke check via torch.
#
# It deliberately does NOT install torch_npu / triton-ascend (NPU-only).
#
# Usage:
#   bash scripts/gpu_pipeline/00_setup_env.sh            # full setup
#   bash scripts/gpu_pipeline/00_setup_env.sh --check    # prechecks only
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${_HERE}/lib_common.sh"

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

# Frozen pins -- the COMPLETE set the bench is verified against. Single
# source of truth is infra/version_matrix.toml + infra/Dockerfile.gpu;
# duplicated here as explicit literals so `pip install` does not depend on
# a TOML parser. verify_version_matrix.py cross-checks the toml at the end
# of the pipeline -- if these drift the operator will be told.
#
# Keep this list IN SYNC with infra/Dockerfile.gpu's pip install block.
FROZEN_PINS=(
    "torch==2.5.1"          # version_matrix [gpu].torch
    "triton==3.2.0"         # version_matrix [gpu].triton
    "numpy==1.26.0"         # version_matrix [python_deps].numpy
    "pyyaml==6.0"           # version_matrix [python_deps].pyyaml
    "safetensors==0.4.5"    # version_matrix [python_deps].safetensors
    "click==8.1.7"          # version_matrix [python_deps].click
    "rich==13.7"            # version_matrix [python_deps].rich
    "einops==0.8.0"         # version_matrix [gpu_capture].extra_pins
)

step_banner "0" "GPU environment setup"

# --- 1. host prechecks ------------------------------------------------------
log_info "checking python3 ..."
require_python
log_ok "python3 $(python3 -V 2>&1 | awk '{print $2}')"

log_info "checking NVIDIA driver ..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
        | sed 's/^/  GPU: /'
    log_ok "nvidia-smi present"
else
    log_warn "nvidia-smi not found -- if this host genuinely has no GPU the"
    log_warn "  Phase A' GPU-authoritative steps cannot run here."
fi

log_info "checking Nsight Compute (ncu) ..."
if command -v ncu >/dev/null 2>&1 || [ -x /usr/local/cuda/bin/ncu ]; then
    log_ok "ncu present"
else
    log_warn "ncu not found -- step 20 (cost-model GPU cross-check) measures"
    log_warn "  HBM traffic via Nsight Compute DRAM counters. Without ncu it"
    log_warn "  is fail-closed: every cross-checked task is BLOCKED (never a"
    log_warn "  fabricated PASS). Install the CUDA toolkit's Nsight Compute."
fi

# --- 2. frozen upstream checkouts -------------------------------------------
log_info "checking frozen upstream checkouts at ${INFER_ROOT} ..."
INFER_MISSING=0
for repo in vllm sglang; do
    if [ -d "${INFER_ROOT}/${repo}" ]; then
        log_ok "found ${INFER_ROOT}/${repo}"
    else
        log_warn "missing ${INFER_ROOT}/${repo}"
        INFER_MISSING=1
    fi
done
if [ "${INFER_MISSING}" = "1" ]; then
    cat >&2 <<EOF

  ----------------------------------------------------------------------
  ACTION REQUIRED: the Phase A' entrypoint resolver and severing flow
  read the FROZEN upstream sources from:

      ${INFER_ROOT}/vllm
      ${INFER_ROOT}/sglang

  These are NOT vendored into this repo. Clone them as siblings of the
  bench checkout and pin each to the commit recorded in the task metas:

      mkdir -p "${INFER_ROOT}"
      git -C "${INFER_ROOT}" clone https://github.com/vllm-project/vllm.git
      git -C "${INFER_ROOT}" clone https://github.com/sgl-project/sglang.git

  Then, per task, check out the commit in tasks/<level>/<id>/meta.yaml
  (upstream.commit_sha). The resolver is FAIL-CLOSED: a dirty or
  unpinned tree raises StaleResolverCacheError -- it will not silently
  use a wrong upstream.
  ----------------------------------------------------------------------
EOF
fi

if [ "${CHECK_ONLY}" = "1" ]; then
    log_info "--check given: stopping after prechecks (no install performed)"
    [ "${INFER_MISSING}" = "1" ] && exit 1
    exit 0
fi

# --- 3. install the frozen GPU stack ----------------------------------------
if is_done "setup_pip"; then
    log_skip "pip install already completed (marker present; FORCE=1 to redo)"
else
    step_banner "0.1" "pip install (complete frozen stack + bench)"
    log_info "installing the ${#FROZEN_PINS[@]} pinned dependencies ..."
    for pin in "${FROZEN_PINS[@]}"; do
        log_info "  - ${pin}"
    done
    python3 -m pip install --no-cache-dir "${FROZEN_PINS[@]}"
    # The bench's only declared extra is [dev]; the GPU-side numerics
    # stack is pinned explicitly above (FROZEN_PINS) rather than via a
    # pyproject extra, so the editable install uses [dev].
    log_info "installing the bench in editable mode (.[dev]) ..."
    python3 -m pip install --no-cache-dir -e "${REPO_ROOT}[dev]"
    mark_done "setup_pip"
    log_ok "pip install complete"
fi

# --- 4. CUDA smoke check ----------------------------------------------------
step_banner "0.2" "CUDA smoke check"
require_cuda
log_ok "CUDA is available to torch"

# --- 5. bench import + CLI smoke check --------------------------------------
log_info "verifying the 'tab' CLI entrypoint ..."
python3 -c "import triton_ascend_bench; print('  triton_ascend_bench import OK')"
if command -v tab >/dev/null 2>&1; then
    log_ok "tab CLI on PATH"
else
    log_warn "tab not on PATH -- use 'python3 -m triton_ascend_bench.cli' instead"
fi

echo
log_ok "environment setup finished."
if [ "${INFER_MISSING}" = "1" ]; then
    log_warn "upstream checkouts still missing -- Phase A' steps WILL fail"
    log_warn "until ${INFER_ROOT}/{vllm,sglang} are cloned and pinned."
    exit 1
fi
log_info "next: bash scripts/gpu_pipeline/run_all.sh"
