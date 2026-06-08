#!/usr/bin/env bash
# 00_setup_env.sh -- GPU-host environment setup for triton-ascend-bench.
#
# Run this FIRST, right after `git clone`, on the NVIDIA CUDA GPU host.
# It is idempotent: re-running only does what is missing.
#
# What it does:
#   1. Verifies python3 >= 3.11 and a CUDA driver / nvidia-smi.
#   2. Installs the bench in editable mode with the [dev] extra. The
#      bench declares only >= lower-bound deps in pyproject.toml
#      [project].dependencies; it does NOT pin an exact GPU numerics
#      stack. pip installs the bench plus any MISSING dependency and
#      leaves versions the host already satisfies untouched -- the
#      bench runs on whatever torch / triton the host already has.
#   3. Checks for the frozen upstream vllm/sglang checkouts that the
#      Phase A' resolver needs at <repo>/../infer/{vllm,sglang}; if they
#      are missing it prints exactly what to clone (it does NOT guess a
#      commit -- the per-task commit SHAs live in tasks/*/meta.yaml and
#      the checkout must be pinned to a clean state by the operator).
#   4. Runs a CUDA smoke check via torch.
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

# --- 3. install the bench (compatible with the host GPU stack) --------------
if is_done "setup_pip"; then
    log_skip "pip install already completed (marker present; FORCE=1 to redo)"
else
    step_banner "0.1" "install the bench (host-compatible)"
    log_info "host GPU stack vs infra/version_matrix.toml:"
    # Compare the host's actual torch/triton/python/cuda + Python deps
    # against what version_matrix.toml records. A mismatch is a WARNING,
    # not a hard fail: the bench runs on the host stack by design. The
    # operator must still be told so a baseline captured on this host is
    # accompanied by a version_matrix.toml update reflecting the real
    # stack. tomllib is stdlib on python>=3.11; no third-party dep.
    VM_REPORT="$(python3 - "${REPO_ROOT}/infra/version_matrix.toml" <<'PY'
import importlib
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover -- python<3.11
    print("STATUS warn version_matrix comparison skipped (no tomllib)")
    sys.exit(0)

matrix_path = Path(sys.argv[1])
try:
    matrix = tomllib.loads(matrix_path.read_text())
except (OSError, tomllib.TOMLDecodeError) as exc:
    print(f"STATUS warn cannot read {matrix_path}: {exc}")
    sys.exit(0)

gpu = matrix.get("gpu", {})
python_deps = matrix.get("python_deps", {})


def host_version(module_name):
    try:
        mod = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 -- a probe import must never abort setup
        return None
    return getattr(mod, "__version__", "?")


# (label, host value, matrix-expected value)
checks = []

py_host = ".".join(str(x) for x in sys.version_info[:3])
checks.append(("python", py_host, gpu.get("python", "")))

torch_host = host_version("torch")
checks.append(("torch", torch_host, gpu.get("torch", "")))

triton_host = host_version("triton")
checks.append(("triton", triton_host, gpu.get("triton", "")))

cuda_host = None
try:
    import torch as _torch

    cuda_host = _torch.version.cuda
except Exception:  # noqa: BLE001 -- torch missing or no cuda build
    cuda_host = None
checks.append(("cuda", cuda_host, gpu.get("cuda", "")))

# numpy / pyyaml / safetensors / click from [python_deps]
for label, module_name, key in (
    ("numpy", "numpy", "numpy"),
    ("pyyaml", "yaml", "pyyaml"),
    ("safetensors", "safetensors", "safetensors"),
    ("click", "click", "click"),
):
    checks.append((label, host_version(module_name), python_deps.get(key, "")))

mismatches = 0
for label, host_val, expected in checks:
    host_disp = host_val if host_val else "(missing)"
    if not expected:
        # No recorded expectation -- print for visibility only.
        print(f"LINE info     {label:13s}host={host_disp}  matrix=(unset)")
        continue
    # python: compare only major.minor against the matrix value.
    host_cmp = host_val
    if label == "python" and host_val:
        host_cmp = ".".join(host_val.split(".")[:2])
    if host_cmp == expected:
        print(f"LINE ok       {label:13s}host={host_disp}  matrix={expected}")
    else:
        mismatches += 1
        print(f"LINE warn     {label:13s}host={host_disp}  matrix={expected}  MISMATCH")

if mismatches:
    print(f"STATUS warn {mismatches} version(s) differ from version_matrix.toml")
else:
    print("STATUS ok host stack matches version_matrix.toml")
PY
)"
    while IFS= read -r vline; do
        case "${vline}" in
            "LINE ok "*)    log_ok   "${vline#LINE ok }" ;;
            "LINE warn "*)  log_warn "${vline#LINE warn }" ;;
            "LINE info "*)  log_info "${vline#LINE info }" ;;
            "STATUS ok "*)  log_ok   "${vline#STATUS ok }" ;;
            "STATUS warn "*)
                log_warn "${vline#STATUS warn }"
                log_warn "  the bench runs on the host stack by design -- this is"
                log_warn "  not fatal, but any baseline captured on this host must"
                log_warn "  be accompanied by a version_matrix.toml update that"
                log_warn "  records the actual host versions."
                ;;
            "") ;;
            *)              log_info "${vline}" ;;
        esac
    done <<< "${VM_REPORT}"
    # The bench declares only >= lower-bound deps (pyproject.toml); it
    # does NOT pin an exact numerics stack. `pip install -e .[dev]`
    # installs the bench plus any missing dependency and leaves an
    # already-satisfied torch / triton untouched -- the bench runs on
    # whatever GPU stack the host already has.
    log_info "installing the bench in editable mode (.[dev]) ..."
    python3 -m pip install --no-cache-dir -e "${REPO_ROOT}[dev]"
    mark_done "setup_pip"
    log_ok "bench installed"
fi

# --- 4. CUDA smoke check ----------------------------------------------------
step_banner "0.2" "CUDA smoke check"
require_cuda
log_ok "CUDA is available to torch"

# --- 5. bench import + CLI smoke check --------------------------------------
log_info "verifying the 'tab' CLI entrypoint ..."
python3 -c "import ascendbench; print('  ascendbench import OK')"
if command -v tab >/dev/null 2>&1; then
    log_ok "tab CLI on PATH"
else
    log_warn "tab not on PATH -- use 'python3 -m ascendbench.engine.cli' instead"
fi

echo
log_ok "environment setup finished."
if [ "${INFER_MISSING}" = "1" ]; then
    log_warn "upstream checkouts still missing -- Phase A' steps WILL fail"
    log_warn "until ${INFER_ROOT}/{vllm,sglang} are cloned and pinned."
    exit 1
fi
log_info "next: bash scripts/gpu_pipeline/run_all.sh"
