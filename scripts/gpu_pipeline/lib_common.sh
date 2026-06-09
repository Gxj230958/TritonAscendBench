# shellcheck shell=bash
# lib_common.sh -- shared helpers for the triton-ascend-bench GPU pipeline.
#
# Sourced by every gpu_pipeline/*.sh script. Provides: repo-root discovery,
# colourised logging, step framing, a CUDA precheck, and an idempotent
# marker mechanism so a re-run can skip already-finished steps.
#
# This file is sourced, never executed -- it has no shebang and sets no
# `set -euo pipefail` of its own (the caller owns that).

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# REPO_ROOT = the triton-ascend-bench checkout. lib_common.sh lives at
# <repo>/scripts/gpu_pipeline/lib_common.sh, so the repo root is two
# directories up from this file.
_LIB_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_LIB_COMMON_DIR}/../.." && pwd)"
export REPO_ROOT

# INFER_ROOT = the frozen upstream vllm/sglang checkouts. The bench code
# (scripts/verify/resolve_upstream_entrypoints.py:69) hard-codes this as a sibling
# of the repo: <repo>/../infer/{vllm,sglang}. It is NOT vendored into the
# bench and MUST be present (clean pinned git checkout) for Phase A'.
INFER_ROOT="$(cd "${REPO_ROOT}/.." && pwd)/infer"
export INFER_ROOT

# Pipeline state: per-step done-markers + log files live here. Kept inside
# the repo (gitignored) so a re-run is idempotent.
PIPELINE_STATE_DIR="${REPO_ROOT}/results/gpu_pipeline"
export PIPELINE_STATE_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
    _C_RED=$'\033[31m'; _C_GRN=$'\033[32m'; _C_YEL=$'\033[33m'
    _C_BLU=$'\033[34m'; _C_BLD=$'\033[1m'; _C_RST=$'\033[0m'
else
    _C_RED=''; _C_GRN=''; _C_YEL=''; _C_BLU=''; _C_BLD=''; _C_RST=''
fi

log_info()  { printf '%s[INFO]%s  %s\n'  "${_C_BLU}" "${_C_RST}" "$*"; }
log_ok()    { printf '%s[ OK ]%s  %s\n'  "${_C_GRN}" "${_C_RST}" "$*"; }
log_warn()  { printf '%s[WARN]%s  %s\n'  "${_C_YEL}" "${_C_RST}" "$*" >&2; }
log_err()   { printf '%s[FAIL]%s  %s\n'  "${_C_RED}" "${_C_RST}" "$*" >&2; }
log_skip()  { printf '%s[SKIP]%s  %s\n'  "${_C_YEL}" "${_C_RST}" "$*"; }

# step_banner <n> <title> -- prints a framed step header.
step_banner() {
    printf '\n%s========================================================%s\n' \
        "${_C_BLD}" "${_C_RST}"
    printf '%s STEP %s : %s%s\n' "${_C_BLD}" "$1" "$2" "${_C_RST}"
    printf '%s========================================================%s\n' \
        "${_C_BLD}" "${_C_RST}"
}

# die <msg> -- print error and exit non-zero. Callers use `set -e` too, but
# an explicit die makes the failing step unambiguous.
die() { log_err "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Idempotency markers
# ---------------------------------------------------------------------------
# Each orchestrated step writes <PIPELINE_STATE_DIR>/<name>.done on success.
# A re-run consults this so finished steps are skipped unless FORCE=1.

_marker_path() { printf '%s/%s.done' "${PIPELINE_STATE_DIR}" "$1"; }

mark_done() {
    mkdir -p "${PIPELINE_STATE_DIR}"
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "$(_marker_path "$1")"
}

# is_done <name> -- returns 0 if the step already succeeded and FORCE is unset.
is_done() {
    [ "${FORCE:-0}" = "1" ] && return 1
    [ -f "$(_marker_path "$1")" ]
}

# clear_markers -- wipe all done-markers (used by `run_all.sh --fresh`).
clear_markers() {
    rm -f "${PIPELINE_STATE_DIR}"/*.done 2>/dev/null || true
    log_info "cleared pipeline done-markers in ${PIPELINE_STATE_DIR}"
}

# ---------------------------------------------------------------------------
# Environment prechecks
# ---------------------------------------------------------------------------
# require_python -- the bench mandates python3 (CLAUDE.md). Confirm >= 3.11.
require_python() {
    command -v python3 >/dev/null 2>&1 || die "python3 not found on PATH"
    python3 - <<'PY' || die "python3 must be >= 3.11 (pyproject requires-python)"
import sys
raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)
PY
}

# require_cuda -- a real torch.cuda.is_available() check. Phase A' GPU-
# authoritative verify and the cost-model cross-check both need a live CUDA
# device; without one phase_a_prime honestly returns UNRUN(needs-gpu) and
# nothing gets promoted. We fail fast so the operator is not misled.
require_cuda() {
    require_python
    python3 - <<'PY'
import sys
try:
    import torch
except Exception as e:  # noqa: BLE001
    print(f"  torch import failed: {e}", file=sys.stderr)
    sys.exit(2)
if not torch.cuda.is_available():
    print("  torch.cuda.is_available() is False -- no CUDA device", file=sys.stderr)
    sys.exit(3)
dev = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"  CUDA OK: {dev} (sm_{cap[0]}{cap[1]}), torch {torch.__version__}")
PY
    case "$?" in
        0) return 0 ;;
        2) die "torch is not installed -- run 00_setup_env.sh first" ;;
        3) die "no CUDA GPU visible -- this host cannot run GPU-authoritative steps" ;;
        *) die "CUDA precheck failed" ;;
    esac
}

# require_infer_roots -- the frozen upstream checkouts must exist and be
# clean pinned git checkouts (the resolver is fail-closed on a dirty tree:
# resolve_upstream_entrypoints.py raises StaleResolverCacheError otherwise).
require_infer_roots() {
    [ -d "${INFER_ROOT}/vllm" ]   || die "missing upstream checkout: ${INFER_ROOT}/vllm (see 00_setup_env.sh)"
    [ -d "${INFER_ROOT}/sglang" ] || die "missing upstream checkout: ${INFER_ROOT}/sglang (see 00_setup_env.sh)"
    local root
    for root in "${INFER_ROOT}/vllm" "${INFER_ROOT}/sglang"; do
        if [ -d "${root}/.git" ]; then
            if [ -n "$(git -C "${root}" status --porcelain 2>/dev/null)" ]; then
                log_warn "upstream checkout ${root} has a DIRTY working tree --"
                log_warn "  the Phase A' resolver is fail-closed and will raise"
                log_warn "  StaleResolverCacheError. Pin it to a clean commit first."
            fi
        else
            log_warn "${root} is not a git checkout -- resolver provenance"
            log_warn "  (infer_state) cannot be verified; expect fail-closed errors."
        fi
    done
}

# run_logged <name> <cmd...> -- run a command, tee its output to a per-step
# log file, and on success drop the done-marker. Returns the command's
# exit status so the caller's `set -e` still trips on failure.
run_logged() {
    local name="$1"; shift
    mkdir -p "${PIPELINE_STATE_DIR}"
    local logf="${PIPELINE_STATE_DIR}/${name}.log"
    log_info "running: $* "
    log_info "  log -> ${logf}"
    if "$@" 2>&1 | tee "${logf}"; then
        mark_done "${name}"
        return 0
    fi
    return "${PIPESTATUS[0]}"
}
