#!/usr/bin/env bash
# 30_golden_capture.sh -- offline golden capture for upstream_oracle_required.
#
# GPU-host P0 #3 (Phase B' ground-truth capture). For each task whose
# meta.yaml: oracle_simplified == upstream_oracle_required (19 tasks
# today), the PyTorch oracle is KNOWN to deviate from the frozen
# upstream kernel (DESIGN.md §5.7). Their evaluation-time golden must
# therefore be captured ONCE, offline, on the GPU, by running the
# frozen upstream kernel over the task's COMPLETE eval shape set
# (public + sealed hidden), and written to:
#
#     tasks/<level>/<id>/golden/<shape_id>.safetensors
#
# eval._resolve_golden + eligibility.golden_complete read exactly this
# layout (a flat {tensor_key: tensor} safetensors dict per shape).
#
# This wrapper drives scripts/phase_b_golden_capture.py, which:
#   1. enumerates the upstream_oracle_required tasks from trusted repo
#      meta (eligibility.is_upstream_oracle_required);
#   2. per task enumerates ALL eval shape ids -- public AND sealed
#      hidden (inputs.shape_ids);
#   3. materialises each shape's inputs with the verified-runner seed
#      (inputs.build_inputs, seed=0 -- byte-identical to eval's
#      correctness_inputs), runs the PROMOTED frozen upstream
#      kernel.py::run on the GPU in an isolated subprocess;
#   4. writes the output dict to golden/<shape_id>.safetensors
#      (safetensors-only, no pickle -- matches the IPC constraint).
#
# It MUST run AFTER Phase A' promote (step 10): the capture uses the
# severed/promoted standalone kernel.py. phase_b_golden_capture.py
# gates each task on meta.phase_a_prime.status == PASS and honestly
# fails any task that is not promoted yet -- it never fabricates a
# golden artifact, and a no-CUDA host is a hard error.
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${_HERE}/lib_common.sh"

TASK_FILTER=""
if [ "${1:-}" = "--task" ]; then
    [ -n "${2:-}" ] || die "--task needs a task id"
    TASK_FILTER="$2"
fi

CAPTURE_SCRIPT="${REPO_ROOT}/scripts/phase_b_golden_capture.py"
[ -f "${CAPTURE_SCRIPT}" ] || die "missing ${CAPTURE_SCRIPT}"

step_banner "C" "offline golden capture (upstream_oracle_required)"

# --- prechecks --------------------------------------------------------------
# golden capture runs the promoted standalone kernel.py on a real CUDA
# GPU; without one the script honestly hard-errors, so fail fast here.
require_cuda

# Count the affected tasks so the operator sees the exact scope.
N_UOR="$(
    grep -rl 'oracle_simplified: *upstream_oracle_required' \
        "${REPO_ROOT}"/tasks/*/*/meta.yaml 2>/dev/null | wc -l | tr -d ' '
)"
log_info "upstream_oracle_required tasks needing golden capture: ${N_UOR}"

# --- run golden capture -----------------------------------------------------
if is_done "golden_capture" && [ -z "${TASK_FILTER}" ]; then
    log_skip "golden capture already done (FORCE=1 to redo)"
else
    if [ -n "${TASK_FILTER}" ]; then
        log_info "capturing single task: ${TASK_FILTER}"
        run_logged "golden_capture_${TASK_FILTER}" \
            python3 "${CAPTURE_SCRIPT}" --task "${TASK_FILTER}"
    else
        log_info "capturing all ${N_UOR} upstream_oracle_required tasks"
        run_logged "golden_capture" \
            python3 "${CAPTURE_SCRIPT}" --all
    fi
    log_ok "golden capture complete"
fi

echo
log_ok "Phase B' golden capture finished."
log_info "verify artifacts: ls tasks/*/*/golden/*.safetensors"
log_info "next: bash scripts/gpu_pipeline/40_release_gate.sh"
