#!/usr/bin/env bash
# 20_cost_model_xcheck.sh -- cost-model GPU cross-check (DESIGN.md §5.6).
#
# GPU-host P0 #2: cross-check each task's ANALYTIC cost model
# (meta.yaml.cost_model.flops / bytes formulas) against INSTRUMENTED
# GPU-Triton measurements, and write meta.yaml: cost_model.status
# (PASS / BLOCKED / FLOOR_DOMINATED) -- ncu-evidenced PERFORMANCE metadata, NOT a
# release gate. Conditional access, L2 reuse, tile geometry, and the
# HBM burst floor make a portable analytic bytes formula unable to hit
# per-shape tol everywhere, so most tasks are cost_model BLOCKED; that
# does not block release. This step only records the evidence.
#
# The release gate keys on the two REAL correctness proofs:
#     release_status == PASS
#         <=> phase_a_prime.status == PASS (S2 severance)
#             AND oracle_kernel_equivalence in {VERIFIED, CEILING_ACCEPTED}
#                 (C2 oracle-equivalence)
#
# Implementation: scripts/phase_a_cost_xcheck.py. Per task it
#   1. evaluates the analytic W_c / M_r formulas at each public eval
#      shape (reuses ascendbench.perf.cost_model verbatim);
#   2. runs the promoted (Phase A' PASS) kernel.py on a live CUDA device
#      under Nsight Compute (ncu), measuring HBM traffic via the
#      hardware DRAM counters dram__bytes_read.sum + dram__bytes_write.sum;
#   3. reconciles |analytic - measured| / measured <= gpu_xcheck_tol for
#      the bytes model (FLOPs are recorded analytic-only, not GPU-measured);
#   4. atomically writes meta.yaml: cost_model.status = PASS|BLOCKED|
#      FLOOR_DOMINATED (the last for sub-burst index kernels with no asymptotic
#      HBM regime, source-audited; never a PASS) with a per-shape evidence
#      block (mirrors update_meta_phase_a_prime).
#
# It is fail-closed: if ncu is unavailable or yields no DRAM metric the
# affected shape/task is BLOCKED with an explicit reason -- never a
# fabricated measurement.
#
# A non-score-eligible task (stub oracle / PHASE-C-MANUAL shapes, e.g.
# L2_08_gumbel_sample) carries only a placeholder cost_model and is
# skipped -- it never enters the scored denominator, so it must not be
# able to BLOCK the gate.
#
# Honesty: the cross-check is GPU-authoritative. With no live CUDA
# device phase_a_cost_xcheck.py exits non-zero (it never fabricates a
# PASS); this wrapper's require_cuda precheck fails fast first.
#
# Prerequisites: 00_setup_env.sh done, CUDA visible, step 10
# (10_phase_a_prime.sh) has promoted kernel.py for every task.
#
# Usage:
#   bash scripts/gpu_pipeline/20_cost_model_xcheck.sh           # full sweep
#   bash scripts/gpu_pipeline/20_cost_model_xcheck.sh --task L1_12_rms_norm
#   FORCE=1 bash scripts/gpu_pipeline/20_cost_model_xcheck.sh   # ignore markers
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${_HERE}/lib_common.sh"

TASK_FILTER=""
if [ "${1:-}" = "--task" ]; then
    [ -n "${2:-}" ] || die "--task needs a task id"
    TASK_FILTER="$2"
fi

XCHECK_SCRIPT="${REPO_ROOT}/scripts/phase_a_cost_xcheck.py"
[ -f "${XCHECK_SCRIPT}" ] || die "missing ${XCHECK_SCRIPT}"

step_banner "B" "cost-model GPU cross-check"

# --- prechecks --------------------------------------------------------------
require_cuda

# --- cost-model cross-check + atomic meta.yaml write ------------------------
if is_done "cost_xcheck" && [ -z "${TASK_FILTER}" ]; then
    log_skip "cost-model cross-check already done (FORCE=1 to redo)"
else
    if [ -n "${TASK_FILTER}" ]; then
        step_banner "B.1" "cross-check task ${TASK_FILTER} (--apply)"
        run_logged "cost_xcheck_${TASK_FILTER}" \
            python3 "${XCHECK_SCRIPT}" --task "${TASK_FILTER}" --apply
    else
        step_banner "B.1" "cross-check every task (--all --apply)"
        log_info "evaluates analytic W_c/M_r + instruments GPU-Triton bytes;"
        log_info "only tol-PASS tasks get cost_model.status=PASS in meta.yaml."
        run_logged "cost_xcheck" \
            python3 "${XCHECK_SCRIPT}" --all --apply
    fi
    log_ok "cost-model cross-check complete"
fi

echo
log_ok "cost-model GPU cross-check stage finished."
log_info "verify verdicts: grep -A1 'cost_model:' tasks/*/*/meta.yaml | grep status"
log_info "next: bash scripts/gpu_pipeline/30_golden_capture.sh"
