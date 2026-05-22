#!/usr/bin/env bash
# 10_phase_a_prime.sh -- Phase A' GPU-authoritative reproducible-freeze.
#
# This is GPU-host P0 #1 + #3 (entrypoint resolve + severing + S2-gpu
# verify + promote). All steps below have a real script entrypoint in
# the bench; this wrapper just sequences them, logs each, and is
# idempotent via done-markers.
#
# Pipeline (DESIGN.md §6.10 offline build order, GPU-relevant subset):
#
#   A. resolve_upstream_entrypoints --refresh-cache --apply
#        Dual-anchor entrypoint resolution. Writes meta.yaml:
#        oracle.gpu_triton for every oracle-verified task, replacing the
#        ::placeholder entries (95/101 tasks today). The CPU equivalence
#        pre-check anchor runs here; GPU is used for the oracle anchor.
#
#   B. phase_a_prime --rewrite        (== --rewrite __ALL__)
#        For every task: U1 classify -> build severed candidate ->
#        S2-smoke (CPU pre-filter) -> S2-gpu-authoritative verify ->
#        S3 N=5 determinism -> on PASS atomic-promote the real kernel.py
#        and write meta.yaml: phase_a_prime.{status,standalone_sha256}.
#        This is the ONLY path that flips phase_a_prime.status to PASS.
#        On a no-GPU host every verdict is UNRUN and nothing is promoted.
#
#   C. phase_a_prime --run
#        Regenerate the human-readable Phase A' status report
#        (results/.../phase_a_prime.status.md or PLANS_DIR) summarising
#        per-task verdicts + the release-blocker count.
#
# Prerequisites: 00_setup_env.sh done, CUDA visible, infer/ checkouts
# present and pinned clean.
#
# Usage:
#   bash scripts/gpu_pipeline/10_phase_a_prime.sh           # full sweep
#   bash scripts/gpu_pipeline/10_phase_a_prime.sh --task L1_02_xxx
#   FORCE=1 bash scripts/gpu_pipeline/10_phase_a_prime.sh   # ignore markers
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${_HERE}/lib_common.sh"

TASK_FILTER=""
if [ "${1:-}" = "--task" ]; then
    [ -n "${2:-}" ] || die "--task needs a task id"
    TASK_FILTER="$2"
fi

PA="${REPO_ROOT}/scripts/phase_a_prime.py"
[ -f "${PA}" ] || die "missing ${PA}"

step_banner "A" "Phase A' GPU-authoritative reproducible-freeze"

# --- prechecks --------------------------------------------------------------
require_cuda
require_infer_roots

# --- A. resolve upstream entrypoints ----------------------------------------
if is_done "pa_resolve" && [ -z "${TASK_FILTER}" ]; then
    log_skip "entrypoint resolution already done (FORCE=1 to redo)"
else
    step_banner "A.1" "resolve upstream entrypoints (dual-anchor, --apply)"
    if [ -n "${TASK_FILTER}" ]; then
        run_logged "pa_resolve_${TASK_FILTER}" \
            python3 "${PA}" --resolve-entrypoint --task "${TASK_FILTER}" --apply
    else
        # --refresh-cache implies --all + writes the canonical resolver
        # cache; pair with --apply so verified entrypoints land in
        # meta.yaml. resolve_upstream_entrypoints owns --refresh-cache,
        # so call it directly (the phase_a_prime delegate forwards only
        # --task/--all/--apply, not --refresh-cache).
        run_logged "pa_resolve" \
            python3 "${REPO_ROOT}/scripts/resolve_upstream_entrypoints.py" \
            --refresh-cache --apply
    fi
    log_ok "entrypoint resolution complete"
fi

# --- B. S2-gpu-authoritative verify + promote -------------------------------
if is_done "pa_verify" && [ -z "${TASK_FILTER}" ]; then
    log_skip "Phase A' verify+promote already done (FORCE=1 to redo)"
else
    step_banner "A.2" "S2-gpu verify + atomic promote (phase_a_prime --rewrite)"
    log_info "this runs CUDA-authoritative equivalence + N=5 determinism."
    log_info "only S2-gpu-PASS tasks get promoted (phase_a_prime.status=PASS)."
    if [ -n "${TASK_FILTER}" ]; then
        run_logged "pa_verify_${TASK_FILTER}" \
            python3 "${PA}" --rewrite "${TASK_FILTER}"
    else
        # No --task => --rewrite sweeps every task (const="__ALL__").
        run_logged "pa_verify" python3 "${PA}" --rewrite
    fi
    log_ok "Phase A' verify+promote sweep complete"
fi

# --- C. status report -------------------------------------------------------
step_banner "A.3" "regenerate Phase A' status report"
run_logged "pa_status" python3 "${PA}" --run ${TASK_FILTER:+--task "${TASK_FILTER}"}
log_ok "Phase A' status report regenerated"

echo
log_ok "Phase A' GPU stage finished."
log_info "verify promotions: grep 'phase_a_prime:' tasks/*/*/meta.yaml"
log_info "next: bash scripts/gpu_pipeline/20_cost_model_xcheck.sh"
