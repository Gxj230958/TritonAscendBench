#!/usr/bin/env bash
# run_all.sh -- master orchestrator for the triton-ascend-bench GPU pipeline.
#
# Runs every GPU-host release-readiness step in dependency order. Each
# step is also independently runnable; this just sequences them and
# produces one consolidated summary at the end.
#
# Order (see DESIGN.md §6.10 offline build pipeline):
#   00_setup_env.sh        environment + frozen-stack install   [ready]
#   10_phase_a_prime.sh    resolve entrypoints -> S2-gpu verify
#                          -> promote phase_a_prime.status      [ready]
#   20_cost_model_xcheck.sh  cost-model GPU cross-check          [ready]
#   30_golden_capture.sh   upstream_oracle_required golden       [ready]
#   40_release_gate.sh     validate_tasks --release-gate         [ready]
#
# Every step is implemented. If a ready step fails, run_all.sh stops
# immediately so the operator fixes it before later steps run on bad
# state. The release gate (step 40) is EXPECTED to be red until every
# earlier step has actually run on a GPU host and stamped its status
# into meta.yaml -- that is the gate working, not step 40 failing, and
# the summary says so. run_all.sh NEVER reports a green pipeline while
# the gate is red.
#
# Idempotent: finished steps are skipped via done-markers in
# results/gpu_pipeline/. `--fresh` clears them; `FORCE=1` ignores them.
#
# Usage:
#   bash scripts/gpu_pipeline/run_all.sh            # full pipeline
#   bash scripts/gpu_pipeline/run_all.sh --fresh    # wipe markers, redo all
#   bash scripts/gpu_pipeline/run_all.sh --from 10  # start at step 10
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${_HERE}/lib_common.sh"

FROM_STEP="00"
for arg in "$@"; do
    case "${arg}" in
        --fresh) clear_markers ;;
        --from)  FROM_STEP="__NEXT__" ;;
        *)
            if [ "${FROM_STEP}" = "__NEXT__" ]; then
                FROM_STEP="${arg}"
            else
                die "unknown argument: ${arg} (use --fresh | --from <NN>)"
            fi
            ;;
    esac
done

# Step registry: id | script | human label
STEPS=(
    "00|00_setup_env.sh|environment setup"
    "10|10_phase_a_prime.sh|Phase A' GPU-authoritative freeze"
    "20|20_cost_model_xcheck.sh|cost-model GPU cross-check"
    "30|30_golden_capture.sh|offline golden capture"
    "40|40_release_gate.sh|release-readiness gate"
)

mkdir -p "${PIPELINE_STATE_DIR}"
SUMMARY=()      # "id|label|verdict"
OVERALL_OK=1
GATE_RED=0      # 1 iff step 40 (release gate) finished red

printf '\n%s########################################################%s\n' \
    "${_C_BLD}" "${_C_RST}"
printf '%s# triton-ascend-bench GPU pipeline -- run_all.sh%s\n' \
    "${_C_BLD}" "${_C_RST}"
printf '%s#   repo : %s%s\n' "${_C_BLD}" "${REPO_ROOT}" "${_C_RST}"
printf '%s#   from : step %s%s\n' "${_C_BLD}" "${FROM_STEP}" "${_C_RST}"
printf '%s########################################################%s\n' \
    "${_C_BLD}" "${_C_RST}"

for entry in "${STEPS[@]}"; do
    IFS='|' read -r sid sscript slabel <<< "${entry}"
    [ "${sid}" -lt "${FROM_STEP}" ] 2>/dev/null && continue

    rc=0
    bash "${_HERE}/${sscript}" || rc=$?

    if [ "${rc}" = "0" ]; then
        SUMMARY+=("${sid}|${slabel}|PASS")
    elif [ "${sid}" = "40" ]; then
        # The release gate finished red. The gate is the bench's hard
        # release predicate -- a red gate means score-eligible tasks are
        # not yet release-ready (e.g. an earlier step has not been run on
        # this host yet). That is the gate doing its job, not step 40
        # itself being broken, so it is not a pipeline-stopping failure;
        # it is reported as RED and run_all.sh exits non-zero.
        SUMMARY+=("${sid}|${slabel}|RED (release gate blocking)")
        GATE_RED=1
        log_warn "step ${sid} release gate is RED -- score-eligible tasks"
        log_warn "  are not all release-ready (see release_gate.log)."
    else
        # A genuine failure of a ready step: stop immediately so the
        # operator fixes it before later steps run on bad state.
        SUMMARY+=("${sid}|${slabel}|FAIL (rc=${rc})")
        OVERALL_OK=0
        log_err "step ${sid} (${slabel}) FAILED with rc=${rc} -- stopping."
        break
    fi
done

# --- consolidated summary ---------------------------------------------------
printf '\n%s########################################################%s\n' \
    "${_C_BLD}" "${_C_RST}"
printf '%s# PIPELINE SUMMARY%s\n' "${_C_BLD}" "${_C_RST}"
printf '%s########################################################%s\n' \
    "${_C_BLD}" "${_C_RST}"
for row in "${SUMMARY[@]}"; do
    IFS='|' read -r sid slabel sverdict <<< "${row}"
    case "${sverdict}" in
        PASS*) log_ok   "step ${sid}  ${slabel}  -> ${sverdict}" ;;
        RED*)  log_warn "step ${sid}  ${slabel}  -> ${sverdict}" ;;
        *)     log_err  "step ${sid}  ${slabel}  -> ${sverdict}" ;;
    esac
done

echo
if [ "${OVERALL_OK}" = "0" ]; then
    log_err "PIPELINE: a ready step FAILED. Fix it and re-run (idempotent)."
    exit 1
fi
if [ "${GATE_RED}" = "1" ]; then
    log_warn "PIPELINE: every step ran, but the release gate (step 40) is RED:"
    log_warn "  one or more score-eligible tasks are not release-ready."
    log_warn "  The gate is correctly blocking, not failing. The authoritative"
    log_warn "  blocking task list is in:"
    log_warn "    ${PIPELINE_STATE_DIR}/release_gate.log"
    log_warn "  Address the listed tasks (re-run the relevant earlier step on"
    log_warn "  this GPU host if a status is missing) and re-run -- the gate"
    log_warn "  turns green when every score-eligible task is truly ready."
    exit 2
fi
log_ok "PIPELINE: every step PASSED -- benchmark is release-ready."
exit 0
