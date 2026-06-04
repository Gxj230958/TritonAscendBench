#!/usr/bin/env bash
# 40_release_gate.sh -- run the release-readiness gate and report.
#
# Final pipeline step. Runs the two validate_tasks.py modes:
#
#   1. schema/hash validation -- 5-file contract, schema, ast.parse,
#      kernel.py SHA-256 (validate_tasks.py default mode);
#   2. --release-gate -- the HARD gate (DESIGN.md §4.4): keys on the
#      two real correctness proofs. The release-eligible set is the
#      score-eligible set INTERSECT C2 oracle-equivalence
#      (oracle_kernel_equivalence == VERIFIED/CEILING_ACCEPTED); over
#      that set every task must have release_status == PASS, where
#      release_status == PASS <=> phase_a_prime.status == PASS (S2
#      severance). cost_model.status is ncu-evidenced perf metadata,
#      not a correctness blocker, so it is NOT gated here. A non-empty
#      blocking set => non-zero exit + a blocking task list.
#
# Also runs verify_version_matrix.py so infra/version_matrix.toml is
# checked for missing keys / unresolved placeholders.
#
# This step is EXPECTED to fail (non-zero) until the gating GPU steps
# (10 Phase A' freeze for S2 severance, 30 golden capture for C2
# oracle-equivalence) have actually run on a GPU host and stamped
# their status into meta.yaml -- that is the gate doing its job, not a
# script bug. It is always run so the operator gets the authoritative
# blocking list.
#
# Exit code: non-zero if ANY of the three checks failed -- the version
# matrix check, the schema/hash check, or the release gate (no failure
# may be masked by another check's 0 rc).
#
# Usage:
#   bash scripts/gpu_pipeline/40_release_gate.sh
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_common.sh
source "${_HERE}/lib_common.sh"

VT="${REPO_ROOT}/scripts/validate_tasks.py"
VVM="${REPO_ROOT}/scripts/verify_version_matrix.py"
[ -f "${VT}" ] || die "missing ${VT}"
# VVM is part of the release gate (its rc feeds the exit code below); a
# missing script must hard-fail, never silently leave VVM_RC at 0.
[ -f "${VVM}" ] || die "missing ${VVM}"

step_banner "D" "release-readiness gate"

require_python
# Ensure the log destination exists -- this step writes logs via `tee`
# directly (not run_logged), and may be invoked standalone before any
# other step has created PIPELINE_STATE_DIR.
mkdir -p "${PIPELINE_STATE_DIR}"

# --- 1. version matrix check ------------------------------------------------
VVM_RC=0
if [ -f "${VVM}" ]; then
    step_banner "D.1" "verify version matrix"
    python3 "${VVM}" 2>&1 | tee "${PIPELINE_STATE_DIR}/version_matrix.log" \
        || VVM_RC="${PIPESTATUS[0]}"
    if [ "${VVM_RC}" = "0" ]; then
        log_ok "version matrix consistent"
    else
        log_warn "version matrix check FAILED (rc=${VVM_RC}) -- see log above"
    fi
fi

# --- 2. schema / hash validation --------------------------------------------
step_banner "D.2" "schema + hash validation (validate_tasks.py)"
SCHEMA_RC=0
python3 "${VT}" 2>&1 | tee "${PIPELINE_STATE_DIR}/validate_schema.log" \
    || SCHEMA_RC="${PIPESTATUS[0]}"
if [ "${SCHEMA_RC}" = "0" ]; then
    log_ok "schema/hash validation passed"
else
    log_warn "schema/hash validation FAILED (rc=${SCHEMA_RC}) -- see log"
fi

# --- 3. the hard release gate -----------------------------------------------
step_banner "D.3" "release gate (validate_tasks.py --release-gate)"
GATE_RC=0
python3 "${VT}" --release-gate 2>&1 \
    | tee "${PIPELINE_STATE_DIR}/release_gate.log" || GATE_RC="${PIPESTATUS[0]}"

echo
if [ "${GATE_RC}" = "0" ] && [ "${SCHEMA_RC}" = "0" ] && [ "${VVM_RC}" = "0" ]; then
    log_ok "RELEASE GATE: PASS -- all score-eligible tasks are release-ready."
    mark_done "release_gate"
    exit 0
fi

log_err "RELEASE GATE: NOT PASS."
log_err "  version-matrix rc = ${VVM_RC}"
log_err "  schema/hash rc = ${SCHEMA_RC}"
log_err "  release-gate rc = ${GATE_RC}"
log_err ""
log_err "  This is expected until the gating GPU steps (10 Phase A' freeze,"
log_err "  30 golden capture) have actually run on a GPU host -- the gate is"
log_err "  correctly blocking, not a script bug."
log_err "  The blocking task list is in ${PIPELINE_STATE_DIR}/release_gate.log"

# Exit non-zero if ANY stage failed. A version-matrix or schema/hash
# failure with a 0 release-gate rc must NOT exit 0 -- that would let
# run_all.sh record step 40 as PASS (a false green). Prefer the
# release-gate rc when it is itself non-zero so the operator sees the
# gate's own status code; otherwise surface whichever check failed.
if [ "${GATE_RC}" != "0" ]; then
    exit "${GATE_RC}"
fi
if [ "${SCHEMA_RC}" != "0" ]; then
    exit "${SCHEMA_RC}"
fi
# GATE_RC and SCHEMA_RC are 0 here, so VVM_RC must be the non-zero one.
exit "${VVM_RC:-1}"
