#!/usr/bin/env bash
# oracle_equivalence_verify.sh -- R10 GPU equivalence verify (V9 M2.1b).
#
# Standalone GPU-host operator script. Not part of run_all.sh numbered
# pipeline (00/10/20/30/40); run this on demand.
#
# Reads precomputed PyTorch reference artifacts (produced by dev-side
# scripts/oracle_pyref_precompute.py) from tasks/<id>/r10_pyref/ and
# compares against kernel.py upstream entrypoint on real GPU.
#
# Usage on GPU host:
#   bash scripts/gpu_pipeline/oracle_equivalence_verify.sh --task L1_12_rms_norm
#   bash scripts/gpu_pipeline/oracle_equivalence_verify.sh --task-list L1_01,L1_12,L1_16
#   bash scripts/gpu_pipeline/oracle_equivalence_verify.sh --all
#
# Output:
#   Per-task verdict JSON at /tmp/r10_verdict/<task_id>.json containing
#   per-shape PASS/FAIL + numeric diff (max_abs, max_rel) per output.
#   Operator then rsyncs /tmp/r10_verdict/ back to dev machine for
#   apply_oracle_equivalence_verdict.py to write into meta.yaml.
#
# Prerequisites:
#   1. infer/ sibling dir present at <repo>/../infer/{vllm,sglang}
#      (same as run_all.sh and 10_phase_a_prime.sh)
#   2. tasks/<id>/r10_pyref/{inputs_<sid>,ref_<sid>}.safetensors committed
#      or rsync'd from dev (produced by oracle_pyref_precompute.py)
#   3. GPU available (torch.cuda.is_available() == True)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Quick GPU sanity check
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "ERROR: torch.cuda.is_available() is False; this script requires a GPU host." >&2
  exit 2
fi

# Quick infer/ presence check
if [[ ! -d "$REPO_ROOT/../infer/vllm" ]] && [[ ! -d "$REPO_ROOT/../infer/sglang/python" ]]; then
  echo "ERROR: infer/ sibling dir missing at $REPO_ROOT/../infer/{vllm,sglang}" >&2
  echo "Same as run_all.sh: clone vllm + sglang into sibling directory first." >&2
  exit 2
fi

OUTPUT_DIR="${OUTPUT_DIR:-/tmp/r10_verdict}"
mkdir -p "$OUTPUT_DIR"

cd "$REPO_ROOT"

# Self-bootstrap R10 stage-1 (CPU pre-compute) for any task missing its
# r10_pyref/ artifacts. The artifacts are *.safetensors which the repo
# .gitignore excludes, so a fresh git pull or a manual `rm -rf` of one
# task's r10_pyref leaves the GPU runner with nothing to verify. We
# now scan TWO modes:
#
#   1. --task / --task-list explicit list: walk the list, precompute
#      each task whose r10_pyref is empty.
#   2. --all (or no task-selector): if NO task has ref_*.safetensors,
#      precompute the whole corpus once.
#
# build_inputs uses the fixed GOLDEN_SEED constant so inputs are
# deterministic across dev/GPU hosts.

# Parse the verify args to find an explicit task selector. We don't
# consume the args -- they are forwarded verbatim to the python runner.
# Codex review 2026-05-27:
#   * ``--task-list all`` is supported by the python runner as a
#     synonym for --all; treat it that way here too so we don't try
#     to precompute a task literally named ``all``.
#   * A no-arg invocation must keep the original ``--all``-style
#     bootstrap so a fresh GPU host can rebuild every ref. Default
#     ``_saw_all=1`` when no selector was passed.
_explicit_tasks=""
_saw_all=0
_saw_any_selector=0
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      _saw_any_selector=1
      _explicit_tasks+="${_explicit_tasks:+,}$2"
      ARGS+=("$1" "$2"); shift 2;;
    --task-list)
      _saw_any_selector=1
      if [[ "$2" == "all" ]]; then
        _saw_all=1
      else
        _explicit_tasks+="${_explicit_tasks:+,}$2"
      fi
      ARGS+=("$1" "$2"); shift 2;;
    --all)
      _saw_any_selector=1
      _saw_all=1
      ARGS+=("$1"); shift;;
    *)
      ARGS+=("$1"); shift;;
  esac
done
# No --task / --task-list / --all -> emulate the historical "run
# everything" behaviour so a fresh host re-bootstraps automatically.
if [[ "$_saw_any_selector" -eq 0 ]]; then
  _saw_all=1
fi

_precompute_missing() {
  local missing=()
  IFS=',' read -ra _tids <<<"$1"
  for tid in "${_tids[@]}"; do
    # locate task dir under tasks/level{1,2,3}/
    local td=""
    for level in level1 level2 level3; do
      if [[ -d "tasks/$level/$tid" ]]; then td="tasks/$level/$tid"; break; fi
    done
    if [[ -z "$td" ]]; then
      echo "[stage 0] WARN: task $tid not present in tasks/; skipping" >&2
      continue
    fi
    # Refresh if ref tensors OR the H2 input sidecar is missing (legacy
    # artifacts predate the sidecar; R10.b blocks them INPUT_DRIFT otherwise).
    if ! compgen -G "$td/r10_pyref/ref_*.safetensors" > /dev/null \
       || ! compgen -G "$td/r10_pyref/inputs_*.sidecar.json" > /dev/null; then
      missing+=("$tid")
    fi
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "[stage 0] re-precomputing R10.a for ${#missing[@]} task(s): ${missing[*]}"
    for tid in "${missing[@]}"; do
      python3 scripts/oracle_pyref_precompute.py --task "$tid" \
        --json-out "/tmp/oracle_pyref_audit_${tid}.json" \
        --report "docs/triage_backlog/2026-05-26-oracle-pyref-audit.md" \
        || echo "[stage 0] WARN: R10.a failed for $tid (continuing)" >&2
    done
  fi
}

if [[ -n "$_explicit_tasks" ]]; then
  _precompute_missing "$_explicit_tasks"
elif [[ "$_saw_all" -eq 1 ]]; then
  if [[ -z $(find tasks/level1 tasks/level2 tasks/level3 \
              -path '*/r10_pyref/ref_*.safetensors' -print -quit 2>/dev/null) ]]; then
    echo "[stage 0] no r10_pyref artifacts found anywhere; running CPU pre-compute first..."
    python3 scripts/oracle_pyref_precompute.py --all --workers 8 \
      --json-out /tmp/oracle_pyref_audit.json \
      --report docs/triage_backlog/2026-05-26-oracle-pyref-audit.md
  else
    # Legacy artifacts may have ref_* but no inputs_*.sidecar.json (H2);
    # refresh per-task so R10.b doesn't block them as INPUT_DRIFT.
    # Comma-separated to match _precompute_missing's IFS=',' split.
    echo "[stage 0] artifacts present; refreshing tasks missing ref/sidecar..."
    _all_ids=$(ls -d tasks/level*/L*/ 2>/dev/null | xargs -n1 basename | paste -sd,)
    _precompute_missing "$_all_ids"
  fi
fi

exec python3 scripts/oracle_equivalence_gpu_runner.py \
  --output-dir "$OUTPUT_DIR" \
  "${ARGS[@]}"
