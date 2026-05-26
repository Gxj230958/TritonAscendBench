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

# Self-bootstrap R10 stage-1 (CPU pre-compute) if r10_pyref/ artifacts are
# missing on this host. The artifacts are *.safetensors which the repo
# .gitignore excludes, so a fresh git pull will not transfer them. Re-run
# the pre-compute here; build_inputs uses the fixed GOLDEN_SEED constant
# so the inputs are deterministic across dev/GPU hosts.
if [[ -z $(find tasks/level1 tasks/level2 tasks/level3 \
              -path '*/r10_pyref/ref_*.safetensors' -print -quit 2>/dev/null) ]]; then
  echo "[stage 0] no r10_pyref artifacts found; running CPU pre-compute first..."
  python3 scripts/oracle_pyref_precompute.py --all --workers 8 \
    --json-out /tmp/oracle_pyref_audit.json \
    --report docs/triage_backlog/2026-05-26-oracle-pyref-audit.md
fi

exec python3 scripts/oracle_equivalence_gpu_runner.py \
  --output-dir "$OUTPUT_DIR" \
  "$@"
