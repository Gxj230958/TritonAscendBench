"""Integration tests for the R10.a wrapper-contract classification.

A set of L3 tasks once LAUNCH_BLOCKED with ``not enough values to
unpack`` because their oracle/inputs were 3-D while the kernel wrapper
unpacked a 4-D shape. They have since been re-authored (oracle rewritten
+ shapes.yaml aligned to the wrapper's 4-D [B, H, L, D] layout) and now
return ALIGNED_READY.

The spot-check tests below run the real R10.a precompute script and
assert these recovered tasks stay ALIGNED_READY -- a regression here
means an oracle/shapes patch was reverted or a new wrapper-contract
heuristic mis-classifies them.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PRECOMPUTE = REPO_ROOT / "scripts" / "oracle" / "precompute_oracle_reference.py"


@pytest.mark.parametrize(
    "task_id",
    [
        # Re-authored: oracle rewritten + shapes.yaml aligned to the
        # wrapper-required 4-D layouts. Spot-check guards against a
        # regression that reintroduces a rank-mismatch verdict.
        "L2_35_chunk_local_cumsum_vector",
        "L2_34_solve_tril_16x16",
        "L3_30_merge_16x16_to_32x32_inverse",
        # Previously classified WRAPPER_CONTRACT_MISMATCH /
        # ORACLE_KERNEL_LAYOUT_MISMATCH; their oracles + inputs were
        # re-authored to 4-D and they now return ALIGNED_READY.
        "L3_22_chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        "L3_25_chunk_fwd_kernel_o",
        "L3_26_chunk_gla_fwd_kernel_o",
        "L3_27_chunk_scaled_dot_kkt_fwd",
        "L3_29_recompute_w_u_fwd",
    ],
)
def test_recovered_tasks_now_aligned(task_id):
    """Spot check tasks that were patched out of the
    WRAPPER_CONTRACT_MISMATCH / ORACLE_KERNEL_LAYOUT_MISMATCH buckets.
    A regression here means the oracle/shapes patch was reverted or a
    new wrapper-contract heuristic mis-classifies them."""
    # task_id prefix encodes the level (``L2_...`` -> level2).
    level = f"level{task_id.split('_', 1)[0][1:]}"
    td = REPO_ROOT / "tasks" / level / task_id
    if not td.is_dir():
        pytest.skip(f"{task_id} missing")
    proc = subprocess.run(
        [
            sys.executable,
            str(PRECOMPUTE),
            "--task", task_id,
            "--json-out", f"/tmp/_{task_id}_test.json",
            "--report", f"/tmp/_{task_id}_test.md",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        # L3_29 reference precompute runs ~75s; keep generous headroom.
        timeout=180,
    )
    assert proc.returncode == 0
    combined = proc.stdout + proc.stderr
    assert "ALIGNED_READY" in combined, (
        f"{task_id} should now be ALIGNED_READY: {combined}"
    )
    assert "WRAPPER_CONTRACT_MISMATCH" not in combined
    assert "ORACLE_KERNEL_LAYOUT_MISMATCH" not in combined


def test_wrapper_contract_classifier_does_not_regress_aligned_tasks():
    """Tasks already at ALIGNED_READY (with built refs) must keep that
    status -- the new checker only flips on real rank mismatches."""
    # L1_12_rms_norm is a stable ALIGNED_READY case.
    proc = subprocess.run(
        [
            sys.executable,
            str(PRECOMPUTE),
            "--task", "L1_12_rms_norm",
            "--json-out", "/tmp/test_aligned.json",
            "--report", "/tmp/test_aligned.md",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    assert proc.returncode == 0
    combined = proc.stdout + proc.stderr
    assert "ALIGNED_READY" in combined, (
        f"L1_12_rms_norm should remain ALIGNED_READY: {combined}"
    )
    assert "WRAPPER_CONTRACT_MISMATCH" not in combined
    assert "ORACLE_KERNEL_LAYOUT_MISMATCH" not in combined
