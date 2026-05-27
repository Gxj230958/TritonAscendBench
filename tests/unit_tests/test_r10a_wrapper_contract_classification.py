"""Integration tests for V9 P3 -- R10.a wrapper-contract classification.

Verifies that the 7 GPU-host L3 tasks that previously LAUNCH_BLOCKED
with ``not enough values to unpack`` get the right new R10.a status:

* L3_13 -> ORACLE_KERNEL_LAYOUT_MISMATCH (oracle is 3-D, wrapper 4-D)
* L3_16 / L3_17 / L3_18 / L3_20 / L3_22 / L3_23 -> WRAPPER_CONTRACT_MISMATCH

The L3_16/17 case is a known partial false positive (their oracle uses
einsum subscripts that hard-code 3-D; the static checker doesn't parse
einsum so credits them as rank-agnostic). The status's ``suggested_fix``
text instructs the operator to re-run R10.a after patching shapes and
reclassify if the oracle then RUNTIME_ERROR's.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PRECOMPUTE = REPO_ROOT / "scripts" / "oracle_pyref_precompute.py"


CLASSIFICATION_CASES = [
    # L3_22 was patched 2026-05-27 (oracle rewritten as per-chunk
    # cumsum + shapes.yaml aligned to [B,H,L,D]); it now returns
    # ALIGNED_READY and is no longer expected in this rank-mismatch
    # bucket.
    ("L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        "ORACLE_KERNEL_LAYOUT_MISMATCH"),
    ("L3_16_chunk_fwd_kernel_o", "WRAPPER_CONTRACT_MISMATCH"),
    ("L3_17_chunk_gla_fwd_kernel_o", "WRAPPER_CONTRACT_MISMATCH"),
    ("L3_18_chunk_scaled_dot_kkt_fwd", "WRAPPER_CONTRACT_MISMATCH"),
    ("L3_20_recompute_w_u_fwd", "WRAPPER_CONTRACT_MISMATCH"),
    ("L3_23_merge_16x16_to_32x32_inverse", "WRAPPER_CONTRACT_MISMATCH"),
]


def test_l3_22_now_aligned():
    """Spot check: L3_22 was patched out of the WRAPPER_CONTRACT_MISMATCH
    bucket on 2026-05-27 (oracle rewritten as per-chunk cumsum, shapes
    aligned to [B,H,L,D]). Guard against regression that reintroduces
    a rank-mismatch verdict here."""
    import sys

    td = REPO_ROOT / "tasks" / "level3" / "L3_22_chunk_local_cumsum_vector"
    if not td.is_dir():
        pytest.skip("L3_22 missing")
    proc = subprocess.run(
        [
            sys.executable,
            str(PRECOMPUTE),
            "--task", "L3_22_chunk_local_cumsum_vector",
            "--json-out", "/tmp/_l3_22_test.json",
            "--report", "/tmp/_l3_22_test.md",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    assert proc.returncode == 0
    combined = proc.stdout + proc.stderr
    assert "ALIGNED_READY" in combined, (
        f"L3_22 should now be ALIGNED_READY: {combined}"
    )
    assert "WRAPPER_CONTRACT_MISMATCH" not in combined


@pytest.mark.parametrize("task_id,expected_status", CLASSIFICATION_CASES)
def test_value_error_tasks_get_new_r10a_status(
    task_id, expected_status, tmp_path
):
    """Run the real R10.a precompute script for each task and assert
    the new wrapper-contract classifier picks the right bucket."""
    td = REPO_ROOT / "tasks" / "level3" / task_id
    if not td.is_dir():
        pytest.skip(f"corpus task {task_id} missing")
    json_out = tmp_path / "audit.json"
    report = tmp_path / "report.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(PRECOMPUTE),
            "--task", task_id,
            "--json-out", str(json_out),
            "--report", str(report),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"precompute exited {proc.returncode}: {proc.stderr}"
    )
    # The single-task INFO line carries the verdict.
    assert expected_status in proc.stdout or expected_status in proc.stderr, (
        f"expected status {expected_status} not found in output:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


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
