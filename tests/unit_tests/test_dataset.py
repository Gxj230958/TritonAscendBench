"""Unit tests for triton_ascend_bench.dataset."""
from __future__ import annotations

import pytest

from triton_ascend_bench.dataset import iter_all_tasks, load_task


POC_TASK_IDS = [
    "L1_01_bias_kernel",
    "L1_12_rms_norm_kernel",
    "L1_16_silu_and_mul",
    "L2_10_w8a8_block_int8_matmul",
    "L3_04_fwd_kernel",
]


def test_iter_all_tasks_finds_pocs():
    tasks = iter_all_tasks()
    found_ids = {t.task_id for t in tasks}
    for tid in POC_TASK_IDS:
        assert tid in found_ids


def test_load_unknown_task_raises():
    with pytest.raises(FileNotFoundError):
        load_task("DOES_NOT_EXIST")


@pytest.mark.parametrize("task_id", POC_TASK_IDS)
def test_each_task_has_mandatory_meta(task_id):
    task = load_task(task_id)
    assert task.level in {1, 2, 3}
    assert task.tolerance_class
    assert task.oracle_tier in {
        "simple_pytorch_ok",
        "formula_exact_manual",
        "upstream_oracle_required",
    }
    assert task.upstream_sha           # frozen-snapshot placeholder counts
    assert task.kernel_py_sha256       # always derived
    assert len(task.public_shapes) >= 4


@pytest.mark.parametrize("task_id", POC_TASK_IDS)
def test_tolerance_class_exists_in_policy(task_id):
    from triton_ascend_bench.tolerance import load_policy

    pol = load_policy()
    task = load_task(task_id)
    assert task.tolerance_class in pol, (
        f"{task_id} tolerance class {task.tolerance_class!r} "
        f"not in policy: {sorted(pol.keys())}"
    )
