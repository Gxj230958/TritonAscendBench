"""Unit tests for triton_ascend_bench.dataset."""
from __future__ import annotations

import shutil

import pytest
import yaml

from triton_ascend_bench import dataset as dataset_mod
from triton_ascend_bench.dataset import iter_all_tasks, load_task

POC_TASK_IDS = [
    "L1_01_bias",
    "L1_12_rms_norm",
    "L1_16_silu_and_mul",
    "L2_10_w8a8_block_int8_matmul",
    "L3_04_fwd",
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


# --- W1.2: cost_model parsing on TaskDef -----------------------------------

def test_cost_model_missing_is_none(tmp_path, monkeypatch):
    # Backward compat: a task whose meta.yaml has no `cost_model` block
    # loads fine with `TaskDef.cost_model = None` — the loader must never
    # crash on an absent block. With the W3.1 cost_model fill every live
    # corpus task now carries a cost_model, so this loader contract is
    # exercised against a synthetic task dir (a real task dir copied with
    # its `cost_model` block stripped) rather than a live task.
    src = dataset_mod.TASKS_ROOT / "level1" / "L1_16_silu_and_mul"
    fake_root = tmp_path / "tasks"
    dst = fake_root / "level1" / "L1_16_silu_and_mul"
    dst.parent.mkdir(parents=True)
    shutil.copytree(src, dst)
    meta = yaml.safe_load((dst / "meta.yaml").read_text())
    meta.pop("cost_model", None)
    (dst / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    monkeypatch.setattr(dataset_mod, "TASKS_ROOT", fake_root)

    task = load_task("L1_16_silu_and_mul")
    assert task.cost_model is None


def test_cost_model_scalar_form_parsed():
    # L1_01 carries a hand-authored scalar-form cost_model (W1.2 sample).
    task = load_task("L1_01_bias")
    assert task.cost_model is not None
    assert task.cost_model.flops_scalar is True
    assert task.cost_model.bytes_scalar is True
    # scalar form is normalised to a one-entry map keyed by the companion
    # class / resource — a real peak-table key (bias is a vector-unit
    # elementwise op over HBM-resident tensors).
    assert set(task.cost_model.flops) == {"vector"}
    assert set(task.cost_model.bytes) == {"hbm"}


def test_cost_model_map_form_parsed():
    # L1_12 carries a hand-authored map-form cost_model (W1.2 sample).
    task = load_task("L1_12_rms_norm")
    assert task.cost_model is not None
    assert task.cost_model.flops_scalar is False
    assert task.cost_model.bytes_scalar is False
    assert "vector" in task.cost_model.flops
    assert "hbm" in task.cost_model.bytes


def test_iter_all_tasks_tolerates_missing_cost_model():
    # The shared `iter_all_tasks` path must not crash on the (majority)
    # pre-W3.1 tasks that have no cost_model block.
    tasks = iter_all_tasks()
    assert len(tasks) > 0
    with_cm = [t for t in tasks if t.cost_model is not None]
    # at least the two W1.2 sample tasks carry one
    assert {"L1_01_bias", "L1_12_rms_norm"} <= {
        t.task_id for t in with_cm
    }
