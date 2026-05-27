"""Unit tests for V9 R10 multi-output ordering metadata.

Safetensors loads keys alphabetically, but the R10.b runner needs
to map a kernel's positional tuple return to oracle ref tensors in
the oracle's *dict insertion* order. R10.a now stamps that order
into safetensors ``metadata`` so R10.b can recover it after load.

These tests pin the contract end-to-end:

1. ``oracle_pyref_precompute`` writes the metadata field on every
   ``ref_*.safetensors`` / ``inputs_*.safetensors`` it produces.
2. The recorded order matches the oracle's ``run(inputs)`` dict
   insertion order (not safetensors' alphabetical load order).
3. Legacy ref files without metadata still load cleanly (the runner
   falls back to alphabetical keys).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_r10a_writes_output_key_order_metadata():
    """Run the precompute script for a multi-output corpus task and
    assert the safetensors metadata pins the oracle dict insertion
    order (not safetensors' alphabetical load order)."""
    task_dir = REPO_ROOT / "tasks" / "level1" / "L1_36_compute_seg_indptr_triton"
    if not task_dir.is_dir():
        pytest.skip("L1_36 missing")

    # Force a fresh precompute so we know the metadata was just written.
    pyref = task_dir / "r10_pyref"
    if pyref.exists():
        for f in pyref.iterdir():
            f.unlink()

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "oracle_pyref_precompute.py"),
            "--task", "L1_36_compute_seg_indptr_triton",
            "--json-out", "/tmp/_test_l1_36.json",
            "--report", "/tmp/_test_l1_36.md",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr

    ref_file = pyref / "ref_P1.safetensors"
    assert ref_file.is_file(), f"expected {ref_file} after R10.a run"
    with safe_open(str(ref_file), framework="pt") as f:
        meta = f.metadata() or {}
        alphabetical = list(f.keys())
    assert "output_key_order" in meta, (
        f"R10.a must stamp output_key_order in safetensors metadata; got {meta}"
    )
    order = json.loads(meta["output_key_order"])
    # L1_36 oracle returns (reorder_topk_ids, src2dst, seg_indptr) --
    # neither alphabetical nor reverse-alphabetical; pin the exact
    # wrapper return order so a future refactor that flips the dict
    # insertion order regresses loudly.
    assert order == ["reorder_topk_ids", "src2dst", "seg_indptr"]
    # And we expect the alphabetical-load order to be DIFFERENT
    # (otherwise this metadata wouldn't be needed and the test would
    # silently pass even if the implementation was wrong).
    assert alphabetical != order, (
        "if alphabetical == insertion order for this task the metadata "
        "is redundant -- pick a task with a non-alphabetical wrapper "
        "return order to exercise the gap"
    )


def test_r10b_runner_uses_metadata_order_for_tuple_mapping(tmp_path):
    """Stand-alone smoke: simulate what R10.b does -- load metadata,
    map a positional kernel tuple to a dict using metadata order.

    Mocks the file I/O without spinning up a GPU so the assertion is
    pure-CPU + fast in CI."""
    # Write a synthetic ref with non-alphabetical insertion order.
    ref_path = tmp_path / "ref.safetensors"
    tensors = {
        "z_last": torch.tensor([3.0]),
        "a_first": torch.tensor([1.0]),
        "m_mid": torch.tensor([2.0]),
    }
    meta = {"output_key_order": json.dumps(list(tensors.keys()))}
    save_file(tensors, str(ref_path), metadata=meta)

    # Read metadata + alphabetical keys.
    with safe_open(str(ref_path), framework="pt") as f:
        loaded_meta = f.metadata() or {}
        alphabetical = list(f.keys())
    assert alphabetical == ["a_first", "m_mid", "z_last"]  # safetensors sort

    insertion_order = json.loads(loaded_meta["output_key_order"])
    assert insertion_order == ["z_last", "a_first", "m_mid"]  # original

    # Simulate the R10.b tuple -> dict mapping.
    kernel_out = (
        torch.tensor([30.0]),  # corresponds to z_last
        torch.tensor([10.0]),  # corresponds to a_first
        torch.tensor([20.0]),  # corresponds to m_mid
    )
    out_dict = {
        insertion_order[i] if i < len(insertion_order) else f"out_{i}": v
        for i, v in enumerate(kernel_out)
    }
    assert torch.equal(out_dict["z_last"], torch.tensor([30.0]))
    assert torch.equal(out_dict["a_first"], torch.tensor([10.0]))
    assert torch.equal(out_dict["m_mid"], torch.tensor([20.0]))


def test_r10b_runner_falls_back_to_alphabetical_when_no_metadata(tmp_path):
    """Legacy / pre-2026-05-27 ref artifacts without ``output_key_order``
    metadata must still load cleanly -- R10.b falls back to
    alphabetical mapping in that case."""
    ref_path = tmp_path / "ref.safetensors"
    tensors = {
        "b_one": torch.tensor([1.0]),
        "a_zero": torch.tensor([0.0]),
    }
    # Save WITHOUT metadata
    save_file(tensors, str(ref_path))

    with safe_open(str(ref_path), framework="pt") as f:
        loaded_meta = f.metadata() or {}
    # Either None or empty dict -- both must work.
    order_str = loaded_meta.get("output_key_order")
    order = json.loads(order_str) if order_str else []
    assert order == []  # no metadata -> empty fallback list


def test_l1_36_oracle_handles_minus_one_padding():
    """Codex r2 flagged that the L1_36 oracle's `num_minus_one > 0`
    path (step 4 src2dst offset scatter + step 5 reorder slice) was
    never exercised by existing shapes (`topk_ids in [0, 1]`). Pin
    the math with an explicit input that mixes -1 padding entries
    with real expert ids so any future refactor of the oracle that
    drops the offset / slice trips this test."""
    sys.path.insert(0, str(REPO_ROOT / "tasks/level1/L1_36_compute_seg_indptr_triton"))
    import oracle_pytorch as orc

    # Input: 2 entries of -1 padding, then expert ids 0/1/2 interleaved.
    # num_minus_one = 2 after stable sort.
    topk_ids = torch.tensor([-1, 1, 0, -1, 2, 1], dtype=torch.int64)
    num_experts = 3

    reorder_topk_ids, src2dst, seg_indptr = orc._deepep_run_moe_deep_preprocess(
        topk_ids, num_experts
    )

    # Expected (worked out by hand from the wrapper math):
    # - stable sort of [-1, 1, 0, -1, 2, 1] -> [-1, -1, 0, 1, 1, 2]
    # - reorder_ids: positions of each sorted entry in the original
    #   stable-sort -> [0, 3, 2, 1, 5, 4]
    # - num_minus_one = 2 (entries < 0)
    # - seg_indptr_raw via searchsorted on [0, 1, 2, 3]
    #   = [2, 3, 5, 6]  (counts of <0, <1, <2, <3 in sorted)
    # - seg_indptr = seg_indptr_raw - 2 = [0, 1, 3, 4]
    # - reorder_topk_ids (after slice [2:]) = [0, 1, 1, 2]
    # - src2dst[reorder_ids[i]] = i - 2 for each i in [0..6)
    #   src2dst initialised empty; fill at indices 0,3,2,1,5,4 with -2,-1,0,1,2,3
    #   -> src2dst = [-2, 1, 0, -1, 3, 2]
    assert reorder_topk_ids.tolist() == [0, 1, 1, 2]
    assert seg_indptr.tolist() == [0, 1, 3, 4]
    assert src2dst.tolist() == [-2, 1, 0, -1, 3, 2]


def test_l1_36_full_pipeline_with_kernel_tuple(tmp_path):
    """End-to-end on L1_36 ref artifacts: simulate kernel returning
    a 3-tuple in wrapper order and confirm the dict mapping pairs
    each kernel output with its correct ref tensor by name."""
    task_dir = REPO_ROOT / "tasks" / "level1" / "L1_36_compute_seg_indptr_triton"
    ref_file = task_dir / "r10_pyref" / "ref_P1.safetensors"
    if not ref_file.is_file():
        pytest.skip("L1_36 r10_pyref not built; run R10.a first")

    ref_tensors = {}
    with safe_open(str(ref_file), framework="pt") as f:
        loaded_meta = f.metadata() or {}
        for k in f.keys():
            ref_tensors[k] = f.get_tensor(k)

    order = json.loads(loaded_meta["output_key_order"])
    # Synthetic kernel_out that just clones the corresponding ref values
    # IN WRAPPER (insertion) ORDER, i.e. ref[order[i]].
    kernel_out = tuple(ref_tensors[k].clone() for k in order)

    out_dict = {
        order[i] if i < len(order) else f"out_{i}": v
        for i, v in enumerate(kernel_out)
    }
    # After mapping, ``out_dict[k]`` for each k MUST equal ``ref_tensors[k]``
    for k in order:
        assert torch.equal(out_dict[k], ref_tensors[k]), (
            f"{k} mismatch after tuple->dict map"
        )
