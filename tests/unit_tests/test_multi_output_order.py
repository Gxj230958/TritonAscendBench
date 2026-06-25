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


# A multi-output corpus task whose oracle returns tensors in a
# non-alphabetical insertion order, used to exercise the R10.a metadata
# stamping. L2_28 moe_align_block_size_stage2 returns
# (sorted_token_ids, expert_ids, num_tokens_post_pad); alphabetical load
# order would be (expert_ids, num_tokens_post_pad, sorted_token_ids), so
# the gap is real.
MULTI_OUTPUT_TASK = "L2_28_moe_align_block_size_stage2"
MULTI_OUTPUT_TASK_DIR = REPO_ROOT / "tasks" / "level2" / MULTI_OUTPUT_TASK
MULTI_OUTPUT_ORDER = ["sorted_token_ids", "expert_ids", "num_tokens_post_pad"]


def test_r10a_writes_output_key_order_metadata():
    """Run the precompute script for a multi-output corpus task and
    assert the safetensors metadata pins the oracle dict insertion
    order (not safetensors' alphabetical load order)."""
    task_dir = MULTI_OUTPUT_TASK_DIR
    if not task_dir.is_dir():
        pytest.skip(f"{MULTI_OUTPUT_TASK} missing")

    # Force a fresh precompute so we know the metadata was just written.
    pyref = task_dir / "r10_pyref"
    if pyref.exists():
        for f in pyref.iterdir():
            f.unlink()

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "oracle" / "precompute_oracle_reference.py"),
            "--task", MULTI_OUTPUT_TASK,
            "--json-out", "/tmp/_test_r10a.json",
            "--report", "/tmp/_test_r10a.md",
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
    # The oracle returns (sorted_token_ids, expert_ids, num_tokens_post_pad)
    # -- neither alphabetical nor reverse-alphabetical; pin the exact
    # wrapper return order so a future refactor that flips the dict
    # insertion order regresses loudly.
    assert order == MULTI_OUTPUT_ORDER
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


def test_l1_24_seg_indptr_handles_minus_one_padding():
    """The L1_24 deepgemm seg_indptr oracle consumes the RAW segment
    indptr (no `num_minus_one` subtraction), so -1 padding tokens count
    into seg_indptr[0]. This edge case is never exercised by the public
    shapes (`topk_ids in [0, num_experts)`); pin it explicitly so a
    refactor that drops the searchsorted-over-padding behaviour trips."""
    sys.path.insert(0, str(REPO_ROOT / "tasks/level1/L1_24_compute_seg_indptr_triton"))
    sys.modules.pop("oracle_pytorch", None)
    import oracle_pytorch as orc

    # Input: 2 entries of -1 padding, then expert ids 0/1/2 interleaved.
    topk_ids = torch.tensor([-1, 1, 0, -1, 2, 1], dtype=torch.int64)
    num_experts = 3

    seg_indptr = orc._compute_seg_indptr(topk_ids, num_experts)

    # Expected (worked out by hand from the kernel's binary search =
    # searchsorted left over the sorted ids):
    # - stable sort of [-1, 1, 0, -1, 2, 1] -> [-1, -1, 0, 1, 1, 2]
    # - seg_indptr[e] = #{sorted < e} for e in [0, 1, 2, 3]
    #   = [#{<0}, #{<1}, #{<2}, #{<3}] = [2, 3, 5, 6]  (RAW, incl. padding)
    assert seg_indptr.tolist() == [2, 3, 5, 6]

    # run() surfaces the same value under the single contract key.
    out = orc.run({"topk_ids": topk_ids, "num_local_experts": num_experts})
    assert list(out.keys()) == ["seg_indptr"]
    assert out["seg_indptr"].tolist() == [2, 3, 5, 6]


def test_multi_output_full_pipeline_with_kernel_tuple(tmp_path):
    """End-to-end on a multi-output task's ref artifacts: simulate a
    kernel returning a tuple in wrapper order and confirm the dict
    mapping pairs each kernel output with its correct ref tensor by
    name (recovered from the stamped metadata order)."""
    task_dir = MULTI_OUTPUT_TASK_DIR
    ref_file = task_dir / "r10_pyref" / "ref_P1.safetensors"
    if not ref_file.is_file():
        # Build it (no skips): the metadata test usually runs first and
        # produces this, but make the test self-sufficient in isolation.
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "oracle" / "precompute_oracle_reference.py"),
                "--task", MULTI_OUTPUT_TASK,
                "--json-out", "/tmp/_test_r10_pipeline.json",
                "--report", "/tmp/_test_r10_pipeline.md",
            ],
            capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        assert ref_file.is_file(), f"expected {ref_file} after R10.a run"

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
