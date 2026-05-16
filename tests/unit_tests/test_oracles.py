"""Unit tests for each PoC task's oracle_pytorch.run() — confirm that the
reference oracle produces output of the right shape/dtype on at least one
public shape.

These tests are CPU-only (no triton); they validate the oracle in isolation.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from triton_ascend_bench.dataset import load_task


def _load_oracle_module(task):
    spec = importlib.util.spec_from_file_location(
        f"oracle_{task.task_id}",
        task.task_dir / "oracle_pytorch.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_l1_12_rms_norm_oracle_shape_dtype():
    task = load_task("L1_12_rms_norm_kernel")
    mod = _load_oracle_module(task)
    x = torch.randn(4, 4096, dtype=torch.float16)
    w = torch.ones(4096, dtype=torch.float16)
    out = mod.run({"input": x, "weight": w, "eps": 1e-6})["output"]
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_l1_12_rms_norm_metamorphic_weight_one():
    """rms_norm(x, ones, eps≈0) should equal x / rms(x) — exercise the
    metamorphic-check definition."""
    task = load_task("L1_12_rms_norm_kernel")
    mod = _load_oracle_module(task)
    x = torch.randn(2, 128, dtype=torch.float32)
    w = torch.ones(128, dtype=torch.float32)
    out = mod.run({"input": x, "weight": w, "eps": 1e-12})["output"]
    rms = (x ** 2).mean(dim=-1, keepdim=True).sqrt()
    expected = x / rms
    assert torch.allclose(out, expected, atol=1e-4, rtol=1e-4)


def test_l1_16_silu_and_mul_oracle_zero_x1():
    """If x1 = 0, silu(0)=0 → output is zero everywhere."""
    task = load_task("L1_16_silu_and_mul")
    mod = _load_oracle_module(task)
    h = torch.zeros(8, 4096, dtype=torch.float32)
    h[:, 2048:] = 1.0   # x3 = 1, x1 = 0
    out = mod.run({"hidden_states": h})["output"]
    assert torch.allclose(out, torch.zeros_like(out))


def test_l3_04_fwd_attention_single_token_per_seq():
    """Each sequence has length 1 ⇒ output == v."""
    task = load_task("L3_04_fwd_kernel")
    mod = _load_oracle_module(task)
    batch, h, d = 4, 8, 64
    total = batch  # 1 token per seq
    q = torch.randn(total, h, d, dtype=torch.float32)
    k = torch.randn(total, h, d, dtype=torch.float32)
    v = torch.randn(total, h, d, dtype=torch.float32)
    b_start = torch.arange(batch, dtype=torch.int32)
    b_seqlen = torch.ones(batch, dtype=torch.int32)
    out = mod.run({
        "q": q, "k": k, "v": v,
        "b_start_loc": b_start, "b_seq_len": b_seqlen,
        "is_causal": True,
    })["o"]
    # softmax over a single-element axis = 1 → output = v
    assert torch.allclose(out, v, atol=1e-4, rtol=1e-4)
