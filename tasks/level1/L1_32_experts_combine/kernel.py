"""Task L1_32_experts_combine_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `experts_combine_kernel`
Source path: `sglang/srt/layers/elementwise.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): 62126ca8893757ae05a0d75da2087dec4cbf289f4af3d248fa84afe4fac739a9
"""
from typing import Optional, Tuple
import torch
import triton
import triton.language as tl
from ascendbench.frozen.shim import is_hip  # Phase A': platform/math predicate (B-inert-shim)
from ascendbench.frozen.shim import register_custom_op  # Phase A': inert custom-op decorator (B-inert-shim)

@triton.jit
def experts_combine_kernel(out_hidden_states, moe_hidden_states, mlp_hidden_states, combine_k: tl.constexpr, hidden_dim: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start_index_mlp = pid * hidden_dim
    start_index_rmoe = pid * hidden_dim * combine_k
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    combine_k_offsets = tl.arange(0, combine_k)
    moe_x = tl.load(moe_hidden_states + start_index_rmoe + combine_k_offsets[:, None] * hidden_dim + offsets[None, :], mask=mask[None, :], other=0.0)
    moe_x = tl.sum(moe_x, axis=0)
    mlp_x = tl.load(mlp_hidden_states + start_index_mlp + offsets, mask=mask, other=0.0)
    combined_x = (moe_x + mlp_x) / 1.4142135623730951
    tl.store(out_hidden_states + start_index_mlp + offsets, combined_x, mask=mask)

import triton

import torch


# Authored launcher: drives the in-module @triton.jit `experts_combine_kernel`
# exactly as the upstream host wrapper `experts_combine_triton` does (SGLang
# sglang/srt/layers/elementwise.py, commit
# 409d350fb6f6a1e7c7546e39028f811092a8e489).
#
# Why a launcher rather than the registered op: meta.oracle.gpu_triton is
# `kernel.py::placeholder` (no resolvable callable), so the bench cannot
# synthesise a run() for the bare @triton.jit. The host wrapper itself is
# severable (its only module-scope deps are `is_hip` -- a constant backend gate
# -- and the inert `register_custom_op` shim), and it touches only plain dense
# tensors. We reproduce its invocation verbatim.
#
# Upstream invocation reproduced here (experts_combine_triton):
#   moe_hidden_states: (bs, combine_k, hidden_dim) when 3D, else (bs,
#     hidden_dim) "pre-combined" with combine_k = 1.
#   mlp_hidden_states: (bs, hidden_dim) -- defines bs, hidden_dim.
#   combine_k = moe_hidden_states.shape[1] if 3D else 1.
#   out_hidden_states = empty_like(mlp_hidden_states), shape (bs, hidden_dim).
#   config = {BLOCK_SIZE = next_power_of_2(hidden_dim),
#             num_warps = max(min(next_power_of_2(cdiv(hidden_dim,1024)),8),4)}.
#   grid = (bs,).
#   experts_combine_kernel[(bs,)](out, moe, mlp, combine_k, hidden_dim,
#                                 **config).
#   Kernel math (per row pid): moe_x = sum_{k<combine_k} moe[pid,k,:];
#     combined = (moe_x + mlp[pid,:]) / sqrt(2). Output key "output", matching
#     the oracle contract; out dtype == mlp dtype.
def run(inputs: dict) -> dict:
    moe_hidden_states = inputs["moe_hidden_states"].contiguous()
    mlp_hidden_states = inputs["mlp_hidden_states"].contiguous()

    assert moe_hidden_states.is_contiguous()
    assert mlp_hidden_states.is_contiguous()

    if len(moe_hidden_states.shape) == 2:
        combine_k = 1  # pre-combined
    else:
        combine_k = moe_hidden_states.shape[1]

    out_hidden_states = torch.empty_like(mlp_hidden_states)

    bs, hidden_dim = mlp_hidden_states.shape

    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 1024)), 8), 4
        ),
    }

    experts_combine_kernel[(bs,)](
        out_hidden_states,
        moe_hidden_states,
        mlp_hidden_states,
        combine_k,
        hidden_dim,
        **config,
    )

    return {"output": out_hidden_states}
