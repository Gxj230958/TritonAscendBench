"""Task L2_26_fwd_kernel_ep_scatter_1 — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_fwd_kernel_ep_scatter_1`
Source path: `vllm/model_executor/layers/fused_moe/deep_gemm_utils.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): 502a66386d0be6673d2f582daca0c10e56e2e8c4a0b94cb847c41d612c37e101
Frozen by: scripts/phase_a_freeze.py
"""
'\nTaken from https://github.com/ModelTC/LightLLM/blob/8ed97c74c18f11505b048b1ba00ba5c0cef8bff6/lightllm/common/fused_moe/deepep_scatter_gather.py\nand updated to fit vllm needs and terminology.\n'
import torch
import ascendbench.frozen.shim.modular_kernel as mk  # Phase A': off-path modular_kernel annotation shim (B-inert-shim)
from ascendbench.frozen.shim import count_expert_num_tokens  # Phase A': off-path host-wrapper helper (B-inert-shim)
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
from ascendbench.frozen.shim import get_mk_alignment_for_contiguous_layout  # Phase A': off-path deep_gemm accel surface (B-inert-shim)
from ascendbench.frozen.shim import round_up  # Phase A': pure-math shim (B-inert-shim)

@triton.jit
def round_up_128(x: int) -> int:
    y = 128
    return (x + y - 1) // y * y

@triton.jit
def _fwd_kernel_ep_scatter_1(num_recv_tokens_per_expert, expert_start_loc, m_indices, num_experts: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_EXPERT_NUM: tl.constexpr):
    cur_expert = tl.program_id(0)
    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(num_recv_tokens_per_expert + offset_cumsum, mask=offset_cumsum < num_experts, other=0)
    tokens_per_expert = round_up_128(tokens_per_expert)
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert
    cur_expert_start = tl.sum(tl.where(offset_cumsum == cur_expert, cumsum, tl.zeros_like(cumsum)))
    tl.store(expert_start_loc + cur_expert, cur_expert_start)
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)
    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)
    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E):
        offs = start_m + off_expert
        mask = offs < cur_expert_token_num
        tl.store(m_indices_start_ptr + offs, cur_expert, mask=mask)

import torch
import triton


# Authored launcher: drives the in-module @triton.jit
# `_fwd_kernel_ep_scatter_1` exactly as the upstream host wrapper
# `ep_scatter` does (vLLM
# vllm/model_executor/layers/fused_moe/deep_gemm_utils.py, commit 18f6bf5).
#
# Upstream call site (the only part of `ep_scatter` that touches kernel_1):
#     num_experts = num_recv_tokens_per_expert.shape[0]
#     grid = num_experts
#     assert m_indices.shape[0] % BLOCK_E == 0
#     assert expert_start_loc.shape[0] == num_experts
#     _fwd_kernel_ep_scatter_1[(grid,)](
#         num_recv_tokens_per_expert,
#         expert_start_loc,
#         m_indices,
#         num_experts=num_experts,
#         num_warps=8,
#         BLOCK_E=128,
#         BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
#     )
# expert_start_loc / m_indices are pre-allocated by the caller
# (`deepgemm_moe_permute`): expert_start_loc = torch.empty(num_experts, int32)
# and m_indices = torch.full((M_sum,), -1, int32) -- the -1 sentinel is part
# of the contract (padding rows DeepGEMM skips), so the launcher reproduces it.
#
# M_sum is the size of the contiguous-group output the kernel indexes into:
#     M_sum = sum_e round_up_128(num_recv_tokens_per_expert[e])
# which equals expert_start_loc[-1] + round_up_128(counts[-1]). The launcher
# derives it from the input so the kernel's `m_indices.shape[0] % 128 == 0`
# invariant holds for any fixture.
#
# Output keys match the oracle contract exactly:
#   expert_start_loc : [num_experts] int32 -- padded exclusive prefix sum
#   m_indices        : [M_sum]       int32 -- expert-id fill, -1 padding
def _round_up_128(x: torch.Tensor) -> torch.Tensor:
    return ((x + 127) // 128) * 128


def run(inputs: dict) -> dict:
    counts = inputs["num_recv_tokens_per_expert"]
    # Kernel loads num_recv_tokens_per_expert with int math; keep it integral.
    if counts.dtype not in (torch.int32, torch.int64):
        counts = counts.to(torch.int32)
    counts = counts.contiguous()

    num_experts = counts.shape[0]
    BLOCK_E = 128
    num_warps = 8

    padded = _round_up_128(counts.to(torch.int64))
    m_sum = int(padded.sum().item())
    # m_sum is a sum of multiples of 128, so the % BLOCK_E == 0 invariant holds.
    # Guarantee at least one BLOCK_E row so the empty-fixture grid still runs.
    if m_sum == 0:
        m_sum = BLOCK_E

    device = counts.device
    expert_start_loc = torch.empty(
        (num_experts,), device=device, dtype=torch.int32
    )
    m_indices = torch.full(
        (m_sum,), -1, device=device, dtype=torch.int32
    )

    grid = num_experts

    _fwd_kernel_ep_scatter_1[(grid,)](
        counts,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=num_warps,
        BLOCK_E=BLOCK_E,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )

    return {"expert_start_loc": expert_start_loc, "m_indices": m_indices}
