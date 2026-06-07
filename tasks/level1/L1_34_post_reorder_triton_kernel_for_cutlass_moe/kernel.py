"""Task L1_34_post_reorder_triton_kernel_for_cutlass_moe — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `post_reorder_triton_kernel_for_cutlass_moe`
Source path: `sglang/srt/layers/moe/ep_moe/kernels.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): 4e5622e70e0050e5e9d66dae898eb3ce8f06678da90d81777414ab4337cb49b8
Frozen by: scripts/phase_a_freeze.py
"""
import logging
import torch
import triton
from tasks._frozen_shim import ceil_div, is_cuda, is_musa  # Phase A': platform/math predicate (B-inert-shim)
_is_cuda = is_cuda()
_is_musa = is_musa()
if _is_cuda or _is_musa:
    from tasks._frozen_shim import sglang_per_token_group_quant_fp8 as per_token_group_quant_fp8  # Phase A': fp8 quant helpers -> shim (B-inert-shim)
import triton.language as tl

@triton.jit
def post_reorder_triton_kernel_for_cutlass_moe(down_output_ptr, output_ptr, src2dst_ptr, topk_ids_ptr, topk_weights_ptr, num_local_experts, topk, num_tokens, hidden_size, routed_scaling_factor: float, BLOCK_SIZE: tl.constexpr, NUM_STAGES: tl.constexpr):
    OutDtype = output_ptr.dtype.element_ty
    offset = BLOCK_SIZE * tl.program_id(1) + tl.arange(0, BLOCK_SIZE)
    mask = offset < hidden_size
    down_output_ptr_offs = down_output_ptr + offset
    output_ptr_offs = output_ptr + offset
    start_src_idx = tl.program_id(0)
    step = tl.num_programs(0)
    for src_idx_int32 in tl.range(start_src_idx, num_tokens, step, num_stages=NUM_STAGES):
        src_idx = src_idx_int32.to(tl.int64)
        token_src2dst_ptr = src2dst_ptr + src_idx * topk
        token_topk_ids_ptr = topk_ids_ptr + src_idx * topk
        token_topk_weights_ptr = topk_weights_ptr + src_idx * topk
        sum_vec = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for idx in range(topk):
            expert_id = tl.load(token_topk_ids_ptr + idx)
            if expert_id != num_local_experts:
                dst_idx_int32 = tl.load(token_src2dst_ptr + idx)
                dst_idx = dst_idx_int32.to(tl.int64)
                dst_idx = dst_idx
                weight_scale = tl.load(token_topk_weights_ptr + idx).to(tl.float32)
                load_ptr_offs = down_output_ptr_offs + dst_idx * hidden_size
                in_data = tl.load(load_ptr_offs, mask=mask).to(tl.float32)
                sum_vec += in_data * weight_scale
        sum_vec *= routed_scaling_factor
        store_ptr_offs = output_ptr_offs + src_idx * hidden_size
        tl.store(store_ptr_offs, sum_vec.to(OutDtype), mask=mask)

import torch
import triton


# Drives the graded @triton.jit `post_reorder_triton_kernel_for_cutlass_moe`
# directly. The grid is pure scheduling (the kernel strides tokens by
# num_programs(0) and masks the hidden tail), so any valid cover reproduces the
# result: one program per token, one hidden block.
#
# Faithful invocation: the expert gate AND the router weighting AND the routed
# scaling are all exercised. topk_ids carries real expert ids including the
# padding sentinel num_local_experts (gated slots are skipped); topk_weights are
# real per-slot router probabilities; routed_scaling_factor != 1; src2dst is a
# real permutation gather.
def run(inputs: dict) -> dict:
    down_output = inputs["down_output"].contiguous()
    src2dst = inputs["src2dst"].to(torch.int64).contiguous()
    topk_ids = inputs["topk_ids"].to(torch.int64).contiguous()
    topk_weights = inputs["topk_weights"].to(torch.float32).contiguous()
    num_local_experts = int(inputs["num_local_experts"])
    topk = int(inputs["topk"])
    routed_scaling_factor = float(inputs["routed_scaling_factor"])

    num_tokens = src2dst.numel() // topk
    hidden_size = down_output.shape[1]
    output = down_output.new_zeros((num_tokens, hidden_size))

    block_dim = triton.next_power_of_2(hidden_size)
    grid = (num_tokens, triton.cdiv(hidden_size, block_dim))

    post_reorder_triton_kernel_for_cutlass_moe[grid](
        down_output_ptr=down_output,
        output_ptr=output,
        src2dst_ptr=src2dst,
        topk_ids_ptr=topk_ids,
        topk_weights_ptr=topk_weights,
        num_local_experts=num_local_experts,
        topk=topk,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        routed_scaling_factor=routed_scaling_factor,
        BLOCK_SIZE=block_dim,
        NUM_STAGES=3,
    )
    return {"output": output}
