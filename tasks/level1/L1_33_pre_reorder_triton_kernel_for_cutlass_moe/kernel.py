"""Task L1_33_pre_reorder_triton_kernel_for_cutlass_moe — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `pre_reorder_triton_kernel_for_cutlass_moe`
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
def pre_reorder_triton_kernel_for_cutlass_moe(input_ptr, gateup_input_ptr, src2dst_ptr, topk_ids_ptr, a1_scales_ptr, num_local_experts, topk, num_tokens, hidden_size, BLOCK_SIZE: tl.constexpr, NUM_STAGES: tl.constexpr):
    OutDtype = gateup_input_ptr.dtype.element_ty
    if a1_scales_ptr is not None:
        a1_scale = 1.0 / tl.load(a1_scales_ptr)
    else:
        a1_scale = 1.0
    offset = BLOCK_SIZE * tl.program_id(1) + tl.arange(0, BLOCK_SIZE)
    mask = offset < hidden_size
    start_src_idx = tl.program_id(0)
    step = tl.num_programs(0)
    for src_idx_int32 in tl.range(start_src_idx, num_tokens, step, num_stages=NUM_STAGES):
        src_idx = src_idx_int32.to(tl.int64)
        token_src2dst_ptr = src2dst_ptr + src_idx * topk
        token_topk_ids_ptr = topk_ids_ptr + src_idx * topk
        src_ptr_offs = input_ptr + src_idx * hidden_size + offset
        dst_ptr_offs = gateup_input_ptr + offset
        in_data = tl.load(src_ptr_offs, mask=mask).to(tl.float32)
        out_data = (in_data * a1_scale).to(OutDtype)
        for idx in range(topk):
            expert_id = tl.load(token_topk_ids_ptr + idx)
            if expert_id != num_local_experts:
                dst_idx = tl.load(token_src2dst_ptr + idx)
                tl.store(dst_ptr_offs + dst_idx * hidden_size, out_data, mask=mask)

import torch
import triton


# Drives the graded @triton.jit `pre_reorder_triton_kernel_for_cutlass_moe`
# directly, NOT through the upstream host wrapper `pre_reorder_for_cutlass_moe`:
# that wrapper sizes the grid via `_get_launch_config_2d`, which calls
# torch.cuda.get_device_properties(input.device) and aborts on the CPU
# oracle-consistency anchor. The grid is pure scheduling -- the kernel strides
# tokens by num_programs(0) and masks the hidden tail -- so any valid cover
# reproduces the result: one program per token, one hidden block.
#
# Faithful invocation: the kernel's expert gate AND activation scaling are
# exercised, NOT pinned to a trivial subcase. topk_ids carries real expert ids
# including the padding sentinel num_local_experts (gated slots are skipped);
# a1_scales is a real per-tensor scale; src2dst is a real permutation.
def run(inputs: dict) -> dict:
    x = inputs["input"].contiguous()
    src2dst = inputs["src2dst"].to(torch.int64).contiguous()
    topk_ids = inputs["topk_ids"].to(torch.int64).contiguous()
    a1_scales = inputs["a1_scales"].to(torch.float32).contiguous()
    num_local_experts = int(inputs["num_local_experts"])
    topk = int(inputs["topk"])

    num_tokens, hidden_size = x.shape
    gateup_input = x.new_zeros((num_tokens * topk, hidden_size))

    block_dim = triton.next_power_of_2(hidden_size)
    grid = (num_tokens, triton.cdiv(hidden_size, block_dim))

    pre_reorder_triton_kernel_for_cutlass_moe[grid](
        input_ptr=x,
        gateup_input_ptr=gateup_input,
        src2dst_ptr=src2dst,
        topk_ids_ptr=topk_ids,
        a1_scales_ptr=a1_scales,
        num_local_experts=num_local_experts,
        topk=topk,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        BLOCK_SIZE=block_dim,
        NUM_STAGES=3,
    )
    return {"gateup_input": gateup_input}
