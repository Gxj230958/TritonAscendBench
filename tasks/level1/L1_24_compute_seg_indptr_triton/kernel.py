"""Task L1_24_compute_seg_indptr_triton_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `compute_seg_indptr_triton_kernel`
Source path: `sglang/srt/layers/moe/ep_moe/kernels.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): 4e5622e70e0050e5e9d66dae898eb3ce8f06678da90d81777414ab4337cb49b8
"""
import logging
import torch
import triton
from ascendbench.frozen.shim import ceil_div, is_cuda, is_musa  # Phase A': platform/math predicate (B-inert-shim)
_is_cuda = is_cuda()
_is_musa = is_musa()
if _is_cuda or _is_musa:
    from ascendbench.frozen.shim import sglang_per_token_group_quant_fp8 as per_token_group_quant_fp8  # Phase A': fp8 quant helpers -> shim (B-inert-shim)
import triton.language as tl

@triton.jit
def compute_seg_indptr_triton_kernel(reorder_topk_ids, seg_indptr, num_toks):
    expert_id_minus_1 = tl.program_id(0) - 1
    low = 0
    high = num_toks - 1
    target_location = -1
    while low <= high:
        mid = (low + high) // 2
        if tl.load(reorder_topk_ids + mid) > expert_id_minus_1:
            high = mid - 1
        else:
            low = mid + 1
            target_location = mid
    tl.store(seg_indptr + expert_id_minus_1 + 1, target_location + 1)

import torch
import triton


# Authored launcher: drives the in-module @triton.jit
# `compute_seg_indptr_triton_kernel` exactly as the upstream host call site does
# (sglang/srt/layers/moe/ep_moe/kernels.py::moe_ep_deepgemm_preprocess,
# SGLang commit 409d350fb6f6a1e7c7546e39028f811092a8e489).
#
# Why a launcher rather than the host wrapper: the named kernel ships inside a
# 1464-LOC multi-kernel MoE/DeepGEMM pipeline whose host orchestration imports
# the full sglang runtime (sgl_kernel, deep_gemm, flashinfer, vllm._C, fp8
# quant, ...). That closure is unbuildable/standalone-severed (closure cycle +
# vllm._C unvendorable, classifier tier T3), and none of it is needed to
# exercise the unit under test. The slice of the wrapper that is relevant to
# this kernel is tiny and reproduced verbatim below.
#
# Upstream call site reproduced here (moe_ep_deepgemm_preprocess, lines
# 1071-1080 of the frozen kernel.py):
#     reorder_topk_ids, reorder_ids = torch.sort(topk_ids.view(-1), stable=True)
#     seg_indptr = torch.zeros(
#         num_local_experts + 1, device=topk_ids.device, dtype=torch.int64)
#     compute_seg_indptr_triton_kernel[(num_local_experts + 1,)](
#         reorder_topk_ids, seg_indptr, topk_ids.numel())
#
# The named kernel is launched with grid (num_local_experts + 1,). For each
# program id e it binary-searches the SORTED reorder_topk_ids for the last
# position whose value is <= (e - 1) and writes target_location + 1 into
# seg_indptr[e]. That equals the count of sorted entries strictly less than e,
# i.e. searchsorted(reorder_topk_ids, e, side='left') — the RAW segment indptr
# with NO num_minus_one subtraction (the deepgemm caller consumes it raw).
#
# The torch.sort is part of this upstream call site (it feeds the kernel's
# sorted-array precondition), so it is reproduced here rather than pushed into
# the dense input dict. Output key matches the oracle contract: seg_indptr.
def run(inputs: dict) -> dict:
    topk_ids = inputs["topk_ids"]
    num_local_experts = int(inputs["num_local_experts"])

    reorder_topk_ids, _reorder_ids = torch.sort(topk_ids.view(-1), stable=True)
    seg_indptr = torch.zeros(
        num_local_experts + 1, device=topk_ids.device, dtype=torch.int64
    )

    compute_seg_indptr_triton_kernel[(num_local_experts + 1,)](
        reorder_topk_ids, seg_indptr, topk_ids.numel()
    )
    return {"seg_indptr": seg_indptr}
