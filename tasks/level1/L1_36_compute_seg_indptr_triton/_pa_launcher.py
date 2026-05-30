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
