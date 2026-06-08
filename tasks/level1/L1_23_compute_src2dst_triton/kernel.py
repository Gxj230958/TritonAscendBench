"""Task L1_23_compute_src2dst_triton_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `compute_src2dst_triton_kernel`
Source path: `sglang/srt/layers/moe/ep_moe/kernels.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): 4e5622e70e0050e5e9d66dae898eb3ce8f06678da90d81777414ab4337cb49b8
Frozen by: scripts/phase_a_freeze.py
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
def compute_src2dst_triton_kernel(reorder_ids, src2dst, num_toks, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    dst_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_id < num_toks
    src_id = tl.load(reorder_ids + dst_id, mask=mask)
    tl.store(src2dst + src_id, dst_id, mask=mask)

import torch
import triton


# Authored launcher: drives the in-module @triton.jit
# `compute_src2dst_triton_kernel` exactly as the upstream host call site does
# (sglang/srt/layers/moe/ep_moe/kernels.py::cutlass_w4_run_moe_ep_preproess,
# SGLang commit 409d350fb6f6a1e7c7546e39028f811092a8e489).
#
# Why a launcher rather than the host wrapper: the kernel ships inside a 1464-LOC
# multi-kernel MoE pipeline whose host orchestration imports the full sglang
# runtime (sgl_kernel, deep_gemm, flashinfer, vllm._C, ...). That closure is
# unbuildable/standalone-severed (closure cycle + vllm._C unvendorable), and
# none of it is needed to exercise the unit under test. The wrapper's body that
# is relevant to this kernel is tiny and reproduced verbatim below.
#
# Upstream call site reproduced here (cutlass_w4_run_moe_ep_preproess):
#     _, reorder_ids = torch.sort(topk_ids.view(-1), stable=True)
#     BLOCK_SIZE = 512
#     grid = (triton.cdiv(topk_ids.numel(), BLOCK_SIZE),)
#     src2dst = torch.empty(topk_ids.numel(), device=..., dtype=torch.int32)
#     compute_src2dst_triton_kernel[grid](
#         reorder_ids, src2dst, topk_ids.numel(), BLOCK_SIZE)
#     return src2dst
#
# The dense input dict supplies `reorder_ids` (the already-materialised sort
# permutation that the upstream derives from torch.sort) and `num_toks`
# (= topk_ids.numel() = reorder_ids.numel()), so the host sort/preprocessing
# is bypassed and the bare jit is driven directly.
#
# Kernel math (matches oracle_pytorch): src2dst[reorder_ids[i]] = i, a pure
# inverse-permutation scatter. `src2dst` is allocated int32 and every slot is
# written because reorder_ids is a full permutation of [0, num_toks). Output
# key matches the oracle contract: src2dst.
def run(inputs: dict) -> dict:
    reorder_ids = inputs["reorder_ids"]
    num_toks = int(inputs["num_toks"])

    BLOCK_SIZE = 512
    grid = (triton.cdiv(num_toks, BLOCK_SIZE),)
    src2dst = torch.empty(num_toks, device=reorder_ids.device, dtype=torch.int32)

    compute_src2dst_triton_kernel[grid](
        reorder_ids, src2dst, num_toks, BLOCK_SIZE
    )
    return {"src2dst": src2dst}
