"""Task L1_21_create_flashmla_kv_indices_triton — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `create_flashmla_kv_indices_triton`
Source path: `sglang/srt/layers/attention/utils.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): 6bae800d0cb2f1b4a2abb448bec90e2c4664d0b882e8bbcdfbe06a3bc8834cc1
Frozen by: scripts/phase_a_freeze.py
"""
import torch
import triton
import triton.language as tl
from ascendbench.frozen.shim import is_cuda  # Phase A': platform/math predicate (B-inert-shim)
_FLASHMLA_CREATE_KV_BLOCK_SIZE = 4096
FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON = tl.constexpr(_FLASHMLA_CREATE_KV_BLOCK_SIZE)
_is_cuda = is_cuda()
if _is_cuda:
    from ascendbench.frozen.shim import concat_mla_absorb_q  # Phase A': sibling-only CUDA absorb-concat (B-inert-shim)
from ascendbench.frozen.shim import is_arch_support_pdl  # Phase A': sibling-only PDL arch gate (B-inert-shim)

@triton.jit
def create_flashmla_kv_indices_triton(req_to_token_ptr, req_pool_indices_ptr, page_kernel_lens_ptr, kv_start_idx, kv_indices_ptr, req_to_token_ptr_stride: tl.constexpr, kv_indices_ptr_stride: tl.constexpr, PAGED_SIZE: tl.constexpr=64):
    NUM_PAGE_PER_BLOCK: tl.constexpr = FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON // PAGED_SIZE
    pid = tl.program_id(axis=0)
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)
    num_paged = tl.cdiv(kv_end - kv_start, PAGED_SIZE)
    num_pages_loop = tl.cdiv(kv_end - kv_start, FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON)
    for i in range(num_pages_loop):
        paged_offset = (tl.arange(0, NUM_PAGE_PER_BLOCK).to(tl.int64) + i * NUM_PAGE_PER_BLOCK) * PAGED_SIZE
        paged_offset_out = tl.arange(0, NUM_PAGE_PER_BLOCK) + i * NUM_PAGE_PER_BLOCK
        mask = paged_offset < num_paged * PAGED_SIZE
        mask_out = paged_offset_out < num_paged
        data = tl.load(req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + kv_start + paged_offset, mask=mask)
        tl.store(kv_indices_ptr + pid * kv_indices_ptr_stride + paged_offset_out, data // PAGED_SIZE, mask=mask_out)

import torch

# Authored jit-only launcher: drives the in-module @triton.jit
# `create_flashmla_kv_indices_triton` exactly as the upstream FlashMLA decode
# call site does (sglang/srt/layers/attention/utils.py). The kernel has no
# host wrapper in the module, so this run(inputs) reproduces the call site:
#
#   grid = (n_reqs,)
#   create_flashmla_kv_indices_triton[grid](
#       req_to_token,            # [max_batch, max_context_len]
#       req_pool_indices,        # [n_reqs]
#       page_kernel_lens,        # [n_reqs]  (seq_lens)
#       None,                    # kv_start_idx  (decode path passes null)
#       kv_indices,              # [n_reqs, kv_indices_stride]  (out, pre-zeroed)
#       req_to_token.stride(0),  # req_to_token_ptr_stride  (constexpr)
#       kv_indices.stride(0),    # kv_indices_ptr_stride     (constexpr)
#       PAGED_SIZE=page_size,    # page granularity          (constexpr)
#   )
#
# kv_start_idx=None -> the kernel's `if kv_start_idx:` is False, so kv_start=0
# and kv_end = page_kernel_lens[pid]; pages beyond num_paged are never stored,
# so the pre-zeroed buffer supplies the padding. Output key: "kv_indices".
def run(inputs: dict) -> dict:
    req_to_token = inputs["req_to_token"]
    req_pool_indices = inputs["req_pool_indices"]
    page_kernel_lens = inputs["page_kernel_lens"]
    page_size = int(inputs["page_size"])

    n_reqs = req_pool_indices.shape[0]
    max_ctx = req_to_token.shape[1]
    # kv_indices column count = max pages addressable in req_to_token width.
    kv_indices_stride = (max_ctx + page_size - 1) // page_size

    kv_indices = torch.zeros(
        (n_reqs, kv_indices_stride),
        dtype=torch.int32,
        device=req_to_token.device,
    )

    grid = (n_reqs,)
    create_flashmla_kv_indices_triton[grid](
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        None,  # kv_start_idx
        kv_indices,
        req_to_token.stride(0),
        kv_indices.stride(0),
        PAGED_SIZE=page_size,
    )

    return {"kv_indices": kv_indices}
