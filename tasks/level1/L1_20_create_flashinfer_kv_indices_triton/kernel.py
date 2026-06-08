"""Task L1_20_create_flashinfer_kv_indices_triton — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `create_flashinfer_kv_indices_triton`
Source path: `sglang/srt/layers/attention/utils.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): 6bae800d0cb2f1b4a2abb448bec90e2c4664d0b882e8bbcdfbe06a3bc8834cc1
Frozen by: scripts/phase_a_freeze.py
"""
import torch
import triton
import triton.language as tl
from triton_ascend_bench.frozen.shim import is_cuda  # Phase A': platform/math predicate (B-inert-shim)
_is_cuda = is_cuda()
if _is_cuda:
    from triton_ascend_bench.frozen.shim import concat_mla_absorb_q  # Phase A': sibling-only CUDA absorb-concat (B-inert-shim)
from triton_ascend_bench.frozen.shim import is_arch_support_pdl  # Phase A': sibling-only PDL arch gate (B-inert-shim)

@triton.jit
def create_flashinfer_kv_indices_triton(req_to_token_ptr, req_pool_indices_ptr, page_kernel_lens_ptr, kv_indptr, kv_start_idx, kv_indices_ptr, req_to_token_ptr_stride: tl.constexpr):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)
    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)
    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + kv_start + offset, mask=mask)
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)

import torch


# Authored launcher: drives the in-module @triton.jit
# `create_flashinfer_kv_indices_triton` exactly as the upstream host call site
# does (sglang/srt/layers/attention/flashinfer_backend.py:1129, normal
# spec_info=None path; SGLang commit 409d350fb6f6a1e7c7546e39028f811092a8e489).
#
# Why a launcher rather than the host wrapper: the kernel ships inside the
# flashinfer attention backend whose host orchestration imports the full
# sglang/flashinfer/sgl_kernel runtime (>800-symbol closure with a cycle, plus
# unvendorable native deps). None of that is needed to exercise the unit under
# test. The wrapper's body that is relevant to this kernel is tiny and
# reproduced verbatim below.
#
# Upstream call site reproduced here (call_begin_forward, spec_info is None):
#     bs = len(req_pool_indices)
#     kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
#     kv_indptr = kv_indptr[: bs + 1]
#     kv_indices = torch.empty(paged_kernel_lens_sum, dtype=torch.int32, ...)
#     create_flashinfer_kv_indices_triton[(bs,)](
#         self.req_to_token,        # [max_batch, max_context_len]
#         req_pool_indices,         # [bs]
#         paged_kernel_lens,        # [bs]
#         kv_indptr,                # [bs + 1]
#         kv_start_idx,             # None on this path -> kernel kv_start == 0
#         kv_indices,               # output [sum(lens)]
#         self.req_to_token.shape[1],  # req_to_token_ptr_stride (constexpr)
#     )
#
# The dense input dict supplies `req_to_token` (the token-location table),
# `req_pool_indices` (which pool row each request reads), and `seq_lens`
# (== paged_kernel_lens, the per-request copy length, 1 <= lens < N <=
# max_context_len so the gather never reads past the row). kv_start_idx is
# None, matching the normal non-sliding-window host path. Output keys match
# the oracle contract: kv_indptr, kv_indices.
def run(inputs: dict) -> dict:
    req_to_token = inputs["req_to_token"]
    req_pool_indices = inputs["req_pool_indices"]
    paged_kernel_lens = inputs["seq_lens"]

    device = req_to_token.device
    bs = req_pool_indices.shape[0]

    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens.to(torch.long), dim=0)

    paged_kernel_lens_sum = int(kv_indptr[bs].item())
    kv_indices = torch.empty(
        paged_kernel_lens_sum, dtype=torch.int32, device=device
    )

    create_flashinfer_kv_indices_triton[(bs,)](
        req_to_token,
        req_pool_indices,
        paged_kernel_lens,
        kv_indptr,
        None,  # kv_start_idx: normal non-sliding path -> kv_start == 0
        kv_indices,
        req_to_token.shape[1],
    )
    return {"kv_indptr": kv_indptr, "kv_indices": kv_indices}
