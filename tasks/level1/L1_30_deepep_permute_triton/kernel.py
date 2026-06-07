"""Task L1_30_deepep_permute_triton_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `deepep_permute_triton_kernel`
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
def deepep_permute_triton_kernel(input_ptr, gateup_input_ptr, src2dst_ptr, topk_ids_ptr, a1_scales_ptr, topk, hidden_size, BLOCK_SIZE: tl.constexpr):
    OutDtype = gateup_input_ptr.dtype.element_ty
    src_idx = tl.program_id(0)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    src_ptr = input_ptr + src_idx * hidden_size
    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size
        in_data = tl.load(src_ptr + offset, mask=mask).to(OutDtype)
        for idx in range(topk):
            dst_idx = tl.load(src2dst_ptr + idx)
            if dst_idx >= 0:
                dst_ptr = gateup_input_ptr + dst_idx * hidden_size
                tl.store(dst_ptr + offset, in_data, mask=mask)

import torch
import triton


# Authored launcher: drives the in-module @triton.jit
# `deepep_permute_triton_kernel` (frozen kernel.py) the way upstream SGLang
# drives it during DeepEP dispatch (sgl-project/sglang
# sglang/srt/layers/moe/ep_moe/kernels.py, commit 409d350). The kernel
# realises the token permute out[src2dst[i*topk + j]] = x[i]: one grid program
# per source token i, copying row x[i] into every destination slot listed in
# src2dst[i*topk : i*topk+topk] (a dst_idx < 0 slot is skipped).
#
# Kernel signature (verbatim, kernel.py:90):
#     deepep_permute_triton_kernel(
#         input_ptr,         # x            [n_tokens, hidden_size]
#         gateup_input_ptr,  # output       [n_tokens*topk, hidden_size]
#         src2dst_ptr,       # permutation  [n_tokens*topk]  (advanced by i*topk)
#         topk_ids_ptr,      #              [n_tokens*topk]  (pointer advanced,
#                            #                                never loaded)
#         a1_scales_ptr,     #              (in signature, never referenced in
#                            #                this permute path)
#         topk, hidden_size, BLOCK_SIZE: tl.constexpr)
#
# The kernel body only LOADS from input_ptr and src2dst_ptr and only STORES to
# gateup_input_ptr; topk_ids_ptr is advanced but never dereferenced and
# a1_scales_ptr is unused. We still pass real-shaped dummy tensors for both so
# the launch matches the upstream ABI (and so a future kernel byte change that
# starts reading them would not silently read freed memory).
#
# src2dst is the LOAD-BEARING input: out[src2dst[i*topk+j]] = x[i]. shapes.yaml
# supplies it as a randperm of [0, n_tokens*topk) (a real bijection), so every
# output slot is the image of exactly one source row -- a kernel that ignored
# src2dst (e.g. wrote out[i*topk+j] = x[i], the identity layout) produces a
# DIFFERENT output and fails the oracle compare.
#
# Output buffer: gateup_input_ptr dtype == OutDtype the kernel casts the copied
# row to. The oracle is out = x.new_zeros(...) (float32, = x.dtype), so the
# launcher allocates float32 to match bit-for-bit. It is pre-zeroed because the
# kernel only writes slots with dst_idx >= 0; with a full randperm every slot is
# covered, but zero-init keeps the contract identical to the oracle for any
# (e.g. partially-masked) src2dst.
#
# BLOCK_SIZE tiles hidden_size: the kernel loops `tl.range(0, hidden_size,
# BLOCK_SIZE)` and masks `offset < hidden_size`, so ANY BLOCK_SIZE drives it
# bit-identically. `tl.arange(0, BLOCK_SIZE)` requires a power of 2 (enforced at
# trace time on every triton 3.6 backend, not just TRITON_INTERPRET), so we pick
# the smallest power-of-2 >= hidden_size, capped at 1024; the per-row mask
# handles the remainder. Output key: "output" (the oracle contract key).
def run(inputs: dict) -> dict:
    x = inputs["x"]                       # [n_tokens, hidden_size] float32
    src2dst = inputs["src2dst"]           # [n_tokens*topk] int64 (permutation)
    topk = int(inputs["topk"])

    assert x.dim() == 2, f"x must be [n_tokens, hidden_size], got {tuple(x.shape)}"
    n_tokens, hidden_size = x.shape

    x = x.contiguous()
    # Kernel indexes src2dst with int math; keep it integral and flat.
    src2dst = src2dst.to(torch.int64).reshape(-1).contiguous()
    assert src2dst.numel() == n_tokens * topk, (
        f"src2dst must have n_tokens*topk={n_tokens * topk} entries, "
        f"got {src2dst.numel()}"
    )

    device = x.device
    output = torch.zeros(
        (n_tokens * topk, hidden_size), device=device, dtype=x.dtype
    )

    # topk_ids_ptr is advanced (i*topk) but never loaded; a1_scales_ptr is
    # unused in this path. Pass correctly-shaped dummies so the ABI matches.
    topk_ids = torch.zeros(
        (n_tokens * topk,), device=device, dtype=torch.int64
    )
    a1_scales = torch.ones((n_tokens,), device=device, dtype=torch.float32)

    BLOCK_SIZE = min(triton.next_power_of_2(hidden_size), 1024)

    grid = (n_tokens,)
    deepep_permute_triton_kernel[grid](  # noqa: F821 (injected into kernel.py)
        x,
        output,
        src2dst,
        topk_ids,
        a1_scales,
        topk,
        hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return {"output": output}
