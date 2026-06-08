"""Task L1_31_deepep_post_reorder_triton_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `deepep_post_reorder_triton_kernel`
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
def deepep_post_reorder_triton_kernel(down_output_ptr, output_ptr, src2dst_ptr, topk_ids_ptr, topk_weights_ptr, topk, hidden_size, BLOCK_SIZE: tl.constexpr):
    InDtype = down_output_ptr.dtype.element_ty
    src_idx = tl.program_id(0)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    topk_weights_ptr = topk_weights_ptr + src_idx * topk
    store_ptr = output_ptr + src_idx * hidden_size
    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size
        sum_vec = tl.zeros([BLOCK_SIZE], dtype=InDtype)
        for idx in range(topk):
            dst_idx = tl.load(src2dst_ptr + idx)
            if dst_idx >= 0:
                weigh_scale = tl.load(topk_weights_ptr + idx).to(InDtype)
                load_ptr = down_output_ptr + dst_idx * hidden_size
                in_data = tl.load(load_ptr + offset, mask=mask)
                sum_vec += in_data * weigh_scale
        tl.store(store_ptr + offset, sum_vec, mask=mask)

import torch
import triton


# Authored launcher: drives the in-module @triton.jit
# `deepep_post_reorder_triton_kernel` exactly as the upstream host wrapper
# `deepep_post_reorder` does (SGLang sglang/srt/layers/moe/ep_moe/kernels.py,
# commit 409d350). The frozen kernel.py is jit-only (no host wrapper), so the
# resolver execs this run(inputs) into the kernel module and the jit symbol is
# in scope (injected -> noqa: F821).
#
# DeepEP combine (post-reorder): expert-major rows x[n_tokens*topk, D] are
# gathered back into token order and weighted-summed:
#   out[i, :] = sum_j x[src2dst[i*topk + j], :] * topk_weights[i, j]
# src2dst is a FLAT [n_tokens*topk] index map (a permutation of token slots into
# the expert-major buffer); topk_weights is [n_tokens, topk]. The kernel body:
#   src_idx = program_id(0)                       # one program per out token
#   src2dst_ptr += src_idx*topk                   # flat src2dst[src_idx*topk+j]
#   topk_weights_ptr += src_idx*topk              # flat tw[src_idx*topk+j]
#   for j in range(topk):
#       dst = load(src2dst_ptr+j)                 # row into x; <0 -> skip
#       if dst >= 0: acc += x[dst, off] * tw[j]
#   store(out + src_idx*hidden_size + off, acc)
# so src2dst is load-bearing: it selects WHICH expert-major rows feed each
# output token. The kernel also takes a topk_ids_ptr arg that its body advances
# but never dereferences (masking is done via the dst>=0 guard); build_inputs
# does not supply topk_ids, so we pass a 1-element scratch tensor purely to
# satisfy the signature -- it is never read.
#
# Grid / BLOCK_SIZE (verbatim upstream semantics):
#   grid = (n_tokens,)            # one program per output token
#   BLOCK_SIZE = 512             # upstream default; power-of-2 (tl.arange needs
#                                  # pow2 at trace time on every backend). The
#                                  # kernel masks `off < hidden_size`, so a
#                                  # BLOCK_SIZE that exceeds hidden_size is still
#                                  # bit-exact (the D-loop is a masked copy).
# Output key: "output" (the oracle contract key), [n_tokens, D] in x.dtype.
def run(inputs: dict) -> dict:
    x = inputs["x"]                          # [n_tokens*topk, D]
    src2dst = inputs["src2dst"]              # [n_tokens*topk] int (flat map)
    topk_weights = inputs["topk_weights"]    # [n_tokens, topk] float32
    topk = int(inputs["topk"])

    n_tokens = topk_weights.shape[0]
    hidden_size = x.shape[1]

    # Kernel indexes src2dst/topk_weights with int32 pointer math; keep src2dst
    # integral and contiguous. topk_weights stays float (loaded + cast to x dt).
    if src2dst.dtype not in (torch.int32, torch.int64):
        src2dst = src2dst.to(torch.int64)
    src2dst = src2dst.contiguous()
    topk_weights = topk_weights.contiguous()
    x = x.contiguous()

    output = torch.empty(
        (n_tokens, hidden_size), device=x.device, dtype=x.dtype
    )

    # topk_ids: signature-only arg the kernel advances but never loads. Supply a
    # correctly-typed buffer the same flat length as src2dst so any (absent)
    # access stays in bounds; build_inputs does not produce it.
    topk_ids = torch.zeros_like(src2dst)

    BLOCK_SIZE = 512
    grid = (n_tokens,)

    deepep_post_reorder_triton_kernel[grid](  # noqa: F821 (injected into kernel.py)
        x,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        topk,
        hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return {"output": output}
