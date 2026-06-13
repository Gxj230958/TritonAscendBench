"""Task L1_09_quantize_k_cache_fast_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_quantize_k_cache_fast_kernel`
Source path: `sglang/srt/layers/attention/nsa/quant_k_cache.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): 23dccba0f293449067a25b90e4ac009ce4e1d424c3358aedd9396fd35c995fc6
"""

import torch
import triton
import triton.language as tl






# Copied from original








@triton.jit
def _quantize_k_cache_fast_kernel(
    output_nope_q_ptr,
    output_nope_s_ptr,
    output_rope_ptr,
    k_nope_ptr,
    k_rope_ptr,
    output_nope_q_stride_0: int,
    output_nope_s_stride_0: int,
    output_rope_stride_0: int,
    k_nope_stride_0: int,
    k_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    token_id = tl.program_id(0)
    raw_block_id = tl.program_id(1)

    if raw_block_id < NUM_NOPE_BLOCKS:
        # a. quant nope
        effective_block_id = raw_block_id

        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_NOPE
        ptr = k_nope_ptr + token_id * k_nope_stride_0 + offs

        y = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)

        # the ref impl do not have a `tl.maximum(... eps)`, so we remove it here
        y_s = tl.max(tl.abs(y)) / FP8_MAX
        y_s_inv = 1.0 / y_s
        y_q = tl.clamp(y * y_s_inv, FP8_MIN, FP8_MAX).to(
            output_nope_q_ptr.dtype.element_ty
        )

        dst_q_ptr = output_nope_q_ptr + token_id * output_nope_q_stride_0 + offs
        dst_s_ptr = (
            output_nope_s_ptr + token_id * output_nope_s_stride_0 + effective_block_id
        )

        tl.store(dst_q_ptr, y_q, mask=mask)
        tl.store(dst_s_ptr, y_s)
    else:
        # b. copy rope
        effective_block_id = raw_block_id - NUM_NOPE_BLOCKS

        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE

        src_ptr = k_rope_ptr + token_id * k_rope_stride_0 + offs
        dst_ptr = output_rope_ptr + token_id * output_rope_stride_0 + offs

        data = tl.load(src_ptr, mask=mask)
        tl.store(dst_ptr, data, mask=mask)



import torch
import triton


# Authored launcher: drives the in-module @triton.jit
# `_quantize_k_cache_fast_kernel` exactly as the upstream host call site
# `_quantize_k_cache_fast` does (SGLang
# sglang/srt/layers/attention/nsa/quant_k_cache.py, commit 409d350).
#
# Upstream contract reproduced verbatim from the host launch site:
#   * k_nope: (num_tokens, 512) bfloat16, k_rope: (num_tokens, 64) bfloat16
#   * group_size == 128  -> NUM_NOPE_BLOCKS = 512 // 128 = 4
#   * DIM_NOPE = 512, DIM_ROPE = 64
#   * num_blocks_per_token = cdiv(512 + 64, 128) = 5
#   * grid = (num_tokens, num_blocks_per_token)
#   * FP8_MIN/FP8_MAX = finfo(float8_e4m3fn).min / .max  (= -448 / 448)
#
# The kernel computes, per nope group:
#     y_s = max(|y|) / FP8_MAX ; y_q = clamp(y / y_s, FP8_MIN, FP8_MAX) -> fp8
# and copies rope through unchanged. The upstream allocates one packed fp8
# buffer and carves typed views (output_nope_q fp8, output_nope_s fp32,
# output_rope bf16) out of it; we allocate the three logical views directly
# so each is returned as the oracle's contract output. The on-HBM bytes and
# kernel writes are identical to the packed layout (the views differ only in
# being separate allocations, which the kernel never observes -- it writes
# through the three independent base pointers + stride_0 it is handed).

import torch
import triton


# Authored launcher: drives the in-module @triton.jit
# `_quantize_k_cache_fast_kernel` exactly as the upstream host call site
# `_quantize_k_cache_fast` does (SGLang
# sglang/srt/layers/attention/nsa/quant_k_cache.py, commit 409d350).
#
# Upstream contract reproduced verbatim from the host launch site:
#   * k_nope: (num_tokens, 512) bfloat16, k_rope: (num_tokens, 64) bfloat16
#   * group_size == 128  -> NUM_NOPE_BLOCKS = 512 // 128 = 4
#   * DIM_NOPE = 512, DIM_ROPE = 64
#   * num_blocks_per_token = cdiv(512 + 64, 128) = 5
#   * grid = (num_tokens, num_blocks_per_token)
#   * FP8_MIN/FP8_MAX = finfo(float8_e4m3fn).min / .max  (= -448 / 448)
#
# The kernel computes, per nope group:
#     y_s = max(|y|) / FP8_MAX ; y_q = clamp(y / y_s, FP8_MIN, FP8_MAX) -> fp8
# and copies rope through unchanged. The upstream allocates one packed fp8
# buffer and carves typed views (output_nope_q fp8, output_nope_s fp32,
# output_rope bf16) out of it; we allocate the three logical views directly
# so each is returned as the oracle's contract output. The on-HBM bytes and
# kernel writes are identical to the packed layout (the views differ only in
# being separate allocations, which the kernel never observes -- it writes
# through the three independent base pointers + stride_0 it is handed).
def run(inputs: dict) -> dict:
    k_nope = inputs["k_nope"]
    k_rope = inputs["k_rope"]
    group_size = int(inputs["group_size"])

    # Upstream hard-asserts these; reproduce them so a wrong shape cannot
    # silently pass through a mismatched launch.
    assert k_nope.dtype == torch.bfloat16
    assert k_rope.dtype == torch.bfloat16

    k_nope = k_nope.contiguous()
    k_rope = k_rope.contiguous()

    num_tokens, dim_nope = k_nope.shape
    num_tokens_, dim_rope = k_rope.shape
    assert num_tokens == num_tokens_
    assert dim_nope == 512
    assert dim_rope == 64

    num_tiles = dim_nope // group_size  # 4

    # Three logical outputs matching the upstream packed views' dtypes.
    output_nope_q = torch.empty(
        (num_tokens, dim_nope), dtype=torch.float8_e4m3fn, device=k_nope.device
    )
    output_nope_s = torch.empty(
        (num_tokens, num_tiles), dtype=torch.float32, device=k_nope.device
    )
    output_rope = torch.empty(
        (num_tokens, dim_rope), dtype=torch.bfloat16, device=k_rope.device
    )

    num_blocks_per_token = triton.cdiv(dim_nope + dim_rope, group_size)
    assert num_blocks_per_token == 5
    assert dim_nope % group_size == 0
    NUM_NOPE_BLOCKS = dim_nope // group_size

    _quantize_k_cache_fast_kernel[(num_tokens, num_blocks_per_token)](
        output_nope_q,
        output_nope_s,
        output_rope,
        k_nope,
        k_rope,
        output_nope_q.stride(0),
        output_nope_s.stride(0),
        output_rope.stride(0),
        k_nope.stride(0),
        k_rope.stride(0),
        NUM_NOPE_BLOCKS=NUM_NOPE_BLOCKS,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
        FP8_MIN=torch.finfo(torch.float8_e4m3fn).min,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
    )

    return {"nope_q": output_nope_q, "nope_s": output_nope_s, "rope": output_rope}
