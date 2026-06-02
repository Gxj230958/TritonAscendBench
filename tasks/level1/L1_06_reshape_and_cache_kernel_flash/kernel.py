"""Task L1_06_reshape_and_cache_kernel_flash — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `reshape_and_cache_kernel_flash`
Source path: `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): a2792421a61c0b20acf5bb21dc9d5cfea8464da17f44a5eee4ed78c35e549bf3
Frozen by: scripts/phase_a_freeze.py
"""
import torch
from tasks._frozen_shim import FP8_DTYPE, get_fp8_min_max  # Phase A': quant symbols, symbol-aware (B-inert-shim)
from tasks._frozen_shim import current_platform  # Phase A': platform stub (B-inert-shim)
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
from tasks._frozen_shim import is_quantized_kv_cache  # Phase A': pure predicate (B-inert-shim)

@triton.jit
def reshape_and_cache_kernel_flash(key_ptr, value_ptr, key_cache_ptr, value_cache_ptr, slot_mapping_ptr, k_scale, v_scale, key_stride: tl.int64, value_stride: tl.int64, block_stride: tl.int64, head_stride: tl.int64, dim_stride_k: tl.int64, dim_stride_v: tl.int64, page_stride: tl.int64, num_heads: tl.constexpr, head_size: tl.constexpr, block_size: tl.constexpr, x: tl.constexpr, USE_HEAD_MAJOR_LAYOUT: tl.constexpr, FP8_KV_CACHE: tl.constexpr, TILE_SIZE: tl.constexpr):
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    tile_i = tl.program_id(axis=1)
    tile_offs = tl.arange(0, TILE_SIZE)
    tile_pos = tile_i * TILE_SIZE + tile_offs
    src_key_idx = token_idx * key_stride
    src_value_idx = token_idx * value_stride
    if USE_HEAD_MAJOR_LAYOUT:
        cur_head = tile_pos // head_size
        cur_dim = tile_pos % head_size
        tgt_idx_v = block_idx * block_stride + cur_head * head_stride + cur_dim * dim_stride_v + block_offset * 1
        tgt_idx_k = block_idx * block_stride + cur_head * head_stride + cur_dim // x * dim_stride_k + block_offset * x + cur_dim % x
    else:
        cur_head = tile_pos // head_size
        cur_dim = tile_pos % head_size
        tgt_idx_k = block_idx * block_stride + block_offset * page_stride + cur_head * head_stride + cur_dim
        tgt_idx_v = tgt_idx_k
    key_load = tl.load(key_ptr + src_key_idx + tile_pos, mask=tile_pos < num_heads * head_size)
    if FP8_KV_CACHE:
        key_tile = key_load if key_load.dtype.is_fp8() else key_load / tl.load(k_scale)
    else:
        key_tile = key_load
    value_load = tl.load(value_ptr + src_value_idx + tile_pos, mask=tile_pos < num_heads * head_size)
    if FP8_KV_CACHE:
        if value_load.dtype.is_fp8():
            value_tile = value_load
        else:
            value_tile = value_load / tl.load(v_scale)
    else:
        value_tile = value_load
    tl.store(key_cache_ptr + tgt_idx_k, key_tile, mask=tile_pos < num_heads * head_size)
    tl.store(value_cache_ptr + tgt_idx_v, value_tile, mask=tile_pos < num_heads * head_size)
    return

import torch
import triton


# Authored launcher: drives the in-module @triton.jit
# `reshape_and_cache_kernel_flash` exactly as the upstream host call site
# `triton_reshape_and_cache_flash` does (vLLM
# vllm/v1/attention/ops/triton_reshape_and_cache_flash.py, commit 18f6bf5),
# for the non-head-major rank-4 paged-KV layout that shapes.yaml supplies.
#
# Why a launcher instead of calling the upstream wrapper directly:
#   * The upstream wrapper mutates key_cache/value_cache IN PLACE and
#     returns None; the GPU oracle-equivalence runner expects a returned
#     dict/tensor, so the in-place None return cannot be consumed.
#   * The static wrapper-rank extractor is branch-insensitive: it sees the
#     `key_cache.shape[4]` access inside the (unreachable, head-major)
#     `if key_cache.ndim == 5` branch and conservatively demands rank >= 5,
#     producing a false WRAPPER_CONTRACT_MISMATCH even though the rank-4
#     `else` branch is the correct, oracle-matching path.
# The launcher allocates its own output by cloning the caches and returns
# them, sidestepping both shared-runner limitations while reproducing the
# upstream kernel math bit-for-bit.
#
# Upstream non-head-major (key_cache.ndim == 4) call-site contract:
#   num_heads     = key.shape[1]
#   head_size     = key.shape[2]
#   block_size    = key_cache.shape[1]
#   x             = 1
#   dim_stride_k  = 0
#   dim_stride_v  = 0
#   head_stride   = key_cache.stride(2)
#   key_stride    = key.stride(0)
#   value_stride  = value.stride(0)
#   block_stride  = key_cache.stride(0)
#   page_stride   = key_cache.stride(1)
#   USE_HEAD_MAJOR_LAYOUT = False
#   FP8_KV_CACHE          = False        (kv_cache_dtype == "auto")
#   n         = num_heads * head_size
#   TILE_SIZE = min(2048, next_power_of_2(n)); capped to 512 on pre-sm90
#   grid      = (slot_mapping.shape[0], cdiv(n, TILE_SIZE))
# Output keys: key_cache, value_cache (the mutated paged caches).
def run(inputs: dict) -> dict:
    key = inputs["key"]
    value = inputs["value"]
    # Clone so the launcher returns a fresh result (the kernel scatters in
    # place); mirrors the oracle's clone-then-scatter semantics.
    key_cache = inputs["key_cache"].clone()
    value_cache = inputs["value_cache"].clone()
    slot_mapping = inputs["slot_mapping"]
    k_scale = inputs["k_scale"]
    v_scale = inputs["v_scale"]

    num_heads = key.shape[1]
    head_size = key.shape[2]

    # Non-head-major (rank-4) paged-KV layout.
    block_size = key_cache.shape[1]
    x = 1
    dim_stride_k = 0
    dim_stride_v = 0
    head_stride = key_cache.stride()[2]

    n = num_heads * head_size
    key_stride = key.stride()[0]
    value_stride = value.stride()[0]
    block_stride = key_cache.stride()[0]
    page_stride = key_cache.stride()[1]

    # kv_cache_dtype == "auto" -> no fp8 quantization.
    FP8_KV_CACHE = False

    # heuristics instead of autotuning (cuda branch of upstream wrapper).
    TILE_SIZE = min(2048, triton.next_power_of_2(n))
    num_stages = 10
    num_warps = 16
    if key.is_cuda and torch.cuda.get_device_capability(key.device)[0] < 9:
        TILE_SIZE = min(512, TILE_SIZE)

    grid = lambda meta: (
        slot_mapping.shape[0],
        triton.cdiv(n, meta["TILE_SIZE"]),
    )

    reshape_and_cache_kernel_flash[grid](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping,
        k_scale=k_scale,
        v_scale=v_scale,
        # strides
        key_stride=key_stride,
        value_stride=value_stride,
        block_stride=block_stride,
        head_stride=head_stride,
        dim_stride_k=dim_stride_k,
        dim_stride_v=dim_stride_v,
        page_stride=page_stride,
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        x=x,
        USE_HEAD_MAJOR_LAYOUT=False,
        FP8_KV_CACHE=FP8_KV_CACHE,
        # autotune parameters
        TILE_SIZE=TILE_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return {"key_cache": key_cache, "value_cache": value_cache}
