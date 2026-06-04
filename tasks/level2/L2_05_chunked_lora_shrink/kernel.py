"""Task L2_05_chunked_lora_shrink_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_chunked_lora_shrink_kernel`
Source path: `sglang/srt/lora/triton_ops/chunked_sgmv_shrink.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): 23cb7b2a5868763fb0d5ab519a0e839ba992dd4ae43076ec8b288925c34823ec
Frozen by: scripts/phase_a_freeze.py
"""

import torch
import triton
import triton.language as tl

from tasks._frozen_shim import get_lora_shrink_config  # Phase A': LoRA block-size config (B-inert-shim)
from tasks._frozen_shim import LoRABatchInfo  # Phase A': sibling LoRA batch-state stand-in (B-inert-shim)
from tasks._frozen_shim import cached_triton_kernel  # Phase A': platform/math predicate (B-inert-shim)


@cached_triton_kernel(
    lambda _, kwargs: (kwargs["K"], kwargs["NUM_SLICES"], kwargs["BLOCK_M"])
)
@triton.jit(do_not_specialize=["num_segs"])
def _chunked_lora_shrink_kernel(
    # Pointers to matrices
    x,
    weights,
    output,
    # Information on sequence lengths,ranks and weight id
    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    num_segs,
    # Meta parameters
    N: tl.constexpr,  # num_slices * r
    K: tl.constexpr,  # input_dim
    NUM_SLICES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes a chunked SGMV for LoRA shrink operations.

    The kernel ensures that output[seg_start:seg_start + seg_len, :rank * num_slices]
    stores the product of the input `x` and the LoRA weights for the corresponding
    sequence. This implies that when rank is 0, the kernel is essentially a no-op,
    as output[seg_start:seg_start + seg_len, :0] is trivially correct (empty).

    Args:
        x (torch.Tensor): The input activations tensor of shape `(s, K)`, where `s`
            is the sum of all sequence lengths in the batch.
        weights (torch.Tensor): The LoRA A weights for all available adapters,
            with shape `(num_lora, N, K)` where N = num_slices * r.
        output (torch.Tensor): The output tensor of shape `(s, N)`.
    """
    x_stride_1: tl.constexpr = 1
    x_stride_0: tl.constexpr = K

    w_stride_0: tl.constexpr = N * K
    w_stride_1: tl.constexpr = K
    w_stride_2: tl.constexpr = 1

    output_stride_0: tl.constexpr = N
    output_stride_1: tl.constexpr = 1

    pid_s = tl.program_id(1)
    if pid_s >= num_segs:
        return

    pid_n = tl.program_id(0)

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len
    w_index = tl.load(weight_indices + pid_s)
    rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel becomes a no-op as the output is always trivially correct.
    if rank == 0:
        return

    seg_start = tl.load(seg_indptr + pid_s)
    seg_end = tl.load(seg_indptr + pid_s + 1)

    # Adjust N dim according to the specific LoRA adapter
    cur_n = tl.minimum(N, rank * NUM_SLICES)

    # Map logical sequence index to physical index
    s_offset_logical = tl.arange(0, BLOCK_M) + seg_start
    s_offset_physical = tl.load(
        permutation + s_offset_logical, mask=s_offset_logical < seg_end
    )

    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)
    x_ptrs = x + (
        s_offset_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1
    )
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    # Iterate to compute the block in output matrix
    partial_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset_logical[:, None] < seg_end)
            & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < cur_n),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # Store result to output matrix
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = output + (
        s_offset_physical[:, None] * output_stride_0
        + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset_logical[:, None] < seg_end) & (n_offset[None, :] < cur_n)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def chunked_sgmv_lora_shrink_forward(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    num_slices: int,
) -> torch.Tensor:
    # x: (s, input_dim)
    # weights: (num_lora, num_slices * r, input_dim)
    # output: (s, num_slices * r)
    # num_slices: qkv=3, gate_up=2, others=1
    # when called with multiple slices, the weights.shape[-2] will be num_slices * r
    # input_dim is much larger than r

    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert len(x.shape) == 2
    assert len(weights.shape) == 3

    # Block shapes — use auto-tuned config if available, else defaults
    BLOCK_M = batch_info.max_len
    # weights shape is (num_lora, num_slices * rank, input_dim)
    MAX_RANK = weights.shape[1] // num_slices
    config = get_lora_shrink_config(
        K=weights.shape[2], R=MAX_RANK, num_slices=num_slices, chunk_size=BLOCK_M
    )
    BLOCK_N = config["BLOCK_N"]
    BLOCK_K = config["BLOCK_K"]

    S = x.shape[0]
    N = weights.shape[1]
    K = weights.shape[2]
    assert x.shape[-1] == K

    num_segments = batch_info.num_segments
    grid = (
        triton.cdiv(N, BLOCK_N),
        batch_info.bs if batch_info.use_cuda_graph else num_segments,
    )

    # Optional launch params from tuned config
    extra_kwargs = {}
    if "num_warps" in config:
        extra_kwargs["num_warps"] = config["num_warps"]
    if "num_stages" in config:
        extra_kwargs["num_stages"] = config["num_stages"]

    output = torch.empty((S, N), device=x.device, dtype=x.dtype)
    _chunked_lora_shrink_kernel[grid](
        x=x,
        weights=weights,
        output=output,
        seg_indptr=batch_info.seg_indptr,
        weight_indices=batch_info.weight_indices,
        lora_ranks=batch_info.lora_ranks,
        permutation=batch_info.permutation,
        num_segs=num_segments,
        # constants
        N=N,
        K=K,
        NUM_SLICES=num_slices,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        **extra_kwargs,
    )

    return output

import torch
import triton


# Drives the in-module @triton.jit `_chunked_lora_shrink_kernel` exactly as the
# upstream host wrapper `chunked_sgmv_lora_shrink_forward` does (SGLang
# sglang/srt/lora/triton_ops/chunked_sgmv_shrink.py, commit 409d350f).
#
# Why a launcher rather than the host wrapper: the wrapper calls
# `get_lora_shrink_config` -> `get_device_name()`, which returns None off-GPU
# (AttributeError: 'NoneType'.replace), so the wrapper is not invocable here.
# It also takes a `sglang.srt.lora.utils.LoRABatchInfo` whose import pulls the
# full sglang runtime (a T3 BLOCKED closure). The kernel itself only reads plain
# dense tensors plus a derived num_segments and a permutation, so we reconstruct
# exactly the batch_info fields the wrapper forwards from the dense oracle
# inputs. (Same pattern as the passed siblings L2_04_chunked_lora_expand /
# L2_06_qkv_lora_b.)
#
# Layout reconciliation (dense fixtures -> upstream batch_info contract):
#   seg_indptr is the authoritative segment-boundary source (the oracle's
#   boundary source); the real LoRABatchInfo always satisfies
#   seg_indptr == cumsum([0, seg_lens]). num_segments = len(seg_indptr) - 1.
#   BLOCK_M = max_len = max segment length (the grid/tiling axis-0 extent: it
#   must cover the longest segment or that segment under-launches). The kernel
#   always dereferences `permutation` (output[seg_start:seg_end] is stored at
#   permutation-mapped physical rows); the oracle uses no reordering, i.e. the
#   identity row mapping, so we supply the identity arange(S) -- matching the
#   oracle exactly. The kernel reads weights[weight_indices[seg]] and masks
#   output columns by lora_ranks[weight_indices[seg]] * NUM_SLICES, so both
#   index vectors are load-bearing and forwarded verbatim.
#
# Block sizes are the upstream untuned shrink defaults (BLOCK_N=16, BLOCK_K=256,
# mirroring lora_tuning_config.DEFAULT_SHRINK_CONFIG); tiling only, masked, never
# the result. Grid = (cdiv(N, BLOCK_N), num_segments), as the wrapper builds it
# off the CUDA-graph path (use_cuda_graph=False).
def run(inputs: dict) -> dict:
    x = inputs["x"].contiguous()
    weights = inputs["weights"].contiguous()
    seg_indptr = inputs["seg_indptr"]
    weight_indices = inputs["weight_indices"]
    lora_ranks = inputs["lora_ranks"]
    num_slices = int(inputs.get("num_slices", 1))

    S = x.shape[0]
    N = weights.shape[1]
    K = weights.shape[2]
    assert x.shape[-1] == K, (x.shape, weights.shape)

    seg_indptr = seg_indptr.to(torch.int64).contiguous()
    num_segments = seg_indptr.numel() - 1
    seg_lens = seg_indptr[1:] - seg_indptr[:-1]
    BLOCK_M = int(seg_lens.max().item()) if num_segments > 0 else 0

    weight_indices = weight_indices.to(torch.int64).contiguous()
    lora_ranks = lora_ranks.to(torch.int64).contiguous()

    # permutation is identity (no adapter-sorted reorder) -- the oracle's row
    # mapping. The kernel indexes x and output through permutation[logical], so
    # an identity arange(S) reproduces the unpermuted oracle layout exactly.
    permutation = torch.arange(S, device=x.device, dtype=torch.int64)

    BLOCK_N = 16
    BLOCK_K = 256

    # The kernel only stores the active window output[seg, :rank*NUM_SLICES];
    # rows outside any segment and columns >= cur_n are left untouched. The
    # oracle returns those positions as zero, so zero-init (not torch.empty,
    # which the upstream wrapper uses but whose downstream consumer ignores the
    # tail columns) to reproduce the oracle's full-tensor zero padding.
    output = torch.zeros((S, N), device=x.device, dtype=x.dtype)

    if BLOCK_M > 0 and num_segments > 0:
        grid = (
            triton.cdiv(N, BLOCK_N),
            num_segments,
        )
        _chunked_lora_shrink_kernel[grid](
            x=x,
            weights=weights,
            output=output,
            seg_indptr=seg_indptr,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            permutation=permutation,
            num_segs=num_segments,
            N=N,
            K=K,
            NUM_SLICES=num_slices,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

    return {"output": output}
