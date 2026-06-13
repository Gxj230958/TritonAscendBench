"""Task L3_03_chunked_lora_expand_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_chunked_lora_expand_kernel`
Source path: `sglang/srt/lora/triton_ops/chunked_sgmv_expand.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): e4b90c63ab3a04d78c5a8dd9ed453ab9409ef9873ec889a161dfd76303dbac50
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from ascendbench.frozen.shim import get_lora_expand_config  # Phase A': LoRA block-size config (B-inert-shim)
from ascendbench.frozen.shim import LoRABatchInfo  # Phase A': sibling LoRA batch-state stand-in (B-inert-shim)
from ascendbench.frozen.shim import cached_triton_kernel  # Phase A': platform/math predicate (B-inert-shim)


@cached_triton_kernel(
    lambda _, kwargs: (kwargs["NUM_SLICES"], kwargs["BLOCK_M"], kwargs["OUTPUT_DIM"])
)
@triton.jit(do_not_specialize=["num_segs", "output_stride_0", "output_stride_1"])
def _chunked_lora_expand_kernel(
    # Pointers to matrices
    x,
    weights,
    output,
    # Output strides may differ from OUTPUT_DIM when compact LoRA output is
    # accumulated into a wider base projection.
    output_stride_0,
    output_stride_1,
    # Information on sequence lengths and weight id
    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    num_segs,
    # For fused output scaling
    scalings,
    # Offsets of q/k/v slice on output dimension
    slice_offsets,
    # Meta parameters
    NUM_SLICES: tl.constexpr,
    OUTPUT_DIM: tl.constexpr,
    MAX_RANK: tl.constexpr,  # K = R
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes a chunked SGMV for LoRA expand operations.

    When a sequence's rank is 0, the kernel is essentially a no-op, following
    the convention in pytorch where the product of two matrices of shape (m, 0)
    and (0, n) is an all-zero matrix of shape (m, n).

    Args:
        x (Tensor): The input tensor, which is the result of the LoRA A projection.
            Shape: (s, num_slices * K), where s is the sum of all sequence lengths in the
            batch and K is the maximum LoRA rank.
        weights (Tensor): The LoRA B weights for all adapters.
            Shape: (num_lora, output_dim, K).
        output (Tensor): The output tensor where the result is stored.
            Shape: (s, output_dim) or a wider base output.
    """
    x_stride_0: tl.constexpr = NUM_SLICES * MAX_RANK
    x_stride_1: tl.constexpr = 1

    w_stride_0: tl.constexpr = OUTPUT_DIM * MAX_RANK
    w_stride_1: tl.constexpr = MAX_RANK
    w_stride_2: tl.constexpr = 1

    pid_s = tl.program_id(axis=2)
    if pid_s >= num_segs:
        return

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len.
    # qkv_id decides which of q,k,v to compute (0: q, 1: k, 2: v)
    w_index = tl.load(weight_indices + pid_s)
    cur_rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel is a no-op.
    if cur_rank == 0:
        return

    seg_start = tl.load(seg_indptr + pid_s)
    seg_end = tl.load(seg_indptr + pid_s + 1)

    slice_id = tl.program_id(axis=1)
    slice_start = tl.load(slice_offsets + slice_id)
    slice_end = tl.load(slice_offsets + slice_id + 1)

    scaling = tl.load(scalings + w_index)
    # Adjust K (rank) according to the specific LoRA adapter
    cur_rank = tl.minimum(MAX_RANK, cur_rank)

    # Map logical sequence index to physical index
    s_offset_logical = tl.arange(0, BLOCK_M) + seg_start
    s_offset_physical = tl.load(
        permutation + s_offset_logical, mask=s_offset_logical < seg_end
    )

    # Create pointers for the first block of x and weights[batch_id][n_start: n_end][:]
    # The pointers will be advanced as we move in the K direction
    # and accumulate
    pid_n = tl.program_id(axis=0)
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + slice_start
    k_offset = tl.arange(0, BLOCK_K)

    x_ptrs = (
        x
        + slice_id * cur_rank * x_stride_1
        + (s_offset_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    )
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    # Iterate to compute the block in output matrix
    partial_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(cur_rank, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset_logical[:, None] < seg_end)
            & (k_offset[None, :] < cur_rank - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < cur_rank - k * BLOCK_K)
            & (n_offset[None, :] < slice_end),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # Store result to output matrix
    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = output + (
        s_offset_physical[:, None] * output_stride_0
        + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset_logical[:, None] < seg_end) & (
        n_offset[None, :] < slice_end
    )
    partial_sum += tl.load(output_ptr, mask=output_mask, other=0.0)
    tl.store(output_ptr, partial_sum, mask=output_mask)



import torch
import triton


# Drives the in-module @triton.jit `_chunked_lora_expand_kernel` exactly as the
# upstream host wrapper `chunked_sgmv_lora_expand_forward` does (SGLang
# sglang/srt/lora/triton_ops/chunked_sgmv_expand.py, commit 409d350f).
#
# The wrapper takes a `sglang.srt.lora.utils.LoRABatchInfo` (BLOCKED closure);
# the kernel only reads the dense tensors plus a derived seg_lens/max_len and an
# identity permutation, so we reconstruct exactly the batch_info fields it
# forwards from the dense oracle inputs. seg_indptr is the authoritative segment
# boundary source (invariant seg_indptr == cumsum([0, seg_lens])); BLOCK_M =
# max_len = max segment length. The kernel always dereferences `permutation`, so
# we supply the identity arange(S) -- the deterministic upstream lm_head case,
# matching the oracle's identity default. Block sizes are the untuned defaults
# (tiling only, not the result); the grid derives max_slice_size from
# slice_offsets so every output slice column is covered.

import torch
import triton


# Drives the in-module @triton.jit `_chunked_lora_expand_kernel` exactly as the
# upstream host wrapper `chunked_sgmv_lora_expand_forward` does (SGLang
# sglang/srt/lora/triton_ops/chunked_sgmv_expand.py, commit 409d350f).
#
# The wrapper takes a `sglang.srt.lora.utils.LoRABatchInfo` (BLOCKED closure);
# the kernel only reads the dense tensors plus a derived seg_lens/max_len and an
# identity permutation, so we reconstruct exactly the batch_info fields it
# forwards from the dense oracle inputs. seg_indptr is the authoritative segment
# boundary source (invariant seg_indptr == cumsum([0, seg_lens])); BLOCK_M =
# max_len = max segment length. The kernel always dereferences `permutation`, so
# we supply the identity arange(S) -- the deterministic upstream lm_head case,
# matching the oracle's identity default. Block sizes are the untuned defaults
# (tiling only, not the result); the grid derives max_slice_size from
# slice_offsets so every output slice column is covered.
def run(inputs: dict) -> dict:
    x = inputs["x"].contiguous()
    weights = inputs["weights"].contiguous()
    seg_indptr = inputs["seg_indptr"]
    weight_indices = inputs["weight_indices"]
    lora_ranks = inputs["lora_ranks"]
    scalings = inputs["scalings"]
    slice_offsets = inputs["slice_offsets"]

    S = x.shape[0]
    OUTPUT_DIM = weights.shape[1]
    MAX_RANK = weights.shape[2]

    slice_offsets = slice_offsets.to(torch.int32).contiguous()
    num_slices = slice_offsets.numel() - 1

    seg_indptr = seg_indptr.to(torch.int64).contiguous()
    num_segs = seg_indptr.numel() - 1
    seg_lens = seg_indptr[1:] - seg_indptr[:-1]
    BLOCK_M = int(seg_lens.max().item()) if num_segs > 0 else 0

    weight_indices = weight_indices.to(torch.int64).contiguous()
    lora_ranks = lora_ranks.to(torch.int64).contiguous()
    scalings = scalings.to(torch.float32).contiguous()

    permutation = torch.arange(S, device=x.device, dtype=torch.int32)

    max_slice_size = (
        int((slice_offsets[1:] - slice_offsets[:-1]).max().item())
        if num_slices > 0
        else 0
    )

    BLOCK_N = 64
    BLOCK_K = 16

    output = torch.zeros((S, OUTPUT_DIM), device=x.device, dtype=x.dtype)

    if BLOCK_M > 0 and num_segs > 0 and max_slice_size > 0:
        grid = (
            triton.cdiv(max_slice_size, BLOCK_N),
            num_slices,
            num_segs,
        )
        _chunked_lora_expand_kernel[grid](
            x=x,
            weights=weights,
            output=output,
            output_stride_0=output.stride(0),
            output_stride_1=output.stride(1),
            seg_indptr=seg_indptr,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            permutation=permutation,
            num_segs=num_segs,
            scalings=scalings,
            slice_offsets=slice_offsets,
            NUM_SLICES=num_slices,
            OUTPUT_DIM=OUTPUT_DIM,
            MAX_RANK=MAX_RANK,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

    return {"output": output}
