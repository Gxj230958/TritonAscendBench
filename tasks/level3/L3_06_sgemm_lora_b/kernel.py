"""Task L3_06_sgemm_lora_b_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_sgemm_lora_b_kernel`
Source path: `sglang/srt/lora/triton_ops/sgemm_lora_b.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): cb5365bab689adcca18ce70b6ed193a53f54bea137c24167ee9bb99a7100c4a5
"""

import torch
import triton
import triton.language as tl

from ascendbench.frozen.deps.h_ebda9fec13c5.sglang_lora_triton_ops.kernel_utils import _resolve_token_positions  # Phase A': U2 vendored sglang LoRA helper
from ascendbench.frozen.shim import LoRABatchInfo  # Phase A': sibling LoRA batch-state stand-in (B-inert-shim)


@triton.jit
def _sgemm_lora_b_kernel(
    # Pointers to matrices
    x,
    weights,
    output,
    # Matrix dimensions
    N,  # output_dim
    K,  # r
    # Strides
    x_stride_0,
    x_stride_1,
    w_stride_0,
    w_stride_1,
    w_stride_2,
    output_stride_0,
    output_stride_1,
    # Information on sequence lengths and weight id
    seg_lens,
    seg_indptr,
    weight_indices,
    lora_ranks,
    sorted_token_ids,
    # Meta parameters
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # For fused output scaling
    scalings,
):
    """
    Computes a segmented batched matrix multiplication for the LoRA B matrix
    and adds the result to the output in-place.

    When a sequence's rank is 0, the kernel is essentially a no-op, following
    the convention in pytorch where the product of two matrices of shape (m, 0)
    and (0, n) is an all-zero matrix of shape (m, n).

    Args:
        x (torch.Tensor): The intermediate tensor from the LoRA 'A' multiplication,
            of shape `(s, K)`, where `s` is the total number of tokens.
        weights (torch.Tensor): The LoRA 'B' weights for all available adapters,
            with shape `(num_lora, N, K)`.
        output (torch.Tensor): The output tensor of shape `(s, N)`. This can be
            the base model's output for a fused add operation.
    """

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len
    batch_id = tl.program_id(axis=1)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel is a no-op.
    if rank == 0:
        return

    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return
    seg_start = tl.load(seg_indptr + batch_id)
    scaling = tl.load(scalings + w_index)
    # Adjust K (rank) according to the specific LoRA adapter
    K = tl.minimum(K, rank)

    # The tile in output matrix will have (pid_s, pid_n) as id
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    # Create pointers for the first block of x and weights[batch_id]
    # The pointers will be advanced as we move in the K direction
    # and accumulate
    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)
    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )
    x_ptrs = x + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    w_ptrs = (weights + w_index * w_stride_0) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    # Iterate to compute the block in output matrix
    n_mask = n_offset[None, :] < N
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & n_mask,
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # Store result to output matrix
    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = output + (
        s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset[:, None] < seg_len) & n_mask
    partial_sum += tl.load(output_ptr, mask=output_mask, other=0.0)
    tl.store(output_ptr, partial_sum, mask=output_mask)



import torch
import triton


# Drives the in-module @triton.jit `_sgemm_lora_b_kernel` exactly as the
# upstream host wrapper `sgemm_lora_b_fwd` does (SGLang
# sglang/srt/lora/triton_ops/sgemm_lora_b.py, commit 409d350f).
#
# The wrapper takes a `sglang.srt.lora.utils.LoRABatchInfo`, whose import pulls
# in the full sglang runtime (BLOCKED closure). The kernel itself only reads
# the dense tensors plus a derived seg_lens/max_len, so we reconstruct exactly
# the batch_info fields it forwards from the dense oracle inputs.
#
# permutation is None -> SORTED_BY_ADAPTER False, so `_resolve_token_positions`
# returns seg_start + s_offset, the identity row map the oracle uses. seg_lens
# is derived from the authoritative seg_indptr to restore the invariant
# seg_indptr == cumsum([0, seg_lens]). The kernel accumulates into `output`;
# starting from zeros makes that equal the oracle's base-free reference.

import torch
import triton


# Drives the in-module @triton.jit `_sgemm_lora_b_kernel` exactly as the
# upstream host wrapper `sgemm_lora_b_fwd` does (SGLang
# sglang/srt/lora/triton_ops/sgemm_lora_b.py, commit 409d350f).
#
# The wrapper takes a `sglang.srt.lora.utils.LoRABatchInfo`, whose import pulls
# in the full sglang runtime (BLOCKED closure). The kernel itself only reads
# the dense tensors plus a derived seg_lens/max_len, so we reconstruct exactly
# the batch_info fields it forwards from the dense oracle inputs.
#
# permutation is None -> SORTED_BY_ADAPTER False, so `_resolve_token_positions`
# returns seg_start + s_offset, the identity row map the oracle uses. seg_lens
# is derived from the authoritative seg_indptr to restore the invariant
# seg_indptr == cumsum([0, seg_lens]). The kernel accumulates into `output`;
# starting from zeros makes that equal the oracle's base-free reference.
def run(inputs: dict) -> dict:
    x = inputs["x"].contiguous()
    weights = inputs["weights"].contiguous()
    seg_indptr = inputs["seg_indptr"]
    weight_indices = inputs["weight_indices"]
    lora_ranks = inputs["lora_ranks"]
    scalings = inputs["scalings"]

    S = x.shape[0]
    N = weights.shape[1]
    R = weights.shape[2]

    seg_indptr_l = seg_indptr.to(torch.int64)
    bs = seg_indptr_l.numel() - 1
    seg_lens = (seg_indptr_l[1:] - seg_indptr_l[:-1]).to(torch.int32).contiguous()
    max_len = int(seg_lens.max().item()) if bs > 0 else 0

    weight_indices = weight_indices.to(torch.int64).contiguous()
    lora_ranks = lora_ranks.to(torch.int64).contiguous()
    scalings = scalings.to(torch.float32).contiguous()
    seg_indptr_k = seg_indptr_l.contiguous()

    permutation = None
    sorted_by_adapter = permutation is not None

    BLOCK_S = 16
    BLOCK_N = 256
    BLOCK_R = 16

    output = torch.zeros((S, N), device=x.device, dtype=x.dtype)

    if max_len > 0 and bs > 0:
        grid = (
            triton.cdiv(max_len, BLOCK_S) * triton.cdiv(N, BLOCK_N),
            bs,
        )
        _sgemm_lora_b_kernel[grid](
            x,
            weights,
            output,
            N,
            R,
            x.stride(0),
            x.stride(1),
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            output.stride(0),
            output.stride(1),
            seg_lens,
            seg_indptr_k,
            weight_indices,
            lora_ranks,
            permutation,
            sorted_by_adapter,
            BLOCK_S,
            BLOCK_N,
            BLOCK_R,
            scalings,
        )

    return {"output": output}
