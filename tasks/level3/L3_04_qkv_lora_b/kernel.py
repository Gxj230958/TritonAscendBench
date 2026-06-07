"""Task L3_04_qkv_lora_b_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_qkv_lora_b_kernel`
Source path: `sglang/srt/lora/triton_ops/qkv_lora_b.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): 151793ef5fbd2910c1e5ffe6d95d6d6f48df73a6459d17239526ab83dac39191
Frozen by: scripts/phase_a_freeze.py
"""

import torch
import triton
import triton.language as tl

from tasks._frozen_deps_pool.h_ebda9fec13c5.sglang_lora_triton_ops.kernel_utils import _resolve_token_positions  # Phase A': U2 vendored sglang LoRA helper
from tasks._frozen_shim import LoRABatchInfo  # Phase A': sibling LoRA batch-state stand-in (B-inert-shim)


@triton.jit
def _qkv_lora_b_kernel(
    # Pointers to matrices
    x,
    weights,
    output,
    # Parameters of size
    K,  # K = R
    max_qkv_out_dim,  # max(output_q_dim, output_kv_dim)
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
    # Offsets of q/k/v slice on output dimension
    n_offs,
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
    This kernel packs 3 sgemms (q/k/v) into a single kernel. The multiplication
    results are accumulated into the output tensor.

    When a sequence's rank is 0, the kernel is essentially a no-op, following
    the convention in pytorch where the product of two matrices of shape (m, 0)
    and (0, n) is an all-zero matrix of shape (m, n).

    Args:
        x (Tensor): The input tensor, which is the result of the LoRA A projection.
            Shape: (s, 3 * K), where s is the sum of all sequence lengths in the
            batch and K is the maximum LoRA rank. The second dimension is partitioned
            for Q, K, and V.
        weights (Tensor): The LoRA B weights for all adapters.
            Shape: (num_lora, N_Q + 2 * N_KV, K).
        output (Tensor): The output tensor where the result is stored.
            Shape: (s, N_Q + 2 * N_KV).
    """

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len.
    # qkv_id decides which of q,k,v to compute (0: q, 1: k, 2: v)
    batch_id = tl.program_id(axis=2)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel is a no-op.
    if rank == 0:
        return

    qkv_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return
    seg_start = tl.load(seg_indptr + batch_id)
    n_start = tl.load(n_offs + qkv_id)
    n_size = tl.load(n_offs + qkv_id + 1) - n_start
    scaling = tl.load(scalings + w_index)
    # Adjust K (rank) according to the specific LoRA adapter
    K = tl.minimum(K, rank)

    # The tile in output matrix will have (pid_s, pid_n) as id
    num_pid_n = tl.cdiv(max_qkv_out_dim, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    # Create pointers for the first block of x and weights[batch_id][n_start: n_end][:]
    # The pointers will be advanced as we move in the K direction
    # and accumulate
    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)

    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )
    x_ptrs = (
        x
        + (qkv_id * K) * x_stride_1
        + (s_physical[:, None] * x_stride_0 + k_offset[None, :] * x_stride_1)
    )
    w_ptrs = (weights + w_index * w_stride_0 + n_start * w_stride_1) + (
        k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1
    )

    # Iterate to compute the block in output matrix
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < n_size),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # Store result to output matrix
    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = (
        output
        + n_start * output_stride_1
        + (s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1)
    )
    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < n_size)
    partial_sum += tl.load(output_ptr, mask=output_mask)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def qkv_lora_b_fwd(
    x: torch.Tensor,
    qkv_lora_b: torch.Tensor,
    batch_info: LoRABatchInfo,
    output_offset: torch.Tensor,
    max_qkv_out_dim: int,
    base_output: torch.Tensor = None,
    n_slices: int = 3,
) -> torch.Tensor:

    # x: (s, n_slices * r)
    # qkv_lora_b: (num_lora, output_dim_q + 2 * output_dim_kv, r)
    # output_offset = [0, output_dim_q, output_dim_q + output_dim_kv,
    #                     output_dim_q + 2 * output_dim_kv]  (length n_slices + 1)
    # max_qkv_out_dim = max(output_dim_q, output_dim_kv)
    # output: (s, output_dim_q + 2 * output_dim_kv)

    # Compute lora_output with shape (s, output_dim) as follows:
    # lora_output[:, :output_dim_q] = sgemm(x[:, :r], qkv_lora_b[:, :outptu_dim_q, :])
    # lora_output[:, output_dim_q: output_dim_q + output_dim_kv]
    #      = sgemm(x[:, r: 2 * r], qkv_lora_b[:, outptu_dim_q: output_dim_q + output_dim_kv, :])
    # lora_output[:, output_dim_q + output_dim_kv: ]
    #      = sgemm(x[:, 2 * r: , qkv_lora_b[:, output_dim_q + output_dim_kv: , :])

    # Get dims
    s = x.shape[0]
    input_dim = x.shape[1]
    r = qkv_lora_b.shape[-1]
    output_dim = qkv_lora_b.shape[-2]
    assert input_dim == n_slices * r
    assert output_offset.shape[0] == n_slices + 1

    BLOCK_S = 16
    BLOCK_R = 16
    BLOCK_OUT = 64

    grid_b = (
        triton.cdiv(batch_info.max_len, BLOCK_S)
        * triton.cdiv(max_qkv_out_dim, BLOCK_OUT),
        n_slices,
        batch_info.bs,
    )

    if base_output is None:
        output = torch.zeros((s, output_dim), device=x.device, dtype=x.dtype)
    else:
        output = base_output

    sorted_by_adapter = batch_info.permutation is not None
    _qkv_lora_b_kernel[grid_b](
        x,
        qkv_lora_b,
        output,
        r,
        max_qkv_out_dim,
        x.stride(0),
        x.stride(1),
        qkv_lora_b.stride(0),
        qkv_lora_b.stride(1),
        qkv_lora_b.stride(2),
        output.stride(0),
        output.stride(1),
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        output_offset,
        batch_info.permutation,
        sorted_by_adapter,
        BLOCK_S,
        BLOCK_OUT,
        BLOCK_R,
        batch_info.scalings,
    )

    return output

import torch
import triton


# Authored launcher: drives the in-module @triton.jit `_qkv_lora_b_kernel`
# exactly as the upstream host wrapper `qkv_lora_b_fwd` does (SGLang
# sglang/srt/lora/triton_ops/qkv_lora_b.py, commit
# 409d350fb6f6a1e7c7546e39028f811092a8e489).
#
# Why a launcher rather than the host wrapper: the host entry
# `qkv_lora_b_fwd(x, qkv_lora_b, batch_info, output_offset, max_qkv_out_dim,
# ...)` takes a `sglang.srt.lora.utils.LoRABatchInfo` dataclass whose import
# transitively pulls the full sglang runtime (ForwardBatch, vllm._C, fastapi,
# prometheus_client, ...). That closure is T3 BLOCKED-candidate (closure cycle,
# >800 symbols) and none of it is needed to exercise the unit under test: the
# kernel only reads plain dense tensors plus a derived seg_lens/max_len. We
# reconstruct exactly the batch_info fields the wrapper forwards, straight from
# the dense oracle inputs. (Same pattern as L2_16_sgemm_lora_a.)
#
# Upstream invocation reproduced here (qkv_lora_b_fwd):
#   x: (s, n_slices * r) -- the LoRA-A output, partitioned into n_slices=3
#     column blocks for Q/K/V.
#   qkv_lora_b: (num_lora, output_dim_q + 2*output_dim_kv, r).
#   output_offset = [0, output_dim_q, output_dim_q + output_dim_kv,
#                    output_dim_q + 2*output_dim_kv] (length n_slices + 1).
#   max_qkv_out_dim = max(output_dim_q, output_dim_kv).
#   output: (s, output_dim_q + 2*output_dim_kv).
#   K (kernel arg) = r = qkv_lora_b.shape[-1]; BLOCK_S=16, BLOCK_OUT=64,
#   BLOCK_R=16.
#   grid = (cdiv(max_len, BLOCK_S) * cdiv(max_qkv_out_dim, BLOCK_OUT),
#           n_slices, bs).
#   The wrapper passes batch_info.{seg_lens, seg_indptr, weight_indices,
#   lora_ranks, scalings, permutation}; sorted_by_adapter = permutation is not
#   None.
#
# Layout reconciliation (dense fixtures -> upstream batch_info contract):
#   A segment's row span is defined the SAME way by oracle and kernel when the
#   two segment descriptors are consistent: the wrapper drives row offsets from
#   seg_indptr[batch_id] (seg_start) with length seg_lens[batch_id]. The real
#   LoRABatchInfo always satisfies seg_indptr == cumsum([0, seg_lens]), i.e.
#   seg_lens[i] = seg_indptr[i+1] - seg_indptr[i]. The dense fixture supplies
#   the authoritative seg_indptr (the oracle's boundary source), so we DERIVE
#   seg_lens from it to restore that invariant exactly. bs = num_segments =
#   len(seg_indptr) - 1; max_len = max(seg_lens) (the grid axis-0 extent --
#   it must cover the longest segment or that segment under-launches).
#   permutation is None (tokens contiguous, SORTED_BY_ADAPTER=False), so
#   `_resolve_token_positions` returns seg_start + s_offset -- the identity row
#   mapping the oracle uses (no adapter-sorted reordering).
#
# Per (segment batch_id, slice qkv_id): w_index = weight_indices[batch_id]
# selects the adapter; rank = lora_ranks[w_index]; K is clamped to
# min(r, rank); x reads columns [qkv_id*K : qkv_id*K + K] (note the CLAMPED K
# stride, matching the kernel's `qkv_id * K` offset after `K = min(K, rank)`);
# weights read qkv_lora_b[w_index, n_start:n_start+n_size, :K] for the slice's
# output_offset window; output[:, n_start:n_start+n_size] += scaling * (x @
# w.T). Output key matches the oracle contract: output, fp32 [s, 3*N].
def run(inputs: dict) -> dict:
    x = inputs["x"].contiguous()
    qkv_lora_b = inputs["qkv_lora_b"].contiguous()
    seg_indptr = inputs["seg_indptr"]
    weight_indices = inputs["weight_indices"]
    lora_ranks = inputs["lora_ranks"]
    scalings = inputs["scalings"]

    s = x.shape[0]
    r = qkv_lora_b.shape[-1]
    output_dim = qkv_lora_b.shape[-2]
    n_slices = 3
    assert x.shape[1] == n_slices * r, (x.shape, r)
    assert output_dim % n_slices == 0, output_dim
    per_slice = output_dim // n_slices  # output_dim_q == output_dim_kv
    max_qkv_out_dim = per_slice

    # output_offset = [0, per, 2*per, 3*per]; length n_slices + 1.
    output_offset = torch.tensor(
        [i * per_slice for i in range(n_slices + 1)],
        device=x.device, dtype=torch.int32,
    ).contiguous()

    # Restore the upstream invariant seg_indptr == cumsum([0, seg_lens]).
    seg_indptr_l = seg_indptr.to(torch.int64)
    num_segments = seg_indptr_l.numel() - 1
    seg_lens = (seg_indptr_l[1:] - seg_indptr_l[:-1]).to(torch.int32).contiguous()
    bs = num_segments
    max_len = int(seg_lens.max().item()) if num_segments > 0 else 0

    weight_indices = weight_indices.to(torch.int64).contiguous()
    lora_ranks = lora_ranks.to(torch.int64).contiguous()
    scalings = scalings.to(torch.float32).contiguous()
    seg_indptr_k = seg_indptr_l.contiguous()

    # permutation is None -> SORTED_BY_ADAPTER is False (contiguous tokens).
    permutation = None
    sorted_by_adapter = permutation is not None

    BLOCK_S = 16
    BLOCK_OUT = 64
    BLOCK_R = 16

    # Upstream allocates output with zeros and the kernel accumulates into it
    # (output += partial). Zero-init matches the oracle's base==None reference.
    output = torch.zeros((s, output_dim), device=x.device, dtype=x.dtype)

    if max_len > 0 and bs > 0:
        grid = (
            triton.cdiv(max_len, BLOCK_S) * triton.cdiv(max_qkv_out_dim, BLOCK_OUT),
            n_slices,
            bs,
        )
        _qkv_lora_b_kernel[grid](
            x,
            qkv_lora_b,
            output,
            r,
            max_qkv_out_dim,
            x.stride(0),
            x.stride(1),
            qkv_lora_b.stride(0),
            qkv_lora_b.stride(1),
            qkv_lora_b.stride(2),
            output.stride(0),
            output.stride(1),
            seg_lens,
            seg_indptr_k,
            weight_indices,
            lora_ranks,
            output_offset,
            permutation,
            sorted_by_adapter,
            BLOCK_S,
            BLOCK_OUT,
            BLOCK_R,
            scalings,
        )

    return {"output": output}
