"""Task L2_16_sgemm_lora_a_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_sgemm_lora_a_kernel`
Source path: `sglang/srt/lora/triton_ops/sgemm_lora_a.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): ddfb940899cfdd908dc1bf122e8c379b72fe18a54d9d408969e88f46be939b05
Frozen by: scripts/phase_a_freeze.py
"""

import torch
import triton
import triton.language as tl

from tasks._frozen_deps_pool.h_ebda9fec13c5.sglang_lora_triton_ops.kernel_utils import _resolve_token_positions  # Phase A': U2 vendored sglang LoRA helper
from tasks._frozen_shim import LoRABatchInfo  # Phase A': sibling LoRA batch-state stand-in (B-inert-shim)


@triton.jit
def _sgemm_lora_a_kernel(
    # Pointers to matrices
    x,
    weights,
    output,
    # Matrix dimensions
    N,  # stack_num * r
    K,  # input_dim
    stack_num,
    # Strides
    x_stride_0,
    x_stride_1,
    w_stride_0,
    w_stride_1,
    w_stride_2,
    output_stride_0,
    output_stride_1,
    # Information on sequence lengths,ranks and weight id
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
):
    """
    Computes a segmented batched matrix multiplication for the LoRA A matrix.

    The kernel ensures that output[seg_start:seg_start + seg_len, :rank * stack_num]
    stores the product of the input `x` and the LoRA weights for the corresponding
    sequence. This implies that when rank is 0, the kernel is essentially a no-op,
    as output[seg_start:seg_start + seg_len, :0] is trivially correct (empty).

    Args:
        x (torch.Tensor): The input activations tensor of shape `(s, K)`, where `s`
            is the sum of all sequence lengths in the batch.
        weights (torch.Tensor): The LoRA 'A' weights for all available adapters,
            with shape `(num_lora, N, K)`.
        output (torch.Tensor): The output tensor of shape `(s, N)`.
    """

    # Current block computes sequence with batch_id,
    # which starts from row seg_start of x with length seg_len
    batch_id = tl.program_id(axis=1)
    w_index = tl.load(weight_indices + batch_id)
    rank = tl.load(lora_ranks + w_index)

    # If rank is 0, this kernel becomes a no-op as the output is always trivially correct.
    if rank == 0:
        return

    pid = tl.program_id(axis=0)
    seg_start = tl.load(seg_indptr + batch_id)
    seg_len = tl.load(seg_lens + batch_id)
    if seg_len == 0:
        return

    # Adjust N (stack_num * max_rank) according to the specific LoRA adapter
    N = tl.minimum(N, rank * stack_num)

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
    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset[:, None] < seg_len) & (k_offset[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < K - k * BLOCK_K) & (n_offset[None, :] < N),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * x_stride_1
        w_ptrs += BLOCK_K * w_stride_2

    # Store result to output matrix
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_mask = (s_offset[:, None] < seg_len) & (n_offset[None, :] < N)
    output_ptr = output + (
        s_physical[:, None] * output_stride_0 + n_offset[None, :] * output_stride_1
    )
    tl.store(output_ptr, partial_sum, mask=output_mask)


def sgemm_lora_a_fwd(
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    stack_num: int = 1,
) -> torch.Tensor:
    # x: (s, input_dim)
    # weights: (num_lora, stack_num * r, input_dim)
    # output: (s, stack_num * r)
    # stack_num: run_qkv_lora: 3, run_gate_up_lora: 2
    # when called by run_qkv_lora, the weights.shape[-2] will be 3 * r
    # input_dim is much larger than r

    assert x.is_contiguous()
    assert weights.is_contiguous()
    assert len(x.shape) == 2
    assert len(weights.shape) == 3

    S = x.shape[0]
    R = weights.shape[-2]
    K = weights.shape[-1]
    assert x.shape[-1] == K

    # Block shapes
    BLOCK_S = 16
    BLOCK_K = 256
    BLOCK_R = 16

    grid = (
        triton.cdiv(batch_info.max_len, BLOCK_S) * triton.cdiv(R, BLOCK_R),
        batch_info.bs,
    )

    sorted_by_adapter = batch_info.permutation is not None

    output = torch.empty((S, R), device=x.device, dtype=x.dtype)
    _sgemm_lora_a_kernel[grid](
        x,
        weights,
        output,
        R,
        K,
        stack_num,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        batch_info.seg_lens,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        sorted_by_adapter,
        BLOCK_S,
        BLOCK_R,
        BLOCK_K,
    )
    return output

import torch
import triton


# Authored launcher: drives the in-module @triton.jit `_sgemm_lora_a_kernel`
# exactly as the upstream host wrapper `sgemm_lora_a_fwd` does (SGLang
# sglang/srt/lora/triton_ops/sgemm_lora_a.py, commit
# 409d350fb6f6a1e7c7546e39028f811092a8e489).
#
# Why a launcher rather than the host wrapper: the host entry
# `sgemm_lora_a_fwd(x, weights, batch_info, stack_num)` takes a
# `sglang.srt.lora.utils.LoRABatchInfo` dataclass whose import transitively
# pulls in the full sglang runtime (ForwardBatch, hf_transformers_utils,
# vllm._C, fastapi, prometheus_client, ...). That closure is T3
# BLOCKED-candidate (closure cycle, >800 symbols) and none of it is needed to
# exercise the unit under test: the kernel only reads the plain dense tensors
# (x, weights, seg_indptr, weight_indices, lora_ranks) plus a derived
# seg_lens/max_len. We reconstruct exactly the batch_info fields the wrapper
# forwards, straight from the dense oracle inputs.
#
# Upstream invocation reproduced here (sgemm_lora_a_fwd):
#   x: (S, K=input_dim) contiguous; weights: (num_lora, R=stack_num*r, K)
#   contiguous; output: (S, R). With stack_num=1, R == max_rank == G.
#   N = R, K = weights.shape[-1]; BLOCK_S=16, BLOCK_K=256, BLOCK_R=16.
#   grid = (cdiv(max_len, BLOCK_S) * cdiv(R, BLOCK_R), bs).
#   The wrapper passes batch_info.{seg_lens, seg_indptr, weight_indices,
#   lora_ranks, permutation} and sorted_by_adapter = permutation is not None.
#
# Layout reconciliation (dense fixtures -> upstream batch_info contract):
#   The oracle and the kernel define a segment's row span the SAME way when
#   the two segment descriptors are consistent: the wrapper drives row offsets
#   from seg_indptr[batch_id] (seg_start) with length seg_lens[batch_id]. The
#   real LoRABatchInfo always satisfies seg_indptr == cumsum([0, seg_lens]),
#   i.e. seg_lens[i] = seg_indptr[i+1] - seg_indptr[i]. The dense fixture only
#   supplies the authoritative seg_indptr (the oracle's boundary source:
#   out[seg_indptr[i]:seg_indptr[i+1]]), so we DERIVE seg_lens from it to
#   restore that invariant exactly. bs = num_segments = len(seg_indptr) - 1.
#   permutation is None (tokens contiguous, SORTED_BY_ADAPTER=False), so
#   `_resolve_token_positions` returns seg_start + s_offset -- the identity
#   row mapping the oracle uses (no adapter-sorted reordering).
#
# weight_indices[batch_id] selects the adapter (kernel: w_index); lora_ranks
# is indexed by that adapter id (kernel: rank = lora_ranks[w_index]); rows
# beyond `rank` stay zero -- identical to the oracle's out[s:e, :r] write.
# Output key matches the oracle contract: output, fp32 [S, R].
def run(inputs: dict) -> dict:
    x = inputs["x"]
    weights = inputs["weights"]
    seg_indptr = inputs["seg_indptr"]
    weight_indices = inputs["weight_indices"]
    lora_ranks = inputs["lora_ranks"]

    x = x.contiguous()
    weights = weights.contiguous()

    S, K = x.shape
    num_lora, R, Kw = weights.shape
    assert Kw == K, (Kw, K)

    stack_num = 1

    # Restore the upstream invariant seg_indptr == cumsum([0, seg_lens]).
    seg_indptr_l = seg_indptr.to(torch.int64)
    num_segments = seg_indptr_l.numel() - 1
    seg_lens = (seg_indptr_l[1:] - seg_indptr_l[:-1]).to(torch.int32).contiguous()
    bs = num_segments
    max_len = int(seg_lens.max().item()) if num_segments > 0 else 0

    weight_indices = weight_indices.to(torch.int64).contiguous()
    lora_ranks = lora_ranks.to(torch.int64).contiguous()
    seg_indptr_k = seg_indptr_l.contiguous()

    # permutation is None -> SORTED_BY_ADAPTER is False (contiguous tokens).
    permutation = None
    sorted_by_adapter = permutation is not None

    BLOCK_S = 16
    BLOCK_K = 256
    BLOCK_R = 16

    # Upstream wrapper allocates with torch.empty and the kernel writes only
    # output[seg_start:seg_start+seg_len, :rank] (the contract's defined
    # region; downstream LoRA forward reads only those columns). Undefined
    # cells (rows outside any segment, columns >= rank) are zero-filled here
    # to match the oracle's deterministic zero reference exactly -- the
    # benched region [seg, :rank] is computed by the kernel identically.
    output = torch.zeros((S, R), device=x.device, dtype=x.dtype)

    if max_len > 0 and bs > 0:
        grid = (
            triton.cdiv(max_len, BLOCK_S) * triton.cdiv(R, BLOCK_R),
            bs,
        )
        _sgemm_lora_a_kernel[grid](
            x,
            weights,
            output,
            R,
            K,
            stack_num,
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
            BLOCK_R,
            BLOCK_K,
        )

    return {"output": output}
