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
