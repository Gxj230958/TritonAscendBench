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
