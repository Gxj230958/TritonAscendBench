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
