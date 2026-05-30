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
