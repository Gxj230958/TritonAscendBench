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
# inputs. (Same pattern as the passed siblings L3_03_chunked_lora_expand /
# L3_04_qkv_lora_b.)
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
