import torch

import triton


# Authored jit-only launcher: drives the in-module @triton.jit `pack_bitmatrix`
# exactly as the upstream host call site does
# (vllm gpt_oss_triton_kernels_moe.py::make_routing_data, commit 18f6bf5).
#
# The upstream wrapper `make_routing_data` cannot serve as the entrypoint: it
# constructs unvendorable triton_kernels `Bitmatrix` / `SparseMatrix` objects
# and returns routing metadata, not the raw bitmatrix the oracle contract
# wants. The faithful, vendorable path is to drive the bare @triton.jit
# directly, mirroring the upstream invocation:
#
#   topk_ids = topk_ids.to(torch.int16)
#   n_rows, num_topk = topk_ids.size()
#   BLOCK_SIZE_M = 512; BLOCK_SIZE_K = 32
#   bm_cols = triton.cdiv(num_local_experts, BLOCK_SIZE_K)
#   bitmatrix = torch.zeros((n_rows, bm_cols), dtype=torch.uint32, ...)
#   grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
#   pack_bitmatrix[grid](bitmatrix, topk_ids, n_rows, bm_cols, num_topk,
#                        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_K=BLOCK_SIZE_K)
#
# Faithfulness notes:
#   * num_local_experts (upstream) == inputs["num_experts"] here; bm_cols and
#     the per-row bit layout depend only on it.
#   * topk_ids is narrowed to int16 before launch, exactly as upstream; the
#     bit-packing reads the same expert ids (ids in [0, num_experts) fit int16).
#   * the kernel writes uint32 words; the oracle stores the identical 32-bit
#     pattern as int32. Returning bitmatrix.to(torch.int32) is a bit-pattern
#     preserving recast (e.g. 0x80000000 <-> -2147483648) so values match the
#     oracle's int32 contract; it does not alter the produced 32-bit words.
# Output key matches the oracle contract: "bitmatrix".
def run(inputs: dict) -> dict:
    topk_ids = inputs["topk_ids"]
    num_experts = int(inputs["num_experts"])

    # Mirror the upstream int16 narrowing of topk_ids prior to launch.
    topk_ids = topk_ids.to(torch.int16)
    n_rows, num_topk = topk_ids.shape

    BLOCK_SIZE_M = 512
    BLOCK_SIZE_K = 32
    bm_cols = triton.cdiv(num_experts, BLOCK_SIZE_K)
    bitmatrix = torch.zeros(
        (n_rows, bm_cols), dtype=torch.uint32, device=topk_ids.device
    )

    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
    pack_bitmatrix[grid](
        bitmatrix,
        topk_ids,
        n_rows,
        bm_cols,
        num_topk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return {"bitmatrix": bitmatrix.to(torch.int32)}
