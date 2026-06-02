# Phase A': sliced to the graded entrypoint's module-level closure
# (whole-module vendored kernel; dead upstream siblings + their
# upstream imports dropped -- the kept defs are byte-identical).
import torch
import triton
import triton.language as tl


@triton.jit
def _copy_page_indices_kernel(page_indices, block_table, block_table_stride, cu_num_blocks, BLOCK_SIZE: tl.constexpr):
    req_idx = tl.program_id(0)
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_blocks + req_idx)
    end_idx = tl.load(cu_num_blocks + req_idx + 1)
    num_blocks = end_idx - start_idx
    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        block_ids = tl.load(row_ptr + i + offset, mask=i + offset < num_blocks)
        tl.store(page_indices + start_idx + i + offset, block_ids, mask=i + offset < num_blocks)

import torch


# Authored jit-only launcher: drives the in-module @triton.jit
# `_copy_page_indices_kernel` exactly as the upstream host call site does
# (FlashInferMetadataBuilder._compute_flashinfer_kv_metadata, vLLM
# vllm/v1/attention/backends/flashinfer.py, commit 18f6bf5).
#
# Upstream contract reproduced here:
#   * num_blocks_per_req[r] = ceil(seq_lens[r] / page_size)
#       (build(): num_blocks_np = (seq_lens_np + (page_size - 1)) // page_size)
#   * cu_num_blocks = [0, cumsum(num_blocks_per_req)], int32, length num_reqs+1
#       (paged_kv_indptr; np.cumsum(..., dtype=np.int32) into [1:num_reqs+1],
#        0-index is always 0)
#   * page_indices = flat int32 output of length cu_num_blocks[-1]
#       (paged_kv_indices buffer is int32; sliced to num_actual_pages)
#   * block_table_stride = block_table.stride(0)
#   * grid = (num_reqs,), BLOCK_SIZE = 1024
#
# The kernel copies block_table[r, 0:num_blocks_per_req[r]] into the flat
# page_indices vector; storing int64 block ids into an int32 buffer truncates
# to int32, matching the oracle's `.to(torch.int32)`. Output key: kv_indices.
def run(inputs: dict) -> dict:
    block_table = inputs["block_table"]
    seq_lens = inputs["seq_lens"]
    page_size = int(inputs["page_size"])

    num_reqs = seq_lens.shape[0]
    device = block_table.device

    # ceil(seq_lens / page_size) per request (upstream build()).
    num_blocks_per_req = (seq_lens.to(torch.int64) + (page_size - 1)) // page_size

    # cu_num_blocks: int32 indptr, [0, cumsum(...)], length num_reqs + 1.
    cu_num_blocks = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    cu_num_blocks[1:] = torch.cumsum(num_blocks_per_req, dim=0).to(torch.int32)

    num_actual_pages = int(cu_num_blocks[num_reqs].item())
    page_indices = torch.empty(num_actual_pages, dtype=torch.int32, device=device)

    _copy_page_indices_kernel[(num_reqs,)](
        page_indices,
        block_table,
        block_table.stride(0),
        cu_num_blocks,
        BLOCK_SIZE=1024,
    )
    return {"kv_indices": page_indices}
