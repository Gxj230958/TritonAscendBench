"""Task L1_05_gather_block_tables_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_gather_block_tables_kernel`
Source path: `vllm/v1/worker/gpu/block_table.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): f7662f59feb24dcb8622092809f1fb8ece7790b6443f553f6779df3acf01f5ee
"""
from collections.abc import Iterable
import torch
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
from ascendbench.frozen.shim import PAD_SLOT_ID  # Phase A': attention sentinel constants (B-inert-shim)
from ascendbench.frozen.shim import StagedWriteTensor, UvaBackedTensor  # Phase A': setup-only buffer helpers (B-inert-shim)

@triton.jit(do_not_specialize=['num_reqs'])
def _gather_block_tables_kernel(batch_idx_to_req_idx, src_block_table_ptrs, dst_block_table_ptrs, block_table_strides, num_blocks_ptr, num_blocks_stride, num_reqs, max_num_blocks, BLOCK_SIZE: tl.constexpr):
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    stride = tl.load(block_table_strides + group_id)
    dst_block_table_ptr = _load_ptr(dst_block_table_ptrs + group_id, tl.int32)
    dst_row_ptr = dst_block_table_ptr + batch_idx * stride
    if batch_idx >= num_reqs:
        for i in tl.range(0, max_num_blocks, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            tl.store(dst_row_ptr + offset, 0, mask=offset < max_num_blocks)
        return
    req_idx = tl.load(batch_idx_to_req_idx + batch_idx)
    group_num_blocks_ptr = num_blocks_ptr + group_id * num_blocks_stride
    num_blocks = tl.load(group_num_blocks_ptr + req_idx)
    src_block_table_ptr = _load_ptr(src_block_table_ptrs + group_id, tl.int32)
    src_row_ptr = src_block_table_ptr + req_idx * stride
    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        block_ids = tl.load(src_row_ptr + offset, mask=offset < num_blocks)
        tl.store(dst_row_ptr + offset, block_ids, mask=offset < num_blocks)

@triton.jit
def _load_ptr(ptr_to_ptr, elem_dtype):
    ptr = tl.load(ptr_to_ptr)
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
    return tl.multiple_of(ptr, 16)

import torch
import triton


# Authored launcher: drives the in-module @triton.jit
# `_gather_block_tables_kernel` exactly as the upstream host method
# `BlockTables.gather_block_tables` does (vLLM
# vllm/v1/worker/gpu/block_table.py, commit
# 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19).
#
# Why a launcher rather than the host method: the real entry
# `BlockTables.gather_block_tables(idx_mapping, num_reqs_padded)` reads
# kernel-launch state off a `BlockTables` instance whose __init__ builds
# `StagedWriteTensor` / `UvaBackedTensor` UVA buffers and raw `data_ptr()`
# pointer tensors. None of that bookkeeping is the unit under test: the
# kernel is a pure per-row block-table gather. We reconstruct exactly the
# flat tensors the host method forwards, straight from the dense oracle
# inputs.
#
# Upstream invocation reproduced here (gather_block_tables, single kv-cache
# group):
#   _gather_block_tables_kernel[(num_kv_cache_groups, num_reqs_padded)](
#       idx_mapping,                       # batch_idx_to_req_idx [B]
#       block_table_ptrs,                  # src ptr tensor       [G]
#       input_block_table_ptrs,            # dst ptr tensor       [G]
#       block_table_strides,               # row stride (elems)   [G]
#       num_blocks.gpu,                    # [G, max_num_reqs]
#       num_blocks.gpu.stride(0),
#       num_reqs,                          # actual req count
#       max_num_blocks,                    # dst row width
#       BLOCK_SIZE=1024)
#
# Layout reconciliation (dense fixtures -> upstream kernel contract):
#   The oracle is the full per-row gather output = block_table[idx_mapping]
#   (every one of the N columns copied). The kernel copies exactly
#   num_blocks[req_idx] leading entries of each selected source row into the
#   destination row, leaving the rest at the dst's pre-launch contents. The
#   upstream block tables are torch.int32 with row stride = max_num_blocks
#   (== N here) and the dst tables are zero-initialised. To reproduce the
#   FULL-row gather the oracle defines we set num_blocks[req] = N for every
#   request (the maximal, complete-copy case that exercises the entire
#   per-row loop) and num_reqs_padded = num_reqs = B (no padded rows). With
#   a zero-initialised dst, dst[b, :] = src[idx_mapping[b], :N] exactly, i.e.
#   block_table[idx_mapping] -- the oracle op, byte for byte.
#
#   dtype: the kernel reads/writes the tables through int32 pointers
#   (`_load_ptr(..., tl.int32)`), so source/dst are materialised as int32
#   (upstream block tables are int32; block ids lie in [0, N) so int64->int32
#   is lossless). The gathered output is cast back to int64 to match the
#   oracle output dtype.
def run(inputs: dict) -> dict:
    block_table = inputs["block_table"]   # [E, N] int64
    idx_mapping = inputs["idx_mapping"]   # [B]   int64, each in [0, E)

    device = block_table.device
    E, N = int(block_table.shape[0]), int(block_table.shape[1])
    B = int(idx_mapping.shape[0])

    # Single kv-cache group. Source table is the upstream-native int32 layout
    # (contiguous, row stride == N). Block ids are in [0, N): int64->int32 is
    # value-preserving.
    src = block_table.to(torch.int32).contiguous()
    # Destination is zero-initialised [B, N] int32, exactly like the upstream
    # input_block_tables row span the kernel writes into.
    dst = torch.zeros((B, N), dtype=torch.int32, device=device)

    # idx_mapping = batch_idx_to_req_idx [B] (int32, the kernel loads it raw).
    batch_idx_to_req_idx = idx_mapping.to(torch.int32).contiguous()

    # num_blocks[group, req] = N for every request -> copy the whole row.
    num_blocks = torch.full((1, E), N, dtype=torch.int32, device=device)

    # Raw data_ptr() pointer tensors (uint64), one entry per group.
    src_block_table_ptrs = torch.tensor(
        [src.data_ptr()], dtype=torch.uint64, device=device
    )
    dst_block_table_ptrs = torch.tensor(
        [dst.data_ptr()], dtype=torch.uint64, device=device
    )
    # Row stride in ELEMENTS (matches block_table.stride(0) of an [_, N] int32
    # table).
    block_table_strides = torch.tensor(
        [src.stride(0)], dtype=torch.int64, device=device
    )

    num_kv_cache_groups = 1
    num_reqs_padded = B            # no padded rows
    num_reqs = B                   # all rows are real requests

    # Import the in-module jit kernel (exec'd into this namespace by the
    # loader). Reference it by name; both S2 sides share the same symbol.
    _gather_block_tables_kernel[(num_kv_cache_groups, num_reqs_padded)](
        batch_idx_to_req_idx,
        src_block_table_ptrs,
        dst_block_table_ptrs,
        block_table_strides,
        num_blocks,
        num_blocks.stride(0),
        num_reqs,
        N,                         # max_num_blocks == dst row width
        BLOCK_SIZE=1024,
    )

    # Cast back to the oracle's int64 output dtype (lossless; values in [0, N)).
    return {"output": dst.to(torch.int64)}
