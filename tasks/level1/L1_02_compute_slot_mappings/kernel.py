"""Task L1_02_compute_slot_mappings_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_compute_slot_mappings_kernel`
Source path: `vllm/v1/worker/gpu/block_table.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): f7662f59feb24dcb8622092809f1fb8ece7790b6443f553f6779df3acf01f5ee
"""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import torch

import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
from ascendbench.frozen.shim import PAD_SLOT_ID  # Phase A': attention sentinel constants (B-inert-shim)
from ascendbench.frozen.shim import StagedWriteTensor, UvaBackedTensor  # Phase A': setup-only buffer helpers (B-inert-shim)


class BlockTables:
    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        max_num_blocks_per_group: list[int],
        device: torch.device,
        cp_size: int = 1,
        cp_rank: int = 0,
        cp_interleave: int = 1,
    ):
        self.block_sizes = block_sizes
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device

        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.cp_interleave = cp_interleave

        self.num_kv_cache_groups = len(self.block_sizes)
        assert len(max_num_blocks_per_group) == self.num_kv_cache_groups
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.block_tables: list[StagedWriteTensor] = []
        for i in range(self.num_kv_cache_groups):
            max_num_blocks = max_num_blocks_per_group[i]
            block_table = StagedWriteTensor(
                (self.max_num_reqs, max_num_blocks),
                dtype=torch.int32,
                device=device,
            )
            self.block_tables.append(block_table)

        self.num_blocks = UvaBackedTensor(
            (self.num_kv_cache_groups, self.max_num_reqs),
            dtype=torch.int32,
        )

        # Block tables used for model's forward pass.
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.input_block_tables: list[torch.Tensor] = [
            torch.zeros_like(b.gpu) for b in self.block_tables
        ]

        self.slot_mappings = torch.zeros(
            self.num_kv_cache_groups,
            self.max_num_batched_tokens,
            dtype=torch.int64,
            device=self.device,
        )

        self.init_block_table_layout_tensors()

    def _make_ptr_tensor(self, x: Iterable[torch.Tensor]) -> torch.Tensor:
        # NOTE(woosuk): Use uint64 instead of int64 to cover all possible addresses.
        return torch.tensor(
            [t.data_ptr() for t in x], dtype=torch.uint64, device=self.device
        )

    def init_block_table_layout_tensors(self) -> None:
        # Called at init and after a CuMem kv_cache wake-up. The ptr tensors
        # cache raw data_ptr() values that go stale once the underlying tensors
        # are reallocated on wake; block_sizes_tensor needs re-populating
        # because its storage lives under the kv_cache pool tag and comes back
        # with undefined contents.
        self.block_table_ptrs = self._make_ptr_tensor(
            [b.gpu for b in self.block_tables]
        )
        self.block_table_strides = torch.tensor(
            [b.gpu.stride(0) for b in self.block_tables],
            dtype=torch.int64,
            device=self.device,
        )
        self.block_sizes_tensor = torch.tensor(
            self.block_sizes, dtype=torch.int32, device=self.device
        )
        self.input_block_table_ptrs = self._make_ptr_tensor(self.input_block_tables)

    def append_block_ids(
        self,
        req_index: int,
        new_block_ids: tuple[list[int], ...],
        overwrite: bool,
    ) -> None:
        for i in range(self.num_kv_cache_groups):
            start = self.num_blocks.np[i, req_index] if not overwrite else 0
            block_ids = new_block_ids[i]
            self.block_tables[i].stage_write(req_index, start, block_ids)
            self.num_blocks.np[i, req_index] = start + len(block_ids)

    def apply_staged_writes(self) -> None:
        # TODO(woosuk): This can be inefficient since it launches one kernel per
        # block table. Implement a kernel to handle all block tables at once.
        for block_table in self.block_tables:
            block_table.apply_write()
        self.num_blocks.copy_to_uva()

    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,
        num_reqs_padded: int,
    ) -> tuple[torch.Tensor, ...]:
        num_reqs = idx_mapping.shape[0]
        # Launch kernel with num_reqs_padded to fuse zeroing of padded rows.
        _gather_block_tables_kernel[(self.num_kv_cache_groups, num_reqs_padded)](
            idx_mapping,
            self.block_table_ptrs,
            self.input_block_table_ptrs,
            self.block_table_strides,
            self.num_blocks.gpu,
            self.num_blocks.gpu.stride(0),
            num_reqs,
            self.input_block_tables[0].shape[1],  # max_num_blocks
            BLOCK_SIZE=1024,  # type: ignore
        )
        return tuple(bt[:num_reqs_padded] for bt in self.input_block_tables)

    def get_dummy_block_tables(self, num_reqs: int) -> tuple[torch.Tensor, ...]:
        # NOTE(woosuk): The output may be used for CUDA graph capture.
        # Therefore, this method must return the persistent tensor
        # with the same memory address as that used during the model's forward pass,
        # rather than allocating a new tensor.
        return tuple(block_table[:num_reqs] for block_table in self.input_block_tables)

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
    ) -> torch.Tensor:
        num_reqs = idx_mapping.shape[0]
        num_groups = self.num_kv_cache_groups
        _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
            self.max_num_batched_tokens,
            idx_mapping,
            query_start_loc,
            positions,
            self.block_table_ptrs,
            self.block_table_strides,
            self.block_sizes_tensor,
            self.slot_mappings,
            self.slot_mappings.stride(0),
            self.cp_rank,
            CP_SIZE=self.cp_size,
            CP_INTERLEAVE=self.cp_interleave,
            PAD_ID=PAD_SLOT_ID,
            TRITON_BLOCK_SIZE=1024,  # type: ignore
        )
        return self.slot_mappings[:, :num_tokens_padded]

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        # Fill the entire slot_mappings tensor, not just the first `num_tokens` entries.
        # This is because the padding logic is complex and kernels may access beyond
        # the requested range.
        self.slot_mappings.fill_(PAD_SLOT_ID)
        # NOTE(woosuk): The output may be used for CUDA graph capture.
        # Therefore, this method must return the persistent tensor
        # with the same memory address as that used during the model's forward pass,
        # rather than allocating a new tensor.
        return self.slot_mappings[:, :num_tokens]


@triton.jit(do_not_specialize=["num_reqs"])
def _gather_block_tables_kernel(
    batch_idx_to_req_idx,  # [batch_size]
    src_block_table_ptrs,  # [num_kv_cache_groups]
    dst_block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    num_blocks_ptr,  # [num_kv_cache_groups, max_num_reqs]
    num_blocks_stride,
    num_reqs,  # actual number of requests (for padding)
    max_num_blocks,  # stride for zeroing padded rows
    BLOCK_SIZE: tl.constexpr,
):
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)

    stride = tl.load(block_table_strides + group_id)
    dst_block_table_ptr = _load_ptr(dst_block_table_ptrs + group_id, tl.int32)
    dst_row_ptr = dst_block_table_ptr + batch_idx * stride

    if batch_idx >= num_reqs:
        # Zero out padded rows.
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
def _compute_slot_mappings_kernel(
    max_num_tokens,
    idx_mapping,  # [num_reqs]
    query_start_loc,  # [num_reqs + 1]
    pos,  # [num_tokens]
    block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    block_sizes,  # [num_kv_cache_groups]
    slot_mappings_ptr,  # [num_kv_cache_groups, max_num_tokens]
    slot_mappings_stride,
    cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride

    if batch_idx == tl.num_programs(1) - 1:
        # Pad remaining slots to -1. This is needed for CUDA graphs.
        # Start from actual token count (not padded) to cover the gap
        # between actual tokens and padded tokens that can contain stale
        # valid slot IDs from previous chunks during chunked prefill.
        actual_num_tokens = tl.load(query_start_loc + batch_idx)
        for i in range(actual_num_tokens, max_num_tokens, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens)
        return

    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    block_size = tl.load(block_sizes + group_id)

    req_state_idx = tl.load(idx_mapping + batch_idx)
    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)
    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)

        block_indices = positions // (block_size * CP_SIZE)
        block_offsets = positions % (block_size * CP_SIZE)
        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices
        )

        if CP_SIZE == 1:
            # Common case: Context parallelism is not used.
            slot_ids = block_numbers * block_size + block_offsets
        else:
            # Context parallelism is used.
            is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
            rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)
            remainder = block_offsets % CP_INTERLEAVE
            local_offsets = rounds * CP_INTERLEAVE + remainder
            slot_ids = block_numbers * block_size + local_offsets
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)

        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)


@triton.jit
def _load_ptr(ptr_to_ptr, elem_dtype):
    ptr = tl.load(ptr_to_ptr)
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
    return tl.multiple_of(ptr, 16)

import torch


# Authored launcher: drives the graded @triton.jit `_compute_slot_mappings_kernel`
# (vLLM v1 paged KV slot mapping, vllm/v1/worker/gpu/block_table.py, commit
# 18f6bf5) directly, mirroring the in-module host method
# `BlockTables.compute_slot_mappings` (kernel.py lines 151-176) EXACTLY.
#
# Why drive the @triton.jit directly instead of calling the host method:
#   `BlockTables.__init__` constructs `StagedWriteTensor` / `UvaBackedTensor`
#   (vllm.v1.worker.gpu.buffer_utils) and reaches into the live vllm runtime
#   to build the pointer-table machinery. Those buffer helpers are the
#   `ascendbench.frozen.shim` redirect target (B-inert-shim), not real allocators,
#   so the host method cannot be invoked in the severed grading environment.
#   The only faithful way to grade the kernel is to reproduce the launch:
#   build the exact pointer / stride / segment tensors the method forwards,
#   straight from the dense oracle inputs, and call the kernel on the same
#   grid `(num_groups, num_reqs + 1)`.
#
# Input/layout reconciliation (the crux):
#   The oracle treats the operation as PER-TOKEN over B tokens:
#     slot[i] = block_table[idx_mapping[i], positions[i] // bs] * bs
#               + positions[i] % bs        for i in [0, B)
#   The kernel instead is PER-REQUEST: grid axis-1 (`batch_idx`) indexes a
#   request, and request `batch_idx` processes the token span
#   [query_start_loc[batch_idx], query_start_loc[batch_idx + 1]). For each
#   token `i` in that span it computes (with CP_SIZE == 1, the common case):
#     req_state_idx = idx_mapping[batch_idx]
#     block_numbers = block_table[req_state_idx, positions[i] // bs]
#     slot_ids      = block_numbers * bs + positions[i] % bs
#   To make the kernel reproduce the oracle's per-token result EXACTLY we map
#   one request to one token: num_reqs = B and
#   query_start_loc = arange(0, B + 1), so request `i` owns exactly token `i`.
#   Then idx_mapping[i] (per request) == the oracle's idx_mapping[i] (per
#   token), positions[i] is read at the same index, and the kernel's per-token
#   arithmetic is bit-identical to the oracle's. Single KV-cache group
#   (num_groups = 1) since the oracle takes a single [E, N] block_table.
#
# Pointer-table machinery (mirrors `init_block_table_layout_tensors`):
#   The kernel reads the block table through a uint64 ptr indirection:
#   `block_table_ptrs[g]` holds block_table[g].data_ptr(); `_load_ptr` casts it
#   to an int32* and the kernel indexes it as a flat row-major buffer with
#   `block_table_stride = block_table.stride(0)`. The upstream block tables are
#   int32, so we materialise an int32, contiguous copy of the [E, N] table and
#   register its data_ptr (values are KV block ids in [0, N), losslessly
#   representable in int32). `slot_mappings_ptr` is the int64 output buffer of
#   shape [num_groups, max_num_tokens]; the kernel's last program_id pads
#   unused slots to PAD_ID, so we allocate the full [1, max_num_tokens] buffer
#   and slice the first B entries for the [B] int64 oracle output.
#
# CP disabled (CP_SIZE = 1, cp_rank = 0, CP_INTERLEAVE = 1) -- matches the
# oracle, which has no context-parallel term. PAD_ID = PAD_SLOT_ID (the shim
# constant, -1); TRITON_BLOCK_SIZE = 1024 (the upstream launch literal).
#
# Output key: "slot_mapping" -- the oracle's [B] int64 per-token slot indices.
def run(inputs: dict) -> dict:
    idx_mapping_in = inputs["idx_mapping"]          # [B] int64, request id in [0, E)
    positions_in = inputs["positions"]              # [B] int64, token positions
    block_table_in = inputs["block_table"]          # [E, N] int64 block ids in [0, N)
    block_size = int(inputs["block_size"])
    max_num_tokens = int(inputs["max_num_tokens"])

    device = idx_mapping_in.device
    B = idx_mapping_in.shape[0]
    num_groups = 1

    # PAD_SLOT_ID is the upstream padding sentinel; the kernel.py rewrite pulls
    # it from ascendbench.frozen.shim. Read it from the executing kernel module so
    # the launcher and kernel agree on the same constant.
    import sys
    _kmod = sys.modules[run.__module__]
    PAD_SLOT_ID = getattr(_kmod, "PAD_SLOT_ID", -1)

    # One request per token: each request owns exactly its own token, so the
    # kernel's per-request span [qsl[i], qsl[i+1]) == {i} and its arithmetic
    # matches the oracle's per-token formula.
    idx_mapping = idx_mapping_in.to(torch.int64).contiguous()
    query_start_loc = torch.arange(0, B + 1, device=device, dtype=torch.int64)
    positions = positions_in.to(torch.int64).contiguous()

    # Upstream block tables are int32; the ptr indirection casts to int32*.
    # block ids are in [0, N), losslessly held in int32.
    block_table_i32 = block_table_in.to(torch.int32).contiguous()
    block_table_ptrs = torch.tensor(
        [block_table_i32.data_ptr()], dtype=torch.uint64, device=device
    )
    block_table_strides = torch.tensor(
        [block_table_i32.stride(0)], dtype=torch.int64, device=device
    )
    block_sizes_tensor = torch.tensor(
        [block_size], dtype=torch.int32, device=device
    )

    slot_mappings = torch.zeros(
        num_groups, max_num_tokens, dtype=torch.int64, device=device
    )

    _kmod._compute_slot_mappings_kernel[(num_groups, B + 1)](
        max_num_tokens,
        idx_mapping,
        query_start_loc,
        positions,
        block_table_ptrs,
        block_table_strides,
        block_sizes_tensor,
        slot_mappings,
        slot_mappings.stride(0),
        0,                       # cp_rank
        CP_SIZE=1,
        CP_INTERLEAVE=1,
        PAD_ID=PAD_SLOT_ID,
        TRITON_BLOCK_SIZE=1024,
    )

    # The oracle output is the per-token slot mapping of length B (int64).
    return {"slot_mapping": slot_mappings[0, :B].contiguous()}
