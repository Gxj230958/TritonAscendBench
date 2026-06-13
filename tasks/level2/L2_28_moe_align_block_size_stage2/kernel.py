"""Task L2_28_moe_align_block_size_stage2 — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `moe_align_block_size_stage2`
Source path: `sglang/jit_kernel/tests/test_moe_align_block_size.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): 06564b45a3dfa6be2e2ca246182b3ba6a75c5c265e1c5a1bb2582e254424a38a
"""

import itertools
import sys

import pytest
import torch
import triton
import triton.language as tl

from ascendbench.frozen.shim import moe_align_block_size  # Phase A': sibling-only CUDA reference (B-inert-shim)
from ascendbench.frozen.shim import register_cuda_ci  # Phase A': CI-registration no-op (B-inert-shim)

register_cuda_ci(est_time=28, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)




@triton.jit
def moe_align_block_size_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts

    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)
    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    last_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last_cumsum)
    tl.store(total_tokens_post_pad_ptr, last_cumsum)


@triton.jit
def moe_align_block_size_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)









import torch


# Authored launcher: drives the in-module 4-stage @triton.jit pipeline
# exactly as the upstream host wrapper `moe_align_block_size_triton` does
# (SGLang sglang/jit_kernel/tests/test_moe_align_block_size.py,
# commit 409d350fb6f6a1e7c7546e39028f811092a8e489 -- the same file frozen
# into kernel.py).
#
# Upstream host wrapper (kernel.py, verbatim launch sequence):
#     numel = topk_ids.numel()
#     grid = (num_experts,)
#     tokens_cnts = torch.zeros((num_experts + 1, num_experts), int32, dev)
#     cumsum = torch.zeros((num_experts + 1,), int32, dev)
#     tokens_per_thread = ceil_div(numel, num_experts)
#     moe_align_block_size_stage1[grid](topk_ids, tokens_cnts, num_experts,
#                                       numel, tokens_per_thread)
#     moe_align_block_size_stage2[grid](tokens_cnts, num_experts)
#     moe_align_block_size_stage3[(1,)](num_tokens_post_pad, tokens_cnts,
#                                       cumsum, num_experts, block_size)
#     moe_align_block_size_stage4[grid](topk_ids, sorted_token_ids,
#                                       expert_ids, tokens_cnts, cumsum,
#                                       num_experts, block_size, numel,
#                                       tokens_per_thread)
#
# The output buffers `sorted_token_ids` / `expert_ids` / `num_tokens_post_pad`
# are caller-allocated. The upstream test allocates them OVER-SIZED to the
# worst-case `max_num_tokens_padded` and then compares only the valid prefix
# (`sorted_ids[:num_tokens_post_pad]`, `expert_ids[:num_tokens_post_pad //
# block_size]`). The padded sentinel for `sorted_token_ids` is `numel`
# (upstream: `sorted_ids.fill_(topk_ids.numel())`).
#
# Output keys + the trimmed valid region match the oracle contract
# (oracle_pytorch.py::run) exactly:
#   sorted_token_ids   : [num_tokens_post_pad] int32, padding slots = numel
#   expert_ids         : [num_tokens_post_pad // block_size] int32
#   num_tokens_post_pad: [1] int32
def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

import torch


# Authored launcher: drives the in-module 4-stage @triton.jit pipeline
# exactly as the upstream host wrapper `moe_align_block_size_triton` does
# (SGLang sglang/jit_kernel/tests/test_moe_align_block_size.py,
# commit 409d350fb6f6a1e7c7546e39028f811092a8e489 -- the same file frozen
# into kernel.py).
#
# Upstream host wrapper (kernel.py, verbatim launch sequence):
#     numel = topk_ids.numel()
#     grid = (num_experts,)
#     tokens_cnts = torch.zeros((num_experts + 1, num_experts), int32, dev)
#     cumsum = torch.zeros((num_experts + 1,), int32, dev)
#     tokens_per_thread = ceil_div(numel, num_experts)
#     moe_align_block_size_stage1[grid](topk_ids, tokens_cnts, num_experts,
#                                       numel, tokens_per_thread)
#     moe_align_block_size_stage2[grid](tokens_cnts, num_experts)
#     moe_align_block_size_stage3[(1,)](num_tokens_post_pad, tokens_cnts,
#                                       cumsum, num_experts, block_size)
#     moe_align_block_size_stage4[grid](topk_ids, sorted_token_ids,
#                                       expert_ids, tokens_cnts, cumsum,
#                                       num_experts, block_size, numel,
#                                       tokens_per_thread)
#
# The output buffers `sorted_token_ids` / `expert_ids` / `num_tokens_post_pad`
# are caller-allocated. The upstream test allocates them OVER-SIZED to the
# worst-case `max_num_tokens_padded` and then compares only the valid prefix
# (`sorted_ids[:num_tokens_post_pad]`, `expert_ids[:num_tokens_post_pad //
# block_size]`). The padded sentinel for `sorted_token_ids` is `numel`
# (upstream: `sorted_ids.fill_(topk_ids.numel())`).
#
# Output keys + the trimmed valid region match the oracle contract
# (oracle_pytorch.py::run) exactly:
#   sorted_token_ids   : [num_tokens_post_pad] int32, padding slots = numel
#   expert_ids         : [num_tokens_post_pad // block_size] int32
#   num_tokens_post_pad: [1] int32
def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def run(inputs: dict) -> dict:
    topk_ids = inputs["topk_ids"]
    num_experts = int(inputs["num_experts"])
    block_size = int(inputs["block_size"])

    # The kernel loads topk_ids as a flat integer pointer; keep it integral
    # and contiguous (upstream passes the [num_tokens, topk] tensor directly,
    # which is contiguous).
    if topk_ids.dtype not in (torch.int32, torch.int64):
        topk_ids = topk_ids.to(torch.int64)
    topk_ids = topk_ids.contiguous()
    device = topk_ids.device

    numel = topk_ids.numel()
    grid = (num_experts,)

    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts), dtype=torch.int32, device=device
    )
    cumsum = torch.zeros(
        (num_experts + 1,), dtype=torch.int32, device=device
    )
    tokens_per_thread = _ceil_div(numel, num_experts)

    # Worst-case caller allocation, mirroring the upstream test driver.
    max_num_tokens_padded = numel + (num_experts + 1) * (block_size - 1)
    if numel < num_experts + 1:
        max_num_tokens_padded = numel * block_size
    max_num_m_blocks = max_num_tokens_padded // block_size

    sorted_token_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=device
    )
    sorted_token_ids.fill_(numel)  # sentinel for unwritten padding slots
    expert_ids = torch.zeros(
        (max_num_m_blocks,), dtype=torch.int32, device=device
    )
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)

    moe_align_block_size_stage1[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
    )
    moe_align_block_size_stage2[grid](
        tokens_cnts,
        num_experts,
    )
    moe_align_block_size_stage3[(1,)](
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
    )
    moe_align_block_size_stage4[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
    )

    # Trim the over-allocated buffers to the logical region the oracle returns.
    npp = int(num_tokens_post_pad.item())
    n_blocks = npp // block_size
    return {
        "sorted_token_ids": sorted_token_ids[:npp],
        "expert_ids": expert_ids[:n_blocks],
        "num_tokens_post_pad": num_tokens_post_pad,
    }
