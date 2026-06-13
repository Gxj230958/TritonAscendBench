"""Task L2_12_pack_bitmatrix — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `pack_bitmatrix`
Source path: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): 1a839b9706673f8a54e7d21818952a190a853cf7021a6a320d955ce66a927ed5
"""
import torch
import ascendbench.frozen.shim.modular_kernel as mk  # Phase A': off-path modular_kernel annotation shim (B-inert-shim)
from ascendbench.frozen.shim import ops as ops  # Phase A': off-path custom-ops shim (B-inert-shim)
from ascendbench.frozen.shim import init_logger  # Phase A': inert no-op sink (B-inert-shim)
from ascendbench.frozen.shim import MoEActivation  # Phase A': off-path activation enum (B-inert-shim)
from ascendbench.frozen.shim import FUSED_MOE_UNQUANTIZED_CONFIG, FusedMoEConfig, FusedMoEParallelConfig, FusedMoEQuantConfig, RoutingMethodType  # Phase A': off-path fused_moe config (B-inert-shim)
from ascendbench.frozen.shim import LoRAExpertsMixin  # Phase A': off-path mixin base (B-inert-shim)
from ascendbench.frozen.shim import TopKWeightAndReduceNoOP  # Phase A': off-path reduce helper (B-inert-shim)
from ascendbench.frozen.shim import _resize_cache  # Phase A': off-path workspace reshape (B-inert-shim)
from ascendbench.frozen.shim import QuantKey, kMxfp4Static  # Phase A': quant symbols, symbol-aware (B-inert-shim)
from ascendbench.frozen.shim import current_platform  # Phase A': platform stub (B-inert-shim)
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
from ascendbench.frozen.shim import has_triton_kernels  # Phase A': optional-package predicate (B-inert-shim)
from ascendbench.frozen.shim import swiglu_limit_func  # Phase A': off-path swiglu helper (B-inert-shim)


@triton.jit
def pack_bitmatrix(bitmatrix, topk_ids, n_rows, bm_cols: tl.constexpr, n_expts_act, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    """
    Packs topk_ids into a bitmatrix.
    code reference:
    https://github.com/triton-lang/triton/blob/dd1bbc52b34d202dfe5ffea1e04fb16166c5c04e/python/triton_kernels/bench/distributed.py#L264
    """
    pid_m = tl.program_id(0)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    offsets = offsets_m[:, None] * n_expts_act + offsets_k[None, :]
    mask = (offsets_m < n_rows)[:, None] & (offsets_k < n_expts_act)[None, :]
    indices = tl.load(topk_ids + offsets, mask=mask, other=-1)
    valid = indices >= 0
    div = indices // 32
    rem = indices % 32
    one = tl.cast(1, tl.uint32)
    for i in range(bm_cols):
        offs = tl.arange(0, BLOCK_SIZE_K // 32) + i * (BLOCK_SIZE_K // 32)
        x = tl.where(valid[:, :, None] & (div[:, :, None] == offs[None, None, :]), (one << rem)[:, :, None], 0)
        y = tl.reduce_or(x, axis=1)
        bitmatrix_ptrs = bitmatrix + offsets_m[:, None] * bm_cols + offs[None, :]
        tl.store(bitmatrix_ptrs, y, mask=offsets_m[:, None] < n_rows)

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
