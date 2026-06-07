"""Task L2_12_pack_bitmatrix — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `pack_bitmatrix`
Source path: `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): 1a839b9706673f8a54e7d21818952a190a853cf7021a6a320d955ce66a927ed5
Frozen by: scripts/phase_a_freeze.py
"""
import torch
import tasks._frozen_shim.modular_kernel as mk  # Phase A': off-path modular_kernel annotation shim (B-inert-shim)
from tasks._frozen_shim import ops as ops  # Phase A': off-path custom-ops shim (B-inert-shim)
from tasks._frozen_shim import init_logger  # Phase A': inert no-op sink (B-inert-shim)
from tasks._frozen_shim import MoEActivation  # Phase A': off-path activation enum (B-inert-shim)
from tasks._frozen_shim import FUSED_MOE_UNQUANTIZED_CONFIG, FusedMoEConfig, FusedMoEParallelConfig, FusedMoEQuantConfig, RoutingMethodType  # Phase A': off-path fused_moe config (B-inert-shim)
from tasks._frozen_shim import LoRAExpertsMixin  # Phase A': off-path mixin base (B-inert-shim)
from tasks._frozen_shim import TopKWeightAndReduceNoOP  # Phase A': off-path reduce helper (B-inert-shim)
from tasks._frozen_shim import _resize_cache  # Phase A': off-path workspace reshape (B-inert-shim)
from tasks._frozen_shim import QuantKey, kMxfp4Static  # Phase A': quant symbols, symbol-aware (B-inert-shim)
from tasks._frozen_shim import current_platform  # Phase A': platform stub (B-inert-shim)
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
from tasks._frozen_shim import has_triton_kernels  # Phase A': optional-package predicate (B-inert-shim)
from tasks._frozen_shim import swiglu_limit_func  # Phase A': off-path swiglu helper (B-inert-shim)
logger = init_logger(__name__)

def _patch_make_bitmatrix_metadata() -> None:
    """Monkey-patch make_bitmatrix_metadata to support non-power-of-2 top_k.

    triton's tl.arange requires a power-of-2 range.  The original kernel
    computes BLOCK_SIZE = BLOCK_PER_TOK * TOKS_PER_ROW (= 32 * top_k).  For
    DeepSeek-V4 with top_k=6 this gives 192, which is not a power of 2 and
    causes a compile error at the first forward pass.

    Fix: define a drop-in replacement kernel that accepts an extra constexpr
    BLOCK_SIZE_PADDED (next power of 2 >= BLOCK_SIZE) and uses it for the
    tl.arange call while keeping the actual BLOCK_SIZE as the stride between
    thread-blocks so that all flat indices into NonzeroIndx stay correct.
    Elements beyond BLOCK_SIZE are masked out (col_indx = 0xffff) and ignored.

    This function is called once at module load time and patches the function
    inside the triton_kernels tensor module so that SparseMatrix.__post_init__
    picks up the fixed version transparently.
    """
    import torch
    import triton
    import triton.language as tl
    try:
        from vllm.third_party.triton_kernels.tensor_details import bitmatrix as _bm
        from vllm.third_party.triton_kernels.tensor_details.bitmatrix import BitmatrixMetadata, _keyed_add, cdiv
        from vllm.third_party.triton_kernels.tensor_details.bitmatrix_details.sum_bitmatrix_rows import sum_bitmatrix_rows
    except ImportError:
        return

    @triton.jit
    def _stage2_pow2(ColSortedIndx, RowSortedIndx, NonzeroIndx, n_tokens, ColPartialSum, stride_pm, stride_pn, ColOffs, TOKS_PER_ROW: tl.constexpr, BLOCK_PER_TOK: tl.constexpr, BLOCK_SIZE_PADDED: tl.constexpr):
        BLOCK_SIZE: tl.constexpr = BLOCK_PER_TOK * TOKS_PER_ROW
        tl.static_assert(BLOCK_SIZE_PADDED <= 32768)
        if isinstance(n_tokens, tl.tensor) and n_tokens.dtype.is_ptr():
            n_tokens = tl.load(n_tokens)
        nonzero_indx_size = n_tokens * TOKS_PER_ROW
        pid_m = tl.program_id(0)
        offs_local = tl.arange(0, BLOCK_SIZE_PADDED)
        offs_global = pid_m * BLOCK_SIZE + offs_local
        mask = offs_global < nonzero_indx_size
        col_indx = tl.load(NonzeroIndx + offs_global, mask=mask, other=-1).to(tl.uint32)
        kv_pairs = (col_indx << 16 | offs_local).to(tl.uint32)
        kv_pairs = tl.sort(kv_pairs, 0)
        col_indx = kv_pairs >> 16
        offs_global = pid_m * BLOCK_SIZE + (kv_pairs & 65535)
        mask = col_indx != 65535
        x = kv_pairs & 4294901760 | 1
        cols_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
        exclusive_run_lengths = cols_and_inclusive_run_lengths - 1 & 65535
        row_sorted_indx = tl.load(ColPartialSum + pid_m * stride_pm + col_indx * stride_pn, mask=mask)
        row_sorted_indx += tl.load(ColOffs + col_indx, mask=mask)
        row_sorted_indx += exclusive_run_lengths
        tl.store(RowSortedIndx + offs_global, row_sorted_indx, mask=mask)
        tl.store(ColSortedIndx + row_sorted_indx, offs_global, mask=mask)

    def _make_bitmatrix_metadata_pow2_safe(nonzero_indx, bitmatrix):
        assert nonzero_indx.ndim == 2
        PARTIAL_BLOCK_M = 32
        col_sum, col_partial_sum = sum_bitmatrix_rows(bitmatrix, partials_block_size=PARTIAL_BLOCK_M)
        device = bitmatrix.device
        n_indx = nonzero_indx.numel()
        n_cols = bitmatrix.shape[1]
        col_offs = torch.empty(n_cols, dtype=torch.int32, device=device)
        combined_indx = torch.empty(n_indx * 2, dtype=torch.int32, device=device)
        col_sorted_indx = combined_indx[:n_indx]
        row_sorted_indx = combined_indx[n_indx:]
        MEMSET_BLOCK = 1024
        memset_grid = (cdiv(n_indx * 2, MEMSET_BLOCK) + n_cols + 1,)
        _bm._bitmatrix_metadata_compute_stage1[memset_grid](combined_indx, n_indx * 2, -1, MEMSET_BLOCK, col_sum, col_offs, col_sum.shape[0], col_partial_sum, col_partial_sum.shape[0], col_partial_sum.stride(0), col_partial_sum.stride(1), BLOCK_M=512, BLOCK_N=512)
        toks_per_row = nonzero_indx.shape[-1]
        block_size = PARTIAL_BLOCK_M * toks_per_row
        block_size_padded = 1 << (max(block_size, 1) - 1).bit_length()
        compute_grid = (cdiv(bitmatrix.shape_max[0], PARTIAL_BLOCK_M),)
        _stage2_pow2[compute_grid](col_sorted_indx, row_sorted_indx, nonzero_indx, bitmatrix.shape[0], col_partial_sum, col_partial_sum.stride(0), col_partial_sum.stride(1), col_offs, TOKS_PER_ROW=toks_per_row, BLOCK_PER_TOK=PARTIAL_BLOCK_M, BLOCK_SIZE_PADDED=block_size_padded)
        return BitmatrixMetadata(col_sum=col_sum, col_sorted_indx=col_sorted_indx, row_sorted_indx=row_sorted_indx)
    from triton_kernels.tensor import SparseMatrix as _SparseMatrix
    _SparseMatrix.__post_init__.__globals__['make_bitmatrix_metadata'] = _make_bitmatrix_metadata_pow2_safe
    _bm.make_bitmatrix_metadata = _make_bitmatrix_metadata_pow2_safe
use_legacy_triton_kernels = False
if has_triton_kernels():
    try:
        import triton_kernels.swiglu
        from triton_kernels.matmul_ogs import FnSpecs, FusedActivation, GatherIndx, RoutingData, ScatterIndx, matmul_ogs
        from triton_kernels.tensor import BIT, Bitmatrix
        try:
            from triton_kernels.tensor import SparseMatrix, make_ragged_tensor_metadata
        except ImportError:
            if current_platform.is_rocm():
                logger.warning_once('Using legacy triton_kernels on ROCm')
                use_legacy_triton_kernels = True
            else:
                raise
        if not use_legacy_triton_kernels:
            _patch_make_bitmatrix_metadata()
    except (AttributeError, ImportError) as e:
        logger.error('Failed to import Triton kernels. Please make sure your triton version is compatible. Error: %s', e)

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
