"""Task L1_25_compute_problem_sizes_w4a8_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `compute_problem_sizes_w4a8_kernel`
Source path: `sglang/srt/layers/moe/ep_moe/kernels.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): 4e5622e70e0050e5e9d66dae898eb3ce8f06678da90d81777414ab4337cb49b8
Frozen by: scripts/phase_a_freeze.py
"""
import logging
import torch
import triton
from ascendbench.frozen.shim import ceil_div, is_cuda, is_musa  # Phase A': platform/math predicate (B-inert-shim)
_is_cuda = is_cuda()
_is_musa = is_musa()
if _is_cuda or _is_musa:
    from ascendbench.frozen.shim import sglang_per_token_group_quant_fp8 as per_token_group_quant_fp8  # Phase A': fp8 quant helpers -> shim (B-inert-shim)
import triton.language as tl

@triton.jit
def compute_problem_sizes_w4a8_kernel(masked_m_ptr, problem_sizes1_ptr, problem_sizes2_ptr, n, k, num_experts, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pid < num_experts
    final_occurrences = tl.load(masked_m_ptr + pid, mask=mask, other=0)
    ps1_idx_0 = pid * 3
    ps1_idx_1 = ps1_idx_0 + 1
    ps1_idx_2 = ps1_idx_0 + 2
    ps2_idx_0 = pid * 3
    ps2_idx_1 = ps2_idx_0 + 1
    ps2_idx_2 = ps2_idx_0 + 2
    ps1_mask_0 = ps1_idx_0 < num_experts * 3
    ps1_mask_1 = ps1_idx_1 < num_experts * 3
    ps1_mask_2 = ps1_idx_2 < num_experts * 3
    ps2_mask_0 = ps2_idx_0 < num_experts * 3
    ps2_mask_1 = ps2_idx_1 < num_experts * 3
    ps2_mask_2 = ps2_idx_2 < num_experts * 3
    tl.store(problem_sizes1_ptr + ps1_idx_0, 2 * n, mask=ps1_mask_0)
    tl.store(problem_sizes1_ptr + ps1_idx_1, final_occurrences, mask=ps1_mask_1)
    tl.store(problem_sizes1_ptr + ps1_idx_2, k, mask=ps1_mask_2)
    tl.store(problem_sizes2_ptr + ps2_idx_0, k, mask=ps2_mask_0)
    tl.store(problem_sizes2_ptr + ps2_idx_1, final_occurrences, mask=ps2_mask_1)
    tl.store(problem_sizes2_ptr + ps2_idx_2, n, mask=ps2_mask_2)

import torch
import triton


# Authored jit-only launcher: drives the in-module @triton.jit
# `compute_problem_sizes_w4a8_kernel` through its upstream host wrapper
# `compute_problem_sizes_w4a8`, exactly as the SGLang deepep_ll call site does
# (cutlass_w4a8_moe.py::cutlass_w4a8_moe_deepep_ll ->
#  deepep_ll_get_cutlass_w4a8_moe_mm_data -> compute_problem_sizes_w4a8).
#
# The upstream wrappers cannot serve as the entrypoint directly: the resolver's
# by-name binder needs problem_sizes1/problem_sizes2 (caller-allocated int32
# output buffers the kernel fills in place), which the input contract does not
# carry. The faithful path is to allocate those buffers here and drive the bare
# host wrapper, mirroring the upstream invocation.
#
# Per-expert semantics produced by the frozen kernel (m = masked_m[e]):
#   problem_sizes1[e] = (2*n, m, k)   first GEMM, gate+up fused -> N doubled
#   problem_sizes2[e] = (k,   m, n)   second (down) GEMM
# masked_m is the only load-bearing tensor input; n / k are scalar GEMM dims.
#
# Faithfulness notes:
#   * masked_m is passed as-is (the wrapper does no cast); the kernel loads it
#     and stores the per-expert count into the int32 output buffers, matching
#     upstream where masked_m is the per-expert active-token count.
#   * problem_sizes1/2 are int32, exactly the dtype upstream allocates and the
#     dtype `deepep_ll_get_cutlass_w4a8_moe_mm_data` returns.
# Output keys match the oracle contract: "problem_sizes1", "problem_sizes2".
def run(inputs: dict) -> dict:
    masked_m = inputs["masked_m"].contiguous()
    n = int(inputs["n"])
    k = int(inputs["k"])
    num_experts = int(inputs["num_experts"])

    problem_sizes1 = torch.empty(
        (num_experts, 3), dtype=torch.int32, device=masked_m.device
    )
    problem_sizes2 = torch.empty(
        (num_experts, 3), dtype=torch.int32, device=masked_m.device
    )

    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(num_experts, meta["BLOCK_SIZE"]),)
    compute_problem_sizes_w4a8_kernel[grid](
        masked_m,
        problem_sizes1,
        problem_sizes2,
        n,
        k,
        num_experts,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return {"problem_sizes1": problem_sizes1, "problem_sizes2": problem_sizes2}
