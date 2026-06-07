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
