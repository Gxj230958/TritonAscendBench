import torch
import triton
import triton.language as tl


# Authored launcher: drives the frozen graded @triton.jit `batched_triton_kernel`
# (vLLM vllm/model_executor/layers/fused_moe/experts/fused_batched_moe.py, commit
# 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19) directly from the dense oracle
# tensors. `batched_triton_kernel` -> `expert_triton_kernel` -> `moe_mmk` is the
# only launchable path that exercises the declared graded inner kernel `moe_mmk`
# (the matmul-K accumulator device function); driving it reproduces the full
# per-expert batched GEMM the oracle models.
#
# Why a launcher rather than the in-module host wrapper
# `invoke_moe_batched_triton_kernel`: that wrapper calls
# `normalize_batched_scales_shape` (vllm.model_executor.layers.fused_moe.utils),
# whose import pulls the full vllm runtime -- unavailable off-GPU and a severance
# leak. The wrapper otherwise just marshals strides + grid for the unquantized
# path, which we reconstruct verbatim here. (Same direct-jit pattern as the
# passed siblings L2_05 / L3_13 / L3_14 / L3_20.)
#
# Stride / grid reconciliation (mirrors invoke_moe_batched_triton_kernel exactly
# for the fp32, non-quantized, A_scale=None / B_scale=None path):
#   a [E, T, K] is A; b [E, N, K] is B; output [E, T, N] is C. The kernel reads B
#   transposed via strides -- stride_bk = B.stride(2) (the K axis), stride_bn =
#   B.stride(1) (the N axis) -- so b[e] is contracted over its last axis K against
#   a[e]'s last axis K, i.e. out[e,:m] = a[e,:m] @ b[e].T, the oracle's GEMM.
#   expert_num_tokens[e] bounds the valid rows m per expert (the kernel early-exits
#   tiles past it and masks the partial M tile), reproducing the oracle's
#   `out[e, :m] = af[e,:m] @ bf[e].T` row window. All quant controls are off
#   (use_fp8_w8a8 / use_int8_w8a16 = False, per_act_token_quant = False, every
#   scale/zp ptr None, group_n = group_k = 0), so moe_mmk takes the plain
#   `accumulator += tl.dot(a, b)` branch and the fp32 `accumulator.to(fp32)` cast.
#
# BLOCK_{M,N,K} are tiling-only meta-parameters: the kernel masks every load/store
# by M (mask_m), N (offs_n % N + c_mask < N) and K (offs_k < K - k*BLOCK_K), so the
# tiles never change the numeric result, only the launch geometry. 16 is the
# minimum tile that keeps tl.dot's M/N/K operand dims valid on the smallest shape
# (H2: K=16, N=8, T=8); larger blocks are masked down identically.
_BLOCK_M = 16
_BLOCK_N = 16
_BLOCK_K = 16


def run(inputs: dict) -> dict:
    a = inputs["a"].contiguous()
    b = inputs["b"].contiguous()
    expert_num_tokens = inputs["expert_num_tokens"].to(torch.int64).contiguous()

    E, max_num_tokens, K = a.shape
    N = b.shape[1]
    assert b.shape[0] == E and b.shape[2] == K, (a.shape, b.shape)
    assert expert_num_tokens.shape[0] == E, (expert_num_tokens.shape, E)

    # C [E, max_num_tokens, N]: the kernel only stores rows < expert_num_tokens[e]
    # (and columns < N); rows past each expert's token count are left untouched.
    # The oracle returns those padding rows as zero, so zero-init reproduces the
    # full-tensor zero padding exactly.
    output = torch.zeros((E, max_num_tokens, N), device=a.device, dtype=torch.float32)

    grid = (
        E,
        triton.cdiv(max_num_tokens, _BLOCK_M) * triton.cdiv(N, _BLOCK_N),
    )

    batched_triton_kernel[grid](
        a,
        b,
        output,
        expert_num_tokens,
        tl.float32,  # compute_type (fp32)
        # Dimensions
        max_num_tokens,
        K,
        N,
        # Quantization data (unquantized fp32 path: no scales / zero-point)
        None,  # a_scale_ptr
        None,  # b_scale_ptr
        None,  # b_zp_ptr
        # Strides (mirror invoke_moe_batched_triton_kernel: B is read transposed)
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(2),  # stride_bk = B.stride(2) (K axis)
        b.stride(1),  # stride_bn = B.stride(1) (N axis)
        output.stride(0),
        output.stride(1),
        output.stride(2),
        # Scale strides (all unused, scales are None)
        0,  # stride_ase
        0,  # stride_asm
        0,  # stride_ask
        0,  # stride_bse
        0,  # stride_bsk
        0,  # stride_bsn
        # Blockwise quantization data (no block-wise quant)
        0,  # group_n
        0,  # group_k
        # Quantization schemes
        False,  # use_fp8_w8a8
        False,  # use_int8_w8a16
        False,  # per_act_token_quant
        # Kernel config
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
    )

    return {"output": output}
