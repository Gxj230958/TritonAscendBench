# Phase A': sliced to the graded entrypoint's module-level closure
# (whole-module vendored kernel; dead upstream siblings + their
# upstream imports dropped -- the kept defs are byte-identical).
import torch
import triton
import triton.language as tl


@triton.jit
def moe_mmk(
    a_ptrs,
    b_ptrs,
    K,
    expert_id,
    a_scale_ptr,
    b_scale_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_ak: tl.int64,
    stride_bk: tl.int64,
    stride_ase: tl.int64,
    stride_asm: tl.int64,
    stride_ask: tl.int64,
    stride_bse: tl.int64,
    stride_bsk: tl.int64,
    stride_bsn: tl.int64,
    # Offsets and masks
    offs_m,
    offs_n,
    offs_bn,
    mask_m,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    compute_type: tl.constexpr,
    use_w8a8: tl.constexpr,
    use_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
):
    offs_k = tl.arange(0, BLOCK_K)

    if use_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + expert_id * stride_bse + offs_n[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = b_scale_ptr + offs_bsn * stride_bsn

        # per act token
        elif per_act_token_quant:
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=mask_m, other=0.0)[:, None]

            b_scale_ptrs = b_scale_ptr + offs_bn[None, :] * stride_bsn
            b_scale = tl.load(b_scale_ptrs)

        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & (offs_k[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        # We accumulate along the K dimension.
        if use_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=mask_m, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                # acc used to enable fp8_fast_accum
                accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if use_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)

    return accumulator


@triton.jit
def expert_triton_kernel(
    a_ptr,  # [max_tokens, K]
    b_ptr,  # [K, N]
    c_ptr,  # [max_tokens, N]
    expert_id,
    compute_type: tl.constexpr,
    # Dimensions
    M,
    N,
    K,
    # Quantization data
    a_scale_ptr,
    b_scale_ptr,
    b_zp_ptr,
    # strides
    stride_am: tl.int64,
    stride_ak: tl.int64,
    stride_bk: tl.int64,
    stride_bn: tl.int64,
    stride_cm: tl.int64,
    stride_cn: tl.int64,
    stride_ase: tl.int64,
    stride_asm: tl.int64,
    stride_ask: tl.int64,
    stride_bse: tl.int64,
    stride_bsk: tl.int64,
    stride_bsn: tl.int64,
    # offsets
    offs_bn,
    # Blockwise quantization data
    group_n,
    group_k,
    # Quantization schemes
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
    # Kernel config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N) % N
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    # Make grids of a + b pointers
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = moe_mmk(
        a_ptrs,
        b_ptrs,
        K,
        expert_id,
        a_scale_ptr,
        b_scale_ptr,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_ak,
        stride_bk,
        stride_ase,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Offsets and masks
        offs_m,
        offs_n,
        offs_bn,
        mask_m,
        # Block size for block-wise quantization
        group_n,
        group_k,
        # Meta-parameters
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a16,
        per_act_token_quant,
    )

    # store in C
    offs_cn = tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = mask_m[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def batched_triton_kernel(
    a_ptr,  # [E, max_num_tokens, K]
    b_ptr,  # [E, K, N]
    c_ptr,  # [E, max_num_tokens, N]
    expert_num_tokens,  # [E]
    compute_type: tl.constexpr,
    # Dimensions
    max_num_tokens,
    K,
    N,
    # Quantization data
    a_scale_ptr,
    b_scale_ptr,
    b_zp_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_ae: tl.int64,
    stride_am: tl.int64,
    stride_ak: tl.int64,
    stride_be: tl.int64,
    stride_bk: tl.int64,
    stride_bn: tl.int64,
    stride_ce: tl.int64,
    stride_cm: tl.int64,
    stride_cn: tl.int64,
    stride_ase: tl.int64,
    stride_asm: tl.int64,
    stride_ask: tl.int64,
    stride_bse: tl.int64,
    stride_bsk: tl.int64,
    stride_bsn: tl.int64,
    # Blockwise quantization data
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Quantization schemes
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
    # Kernel config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert_id = tl.program_id(axis=0)
    e_num_tokens = tl.load(expert_num_tokens + expert_id)
    if e_num_tokens == 0:
        # Early exit
        return

    # axis 1 is M_blocks * N_blocks
    pid_mn = tl.program_id(axis=1)
    # num_pid_m = tl.cdiv(max_num_tokens, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    cta_m_start = pid_m * BLOCK_M
    cta_n_start = pid_n * BLOCK_N
    if cta_m_start >= e_num_tokens:
        # Early exit
        return

    cta_m_size = min(BLOCK_M, e_num_tokens - cta_m_start)
    cta_n_size = min(BLOCK_N, N - cta_n_start)

    a_ptr = a_ptr + expert_id * stride_ae + cta_m_start * stride_am
    b_ptr = b_ptr + expert_id * stride_be + cta_n_start * stride_bn
    c_ptr = (
        c_ptr
        + expert_id * stride_ce
        + cta_m_start * stride_cm
        + cta_n_start * stride_cn
    )

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)) % N

    if use_fp8_w8a8:
        a_scale_ptr = a_scale_ptr + expert_id * stride_ase
        b_scale_ptr = b_scale_ptr + expert_id * stride_bse

        # block-wise
        if group_k > 0 and group_n > 0 or per_act_token_quant:
            a_scale_ptr = a_scale_ptr + cta_m_start * stride_asm

    expert_triton_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        expert_id,
        compute_type,
        cta_m_size,  # M
        cta_n_size,  # N
        K,  # K
        a_scale_ptr,
        b_scale_ptr,
        b_zp_ptr,
        # Strides
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_ase,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # offsets
        offs_bn,
        # Blockwise quantization data
        group_n,
        group_k,
        # Quantization schemes
        use_fp8_w8a8,
        use_int8_w8a16,
        per_act_token_quant,
        # Kernel config
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

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
