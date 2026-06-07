"""Task L2_29_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe`
Source path: `sglang/srt/layers/moe/ep_moe/kernels.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): 4e5622e70e0050e5e9d66dae898eb3ce8f06678da90d81777414ab4337cb49b8
Frozen by: scripts/phase_a_freeze.py
"""
import logging
import torch
import triton
from tasks._frozen_shim import ceil_div, is_cuda, is_musa  # Phase A': platform/math predicate (B-inert-shim)
_is_cuda = is_cuda()
_is_musa = is_musa()
if _is_cuda or _is_musa:
    from tasks._frozen_shim import sglang_per_token_group_quant_fp8 as per_token_group_quant_fp8  # Phase A': fp8 quant helpers -> shim (B-inert-shim)
import triton.language as tl

@triton.jit
def _silu_and_mul_post_quant_kernel(input_ptr, stride_input_0, stride_input_1, stride_input_2, output_ptr, stride_output_0, stride_output_1, stride_output_2, output_scale_ptr, stride_output_scale_0, stride_output_scale_1, stride_output_scale_2, masked_m_ptr, size_n, fp8_max, fp8_min, BLOCK_N: tl.constexpr, NUM_STAGE: tl.constexpr, SCALE_UE8M0: tl.constexpr):
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)
    block_num_per_expert = tl.num_programs(1)
    token_num_cur_expert = tl.load(masked_m_ptr + expert_id)
    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)
    offs_in_d = hidden_dim_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    input_ptr_offs = input_ptr + expert_id * stride_input_0 + offs_in_d
    output_ptr_offs = output_ptr + expert_id * stride_output_0 + offs_in_d
    output_scale_offs = output_scale_ptr + expert_id * stride_output_scale_0 + hidden_dim_block_index * stride_output_scale_2
    for token_index in tl.range(token_id, token_num_cur_expert, block_num_per_expert, num_stages=NUM_STAGE):
        gate = tl.load(input_ptr_offs + token_index * stride_input_1, mask=offs_in_d < size_n, other=0.0).to(tl.float32)
        up = tl.load(input_ptr_offs + token_index * stride_input_1 + size_n, mask=offs_in_d < size_n, other=0.0)
        gate = gate / (1 + tl.exp(-gate))
        gate = gate.to(input_ptr.dtype.element_ty)
        gate_up = up * gate
        _absmax = tl.maximum(tl.max(tl.abs(gate_up)), 1e-10)
        output_s = _absmax / fp8_max
        if SCALE_UE8M0:
            output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))
        output_q = tl.clamp(gate_up / output_s, fp8_min, fp8_max).to(output_ptr.dtype.element_ty)
        tl.store(output_ptr_offs + token_index * stride_output_1, output_q, mask=offs_in_d < size_n)
        tl.store(output_scale_offs + token_index * stride_output_scale_1, output_s)

import torch
import triton


# Authored jit-only launcher: drives the in-module @triton.jit
# `_silu_and_mul_post_quant_kernel` exactly as the GRADED host wrapper
# `silu_and_mul_masked_post_quant_fwd` (kernel.py lines 380-445) does
# (SGLang, commit 409d350, sglang/srt/layers/moe/ep_moe/kernels.py, copied
# from lightllm moe_silu_and_mul_mix_quant_ep.py).
#
# Why a launcher rather than calling the host wrapper symbol: the module's
# top-level imports (`from sglang.srt.utils import ...` and, on CUDA/MUSA,
# `sglang.srt.layers.quantization.fp8_kernel`) are an unvendorable closure
# pulled in only by SIBLING helpers, not by THIS kernel or its wrapper. The
# wrapper body is pure grid/stride/scalar arithmetic, so we reproduce its
# invocation verbatim and the @triton.jit kernel is always the unit graded.
#
# Wrapper contract reproduced here (scale_ue8m0=False, the E4M3 path the
# oracle covers):
#   * input [E, T, 2*D] contiguous fp ; masked_m [E] (per-expert valid count)
#   * size_n = input.shape[-1] // 2  (== D) ; size_n % quant_group_size == 0
#   * output       = zeros((E, T, D), float8_e4m3fn)            contiguous
#   * output_scale = zeros((E, T, D // quant_group_size), float32) contiguous
#   * BLOCK_N = quant_group_size ; NUM_STAGES = 6 ; num_warps = 1
#   * BLOCK_NUM_PER_EXPERT = 64 if E < 4 else 32
#   * hidden_dim_split_block_num = cdiv(size_n, BLOCK_N) (== D // group here)
#   * grid = (hidden_dim_split_block_num, BLOCK_NUM_PER_EXPERT, E)
#   * fp8_max = finfo(float8_e4m3fn).max (448.0) ; fp8_min = -fp8_max
#   * strides via *input.stride(), *output.stride(), *output_scale.stride()
#
# Kernel math (matches the re-authored oracle_pytorch): per (expert e, group
# block) and each valid token t < masked_m[e],
#   gate    = silu(input[e, t, :size_n][group]) cast back to input dtype
#   gate_up = up * gate          (up = input[e, t, size_n:][group])
#   absmax  = max(max(|gate_up|), 1e-10)        # 1e-10 floors the AMAX
#   out_s   = absmax / fp8_max
#   out_q   = clamp(gate_up / out_s, fp8_min, fp8_max) -> float8_e4m3fn
# Tokens t >= masked_m[e] are never written by the kernel; we zero-init both
# outputs so invalid rows are deterministic 0 and match the oracle's masked
# (y * valid) invalid rows under the harness full-tensor compare. Valid rows
# are fully overwritten, so zero-init does not change them.
# Output keys: "output" (fp8 quant), "output_scale" (fp32 group scale).
def run(inputs: dict) -> dict:
    x = inputs["input"]
    masked_m = inputs["masked_m"]
    quant_group_size = int(inputs["quant_group_size"])

    assert x.ndim == 3, "input must be (E, T, 2*D)"
    E, T, two_d = x.shape
    assert two_d % 2 == 0, "last dim of input must be even (2*D)"
    size_n = two_d // 2
    assert size_n % quant_group_size == 0, "size_n % quant_group_size != 0"
    assert masked_m.ndim == 1 and masked_m.shape[0] == E

    x = x.contiguous()
    masked_m = masked_m.to(device=x.device, dtype=torch.int64).contiguous()

    fp8_dtype = torch.float8_e4m3fn
    finfo = torch.finfo(fp8_dtype)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    # candidate_inplace: write into the runner-provided buffers. They arrive
    # zero-initialised (shapes.yaml `dist: zeros`); re-zero defensively so any
    # token row never written by the kernel (t >= masked_m[e]) is a
    # deterministic 0, matching the oracle's masked invalid rows.
    output = inputs["output"].to(device=x.device).contiguous()
    output_scale = inputs["output_scale"].to(device=x.device).contiguous()
    assert output.dtype == fp8_dtype
    assert output.shape == (E, T, size_n)
    assert output_scale.shape == (E, T, size_n // quant_group_size)
    output.zero_()
    output_scale.zero_()

    if E < 4:
        BLOCK_NUM_PER_EXPERT = 64
    else:
        BLOCK_NUM_PER_EXPERT = 32

    BLOCK_N = quant_group_size
    num_warps = 1
    NUM_STAGES = 6
    hidden_dim_split_block_num = triton.cdiv(size_n, BLOCK_N)

    grid = (
        hidden_dim_split_block_num,
        BLOCK_NUM_PER_EXPERT,
        E,
    )

    _silu_and_mul_post_quant_kernel[grid](
        x,
        *x.stride(),
        output,
        *output.stride(),
        output_scale,
        *output_scale.stride(),
        masked_m,
        size_n,
        fp8_max,
        fp8_min,
        BLOCK_N=BLOCK_N,
        NUM_STAGE=NUM_STAGES,
        num_warps=num_warps,
        SCALE_UE8M0=False,
    )

    return {"output": output, "output_scale": output_scale}
