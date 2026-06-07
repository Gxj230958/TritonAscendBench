import torch
import triton


# Authored jit-only launcher: drives the in-module @triton.jit
# `_silu_and_mul_post_quant_kernel` exactly as the upstream host wrapper
# `silu_and_mul_masked_post_quant_fwd` does (SGLang, commit 409d350,
# sglang/srt/layers/moe/ep_moe/kernels.py, copied from lightllm
# moe_silu_and_mul_mix_quant_ep.py).
#
# Why a launcher rather than the host wrapper: the module imports
# `sglang.srt.utils` (ceil_div / is_cuda / is_musa) and, on CUDA/MUSA,
# `sglang.srt.layers.quantization.fp8_kernel` -- an unvendorable closure
# (>800 symbols, cyclic). None of that is needed to drive THIS kernel; the
# wrapper body is pure grid/stride/scalar arithmetic. This launcher
# reproduces `silu_and_mul_masked_post_quant_fwd` (kernel.py lines 380-445)
# verbatim so the @triton.jit kernel is always the unit evaluated.
#
# Upstream call site reproduced here (scale_ue8m0=False, the E4M3 path the
# oracle covers):
#   * input [E, T, 2*D] contiguous fp ; masked_m [E] (per-expert valid count)
#   * size_n = input.shape[-1] // 2  (== D) ; size_n % quant_group_size == 0
#   * output     = empty((E, T, D), fp8_e4m3fn)   contiguous
#   * output_scale = empty((E, T, D // quant_group_size), float32) contiguous
#   * BLOCK_N = quant_group_size ; NUM_STAGES = 6 ; num_warps = 1
#   * BLOCK_NUM_PER_EXPERT = 64 if E < 4 else 32
#   * hidden_dim_split_block_num = cdiv(size_n, BLOCK_N) (== D // group here)
#   * grid = (hidden_dim_split_block_num, BLOCK_NUM_PER_EXPERT, E)
#   * fp8_max = finfo(float8_e4m3fn).max (448.0) ; fp8_min = -fp8_max
#   * strides passed by *input.stride(), *output.stride(),
#     *output_scale.stride()
#
# Kernel math (matches the re-authored oracle_pytorch): per (expert e, group
# block hidden_dim_block_index) and each valid token t < masked_m[e],
#   gate = silu(input[e, t, :size_n][group]) cast back to input dtype
#   gate_up = up * gate          (up = input[e, t, size_n:][group])
#   absmax  = max(max(|gate_up|), 1e-10)        # 1e-10 floors the AMAX
#   out_s   = absmax / fp8_max
#   out_q   = clamp(gate_up / out_s, fp8_min, fp8_max) -> fp8_e4m3fn
# Tokens t >= masked_m[e] are never written by the kernel; we zero-init both
# outputs so the invalid rows are deterministic 0 and match the oracle's
# masked (y * valid) invalid rows under the harness full-tensor compare.
# Valid rows are fully overwritten, so zero-init does not change them.
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

    # Zero-init so tokens never written by the kernel (t >= masked_m[e]) are
    # deterministic 0, matching the oracle's masked invalid rows.
    output = torch.zeros((E, T, size_n), dtype=fp8_dtype, device=x.device)
    output_scale = torch.zeros(
        (E, T, size_n // quant_group_size), dtype=torch.float32, device=x.device
    )

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
