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
