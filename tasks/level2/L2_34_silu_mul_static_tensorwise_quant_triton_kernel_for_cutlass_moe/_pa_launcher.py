import torch

# Authored launcher: drives the in-module @triton.jit
# `silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe` exactly as
# the upstream host wrapper `silu_mul_static_tensorwise_quant_for_cutlass_moe`
# does (SGLang sglang/srt/layers/moe/ep_moe/kernels.py, commit
# 409d350fb6f6a1e7c7546e39028f811092a8e489).
#
# Why a launcher rather than the registered op: meta.oracle.gpu_triton is
# `kernel.py::placeholder` (no resolvable callable) and the file is a
# multi-kernel orchestration module whose module-scope imports
# (`from sglang.srt.utils import ...`, the conditional `per_token_group_quant_fp8`
# import) are pulled in only by SIBLING helpers, not by the target kernel or its
# host wrapper. We reproduce the wrapper's invocation verbatim so the bench can
# synthesise run() over plain dense tensors.
#
# Upstream call site (cutlass_w4a8_moe.py):
#   silu_mul_static_tensorwise_quant_for_cutlass_moe(
#       c1, intermediate_q, a2_scale.float(), expert_offsets[-1:], m * topk, n
#   )
#   * c1 (input):  [num_tokens, 2 * intermediate_size], bf16 grouped-GEMM out
#       (gate = [:, :n], up = [:, n:]).
#   * intermediate_q (output): [num_tokens, intermediate_size], float8_e4m3fn,
#       written in place.
#   * a2_scale.float() (scale): scalar fp32.
#   * expert_offsets[-1:] (num_tokens_tensor): 1-elem int tensor == m*topk.
#   * m*topk (expected_num_tokens): grid sizing only.
#   * n (intermediate_size): half of input's last dim.
#
# Kernel math (per output element id, token = id // isz):
#   gate = input[token, 0:isz]; up = input[token, isz:2*isz]
#   out  = gate * sigmoid(gate) * up * (1.0 / scale)  -> .to(out.dtype)
def run(inputs: dict) -> dict:
    input = inputs["input"].contiguous()
    output = inputs["output"].contiguous()
    scale = inputs["scale"]
    num_tokens_tensor = inputs["num_tokens_tensor"]

    assert input.is_contiguous()
    assert output.is_contiguous()
    assert input.shape[-1] % 2 == 0

    intermediate_size = input.shape[-1] // 2
    expected_num_tokens = input.shape[0]

    # `silu_mul_static_tensorwise_quant_for_cutlass_moe` is in-module (spliced
    # alongside this launcher); calling it reproduces the upstream grid/config
    # (`_get_launch_config_1d`) and NUM_STAGES=3 exactly.
    silu_mul_static_tensorwise_quant_for_cutlass_moe(
        input,
        output,
        scale,
        num_tokens_tensor,
        expected_num_tokens,
        intermediate_size,
    )

    return {"output": output}
