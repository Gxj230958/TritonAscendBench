import torch


# Authored jit-only launcher: drives the in-module @triton.jit
# `act_and_mul_kernel` exactly as the upstream host wrapper
# `act_and_mul_triton` does for the UNSORTED routing path (SGLang, commit
# 409d350, sglang/srt/layers/moe/.../fused_moe_triton_kernels.py).
#
# Why a launcher rather than the host wrapper: the module imports
# `sglang.srt.utils` / `sglang.srt.layers.moe.utils` / the fp8+int8 quant
# kernels -- an unvendorable closure pulled in only by the dead sibling
# `invoke_fused_moe_kernel` / `fused_moe_kernel`. None of that is needed to
# drive THIS kernel; the wrapper body for `act_and_mul_triton` is pure
# grid/scalar setup. This launcher reproduces it verbatim for the
# `down_moe_use_tma=False` (unsorted) path so the @triton.jit kernel is the
# unit evaluated. (Same pattern as the passed sibling
# L2_30_silu_and_mul_post_quant.)
#
# Upstream `act_and_mul_triton(gateup_output, down_input, config, topk_ids,
# ..., down_moe_use_tma=False, activation="silu", swiglu_limit=None)` call site
# reproduced here:
#   * grid          = (down_input.shape[0],)                  # one program/row
#   * hidden_size   = gateup_output.shape[1]                  # == 2*half
#   * expert_ids    = topk_ids.view(-1)                       # unsorted layout
#   * expert_step   = 1                                       # unsorted layout
#   * BLOCK_SIZE=512, ACTIVATION_TYPE=activation,
#     SWIGLU_LIMIT=0.0, HAS_SWIGLU_LIMIT=False (no swiglu clamp -- oracle scope)
#
# Kernel math (matches the re-authored oracle_pytorch): per row pid,
#   expert_id = topk_ids[pid]
#   if expert_id == -1: return            # dropped token -> row NOT written
#   gate = gateup_output[pid, :half]; up = gateup_output[pid, half:]
#   down_input[pid] = act(gate).to(in_dtype) * up
# topk_ids == -1 is the load-bearing routing mask: those rows are skipped. We
# zero-init down_input so the skipped rows are deterministic 0 and match the
# oracle's masked (zeroed) rows under the harness full-tensor compare; the
# valid rows are fully overwritten, so zero-init does not change them.
# Output key: "output" (the down-projection input the kernel writes).
def run(inputs: dict) -> dict:
    gateup_output = inputs["gateup_output"].contiguous()
    # topk_ids is the per-token expert assignment; upstream forwards it via
    # `topk_ids.view(-1)` preserving dtype. Keep the input dtype (int64),
    # only flatten + contiguous as the wrapper does.
    topk_ids = inputs["topk_ids"].view(-1).contiguous()
    activation = str(inputs.get("act", "silu"))

    assert gateup_output.ndim == 2, "gateup_output must be (num_tokens, 2*half)"
    num_tokens, hidden_size = gateup_output.shape
    assert hidden_size % 2 == 0, "hidden_size must be even (gate || up)"
    half_hidden_size = hidden_size // 2
    assert topk_ids.numel() == num_tokens, "topk_ids must have one id per token"

    # Zero-init so rows the kernel skips (topk_ids == -1) stay 0, matching the
    # oracle's masked rows. Output dtype follows the input dtype (the kernel
    # casts the activated gate to InDtype then stores as OutDtype == InDtype).
    down_input = torch.zeros(
        (num_tokens, half_hidden_size),
        dtype=gateup_output.dtype,
        device=gateup_output.device,
    )

    grid = (num_tokens,)
    expert_step = 1  # unsorted routing: expert id looked up per row (pid // 1)
    act_and_mul_kernel[grid](
        gateup_output,
        down_input,
        hidden_size,
        topk_ids,
        expert_step,
        BLOCK_SIZE=512,
        ACTIVATION_TYPE=activation,
        SWIGLU_LIMIT=0.0,
        HAS_SWIGLU_LIMIT=False,
    )

    return {"output": down_input}
