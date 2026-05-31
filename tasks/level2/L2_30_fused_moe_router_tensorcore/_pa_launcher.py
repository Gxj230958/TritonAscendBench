import torch
import triton


# Authored jit-only launcher: drives the in-module @triton.jit
# `fused_moe_router_tensorcore_kernel` (SGLang sglang/srt/layers/moe/router.py,
# commit 409d350f) exactly as its host wrapper `fused_moe_router_tensorcore`
# does -- the graded op named by meta.name. We drive the JIT directly rather
# than the wrapper for two reasons:
#   1. `fused_moe_router_tensorcore` imports sglang.srt.layers.dp_attention to
#      read is_dp_attention_enabled(), pulling the sglang runtime off-GPU.
#   2. The wrapper takes BLOCK_SIZE_K from its caller (the shim hardcodes 256/64)
#      and asserts `hidden_dim % BLOCK_SIZE_K == 0`; on the D=16 shape (H2) the
#      fixed BLOCK_SIZE_K from shapes.yaml does not divide D and the assert trips.
# (Same pattern as the passed tensorcore sibling L2_11_w8a8_triton_block_scaled_mm.)
#
# Kernel math the JIT realises (matches the re-authored oracle_pytorch):
#   logits = x @ W.T        accumulated in fp32 via tl.dot over K
#   if softcap > 0:  logits = (exp(2*logits/c)-1)/(exp(2*logits/c)+1) * c   # == c*tanh(logits/c)
#   logits += correction_bias                                              # bias AFTER softcap
#   top1 = argmax(logits);  top1_w = 1/sum(exp(logits - max)) == softmax(logits)[top1]
#   top2 = argmax(logits with top1 masked);  top2_w = exp(top2_v - top1_v) * top1_w
# topk_weights are RAW softmax probabilities at top1/top2 (NOT renormalised).
# Outputs are written as fp32 weights + int32 ids -- the oracle contract keys.
#
# Block sizes (launch metadata only; the kernel accumulates over the FULL K and
# masks padded experts, so the result is independent of the tiling):
#   BLOCK_SIZE_M = 32                upstream shim default.
#   BLOCK_SIZE_N = max(num_experts, 16)   must be >= num_experts; the kernel pads
#                  the N axis to BLOCK_SIZE_N and masks experts >= num_experts via
#                  cond_top1. num_experts (=8 here) <= BLOCK_SIZE_N means exactly
#                  ONE N-block (cdiv(ne, BSN)=1); the flattened [bs, topk] store
#                  offsets (offs_top1/offs_top2) depend only on pid/BLOCK_SIZE_M/
#                  topk, so they stay correct only under this single-N-block
#                  layout -- which num_experts <= BLOCK_SIZE_N guarantees.
#   BLOCK_SIZE_K = largest power-of-2 that divides K, clamped to [16, 256].
#                  The kernel asserts K % BLOCK_SIZE_K == 0; the acc loop sums
#                  `tl.dot(a_tile, b_tile.T)` over K // BLOCK_SIZE_K chunks with no
#                  K masking, so any divisor of K yields the same matmul up to fp32
#                  chunk-accumulation order (tolerance-safe within atomic_scatter,
#                  ~3e-6 observed; NOT claimed bit-exact for arbitrary BSK). >=16
#                  gives tl.dot a valid MMA K tile. Upstream's 256/64 choice is
#                  perf-only and assumes K % BLOCK_SIZE_K == 0; we pick the divisor
#                  so every shape -- including D=16 -- is invocable. The frozen JIT
#                  calls `tl.dot(a, b)` with no input_precision, so we likewise pass
#                  none (forcing "ieee" would change the frozen op's tensor-core
#                  TF32 default -- unfaithful).
# Grid mirrors the wrapper: (cdiv(bs, BLOCK_SIZE_M) * cdiv(num_experts, BLOCK_SIZE_N),).
# dp_attn_workaround_flag = False: the NaN->-1e9 cleanup is the dp-attention path
# (is_dp_attention_enabled() is False here); inputs are randn/rand with no NaNs and
# the oracle has no NaN handling, so skipping it is faithful.
def _block_size_k(hidden_dim: int) -> int:
    cap = min(hidden_dim, 256)
    bsk = 1
    p = 1
    while p <= cap:
        if hidden_dim % p == 0:
            bsk = p
        p *= 2
    return max(bsk, min(16, hidden_dim))


def run(inputs: dict) -> dict:
    x = inputs["x"].contiguous()
    router_weight = inputs["router_weight"].contiguous()
    correction_bias = inputs.get("correction_bias")
    topk = int(inputs["topk"])
    moe_softcapping = float(inputs.get("moe_softcapping", 0.0))

    assert x.ndim == 2 and x.shape[1] == router_weight.shape[1], (
        x.shape, router_weight.shape,
    )
    assert topk <= 2, f"tensorcore kernel only supports topk <= 2, got {topk}"

    bs, hidden_dim = x.shape
    num_experts = router_weight.shape[0]

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = max(num_experts, 16)
    BLOCK_SIZE_K = _block_size_k(hidden_dim)
    assert num_experts <= BLOCK_SIZE_N
    assert hidden_dim % BLOCK_SIZE_K == 0

    if correction_bias is not None:
        correction_bias = correction_bias.contiguous()
    is_correction_bias = correction_bias is not None

    topk_weights = torch.empty((bs, topk), dtype=torch.float32, device=x.device)
    topk_ids = torch.empty((bs, topk), dtype=torch.int32, device=x.device)

    grid = (
        triton.cdiv(bs, BLOCK_SIZE_M) * triton.cdiv(num_experts, BLOCK_SIZE_N),
    )

    fused_moe_router_tensorcore_kernel[grid](  # noqa: F821 (injected into kernel.py module)
        a_ptr=x,
        b_ptr=router_weight,
        topk_weights_ptr=topk_weights,
        topk_ids_ptr=topk_ids,
        bs=bs,
        num_experts=num_experts,
        topk=topk,
        moe_softcapping=moe_softcapping,
        moe_renormalize=False,
        correction_bias_ptr=correction_bias,
        is_correction_bias=is_correction_bias,
        K=hidden_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        stride_am=hidden_dim,
        stride_bn=hidden_dim,
        dp_attn_workaround_flag=False,
    )

    return {"topk_weights": topk_weights, "topk_ids": topk_ids}
