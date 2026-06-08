"""Task L3_10_fused_moe_router_tensorcore_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `fused_moe_router_tensorcore_kernel`
Source path: `sglang/srt/layers/moe/router.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): a2e48dbbc5d4db0f81202c8060fa45ab6ccf6f5d4bc930da9591258740171aeb
Frozen by: scripts/phase_a_freeze.py
"""
from typing import Optional, Tuple
import torch
import triton
import triton.language as tl
from triton_ascend_bench.frozen.shim import fused_topk  # Phase A': sibling-only CUDA topk dispatcher (B-inert-shim)
from triton_ascend_bench.frozen.shim import is_hip  # Phase A': platform/math predicate (B-inert-shim)

@triton.jit
def fused_moe_router_tensorcore_kernel(a_ptr, b_ptr, topk_weights_ptr, topk_ids_ptr, bs, num_experts: tl.constexpr, topk: tl.constexpr, moe_softcapping: tl.constexpr, moe_renormalize: tl.constexpr, correction_bias_ptr, is_correction_bias: tl.constexpr, K: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, stride_am: tl.constexpr, stride_bn: tl.constexpr, dp_attn_workaround_flag: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    bs_mask = offs_m < bs
    offs_k = tl.arange(0, BLOCK_SIZE_K)[None, :]
    a_ptrs = a_ptr + (offs_m * stride_am + offs_k)
    offs_k = tl.arange(0, BLOCK_SIZE_K)[None, :]
    offs_n = tl.arange(0, BLOCK_SIZE_N)[:, None]
    expert_mask = offs_n < num_experts
    b_ptrs = b_ptr + (offs_n * stride_bn + offs_k)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K // BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=bs_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=expert_mask, other=0.0).to(tl.float32).T
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    if moe_softcapping == 0:
        logits_softcapped = acc
    else:
        logits_scaled = acc / moe_softcapping
        exped = tl.exp(2 * logits_scaled)
        logits_softcapped = (exped - 1) / (exped + 1) * moe_softcapping
    if is_correction_bias:
        bias = tl.load(correction_bias_ptr + tl.arange(0, BLOCK_SIZE_N)[None, :], mask=expert_mask.T, other=0.0)
        logits_softcapped = logits_softcapped + bias
    if dp_attn_workaround_flag:
        logits_softcapped = tl.where(logits_softcapped != logits_softcapped, -1000000000.0, logits_softcapped)
    arange_block_size_n = tl.arange(0, BLOCK_SIZE_N)[None, :]
    cond_top1 = arange_block_size_n < num_experts
    top1 = tl.argmax(tl.where(cond_top1, logits_softcapped, float('-inf')), axis=1)
    top1_v = tl.max(tl.where(cond_top1, logits_softcapped, float('-inf')), axis=1, keep_dims=True)
    top1_invsumexp = 1.0 / tl.sum(tl.where(cond_top1, tl.exp(logits_softcapped - top1_v), 0.0), axis=1)
    offs_top1 = pid * topk * BLOCK_SIZE_M + topk * tl.arange(0, BLOCK_SIZE_M)
    top1_mask = offs_top1 < bs * topk
    tl.store(topk_ids_ptr + offs_top1, top1, mask=top1_mask)
    tl.store(topk_weights_ptr + offs_top1, top1_invsumexp, mask=top1_mask)
    if topk == 2:
        cond_top2 = (arange_block_size_n < num_experts) & (arange_block_size_n != top1[:, None])
        top2 = tl.argmax(tl.where(cond_top2, logits_softcapped, float('-inf')), axis=1, keep_dims=True)
        top2_v = tl.sum(logits_softcapped * (arange_block_size_n == top2), axis=1, keep_dims=True)
        top2_invsumexp = tl.exp(top2_v - top1_v) * top1_invsumexp[:, None]
        offs_top2 = pid * topk * BLOCK_SIZE_M + topk * tl.arange(0, BLOCK_SIZE_M)[:, None] + 1
        top2_mask = offs_top2 < bs * topk
        tl.store(topk_ids_ptr + offs_top2, top2, mask=top2_mask)
        tl.store(topk_weights_ptr + offs_top2, top2_invsumexp, mask=top2_mask)

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
# (Same pattern as the passed tensorcore sibling L2_18_w8a8_triton_block_scaled_mm.)
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
