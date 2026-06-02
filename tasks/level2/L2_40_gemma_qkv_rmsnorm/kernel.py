"""Task L2_40_gemma_qkv_rmsnorm — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_gemma_qkv_rmsnorm_kernel`
Source path: `sglang/srt/layers/gemma4_fused_ops.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): 169a498ad0326b55802e7382ce66102317fb3d66808d57d3daf601f741497c53
Frozen by: scripts/phase_a_freeze.py
"""
'Fused triton kernels for Gemma4 decoder layer operations.\n\nFuses standard RMSNorm + residual-add (+ optional scalar multiply) into\na single kernel pass to reduce kernel launch overhead.\n'
from typing import Optional
import torch
import triton
import triton.language as tl

@triton.jit
def _gemma_qkv_rmsnorm_kernel(Q_ptr, K_ptr, V_ptr, Q_w_ptr, K_w_ptr, stride_q_m, stride_k_m, stride_v_m, NUM_Q_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, eps, HAS_KV: tl.constexpr, BLOCK: tl.constexpr):
    """Per-token fused RMSNorm of Q (with q_w), K (with k_w), V (no scale).

    Layout assumption: each tensor's last dim packs (num_heads, head_dim) contiguously
    so per-head offset is `h * HEAD_DIM`. The token (M) stride is taken from
    stride_*_m so the kernel works on strided views (e.g. slices of a larger
    qkv buffer produced by `qkv.split`) without requiring `.contiguous()` copies.
    V uses `weight=ones` semantics so the multiply-by-weight is omitted.
    """
    m = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < HEAD_DIM
    qw = tl.load(Q_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    for h in tl.static_range(NUM_Q_HEADS):
        off = m * stride_q_m + h * HEAD_DIM + cols
        x = tl.load(Q_ptr + off, mask=mask, other=0.0).to(tl.float32)
        rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
        out = x * rrms * qw
        tl.store(Q_ptr + off, out.to(Q_ptr.dtype.element_ty), mask=mask)
    if HAS_KV:
        kw = tl.load(K_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        for h in tl.static_range(NUM_KV_HEADS):
            off = m * stride_k_m + h * HEAD_DIM + cols
            x = tl.load(K_ptr + off, mask=mask, other=0.0).to(tl.float32)
            rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
            out = x * rrms * kw
            tl.store(K_ptr + off, out.to(K_ptr.dtype.element_ty), mask=mask)
        for h in tl.static_range(NUM_KV_HEADS):
            off = m * stride_v_m + h * HEAD_DIM + cols
            x = tl.load(V_ptr + off, mask=mask, other=0.0).to(tl.float32)
            rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
            out = x * rrms
            tl.store(V_ptr + off, out.to(V_ptr.dtype.element_ty), mask=mask)

import triton


# Authored launcher: drives the in-module @triton.jit `_gemma_qkv_rmsnorm_kernel`
# (SGLang 409d350, sglang/srt/layers/gemma4_fused_ops.py) exactly as its host
# wrapper `gemma_qkv_rmsnorm` does on the full Q+K+V path (HAS_KV=True). The
# wrapper pulls a stray `from sglang.srt.utils import is_cuda` into the module;
# this launcher bypasses the wrapper and drives the @triton.jit kernel directly
# so the graded surface is just the kernel + plain triton.
#
# Math (matches oracle_pytorch): per token, per head, RMS-normalize each head's
# vector -- out = x * rsqrt(mean(x^2)+eps) * w -- with Q scaled by q_weight,
# K by k_weight, V by 1. The kernel mutates q, k, v in place; output keys mirror
# the input names so mutated-input comparison sees the post-norm tensors.
EPS = 1e-6


def run(inputs: dict) -> dict:
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    q_weight = inputs["q_weight"]
    k_weight = inputs["k_weight"]

    M = q.shape[0]
    head_dim = q_weight.shape[-1]
    num_q_heads = q.shape[1] // head_dim
    num_kv_heads = k.shape[1] // head_dim
    BLOCK = triton.next_power_of_2(head_dim)

    _gemma_qkv_rmsnorm_kernel[(M,)](
        q,
        k,
        v,
        q_weight,
        k_weight,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        eps=EPS,
        HAS_KV=True,
        BLOCK=BLOCK,
    )

    return {"q": q, "k": k, "v": v}
