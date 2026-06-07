"""Oracle reference for L1_36_gemma_qkv_rmsnorm (_gemma_qkv_rmsnorm_kernel).

Tier: formula_exact_manual
Tolerance class: vector_reduce
Category: normalization

Independent eager reference for SGLang's Gemma3 fused per-head QKV RMSNorm.
Each attention head's vector is RMS-normalized independently:

  out = x * rsqrt(mean(x^2, over head_dim) + eps) * weight

Q heads are scaled by `q_weight`, K heads by `k_weight`, V heads by 1
(Gemma's V-norm has `with_scale=False`). Layout: q is [M, QH*head_dim],
k/v are [M, KVH*head_dim] with heads packed contiguously in the last dim;
q_weight/k_weight are [head_dim]. The kernel mutates q, k, v in place, so the
oracle returns the post-norm tensors under the same keys for comparison. The
reduction runs in fp32 (matching the kernel's `.to(tl.float32)` loads); no
upstream code is copied.
"""
from __future__ import annotations

import torch

EPS = 1e-6


def _rmsnorm_per_head(
    x: torch.Tensor, weight: torch.Tensor | None, head_dim: int,
) -> torch.Tensor:
    M, packed = x.shape
    num_heads = packed // head_dim
    xr = x.reshape(M, num_heads, head_dim).to(torch.float32)
    rrms = torch.rsqrt(xr.pow(2).mean(dim=-1, keepdim=True) + EPS)
    out = xr * rrms
    if weight is not None:
        out = out * weight.reshape(1, 1, head_dim).to(torch.float32)
    return out.reshape(M, packed).to(x.dtype)


def run(inputs: dict) -> dict:
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    head_dim = inputs["q_weight"].shape[-1]
    nq = _rmsnorm_per_head(q, inputs["q_weight"], head_dim)
    nk = _rmsnorm_per_head(k, inputs["k_weight"], head_dim)
    nv = _rmsnorm_per_head(v, None, head_dim)
    return {"q": nq, "k": nk, "v": nv}
