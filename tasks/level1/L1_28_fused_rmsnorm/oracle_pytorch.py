"""Oracle reference for L1_28_fused_rmsnorm.

Tier: formula_exact_manual
Tolerance class: vector_reduce
Category: normalization

Independent eager PyTorch reference for the frozen upstream
``fused_rmsnorm_kernel`` (sglang/srt/layers/elementwise.py, commit
409d350). The kernel and its host wrapper ``fused_rmsnorm(x, weight,
eps)`` compute a plain per-row RMSNorm scaled by ``weight`` -- there is
NO residual add and NO second weight (those belong to the sibling
``fused_dual_residual_rmsnorm_kernel``). One row per program; the
reduction is over the last (hidden) dimension.

Upstream math (kernel.py:247-266):
    a   = x.to(float32)
    rms = sqrt(sum(a*a, dim=-1) / hidden_dim + eps)
    out = a / rms * weight          # weight upcast to float32
    store(out)                       # implicit cast back to x.dtype

This is authored from the operation's math, not copied from the kernel.
"""
from __future__ import annotations

import torch


def _fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-row RMSNorm with a learnable scale, matching fused_rmsnorm_kernel.

    rms = sqrt(mean(x^2, last_dim) + eps); out = x / rms * weight.
    All reduction arithmetic is done in float32, then cast back to the
    input dtype (the kernel stores into ``torch.empty_like(x)``).
    """
    a = x.to(torch.float32)
    hidden_dim = a.shape[-1]
    rms = torch.sqrt(a.pow(2).sum(dim=-1, keepdim=True) / hidden_dim + float(eps))
    out = a / rms * weight.to(torch.float32)
    return out.to(x.dtype)


def run(inputs: dict) -> dict:
    out = _fused_rmsnorm(
        inputs["x"], inputs["weight"], float(inputs.get("eps", 1e-6))
    )
    return {"output": out}
