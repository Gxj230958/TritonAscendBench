"""Oracle reference for L2_15_fused_dual_residual_rmsnorm_kernel.

Tier: formula_exact_manual
Tolerance class: vector_reduce
Category: normalization

Independent eager PyTorch reference for the frozen upstream kernel
``fused_dual_residual_rmsnorm_kernel`` (SGLang
``sglang/srt/layers/elementwise.py``). The kernel and its host wrapper
``fused_dual_residual_rmsnorm(x, residual, weight1, weight2, eps)``
implement the fused identity documented in the upstream
``FusedDualResidualRMSNorm`` class:

    y   = RMSNorm2(RMSNorm1(x) + residual)
    mid = RMSNorm1(x) + residual

i.e. TWO sequential RMSNorms with a single residual add in between, NOT a
sum of two residuals followed by one RMSNorm. This file re-derives that
math in pure PyTorch from the kernel's documented contract (no upstream
code copy, never calls the kernel).
"""
from __future__ import annotations

import torch


def _rmsnorm(a: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim: a / sqrt(mean(a**2) + eps) * weight.

    Matches the kernel's per-row reduction
    ``rms = sqrt(sum(a*a) / hidden_dim + eps)`` computed in float32.
    """
    a32 = a.to(torch.float32)
    rms = torch.sqrt(a32.pow(2).mean(dim=-1, keepdim=True) + float(eps))
    return a32 / rms * weight.to(torch.float32)


def _fused_dual_residual_rms(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """y = RMSNorm2(RMSNorm1(x) + residual); mid = RMSNorm1(x) + residual.

    Mirrors the kernel's dtype handling: the first RMSNorm result is cast
    back to the residual dtype BEFORE the residual add (so ``mid`` carries
    the input dtype), then the second RMSNorm runs in float32 and its
    result is cast to the output (== input) dtype.
    """
    # RMSNorm1(x), cast to residual dtype before the add (kernel: a2r =
    # r + (a / rms * w1).to(r.dtype)).
    normed1 = _rmsnorm(x, weight1, eps).to(residual.dtype)
    mid = residual + normed1  # the second-stage residual / next-layer input

    # RMSNorm2 over the post-residual activation, cast to the output dtype
    # (kernel: output = a2r / rms2 * w2, stored as torch.empty_like(x)).
    out = _rmsnorm(mid, weight2, eps).to(x.dtype)
    return out, mid.to(x.dtype)


def run(inputs: dict) -> dict:
    out, mid = _fused_dual_residual_rms(
        inputs["x"],
        inputs["residual"],
        inputs["weight1"],
        inputs["weight2"],
        float(inputs.get("eps", 1e-6)),
    )
    return {"output": out, "mid": mid}
