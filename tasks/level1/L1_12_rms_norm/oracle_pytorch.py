"""Oracle reference for L1_12_rms_norm_kernel.

This oracle is tier-1
(`simple_pytorch_ok`) — the formula is well-defined and a straightforward
PyTorch implementation matches the GPU kernel semantics to within the
declared tolerance class.

Tier: simple_pytorch_ok
Tolerance class: vector_reduce  (see docs/reference/tolerance_policy.md)

Runs on CPU (or any device); does NOT depend on triton, CUDA, or NPU.
"""
from __future__ import annotations

import torch


def oracle_rms_norm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Reference RMS norm: y = x / sqrt(mean(x^2) + eps) * weight.

    Accumulates in float32 to match the kernel's internal precision, then
    casts back to input dtype.
    """
    assert weight.dim() == 1
    assert input.shape[-1] == weight.shape[0]

    orig_dtype = input.dtype
    orig_shape = input.shape

    x = input.reshape(-1, input.shape[-1]).to(torch.float32)
    w = weight.to(torch.float32)

    mean_sq = (x * x).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_sq + eps)
    y = x * inv_rms * w
    return y.to(orig_dtype).reshape(orig_shape)


def run(inputs: dict) -> dict:
    """Standard oracle entrypoint.

    Args:
        inputs: dict with keys `input`, `weight`, `eps`.

    Returns:
        dict with key `output`.
    """
    out = oracle_rms_norm(inputs["input"], inputs["weight"], inputs["eps"])
    return {"output": out}
