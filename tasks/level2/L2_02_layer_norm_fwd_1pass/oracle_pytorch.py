"""Oracle reference for L2_02_layer_norm_fwd_1pass_kernel.

Tier: upstream_oracle_required
Tolerance class: vector_reduce
Category: normalization

Eager PyTorch reference authored from the kernel's documented operation
(no upstream code copy).

ORACLE-SCOPE: covers single-pass LayerNorm with affine (HAS_BIAS) AND the SiLU
gating branch (HAS_Z) with the wrapper default norm_before_gate=True (gate
applied AFTER normalisation). The non-default IS_RMS_NORM and gate-before-norm
paths are separate kernel modes a single task config cannot also drive.
"""
from __future__ import annotations

import torch


def _layer_norm_1pass(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
    z: torch.Tensor | None, eps: float,
) -> torch.Tensor:
    """Single-pass gated LayerNorm with affine, fp32 intermediate.

    Mirrors ``_layer_norm_fwd`` / ``_layer_norm_fwd_1pass_kernel`` (non-RMS,
    norm_before_gate=True): mean/var normalise along the last dim, scale +
    shift, then -- when a gating tensor ``z`` is present -- multiply by the
    SiLU gate ``z * sigmoid(z)``. This is the gate-after-norm branch
    (``if HAS_Z and NORM_BEFORE_GATE: y *= z * sigmoid(z)``).
    """
    x_f = x.to(torch.float32)
    mean = x_f.mean(dim=-1, keepdim=True)
    var = x_f.var(dim=-1, keepdim=True, unbiased=False)
    inv = torch.rsqrt(var + float(eps))
    out = (x_f - mean) * inv * weight.to(torch.float32)
    if bias is not None:
        out = out + bias.to(torch.float32)
    if z is not None:
        z_f = z.to(torch.float32)
        # SiLU gate z*sigmoid(z) expanded to low-level primitives (spec L195):
        # z/(1+exp(-z)) -- exp-stable, no overflow at the SiLU limits.
        out = out * (z_f / (1.0 + torch.exp(-z_f)))
    return out.to(x.dtype)


def run(inputs: dict) -> dict:
    return {
        "output": _layer_norm_1pass(
            inputs["x"], inputs["weight"], inputs.get("bias"),
            inputs.get("z"), float(inputs.get("eps", 1e-5)),
        )
    }
