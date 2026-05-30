"""Oracle reference for L2_32_experts_combine_kernel.

Tier: simple_pytorch_ok
Tolerance class: atomic_scatter
Category: moe

Independent eager-PyTorch reference for the upstream SGLang
`experts_combine_kernel` (sglang/srt/layers/elementwise.py, commit
409d350fb6f6a1e7c7546e39028f811092a8e489). Authored from the kernel's
documented math, NOT a copy of upstream code and never calling the kernel.

Kernel contract (per output row pid):
    moe_x = sum_{k < combine_k} moe_hidden_states[pid, k, :]
    combined = (moe_x + mlp_hidden_states[pid, :]) / sqrt(2)
where moe_hidden_states is (bs, combine_k, hidden_dim) when 3-D, or
(bs, hidden_dim) "pre-combined" (combine_k == 1) when 2-D. mlp_hidden_states
is (bs, hidden_dim) and fixes the output shape/dtype. The 1/sqrt(2) factor
is the kernel's literal divisor 1.4142135623730951.
"""
from __future__ import annotations

import math

import torch

_INV_SQRT2_DIVISOR = 1.4142135623730951  # kernel literal: combined / sqrt(2)


def _experts_combine(
    moe_hidden: torch.Tensor, mlp_hidden: torch.Tensor,
) -> torch.Tensor:
    """Residual combine of MoE (summed over the combine_k axis) and the
    shared-MLP branch, scaled by 1/sqrt(2) -- matching the upstream kernel."""
    if moe_hidden.dim() == 3:
        # (bs, combine_k, hidden_dim) -> sum over the combine_k axis.
        moe_x = moe_hidden.to(torch.float32).sum(dim=1)
    else:
        # (bs, hidden_dim): combine_k == 1, sum over a singleton axis = identity.
        moe_x = moe_hidden.to(torch.float32)

    combined = (moe_x + mlp_hidden.to(torch.float32)) / _INV_SQRT2_DIVISOR
    return combined.to(mlp_hidden.dtype)


def run(inputs: dict) -> dict:
    return {"output": _experts_combine(
        inputs["moe_hidden_states"], inputs["mlp_hidden_states"])}
