"""Oracle reference for L1_16_silu_and_mul.

Tier: simple_pytorch_ok
Tolerance class: elementwise

The upstream kernel computes silu in fp32 then casts back to input dtype
before the multiply. This oracle mirrors that order.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def oracle_silu_and_mul(hidden_states: torch.Tensor) -> torch.Tensor:
    bs, in_hidden_dim = hidden_states.shape
    hidden_dim = in_hidden_dim // 2
    x1 = hidden_states[:, :hidden_dim]
    x3 = hidden_states[:, hidden_dim:]
    # silu in fp32 then cast back to input dtype, matching upstream
    silu_x1 = F.silu(x1.to(torch.float32)).to(hidden_states.dtype)
    return x3 * silu_x1


def run(inputs: dict) -> dict:
    return {"output": oracle_silu_and_mul(inputs["hidden_states"])}
