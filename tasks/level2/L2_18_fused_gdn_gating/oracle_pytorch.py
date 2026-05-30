"""Oracle reference for L2_18_fused_gdn_gating.

Tier: formula_exact_manual
Tolerance class: recurrent_chunk
Category: ssm

Independent eager-PyTorch reference that exactly reproduces the upstream
host wrapper ``fused_gdn_gating`` (and its ``@triton.jit``
``fused_gdn_gating_kernel``) vendored in kernel.py. The upstream kernel is
the GROUND TRUTH; this file is authored from its math/contract, never by
calling it.

Upstream contract (vLLM, gdn_linear_attn.py::fused_gdn_gating):
  batch, num_heads = a.shape          # a, b are [B, H]; A_log, dt_bias are [H]
  g          -> empty(1, batch, num_heads, float32)
  beta_output-> empty(1, batch, num_heads, dtype=b.dtype)
  per element (head h broadcast over batch):
    x          = a.float() + dt_bias.float()
    softplus_x = where(beta*x <= threshold,
                       (1/beta)*log(1 + exp(beta*x)),
                       x)
    g          = -exp(A_log.float()) * softplus_x          # stays float32
    beta_output= sigmoid(b.float())                        # cast to b.dtype
  returns (g, beta_output)
"""
from __future__ import annotations

import torch


def _fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Faithful reproduction of the upstream fused GDN gating kernel.

    A_log / dt_bias are per-head ``[H]`` and broadcast over the batch dim;
    a / b are ``[B, H]``. Returns ``(g, beta_output)`` each ``[1, B, H]``.
    """
    a_f = a.to(torch.float32)
    dt_bias_f = dt_bias.to(torch.float32)
    A_log_f = A_log.to(torch.float32)

    # x = a + dt_bias (dt_bias is [H], broadcasts over the batch rows of a).
    x = a_f + dt_bias_f
    # softplus_x mirrors the kernel's explicit op sequence (tl.log(1 + tl.exp))
    # with the same beta-scaled overflow guard, NOT F.softplus / log1p.
    softplus_x = torch.where(
        beta * x <= threshold,
        (1.0 / beta) * torch.log(1 + torch.exp(beta * x)),
        x,
    )
    # A_log is [H], broadcasts over the batch rows; g buffer is float32.
    g = (-torch.exp(A_log_f) * softplus_x).unsqueeze(0)

    beta_output = torch.sigmoid(b.to(torch.float32)).to(b.dtype).unsqueeze(0)
    return g, beta_output


def run(inputs: dict) -> dict:
    g, beta_output = _fused_gdn_gating(
        inputs["A_log"],
        inputs["a"],
        inputs["b"],
        inputs["dt_bias"],
        float(inputs.get("beta", 1.0)),
        float(inputs.get("threshold", 20.0)),
    )
    return {"g": g, "beta_output": beta_output}
