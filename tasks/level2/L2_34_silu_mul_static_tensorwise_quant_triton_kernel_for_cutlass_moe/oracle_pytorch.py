"""Oracle reference for L2_34_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe.

Tier: formula_exact_manual
Tolerance class: atomic_scatter
Category: moe

The wrapper under test is ``silu_and_mul_masked_post_per_tensor_quant_fwd``:

* ``input``  shape ``[E, T, 2*D]`` where the first half is ``gate`` and the
  second half is ``up``.
* ``output`` shape ``[E, T, D]`` with dtype ``torch.float8_e4m3fn``.
* ``masked_m[e]`` gives the number of valid tokens for expert ``e``.
* ``scale`` is either a single tensor-wise scale or one scale per expert.
"""
from __future__ import annotations

import torch


def _silu_mul_static_tensorwise_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    masked_m: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    half = input.shape[-1] // 2
    gate = input[..., :half].to(torch.float32)
    up = input[..., half:].to(torch.float32)

    gate = gate * torch.sigmoid(gate)
    if scale.numel() == 1:
        scaled = up * gate * (1.0 / scale.reshape(1, 1, 1).to(torch.float32))
    else:
        scaled = up * gate * (1.0 / scale.reshape(-1, 1, 1).to(torch.float32))

    out = output.clone()
    for expert_idx in range(input.shape[0]):
        valid_tokens = int(masked_m[expert_idx].item())
        if valid_tokens <= 0:
            continue
        out[expert_idx, :valid_tokens] = scaled[expert_idx, :valid_tokens].clamp(
            min=-torch.finfo(torch.float8_e4m3fn).max,
            max=torch.finfo(torch.float8_e4m3fn).max,
        ).to(torch.float8_e4m3fn)
    return out


def run(inputs: dict) -> dict:
    return {
        "output": _silu_mul_static_tensorwise_quant(
            inputs["input"],
            inputs["output"],
            inputs["masked_m"],
            inputs["scale"],
        )
    }
