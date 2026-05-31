"""Oracle reference for L2_34_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe.

Tier: formula_exact_manual
Tolerance class: atomic_scatter
Category: moe

Independent eager-PyTorch reference for the frozen upstream GPU triton
kernel ``silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe``
(SGLang ``sglang/srt/layers/moe/ep_moe/kernels.py``, commit
409d350fb6f6a1e7c7546e39028f811092a8e489). This file authors the math from
the kernel's documented contract; it never imports or calls the kernel.

Upstream host wrapper ``silu_mul_static_tensorwise_quant_for_cutlass_moe`` is
invoked from ``cutlass_w4a8_moe.py`` as::

    silu_mul_static_tensorwise_quant_for_cutlass_moe(
        c1, intermediate_q, a2_scale.float(), expert_offsets[-1:], m * topk, n
    )

so the buffer layout is:

* ``input``  (``c1``):  ``[num_tokens, 2 * intermediate_size]``, the grouped-GEMM
  output where the first half along the last dim is ``gate`` and the second
  half is ``up`` (NOT interleaved).
* ``output`` (``intermediate_q``): ``[num_tokens, intermediate_size]``,
  dtype ``torch.float8_e4m3fn``; the candidate writes here in place.
* ``scale``  (``a2_scale.float()``): a single scalar fp32 tensor.
* ``num_tokens_tensor`` (``expert_offsets[-1:]``): a 1-element int tensor whose
  value is the active token count (``m * topk``); here it equals ``num_tokens``.

Kernel math (per output element):
    gate = input[token, 0:isz];  up = input[token, isz:2*isz]
    s    = 1.0 / scale
    out  = silu(gate) * up * s   = gate * sigmoid(gate) * up * s
then cast to the output dtype (``float8_e4m3fn``). The upstream kernel performs
a bare ``.to(OutDtype)`` with no clamp.
"""
from __future__ import annotations

import torch


def _silu_mul_static_tensorwise_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    scale: torch.Tensor,
    num_tokens_tensor: torch.Tensor,
) -> torch.Tensor:
    intermediate_size = input.shape[-1] // 2
    gate = input[..., :intermediate_size].to(torch.float32)
    up = input[..., intermediate_size:].to(torch.float32)

    inv_scale = 1.0 / scale.reshape(()).to(torch.float32)
    silu = gate * torch.sigmoid(gate)
    z = silu * up * inv_scale

    out = output.clone()
    num_tokens = int(num_tokens_tensor.reshape(-1)[0].item())
    num_tokens = max(0, min(num_tokens, input.shape[0]))
    if num_tokens > 0:
        out[:num_tokens] = z[:num_tokens].to(output.dtype)
    return out


def run(inputs: dict) -> dict:
    return {
        "output": _silu_mul_static_tensorwise_quant(
            inputs["input"],
            inputs["output"],
            inputs["scale"],
            inputs["num_tokens_tensor"],
        )
    }
