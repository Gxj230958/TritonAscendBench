"""Oracle reference for L1_09_quantize_k_cache_fast_kernel.

Tier: formula_exact_manual
Tolerance class: vector_reduce
Category: quantization

Independent eager-PyTorch reference for the SGLang
`_quantize_k_cache_fast_kernel` (sglang/srt/layers/attention/nsa/
quant_k_cache.py, commit 409d350). Authored from the kernel's
mathematical definition; never calls the kernel.

Upstream kernel math (the ground truth), per nope group of GROUP_SIZE=128
contiguous columns (DIM_NOPE=512 -> NUM_NOPE_BLOCKS=4):

    y     = k_nope_group.to(fp32)         # loaded from bf16
    y_s   = max(|y|) / FP8_MAX            # per-group scale (no eps; upstream
                                          #   deliberately drops the eps guard)
    y_inv = 1.0 / y_s                     # kernel multiplies by the reciprocal
    y_q   = clamp(y * y_inv, FP8_MIN, FP8_MAX).to(float8_e4m3fn)

The rope half is a pure passthrough copy (dtype preserved, bf16). The
kernel uses FP8_MIN/FP8_MAX = finfo(float8_e4m3fn).min/.max (= -448 / 448).
"""
from __future__ import annotations

import torch

_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0


def _quantize_k_cache(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    group_size: int,
    fp8_max: float = _FP8_MAX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-group symmetric FP8 quant of the nope part; rope passed through.

    Returns (nope_q float8_e4m3fn, nope_s fp32, rope passthrough).
    """
    num_tokens, dim_nope = k_nope.shape
    gs = int(group_size)
    fp8_min = -float(fp8_max)
    fp8_max = float(fp8_max)

    # Match the kernel's bf16 -> fp32 load.
    xv = k_nope.to(torch.float32).reshape(num_tokens, dim_nope // gs, gs)
    # Per-group scale: amax / FP8_MAX (no eps -- upstream removed it).
    s = xv.abs().amax(dim=-1, keepdim=True) / fp8_max
    # Kernel computes y_s_inv = 1/y_s then y * y_s_inv (reciprocal-multiply,
    # not a true divide); mirror that arithmetic order for bit-faithfulness.
    q = (xv * (1.0 / s)).clamp(fp8_min, fp8_max)
    q_fp8 = q.reshape(num_tokens, dim_nope).to(torch.float8_e4m3fn)
    nope_s = s.squeeze(-1).reshape(num_tokens, dim_nope // gs)
    # Rope passthrough: copy unchanged, preserving the bf16 dtype/contract.
    rope = k_rope.clone()
    return q_fp8, nope_s, rope


def run(inputs: dict) -> dict:
    nq, ns, rope = _quantize_k_cache(
        inputs["k_nope"],
        inputs["k_rope"],
        int(inputs["group_size"]),
        float(inputs.get("fp8_max", _FP8_MAX)),
    )
    return {"nope_q": nq, "nope_s": ns, "rope": rope}
