"""Oracle reference for L1_33_pre_reorder_triton_kernel_for_cutlass_moe.

Tier: formula_exact_manual
Tolerance class: atomic_scatter
Category: moe

Eager PyTorch reference authored from the kernel's documented operation
(no upstream code copy).
"""
from __future__ import annotations

import torch


def _pre_reorder_cutlass_moe(
    x: torch.Tensor, src2dst: torch.Tensor, topk_ids: torch.Tensor,
    a1_scales: torch.Tensor, num_local_experts: int, topk: int,
) -> torch.Tensor:
    """Pre-reorder activations into per-expert slots for the cutlass grouped
    GEMM, gated by expert id and scaled by the activation scale.

    For token ``i`` and slot ``j`` (flat index ``i*topk + j``): when
    ``topk_ids[i*topk+j] != num_local_experts`` (a real expert, not the padding
    sentinel), write ``x[i] * (1 / a1_scales[0])`` into
    ``gateup_input[src2dst[i*topk+j]]``. Padding slots are skipped; their
    destination rows stay zero. ``a1_scales[0]`` is the per-tensor activation
    scale (the kernel divides by it).
    """
    n_tokens, hidden = x.shape
    out = x.new_zeros((n_tokens * topk, hidden), dtype=torch.float32)
    a1_scale = float((1.0 / a1_scales.reshape(-1)[0].to(torch.float32)).item())
    s2d = src2dst.to(torch.long).reshape(-1)
    tids = topk_ids.to(torch.long).reshape(-1)
    xf = x.to(torch.float32)
    for i in range(n_tokens):
        scaled = xf[i] * a1_scale
        for j in range(topk):
            slot = i * topk + j
            if int(tids[slot]) != num_local_experts:
                out[int(s2d[slot])] = scaled
    return out.to(x.dtype)


def run(inputs: dict) -> dict:
    return {
        "gateup_input": _pre_reorder_cutlass_moe(
            inputs["input"], inputs["src2dst"], inputs["topk_ids"],
            inputs["a1_scales"], int(inputs["num_local_experts"]),
            int(inputs["topk"]),
        )
    }
