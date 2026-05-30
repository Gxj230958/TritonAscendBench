"""Oracle reference for L2_36_post_reorder_triton_kernel_for_cutlass_moe.

Tier: formula_exact_manual
Tolerance class: atomic_scatter
Category: moe

Eager PyTorch reference authored from the kernel's documented operation
(no upstream code copy).
"""
from __future__ import annotations

import torch


def _post_reorder_cutlass_moe(
    down_output: torch.Tensor, src2dst: torch.Tensor, topk_ids: torch.Tensor,
    topk_weights: torch.Tensor, num_local_experts: int,
    routed_scaling_factor: float, topk: int,
) -> torch.Tensor:
    """Gather the cutlass grouped-GEMM expert outputs back to token order,
    weighting by the router probabilities and skipping padding experts.

    For output token ``i``: accumulate over slot ``j`` (flat index
    ``i*topk + j``) only when ``topk_ids[i*topk+j] != num_local_experts``
    (a real expert): ``acc += down_output[src2dst[i*topk+j]] *
    topk_weights[i*topk+j]``. The token output is ``acc *
    routed_scaling_factor``; a token with all slots padded stays zero.
    """
    n_tokens = src2dst.numel() // topk
    hidden = down_output.shape[1]
    out = down_output.new_zeros((n_tokens, hidden), dtype=torch.float32)
    s2d = src2dst.to(torch.long).reshape(-1)
    tids = topk_ids.to(torch.long).reshape(-1)
    tw = topk_weights.to(torch.float32).reshape(-1)
    do = down_output.to(torch.float32)
    for i in range(n_tokens):
        acc = torch.zeros(hidden, dtype=torch.float32, device=do.device)
        for j in range(topk):
            slot = i * topk + j
            if int(tids[slot]) != num_local_experts:
                acc = acc + do[int(s2d[slot])] * float(tw[slot])
        out[i] = acc * routed_scaling_factor
    return out.to(down_output.dtype)


def run(inputs: dict) -> dict:
    return {
        "output": _post_reorder_cutlass_moe(
            inputs["down_output"], inputs["src2dst"], inputs["topk_ids"],
            inputs["topk_weights"], int(inputs["num_local_experts"]),
            float(inputs["routed_scaling_factor"]), int(inputs["topk"]),
        )
    }
