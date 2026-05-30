"""Oracle reference for L2_26_fwd_kernel_ep_scatter_1.

Tier: formula_exact_manual
Tolerance class: atomic_scatter
Category: moe

Independent eager-PyTorch reference authored from the documented math of
the frozen upstream entrypoint ``_fwd_kernel_ep_scatter_1`` (vLLM
vllm/model_executor/layers/fused_moe/deep_gemm_utils.py, commit 18f6bf5).
No upstream code copy and no triton.

What the kernel actually computes (NOT a row scatter): given the per-expert
token counts ``num_recv_tokens_per_expert[num_experts]`` it builds the
DeepGEMM contiguous-group layout:

  * ``expert_start_loc[e]`` = exclusive prefix sum over experts of
    ``round_up_128(num_recv_tokens_per_expert)`` -- i.e. each expert's
    region is padded up to a multiple of 128 (BLOCK_E) and laid out
    back-to-back. (Kernel lines: ``tokens_per_expert = round_up_128(...)``;
    ``cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert``;
    ``tl.store(expert_start_loc + cur_expert, cur_expert_start)``.)
  * ``m_indices[M_sum]`` = for each expert ``e`` the rows
    ``[expert_start_loc[e] : expert_start_loc[e] + num_recv_tokens_per_expert[e])``
    are filled with the value ``e``; every other (padding) row keeps the
    caller-supplied sentinel ``-1`` so DeepGEMM skips it. (Kernel loop over
    ``cur_expert_token_num`` storing ``cur_expert`` into ``m_indices``.)

``M_sum = sum_e round_up_128(num_recv_tokens_per_expert[e])``.
"""
from __future__ import annotations

import torch


def _round_up_128(x: torch.Tensor) -> torch.Tensor:
    y = 128
    return ((x + y - 1) // y) * y


def _ep_scatter_1(
    num_recv_tokens_per_expert: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build expert_start_loc + m_indices exactly as the kernel does."""
    counts = num_recv_tokens_per_expert.to(torch.int64)
    num_experts = counts.shape[0]

    padded = _round_up_128(counts)                       # [num_experts]
    # exclusive prefix sum (cumsum - self), matching the kernel.
    expert_start_loc = (torch.cumsum(padded, 0) - padded).to(torch.int32)
    m_sum = int(padded.sum().item())

    m_indices = torch.full(
        (m_sum,), -1, dtype=torch.int32, device=counts.device
    )
    starts = expert_start_loc.to(torch.int64)
    for e in range(num_experts):
        n = int(counts[e].item())
        if n > 0:
            s = int(starts[e].item())
            m_indices[s : s + n] = e
    return expert_start_loc, m_indices


def run(inputs: dict) -> dict:
    expert_start_loc, m_indices = _ep_scatter_1(
        inputs["num_recv_tokens_per_expert"]
    )
    return {"expert_start_loc": expert_start_loc, "m_indices": m_indices}
