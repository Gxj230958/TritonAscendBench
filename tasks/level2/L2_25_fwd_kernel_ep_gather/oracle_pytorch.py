"""Oracle reference for L2_25_fwd_kernel_ep_gather.

Tier: upstream_oracle_required
Tolerance class: atomic_scatter
Category: moe

Independent eager-PyTorch reference for the frozen upstream GPU triton
kernel ``_fwd_kernel_ep_gather`` (vLLM
``vllm/model_executor/layers/fused_moe/deep_gemm_utils.py``, commit
18f6bf5; ported from lightllm ``deepep_scatter_gather.py``). It is
authored from the kernel's documented math contract; it NEVER calls the
kernel and never trivially echoes inputs.

Kernel contract (expert_map=None / HAS_EXPERT_MAP=False path):

    for each output token row t and feature d:
        out[t, d] = sum over k in [0, topk_num):
            1[recv_topk_ids[t, k] >= 0]
              * recv_topk_weight[t, k]
              * input_tensor[input_index[t, k], d]

    accumulation is in float32, then cast to output dtype. Entries with
    recv_topk_ids[t, k] < 0 (i.e. -1, the invalid/padded marker) are
    skipped entirely: their input_index[t, k] is never dereferenced.

This is an unpermute-and-weighted-reduce, NOT a plain row gather: each
output row sums topk weighted source rows of ``input_tensor`` selected by
``input_index`` and masked by the sign of ``recv_topk_ids``.

ORACLE-SCOPE (tier=upstream_oracle_required): covers the expert_map=None
path; NOT covered: HAS_EXPERT_MAP expert-id remap.
"""
from __future__ import annotations

import torch


def _ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
) -> torch.Tensor:
    """Topk weighted unpermute-and-reduce (expert_map=None path).

    Args:
        input_tensor: [M, D] source rows (the permuted grouped-gemm output).
        recv_topk_ids: [T, K] signed expert ids; values < 0 mark invalid
            (skipped) slots.
        recv_topk_weight: [T, K] per-slot combine weights.
        input_index: [T, K] source-row indices into ``input_tensor`` rows.

    Returns:
        [T, D] tensor, ``input_tensor.dtype``: per-token weighted sum of the
        valid topk source rows.
    """
    M, D = input_tensor.shape
    T, K = recv_topk_ids.shape

    out_dtype = input_tensor.dtype
    acc = torch.zeros((T, D), dtype=torch.float32, device=input_tensor.device)

    valid = recv_topk_ids >= 0  # [T, K] bool mask
    # Clamp source indices into range so the masked-out (invalid) slots,
    # whose input_index value is irrelevant, never trigger an OOB gather.
    # The mask below zeroes their contribution exactly as the kernel's
    # `if expert_id >= 0` guard skips the load.
    src = input_index.to(torch.long).clamp_(0, M - 1)  # [T, K]

    src_f32 = input_tensor.to(torch.float32)
    w = recv_topk_weight.to(torch.float32) * valid.to(torch.float32)  # [T, K]

    for k in range(K):
        gathered = src_f32.index_select(0, src[:, k])  # [T, D]
        acc += gathered * w[:, k].unsqueeze(1)

    return acc.to(out_dtype).contiguous()


def run(inputs: dict) -> dict:
    return {
        "output": _ep_gather(
            inputs["input_tensor"],
            inputs["recv_topk_ids"],
            inputs["recv_topk_weight"],
            inputs["input_index"],
        )
    }
