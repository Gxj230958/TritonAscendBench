"""Oracle reference for L1_36_compute_seg_indptr_triton_kernel.

Tier: formula_exact_manual
Tolerance class: atomic_scatter
Category: moe

The kernel under test is the GPU triton implementation vendored in
kernel.py; this file authors an eager PyTorch reference from the
operation's documented mathematical definition (no upstream code copy,
never calls the kernel).

The named @triton.jit kernel is ``compute_seg_indptr_triton_kernel``
(kernel.py line 199), launched by the ``moe_ep_deepgemm_preprocess``
host wrapper with grid ``(num_local_experts + 1,)``::

    @triton.jit
    def compute_seg_indptr_triton_kernel(reorder_topk_ids, seg_indptr, num_toks):
        expert_id_minus_1 = tl.program_id(0) - 1
        low, high, target_location = 0, num_toks - 1, -1
        while low <= high:                       # binary search over the
            mid = (low + high) // 2              # SORTED reorder_topk_ids
            if tl.load(reorder_topk_ids + mid) > expert_id_minus_1:
                high = mid - 1
            else:
                low = mid + 1
                target_location = mid
        tl.store(seg_indptr + expert_id_minus_1 + 1, target_location + 1)

For program id ``e`` (in ``[0, num_local_experts]``) the kernel finds the
last index whose value is ``<= e - 1`` and stores ``target_location + 1``
into ``seg_indptr[e]``. That is exactly the number of sorted entries
strictly less than ``e``, i.e. ``searchsorted(reorder_topk_ids, e,
side='left')`` — the RAW segment indptr. The deepgemm caller consumes it
without the ``num_minus_one`` subtraction the *other* (deepep) wrapper
applies, so the reference returns the raw counts.
"""
from __future__ import annotations

import torch


def _compute_seg_indptr(
    topk_ids: torch.Tensor, num_local_experts: int
) -> torch.Tensor:
    """Raw expert segment indptr over the sorted token->expert ids.

    ``seg_indptr[e] = #{ reorder_topk_ids < e }`` for ``e`` in
    ``[0, num_local_experts]`` — equivalently a left-side searchsorted of
    the sorted ids against ``[0, 1, ..., num_local_experts]``. Independent
    of (and never invoking) the triton kernel.
    """
    reorder_topk_ids, _ = torch.sort(topk_ids.view(-1).to(torch.int64), stable=True)
    expert_ids = torch.arange(
        num_local_experts + 1,
        device=topk_ids.device,
        dtype=reorder_topk_ids.dtype,
    )
    seg_indptr = torch.searchsorted(
        reorder_topk_ids, expert_ids, right=False
    ).to(torch.int64)
    return seg_indptr


def run(inputs: dict) -> dict:
    return {
        "seg_indptr": _compute_seg_indptr(
            inputs["topk_ids"], int(inputs["num_local_experts"])
        )
    }
