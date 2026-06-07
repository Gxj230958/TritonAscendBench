"""Oracle reference for L1_21_create_flashmla_kv_indices_triton.

Tier: formula_exact_manual
Tolerance class: attention_flash
Category: attention

Independent eager-PyTorch reference authored from the frozen upstream
``create_flashmla_kv_indices_triton`` @triton.jit (kernel.py). The kernel
builds the FlashMLA paged-KV *page-index* table: for each request it walks
the ``req_to_token`` slot table at ``page_size`` stride and emits the page id
(``token_slot // page_size``) for every page the request occupies.

This file re-derives that integer math directly (gather + floor-divide +
per-request page mask); it never calls the kernel and never echoes inputs.

Upstream contract (kernel.py::create_flashmla_kv_indices_triton):
  grid = (n_reqs,)
  for request pid:
      req_pool_index = req_pool_indices[pid]
      kv_start       = kv_start_idx[pid] if kv_start_idx else 0
      seq_len        = page_kernel_lens[pid]
      num_paged      = cdiv(seq_len, PAGED_SIZE)
      for page p in [0, num_paged):
          tok  = req_to_token[req_pool_index, kv_start + p * PAGED_SIZE]
          kv_indices[pid, p] = tok // PAGED_SIZE
  Page slots p >= num_paged are left at their pre-allocated value (0 here).
The output is the dense ``[n_reqs, kv_indices_stride]`` page-index table
(NO indptr; that is the *flashinfer* sibling's contract, not FlashMLA's).
"""
from __future__ import annotations

import torch


def _create_flashmla_kv_indices(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    page_kernel_lens: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Build the FlashMLA paged-KV page-index table.

    Args:
        req_to_token: [max_batch, max_context_len] token-slot table.
        req_pool_indices: [n_reqs] physical pool row per request.
        page_kernel_lens: [n_reqs] sequence length per request.
        page_size: page granularity (the kernel's PAGED_SIZE constexpr).

    Returns:
        [n_reqs, kv_indices_stride] int32 page-index table, where
        kv_indices_stride = cdiv(max_context_len, page_size).
    """
    n_reqs = req_pool_indices.shape[0]
    max_ctx = req_to_token.shape[1]
    device = req_to_token.device

    # kv_start_idx is None in the FlashMLA decode call site -> kv_start = 0.
    stride = (max_ctx + page_size - 1) // page_size  # kv_indices_ptr_stride
    counts = (page_kernel_lens.to(torch.long) + page_size - 1) // page_size

    out = torch.zeros((n_reqs, stride), dtype=torch.int32, device=device)

    page_ids = torch.arange(stride, device=device)  # [P]
    tok_off = page_ids * page_size                  # token offset per page
    # Clamp lookup column to the valid req_to_token width; masked-out pages
    # are overwritten with the pre-allocated zero below, so the clamp value
    # is irrelevant numerically.
    gather_col = torch.clamp(tok_off, max=max_ctx - 1)

    rows = req_to_token[req_pool_indices.to(torch.long)]  # [n_reqs, max_ctx]
    gathered = rows[:, gather_col]                        # [n_reqs, P]
    vals = torch.div(gathered, page_size, rounding_mode="floor").to(torch.int32)

    page_mask = page_ids.unsqueeze(0) < counts.unsqueeze(1)  # [n_reqs, P]
    out = torch.where(page_mask, vals, out)
    return out


def run(inputs: dict) -> dict:
    kv_indices = _create_flashmla_kv_indices(
        inputs["req_to_token"],
        inputs["req_pool_indices"],
        inputs["page_kernel_lens"],
        int(inputs["page_size"]),
    )
    return {"kv_indices": kv_indices}
