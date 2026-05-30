"""Oracle reference for L1_32_create_flashinfer_kv_indices_triton.

Tier: formula_exact_manual
Tolerance class: attention_flash
Category: attention

Independent eager-PyTorch reference for the FlashInfer paged-KV index
*gather*, authored from the frozen upstream @triton.jit contract
(sglang/srt/layers/attention/utils.py::create_flashinfer_kv_indices_triton,
SGLang commit 409d350fb6f6a1e7c7546e39028f811092a8e489) and its canonical
host call site (sglang/srt/layers/attention/flashinfer_backend.py:1129).

This does NOT call the kernel and does NOT copy upstream code. The kernel
is a flat gather: for each request ``pid`` it copies the token-location
row ``req_to_token[req_pool_indices[pid], kv_start : kv_start + lens[pid]]``
into ``kv_indices[kv_indptr[pid] : ...]``. With the normal (non-sliding,
``kv_start_idx is None``) host path, ``kv_start == 0`` and
``kv_indptr[0] == 0``, ``kv_indptr[1:bs+1] == cumsum(lens)``. There is no
page_size, no ceil-div, and no [E, N] block table -- those belong to the
sibling create_flashmla_kv_indices_triton (L1_33), not this kernel.
"""
from __future__ import annotations

import torch


def _create_flashinfer_kv_indices(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashInfer paged-KV index gather (kv_start_idx is None path).

    indptr[0]=0, indptr[r+1]=indptr[r]+seq_lens[r];
    indices = concat_r req_to_token[req_pool_indices[r], :seq_lens[r]].
    """
    bs = req_pool_indices.shape[0]
    lens = seq_lens.to(torch.long)
    indptr = torch.zeros(bs + 1, dtype=torch.int32, device=req_to_token.device)
    indptr[1:] = torch.cumsum(lens, 0).to(torch.int32)
    pool = req_pool_indices.to(torch.long)
    parts = [
        req_to_token[int(pool[r]), : int(lens[r])].to(torch.int32)
        for r in range(bs)
    ]
    if parts:
        indices = torch.cat(parts)
    else:
        indices = req_to_token.new_zeros(0, dtype=torch.int32)
    return indptr, indices


def run(inputs: dict) -> dict:
    indptr, indices = _create_flashinfer_kv_indices(
        inputs["req_to_token"],
        inputs["req_pool_indices"],
        inputs["seq_lens"],
    )
    return {"kv_indptr": indptr, "kv_indices": indices}
