"""Oracle reference for L3_06_fwd_grouped_kernel_stage1.

Tier: formula_exact_manual
Tolerance class: attention_flash
Category: other

Authored PyTorch reference (no upstream code copy) for the grouped
(GQA) decode-attention STAGE-1 kernel `_fwd_grouped_kernel_stage1`
(vLLM vllm/v1/attention/ops/triton_decode_attention.py). The kernel is
the first of the two flash-decoding passes: it splits each request's
cached context into NUM_KV_SPLITS contiguous chunks and emits, per
(batch, query-head, split), the chunk-local online-softmax attention
context PLUS its log-sum-exp. Stage 2 (a separate kernel) later merges
the splits; this task grades stage 1 only, so the reference reproduces
the per-split partials, NOT the merged final attention.
"""
from __future__ import annotations

import math

import torch


def _kv_head_for(head_block_id: int, kv_group_num: int, block_h: int) -> int:
    """Reproduce the kernel's query-head -> KV-head mapping.

    kernel.py:312  cur_kv_head = cur_head_id // cdiv(kv_group_num, BLOCK_H)
    where cur_head_id is the head-BLOCK index (a block covers VALID_BLOCK_H
    consecutive query heads). For the common kv_group_num <= BLOCK_H case
    cdiv(kv_group_num, BLOCK_H) == 1 so this reduces to "the kv head that
    owns this query-head block"; the general form is kept so the reference
    stays kernel-faithful when kv_group_num > BLOCK_H.
    """
    return head_block_id // -(-kv_group_num // block_h)


def _grouped_stage1(
    q: torch.Tensor,            # [B, H, D]
    k_buffer: torch.Tensor,     # [B*C, H_KV, D]
    v_buffer: torch.Tensor,     # [B*C, H_KV, Dv]
    req_to_tokens: torch.Tensor,  # [B*C] global slot permutation
    b_seqlen: torch.Tensor,     # [B] per-request cached length
    scale: float,
    num_kv_splits: int,
) -> torch.Tensor:
    """Per-(batch, head, split) stage-1 partials -> [B, H, NUM_KV_SPLITS, Dv+1].

    For split s of request b the kernel scans the contiguous context window
    [split_kv_start, split_kv_end) (kernel.py:348-350) with an online
    softmax; for a single split window that equals a numerically-stable
    softmax over the window. Output column layout (kernel.py:439-455):
      [..., :Lv] = sum_n softmax(qk)_n * v_n   (the context, acc / e_sum)
      [...,  Lv] = e_max + log(e_sum)          (the split log-sum-exp)
    Empty splits (start >= end) are never written by the kernel
    (kernel.py:356 guards both stores) and stay at the zero-initialised
    buffer value, so the reference leaves them zero.
    """
    device = q.device
    B, H, D = q.shape
    H_KV = k_buffer.shape[1]
    Dv = v_buffer.shape[-1]
    C = req_to_tokens.numel() // B
    S = int(num_kv_splits)
    BLOCK_H = 16  # kernel.py:502
    kv_group_num = H // H_KV
    valid_block_h = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num

    qf = q.to(torch.float32)
    kf = k_buffer.to(torch.float32)
    vf = v_buffer.to(torch.float32)
    req = req_to_tokens.to(torch.long).reshape(B, C)
    seqlen = b_seqlen.to(torch.long)

    out = torch.zeros((B, H, S, Dv + 1), dtype=torch.float32, device=device)

    for b in range(B):
        cur_len = int(seqlen[b].item())
        if cur_len <= 0:
            continue
        # cdiv(cur_len, S); split bounds mirror kernel.py:348-350 exactly.
        kv_len_per_split = -(-cur_len // S)
        for s in range(S):
            start = kv_len_per_split * s
            end = min(start + kv_len_per_split, cur_len)
            if end <= start:
                continue  # empty split -> kernel skips stores -> stays zero
            # kv_loc = req_to_tokens[b, n] (page_size=1, kernel.py:374).
            kv_loc = req[b, start:end]            # [Lwin]
            for h in range(H):
                head_block_id = h // valid_block_h
                kvh = _kv_head_for(head_block_id, kv_group_num, BLOCK_H)
                qh = qf[b, h]                      # [D]
                kw = kf[kv_loc, kvh, :]            # [Lwin, D]
                vw = vf[kv_loc, kvh, :]            # [Lwin, Dv]
                qk = (kw @ qh) * scale             # [Lwin]
                m = torch.max(qk)
                p = torch.exp(qk - m)              # [Lwin]
                e_sum = torch.sum(p)
                ctx = (p.unsqueeze(0) @ vw).squeeze(0) / e_sum  # [Dv]
                out[b, h, s, :Dv] = ctx
                out[b, h, s, Dv] = m + torch.log(e_sum)
    return out


def run(inputs: dict) -> dict:
    return {
        "att_out": _grouped_stage1(
            inputs["q"],
            inputs["k_buffer"],
            inputs["v_buffer"],
            inputs["req_to_tokens"],
            inputs["b_seqlen"],
            float(inputs["scale"]),
            int(inputs["num_kv_splits"]),
        )
    }
