"""Oracle reference for L3_04_fwd_kernel (prefill attention).

Tier: upstream_oracle_required
Tolerance class: attention_flash

FlashAttention's online-softmax recurrence is mathematically equivalent to
naive attention but produces slightly different rounding than the standard
reference due to streaming reorder. We use the **naive reference** as the
oracle because it is the ground-truth definition; the tolerance class
admits the bounded drift from the streaming variant.

The oracle handles variable-length sequences packed in [total_tokens, ...]
layout the same way the kernel does, using b_start_loc/b_seq_len.

Cross-oracle drift check: at v0.1, no second oracle is available. The
runner records `cross_oracle_status: single_oracle` and does not auto-
resolve drift between sources.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def naive_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    is_causal: bool = True,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Standard scaled-dot-product attention, batched per-sequence.

    q, k, v: [total_tokens, head, head_dim]
    """
    total_tokens, h_q, head_dim = q.shape
    h_kv = k.shape[1]
    kv_group = h_q // h_kv

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    out = torch.zeros_like(q)
    n_batch = b_seq_len.shape[0]

    for b in range(n_batch):
        start = int(b_start_loc[b].item())
        seq_len = int(b_seq_len[b].item())
        if seq_len == 0:
            continue

        q_b = q[start : start + seq_len]                                   # (S, h_q, d)
        k_b = k[start : start + seq_len]                                   # (S, h_kv, d)
        v_b = v[start : start + seq_len]                                   # (S, h_kv, d)

        # Expand GQA → broadcast kv heads
        k_b_expanded = k_b.repeat_interleave(kv_group, dim=1)              # (S, h_q, d)
        v_b_expanded = v_b.repeat_interleave(kv_group, dim=1)              # (S, h_q, d)

        # (h_q, S, d) layout for batched matmul
        q_t = q_b.transpose(0, 1).to(torch.float32)
        k_t = k_b_expanded.transpose(0, 1).to(torch.float32)
        v_t = v_b_expanded.transpose(0, 1).to(torch.float32)

        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * sm_scale       # (h_q, S, S)

        if is_causal:
            mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=q.device),
                diagonal=1,
            )
            scores = scores + mask

        probs = F.softmax(scores, dim=-1)                                  # (h_q, S, S)
        out_b = torch.matmul(probs, v_t)                                   # (h_q, S, d)
        out_b = out_b.transpose(0, 1).to(q.dtype)                          # (S, h_q, d)

        out[start : start + seq_len] = out_b

    return out


def run(inputs: dict) -> dict:
    out = naive_prefill_attention(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["b_start_loc"],
        inputs["b_seq_len"],
        is_causal=inputs.get("is_causal", True),
        sm_scale=inputs.get("sm_scale"),
    )
    return {"o": out}
