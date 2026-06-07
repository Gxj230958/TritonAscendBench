"""Oracle reference for L3_16_fwd_kernel (prefill attention).

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


def _softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically-stable softmax spelled from its definition -- keeps the
    oracle body free of high-level torch ops (contract §6 / L195). Causal
    rows carry additive -inf entries; exp(-inf - max) -> 0 exactly as
    torch's softmax handles them."""
    e = torch.exp(x - x.amax(dim, keepdim=True))
    return e / e.sum(dim, keepdim=True)


def naive_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    is_causal: bool = True,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Standard scaled-dot-product attention, batched per-sequence.

    Mirrors the kernel's in-place contract: the result is written into the
    caller-supplied ``o`` buffer (and returned for convenience). The kernel
    accumulates in fp32 and ``tl.store``s into the runner-allocated ``o``,
    so the reference casts to ``o.dtype`` (fp16) at the exact same point --
    matching both the value and the storage dtype the kernel produces.

    q, k, v, o: [total_tokens, head, head_dim]
    """
    total_tokens, h_q, head_dim = q.shape
    h_kv = k.shape[1]
    kv_group = h_q // h_kv

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

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

        probs = _softmax(scores, dim=-1)                                   # (h_q, S, S)
        out_b = torch.matmul(probs, v_t)                                   # (h_q, S, d)
        out_b = out_b.transpose(0, 1).to(o.dtype)                          # (S, h_q, d)

        o[start : start + seq_len] = out_b

    return o


def run(inputs: dict) -> dict:
    # The kernel writes attention into the runner-allocated ``o`` buffer in
    # place and returns None; the harness then surfaces every tensor argument
    # it bound (q, k, v, o, b_start_loc, b_seq_len) as the candidate's
    # outputs. The oracle mirrors that exact structure: read-only inputs pass
    # through untouched (the kernel never mutates them) and ``o`` carries the
    # computed result in the buffer's dtype. Returning the full bound set --
    # rather than only ``o`` -- makes the oracle structurally identical to the
    # in-place kernel so the anchor's arity/dtype gate matches faithfully.
    o = inputs["o"]
    naive_prefill_attention(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        o,
        inputs["b_start_loc"],
        inputs["b_seq_len"],
        is_causal=inputs.get("is_causal", True),
        sm_scale=inputs.get("sm_scale"),
    )
    return {
        "q": inputs["q"],
        "k": inputs["k"],
        "v": inputs["v"],
        "o": o,
        "b_start_loc": inputs["b_start_loc"],
        "b_seq_len": inputs["b_seq_len"],
    }
