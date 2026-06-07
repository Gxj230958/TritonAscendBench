"""Oracle reference for L3_25_chunk_fwd_kernel_o.

Tier: formula_exact_manual
Tolerance class: recurrent_chunk
Category: ssm

Independent eager-PyTorch reference for the frozen upstream Triton kernel
``chunk_fwd_kernel_o`` (vLLM fla ops, ``chunk_o.py``). This file does NOT
call the kernel; it reproduces the kernel's per-chunk output math from the
kernel body itself.

Upstream contract (host wrapper ``chunk_fwd_o``, non-varlen path):
  - q: [B, T, Hg, K]   (query, possibly fewer heads than value -- GQA)
  - k: [B, T, Hg, K]   (key, shares the Hg head count with q)
  - v: [B, T, H,  V]   (value)
  - h: [B, NT, H, V, K]  (per-CHUNK state snapshots produced upstream by
                          chunk_fwd_h; chunk i_t reads its OWN h[:, i_t]).
  - g: [B, T, H]       (within-chunk cumulative log-decays; output of
                        chunk_local_cumsum upstream. USE_G branch.)
  - scale: float       (softmax scale, derive 1/sqrt(K)).
  - chunk size BT = FLA_CHUNK_SIZE = 64, NT = ceil(T / BT)
  - output o: [B, T, H, V]

Per chunk i_t (rows [start, stop)), mirroring the kernel body
(i_v/i_k tiling loops collapse to full dense matmuls here):

  Value head h reads key/query head  h // (H // Hg)  (kernel:
  ``i_h // (H // Hg)`` on the q/k offset). State block h[:, i_t] is the
  [V, K] snapshot for that (chunk, head).

  1. inter = q_chunk @ h_state^T              # [BT, V]; b_q @ trans(b_h)
  2. A     = q_chunk @ k_chunk^T              # [BT, BT]; b_q @ b_k
     If USE_G:
       inter *= exp(g_chunk)[:, None]
       A     *= exp(g_chunk[:, None] - g_chunk[None, :])
  3. A = tril(A)  (causal within the chunk + valid rows only)
  4. o_chunk = scale * inter + scale * (A @ v_chunk)

The kernel NEVER mixes chunks in this kernel: the cross-chunk recurrence is
already folded into the precomputed per-chunk state h. So o is a strictly
per-chunk function of (q, k, v, h[chunk], g) -- and h's chunk index is
load-bearing: using a single shared state instead of the per-chunk slice
changes every chunk past the first.
"""
from __future__ import annotations

import torch

# Frozen upstream chunk size (vllm fla ops utils.FLA_CHUNK_SIZE).
_BT = 64


def _bf16(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 operands to bf16 (the activation dtype), matching the kernel's
    bf16-operand tl.dot. (Named _bf16; an earlier docstring wrongly said TF32 — the
    body has always been a bf16 roundtrip, GPU-verified correct at the anchors.)"""
    return x.to(torch.bfloat16).to(torch.float32)


def _chunk_fwd_o(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    h: torch.Tensor, g: torch.Tensor | None, scale: float,
) -> torch.Tensor:
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    hf = h.to(torch.float32)
    gf = g.to(torch.float32) if g is not None else None

    B, T, Hg, K = qf.shape
    H = vf.shape[-2]
    V = vf.shape[-1]
    BT = _BT
    NT = (T + BT - 1) // BT
    # Head grouping: each value head reads key/query head h // (H // Hg).
    group = H // Hg
    head_idx = torch.arange(H, device=qf.device) // group  # [H]

    out = vf.new_zeros(B, T, H, V)

    for i_t in range(NT):
        start = i_t * BT
        stop = min(start + BT, T)
        rows = slice(start, stop)
        n = stop - start

        # Query/key for this chunk, head-grouped to the value heads.
        q_c = qf[:, rows][:, :, head_idx, :]      # [B, n, H, K]
        k_c = kf[:, rows][:, :, head_idx, :]      # [B, n, H, K]
        v_c = vf[:, rows]                          # [B, n, H, V]
        h_state = hf[:, i_t]                       # [B, H, V, K]

        # 1. inter = q_chunk @ h_state^T  -> [B, n, H, V]
        inter = torch.einsum("bnhk,bhvk->bnhv", _bf16(q_c), _bf16(h_state))
        # 2. A = q_chunk @ k_chunk^T  -> [B, n, n, H]
        A = torch.einsum("bnhk,bmhk->bnmh", _bf16(q_c), _bf16(k_c))

        if gf is not None:
            g_c = gf[:, rows]                      # [B, n, H]
            inter = inter * torch.exp(g_c)[..., None]
            # decay[i, j] = exp(g_i - g_j)
            decay = torch.exp(g_c[:, :, None, :] - g_c[:, None, :, :])  # [B,n,n,H]
            A = A * decay

        # 3. causal mask within the chunk (lower triangular, valid rows).
        idx = torch.arange(n, device=qf.device)
        causal = (idx[:, None] >= idx[None, :]).to(torch.float32)  # [n, n]
        A = A * causal[None, :, :, None]

        # 4. o_chunk = scale * inter + scale * (A @ v_chunk)
        intra = torch.einsum("bnmh,bmhv->bnhv", _bf16(A), _bf16(v_c))
        out[:, rows] = scale * inter + scale * intra

    return out.to(v.dtype)


def run(inputs: dict) -> dict:
    return {"output": _chunk_fwd_o(
        inputs["q"], inputs["k"], inputs["v"], inputs["h"],
        inputs.get("g"), float(inputs.get("scale", 1.0)))}
