"""Oracle reference for L3_17_chunk_gla_fwd_kernel_o.

Tier: formula_exact_manual
Tolerance class: recurrent_chunk
Category: ssm

Independent eager-PyTorch reference for the vLLM FLA chunked-GLA output
kernel ``chunk_gla_fwd_kernel_o`` (vllm/model_executor/layers/fla/ops,
commit 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19). Authored from the frozen
Triton kernel's ``make_block_ptr`` contract; never imports or calls the
kernel.

Upstream contract (non-varlen path, ``cu_seqlens is None``):

The kernel is launched on grid ``(cdiv(V, BV), NT, B*H)`` with
``NT = cdiv(T, BT)`` and produces, per ``(batch, head, chunk)``, one
``BT``-sized output tile. The output is the sum of an INTER-chunk gated
state read and an INTRA-chunk causal-masked score x value.

Tensor layouts (matching the kernel's ``make_block_ptr`` strides):
    q: [B, T, H, K]      gated query
    g: [B, T, H, K]      log-gate (the kernel applies exp(g))
    v: [B, T, H, V]      values
    A: [B, T, H, BT]     per-chunk intra score; last axis is the LOCAL
                         within-chunk column index, so A mixes rows only
                         WITHIN a chunk (block-diagonal in time)
    h: [B, NT, H, V, K]  per-chunk hidden state; the kernel reads it as a
                         [V, K] tile (strides (K, 1)) and transposes for
                         the dot, so the stored last two dims are [V, K]
    -> o: [B, T, H, V]

For a chunk ``c`` (global rows ``r = c*BT + i``, local row ``i`` and local
column ``j`` in ``[0, BT)``) and head ``h``::

    o[b, r, h, :] =
        # inter-chunk: gated query read against the chunk state
        sum_k (q[b,r,h,k] * scale * exp(g[b,r,h,k])) * h[b, c, h, :, k]
        # intra-chunk: causal-masked (j <= i) score x value, columns LOCAL
      + sum_{j<=i} A[b, r, h, j] * v[b, c*BT+j, h, :]

The kernel keeps everything in the input dtype with an fp32 dot accumulate;
for the float32 contract this is fp32 end to end.
"""
from __future__ import annotations

import torch


def _chunk_gla_fwd_o(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = A.shape[-1]
    if T % BT != 0:
        raise ValueError(f"T={T} must be divisible by chunk size BT={BT}")
    NT = T // BT

    qf = q.to(torch.float32)
    vf = v.to(torch.float32)
    gf = g.to(torch.float32)
    Af = A.to(torch.float32)
    hf = h.to(torch.float32)

    # Inter-chunk: each row's gated query reads the chunk's [V, K] state.
    # Fold the time axis into (chunk, local-row); the state is indexed per
    # chunk, so row r=c*BT+i reads h[b, c, hh].
    qg = qf * scale * torch.exp(gf)                          # [B, T, H, K]
    qg_c = qg.reshape(B, NT, BT, H, K)                       # [B,NT,row,H,K]
    h_c = hf.reshape(B, NT, H, V, K)                         # [B,NT,H,V,K]
    inter = torch.einsum("bnrhk,bnhvk->bnrhv", qg_c, h_c)    # [B,NT,row,H,V]
    inter = inter.reshape(B, T, H, V)

    # Intra-chunk: A only mixes WITHIN a chunk; its last axis is the LOCAL
    # column, masked to the lower triangle (row local-index >= col).
    Ac = Af.reshape(B, NT, BT, H, BT)                        # [B,NT,row,H,col]
    mask = torch.tril(torch.ones(BT, BT, device=q.device, dtype=torch.float32))
    Ac = Ac * mask.reshape(1, 1, BT, 1, BT)
    vc = vf.reshape(B, NT, BT, H, V)                         # [B,NT,col,H,V]
    intra = torch.einsum("bnrhc,bnchv->bnrhv", Ac, vc)      # [B,NT,row,H,V]
    intra = intra.reshape(B, T, H, V)

    return (inter + intra).to(q.dtype)


def run(inputs: dict) -> dict:
    return {"output": _chunk_gla_fwd_o(
        inputs["q"], inputs["v"], inputs["g"], inputs["A"], inputs["h"],
        float(inputs.get("scale", 1.0)))}
