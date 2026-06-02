"""Oracle reference for L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64.

Tier: formula_exact_manual
Tolerance class: recurrent_chunk
Category: ssm

Independent eager-PyTorch reference for the frozen upstream Triton kernel
``chunk_gated_delta_rule_fwd_kernel_h_blockdim64`` (vLLM fla ops,
``chunk_delta_h.py``). This file does NOT call the kernel; it reproduces
the kernel's chunk-level state recurrence from the kernel's own math.

Upstream contract (host wrapper ``chunk_gated_delta_rule_fwd_h``):
  - k: [B, T, Hg, K]   (key, possibly fewer heads than value)
  - w: [B, T, H,  K]   (WY-representation decay weights)
  - u: [B, T, H,  V]   (passed as ``v`` into the kernel)
  - g: [B, T, H]       (within-chunk cumulative log-decays; output of
                        ``chunk_local_cumsum`` upstream)
  - gk: [B, T, H, K]   (per-key-channel cumulative log-decays; drives the
                        kernel's USE_GK branch. Only the chunk's last row is
                        read, mirroring the kernel.)
  - chunk size BT = FLA_CHUNK_SIZE = 64, NT = ceil(T / BT)
  - output h: [B, NT, H, V, K]  -- the per-CHUNK state snapshot, written
    once per chunk BEFORE that chunk's contribution is folded in.

Per chunk i_t (state ``S`` carried as [V, K], initialised to ``initial_state``
or zeros), mirroring the kernel body:
  1. h[:, i_t] = S                      (snapshot at chunk start)
  2. dv = u_chunk - w_chunk @ S^T       ([BT, V])
  3. last = min((i_t + 1) * BT, T) - 1
  4. dv = dv * exp(g[last] - g_chunk) over valid rows  (gated value, USE_G)
  5. S  = S * exp(g[last])              (scalar state decay, USE_G)
  6. S  = S * exp(gk[last])[None, :]    (per-K-channel state decay, USE_GK;
                                         broadcast over the V rows of S, using
                                         only the chunk's last row of gk)
  7. S  = S + dv^T @ k_chunk            ([V, K] accumulation)

USE_GK mirrors the kernel branch ``b_h *= exp(b_gk_last)[None, :]`` which
loads ``gk[b, last, h, :]`` ([B, T, H, K] layout, same as ``w``) and scales
the carried state per key-channel. It affects only ``S`` (not ``dv``) and
fires before the ``dv^T @ k`` accumulation. The kernel never reads the
non-last rows of ``gk``, so the oracle slices the same last row only.
"""
from __future__ import annotations

import torch

# Frozen upstream chunk size (vllm fla ops utils.FLA_CHUNK_SIZE).
_BT = 64


def _to_tf32(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 to TF32 (drop low 13 mantissa bits), matching the kernel's
    unpinned tl.dot on TF32 cores."""
    return (x.contiguous().view(torch.int32) & ~((1 << 13) - 1)).view(torch.float32)


def _chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor,
    gk: torch.Tensor | None,
    initial_state: torch.Tensor | None,
) -> torch.Tensor:
    kf = k.to(torch.float32)
    wf = w.to(torch.float32)
    uf = u.to(torch.float32)
    gf = g.to(torch.float32)
    gkf = gk.to(torch.float32) if gk is not None else None

    B, T, Hg, K = kf.shape
    H = uf.shape[-2]
    V = uf.shape[-1]
    BT = _BT
    NT = (T + BT - 1) // BT
    # Head grouping: each value head h reads key head h // (H // Hg).
    group = H // Hg

    out = kf.new_zeros(B, NT, H, V, K)
    # State per (B, H): [V, K]. The kernel carries one such block per head.
    if initial_state is not None:
        S = initial_state.to(torch.float32).clone()  # [B, H, V, K]
    else:
        S = kf.new_zeros(B, H, V, K)

    for i_t in range(NT):
        start = i_t * BT
        stop = min(start + BT, T)
        rows = slice(start, stop)
        n = stop - start
        last = stop - 1

        # 1. snapshot the pre-update state for this chunk.
        out[:, i_t] = S

        # Chunk slices: [B, n, H, *] and key head-grouped [B, n, Hg, K].
        w_c = wf[:, rows]                       # [B, n, H, K]
        u_c = uf[:, rows]                        # [B, n, H, V]
        # Map each value head to its (possibly shared) key head.
        head_idx = torch.arange(H, device=kf.device) // group  # [H]
        k_c = kf[:, rows][:, :, head_idx, :]     # [B, n, H, K]

        # 2. dv = u - w @ S^T  ->  [B, n, H, V]
        #    (w_c: [B,n,H,K]) x (S^T: [B,H,K,V]) per head.
        wS = torch.einsum("bnhk,bhvk->bnhv", _to_tf32(w_c), _to_tf32(S))
        dv = u_c - wS

        # 4. gate the value rows by exp(g_last - g_row).
        g_c = gf[:, rows]                        # [B, n, H]
        g_last = gf[:, last]                     # [B, H]
        gate = torch.exp(g_last[:, None, :] - g_c)  # [B, n, H]
        dv = dv * gate[..., None]

        # 5. decay the carried state by exp(g_last).
        S = S * torch.exp(g_last)[:, :, None, None]

        # 6. per-K-channel state decay by exp(gk_last) (USE_GK).
        #    Kernel: b_h *= exp(b_gk_last)[None, :] with b_h = S^T [V, K],
        #    i.e. each key-channel k of S scaled by exp(gk[b, last, h, k]),
        #    broadcast across the V rows. Affects only S, before the
        #    dv^T @ k_chunk accumulation.
        if gkf is not None:
            gk_last = gkf[:, last]                # [B, H, K]
            S = S * torch.exp(gk_last)[:, :, None, :]

        # 7. S += dv^T @ k_chunk  ->  [B, H, V, K]
        S = S + torch.einsum("bnhv,bnhk->bhvk", _to_tf32(dv), _to_tf32(k_c))

    return out.to(k.dtype)


def run(inputs: dict) -> dict:
    return {"output": _chunk_gated_delta_rule_fwd_h(
        inputs["k"], inputs["w"], inputs["u"], inputs["g"],
        inputs.get("gk"), inputs.get("initial_state"))}
