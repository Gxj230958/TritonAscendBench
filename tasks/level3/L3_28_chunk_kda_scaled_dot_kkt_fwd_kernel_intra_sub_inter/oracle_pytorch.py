"""Oracle reference for L3_28_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter.

Tier: formula_exact_manual
Tolerance class: recurrent_chunk
Category: ssm

Independent eager-PyTorch reference for the frozen upstream Triton kernel
``chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter`` (vLLM fla ops, KDA).
This file does NOT call the kernel; it reproduces the kernel's INTER
(off-diagonal sub-chunk) scaled-dot KKT math from the kernel's own body.

Upstream contract (host wrapper ``chunk_kda_scaled_dot_kkt_fwd``,
non-varlen path, ``cu_seqlens is None``):
  - q, k, g(=gk): [B, T, H, K]   (g is a PER-DIMENSION gate, consumed AS-IS;
                                  the cumulative transform happens upstream in
                                  ``chunk_kda_fwd``, NOT in this kernel)
  - beta:         [B, T, H]
  - chunk size BT = FLA_CHUNK_SIZE = 64, NT = ceil(T / BT)
  - sub-chunk width BC = min(16, BT), NC = ceil(BT / BC)
  - outputs A, Aqk: [B, T, H, BT]   (last axis = LOCAL within-chunk column)

The kernel is launched on grid ``(NT, NC*NC, B*H)``. Program ``i_c`` maps to a
row sub-chunk ``i_i = i_c // NC`` and a column sub-chunk ``i_j = i_c % NC``;
the kernel RETURNS for ``i_i <= i_j``, so it writes ONLY the strictly
off-diagonal sub-chunk blocks of each BT-row time chunk. Diagonal sub-blocks
(``i_i == i_j``) and the upper triangle are written by a different named
kernel (``..._intra_sub_intra``) and therefore stay ZERO here.

For a time chunk ``i_t``, a row sub-chunk ``i_i`` (rows
``r = i_t*BT + i_i*BC + a``, ``a in [0, BC)``) and a column sub-chunk
``i_j < i_i`` (cols ``c = i_t*BT + i_j*BC + b``, ``b in [0, BC)``), with the
per-dim anchor gate ``gn_d = g[row 0 of sub-chunk i_i, d]`` (kernel.py:608)::

    A[r, c]   = beta_r * sum_d  ( k_r_d * exp(g_r_d - gn_d) )          # b_k
                              * ( k_c_d * exp(gn_d - g_c_d) )          # b_ktg
    Aqk[r, c] = scale  * sum_d  ( q_r_d * exp(g_r_d - gn_d) )          # b_qg
                              * ( k_c_d * exp(gn_d - g_c_d) )          # b_ktg

The anchor ``gn`` cancels algebraically (exp(g_r-gn)*exp(gn-g_c)=exp(g_r-g_c)),
but it does NOT cancel under the kernel's TF32 truncation: the kernel forms the
two operands ``b_k`` / ``b_qg`` and ``b_ktg`` with the anchor folded in, then
``tl.dot``s them on the TF32 tensor cores. So the oracle keeps the anchor split
and truncates the literal dot operands to TF32 (see ``_to_tf32``). ``beta`` is
applied AFTER the dot (kernel.py:623), so it is NOT folded into the truncated
operand; ``scale`` IS inside the ``b_qg`` operand (kernel.py:620), so it is
folded before truncation.

The output column index is LOCAL to the time chunk: A[b, r, h, j_local] holds
the product of row ``r`` against in-chunk position ``j_local = i_j*BC + b``.
"""
from __future__ import annotations

import torch


def _to_tf32(x: torch.Tensor) -> torch.Tensor:
    """Round a float32 tensor to TF32 precision the way the kernel's GPU
    ``tl.dot`` does.

    The two dots (kernel.py:617, 621) pin NO precision (no
    ``allow_tf32=False`` / ``input_precision="ieee"``), so on Ampere+ each
    float32 operand is reduced to a 19-bit TF32 value before the multiply,
    with float32 accumulate. Triton's default ``tl.dot`` TRUNCATES the low 13
    mantissa bits (it does not round-to-nearest), so emulate truncation. This
    keeps the oracle a faithful reference for the FROZEN kernel; the residual
    oracle-vs-kernel diff is then only float32 accumulation-order noise, well
    inside the recurrent_chunk band (atol/rtol=5e-3). Sibling L3_18 uses the
    identical emulation.
    """
    bits = x.contiguous().view(torch.int32)
    truncated = bits & ~((1 << 13) - 1)
    return truncated.view(torch.float32)


# Frozen upstream chunk size (vllm fla ops utils.FLA_CHUNK_SIZE).
_BT = 64


def _chunk_kda_scaled_dot_kkt_inter(
    q: torch.Tensor, k: torch.Tensor, gk: torch.Tensor,
    beta: torch.Tensor, scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    gf = gk.to(torch.float32)
    bf = beta.to(torch.float32)

    B, T, H, K = kf.shape
    BT = _BT
    BC = min(16, BT)
    NC = (BT + BC - 1) // BC
    NT = (T + BT - 1) // BT

    A = kf.new_zeros(B, T, H, BT)
    Aqk = kf.new_zeros(B, T, H, BT)

    for i_t in range(NT):
        for i_i in range(NC):
            row0 = i_t * BT + i_i * BC          # first global row of sub-chunk i_i
            if row0 >= T:
                continue                        # kernel early-return (line 570)
            r_stop = min(row0 + BC, T)
            n_r = r_stop - row0
            rows = slice(row0, r_stop)

            # Anchor gate gn = g at the FIRST row of sub-chunk i_i, per (B,H,K).
            gn = gf[:, row0:row0 + 1, :, :]                 # [B, 1, H, K]

            # Row operands (rows of sub-chunk i_i).
            g_r = gf[:, rows, :, :]                          # [B, n_r, H, K]
            ek_r = torch.exp(g_r - gn)                       # exp(g_r - gn)
            b_k = _to_tf32(kf[:, rows, :, :] * ek_r)         # left operand (A)
            b_qg = _to_tf32(qf[:, rows, :, :] * ek_r * scale)  # left operand (Aqk)
            b_r = bf[:, rows, :]                             # [B, n_r, H]

            for i_j in range(i_i):              # strictly off-diagonal: i_j < i_i
                col0 = i_t * BT + i_j * BC
                if col0 >= T:
                    continue
                c_stop = min(col0 + BC, T)
                n_c = c_stop - col0
                cols = slice(col0, c_stop)

                g_c = gf[:, cols, :, :]                      # [B, n_c, H, K]
                # b_ktg = k_col * exp(gn - g_col); gn uses sub-chunk i_i's anchor.
                b_ktg = _to_tf32(kf[:, cols, :, :] * torch.exp(gn - g_c))

                # Per-head dot over K: [B, n_r, H] x [B, n_c, H] -> [B, H, n_r, n_c].
                a_blk = torch.einsum("brhk,bchk->bhrc", b_k, b_ktg)
                a_blk = a_blk * b_r.permute(0, 2, 1)[:, :, :, None]   # beta AFTER dot
                aqk_blk = torch.einsum("brhk,bchk->bhrc", b_qg, b_ktg)

                # Scatter into [B, T, H, BT]; column axis is LOCAL (i_j*BC + b).
                j0 = i_j * BC
                A[:, rows, :, j0:j0 + n_c] = a_blk.permute(0, 2, 1, 3)
                Aqk[:, rows, :, j0:j0 + n_c] = aqk_blk.permute(0, 2, 1, 3)

    return A.to(torch.float32), Aqk.to(torch.float32)


def run(inputs: dict) -> dict:
    A, Aqk = _chunk_kda_scaled_dot_kkt_inter(
        inputs["q"], inputs["k"], inputs["gk"], inputs["beta"],
        float(inputs.get("scale", 1.0)),
    )
    return {"A": A, "Aqk": Aqk}
