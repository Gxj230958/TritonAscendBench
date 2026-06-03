"""Oracle reference for L3_10_fused_moe_kernel_gptq_awq.

Tier: formula_exact_manual
Tolerance class: atomic_scatter
Category: moe

Independent PyTorch reference for the GPTQ/AWQ int4-w4a16 fused-MoE expert
GEMM realised by the frozen ``fused_moe_kernel_gptq_awq`` triton kernel. The
math is authored from the quantization spec, not copied from upstream:

  * ``b`` holds 4-bit weights packed two-per-int32 along the contraction axis
    K (== D): ``b[e, n, k//2]``, low nibble for even k, high nibble for odd k
    (shift ``(k % 2) * 4``).  Unpacked value q in [0, 15].
  * ``b_zp`` holds 4-bit asymmetric zero-points packed two-per-int32 along N:
    ``b_zp[e, n//2, g]``, shift ``(n % 2) * 4``.  Unpacked zp in [0, 15].
  * ``b_scale`` is a per-(expert, n, group) fp32 dequant scale, group index
    ``g = k // group_size``.
  * Dequantized weight:  w[e, n, k] = (q[e, n, k] - zp[e, n, g]) * scale[e, n, g].
  * The kernel scatters each routed token into output row == its token id
    (``sorted_token_ids`` value), reading activation row ``token_id // top_k``
    (top_k == 1 here).  Per expert block ``blk`` (block_size rows) the expert
    is ``expert_ids[blk]``; padding slots (token id >= n_tokens) are skipped.
    The output is ``[n_tokens, N]``; rows of un-routed tokens stay zero.
  * If ``apply_weight`` the row is scaled by ``topk_weights[token_id]``.

The fixture (shapes.yaml ``moe_sorted_ids``) draws DISTINCT token ids, so each
output row is written at most once -- the scatter is deterministic and the
``atomic_scatter`` tolerance class applies.
"""
from __future__ import annotations

import torch


def _dequant_weight(
    b: torch.Tensor, b_scale: torch.Tensor, b_zp: torch.Tensor,
    K: int, group_size: int,
) -> torch.Tensor:
    """Dequantize packed int4 weights to fp32 [E, N, K].

    b:       [E, N, K//2] int32 (2 nibbles per elem along K)
    b_scale: [E, N, K//group_size] fp32
    b_zp:    [E, N//2, K//group_size] int32 (2 nibbles per elem along N)
    """
    E, N, _ = b.shape
    bi = b.to(torch.int64)
    # Unpack K nibbles: even k -> low nibble of b[..., k//2], odd k -> high.
    k_idx = torch.arange(K, device=b.device)
    shift_k = (k_idx % 2) * 4                              # [K]
    q = (bi[:, :, k_idx // 2] >> shift_k) & 0xF            # [E, N, K]
    q = q.to(torch.float32)

    # Unpack N nibbles of the zero-point: row n -> nibble (n % 2) of [..., n//2].
    n_idx = torch.arange(N, device=b.device)
    shift_n = (n_idx % 2) * 4                              # [N]
    zp = (b_zp.to(torch.int64)[:, n_idx // 2, :] >> shift_n.view(1, N, 1)) & 0xF
    zp = zp.to(torch.float32)                              # [E, N, K//group_size]

    # Broadcast per-group scale / zp to per-element along K.
    g_idx = k_idx // group_size                            # [K]
    scale_full = b_scale[:, :, g_idx]                      # [E, N, K]
    zp_full = zp[:, :, g_idx]                              # [E, N, K]
    return ((q - zp_full) * scale_full).to(torch.bfloat16).to(torch.float32)                      # [E, N, K]


def _fused_moe_gptq_awq(
    a: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor,
    b_zp: torch.Tensor, topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor,
    block_size: int, group_size: int, apply_weight: bool,
) -> torch.Tensor:
    n_tok, K = a.shape
    N = b.shape[1]
    af = a.to(torch.float32)
    w = _dequant_weight(b, b_scale.to(torch.float32), b_zp, K, group_size)  # [E,N,K]

    st = sorted_token_ids.to(torch.long)
    tw = topk_weights.reshape(-1).to(torch.float32)
    out = a.new_zeros((n_tok, N), dtype=torch.float32)

    nblk = st.shape[0] // block_size
    for blk in range(nblk):
        e = int(expert_ids[blk].item())
        for r in range(block_size):
            tid = int(st[blk * block_size + r].item())
            if tid >= n_tok:            # padding slot -> not written
                continue
            y = af[tid] @ w[e].T        # [N]  (top_k == 1: a-row == token id)
            if apply_weight:
                y = y * tw[tid]
            out[tid] = y
    return out


def run(inputs: dict) -> dict:
    return {"output": _fused_moe_gptq_awq(
        inputs["a"], inputs["b"], inputs["b_scale"], inputs["b_zp"],
        inputs["topk_weights"], inputs["sorted_token_ids"],
        inputs["expert_ids"], int(inputs["block_size"]),
        int(inputs["group_size"]),
        bool(inputs.get("apply_weight", True)))}
