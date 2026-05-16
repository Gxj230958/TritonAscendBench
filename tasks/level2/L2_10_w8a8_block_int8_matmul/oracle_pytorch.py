"""Oracle reference for L2_10_w8a8_block_int8_matmul.

Tier: formula_exact_manual
Tolerance class: int8_w8a8

The op is the composition of:
1. Per-(token, k_group) dequant of A: A_fp = A_int8 * As[:, k_group]
2. Per-(n_block, k_group) dequant of B: B_fp = B_int8 * Bs[n_block, k_group]
3. fp32 matmul, accumulate, cast to output_dtype

The kernel does this in fused fp32 accumulation; the oracle dequantizes
explicitly and uses a plain PyTorch matmul. Both share the property that
the accumulator is fp32 and the cast happens once at the end.

This belongs to tier `formula_exact_manual` (not `simple_pytorch_ok`)
because:
- The block-scale layout (As: per-token×k_group, Bs: n_block×k_group) must
  be expanded carefully to per-element scales.
- The kernel uses `% M` and `% N` for masking; the oracle must mirror that
  to be byte-equivalent on out-of-bound tiles, which we do by truncating
  inputs to valid (M, N, K) ranges before computing.
"""
from __future__ import annotations

import torch


def _expand_scale_a(As: torch.Tensor, block_k: int, K: int) -> torch.Tensor:
    """As shape (M, K // block_k) → (M, K), repeating each scale block_k times."""
    M, n_k_groups = As.shape
    expanded = As.repeat_interleave(block_k, dim=-1)
    # Truncate to K (in case K is not exact multiple)
    return expanded[:, :K]


def _expand_scale_b(Bs: torch.Tensor, block_n: int, block_k: int, N: int, K: int) -> torch.Tensor:
    """Bs shape (N // block_n, K // block_k) → (N, K)."""
    # Bs is laid out (n_blocks, k_blocks); expand both dims
    expanded = Bs.repeat_interleave(block_k, dim=-1).repeat_interleave(block_n, dim=0)
    return expanded[:N, :K]


def oracle_w8a8_block_int8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    block_n, block_k = block_size[0], block_size[1]
    N, K = B.shape
    M = A.numel() // A.shape[-1]
    A_2d = A.reshape(M, K)
    As_2d = As.reshape(M, As.shape[-1])

    A_scale = _expand_scale_a(As_2d.to(torch.float32), block_k, K)
    B_scale = _expand_scale_b(Bs.to(torch.float32), block_n, block_k, N, K)

    A_fp = A_2d.to(torch.float32) * A_scale
    B_fp = B.to(torch.float32) * B_scale

    # C = A_fp @ B_fp^T
    C = A_fp @ B_fp.T
    C = C.to(output_dtype)

    out_shape = A.shape[:-1] + (N,)
    return C.reshape(out_shape)


def run(inputs: dict) -> dict:
    out = oracle_w8a8_block_int8_matmul(
        inputs["A"],
        inputs["B"],
        inputs["As"],
        inputs["Bs"],
        inputs["block_size"],
        inputs.get("output_dtype", torch.float16),
    )
    return {"C": out}
