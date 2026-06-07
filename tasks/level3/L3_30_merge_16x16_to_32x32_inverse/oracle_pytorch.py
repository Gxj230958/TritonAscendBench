"""Oracle reference for L3_30_merge_16x16_to_32x32_inverse_kernel.

Tier: formula_exact_manual
Tolerance class: recurrent_chunk
Category: ssm

The kernel under test is the vendored vLLM/FLA upstream
``solve_tril`` wrapper, which dispatches to
``merge_16x16_to_32x32_inverse_kernel`` when ``BT == 32`` (see
kernel.py:556-557). Wrapper signature (kernel.py:521):

    solve_tril(A, cu_seqlens=None, chunk_indices=None,
               output_dtype=torch.float) -> Tensor

with ``A.shape == [B, T, H, BT]`` and the documented math
(kernel.py:528):

    Compute the inverse of the matrix I + A
    A should be strictly lower triangular, i.e., A.triu() == 0.

Mathematically identical to the BT=16 path (L3_21) at the wrapper
level -- only the internal kernel strategy differs (the BT=32
kernel merges two 16x16 inverses via the block formula). The
oracle therefore mirrors the wrapper's mathematical contract, not
the internal merge.
"""
from __future__ import annotations

import torch


def _solve_tril(A: torch.Tensor) -> torch.Tensor:
    """Compute ``(I + tril(A, diagonal=-1))^{-1}`` per chunk.

    Args:
        A: 4-D tensor ``[B, T, H, BT]`` with ``BT == 32``. ``T`` must
           be a multiple of ``BT``. Each ``[BT, BT]`` chunk along ``T``
           is inverted independently and the per-chunk results are
           stitched back into the original layout.
    """
    if A.ndim != 4:
        raise ValueError(
            f"solve_tril oracle: A must be 4-D [B,T,H,BT], got shape {tuple(A.shape)}"
        )
    B, T, H, BT = A.shape
    if BT != 32:
        raise ValueError(
            f"merge_16x16_to_32x32 oracle: BT must be 32 for this task, got {BT}"
        )
    if T % BT != 0:
        raise ValueError(
            f"solve_tril oracle: T={T} must be a multiple of BT={BT}"
        )

    Af = A.to(torch.float32)
    NT = T // BT
    chunks = Af.view(B, NT, BT, H, BT)  # [B, NT, BT_row, H, BT_col]
    chunks = chunks.permute(0, 1, 3, 2, 4).contiguous()  # [B, NT, H, BT, BT]

    A_lower = torch.tril(chunks, diagonal=-1)
    eye = torch.eye(BT, device=A.device, dtype=torch.float32)
    M = eye + A_lower
    inv = torch.linalg.solve_triangular(
        M, eye.expand_as(M), upper=False, unitriangular=True
    )

    inv = inv.permute(0, 1, 3, 2, 4).contiguous()
    inv = inv.view(B, T, H, BT)
    return inv.to(A.dtype)


def run(inputs: dict) -> dict:
    return {"output": _solve_tril(inputs["A"])}
