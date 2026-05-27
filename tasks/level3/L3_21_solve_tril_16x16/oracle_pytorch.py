"""Oracle reference for L3_21_solve_tril_16x16_kernel.

Tier: formula_exact_manual
Tolerance class: recurrent_chunk
Category: ssm

The kernel under test is the vendored vLLM/FLA upstream
``solve_tril`` wrapper, which dispatches to
``solve_tril_16x16_kernel`` when ``BT == 16``. Wrapper signature
(kernel.py:521):

    solve_tril(A, cu_seqlens=None, chunk_indices=None,
               output_dtype=torch.float) -> Tensor

with ``A.shape == [B, T, H, BT]`` and the documented math
(kernel.py:528):

    Compute the inverse of the matrix I + A
    A should be strictly lower triangular, i.e., A.triu() == 0.

The previous oracle inverted ``A`` directly via
``solve_triangular(A, eye)``, which is a different operator from
the wrapper and produced max_rel ~ 4e13 on the GPU verdict.
"""
from __future__ import annotations

import torch


def _solve_tril(A: torch.Tensor) -> torch.Tensor:
    """Compute ``(I + tril(A, diagonal=-1))^{-1}`` per chunk.

    Args:
        A: 4-D tensor ``[B, T, H, BT]`` with ``BT == 16``. ``T`` must
           be a multiple of ``BT``. Each ``[BT, BT]`` chunk along ``T``
           is inverted independently and the per-chunk results are
           stitched back into the original layout.

    The kernel internally zeroes the upper triangle (kernel.py:96,
    ``b_A = -tl.where(m_A, b_A, 0)`` where ``m_A`` is the strict
    lower-triangular mask). We mirror that explicitly so the oracle
    is robust against shapes.yaml ``dist='tril'`` which already
    fills the upper triangle with zeros but may inject a non-zero
    diagonal.
    """
    if A.ndim != 4:
        raise ValueError(
            f"solve_tril oracle: A must be 4-D [B,T,H,BT], got shape {tuple(A.shape)}"
        )
    B, T, H, BT = A.shape
    if BT != 16:
        raise ValueError(
            f"solve_tril_16x16 oracle: BT must be 16 for this task, got {BT}"
        )
    if T % BT != 0:
        raise ValueError(
            f"solve_tril oracle: T={T} must be a multiple of BT={BT}"
        )

    Af = A.to(torch.float32)
    NT = T // BT
    # Reshape T -> (NT, BT) so the last two axes carry the per-chunk
    # square block. Layout becomes [B, NT, BT, H, BT]; bring (H, NT,
    # BT, BT) to the front so we can solve each (BT, BT) block at
    # once with a single batched solve_triangular.
    chunks = Af.view(B, NT, BT, H, BT)  # [B, NT, BT_row, H, BT_col]
    # Permute to [B, NT, H, BT_row, BT_col] -- the leading axes are
    # batch dims and the last two are the matrix.
    chunks = chunks.permute(0, 1, 3, 2, 4).contiguous()  # [B, NT, H, BT, BT]

    A_lower = torch.tril(chunks, diagonal=-1)
    eye = torch.eye(BT, device=A.device, dtype=torch.float32)
    M = eye + A_lower
    # ``unitriangular=True`` -- M's diagonal is implicitly 1.0 because
    # we constructed it that way; forward-substitution then produces
    # the per-chunk inverse without forming the full LU.
    inv = torch.linalg.solve_triangular(
        M, eye.expand_as(M), upper=False, unitriangular=True
    )  # [B, NT, H, BT, BT]

    # Undo the reshape: [B, NT, H, BT, BT] -> [B, NT, BT, H, BT] ->
    # [B, T, H, BT].
    inv = inv.permute(0, 1, 3, 2, 4).contiguous()  # [B, NT, BT, H, BT]
    inv = inv.view(B, T, H, BT)
    return inv.to(A.dtype)


def run(inputs: dict) -> dict:
    return {"output": _solve_tril(inputs["A"])}
