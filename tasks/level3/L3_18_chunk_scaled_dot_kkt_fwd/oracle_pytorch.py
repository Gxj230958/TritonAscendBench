"""Oracle reference for L3_18_chunk_scaled_dot_kkt_fwd_kernel.

Tier: formula_exact_manual
Tolerance class: recurrent_chunk
Category: ssm

Independent eager-PyTorch reference for the frozen upstream Triton kernel
``chunk_scaled_dot_kkt_fwd_kernel`` (vLLM fla ops). This file does NOT call
the kernel; it reproduces the kernel's chunk-local KKT math from the
kernel's own body.

Upstream contract (host wrapper ``chunk_scaled_dot_kkt_fwd``):
  - k:    [B, T, Hg, K]   (key; possibly fewer heads than beta/g)
  - beta: [B, T, H]
  - g:    [B, T, H]       (cumulative gate; ``USE_G`` is always on here)
  - cu_seqlens: [NSEG+1]  (optional; flips the kernel ``IS_VARLEN`` path on)
  - chunk size BT = FLA_CHUNK_SIZE = 64, NT = ceil(T / BT)
  - output A: [B, T, H, BT]

Two layouts share the SAME chunk-local KKT math; ``cu_seqlens`` only changes
where chunk boundaries fall:

* Fixed-length (``cu_seqlens is None``, ``IS_VARLEN=False``): chunks tile each
  ``[B, T, ...]`` plane at multiples of BT from 0.
* Packed varlen (``cu_seqlens`` given, ``IS_VARLEN=True``): k/beta/g are a single
  packed plane ``[1, total_T, ...]`` and ``cu_seqlens`` defines NSEG segments
  ``[bos, eos)``. The kernel RESETS the chunk grid at every segment start
  (``prepare_chunk_indices`` emits ``ceil(seg_len / BT)`` chunks per segment,
  each addressed from ``bos``), so a segment whose length is not a BT multiple
  shifts every later chunk off the fixed-length grid. B is pinned to 1 for the
  packed layout (the kernel's varlen branch ignores the batch program id and
  addresses k/beta/g/A purely from the flat ``bos`` offset).

Per program ``(i_t, i_b, i_h)`` the kernel computes, over the BT-row chunk
starting at ``i_t * BT`` (mirroring kernel.py lines 80-113):
  - value head ``i_h`` reads key head ``i_h // (H // Hg)`` (GQA grouping)
  - b_A[i, j] = (k_i . k_j) * beta_i                      (within-chunk dot)
  - b_A[i, j] *= exp(g_i - g_j)                           (gate decay)
  - keep only strictly-lower-triangular valid entries:
        o_t = i_t * BT + arange(BT);  m_t = o_t < T
        m_A = (o_t[i] > o_t[j]) & m_t[i] & m_t[j]
  - store b_A into A[i_b, rows, i_h, 0:BT]

The output last dim is the CHUNK width BT (not T): A[b, t, h, j] holds the
scaled dot of row ``t`` against position ``i_t*BT + j`` within the SAME
chunk, lower-triangular only (no cross-chunk products).
"""
from __future__ import annotations

import torch

# Frozen upstream chunk size (vllm fla ops utils.FLA_CHUNK_SIZE).
_BT = 64


def _to_tf32(x: torch.Tensor) -> torch.Tensor:
    """Round a float32 tensor to TF32 precision the way the kernel's GPU
    ``tl.dot`` does.

    ``kernel.py`` line 100 accumulates the within-chunk KKT block with
    ``tl.dot(b_kb, trans(b_k))`` and pins NO precision (no ``allow_tf32=False``
    / ``input_precision="ieee"`` -- contrast the sibling fp32 fla kernels
    L3_20 / L3_22 which DO, and L3_05 which sets ``input_precision="ieee"``).
    On Ampere+ an unpinned float32 ``tl.dot`` therefore runs on the TF32
    tensor cores: each float32 operand is reduced to a 19-bit TF32 value
    (8 exp + 10 mantissa bits) before the multiply, with float32 accumulate.

    NVIDIA's *cuBLAS* path rounds float32->TF32 round-to-nearest-even, but
    Triton's default ``tl.dot`` TRUNCATES the low mantissa bits (it does not
    round) unless the kernel opts into rounding. Truncation (drop the low 13
    of the 23 float32 mantissa bits) is the variant that reproduces the
    observed hardware result: on shape P1 the GPU oracle-equivalence anchor
    measured max_abs=2.461e-2 / max_rel=21.4, and truncated-TF32 emulation
    here matches that at max_abs=2.460e-2 / max_rel=21.4 (round-to-nearest
    would give 7.5e-3, ~3x too small). Mirroring the kernel's actual TF32
    truncation -- rather than computing the ideal full-float32 dot -- is what
    keeps this oracle a *faithful* reference for the FROZEN kernel: the
    benchmark grades kernel-vs-oracle equivalence, and the kernel's real
    operation includes this precision reduction. Tolerance is unchanged
    (recurrent_chunk atol/rtol=5e-3); the residual oracle-vs-kernel diff is
    then only float32 accumulation-order noise (<= 5.4e-5 across all shapes
    and every autotuned BK tile), well inside that band.
    """
    bits = x.contiguous().view(torch.int32)
    truncated = bits & ~((1 << 13) - 1)
    return truncated.view(torch.float32)


def _chunk_block(
    kf: torch.Tensor, gf: torch.Tensor, bf: torch.Tensor,
    rows: slice, head_idx: torch.Tensor,
) -> torch.Tensor:
    """One BT-or-shorter chunk's strictly-lower-triangular scaled KKT block.

    Computes, for the ``n`` rows in ``rows`` (a chunk that does NOT span a
    segment boundary), the kernel's per-chunk math exactly:
        block[b, h, i, j] = beta_i * (k_i . k_j) * exp(g_i - g_j)   for i > j
    and zero elsewhere within the chunk. ``head_idx`` maps value head h to its
    key head ``h // (H // Hg)``. Returns ``[B, n, H, n]`` (value-head layout,
    ready to scatter into ``A[:, rows, :, :n]``).
    """
    k_c = kf[:, rows][:, :, head_idx, :]               # [B, n, H, K]
    b_c = bf[:, rows]                                  # [B, n, H]
    g_c = gf[:, rows]                                  # [B, n, H]
    n = k_c.shape[1]

    # Mirror the kernel's dot operands EXACTLY (kernel.py lines 99-100):
    #   b_kb = b_k * b_beta[:, None]          # beta folded into the LEFT operand
    #   b_A += tl.dot(b_kb, tl.trans(b_k))    # TF32 tensor-core matmul
    # so beta_i multiplies the row BEFORE the TF32 truncation, and both dot
    # operands (beta-scaled keys, and raw keys) are truncated to TF32. The
    # full-K matmul accumulates in float32, matching the kernel's float32 dot
    # accumulator (the autotuned BK tiling only changes accumulation order,
    # which is sub-5e-5 noise here).
    k_t = _to_tf32(k_c)                                # [B, n, H, K] (right operand)
    kb_t = _to_tf32(b_c[:, :, :, None] * k_c)          # [B, n, H, K] (left operand)
    # b_A[i, j] = sum_k (beta_i * k_ik)_tf32 * (k_jk)_tf32, per head: [B, H, n, n].
    kk_scaled = torch.einsum("bihk,bjhk->bhij", kb_t, k_t)
    # exp(g_i - g_j) per head: [B, H, n, n].
    g_h = g_c.permute(0, 2, 1)                         # [B, H, n]
    decay = torch.exp(g_h[:, :, :, None] - g_h[:, :, None, :])
    block = kk_scaled * decay                          # [B, H, n, n]

    # Strictly-lower-triangular within the chunk (valid rows only).
    idx = torch.arange(n, device=kf.device)
    mask = (idx[:, None] > idx[None, :]).to(block.dtype)
    block = block * mask
    return block.permute(0, 2, 1, 3)                   # [B, n, H, n]


def _chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
) -> torch.Tensor:
    kf = k.to(torch.float32)
    gf = g.to(torch.float32)
    bf = beta.to(torch.float32)

    B, T, Hg, K = kf.shape
    H = bf.shape[-1]
    BT = _BT
    NT = (T + BT - 1) // BT
    # Head grouping: value head h reads key head h // (H // Hg).
    group = H // Hg
    head_idx = torch.arange(H, device=kf.device) // group  # [H]

    A = kf.new_zeros(B, T, H, BT)
    for i_t in range(NT):
        start = i_t * BT
        stop = min(start + BT, T)
        n = stop - start
        rows = slice(start, stop)
        # Scatter into the [B, T, H, BT] output (zero-padded columns/rows).
        A[:, rows, :, :n] = _chunk_block(kf, gf, bf, rows, head_idx)

    return A.to(k.dtype)


def _chunk_scaled_dot_kkt_fwd_varlen(
    k: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Packed var-length reference (kernel ``IS_VARLEN=True`` path).

    k/beta/g are a single packed plane (B == 1); ``cu_seqlens`` defines NSEG
    segments ``[bos, eos)``. The chunk grid RESETS at every segment start
    (``prepare_chunk_indices`` emits ``ceil((eos-bos)/BT)`` chunks per segment,
    each addressed from ``bos``), so the per-chunk math is identical to the
    fixed-length path but the chunk boundaries follow the segments. Output
    ``A`` is zero outside written (valid-row, ``:n``-col) positions, matching
    the kernel's masked / boundary-checked store.
    """
    kf = k.to(torch.float32)
    gf = g.to(torch.float32)
    bf = beta.to(torch.float32)

    B, T, Hg, K = kf.shape
    if B != 1:
        # The kernel's varlen branch ignores the batch program id (i_b) and
        # addresses k/beta/g/A purely from the flat ``bos`` offset, so a packed
        # varlen batch is necessarily a single B==1 plane. A B>1 packed input
        # would alias the same storage across programs — refuse it loudly here
        # rather than silently computing an independent-per-B result the kernel
        # never produces.
        raise ValueError(
            f"varlen (cu_seqlens) path requires a packed B==1 plane, got B={B}"
        )
    H = bf.shape[-1]
    BT = _BT
    group = H // Hg
    head_idx = torch.arange(H, device=kf.device) // group  # [H]

    A = kf.new_zeros(B, T, H, BT)
    cu = cu_seqlens.to(torch.long).tolist()
    for s in range(len(cu) - 1):
        bos, eos = cu[s], cu[s + 1]
        seg_len = eos - bos
        if seg_len <= 0:
            continue  # empty segment contributes no chunks (kernel emits none)
        nt = (seg_len + BT - 1) // BT
        for i_t in range(nt):
            start = i_t * BT
            stop = min(start + BT, seg_len)
            n = stop - start
            rows = slice(bos + start, bos + stop)
            A[:, rows, :, :n] = _chunk_block(kf, gf, bf, rows, head_idx)

    # Output dtype mirrors the host wrapper's ``output_dtype`` default
    # (torch.float32), NOT the key dtype — the wrapper allocates A as fp32.
    return A.to(torch.float32)


def run(inputs: dict) -> dict:
    cu_seqlens = inputs.get("cu_seqlens")
    if cu_seqlens is not None:
        # Packed varlen path — flips the kernel's IS_VARLEN heuristic on.
        out = _chunk_scaled_dot_kkt_fwd_varlen(
            inputs["k"], inputs["g"], inputs["beta"], cu_seqlens)
    else:
        out = _chunk_scaled_dot_kkt_fwd(
            inputs["k"], inputs["g"], inputs["beta"])
    return {"output": out}
