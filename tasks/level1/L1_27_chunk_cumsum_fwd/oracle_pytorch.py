"""Oracle reference for L1_27_chunk_cumsum_fwd_kernel.

Tier: formula_exact_manual
Tolerance class: recurrent_chunk
Category: ssm

The kernel under test is the GPU triton implementation vendored in
kernel.py (host wrapper ``_chunk_cumsum_fwd`` launching
``_chunk_cumsum_fwd_kernel``); this file authors an eager PyTorch
reference from the operation's documented mathematical definition
(no upstream code copy).

Contract (matches kernel.py::_chunk_cumsum_fwd):
  dt:                 [seqlen, nheads]              float32
  A:                  [nheads]                      float32
  chunk_size:         int scalar (power of two)
  cu_chunk_seqlens:   [nchunks + 1] int32 exclusive prefix-sum of
                      logical-chunk lengths (leading 0, last == seqlen,
                      each chunk length in (0, chunk_size]).
  dt_softplus:        bool (the DTSP knob)
  outputs:
    dt_out:           [nheads, nchunks, chunk_size]  float32
    dA_cumsum:        [nheads, nchunks, chunk_size]  float32

Per logical chunk c (token rows [s, e), len_c = e - s) and head h, over
the chunk_size axis k in [0, chunk_size) the kernel computes
(kernel.py:100-129):
  1. dt_k = dt[s + k, h]   for k < len_c, else 0.0
  2. if dt_softplus:  dt_k = softplus(dt_k) where dt_k <= 20, else dt_k
  3. dt_k = clamp(dt_k, dt_min, dt_max)            (dt_limit, default (0, inf))
  4. dt_k = dt_k if k < len_c else 0.0             (RE-MASK the chunk tail)
  5. dt_out[h, c, k] = dt_k
  6. dA_cumsum[h, c, :] = cumsum(dt_k * A[h]) over the chunk_size axis
The chunk-tail re-mask (step 4) zeroes positions k >= len_c, so dt_out's
tail is 0 and the dA_cumsum tail carries the final in-chunk prefix value.
The cumsum RESETS at every chunk boundary -- it is per-chunk, not a single
scan over the whole sequence; cu_chunk_seqlens defines those boundaries.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# NOTE: this helper is duplicated byte-for-byte in _pa_launcher.py. Both
# sides derive cu_chunk_seqlens from the SAME (seqlen, chunk_size), so they
# cannot diverge; any drift would surface as a non-zero oracle-vs-kernel
# diff and BLOCK the task (fail-closed), never a silent pass.
def build_cu_chunk_seqlens(seqlen: int, chunk_size: int) -> list[int]:
    """Deterministic NON-UNIFORM chunk-boundary metadata for a contiguous
    token stream of ``seqlen`` tokens, mirroring upstream vLLM
    ``compute_varlen_chunk_metadata`` (mamba2_attn.py): a logical chunk never
    crosses a physical ``chunk_size`` boundary nor a sequence boundary.

    A fixed, varied set of sequence lengths makes the partition non-uniform
    (chunk lengths are not all ``chunk_size``), so the per-chunk-cumsum
    reset is load-bearing: a kernel that ignores cu_chunk_seqlens and assumes
    regular ``chunk_size`` splits produces a different chunk count / values.
    """
    cs = int(chunk_size)
    # Non-uniform per-sequence lengths (sum clamped to exactly seqlen).
    pattern = [cs + cs // 2, cs // 2, 2 * cs, (cs * 3) // 4, cs - 1]
    seq_lens: list[int] = []
    rem = int(seqlen)
    i = 0
    while rem > 0:
        length = pattern[i % len(pattern)] or rem
        length = min(length, rem)
        seq_lens.append(length)
        rem -= length
        i += 1
    # Chunk the contiguous stream at physical chunk_size boundaries.
    chunk_lens: list[int] = []
    pos = 0
    for length in seq_lens:
        end = pos + length
        p = pos
        while p < end:
            room = cs - (p % cs)
            take = min(room, end - p)
            chunk_lens.append(take)
            p += take
        pos = end
    cu = [0]
    for c in chunk_lens:
        cu.append(cu[-1] + c)
    # Fail-closed invariants (mirror the launcher's).
    assert cu[-1] == int(seqlen), (cu[-1], seqlen)
    assert all(0 < c <= cs for c in chunk_lens), chunk_lens
    return cu


def _chunk_cumsum_fwd(
    dt: torch.Tensor, A: torch.Tensor, chunk_size: int,
    cu_chunk_seqlens: list[int], dt_bias: torch.Tensor | None,
    dt_softplus: bool, dt_limit: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    seqlen, nheads = dt.shape
    cs = int(chunk_size)
    nchunks = len(cu_chunk_seqlens) - 1
    df = dt.to(torch.float32)
    Af = A.to(torch.float32)
    bias = dt_bias.to(torch.float32) if dt_bias is not None else None
    dt_min, dt_max = float(dt_limit[0]), float(dt_limit[1])

    dt_out = torch.zeros(nheads, nchunks, cs, dtype=torch.float32, device=dt.device)
    dA_cumsum = torch.zeros(nheads, nchunks, cs, dtype=torch.float32, device=dt.device)
    pos_idx = torch.arange(cs, device=dt.device)

    for c in range(nchunks):
        s, e = cu_chunk_seqlens[c], cu_chunk_seqlens[c + 1]
        len_c = e - s
        block = torch.zeros(nheads, cs, dtype=torch.float32, device=dt.device)
        if len_c > 0:
            block[:, :len_c] = df[s:e, :].transpose(0, 1)  # [nheads, len_c]
        if bias is not None:
            block = block + bias[:, None]
        if dt_softplus:
            block = torch.where(block <= 20.0, F.softplus(block), block)
        block = block.clamp(dt_min, dt_max)
        # Re-mask the chunk tail to 0 (kernel.py:114-116): this overrides any
        # softplus(0)=log(2) / clamp value at positions k >= len_c.
        mask = (pos_idx < len_c).view(1, cs)
        block = torch.where(mask, block, torch.zeros_like(block))
        dt_out[:, c, :] = block
        dA_cumsum[:, c, :] = torch.cumsum(block * Af[:, None], dim=1)

    return dt_out, dA_cumsum


def run(inputs: dict) -> dict:
    dt = inputs["dt"]
    A = inputs["A"]
    chunk_size = int(inputs["chunk_size"])
    seqlen = int(dt.shape[0])
    cu = build_cu_chunk_seqlens(seqlen, chunk_size)
    # Guard the non-degeneracy the fixture relies on (see shapes.yaml): the
    # genuine partition must differ from a uniform ceil(seqlen/chunk_size)
    # split, otherwise a cu-ignoring kernel could not be distinguished.
    assert (len(cu) - 1) != math.ceil(seqlen / chunk_size), (
        f"degenerate (uniform) partition for seqlen={seqlen} "
        f"chunk_size={chunk_size}; pick seqlen so the non-uniform chunk "
        f"count differs from ceil(seqlen/chunk_size)"
    )
    dl = inputs.get("dt_limit")
    dt_limit = tuple(dl) if dl is not None else (0.0, float("inf"))
    dt_out, dA_cumsum = _chunk_cumsum_fwd(
        dt, A, chunk_size, cu, inputs.get("dt_bias"),
        bool(inputs.get("dt_softplus", False)), dt_limit,
    )
    return {"dt_out": dt_out, "dA_cumsum": dA_cumsum}
