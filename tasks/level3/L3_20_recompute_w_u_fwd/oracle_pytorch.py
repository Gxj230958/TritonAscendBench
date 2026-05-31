"""Oracle reference for L3_20_recompute_w_u_fwd_kernel.

Tier: formula_exact_manual
Tolerance class: elementwise
Category: other

Independent eager-PyTorch reference for the vLLM ``recompute_w_u_fwd``
GDN/KDA W/U recompute (flash-linear-attention ``wy_fast``). Authored from
the frozen Triton kernel's contract; never imports or calls the kernel.

Upstream contract (non-varlen path, ``cu_seqlens is None``):

The kernel is launched on grid ``(NT, B*H)`` with ``NT = cdiv(T, BT)`` and
processes one ``BT``-sized chunk of the sequence per ``(batch, head)``.
``A`` is the per-chunk transition matrix of shape ``[B, T, H, BT]`` whose
last axis is the *local* (within-chunk) column index, so ``A`` mixes rows
only WITHIN a chunk — it is block-diagonal along the time axis. The key
tensor carries ``Hg`` (grouped/GQA) heads; value head ``h`` reads key head
``h // (H // Hg)``.

For a chunk ``c`` (rows ``r = c*BT + i``) and head ``h``::

    u[b, r, h, :] = sum_j A[b, r, h, j] * (beta[b, c*BT+j, h] * v[b, c*BT+j, h, :])
    w[b, r, h, :] = sum_j A[b, r, h, j]
                        * (beta[b, c*BT+j, h] * exp(g[b, c*BT+j, h])
                           * k[b, c*BT+j, h//(H//Hg), :])

where ``j`` ranges over the local positions ``[0, BT)`` of chunk ``c``.

Tensor layouts (matching the kernel's ``make_block_ptr`` strides):
    k:        [B, T, Hg, K]
    v:        [B, T, H,  V]
    beta:     [B, T, H]
    g_cumsum: [B, T, H]   (kernel applies exp(g) for w; u uses no gate)
    A:        [B, T, H,  BT]
    -> w:     [B, T, H,  K]
    -> u:     [B, T, H,  V]

The kernel keeps everything in the input dtype with an fp32 dot accumulate.
The two dots differ in precision pinning, which the oracle MUST mirror:

* ``u`` dot (kernel.py:107) is pinned ``tl.dot(b_A, b_vb, allow_tf32=False)``
  -- IEEE/full-fp32 -- so the oracle computes ``u`` in full fp32.
* ``w`` dot (kernel.py:129) is ``tl.dot(b_A, b_kb)`` with NO precision pin.
  On the CUDA daemon (Ampere+/Hopper) Triton's default f32 ``tl.dot`` runs on
  the TF32 tensor cores: each fp32 operand is truncated to a 19-bit TF32 value
  (10 mantissa bits) before the multiply, with fp32 accumulate. The oracle
  therefore truncates BOTH ``w``-dot operands (the A matrix and the already
  gated key operand ``kb``) to TF32 (see ``_to_tf32``), matching the sibling
  fla oracles L3_18 / L3_19. This models the kernel's REAL hardware precision
  -- it is a faithful reference, not a narrowed one.
"""
from __future__ import annotations

import torch


def _to_tf32(x: torch.Tensor) -> torch.Tensor:
    """Round a float32 tensor to TF32 precision the way the kernel's GPU
    ``tl.dot`` does for the UNPINNED ``w`` dot.

    ``kernel.py`` line 129 computes ``b_w = tl.dot(b_A, b_kb)`` with no
    ``allow_tf32=False`` / ``input_precision="ieee"`` pin (contrast the ``u``
    dot at line 107, which IS pinned). On Ampere+ an unpinned float32
    ``tl.dot`` runs on the TF32 tensor cores: each float32 operand is reduced
    to a 19-bit TF32 value (8 exp + 10 mantissa bits) before the multiply,
    with float32 accumulate. Triton's default ``tl.dot`` TRUNCATES the low 13
    of the 23 float32 mantissa bits (it does not round-to-nearest-even), so we
    truncate. The sibling fla oracles L3_18 / L3_19 use the identical
    emulation. Tolerance (elementwise atol/rtol=1e-3) is unchanged; the
    residual oracle-vs-kernel diff is then only float32 accumulation-order
    noise.

    NOTE on the local CPU anchor: under ``TRITON_INTERPRET=1`` the interpreter
    ignores f32 ``input_precision`` and runs the dot in full fp32, so the
    CPU-interpret oracle-consistency anchor can report a (known, documented)
    false-negative for this fp32-output task. The GPU bit-exact re-run -- which
    runs the kernel's real TF32 dot -- is the authoritative confirm.
    """
    bits = x.contiguous().view(torch.int32)
    truncated = bits & ~((1 << 13) - 1)
    return truncated.view(torch.float32)


def _recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, V = v.shape
    K = k.shape[-1]
    Hg = k.shape[-2]
    BT = A.shape[-1]
    if T % BT != 0:
        raise ValueError(f"T={T} must be divisible by chunk size BT={BT}")
    if H % Hg != 0:
        raise ValueError(f"H={H} must be divisible by key-head count Hg={Hg}")
    NT = T // BT
    rep = H // Hg

    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    bf = beta.to(torch.float32)
    gf = g_cumsum.to(torch.float32)
    Af = A.to(torch.float32)

    # Fold the time axis into (chunk, local-position); A only ever mixes
    # within a chunk, so the contraction is block-diagonal per chunk.
    Ac = Af.reshape(B, NT, BT, H, BT)        # [B, NT, row, H, col]
    bc = bf.reshape(B, NT, BT, H)
    gc = gf.reshape(B, NT, BT, H)
    vc = vf.reshape(B, NT, BT, H, V)

    # Expand grouped key heads to the full head count, then chunk.
    kH = kf.repeat_interleave(rep, dim=2)    # [B, T, H, K]
    kc = kH.reshape(B, NT, BT, H, K)

    vb = bc.unsqueeze(-1) * vc                                   # [.., col, H, V]
    kb = (bc * torch.exp(gc)).unsqueeze(-1) * kc                 # [.., col, H, K]

    # u dot is IEEE-pinned in the kernel (allow_tf32=False, kernel.py:107),
    # so keep full fp32. The w dot is UNPINNED (kernel.py:129) and runs in
    # TF32 on the CUDA daemon, so truncate BOTH w-dot operands (the A matrix
    # and the already gated key operand kb) to TF32 before contracting.
    u = torch.einsum("bnrhc,bnchv->bnrhv", Ac, vb).reshape(B, T, H, V)
    w = torch.einsum(
        "bnrhc,bnchk->bnrhk", _to_tf32(Ac), _to_tf32(kb)
    ).reshape(B, T, H, K)

    return w.to(k.dtype), u.to(v.dtype)


def run(inputs: dict) -> dict:
    w, u = _recompute_w_u_fwd(
        inputs["k"],
        inputs["v"],
        inputs["beta"],
        inputs["g_cumsum"],
        inputs["A"],
    )
    return {"w": w, "u": u}
