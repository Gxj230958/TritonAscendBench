"""Oracle reference for L3_01_chunk_scan_fwd_kernel.

Tier: formula_exact_manual
Tolerance class: recurrent_chunk
Category: ssm

The kernel under test is the GPU triton implementation vendored in kernel.py
(host wrapper ``_chunk_scan_fwd`` launching ``_chunk_scan_fwd_kernel``); this
file authors an eager PyTorch reference from the operation's documented
mathematical definition (no upstream code copy).

ORACLE-SCOPE: covers intra (causal in-chunk SSM attention) + inter (off-diag
inter-chunk state contribution) + D skip-connection + Z output gating. The
per-chunk ENTRY ``states`` carry the inter-chunk recurrence; with
initial_states=None the only effect of ``seq_idx`` is to ZERO the previous
state at a sequence restart (seq_idx[c] != seq_idx[c-1]).

Contract (matches kernel.py::_chunk_scan_fwd, vLLM commit 18f6bf5):
  Fixture primitives (shapes.yaml):
    x:  [seqlen, nheads, headdim]   float32
    C:  [seqlen, ngroups, dstate]   float32
    z:  [seqlen, nheads, headdim]   float32   (HAS_Z gating)
    D:  [nheads, headdim]           float32   (HAS_D skip, per-headdim)
    chunk_size: int scalar (power of two)
  Derived (this file + _pa_launcher.py, byte-identical builders):
    cu_chunk_seqlens: [nchunks+1] int32 varlen chunk partition
    seq_idx:          [nchunks]   int32 per-chunk sequence id
    cb:        [nchunks, ngroups, chunk_size, chunk_size]   (= C·B^T per chunk)
    dt:        [nheads, nchunks, chunk_size]   (>=0 discretisation step)
    dA_cumsum: [nheads, nchunks, chunk_size]   (per-chunk NON-MONOTONE decay
               state, bounded in [-6, 6] so the intra min(dA_m-dA_k, 0) clamp
               is exercised on the causal region)
    states:    [nchunks, nheads, headdim, dstate]   (per-chunk entry state)

Per chunk c (token rows [s, e), len_c = e - s), head h, group
g = h // (nheads // ngroups), within-chunk row m in [0, len_c), headdim col n:
  scale_m   = exp2(LOG2E * dA_cumsum[h, c, m])    # kernel fast_exp, not exp
  prev      = zeros if seq_idx[c] != seq_idx[c-1] (or c == 0), else states[c-1, h]
  inter[m]  = (C[s+m, g, :] @ prev[:, n]) * scale_m              # over dstate
  seg[m,k]  = exp2(LOG2E * min(dA_cumsum[h,c,m] - dA_cumsum[h,c,k], 0)) * (m >= k)
  intra[m]  = sum_k cb[c,g,m,k] * seg[m,k] * dt[h,c,k] * x[s+k, h, n]
  out[s+m,h,n] = intra + inter
  if HAS_D: out += x[s+m,h,n] * D[h,n]
  if HAS_Z: out *= z[s+m,h,n] * sigmoid(z[s+m,h,n])
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _bf16(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 operands to bf16 (the activation dtype), matching the kernel's
    bf16-operand tl.dot. (Named _bf16; an earlier docstring wrongly said TF32 — the
    body has always been a bf16 roundtrip, GPU-verified correct at the anchors.)"""
    return x.to(torch.bfloat16).to(torch.float32)


def _partition(seqlen: int, chunk_size: int) -> tuple[list[int], list[int]]:
    """Deterministic NON-UNIFORM varlen chunk partition for a contiguous token
    stream, mirroring upstream vLLM ``compute_varlen_chunk_metadata``: a
    logical chunk never crosses a physical ``chunk_size`` boundary nor a
    sequence boundary.

    Returns ``(cu_chunk_seqlens, seq_idx)`` where ``cu_chunk_seqlens`` is the
    exclusive prefix-sum of logical-chunk lengths (leading 0, last == seqlen)
    and ``seq_idx[c]`` is the sequence id chunk c belongs to. The non-uniform
    sequence lengths make the partition irregular (chunk lengths are not all
    ``chunk_size``) AND span multiple chunks, so (a) a kernel that ignores
    ``cu_chunk_seqlens`` and assumes regular splits produces wrong values, and
    (b) most chunks share seq_idx with their predecessor, keeping the
    inter-chunk ``states`` term load-bearing (prev_states is the real
    states[c-1], not the seq-restart zeros).

    NOTE: byte-identical to the copy in the sibling module -- both sides derive
    the partition from the SAME (seqlen, chunk_size), so they cannot diverge;
    any drift surfaces as a non-zero oracle-vs-kernel diff and BLOCKS the task
    (fail-closed), never a silent pass.
    """
    cs = int(chunk_size)
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
    chunk_lens: list[int] = []
    seq_idx: list[int] = []
    pos = 0
    for sid, length in enumerate(seq_lens):
        end = pos + length
        p = pos
        while p < end:
            room = cs - (p % cs)
            take = min(room, end - p)
            chunk_lens.append(take)
            seq_idx.append(sid)
            p += take
        pos = end
    cu = [0]
    for c in chunk_lens:
        cu.append(cu[-1] + c)
    assert cu[-1] == int(seqlen), (cu[-1], seqlen)
    assert all(0 < c <= cs for c in chunk_lens), chunk_lens
    assert len(seq_idx) == len(chunk_lens), (len(seq_idx), len(chunk_lens))
    # seq_idx must contain at least one adjacent same-sequence pair, else every
    # chunk is a sequence-start and the inter-chunk ``states`` term is dead.
    assert any(
        seq_idx[c] == seq_idx[c - 1] for c in range(1, len(seq_idx))
    ), f"degenerate seq_idx (no multi-chunk sequence): {seq_idx}"
    return cu, seq_idx


def _per_chunk_seed(seqlen: int, nheads: int, ngroups: int,
                    headdim: int, dstate: int, chunk_size: int) -> int:
    """Deterministic seed for the per-chunk tensor RNG, derived from the shape
    so launcher and oracle generate IDENTICAL per-chunk inputs.

    NOTE: byte-identical to the copy in the sibling module.
    """
    h = 0x9E3779B97F4A7C15
    for v in (seqlen, nheads, ngroups, headdim, dstate, chunk_size):
        h = (h * 0x100000001B3) ^ (int(v) + 0x632BE59BD9B4E019)
        h &= (1 << 63) - 1
    return h


def build_chunk_scan_inputs(inputs: dict) -> dict:
    """Materialise the full ``_chunk_scan_fwd`` argument set from the fixture
    primitives. Per-chunk tensors (cb, dt, dA_cumsum, states) are generated
    with a shape-derived seed so both sides agree byte-for-byte.

    NOTE: byte-identical to the copy in the sibling module.
    """
    x = inputs["x"]                       # [seqlen, nheads, headdim] fp32
    C = inputs["C"]                       # [seqlen, ngroups, dstate] fp32
    z = inputs.get("z")                   # [seqlen, nheads, headdim] fp32
    D = inputs.get("D")                   # [nheads, headdim] fp32
    chunk_size = int(inputs["chunk_size"])

    seqlen, nheads, headdim = x.shape
    _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0, (nheads, ngroups)

    cu, seq_idx_list = _partition(seqlen, chunk_size)
    nchunks = len(cu) - 1

    # CPU generator so the CPU oracle (R10.a) and GPU launcher (R10.b) draw the
    # same derived inputs (a CUDA generator yields a different stream per seed).
    gen = torch.Generator()
    gen.manual_seed(_per_chunk_seed(
        seqlen, nheads, ngroups, headdim, dstate, chunk_size))

    # cb = (C @ B^T) per chunk -> [nchunks, ngroups, chunk_size, chunk_size].
    cb = torch.randn(nchunks, ngroups, chunk_size, chunk_size,
                     generator=gen, dtype=torch.float32).to(x.device)
    # dt (post softplus/clamp/tail-mask) >= 0 -> [nheads, nchunks, chunk_size].
    dt = torch.rand(nheads, nchunks, chunk_size,
                    generator=gen, dtype=torch.float32).to(x.device)
    # dA_cumsum -> [nheads, nchunks, chunk_size]. Deliberately NON-MONOTONE: a
    # bounded mixed-sign random walk along the chunk axis, NOT a cumsum of a
    # single-sign dt*A. Rationale: the kernel's intra decay is
    # exp(min(dA_m - dA_k, 0)); a monotone-decreasing dA would make dA_m <= dA_k
    # for every causal entry (m >= k), so dA_m - dA_k <= 0 and the min-clamp
    # would be a no-op -- a kernel that OMITTED the clamp would then pass. A
    # non-monotone dA produces causal entries with dA_m > dA_k (clamped to 0),
    # so the clamp is load-bearing and discriminating. Bounded to keep
    # exp(dA_m) for the inter scale finite.
    steps = (torch.rand(nheads, nchunks, chunk_size,
                        generator=gen, dtype=torch.float32).to(x.device)
             - 0.5)  # mixed-sign in [-0.5, 0.5)
    dA_cumsum = torch.cumsum(steps, dim=-1).clamp(-6.0, 6.0)
    # per-chunk ENTRY states -> [nchunks, nheads, headdim, dstate].
    states = torch.randn(nchunks, nheads, headdim, dstate,
                         generator=gen, dtype=torch.float32).to(x.device)

    # Discrimination guard: at least one CAUSAL (m >= k) entry with
    # dA_m - dA_k > 0 must exist, else the min(.,0) clamp is a no-op on the
    # retained region and a clamp-omitting kernel could not be distinguished.
    cs = chunk_size
    tri = torch.tril(torch.ones(cs, cs, dtype=torch.bool, device=x.device))
    diff = dA_cumsum[..., :, None] - dA_cumsum[..., None, :]   # [H, nc, cs, cs]
    assert (diff[..., tri] > 1e-3).any(), (
        "dA_cumsum is effectively monotone -- the intra min-clamp is not "
        "exercised on the causal region (clamp non-discriminating)")

    seq_idx = torch.tensor(seq_idx_list, dtype=torch.int32, device=x.device)
    cu_chunk_seqlens = torch.tensor(cu, dtype=torch.int32, device=x.device)

    return {
        "cb": cb, "x": x, "dt": dt, "dA_cumsum": dA_cumsum, "C": C,
        "states": states, "cu_chunk_seqlens": cu_chunk_seqlens,
        "seq_idx": seq_idx, "D": D, "z": z,
        "nchunks": nchunks, "chunk_size": chunk_size,
    }


def _chunk_scan_fwd(built: dict) -> torch.Tensor:
    cb = built["cb"].to(torch.float32)          # [nc, G, cs, cs]
    x = built["x"].to(torch.float32)            # [S, H, P]
    dt = built["dt"].to(torch.float32)          # [H, nc, cs]
    dAc = built["dA_cumsum"].to(torch.float32)  # [H, nc, cs]
    C = built["C"].to(torch.float32)            # [S, G, N]
    states = built["states"].to(torch.float32)  # [nc, H, P, N]
    cu = built["cu_chunk_seqlens"].tolist()
    seq_idx = built["seq_idx"].tolist()
    D = built["D"]
    z = built["z"]

    seqlen, nheads, headdim = x.shape
    _, ngroups, dstate = C.shape
    nchunks = len(cu) - 1
    ratio = nheads // ngroups
    device = x.device

    out = torch.zeros(seqlen, nheads, headdim, dtype=torch.float32, device=device)

    for c in range(nchunks):
        s, e = cu[c], cu[c + 1]
        len_c = e - s
        if len_c <= 0:
            continue
        # prev state: seq restart (or c==0) -> zeros, else states[c-1].
        if c == 0 or seq_idx[c] != seq_idx[c - 1]:
            prev = None
        else:
            prev = states[c - 1]                # [H, P, N]

        for h in range(nheads):
            g = h // ratio
            dA = dAc[h, c, :len_c]              # [len_c]
            # Kernel uses fast_exp = exp2(LOG2E*x) (single ex2.approx.f32),
            # NOT torch.exp; model both decay/scale exp sites the same way.
            scale_m = torch.exp2(1.4426950408889634 * dA)  # [len_c]
            # INTER: (C[s:e, g] @ prev[h].T) * scale_m  -> [len_c, P]
            if prev is None:
                inter = torch.zeros(len_c, headdim, dtype=torch.float32,
                                    device=device)
            else:
                # C: [len_c, N]; prev[h]: [P, N] -> [len_c, P]
                inter = (_bf16(C[s:e, g, :]) @ _bf16(prev[h].transpose(0, 1))) \
                    * scale_m[:, None]
            # INTRA: causal in-chunk SSM attention.
            # seg[m,k] = exp(min(dA[m] - dA[k], 0)) * (m >= k)
            diff = dA[:, None] - dA[None, :]            # [len_c, len_c]
            seg = torch.exp2(1.4426950408889634 * torch.clamp(diff, max=0.0))
            causal = torch.tril(torch.ones(len_c, len_c, dtype=torch.bool,
                                           device=device))
            seg = torch.where(causal, seg, torch.zeros_like(seg))
            cb_c = cb[c, g, :len_c, :len_c]            # [len_c, len_c]
            dt_c = dt[h, c, :len_c]                     # [len_c]
            w = cb_c * seg * dt_c[None, :]             # [m, k]
            intra = _bf16(w) @ _bf16(x[s:e, h, :])                   # [len_c, P]

            acc = intra + inter
            if D is not None:
                acc = acc + x[s:e, h, :] * D[h].to(torch.float32)[None, :]
            if z is not None:
                zc = z[s:e, h, :].to(torch.float32)
                acc = acc * (zc * torch.sigmoid(zc))
            out[s:e, h, :] = acc

    return out.to(built["x"].dtype)


def run(inputs: dict) -> dict:
    built = build_chunk_scan_inputs(inputs)
    return {"output": _chunk_scan_fwd(built)}
