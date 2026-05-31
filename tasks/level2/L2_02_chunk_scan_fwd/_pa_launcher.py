"""Authored launcher for L2_02_chunk_scan_fwd.

Drives the in-module host wrapper ``_chunk_scan_fwd`` (kernel.py) -- which
launches the graded @triton.jit ``_chunk_scan_fwd_kernel`` -- exactly as the
upstream Mamba2 SSD call site does (vLLM commit
18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19,
vllm/model_executor/layers/mamba/ops/ssd_combined.py::_mamba_chunk_scan_combined_fwd).

Why a launcher rather than the host wrapper alone: ``_chunk_scan_fwd`` consumes
a varlen chunk partition (``cu_chunk_seqlens``, ``seq_idx``) and per-chunk
tensors (``cb``, ``dt``, ``dA_cumsum``, ``states``) whose chunk-axis length is
DATA-DEPENDENT on (seqlen, chunk_size) -- shapes.yaml cannot materialise a
static chunk count. The launcher reproduces upstream's
``compute_varlen_chunk_metadata`` (mamba2_attn.py) partition for the fixture's
contiguous token stream, derives the per-chunk ``seq_idx``, materialises the
per-chunk tensors deterministically, calls the wrapper verbatim and harvests
the in-place ``out`` it writes.

The ``_partition`` / ``build_chunk_scan_inputs`` helpers below are byte-
identical to the copies in oracle_pytorch.py. Both sides derive the SAME
partition / seq_idx / per-chunk tensors from the SAME (seqlen, nheads,
ngroups, headdim, dstate, chunk_size), so they cannot diverge -- any drift
would make the correct kernel mismatch the oracle and BLOCK the task
(fail-closed), never a silent pass.
"""
import torch


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

    gen = torch.Generator(device=x.device)
    gen.manual_seed(_per_chunk_seed(
        seqlen, nheads, ngroups, headdim, dstate, chunk_size))

    # cb = (C @ B^T) per chunk -> [nchunks, ngroups, chunk_size, chunk_size].
    cb = torch.randn(nchunks, ngroups, chunk_size, chunk_size,
                     generator=gen, dtype=torch.float32, device=x.device)
    # dt (post softplus/clamp/tail-mask) >= 0 -> [nheads, nchunks, chunk_size].
    dt = torch.rand(nheads, nchunks, chunk_size,
                    generator=gen, dtype=torch.float32, device=x.device)
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
                        generator=gen, dtype=torch.float32, device=x.device)
             - 0.5)  # mixed-sign in [-0.5, 0.5)
    dA_cumsum = torch.cumsum(steps, dim=-1).clamp(-6.0, 6.0)
    # per-chunk ENTRY states -> [nchunks, nheads, headdim, dstate].
    states = torch.randn(nchunks, nheads, headdim, dstate,
                         generator=gen, dtype=torch.float32, device=x.device)

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


def run(inputs: dict) -> dict:
    built = build_chunk_scan_inputs(inputs)
    out = torch.empty_like(built["x"])

    _chunk_scan_fwd(  # noqa: F821 (exec'd into kernel module namespace)
        built["cb"],
        built["x"],
        built["dt"],
        built["dA_cumsum"],
        built["C"],
        built["states"],
        built["cu_chunk_seqlens"],
        out,
        built["seq_idx"],
        D=built["D"],
        z=built["z"],
        initial_states=None,
    )
    return {"output": out}
