"""Authored launcher for L2_03_chunk_state_fwd.

Drives the in-module host wrapper ``_chunk_state_fwd`` (kernel.py) -- which
launches the graded @triton.jit ``_chunk_state_fwd_kernel`` -- exactly as the
upstream Mamba2 SSD call site does (vLLM commit
18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19,
vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py).

Why a launcher rather than the bare jit kernel: ``_chunk_state_fwd_kernel`` is
launched over a 3-D grid (headdim/dstate tiles x nchunks x nheads) and reads
its operands through explicit strides plus a ``cu_chunk_seqlens`` chunk
partition; the daemon cannot synthesise that launch. The host wrapper
``_chunk_state_fwd`` allocates ``states`` and issues the launch with the
upstream grid/strides, so this launcher only has to MAP the fixture's
per-chunk primitives onto the wrapper's physical-axis layout and hand the
result back in the oracle's output layout.

Axis mapping (fixture -> upstream physical layout). The fixture (shapes.yaml)
materialises the Mamba2 chunk-state operands already partitioned by chunk:

    B          [C, L, N]   per-chunk state-projection
    x          [C, L, D]   per-chunk input
    dt         [C, L]      per-chunk discretisation step
    dA_cumsum  [C, L]      per-chunk cumulative decay

with C -> nchunks, L -> chunk_size, N -> dstate, D -> headdim, a SINGLE head
and a SINGLE group (nheads == ngroups == 1). The upstream wrapper consumes a
contiguous token stream split into uniform chunks, so the mapping is a pure
reshape (no data invented, no value changed):

    x  [C, L, D] -> [C*L, 1, D]   (seqlen=C*L, nheads=1, headdim=D)
    B  [C, L, N] -> [C*L, 1, N]   (seqlen=C*L, ngroups=1, dstate=N)
    dt [C, L]    -> [1, C, L]     (nheads=1, nchunks=C, chunk_size=L)
    dA [C, L]    -> [1, C, L]
    cu_chunk_seqlens = [0, L, 2L, ..., C*L]   (uniform chunk boundaries)

The wrapper writes ``states`` as [nchunks, nheads, headdim, dstate]; with
nheads==1 that is [C, 1, D, N]. The oracle's declared ``states`` contract is
[C, N, D] (einsum ``cln,clp->cnp`` over the SAME B/x/dt/dA), so the launcher
squeezes the head axis and transposes the trailing (headdim, dstate) pair to
(dstate, headdim) -- a layout reconciliation only, identical numbers. Any drift
between this mapping and the oracle's einsum surfaces as a non-zero
oracle-vs-kernel diff and BLOCKS the task (fail-closed), never a silent pass.
"""
import torch


def _to_upstream_inputs(inputs: dict) -> dict:
    """Reshape the fixture's per-chunk primitives into the upstream
    ``_chunk_state_fwd`` argument layout (nheads == ngroups == 1, uniform
    chunk_size == L). Pure reshape -- no value is altered."""
    B = inputs["B"].to(torch.float32)             # [C, L, N]
    x = inputs["x"].to(torch.float32)             # [C, L, D]
    dt = inputs["dt"].to(torch.float32)           # [C, L]
    dA_cumsum = inputs["dA_cumsum"].to(torch.float32)  # [C, L]

    C, L, N = B.shape
    Cx, Lx, D = x.shape
    assert (Cx, Lx) == (C, L), ((Cx, Lx), (C, L))
    assert dt.shape == (C, L) and dA_cumsum.shape == (C, L), (
        dt.shape, dA_cumsum.shape, (C, L))

    seqlen = C * L
    device = x.device

    # x: [C, L, D] -> [seqlen, nheads=1, headdim=D]; contiguous so a flat
    # reshape lays the C chunks of L tokens out as one stream.
    x_u = x.reshape(seqlen, 1, D).contiguous()
    # B: [C, L, N] -> [seqlen, ngroups=1, dstate=N].
    B_u = B.reshape(seqlen, 1, N).contiguous()
    # dt / dA_cumsum: [C, L] -> [nheads=1, nchunks=C, chunk_size=L].
    dt_u = dt.reshape(1, C, L).contiguous()
    dA_u = dA_cumsum.reshape(1, C, L).contiguous()
    # Uniform chunk partition: chunk c spans tokens [c*L, (c+1)*L).
    cu_chunk_seqlens = torch.arange(
        0, seqlen + 1, L, dtype=torch.int32, device=device)
    assert cu_chunk_seqlens.numel() == C + 1, cu_chunk_seqlens.numel()
    assert int(cu_chunk_seqlens[-1]) == seqlen, cu_chunk_seqlens

    return {
        "B": B_u, "x": x_u, "dt": dt_u, "dA_cumsum": dA_u,
        "cu_chunk_seqlens": cu_chunk_seqlens,
        "C": C, "L": L, "N": N, "D": D,
    }


def run(inputs: dict) -> dict:
    built = _to_upstream_inputs(inputs)

    states = _chunk_state_fwd(  # noqa: F821 (exec'd into kernel module namespace)
        built["B"],
        built["x"],
        built["dt"],
        built["dA_cumsum"],
        built["cu_chunk_seqlens"],
        states=None,
        states_in_fp32=True,
    )
    # states: [nchunks=C, nheads=1, headdim=D, dstate=N]. The oracle contract
    # is [C, N, D] (einsum cln,clp->cnp): drop the head axis, swap the trailing
    # (headdim, dstate) -> (dstate, headdim). Cast to the fixture dtype so the
    # anchor's dtype-equality check matches the oracle output.
    states = states[:, 0].transpose(1, 2).contiguous()  # [C, N, D]
    states = states.to(inputs["x"].dtype)
    return {"states": states}
