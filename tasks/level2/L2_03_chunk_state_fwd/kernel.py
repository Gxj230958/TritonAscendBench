"""Task L2_03_chunk_state_fwd_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_chunk_state_fwd_kernel`
Source path: `vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): 3a81942a6853b21e69cbf7344f6a9504ff8a7b104a203827119ca11b53929f07
Frozen by: scripts/phase_a_freeze.py
"""
import torch
from tasks._frozen_deps_pool.h_119bf34e8937.vllm_model_executor_layers_mamba_ops_triton_helpers import fast_exp  # Phase A': U2 pool-vendored triton helper
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
from triton_ascend_bench.upstream_shims.triton_autotune_interp import make_triton_clone as _pa_make_triton_clone  # Phase A': interpreter-aware autotune fallback
triton = _pa_make_triton_clone()
from tasks._frozen_shim import softplus  # Phase A': off-path mamba softplus (B-inert-shim)

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=2, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=2, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2)], key=['hdim', 'dstate', 'chunk_size'])
@triton.jit
def _chunk_state_fwd_kernel(x_ptr, b_ptr, states_ptr, dt_ptr, dA_cumsum_ptr, cu_chunk_seqlens_ptr, hdim: tl.constexpr, dstate: tl.constexpr, chunk_size: tl.constexpr, seqlen, nheads_ngroups_ratio: tl.constexpr, stride_x_seqlen: tl.int64, stride_x_head: tl.int64, stride_x_hdim: tl.constexpr, stride_b_seqlen: tl.int64, stride_b_head: tl.int64, stride_b_dstate: tl.constexpr, stride_states_chunk: tl.int64, stride_states_head: tl.int64, stride_states_hdim: tl.int64, stride_states_dstate: tl.constexpr, stride_dt_head: tl.int64, stride_dt_chunk: tl.int64, stride_dt_csize: tl.constexpr, stride_dA_cs_head: tl.int64, stride_dA_cs_chunk: tl.int64, stride_dA_cs_csize: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_c = tl.program_id(axis=1).to(tl.int64)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)
    b_ptr += chunk_seqlen_start * stride_b_seqlen + pid_h // nheads_ngroups_ratio * stride_b_head
    x_ptr += chunk_seqlen_start * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        scale = fast_exp(tl.minimum(dA_cs_last - dA_cs_k, 0.0)) * dt_k
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
    states = acc.to(states_ptr.dtype.element_ty)
    states_ptr += pid_c * stride_states_chunk + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate)
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)

def _chunk_state_fwd(B, x, dt, dA_cumsum, cu_chunk_seqlens, states=None, states_in_fp32=True):
    seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (seqlen, ngroups, dstate)
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    if states is not None:
        assert states.shape == (nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty((nchunks, nheads, headdim, dstate), device=x.device, dtype=states_dtype)
    grid = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']), nchunks, nheads)
    with torch.accelerator.device_index(x.device.index):
        _chunk_state_fwd_kernel[grid](x_ptr=x, b_ptr=B, states_ptr=states, dt_ptr=dt, dA_cumsum_ptr=dA_cumsum, cu_chunk_seqlens_ptr=cu_chunk_seqlens, hdim=headdim, dstate=dstate, chunk_size=chunk_size, seqlen=seqlen, nheads_ngroups_ratio=nheads // ngroups, stride_x_seqlen=x.stride(0), stride_x_head=x.stride(1), stride_x_hdim=x.stride(2), stride_b_seqlen=B.stride(0), stride_b_head=B.stride(1), stride_b_dstate=B.stride(2), stride_states_chunk=states.stride(0), stride_states_head=states.stride(1), stride_states_hdim=states.stride(2), stride_states_dstate=states.stride(3), stride_dt_head=dt.stride(0), stride_dt_chunk=dt.stride(1), stride_dt_csize=dt.stride(2), stride_dA_cs_head=dA_cumsum.stride(0), stride_dA_cs_chunk=dA_cumsum.stride(1), stride_dA_cs_csize=dA_cumsum.stride(2))
    return states

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
