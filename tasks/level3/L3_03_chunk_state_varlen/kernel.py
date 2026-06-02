"""Task L3_03_chunk_state_varlen_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_chunk_state_varlen_kernel`
Source path: `sglang/srt/layers/attention/mamba/ops/ssd_chunk_state.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): dc2fed284d4fba3328b45e531a21ce61e94bc5895470acaa9a5ceb1a0fed1d24
Frozen by: scripts/phase_a_freeze.py
"""
import math
import torch
import triton
import triton.language as tl
from tasks._frozen_shim import softplus  # Phase A': off-path mamba softplus (B-inert-shim)

@triton.jit
def _chunk_state_fwd_kernel(x_ptr, b_ptr, states_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr, hdim, dstate, chunk_size, batch, seqlen, nheads_ngroups_ratio, stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim, stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate, stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate, stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize, stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize, stride_seq_idx_batch, stride_seq_idx_seqlen, HAS_SEQ_IDX: tl.constexpr, BLOCK_SIZE_M: tl.constexpr=16, BLOCK_SIZE_N: tl.constexpr=16, BLOCK_SIZE_K: tl.constexpr=16):
    pid_bc = tl.program_id(axis=1).to(tl.int64)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + pid_h // nheads_ngroups_ratio * stride_b_head
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    if HAS_SEQ_IDX:
        seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seqlen
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        if HAS_SEQ_IDX:
            seq_idx_k = tl.load(seq_idx_ptrs, mask=offs_k < chunk_size_limit - k, other=-1)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        if not HAS_SEQ_IDX:
            scale = tl.exp(dA_cs_last - dA_cs_k) * dt_k
        else:
            scale = tl.where(seq_idx_k == seq_idx_last, tl.exp(dA_cs_last - dA_cs_k) * dt_k, 0.0)
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
        if HAS_SEQ_IDX:
            seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seqlen
    states = acc.to(states_ptr.dtype.element_ty)
    states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate)
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)

@triton.jit
def _chunk_state_varlen_kernel(x_ptr, b_ptr, dt_ptr, dA_cumsum_ptr, chunk_states_ptr, cu_seqlens_ptr, states_ptr, initstates_ptr, hdim, dstate, chunk_size, seqlen, nheads_ngroups_ratio, stride_x_seqlen, stride_x_head, stride_x_hdim, stride_b_seqlen, stride_b_head, stride_b_dstate, stride_dt_chunk, stride_dt_head, stride_dt_csize, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize, stride_chunk_states_chunk, stride_chunk_states_head, stride_chunk_states_hdim, stride_chunk_states_dstate, stride_states_batch, stride_states_head, stride_states_hdim, stride_states_dstate, stride_init_states_batch, stride_init_states_head, stride_init_states_hdim, stride_init_states_dstate, HAS_INITSTATES: tl.constexpr, BLOCK_SIZE_M: tl.constexpr=16, BLOCK_SIZE_N: tl.constexpr=16, BLOCK_SIZE_K: tl.constexpr=16):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    end_idx = tl.load(cu_seqlens_ptr + pid_b + 1)
    pid_c = (end_idx - 1) // chunk_size
    b_ptr += pid_c * chunk_size * stride_b_seqlen + pid_h // nheads_ngroups_ratio * stride_b_head
    x_ptr += pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    chunk_states_ptr += pid_c * stride_chunk_states_chunk + pid_h * stride_chunk_states_head
    if HAS_INITSTATES:
        initstates_ptr += pid_h * stride_init_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (end_idx - pid_c * chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    chunk_size_limit = end_idx - pid_c * chunk_size
    start_idx = tl.load(cu_seqlens_ptr + pid_b)
    start_idx_cur = tl.maximum(start_idx - pid_c * chunk_size, 0)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k) & (offs_k[None, :] >= start_idx_cur - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate) & (offs_k[:, None] >= start_idx_cur - k), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        scale = tl.where((offs_k >= start_idx_cur - k) & (offs_k < chunk_size_limit - k), tl.exp(dA_cs_last - dA_cs_k) * dt_k, 0.0)
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
    if start_idx < pid_c * chunk_size or HAS_INITSTATES:
        dA_cs_boundary = 0.0
        if not HAS_INITSTATES:
            past_states_ptrs = chunk_states_ptr + (offs_m[:, None] * stride_chunk_states_hdim + offs_n[None, :] * stride_chunk_states_dstate)
        elif start_idx < pid_c * chunk_size:
            past_states_ptrs = chunk_states_ptr + (offs_m[:, None] * stride_chunk_states_hdim + offs_n[None, :] * stride_chunk_states_dstate)
        else:
            past_states_ptrs = initstates_ptr + (pid_b * stride_init_states_batch + offs_m[:, None] * stride_init_states_hdim + offs_n[None, :] * stride_init_states_dstate)
            if start_idx > pid_c * chunk_size:
                dA_cs_boundary = tl.load(dA_cumsum_ptr + (start_idx - pid_c * chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
        past_states = tl.load(past_states_ptrs, mask=(offs_m[:, None] < hdim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
        scale = tl.exp(dA_cs_last - dA_cs_boundary)
        acc += past_states * scale
    states = acc.to(states_ptr.dtype.element_ty)
    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate)
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)

def _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=None, states=None, states_in_fp32=True):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if states is not None:
        assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty((batch, nchunks, nheads, headdim, dstate), device=x.device, dtype=states_dtype)
    grid = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']), batch * nchunks, nheads)
    with torch.get_device_module(x.device).device(x.device.index):
        _chunk_state_fwd_kernel[grid](x, B, states, dt, dA_cumsum, seq_idx, headdim, dstate, chunk_size, batch, seqlen, nheads // ngroups, x.stride(0), x.stride(1), x.stride(2), x.stride(3), B.stride(0), B.stride(1), B.stride(2), B.stride(-1), states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4), dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3), dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3), *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)), HAS_SEQ_IDX=seq_idx is not None)
    return states

def chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states, initial_states=None):
    total_seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    batch = cu_seqlens.shape[0] - 1
    cu_seqlens = cu_seqlens.contiguous()
    assert nheads % ngroups == 0
    assert B.shape == (total_seqlen, ngroups, dstate)
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert chunk_states.shape == (nchunks, nheads, headdim, dstate)
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    states = torch.empty(batch, nheads, headdim, dstate, dtype=chunk_states.dtype, device=chunk_states.device)
    grid = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']), batch, nheads)
    with torch.get_device_module(x.device).device(x.device.index):
        _chunk_state_varlen_kernel[grid](x, B, dt, dA_cumsum, chunk_states, cu_seqlens, states, initial_states, headdim, dstate, chunk_size, total_seqlen, nheads // ngroups, x.stride(0), x.stride(1), x.stride(2), B.stride(0), B.stride(1), B.stride(2), dt.stride(1), dt.stride(0), dt.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(0), dA_cumsum.stride(2), chunk_states.stride(0), chunk_states.stride(1), chunk_states.stride(2), chunk_states.stride(3), states.stride(0), states.stride(1), states.stride(2), states.stride(3), *((initial_states.stride(0), initial_states.stride(1), initial_states.stride(2), initial_states.stride(3)) if initial_states is not None else (0, 0, 0, 0)), HAS_INITSTATES=initial_states is not None)
    return states

import torch


# Authored launcher: drives the DECLARED graded @triton.jit kernel
# `_chunk_state_varlen_kernel` through its in-module host wrapper
# `chunk_state_varlen` (sglang
# python/sglang/srt/layers/attention/mamba/ops/ssd_chunk_state.py, commit
# 409d350fb6f6a1e7c7546e39028f811092a8e489; adapted from vLLM / state-spaces
# Mamba2 SSD). The launcher never reimplements the SSD recurrence -- it only
# marshals the flat fixture tensors into the chunked varlen layout the wrapper
# requires and assembles the wrapper's return into the oracle contract key.
#
# Why a launcher rather than binding `chunk_state_varlen` directly: its
# signature is
#   chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states,
#                      initial_states=None)
# `chunk_states` has NO default and is NOT a primitive the benchmark fixture
# materialises -- it is the *cumulative* start-of-chunk SSD state that the
# upstream pipeline produces by running the per-chunk state forward
# (`_chunk_state_fwd`) and then carrying it across chunks within each sequence
# (state passing). The strict by-name adapter therefore rejects the wrapper as
# shapes-incomplete. This launcher reproduces that prerequisite from the SAME
# fixture tensors, then calls the wrapper verbatim.
#
# Faithful full-recurrence reconstruction (no single-chunk shortcut):
#   * The varlen kernel partitions the packed [total_seqlen] token stream into
#     chunks of `chunk_size`. dt / dA_cumsum are laid out [nheads, nchunks,
#     chunk_size] with a PER-CHUNK-LOCAL cumulative dA (reset each chunk); the
#     oracle's `dA_cumsum` fixture is a PER-SEQUENCE-GLOBAL cumulative. We
#     recover the per-token dA increments (delta) from the global cumulative,
#     re-chunk them, and take a per-chunk cumsum -- the exact `_chunk_cumsum_fwd`
#     output shape. (delta + per-chunk cumsum is an algebraic identity, so no
#     numeric model is introduced.)
#   * `chunk_states[c]` is the cumulative state at the START of chunk c: zero at
#     each sequence's first chunk, else the previous chunk's start-state decayed
#     by exp(dA_local_last[c-1]) plus the previous chunk's intra-chunk state
#     (`_chunk_state_fwd` output). This is the standard Mamba2 state-passing
#     carry. The graded kernel reads chunk_states[pid_c] (the sequence's last
#     chunk) ONCE and re-decays it to the chunk end -- so chunk_states MUST be
#     cumulative; feeding raw per-chunk states is the single-chunk shortcut and
#     mismatches the oracle for any sequence spanning >1 chunk (fault-injected
#     and confirmed to FAIL).
#
# chunk_size choice: chunk_size = floor(L/2) shrunk to a common divisor of every
# sequence length, so every sequence spans >= 2 chunks AND chunks never straddle
# a sequence boundary. The >= 2 chunks guarantee makes `chunk_states` LOAD-BEARING
# (the kernel's carried-state branch fires); the boundary alignment keeps each
# chunk's dA cumsum / state within one sequence (the varlen invariant).
#
# `_chunk_state_fwd` (the sibling host wrapper in this module) is NOT the graded
# kernel; it is the upstream prerequisite that produces the per-chunk intra
# states, called exactly as upstream does. The graded kernel is
# `_chunk_state_varlen_kernel`, launched by `chunk_state_varlen` below.
#
# Output key (states) matches the oracle contract exactly.
def run(inputs: dict) -> dict:
    Bt = inputs["B"]            # [total_seqlen, ngroups, dstate]
    x = inputs["x"]            # [total_seqlen, nheads, headdim]
    dt = inputs["dt"]           # [total_seqlen, nheads]
    dA = inputs["dA_cumsum"]    # [total_seqlen, nheads]  per-sequence-global cumsum
    cu = inputs["cu_seqlens"].to(torch.long)

    total_seqlen, nheads, headdim = x.shape
    ngroups, dstate = Bt.shape[1], Bt.shape[2]
    seglens = (cu[1:] - cu[:-1]).tolist()
    pos = [int(s) for s in seglens if s > 0]
    if not pos:
        # Degenerate: no tokens -> zero states for every batch slot.
        batch = cu.shape[0] - 1
        out = x.new_zeros((batch, nheads, headdim, dstate))
        return {"states": out}

    # chunk_size: floor(min_seqlen / 2) shrunk to a common divisor of all
    # positive sequence lengths (and hence of total_seqlen). Guarantees
    # >= 2 chunks per sequence (chunk_states load-bearing) and chunk/sequence
    # boundary alignment (varlen invariant).
    chunk_size = max(min(pos) // 2, 1)
    while chunk_size > 1 and any(s % chunk_size != 0 for s in pos):
        chunk_size -= 1
    assert total_seqlen % chunk_size == 0, (total_seqlen, chunk_size)
    nchunks = total_seqlen // chunk_size

    # Recover per-token dA increments from the per-sequence-global cumulative,
    # then re-chunk into [nheads, nchunks, chunk_size] and take a per-chunk
    # cumsum (the _chunk_cumsum_fwd layout the varlen kernel indexes).
    delta = dA.clone()
    for i in range(len(seglens)):
        a, b = int(cu[i]), int(cu[i + 1])
        if b - a > 1:
            delta[a + 1:b] = dA[a + 1:b] - dA[a:b - 1]

    def to_chunks(v):  # [T, H] -> [H, nchunks, chunk_size]
        return v.reshape(nchunks, chunk_size, nheads).permute(2, 0, 1).contiguous()

    dt_c = to_chunks(dt)
    dA_local = torch.cumsum(to_chunks(delta), dim=2)  # per-chunk-local cumsum

    # Per-chunk intra states via the upstream prerequisite host wrapper.
    x_b = x.reshape(1, total_seqlen, nheads, headdim)
    B_b = Bt.reshape(1, total_seqlen, ngroups, dstate)
    dt_b = dt_c.reshape(1, nheads, nchunks, chunk_size)
    dA_b = dA_local.reshape(1, nheads, nchunks, chunk_size)
    states_raw = _chunk_state_fwd(B_b, x_b, dt_b, dA_b)[0]  # noqa: F821 (exec'd into kernel module)
    # states_raw: [nchunks, nheads, headdim, dstate]

    # Cumulative start-of-chunk states (state passing), reset at each
    # sequence's first chunk.
    cu_set = set(int(c) for c in cu.tolist())
    chunk_states = torch.zeros_like(states_raw)
    for c in range(1, nchunks):
        if (c * chunk_size) in cu_set:  # this chunk begins a new sequence
            continue
        dA_last_prev = dA_local[:, c - 1, -1]  # [nheads]
        chunk_states[c] = (
            chunk_states[c - 1] * torch.exp(dA_last_prev)[:, None, None]
            + states_raw[c - 1]
        )

    out = chunk_state_varlen(  # noqa: F821 (exec'd into kernel module)
        B_b[0],
        x_b[0],
        dt_b[0],
        dA_b[0],
        cu.to(torch.int64),
        chunk_states.contiguous(),
    )
    return {"states": out}
