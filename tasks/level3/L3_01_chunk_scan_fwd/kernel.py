"""Task L3_01_chunk_scan_fwd_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_chunk_scan_fwd_kernel`
Source path: `vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): af58095ccee5c173a56395cb8fba4174ed7b0bec32bffbd713b1bf80c0332123
Frozen by: scripts/phase_a_freeze.py
"""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_scan.py

# ruff: noqa: E501,SIM102

from packaging import version

from triton_ascend_bench.frozen.deps.h_119bf34e8937.vllm_model_executor_layers_mamba_ops_triton_helpers import fast_exp  # Phase A': U2 pool-vendored triton helper
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
from triton_ascend_bench.upstream_shims.triton_autotune_interp import make_triton_clone as _pa_make_triton_clone  # Phase A': interpreter-aware autotune fallback
triton = _pa_make_triton_clone()

TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")


@triton.autotune(
    configs=[
        # =================================================================
        # Higher warp count configs for better latency hiding
        # More warps = more instructions in flight = better memory latency hiding
        # =================================================================
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=2,
            num_warps=8,
        ),
        # Smaller tiles with more stages for software pipelining
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64},
            num_stages=2,
            num_warps=4,
        ),
        # =================================================================
        # Low register pressure configs (num_stages=1) for large dstate
        # =================================================================
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=1,
            num_warps=4,
        ),
        # num_stages=2 configs - moderate register pressure
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        # Original configs for larger dstate values
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=2,
        ),
    ],
    key=["chunk_size", "hdim", "dstate", "IS_CAUSAL"],
)
@triton.jit
def _chunk_scan_fwd_kernel(
    # Pointers to matrices
    cb_ptr,
    x_ptr,
    z_ptr,
    out_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    seq_idx_ptr,
    C_ptr,
    states_ptr,
    D_ptr,
    initstates_ptr,
    cu_chunk_seqlens_ptr,
    # Matrix dimensions
    chunk_size: tl.constexpr,
    hdim: tl.constexpr,
    dstate: tl.constexpr,
    seqlen,
    nheads_ngroups_ratio: tl.constexpr,
    # Strides
    stride_cb_chunk: tl.int64,
    stride_cb_head: tl.int64,
    stride_cb_csize_m: tl.int64,
    stride_cb_csize_k: tl.constexpr,
    stride_x_seqlen: tl.int64,
    stride_x_head: tl.int64,
    stride_x_hdim: tl.constexpr,
    stride_z_seqlen: tl.int64,
    stride_z_head: tl.int64,
    stride_z_hdim: tl.constexpr,
    stride_out_seqlen: tl.int64,
    stride_out_head: tl.int64,
    stride_out_hdim: tl.constexpr,
    stride_dt_chunk: tl.int64,
    stride_dt_head: tl.int64,
    stride_dt_csize: tl.constexpr,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    stride_seq_idx_chunk: tl.constexpr,
    stride_C_seqlen: tl.int64,
    stride_C_head: tl.int64,
    stride_C_dstate: tl.constexpr,
    stride_states_chunk: tl.int64,
    stride_states_head: tl.int64,
    stride_states_hdim: tl.int64,
    stride_states_dstate: tl.constexpr,
    stride_init_states_batch: tl.int64,
    stride_init_states_head: tl.int64,
    stride_init_states_hdim: tl.int64,
    stride_init_states_dstate: tl.constexpr,
    stride_D_head: tl.constexpr,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
    HAS_INITSTATES: tl.constexpr,
):
    pid_c = tl.program_id(axis=1).to(tl.int64)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    cb_ptr += pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)
    x_ptr += chunk_seqlen_start * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    C_ptr += (
        chunk_seqlen_start * stride_C_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_C_head
    )

    # M-block offsets and prev states
    #  - logic in next block may override these if there is an active offset
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    seq_idx_ptr += pid_c * stride_seq_idx_chunk
    seq_idx = tl.load(seq_idx_ptr)
    seq_idx_prev = tl.load(
        seq_idx_ptr - stride_seq_idx_chunk, mask=pid_c >= 1, other=-1
    )

    if HAS_INITSTATES and (seq_idx != seq_idx_prev):
        prev_states_ptr = (
            initstates_ptr
            + seq_idx * stride_init_states_batch
            + pid_h * stride_init_states_head
        )
        prev_states_hdim = stride_init_states_hdim
        prev_states_dstate = stride_init_states_dstate
    else:
        prev_states_ptr = (
            states_ptr + (pid_c - 1) * stride_states_chunk + pid_h * stride_states_head
        )
        prev_states_hdim = stride_states_hdim
        prev_states_dstate = stride_states_dstate

    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dA_cs_m = tl.load(
        dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0
    ).to(tl.float32)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
    offs_k_dstate = tl.arange(
        0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K
    )
    C_ptrs = C_ptr + (
        offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate
    )

    scale_m = fast_exp(dA_cs_m)
    if BLOCK_SIZE_DSTATE <= 128:
        C = tl.load(
            C_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit)
            & (offs_k_dstate[None, :] < dstate),
            other=0.0,
        )

        if not HAS_INITSTATES and (seq_idx != seq_idx_prev):
            # if no init states AND starting a new sequence, we need zeros
            prev_states = tl.zeros(
                (BLOCK_SIZE_DSTATE, BLOCK_SIZE_N), dtype=C_ptr.dtype.element_ty
            )
        else:
            # otherwise read the previous state
            prev_states_ptrs = (
                prev_states_ptr
                + offs_n[None, :] * prev_states_hdim
                + offs_k_dstate[:, None] * prev_states_dstate
            )
            prev_states = tl.load(
                prev_states_ptrs,
                mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim),
                other=0.0,
            )
            prev_states = prev_states.to(C_ptr.dtype.element_ty)

        acc = tl.dot(C, prev_states) * scale_m[:, None]

    else:
        prev_states_ptrs = (
            prev_states_ptr
            + offs_n[None, :] * prev_states_hdim
            + offs_k_dstate[:, None] * prev_states_dstate
        )
        for k in range(0, dstate, BLOCK_SIZE_K):
            C = tl.load(
                C_ptrs,
                mask=(offs_m[:, None] < chunk_size_limit)
                & (offs_k_dstate[None, :] < dstate - k),
                other=0.0,
            )
            if not HAS_INITSTATES and (seq_idx != seq_idx_prev):
                prev_states = tl.zeros(
                    (BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=C_ptr.dtype.element_ty
                )
            else:
                prev_states = tl.load(
                    prev_states_ptrs,
                    mask=(offs_k_dstate[:, None] < dstate - k)
                    & (offs_n[None, :] < hdim),
                    other=0.0,
                )
                prev_states = prev_states.to(C_ptr.dtype.element_ty)
            acc += tl.dot(C, prev_states)
            C_ptrs += BLOCK_SIZE_K
            prev_states_ptrs += BLOCK_SIZE_K
        acc *= scale_m[:, None]

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (
        offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k
    )
    x_ptrs = x_ptr + (
        offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = (
        chunk_size_limit
        if not IS_CAUSAL
        else min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)
    )
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        cb = tl.load(
            cb_ptrs,
            mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(
            tl.float32
        )
        # If there's seq_idx, we already set cb[i, j] = 0 for seq_idx[i] != seq_idx[j].
        # So we don't need masking wrt seq_idx here.
        cb *= fast_exp(tl.minimum(dA_cs_m[:, None] - dA_cs_k[None, :], 0.0))
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        cb *= dt_k
        if IS_CAUSAL:
            mask = offs_m[:, None] >= k + offs_k[None, :]
            cb = tl.where(mask, cb, 0.0)
        cb = cb.to(x_ptr.dtype.element_ty)
        x = tl.load(
            x_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < hdim),
            other=0.0,
        )
        acc += tl.dot(cb, x)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(
                D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0
            ).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        x_residual = tl.load(
            x_ptr
            + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim),
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        acc += x_residual * D

    if HAS_Z:
        z_ptr += chunk_seqlen_start * stride_z_seqlen + pid_h * stride_z_head
        z_ptrs = z_ptr + (
            stride_z_seqlen * offs_out_m[:, None] + stride_z_hdim * offs_out_n[None, :]
        )
        z = tl.load(
            z_ptrs,
            mask=(offs_out_m[:, None] < chunk_size_limit)
            & (offs_out_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        acc *= z * tl.sigmoid(z)

    out_ptr += chunk_seqlen_start * stride_out_seqlen + pid_h * stride_out_head
    out_ptrs = out_ptr + (
        stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim
    )
    tl.store(
        out_ptrs,
        acc,
        mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim),
    )


def _chunk_scan_fwd(
    cb,
    x,
    dt,
    dA_cumsum,
    C,
    states,
    cu_chunk_seqlens,
    out,
    seq_idx,
    D=None,
    z=None,
    initial_states=None,
):
    assert seq_idx is not None, "this implementation requires seq_idx"

    seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (seqlen, ngroups, dstate)
    assert cb.shape == (nchunks, ngroups, chunk_size, chunk_size)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if z is not None:
        assert z.shape == x.shape
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (nheads, nchunks, chunk_size)
    assert states.shape == (nchunks, nheads, headdim, dstate)
    assert seq_idx.shape == (nchunks,)

    grid = lambda META: (
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"])
        * triton.cdiv(headdim, META["BLOCK_SIZE_N"]),
        nchunks,
        nheads,
    )

    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    initial_states_strides = (
        (
            initial_states.stride(0),
            initial_states.stride(1),
            initial_states.stride(2),
            initial_states.stride(3),
        )
        if initial_states is not None
        else (0, 0, 0, 0)
    )

    _chunk_scan_fwd_kernel[grid](
        cb_ptr=cb,
        x_ptr=x,
        z_ptr=z,
        out_ptr=out,
        dt_ptr=dt,
        dA_cumsum_ptr=dA_cumsum,
        seq_idx_ptr=seq_idx,
        C_ptr=C,
        states_ptr=states,
        D_ptr=D,
        initstates_ptr=initial_states,
        cu_chunk_seqlens_ptr=cu_chunk_seqlens,
        chunk_size=chunk_size,
        hdim=headdim,
        dstate=dstate,
        seqlen=seqlen,
        nheads_ngroups_ratio=nheads // ngroups,
        stride_cb_chunk=cb.stride(0),
        stride_cb_head=cb.stride(1),
        stride_cb_csize_m=cb.stride(2),
        stride_cb_csize_k=cb.stride(3),
        stride_x_seqlen=x.stride(0),
        stride_x_head=x.stride(1),
        stride_x_hdim=x.stride(2),
        stride_z_seqlen=z_strides[0],
        stride_z_head=z_strides[1],
        stride_z_hdim=z_strides[2],
        stride_out_seqlen=out.stride(0),
        stride_out_head=out.stride(1),
        stride_out_hdim=out.stride(2),
        stride_dt_chunk=dt.stride(1),
        stride_dt_head=dt.stride(0),
        stride_dt_csize=dt.stride(2),
        stride_dA_cs_chunk=dA_cumsum.stride(1),
        stride_dA_cs_head=dA_cumsum.stride(0),
        stride_dA_cs_csize=dA_cumsum.stride(2),
        stride_seq_idx_chunk=seq_idx.stride(0),
        stride_C_seqlen=C.stride(0),
        stride_C_head=C.stride(1),
        stride_C_dstate=C.stride(2),
        stride_states_chunk=states.stride(0),
        stride_states_head=states.stride(1),
        stride_states_hdim=states.stride(2),
        stride_states_dstate=states.stride(3),
        stride_init_states_batch=initial_states_strides[0],
        stride_init_states_head=initial_states_strides[1],
        stride_init_states_hdim=initial_states_strides[2],
        stride_init_states_dstate=initial_states_strides[3],
        stride_D_head=D.stride(0) if D is not None else 0,
        IS_CAUSAL=True,
        HAS_D=D is not None,
        D_HAS_HDIM=D.dim() == 2 if D is not None else True,
        HAS_Z=z is not None,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        IS_TRITON_22=TRITON_22,
        HAS_INITSTATES=initial_states is not None,
    )
    return

"""Authored launcher for L3_01_chunk_scan_fwd.

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

    # CPU generator so the CPU oracle (the pyref-capture stage) and GPU launcher (the GPU-compare stage) draw the
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
