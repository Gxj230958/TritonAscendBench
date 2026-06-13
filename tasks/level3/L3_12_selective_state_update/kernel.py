"""Task L3_12_selective_state_update — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_selective_scan_update_kernel`
Source path: `vllm/model_executor/layers/mamba/ops/mamba_ssm.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): e793eb34f3ef12f3e2ba8185d9fba389aee11f543bf36376297c20d4460c9373
"""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/selective_state_update.py

import torch
from packaging import version

from ascendbench.frozen.shim import ops as ops  # Phase A': off-path custom-ops shim (B-inert-shim)
from ascendbench.frozen.deps.h_119bf34e8937.vllm_model_executor_layers_mamba_ops_triton_helpers import fast_exp  # Phase A': U2 pool-vendored triton helper
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
HAS_TRITON = True  # Phase A': pinned severed-runtime capability flag
from ascendbench.frozen.shim import NULL_BLOCK_ID  # Phase A': attention sentinel constants (B-inert-shim)

TRITON3 = HAS_TRITON and (version.parse(triton.__version__) >= version.parse("3.0.0"))

if TRITON3:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
        return dt
else:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
        return dt


@triton.jit
def convert_rs_fp16x2(x: tl.tensor, rand: tl.tensor) -> tl.tensor:
    y = tl.inline_asm_elementwise(
        asm="""{
cvt.rs.f16x2.f32 $0, $2, $1, $3;
}""",
        constraints="=r,r,r,r,r",
        args=(x, rand),
        dtype=tl.float16,
        is_pure=True,
        pack=2,
    )
    return y


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {
        "HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"]
        is not None
    }
)
@triton.heuristics(
    {"IS_SPEC_DECODING": lambda args: args["num_accepted_tokens_ptr"] is not None}
)
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens_ptr"] is not None})
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.jit(do_not_specialize=["N"])
def _selective_scan_update_kernel(
    # Pointers to matrices
    state_ptr,
    rand_seed_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    state_batch_indices_ptr,
    dst_state_batch_indices_ptr,
    null_block_id,
    num_accepted_tokens_ptr,
    cu_seqlens_ptr,
    # Matrix dimensions
    N,
    nheads,
    dim,
    dstate,
    nheads_ngroups_ratio,
    # Strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_dim,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_A_head,
    stride_A_dim,
    stride_A_dstate,
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_D_dim,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    stride_state_indices_batch,
    stride_state_indices_T,
    stride_dst_state_indices_batch,
    stride_dst_state_indices_T,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    USE_RS_ROUNDING: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + pid_b).to(tl.int64)
        eos = tl.load(cu_seqlens_ptr + pid_b + 1).to(tl.int64)
        seq_len = eos - bos

        if seq_len == 0:
            return
    else:
        bos = pid_b
        seq_len = 1

    state_ptr_base = state_ptr

    # If HAS_STATE_BATCH_INDICES is true, then the ssm state's batch coordinate
    # is taken from the state_batch_indices_ptr Otherwise, the state coordinate
    # is the same as the batch id.
    if HAS_STATE_BATCH_INDICES:
        if IS_SPEC_DECODING:
            num_accepted = tl.load(num_accepted_tokens_ptr + pid_b).to(tl.int64)
            init_token_idx = tl.maximum(num_accepted - 1, 0)
        else:
            init_token_idx = 0

        dst_state_batch_indices_ptr += pid_b * stride_dst_state_indices_batch
        if not IS_SPEC_DECODING:
            dst_state_batch_idx = tl.load(
                dst_state_batch_indices_ptr
                + init_token_idx * stride_dst_state_indices_T
            ).to(tl.int64)
            dst_state_ptr = state_ptr + (
                dst_state_batch_idx * stride_state_batch + pid_h * stride_state_head
            )

        state_batch_indices_ptr += (
            pid_b * stride_state_indices_batch + init_token_idx * stride_state_indices_T
        )
        state_batch_idx = tl.load(state_batch_indices_ptr).to(tl.int64)
        state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    else:
        dst_state_ptr = (
            state_ptr + pid_b * stride_state_batch + pid_h * stride_state_head
        )
        state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head

    x_ptr += bos * stride_x_batch + pid_h * stride_x_head
    dt_ptr += bos * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += bos * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += bos * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += bos * stride_z_batch + pid_h * stride_z_head
    out_ptr += bos * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (
        offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )
    if not IS_SPEC_DECODING:
        dst_state_ptrs = dst_state_ptr + (
            offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
        )

    mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
    if HAS_STATE_BATCH_INDICES:
        mask &= state_batch_idx != null_block_id
    state = tl.load(state_ptrs, mask=mask, other=0.0).to(tl.float32)

    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
        D_ptrs = D_ptr + offs_m * stride_D_dim
    A_ptrs = A_ptr + offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate

    for i_t in range(seq_len):
        x_ptrs = x_ptr + offs_m * stride_x_dim
        dt_ptrs = dt_ptr + offs_m * stride_dt_dim
        B_ptrs = B_ptr + offs_n * stride_B_dstate
        C_ptrs = C_ptr + offs_n * stride_C_dstate
        if HAS_Z:
            z_ptrs = z_ptr + offs_m * stride_z_dim
        out_ptrs = out_ptr + offs_m * stride_out_dim

        x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if not TIE_HDIM:
            dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A = tl.load(
                A_ptrs,
                mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate),
                other=0.0,
            ).to(tl.float32)
            dA = fast_exp(A * dt[:, None])
        else:
            dt = tl.load(dt_ptr).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptr).to(tl.float32)
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A = tl.load(A_ptr).to(tl.float32)
            dA = fast_exp(A * dt)  # scalar, not a matrix

        B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        if HAS_D:
            D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_Z:
            z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

        dB = B[None, :] * dt[:, None] if not TIE_HDIM else B * dt
        state = state * dA + dB * x[:, None]

        if IS_SPEC_DECODING:
            dst_idx_ptr = dst_state_batch_indices_ptr + i_t * stride_dst_state_indices_T
            token_dst_idx = tl.load(dst_idx_ptr).to(tl.int64)
            if token_dst_idx != null_block_id:
                token_dst_ptrs = (
                    state_ptr_base
                    + token_dst_idx * stride_state_batch
                    + pid_h * stride_state_head
                    + offs_m[:, None] * stride_state_dim
                    + offs_n[None, :] * stride_state_dstate
                )
                tl.store(
                    token_dst_ptrs, state.to(token_dst_ptrs.dtype.element_ty), mask=mask
                )

        out = tl.sum(state * C[None, :], axis=1)
        if HAS_D:
            out += x * D
        if HAS_Z:
            out *= z * tl.sigmoid(z)
        tl.store(out_ptrs, out, mask=offs_m < dim)

        x_ptr += stride_x_batch
        dt_ptr += stride_dt_batch
        B_ptr += stride_B_batch
        C_ptr += stride_C_batch
        out_ptr += stride_out_batch
        if HAS_Z:
            z_ptr += stride_z_batch

    if not IS_SPEC_DECODING:
        if USE_RS_ROUNDING:
            # Load random seed
            rand_seed = tl.load(rand_seed_ptr)
            # Generate random offsets for each element in state
            if HAS_STATE_BATCH_INDICES:
                rand_offsets = (
                    state_batch_idx * stride_state_batch + pid_h * stride_state_head
                )
            else:
                rand_offsets = pid_b * stride_state_batch + pid_h * stride_state_head
            rand_offsets += (
                offs_m[:, None] * stride_state_dim
                + offs_n[None, :] * stride_state_dstate
            )
            # Generate random 32-bits for each element in state
            if PHILOX_ROUNDS > 0:
                rand = tl.randint(rand_seed, rand_offsets, PHILOX_ROUNDS)
            else:
                rand = tl.randint(rand_seed, rand_offsets)
            # Convert state to fp16 with RS rounding
            state = convert_rs_fp16x2(state, rand)
            tl.static_assert(state.dtype == tl.float16, "state must be fp16")
            tl.static_assert(
                dst_state_ptrs.dtype.element_ty == tl.float16,
                "dst_state_ptrs must be fp16",
            )
        else:
            state = state.to(dst_state_ptrs.dtype.element_ty)
        tl.store(dst_state_ptrs, state, mask=mask)





import torch
import triton


# Authored launcher: drives the in-module @triton.jit `_selective_scan_update_kernel`
# (vLLM 18f6bf5, vllm/model_executor/layers/mamba/ops/mamba_ssm.py) on the base
# Mamba2 decode path -- HAS_Z / HAS_D / HAS_DT_BIAS true, DT_SOFTPLUS true, no
# state_batch_indices / spec-decoding / varlen / stochastic rounding. Inputs are
# already in canonical rank (4D state, 3D x/dt/z), so the wrapper's unsqueeze
# bookkeeping is unnecessary; this launcher reproduces the wrapper's kernel
# launch verbatim EXCEPT it drops the `torch.accelerator.device_index` context
# (a device-placement detour the runner already manages, and an API the severed
# host's torch need not ship). The kernel's @triton.heuristics derive HAS_*,
# IS_*, and BLOCK_SIZE_DSTATE from the runtime args.
#
# Math (matches oracle_pytorch): state = state*exp(A*dt) + (B*dt)*x ;
# out = sum(state*C) + x*D ; out *= z*sigmoid(z). `out` is candidate-allocated;
# `state` is mutated in place. Output key: "out".

import torch
import triton


# Authored launcher: drives the in-module @triton.jit `_selective_scan_update_kernel`
# (vLLM 18f6bf5, vllm/model_executor/layers/mamba/ops/mamba_ssm.py) on the base
# Mamba2 decode path -- HAS_Z / HAS_D / HAS_DT_BIAS true, DT_SOFTPLUS true, no
# state_batch_indices / spec-decoding / varlen / stochastic rounding. Inputs are
# already in canonical rank (4D state, 3D x/dt/z), so the wrapper's unsqueeze
# bookkeeping is unnecessary; this launcher reproduces the wrapper's kernel
# launch verbatim EXCEPT it drops the `torch.accelerator.device_index` context
# (a device-placement detour the runner already manages, and an API the severed
# host's torch need not ship). The kernel's @triton.heuristics derive HAS_*,
# IS_*, and BLOCK_SIZE_DSTATE from the runtime args.
#
# Math (matches oracle_pytorch): state = state*exp(A*dt) + (B*dt)*x ;
# out = sum(state*C) + x*D ; out *= z*sigmoid(z). `out` is candidate-allocated;
# `state` is mutated in place. Output key: "out".


def run(inputs: dict) -> dict:
    state = inputs["state"]              # [B, H, dim, dstate]
    x = inputs["x"]                      # [B, H, dim]
    dt = inputs["dt"]                    # [B, H, dim]
    A = inputs["A"]                      # [H, dim, dstate]
    B = inputs["B"]                      # [B, G, dstate]
    C = inputs["C"]                      # [B, G, dstate]
    D = inputs["D"]                      # [H, dim]
    z = inputs["z"]                      # [B, H, dim]
    dt_bias = inputs["dt_bias"]          # [H, dim]

    batch, nheads, dim, dstate = state.shape
    ngroups = B.shape[1]
    # ssm output buffer in fp32 (the kernel stores out.to(out_ptr.element_ty)).
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)

    # Hand-tuned launch config (verbatim from selective_state_update), keyed on
    # dstate; no autotune (it would overwrite the state).
    if dstate <= 16:
        BLOCK_SIZE_M, num_warps = 32, 4
    elif dstate <= 32:
        BLOCK_SIZE_M, num_warps = 16, 4
    elif dstate <= 64:
        BLOCK_SIZE_M, num_warps = 8, 4
    elif dstate <= 128:
        BLOCK_SIZE_M, num_warps = 4, 4
    else:
        BLOCK_SIZE_M, num_warps = 4, 8

    tie_hdim = (
        A.stride(-1) == 0 and A.stride(-2) == 0
        and dt.stride(-1) == 0 and dt_bias.stride(-1) == 0
    )

    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)

    _selective_scan_update_kernel[grid](
        state,
        None,            # rand_seed_ptr (no stochastic rounding)
        x,
        dt,
        dt_bias,
        A,
        B,
        C,
        D,
        z,
        out,
        None,            # state_batch_indices_ptr
        None,            # dst_state_batch_indices_ptr
        0,               # null_block_id (unused without state_batch_indices)
        None,            # num_accepted_tokens_ptr
        None,            # cu_seqlens_ptr
        batch,           # N
        nheads,
        dim,
        dstate,
        nheads // ngroups,
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        x.stride(0), x.stride(1), x.stride(2),
        dt.stride(0), dt.stride(1), dt.stride(2),
        dt_bias.stride(0), dt_bias.stride(1),
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        D.stride(0), D.stride(1),
        z.stride(0), z.stride(1), z.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        0, 0,            # state_batch_indices strides
        0, 0,            # dst_state_batch_indices strides
        True,            # DT_SOFTPLUS
        tie_hdim,        # TIE_HDIM
        BLOCK_SIZE_M,
        num_warps=num_warps,
        USE_RS_ROUNDING=False,
        PHILOX_ROUNDS=0,
    )

    return {"out": out}
