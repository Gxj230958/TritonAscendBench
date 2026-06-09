"""Task L1_26_state_passing_fwd_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_state_passing_fwd_kernel`
Source path: `vllm/model_executor/layers/mamba/ops/ssd_state_passing.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): 36b8c8edba437916a5b789717c61fc22077de01e77892cb96dbff3b00a57410e
"""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_state_passing.py

# ruff: noqa: E501

import torch

from ascendbench.frozen.deps.h_119bf34e8937.vllm_model_executor_layers_mamba_ops_triton_helpers import fast_exp  # Phase A': U2 pool-vendored triton helper
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
from ascendbench.upstream_shims.triton_autotune_interp import make_triton_clone as _pa_make_triton_clone  # Phase A': interpreter-aware autotune fallback
triton = _pa_make_triton_clone()


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
    ],
    key=["dim"],
)
@triton.jit
def _state_passing_fwd_kernel(
    # Pointers to matrices
    states_ptr,
    out_ptr,
    dA_cs_ptr,
    initstates_ptr,
    last_chunk_indices_ptr,
    # Matrix dimensions
    dim: tl.constexpr,
    chunk_size: tl.constexpr,
    # Strides
    stride_states_chunk: tl.int64,
    stride_states_head: tl.int64,
    stride_states_dim: tl.constexpr,
    stride_out_chunk: tl.int64,
    stride_out_head: tl.int64,
    stride_out_dim: tl.constexpr,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    stride_initstates_batch: tl.int64,
    stride_initstates_head: tl.int64,
    stride_initstates_dim: tl.constexpr,
    # Meta-parameters
    HAS_INITSTATES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    # Derive this sequence's chunk range from last_chunk_indices
    chunk_end = tl.load(last_chunk_indices_ptr + pid_b) + 1
    chunk_start = (
        tl.load(last_chunk_indices_ptr + pid_b - 1, mask=pid_b > 0, other=-1) + 1
    )

    # Offset pointers to this sequence's first chunk
    states_ptr += chunk_start * stride_states_chunk + pid_h * stride_states_head
    dA_cs_ptr += (
        pid_h * stride_dA_cs_head
        + chunk_start * stride_dA_cs_chunk
        + (chunk_size - 1) * stride_dA_cs_csize
    )
    out_ptr += chunk_start * stride_out_chunk + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim

    # Load initial state once — no per-chunk branching needed
    if HAS_INITSTATES:
        initstates_ptrs = (
            initstates_ptr
            + pid_b * stride_initstates_batch
            + pid_h * stride_initstates_head
            + offs_m * stride_initstates_dim
        )
        states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    else:
        states = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop over only this sequence's chunks — branchless
    nchunks_this_seq = chunk_end - chunk_start
    for _ in range(nchunks_this_seq):
        new_states = tl.load(states_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        states = fast_exp(dA_cs) * states + new_states
        tl.store(out_ptrs, states, mask=offs_m < dim)

        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk


def _state_passing_fwd(
    states,
    dA_cumsum,
    last_chunk_indices,
    initial_states=None,
    out_dtype=None,
):
    nchunks, nheads, dim = states.shape
    chunk_size = dA_cumsum.shape[-1]
    batch = last_chunk_indices.shape[0]
    assert dA_cumsum.shape == (nheads, nchunks, chunk_size)
    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((nchunks, nheads, dim), device=states.device, dtype=out_dtype)

    initial_states_strides = (
        (initial_states.stride(0), initial_states.stride(1), initial_states.stride(2))
        if initial_states is not None
        else (0, 0, 0)
    )

    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE"]), batch, nheads)
    with torch.accelerator.device_index(states.device.index):
        _state_passing_fwd_kernel[grid](
            states_ptr=states,
            out_ptr=out,
            dA_cs_ptr=dA_cumsum,
            initstates_ptr=initial_states,
            last_chunk_indices_ptr=last_chunk_indices,
            dim=dim,
            chunk_size=chunk_size,
            stride_states_chunk=states.stride(0),
            stride_states_head=states.stride(1),
            stride_states_dim=states.stride(2),
            stride_out_chunk=out.stride(0),
            stride_out_head=out.stride(1),
            stride_out_dim=out.stride(2),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            stride_initstates_batch=initial_states_strides[0],
            stride_initstates_head=initial_states_strides[1],
            stride_initstates_dim=initial_states_strides[2],
            HAS_INITSTATES=initial_states is not None,
        )
    return out


# ── Phase A' synthesised standalone entrypoint (HOLE B) ─────────────
# The staged candidate MUST expose run(inputs)->dict so S2 can compare
# it against the UNMODIFIED upstream (which exposes the positional
# entrypoint '_state_passing_fwd', NOT run). This wrapper is a PURE dict->positional
# argument shuffle of the EXISTING upstream entrypoint -- it never
# reimplements numeric logic (changing values = an A' failure). The
# original side runs the same upstream callable via the identical
# signature adapter in _pa_subproc_runner._signature_adapter, so the
# two manifests are structurally comparable. Promotion of this file is
# gated on a genuine S2-gpu-authoritative PASS (never fabricated).
def run(inputs: dict) -> dict:  # noqa: D401
    """Standalone S2 entrypoint -- maps `inputs` onto the upstream
    `_state_passing_fwd` signature STRICTLY BY NAME (never a silent
    skip+positional-shift; a required no-default param with no value is
    a hard BLOCKED(shapes-incomplete), NOT a mis-call), returns a dict
    (mutated args surfaced when the upstream entrypoint mutates in
    place and returns None). This mirrors
    _pa_subproc_runner._signature_adapter EXACTLY so the original and
    severed sides bind identically."""
    import inspect

    import torch as _torch

    _MARK = "BLOCKED(shapes-incomplete"
    _fn = _state_passing_fwd
    _PNM = {}  # meta.interface.param_name_map (wrapper param -> task key)
    try:
        _sig = inspect.signature(_fn)
    except (TypeError, ValueError):
        _sig = None
    if _sig is None:
        _ret = _fn(inputs)
        _bound = []
    else:
        _kw = {}
        _bound = []
        for _n, _p in _sig.parameters.items():
            if _p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if _n in inputs:
                _kw[_n] = inputs[_n]
                _bound.append(_n)
            elif _n in _PNM and _PNM[_n] in inputs:
                # meta.interface.param_name_map (baked): same task tensor
                # under a different key. _bound records the TASK KEY so the
                # in-place output reconstruction indexes inputs correctly.
                _src = _PNM[_n]
                _kw[_n] = inputs[_src]
                _bound.append(_src)
            elif _p.default is not inspect.Parameter.empty:
                continue
            else:
                raise RuntimeError(
                    _MARK + ": missing param " + repr(_n)
                    + " for entrypoint " + repr('_state_passing_fwd')
                    + " -- shapes.yaml / build_inputs provides no value "
                    "and the parameter has no default. Refusing to "
                    "silently skip+shift positional args. Fix: add the "
                    "param to the task's shapes.yaml inputs (or "
                    "meta.interface.param_name_map)."
                )
        _ret = _fn(**_kw)
    if _ret is None:
        _out = {
            p: inputs[p]
            for p in _bound
            if _torch.is_tensor(inputs.get(p))
        }
        return _out or {"_inplace": None}
    if isinstance(_ret, dict):
        return _ret
    if isinstance(_ret, (list, tuple)):
        return {f"_out{i}": v for i, v in enumerate(_ret)}
    return {"_out0": _ret}
