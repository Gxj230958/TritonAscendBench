"""Task L2_01_bmm_chunk_fwd_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_bmm_chunk_fwd_kernel`
Source path: `vllm/model_executor/layers/mamba/ops/ssd_bmm.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): 6dd733f6c6a70a8c4a75c756f404000ab955ca743d6a33ddf2e464e741440607
Frozen by: scripts/phase_a_freeze.py
"""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_bmm.py

# ruff: noqa: E501,SIM102

import torch

import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
from triton_ascend_bench.upstream_shims.triton_autotune_interp import make_triton_clone as _pa_make_triton_clone  # Phase A': interpreter-aware autotune fallback
triton = _pa_make_triton_clone()


@triton.autotune(
    configs=[
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
    key=["chunk_size", "K", "IS_CAUSAL"],
)
@triton.jit
def _bmm_chunk_fwd_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    out_ptr,
    cu_chunk_seqlens_ptr,
    # Matrix dimensions
    seqlen,
    chunk_size: tl.constexpr,
    K: tl.constexpr,
    ngroups: tl.constexpr,
    stride_a_seqlen: tl.int64,
    stride_a_head: tl.int64,
    stride_ak: tl.constexpr,
    stride_b_seqlen: tl.int64,
    stride_b_head: tl.int64,
    stride_bk: tl.constexpr,
    stride_out_chunk: tl.int64,
    stride_out_head: tl.int64,
    stride_outm: tl.int64,
    stride_outn: tl.constexpr,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    dot_dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_ch = tl.program_id(axis=1).to(tl.int64)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - pid_c * ngroups
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    if IS_CAUSAL:
        if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
            return

    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)

    a_ptr += chunk_seqlen_start * stride_a_seqlen + pid_h * stride_a_head
    b_ptr += chunk_seqlen_start * stride_b_seqlen + pid_h * stride_b_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_a_seqlen + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_b_seqlen)
    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # compute a * b.T
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit)
            & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        ).to(dot_dtype)
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K)
            & (offs_n[None, :] < chunk_size_limit),
            other=0.0,
        ).to(dot_dtype)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    out = acc.to(out_ptr.dtype.element_ty)
    out_ptr += pid_c * stride_out_chunk + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_outm * offs_m[:, None] + offs_n[None, :] * stride_outn)
    tl.store(
        out_ptrs,
        out,
        mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size),
    )


def _bmm_chunk_fwd(a, b, chunk_size, cu_chunk_seqlens, causal=False, output_dtype=None):
    """
    Argument:
        a: (seqlen, ngroups, k)
        b: (seqlen, ngroups, k)
        chunk_size: int
        cu_chunk_seq_lens: (nchunks+1,)
        causal: if True, then out[i, j] for i > j will be arbitrary, only out[i, j] for i <= j are
            guaranteed to be correct.
    Return:
        out: (nchunks, ngroups, chunk_size, chunk_size)
    """
    seqlen, ngroups, k = a.shape
    assert b.shape == a.shape
    if a.stride(-1) != 1 and a.stride(0) != 1:
        a = a.contiguous()
    if b.stride(-1) != 1 and b.stride(0) != 1:
        b = b.contiguous()

    nchunks = len(cu_chunk_seqlens) - 1
    # Allocates output.
    out_dtype = a.dtype if output_dtype is None else output_dtype
    out = torch.empty(
        (nchunks, ngroups, chunk_size, chunk_size), device=a.device, dtype=out_dtype
    )
    dot_dtype = (
        tl.bfloat16
        if a.dtype == torch.bfloat16 or b.dtype == torch.bfloat16
        else (
            tl.float16
            if a.dtype == torch.float16 or b.dtype == torch.float16
            else tl.float32
        )
    )
    grid = lambda META: (
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"])
        * triton.cdiv(chunk_size, META["BLOCK_SIZE_N"]),
        nchunks * ngroups,
    )
    with torch.accelerator.device_index(a.device.index):
        _bmm_chunk_fwd_kernel[grid](
            a_ptr=a,
            b_ptr=b,
            out_ptr=out,
            cu_chunk_seqlens_ptr=cu_chunk_seqlens,
            seqlen=seqlen,
            chunk_size=chunk_size,
            K=k,
            ngroups=ngroups,
            stride_a_seqlen=a.stride(0),
            stride_a_head=a.stride(1),
            stride_ak=a.stride(2),
            stride_b_seqlen=b.stride(0),
            stride_b_head=b.stride(1),
            stride_bk=b.stride(2),
            stride_out_chunk=out.stride(0),
            stride_out_head=out.stride(1),
            stride_outm=out.stride(-2),
            stride_outn=out.stride(-1),
            IS_CAUSAL=causal,
            dot_dtype=dot_dtype,
        )
    return out


# ── Phase A' synthesised standalone entrypoint (HOLE B) ─────────────
# The staged candidate MUST expose run(inputs)->dict so S2 can compare
# it against the UNMODIFIED upstream (which exposes the positional
# entrypoint '_bmm_chunk_fwd', NOT run). This wrapper is a PURE dict->positional
# argument shuffle of the EXISTING upstream entrypoint -- it never
# reimplements numeric logic (changing values = an A' failure). The
# original side runs the same upstream callable via the identical
# signature adapter in _pa_subproc_runner._signature_adapter, so the
# two manifests are structurally comparable. Promotion of this file is
# gated on a genuine S2-gpu-authoritative PASS (never fabricated).
def run(inputs: dict) -> dict:  # noqa: D401
    """Standalone S2 entrypoint -- maps `inputs` onto the upstream
    `_bmm_chunk_fwd` signature STRICTLY BY NAME (codex r5 ISSUE 1: never a silent
    skip+positional-shift; a required no-default param with no value is
    a hard BLOCKED(shapes-incomplete), NOT a mis-call), returns a dict
    (mutated args surfaced when the upstream entrypoint mutates in
    place and returns None). This mirrors
    _pa_subproc_runner._signature_adapter EXACTLY so the original and
    severed sides bind identically."""
    import inspect

    import torch as _torch

    _MARK = "BLOCKED(shapes-incomplete"
    _fn = _bmm_chunk_fwd
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
                    + " for entrypoint " + repr('_bmm_chunk_fwd')
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
