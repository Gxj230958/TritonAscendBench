"""Task L1_40_chunk_cumsum_fwd_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_chunk_cumsum_fwd_kernel`
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
from tasks._frozen_deps_pool.h_ccee54797b1f.vllm_model_executor_layers_mamba_ops_mamba_ssm_softplus import softplus  # Phase A': U2 pool-vendored on-path mamba softplus

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_H': 2}), triton.Config({'BLOCK_SIZE_H': 4}), triton.Config({'BLOCK_SIZE_H': 8}), triton.Config({'BLOCK_SIZE_H': 16}), triton.Config({'BLOCK_SIZE_H': 32}), triton.Config({'BLOCK_SIZE_H': 64})], key=['chunk_size', 'nheads'])
@triton.jit
def _chunk_cumsum_fwd_kernel(dt_ptr, A_ptr, dt_bias_ptr, dt_out_ptr, dA_cumsum_ptr, cu_chunk_seqlens_ptr, seqlen, nheads: tl.constexpr, chunk_size: tl.constexpr, dt_min: tl.constexpr, dt_max: tl.constexpr, stride_dt_seqlen: tl.int64, stride_dt_head: tl.constexpr, stride_A_head: tl.constexpr, stride_dt_bias_head: tl.constexpr, stride_dt_out_head: tl.int64, stride_dt_out_chunk: tl.int64, stride_dt_out_csize: tl.constexpr, stride_dA_cs_head: tl.int64, stride_dA_cs_chunk: tl.int64, stride_dA_cs_csize: tl.constexpr, DT_SOFTPLUS: tl.constexpr, HAS_DT_BIAS: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_CHUNK: tl.constexpr):
    pid_c = tl.program_id(axis=0).to(tl.int64)
    pid_h = tl.program_id(axis=1)
    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)
    dt_ptr += chunk_seqlen_start * stride_dt_seqlen
    dt_out_ptr += pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt_ptr + (offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen)
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize)
    dA_cs_ptrs = dA_cumsum_ptr + (offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize)
    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start
    dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, softplus(dt), dt)
    dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0)
    tl.store(dt_out_ptrs, dt, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(dA_cs_ptrs, dA_cs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))

def _chunk_cumsum_fwd(dt, A, chunk_size, cu_chunk_seqlens, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float('inf'))):
    seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    nchunks = cu_chunk_seqlens.shape[0] - 1
    dt_out = torch.empty(nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    dA_cumsum = torch.empty(nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    grid_chunk_cs = lambda META: (nchunks, triton.cdiv(nheads, META['BLOCK_SIZE_H']))
    with torch.accelerator.device_index(dt.device.index):
        _chunk_cumsum_fwd_kernel[grid_chunk_cs](dt_ptr=dt, A_ptr=A, dt_bias_ptr=dt_bias, dt_out_ptr=dt_out, dA_cumsum_ptr=dA_cumsum, cu_chunk_seqlens_ptr=cu_chunk_seqlens, seqlen=seqlen, nheads=nheads, chunk_size=chunk_size, dt_min=dt_limit[0], dt_max=dt_limit[1], stride_dt_seqlen=dt.stride(0), stride_dt_head=dt.stride(1), stride_A_head=A.stride(0), stride_dt_bias_head=dt_bias.stride(0) if dt_bias is not None else 0, stride_dt_out_head=dt_out.stride(0), stride_dt_out_chunk=dt_out.stride(1), stride_dt_out_csize=dt_out.stride(2), stride_dA_cs_head=dA_cumsum.stride(0), stride_dA_cs_chunk=dA_cumsum.stride(1), stride_dA_cs_csize=dA_cumsum.stride(2), DT_SOFTPLUS=dt_softplus, HAS_DT_BIAS=dt_bias is not None, BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size))
    return (dA_cumsum, dt_out)

import torch


# Authored launcher: drives the in-module host wrapper `_chunk_cumsum_fwd`
# (which launches the graded @triton.jit `_chunk_cumsum_fwd_kernel`) exactly
# as the upstream Mamba2 SSD call site does
# (vllm/model_executor/layers/mamba/ops/ssd_combined.py::_chunk_scan_combined_fwd,
# vLLM commit 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19).
#
# Why a launcher rather than the host wrapper alone: the graded entrypoint
# `_chunk_cumsum_fwd` needs a `cu_chunk_seqlens` metadata tensor that the
# upstream pipeline derives from the request layout, not a primitive the
# benchmark fixture materialises directly. The launcher reproduces upstream's
# `compute_varlen_chunk_metadata` (mamba2_attn.py) for the fixture's
# contiguous token stream, then calls the wrapper verbatim. The sibling
# `_chunk_state_fwd` in this module is NOT graded and is never called here.
#
# `build_cu_chunk_seqlens` is byte-identical to the copy in oracle_pytorch.py;
# both derive the partition from the SAME (seqlen, chunk_size), so they cannot
# diverge -- any drift would make the correct kernel mismatch the oracle and
# BLOCK the task (fail-closed), never a silent pass.
def build_cu_chunk_seqlens(seqlen: int, chunk_size: int) -> list[int]:
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
    pos = 0
    for length in seq_lens:
        end = pos + length
        p = pos
        while p < end:
            room = cs - (p % cs)
            take = min(room, end - p)
            chunk_lens.append(take)
            p += take
        pos = end
    cu = [0]
    for c in chunk_lens:
        cu.append(cu[-1] + c)
    assert cu[-1] == int(seqlen), (cu[-1], seqlen)
    assert all(0 < c <= cs for c in chunk_lens), chunk_lens
    return cu


def _pin_cumsum_autotune_config() -> None:
    """Make the @triton.autotune launch config DETERMINISTIC for the strict
    severed-vs-original bit-exact gate.

    ``_chunk_cumsum_fwd_kernel`` (frozen kernel.py) is ``@triton.autotune``'d
    over BLOCK_SIZE_H. Triton's Autotuner selects the config by TIMING
    (``cache[key] = min(timings, ...)``), which is non-deterministic across the
    two independent S2 subprocesses (severed vs original). A different
    BLOCK_SIZE_H gives a different ``[BLOCK_SIZE_H, chunk_size]`` tile layout,
    and that changes how ``tl.cumsum(..., axis=1)`` lowers its scan reduction
    -- float add is non-associative, so a reassociated scan flips a ULP in
    ``dA_cumsum`` (observed at the large shape only; the elementwise ``dt_out``
    stays bit-stable, which pinpoints the reduction as the sole diverging op).

    Collapsing the autotuner to a SINGLE config (one of the kernel's own
    candidates) makes ``len(configs) <= 1``, so the tuner skips benchmarking
    and runs ``configs[0]`` directly -- BOTH sides reduce identically. This
    runs in the SAME launcher on both sides (original exec's it into kernel.py;
    the severed staged candidate appends it), so the pin is symmetric. It
    changes neither the math nor the candidate set.
    """
    kfn = globals().get("_chunk_cumsum_fwd_kernel")
    if kfn is None or not hasattr(kfn, "configs"):
        return  # not an autotuned kernel -> nothing to pin
    try:
        import triton as _triton
    except ImportError:
        return
    # BLOCK_SIZE_H=64 is verbatim one of kernel.py's autotune configs; the
    # head axis is masked (offs_h < nheads) so it is valid for every shape.
    cfg = _triton.Config({"BLOCK_SIZE_H": 64})
    # Two wrapper types reach here:
    #   * real triton Autotuner (GPU path, TRITON_INTERPRET unset + active
    #     driver): plain settable ``.configs`` + a ``.cache`` dict. This is
    #     where the timing-based race lives and where the pin matters.
    #   * the interpreter-fallback _AutotunedKernel (no-driver / interpret
    #     smoke): ``.configs`` is a read-only property over ``._configs`` and
    #     the launch reads ``._first_kwargs`` (built from ``_configs[0]``). It
    #     is ALREADY deterministic, but pin it to the SAME config so both
    #     paths select identically and the assignment never raises.
    try:
        kfn.configs = [cfg]
        if hasattr(kfn, "cache"):
            kfn.cache = {}
        # Drop any stale best_config so the single-config path is taken.
        if hasattr(kfn, "best_config"):
            kfn.best_config = cfg
    except AttributeError:
        if hasattr(kfn, "_configs"):
            kfn._configs = [cfg]
            akw = getattr(cfg, "all_kwargs", None)
            if callable(akw):
                kfn._first_kwargs = dict(akw())


def run(inputs: dict) -> dict:
    _pin_cumsum_autotune_config()
    dt = inputs["dt"]
    A = inputs["A"]
    chunk_size = int(inputs["chunk_size"])
    dt_softplus = bool(inputs.get("dt_softplus", False))

    seqlen = int(dt.shape[0])
    cu_list = build_cu_chunk_seqlens(seqlen, chunk_size)
    cu_chunk_seqlens = torch.tensor(cu_list, dtype=torch.int32, device=dt.device)

    dl = inputs.get("dt_limit")
    dt_limit = tuple(dl) if dl is not None else (0.0, float("inf"))

    dA_cumsum, dt_out = _chunk_cumsum_fwd(  # noqa: F821 (exec'd into kernel module)
        dt,
        A,
        chunk_size,
        cu_chunk_seqlens,
        dt_bias=inputs.get("dt_bias"),
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )
    return {"dt_out": dt_out, "dA_cumsum": dA_cumsum}
