"""Task L3_17_chunk_gla_fwd_kernel_o — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `chunk_gla_fwd_kernel_o`
Source path: `vllm/model_executor/layers/fla/ops/kda.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): 16ccf19821e32783c049d89faa3da6e6dc0ff4bda991e9b5d04b11cfb144f837
Frozen by: scripts/phase_a_freeze.py
"""
import torch
import torch.nn as nn
from tasks._frozen_shim import CustomOp  # Phase A': inert CustomOp base + no-op register (B-inert-shim)
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
from triton_ascend_bench.upstream_shims.triton_autotune_interp import make_triton_clone as _pa_make_triton_clone  # Phase A': interpreter-aware autotune fallback
triton = _pa_make_triton_clone()
from tasks._frozen_shim import cdiv, next_power_of_2  # Phase A': pure-math shim (B-inert-shim)
from tasks._frozen_deps_pool.h_af43637ae818.fla_ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h  # Phase A': U2 FLA-ops package vendoring
from tasks._frozen_deps_pool.h_af43637ae818.fla_ops.cumsum import chunk_local_cumsum  # Phase A': U2 FLA-ops package vendoring
from tasks._frozen_deps_pool.h_af43637ae818.fla_ops.fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel  # Phase A': U2 FLA-ops package vendoring
from tasks._frozen_deps_pool.h_af43637ae818.fla_ops.index import prepare_chunk_indices  # Phase A': U2 FLA-ops package vendoring
from tasks._frozen_deps_pool.h_af43637ae818.fla_ops.l2norm import l2norm_fwd  # Phase A': U2 FLA-ops package vendoring
from tasks._frozen_deps_pool.h_af43637ae818.fla_ops.op import exp, log  # Phase A': U2 FLA-ops package vendoring
from tasks._frozen_deps_pool.h_af43637ae818.fla_ops.solve_tril import solve_tril  # Phase A': U2 FLA-ops package vendoring
from tasks._frozen_deps_pool.h_af43637ae818.fla_ops.utils import FLA_CHUNK_SIZE, is_amd  # Phase A': U2 FLA-ops package vendoring

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages) for BK in [32, 64] for BV in [64, 128] for num_warps in [2, 4, 8] for num_stages in [2, 3, 4]], key=['BT'])
@triton.jit(do_not_specialize=['T'])
def chunk_gla_fwd_kernel_o(q, v, g, h, o, A, cu_seqlens, chunk_indices, scale, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_v, i_t, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_b, i_h = (i_bh // H, i_bh % H)
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = (i_b * T, i_b * T + T)
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + (i_tg * H + i_h) * K * V, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = (b_q * exp(b_g)).to(b_q.dtype)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        if i_k >= 0:
            b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

def chunk_gla_fwd_o_gk(q: torch.Tensor, v: torch.Tensor, g: torch.Tensor, A: torch.Tensor, h: torch.Tensor, o: torch.Tensor, scale: float, cu_seqlens: torch.Tensor | None=None, chunk_size: int=64):
    B, T, H, K, V = (*q.shape, v.shape[-1])
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    def grid(meta):
        return (cdiv(V, meta['BV']), NT, B * H)
    chunk_gla_fwd_kernel_o[grid](q=q, v=v, g=g, h=h, o=o, A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, scale=scale, T=T, H=H, K=K, V=V, BT=BT)
    return o

import torch


# Authored launcher: drives the in-module @triton.jit
# `chunk_gla_fwd_kernel_o` exactly as the upstream host wrapper
# `chunk_gla_fwd_o_gk` does on its NON-VARLEN path (vLLM
# vllm/model_executor/layers/fla/ops, commit
# 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19).
#
# Why a launcher rather than binding the host wrapper directly: the strict
# by-name signature adapter unpacks `B, T, H, K, V = *q.shape, v.shape[-1]`
# in the wrapper, which requires a rank-4 q [B, T, H, K]; it also takes a
# pre-allocated `o` output and a `cu_seqlens` varlen control with no usable
# default. The oracle (oracle_pytorch.py) models ONLY the non-varlen path
# (cu_seqlens is None), where the kernel processes one BT-sized chunk per
# (batch, head) on a dense head-structured layout. We allocate `o`, pass
# cu_seqlens=None, and set chunk_size = A.shape[-1] so the kernel's BT
# matches A's local-column extent exactly — reproducing the wrapper's
# non-varlen branch without touching numeric logic.
#
# Upstream invocation reproduced here (chunk_gla_fwd_o_gk, non-varlen):
#   q: [B, T, H, K]      gated query
#   g: [B, T, H, K]      log-gate (kernel applies exp(g))
#   v: [B, T, H, V]      values
#   A: [B, T, H, BT]     per-chunk intra score; last axis is the LOCAL
#                        within-chunk column (block-diagonal in time)
#   h: [B, NT, H, V, K]  per-chunk state, read as a [V, K] tile
#   o: [B, T, H, V]      output (written in place)
#   B, T, H, K = q.shape; V = v.shape[-1]; BT = A.shape[-1];
#   NT = cdiv(T, BT); grid = (cdiv(V, BV), NT, B * H). The wrapper passes
#   T, H, K, V and BT=chunk_size; BK/BV come from the autotuner.
#
# Output key (output) matches the oracle contract exactly.
def run(inputs: dict) -> dict:
    q = inputs["q"]
    v = inputs["v"]
    g = inputs["g"]
    A = inputs["A"]
    h = inputs["h"]
    scale = float(inputs["scale"])

    BT = A.shape[-1]
    o = torch.empty_like(v)

    # chunk_gla_fwd_o_gk is resolved from the kernel.py module namespace
    # the launcher is exec'd into (see _pa_oracle_check_runner._load).
    chunk_gla_fwd_o_gk(  # noqa: F821
        q=q,
        v=v,
        g=g,
        A=A,
        h=h,
        o=o,
        scale=scale,
        cu_seqlens=None,
        chunk_size=BT,
    )
    return {"output": o}
