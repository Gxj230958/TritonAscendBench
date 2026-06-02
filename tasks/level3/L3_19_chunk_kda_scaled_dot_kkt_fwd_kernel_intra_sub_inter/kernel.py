"""Task L3_19_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter`
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
@triton.autotune(configs=[triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages) for BK in [32, 64] for num_warps in [1, 2, 4, 8] for num_stages in [2, 3, 4]], key=['BC'])
@triton.jit(do_not_specialize=['T'])
def chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter(q, k, g, beta, A, Aqk, scale, cu_seqlens, chunk_indices, T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_c, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_b, i_h = (i_bh // H, i_bh % H)
    i_i, i_j = (i_c // NC, i_c % NC)
    if IS_VARLEN:
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
    else:
        bos, eos = (i_b * T, i_b * T + T)
    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    A += (bos * H + i_h) * BT
    Aqk += (bos * H + i_h) * BT
    p_b = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT + i_i * BC,), (BC,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        b_kt = tl.make_block_ptr(k, (K, T), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g, (K, T), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_gn = tl.load(g + (i_t * BT + i_i * BC) * H * K + o_k, mask=m_k, other=0)
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1)) * exp(b_g - b_gn[None, :])
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kt = tl.load(b_kt, boundary_check=(0, 1))
        b_ktg = b_kt * exp(b_gn[:, None] - b_gk)
        b_A += tl.dot(b_k, b_ktg)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_qg = b_q * exp(b_g - b_gn[None, :]) * scale
        b_Aqk += tl.dot(b_qg, b_ktg)
    b_A *= b_b[:, None]
    p_A = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))
    p_Aqk = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))

import torch
import triton


# Authored launcher: drives the in-module @triton.jit
# `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter` exactly as the
# upstream host wrapper `chunk_kda_scaled_dot_kkt_fwd` does on its
# NON-VARLEN path (vLLM vllm/model_executor/layers/fla/ops/kda.py, commit
# 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19).
#
# Why a launcher (not a host-wrapper bind): the wrapper
# `chunk_kda_scaled_dot_kkt_fwd` launches TWO different jit kernels into the
# same A / Aqk buffers: the INTER kernel
# (`..._intra_sub_inter`, the one meta.name pins) fills the strictly
# off-diagonal sub-chunk blocks (program returns when `i_i <= i_j`), and a
# SEPARATE named kernel (`..._intra_sub_intra`) fills the diagonal
# sub-blocks. meta.name is the INTER kernel only, so this launcher allocates
# zeroed A / Aqk and launches ONLY that kernel with the wrapper's full grid.
# The two kernels write DISJOINT positions, so isolating the named one is not
# a dropped branch -- the diagonal / upper sub-blocks are genuine non-writes
# for THIS kernel and stay zero (matching the oracle).
#
# Non-varlen path: cu_seqlens=None / chunk_indices=None, reproducing the
# wrapper's dense [B, T, H, K] branch byte-for-byte. No numeric logic added.
#
# Upstream invocation reproduced here (chunk_kda_scaled_dot_kkt_fwd, the
# inter launch at the wrapper's grid=(NT, NC*NC, B*H)):
#   q, k, g(=gk): [B, T, H, K]
#   beta:         [B, T, H]
#   A, Aqk:       [B, T, H, BT]   (zero-init; last axis is the LOCAL
#                                  within-chunk column)
#   BT = chunk_size = FLA_CHUNK_SIZE = 64 (pinned across siblings); the
#   wrapper computes BC = min(16, BT), NC = cdiv(BT, BC). BK is NOT passed to
#   the inter launch -- the kernel's @triton.autotune supplies BK from
#   [32, 64], so we must not pin it.
#
# Output keys (A, Aqk) match the oracle contract exactly.
_FLA_CHUNK_SIZE = 64


def run(inputs: dict) -> dict:
    q = inputs["q"]
    k = inputs["k"]
    gk = inputs["gk"]
    beta = inputs["beta"]
    scale = float(inputs["scale"])

    B, T, H, K = k.shape
    assert K <= 256

    BT = _FLA_CHUNK_SIZE
    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)
    NT = triton.cdiv(T, BT)

    A = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)
    Aqk = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)

    grid = (NT, NC * NC, B * H)
    chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        A=A,
        Aqk=Aqk,
        scale=scale,
        cu_seqlens=None,
        chunk_indices=None,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
    )
    return {"A": A, "Aqk": Aqk}
