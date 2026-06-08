"""Task L3_24_fused_recurrent_gated_delta_rule_update_fwd_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `fused_recurrent_gated_delta_rule_update_fwd_kernel`
Source path: `sglang/srt/layers/attention/fla/fused_recurrent.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): 3a36edee5c01e326541e525e2705152b90a96304dd2a43f22e1835b985166e17
Frozen by: scripts/phase_a_freeze.py
"""
from typing import Optional, Tuple
import torch
import triton
import triton.language as tl
from triton_ascend_bench.frozen.deps.h_e36d74582749.sglang_fla_op_utils.op import exp  # Phase A': U2 SGLang FLA op vendoring
from triton_ascend_bench.frozen.deps.h_e36d74582749.sglang_fla_op_utils.utils import input_guard  # Phase A': U2 SGLang FLA utils vendoring

@triton.jit(do_not_specialize=['T'])
def fused_recurrent_gated_delta_rule_update_fwd_kernel(q, k, v, g, beta, o, h0_source, h0_indices, cu_seqlens, scale, intermediate_states_buffer, intermediate_state_indices, cache_steps, retrieve_parent_token_ptr, stride_retrieve_parent_token_seq: tl.constexpr, stride_retrieve_parent_token_token: tl.constexpr, T, NP2_T: tl.constexpr, B: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, IS_BETA_HEADWISE: tl.constexpr, USE_QK_L2NORM_IN_KERNEL: tl.constexpr, IS_VARLEN: tl.constexpr, DISABLE_STATE_UPDATE: tl.constexpr, DISABLE_OUTPUT_CALCULATION: tl.constexpr, CACHE_INTERMEDIATE_STATES: tl.constexpr, HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr):
    i_k, i_v, i_nh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_n, i_hv = (i_nh // HV, i_nh % HV)
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64))
        all = T
        T = eos - bos
    else:
        bos, eos = (i_n * T, i_n * T + T)
        all = B * T
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    p_g = g + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v
    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        token_indices = tl.arange(0, NP2_T)
        mask_retrieve = token_indices < T
        retrieve_parent_token_base = retrieve_parent_token_ptr + i_n * stride_retrieve_parent_token_seq + token_indices * stride_retrieve_parent_token_token
        parent_idx_tokens = tl.load(retrieve_parent_token_base, mask_retrieve)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = h0_source + idx * HV * K * V + i_hv * K * V + o_v[:, None] * K + o_k[None, :]
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(intermediate_state_indices + i_n)
    step_idx = 0
    for _ in range(0, T):
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            if step_idx != 0 and cache_idx >= 0:
                parent_step_idx = tl.sum(tl.where(token_indices == step_idx, parent_idx_tokens, 0))
                step_offset = parent_step_idx * HV * K * V
                cache_ptr = intermediate_states_buffer + cache_idx * cache_steps * HV * K * V + step_offset + i_hv * K * V + o_v[:, None] * K + o_k[None, :]
                b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-06)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-06)
        b_q = b_q * scale
        b_h *= exp(b_g)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        if not DISABLE_OUTPUT_CALCULATION:
            b_o = tl.sum(b_h * b_q[None, :], 1)
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                step_offset = step_idx * HV * K * V
                cache_ptr = intermediate_states_buffer + cache_idx * cache_steps * HV * K * V + step_offset + i_hv * K * V + o_v[:, None] * K + o_k[None, :]
                tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)
        step_idx += 1
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_g += HV
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)
    if not DISABLE_STATE_UPDATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = h0_source + idx * HV * K * V + i_hv * K * V + o_v[:, None] * K + o_k[None, :]
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)

def fused_recurrent_gated_delta_rule_update_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, scale: float, initial_state_source: torch.Tensor, initial_state_indices: torch.Tensor, use_qk_l2norm_in_kernel: bool=False, cu_seqlens: Optional[torch.LongTensor]=None, disable_state_update: bool=False, disable_output_calculation: bool=False, intermediate_states_buffer: Optional[torch.Tensor]=None, intermediate_state_indices: Optional[torch.Tensor]=None, cache_steps: Optional[int]=None, retrieve_parent_token: Optional[torch.Tensor]=None) -> torch.Tensor:
    B, T, H, K, V = (*k.shape, v.shape[-1])
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = (triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8))
    NK, NV = (triton.cdiv(K, BK), triton.cdiv(V, BV))
    assert NK == 1, 'NK > 1 is not supported yet'
    num_stages = 3
    num_warps = 1
    if disable_output_calculation:
        o = q.new_empty(NK, 1, 1, 1, 1)
    else:
        o = q.new_empty(NK, *v.shape)
    grid = (NK, NV, N * HV)
    if retrieve_parent_token is not None:
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token = (retrieve_parent_token.stride(0), retrieve_parent_token.stride(1))
    else:
        stride_retrieve_parent_token_seq = stride_retrieve_parent_token_token = 0
    NP2_T = triton.next_power_of_2(T)
    fused_recurrent_gated_delta_rule_update_fwd_kernel[grid](q=q, k=k, v=v, g=g, beta=beta, o=o, h0_source=initial_state_source, h0_indices=initial_state_indices, cu_seqlens=cu_seqlens, scale=scale, intermediate_states_buffer=intermediate_states_buffer, intermediate_state_indices=intermediate_state_indices, cache_steps=0 if cache_steps is None else cache_steps, retrieve_parent_token_ptr=retrieve_parent_token, stride_retrieve_parent_token_seq=stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token=stride_retrieve_parent_token_token, T=T, NP2_T=NP2_T, B=B, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, USE_INITIAL_STATE=initial_state_source is not None, IS_VARLEN=cu_seqlens is not None, CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None, HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_parent_token is not None, IS_BETA_HEADWISE=beta.ndim == v.ndim, USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel, DISABLE_STATE_UPDATE=disable_state_update, DISABLE_OUTPUT_CALCULATION=disable_output_calculation, num_warps=num_warps, num_stages=num_stages)
    o = o.squeeze(0)
    return o

import torch


# Authored launcher: drives the in-module host wrapper
# ``fused_recurrent_gated_delta_rule_update_fwd`` (SGLang
# sglang/srt/layers/attention/fla/fused_recurrent.py, commit
# 409d350fb6f6a1e7c7546e39028f811092a8e489), which in turn launches the
# DECLARED graded @triton.jit kernel
# ``fused_recurrent_gated_delta_rule_update_fwd_kernel``. The launcher
# never reimplements the recurrence -- it only marshals inputs and
# assembles the wrapper's return + the in-place state pool into the
# oracle's contract keys.
#
# Why a launcher rather than binding the host wrapper directly: the
# wrapper signature is
#   fused_recurrent_gated_delta_rule_update_fwd(q, k, v, g, beta, scale,
#       initial_state_source, initial_state_indices,
#       use_qk_l2norm_in_kernel=False, cu_seqlens=None,
#       disable_state_update=False, disable_output_calculation=False,
#       intermediate_states_buffer=None, intermediate_state_indices=None,
#       cache_steps=None, retrieve_parent_token=None)
# ``initial_state_source`` and ``initial_state_indices`` have NO defaults,
# so the strict by-name adapter rejects the wrapper as shapes-incomplete.
# These are not extra free inputs: the UPDATE kernel always reads its
# per-sequence state from an indexed pool ``h0_source[h0_indices[n]]`` and,
# on the default path, writes the final state back IN-PLACE to that same
# slot. We materialise the pool from the real ``initial_state`` shapes.yaml
# tensor (one slot per sequence, ``initial_state_indices = arange(B)``), so
# sequence ``n`` reads and overwrites slot ``n``. Every other control is
# pinned to the dense, equal-length, scalar-beta, non-KDA, no-L2norm,
# no-cache, no-eagle-tree path the oracle models:
#   cu_seqlens=None / use_qk_l2norm_in_kernel=False / cache=None /
#   retrieve_parent_token=None and disable_state_update=False (final state
#   written) / disable_output_calculation=False (output computed).
#
# Why one dense call covers B>1: unlike L3_14's ``fwd`` (which packs a
# per-token state STACK indexed by ``bos + i_t`` and is only in-bounds for
# B=1), the ``update`` kernel keeps a single [HV, V, K] state per sequence
# in the pool and addresses it by ``i_n`` (kernel.py grid (NK, NV, N*HV),
# bos = i_n * T). With ``initial_state_indices = arange(B)`` each program
# n reads pool slot n and writes its end state back to slot n, so the whole
# batch resolves in one launch.
#
# State / output layout (D = Dv = K = V, HV = H here):
#   initial_state : [B, HV, V, K] in the kernel's V-major state layout
#                   (b_h is [BV, BK] addressed o_v[:, None]*K + o_k[None,:]).
#   o             : the wrapper allocates [NK, B, T, HV, V] and squeezes
#                   the NK axis itself (kernel.py: o = o.squeeze(0)), so it
#                   returns [B, T, HV, V] = [B, T, H, D] -- we pass it
#                   through unchanged (squeezing again would clobber B=1).
#   final_state   : the wrapper mutates ``initial_state_source`` in place
#                   with each sequence's end state; we pass a clone as the
#                   pool so the returned ``final_state`` is the post-update
#                   pool [B, HV, V, K], matching the oracle's final S.
#
# Output keys (output, final_state) match the oracle contract.
def run(inputs: dict) -> dict:
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    g = inputs["g"]
    beta = inputs["beta"]
    scale = float(inputs["scale"])
    initial_state = inputs["initial_state"]

    B = q.shape[0]
    # Real state pool: one [HV, V, K] slot per sequence. The wrapper writes
    # each sequence's FINAL state back here in place (disable_state_update is
    # left False), so this clone IS the returned final_state after the call.
    state_pool = initial_state.to(torch.float32).contiguous().clone()
    state_indices = torch.arange(B, device=q.device, dtype=torch.long)

    o = fused_recurrent_gated_delta_rule_update_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state_source=state_pool,
        initial_state_indices=state_indices,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
        disable_state_update=False,
        disable_output_calculation=False,
        intermediate_states_buffer=None,
        intermediate_state_indices=None,
        cache_steps=None,
        retrieve_parent_token=None,
    )

    # The wrapper already squeezes the NK axis; o is [B, T, HV, V] = [B,T,H,D].
    output = o
    # state_pool now holds each sequence's final state at its own slot.
    final_state = state_pool.to(initial_state.dtype)
    return {"output": output, "final_state": final_state}
