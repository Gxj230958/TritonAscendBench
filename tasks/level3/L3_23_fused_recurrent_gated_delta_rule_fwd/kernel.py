"""Task L3_23_fused_recurrent_gated_delta_rule_fwd_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `fused_recurrent_gated_delta_rule_fwd_kernel`
Source path: `vllm/model_executor/layers/fla/ops/fused_recurrent.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): 3a2a3c5245acba127af326f0b3f287b352776a8eb1e1f6e40c24a767fae3c2ef
"""
import torch
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
from ascendbench.frozen.deps.h_af43637ae818.fla_ops.op import exp  # Phase A': U2 FLA-ops package vendoring

@triton.heuristics({'USE_INITIAL_STATE': lambda args: args['h0'] is not None, 'IS_VARLEN': lambda args: args['cu_seqlens'] is not None, 'IS_CONTINUOUS_BATCHING': lambda args: args['ssm_state_indices'] is not None, 'IS_SPEC_DECODING': lambda args: args['num_accepted_tokens'] is not None})
@triton.jit(do_not_specialize=['N', 'T'])
def fused_recurrent_gated_delta_rule_fwd_kernel(q, k, v, g, beta, o, h0, ht, cu_seqlens, ssm_state_indices, num_accepted_tokens, scale, N: tl.int64, T: tl.int64, B: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, stride_init_state_token: tl.constexpr, stride_final_state_token: tl.constexpr, stride_indices_seq: tl.constexpr, stride_indices_tok: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, INPLACE_FINAL_STATE: tl.constexpr, IS_BETA_HEADWISE: tl.constexpr, USE_QK_L2NORM_IN_KERNEL: tl.constexpr, IS_VARLEN: tl.constexpr, IS_CONTINUOUS_BATCHING: tl.constexpr, IS_SPEC_DECODING: tl.constexpr, IS_KDA: tl.constexpr):
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
    if T == 0:
        return
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    if not IS_KDA:
        p_g = g + bos * HV + i_hv
    else:
        p_gk = g + (bos * HV + i_hv) * K + o_k
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(tl.int64)
            if state_idx <= 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-06)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-06)
        b_q = b_q * scale
        if not IS_KDA:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= exp(b_g)
        else:
            b_gk = tl.load(p_gk).to(tl.float32)
            b_h *= exp(b_gk[None, :])
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        if INPLACE_FINAL_STATE:
            final_state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(tl.int64)
            if final_state_idx > 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        if not IS_KDA:
            p_g += HV
        else:
            p_gk += HV * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)

def fused_recurrent_gated_delta_rule_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, scale: float, initial_state: torch.Tensor, inplace_final_state: bool=True, cu_seqlens: torch.Tensor | None=None, ssm_state_indices: torch.Tensor | None=None, num_accepted_tokens: torch.Tensor | None=None, use_qk_l2norm_in_kernel: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = (*k.shape, v.shape[-1])
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = (triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32))
    NK, NV = (triton.cdiv(K, BK), triton.cdiv(V, BV))
    assert NK == 1, 'NK > 1 is not supported yet'
    num_stages = 3
    num_warps = 1
    o = q.new_empty(NK, *v.shape)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(T, HV, V, K, dtype=initial_state.dtype)
    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)
    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = (1, 1)
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = (ssm_state_indices.stride(0), 1)
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()
    grid = (NK, NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](q=q, k=k, v=v, g=g, beta=beta, o=o, h0=initial_state, ht=final_state, cu_seqlens=cu_seqlens, ssm_state_indices=ssm_state_indices, num_accepted_tokens=num_accepted_tokens, scale=scale, N=N, T=T, B=B, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, stride_init_state_token=stride_init_state_token, stride_final_state_token=stride_final_state_token, stride_indices_seq=stride_indices_seq, stride_indices_tok=stride_indices_tok, IS_BETA_HEADWISE=beta.ndim == v.ndim, USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel, INPLACE_FINAL_STATE=inplace_final_state, IS_KDA=False, num_warps=num_warps, num_stages=num_stages)
    o = o.squeeze(0)
    return (o, final_state)

import torch


# Authored launcher: drives the in-module host wrapper
# ``fused_recurrent_gated_delta_rule_fwd`` (vLLM
# vllm/model_executor/layers/fla/ops/fused_recurrent.py, commit
# 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19), which in turn launches the
# DECLARED graded @triton.jit kernel
# ``fused_recurrent_gated_delta_rule_fwd_kernel``. The launcher never
# reimplements the recurrence -- it only marshals inputs and assembles
# the wrapper's returns into the oracle's contract keys.
#
# Why a launcher rather than binding the host wrapper directly: the
# wrapper signature is
#   fused_recurrent_gated_delta_rule_fwd(q, k, v, g, beta, scale,
#       initial_state, inplace_final_state=True, cu_seqlens=None,
#       ssm_state_indices=None, num_accepted_tokens=None,
#       use_qk_l2norm_in_kernel=False)
# ``initial_state`` has NO default, so the strict by-name adapter rejects
# the wrapper as shapes-incomplete. We supply it (a real shapes.yaml
# input) and pin the remaining controls to the dense, equal-length,
# scalar-beta, non-KDA, no-L2norm path the oracle models.
#
# Why one sequence per call (B=1): the wrapper's dense (non-varlen) path
# is only in-bounds for a single sequence. With inplace_final_state=False
# the kernel writes the per-token state stack into
# ``final_state = q.new_empty(T, HV, V, K)`` indexed by ``bos + i_t`` with
# ``bos = i_n * T`` (kernel.py:177), so a B>1 dense call would index past
# the T-sized buffer. (inplace_final_state=True is also unusable here: its
# store branch unconditionally loads ``ssm_state_indices`` (kernel.py:168),
# which we pass as None.) So we iterate the batch, calling the REAL kernel
# once per sequence; each call's ``bos`` is 0, reading the initial state
# from slot 0 and writing the T-step stack. The final state of a sequence
# is stack slot T-1.
#
# State / output layout (D = Dv = K = V here):
#   o          : wrapper returns [1, T, HV, V] after squeeze -> [T, H, D]
#   final_state: [T, HV, V, K]; slot -1 is the sequence's end state
#                [HV, V, K] = [H, V, K], the kernel's V-major state layout
#                which the oracle mirrors exactly.
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
    outs = []
    finals = []
    for b in range(B):
        o_b, fs_b = fused_recurrent_gated_delta_rule_fwd(
            q=q[b : b + 1].contiguous(),
            k=k[b : b + 1].contiguous(),
            v=v[b : b + 1].contiguous(),
            g=g[b : b + 1].contiguous(),
            beta=beta[b : b + 1].contiguous(),
            scale=scale,
            initial_state=initial_state[b : b + 1].contiguous(),
            inplace_final_state=False,
            cu_seqlens=None,
            ssm_state_indices=None,
            num_accepted_tokens=None,
            use_qk_l2norm_in_kernel=False,
        )
        # o_b: [1, T, H, D]; fs_b: [T, H, V, K] per-token state stack.
        outs.append(o_b[0])
        finals.append(fs_b[-1])

    output = torch.stack(outs, dim=0)        # [B, T, H, D]
    final_state = torch.stack(finals, dim=0)  # [B, H, V, K]
    return {"output": output, "final_state": final_state}
