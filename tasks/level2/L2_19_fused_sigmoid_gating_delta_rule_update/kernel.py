"""Task L2_19_fused_sigmoid_gating_delta_rule_update_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `fused_sigmoid_gating_delta_rule_update_kernel`
Source path: `sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): f4f26ea5cf979b75fd340665e65f1e2d33ab586a501928eb20b1b88f5ed84799
Frozen by: scripts/phase_a_freeze.py
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    # Parameters for target_verify support (unused for decode)
    intermediate_states_buffer,
    intermediate_state_indices,
    cache_steps,
    retrieve_parent_token_ptr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    # ================================================
    scale,
    T,
    stride_a,
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    NP2_T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
    # Optional flags for target_verify support (default False for decode)
    DISABLE_STATE_UPDATE: tl.constexpr = False,
    CACHE_INTERMEDIATE_STATES: tl.constexpr = False,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr = False,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_b = b + bos * stride_b + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    if IS_KDA:
        p_a = a + bos * stride_a + i_hv * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
    else:
        p_a = a + bos * stride_a + i_hv
        p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_v[None, :] * K
                + o_k[:, None]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Preload tree attention data if needed
    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        token_indices = tl.arange(0, NP2_T)
        mask_retrieve = token_indices < T
        retrieve_parent_token_base = (
            retrieve_parent_token_ptr
            + (i_n * stride_retrieve_parent_token_seq)
            + token_indices * stride_retrieve_parent_token_token
        )
        parent_idx_tokens = tl.load(
            retrieve_parent_token_base, mask=mask_retrieve, other=0
        )

    # Prepare intermediate state cache index if enabled
    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(intermediate_state_indices + i_n)

    step_idx = 0
    for _ in range(0, T):
        # Tree attention: load parent's cached state
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            # step_idx == 0 uses b_h from USE_INITIAL_STATE
            if step_idx != 0 and cache_idx >= 0:
                parent_step_idx = tl.sum(
                    tl.where(token_indices == step_idx, parent_idx_tokens, 0)
                )
                step_offset = parent_step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)

        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32)
        if IS_KDA:
            b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias, mask=mask_k, other=0).to(tl.float32)
        else:
            b_a = tl.load(p_a).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        # Apply softplus with numerical stability
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        if IS_KDA:
            b_h *= tl.exp(b_g[:, None])
        else:
            b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Cache intermediate states if enabled
        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                step_offset = step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)

        step_idx += 1

        # Update pointers for next timestep
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_b += stride_b
        p_o += HV * V
        p_a += stride_a

    # Store final state back to h0_source with bounds checking
    if not DISABLE_STATE_UPDATE:
        if USE_INITIAL_STATE:
            idx = tl.load(h0_indices + i_n)
            if idx >= 0:
                p_h0 = (
                    h0_source
                    + idx * HV * K * V
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
    # Optional parameters for target_verify support
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[int] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.

    Supports both decode and target_verify modes:
    - decode: standard single-step update with state write-back
    - target_verify: multi-step with intermediate state caching, optional tree attention,
                     and optional state update disable
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    stride_q = q.stride()[1]
    stride_k = k.stride()[1]
    stride_v = v.stride()[1]
    stride_b = b.stride()[-2]
    # Both paths (KDA/GDN) advance p_a once per token, so use the token-axis stride.
    # For 2D a ([T, ...]) this is stride(0); for 3D a ([B, T, ...]) this is stride(1).
    # Using stride()[-2] covers GDN [T, HV] and KDA layouts ([T, HV*K] / [B, T, HV*K]).
    stride_a = a.stride()[-2]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)

    # Prepare retrieve_parent_token strides
    if retrieve_parent_token is not None:
        stride_retrieve_parent_token_seq = retrieve_parent_token.stride(0)
        stride_retrieve_parent_token_token = retrieve_parent_token.stride(1)
    else:
        stride_retrieve_parent_token_seq = 0
        stride_retrieve_parent_token_token = 0

    NP2_T = triton.next_power_of_2(T)

    grid = (NK, NV, N * HV)

    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=0 if cache_steps is None else cache_steps,
        retrieve_parent_token_ptr=retrieve_parent_token,
        stride_retrieve_parent_token_seq=stride_retrieve_parent_token_seq,
        stride_retrieve_parent_token_token=stride_retrieve_parent_token_token,
        scale=scale,
        T=T,
        stride_a=stride_a,
        stride_q=stride_q,
        stride_k=stride_k,
        stride_v=stride_v,
        stride_b=stride_b,
        NP2_T=NP2_T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state_source is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        IS_KDA=is_kda,
        DISABLE_STATE_UPDATE=disable_state_update,
        CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_parent_token is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o

import torch

# Authored launcher: drives the in-module frozen host wrapper
# `fused_sigmoid_gating_delta_rule_update` (SGLang
# sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py, commit
# 409d350fb6f6a1e7c7546e39028f811092a8e489), which launches the graded
# @triton.jit `fused_sigmoid_gating_delta_rule_update_kernel`.
#
# Why a launcher rather than a direct by-name bind to the wrapper: the wrapper's
# decode ABI is the model-serving layout, NOT the flat single-step fixtures the
# oracle/shapes describe. Reconciling the two is the launcher's whole job:
#
#  * SHAPE/LAYOUT. The kernel is a per-(seq, head) recurrence over a token axis
#    `for _ in range(0, T)`. The oracle is the SINGLE-STEP gated-delta update on
#    flat [B,H,D] tensors + a [B,H,D,D] carried state, so we run T=1. Crucially
#    the kernel reads the gating parameters A_log/dt_bias as PER-HEAD [HV] (the
#    `A_log + i_hv` / `dt_bias + i_hv` pointers carry no batch index), while the
#    benchmark's A_log is per-(batch,head) [B,H]. A single N=B launch would feed
#    one shared A_log row to every batch element -- dropping the per-batch
#    variation the oracle models. We therefore drive ONE sequence per batch
#    element (N=1, T=1, HV=H per call), forwarding that row's own A_log[b] / a[b]
#    / q[b] / k[b] / v[b] / beta[b] / state[b]. The batch axis carries no
#    cross-element coupling in this recurrence, so the per-element loop is
#    mathematically identical to a batched call; every input stays load-bearing.
#
#  * STATE LAYOUT. The kernel's state slot stores element [k,v] at flat offset
#    `o_v*K + o_k`, i.e. h0_source is laid out [idx, HV, V, K] (v outer, k
#    inner), whereas the oracle state is [B,H,k,v]. We materialise the kernel's
#    [1,H,V,K] view by transposing the last two axes in, and transpose back on
#    read of the in-place-updated state. h0_indices=[0] routes the single
#    sequence to slot 0 (idx>=0 enables the USE_INITIAL_STATE load + write-back).
#
#  * BETA. The kernel's `b` input is the PRE-sigmoid gate logit: it computes
#    `b_beta = sigmoid(b)` and uses that as the delta coefficient. The oracle's
#    `beta` (dist rand, in [0,1)) IS the post-sigmoid effective coefficient
#    (delta = beta*(v - k.S)). To drive the kernel to the SAME effective beta we
#    feed `b = logit(beta)` so `sigmoid(b) == beta` exactly. This is the faithful
#    preimage of the kernel's own sigmoid, not a fabricated input -- `b` is
#    load-bearing (perturbing beta shifts the output). The clamp only guards the
#    open-interval endpoints of `rand` (logit(0)/logit(1) = -+inf).
#
#  * SCALE / L2NORM. The oracle is the core update with no query scaling and no
#    L2 norm (ORACLE-SCOPE excludes USE_QK_L2NORM_IN_KERNEL), so we pass
#    scale=1.0 and use_qk_l2norm_in_kernel=False -- matching the oracle exactly.
#    softplus_beta/threshold are forwarded from the fixtures (the kernel reads
#    them in the gating softplus; both are load-bearing).
#
# All optional target_verify / tree-attention / varlen machinery is held inert
# (cu_seqlens=None, is_kda=False, no intermediate-state buffer, no
# retrieve_parent_token) -- NOT removed; the graded jit is invoked verbatim.
def run(inputs: dict) -> dict:
    A_log = inputs["A_log"]
    a = inputs["a"]
    dt_bias = inputs["dt_bias"]
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    beta = inputs["beta"]
    state = inputs["state"]
    softplus_beta = float(inputs.get("softplus_beta", 1.0))
    softplus_threshold = float(inputs.get("softplus_threshold", 20.0))

    B, H, D = q.shape
    Dv = v.shape[-1]

    out = torch.empty(B, H, Dv, dtype=q.dtype, device=q.device)
    new_state = torch.empty_like(state)

    # Per-head gating params are shared across the (one-token) sequence; the
    # kernel reads them as [HV]. dt_bias is already [H]; A_log is sliced per
    # batch element below.
    dt_bias_h = dt_bias.contiguous()

    for bi in range(B):
        # One sequence (N=1), one token (T=1), HV=H. Kernel input layouts:
        #   q/k: [B=1, T=1, H, K]   v: [B=1, T=1, HV, V]
        #   a/b: [B=1, T=1, HV]     A_log/dt_bias: [HV]
        q_b = q[bi].reshape(1, 1, H, D).contiguous()
        k_b = k[bi].reshape(1, 1, H, D).contiguous()
        v_b = v[bi].reshape(1, 1, H, Dv).contiguous()
        a_b = a[bi].reshape(1, 1, H).contiguous()
        A_log_b = A_log[bi].contiguous()  # [H]
        # b = logit(beta): the kernel applies sigmoid(b); the oracle's beta is
        # the post-sigmoid effective coefficient, so this preimage makes the
        # kernel's effective beta equal beta.
        b_b = torch.logit(beta[bi].clamp(1e-6, 1.0 - 1e-6)).reshape(1, 1, H).contiguous()

        # Initial state in the kernel's [idx=1, HV, V, K] layout. Oracle state is
        # [H, k, v] -> permute last two axes to [H, v, k].
        h0 = state[bi].permute(0, 2, 1).reshape(1, H, Dv, D).contiguous()
        h0_indices = torch.zeros(1, dtype=torch.int32, device=q.device)

        o = fused_sigmoid_gating_delta_rule_update(  # noqa: F821 (in kernel.py module)
            A_log=A_log_b,
            a=a_b,
            dt_bias=dt_bias_h,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q_b,
            k=k_b,
            v=v_b,
            b=b_b,
            initial_state_source=h0,
            initial_state_indices=h0_indices,
            scale=1.0,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=None,
            is_kda=False,
        )
        # o: [B=1, T=1, HV, V] -> [H, V]
        out[bi] = o.reshape(H, Dv)
        # h0 was updated in place with the new state; transpose [H, v, k] back
        # to the oracle's [H, k, v].
        new_state[bi] = h0[0].permute(0, 2, 1).contiguous()

    return {"output": out, "state": new_state}
