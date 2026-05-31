"""Oracle reference for L3_14_fused_recurrent_gated_delta_rule_fwd_kernel.

Tier: formula_exact_manual
Tolerance class: recurrent_chunk
Category: ssm

The kernel under test is the GPU triton implementation vendored in
kernel.py (``fused_recurrent_gated_delta_rule_fwd_kernel``); this file
authors an eager PyTorch reference that reproduces that kernel's exact
token-by-token recurrence -- the kernel is the frozen ground truth, so
the oracle models ITS update order, not a textbook variant.

Kernel recurrence (kernel.py token loop, scalar-beta / non-KDA path),
with state ``S`` carried as ``[V, K]`` (the kernel's ``b_h`` is
``[BV, BK]`` addressed ``o_v[:, None] * K + o_k[None, :]``):

    S  = exp(g_t) * S                  # decay applied BEFORE kS
    kS = sum_k (S * k_t)               # contraction over K, uses the
                                       # already-decayed state
    delta = (v_t - kS) * beta_t
    S  = S + delta[:, None] * k_t[None, :]
    o_t = sum_k (S * (scale * q_t))    # contraction over K

This is NOT the "decay-after-kS" formulation a naive reference uses
(kS on the un-decayed state, decay folded into the update); on randn
inputs the two differ by O(1) per element, so the order matters.

q, k, v: [B, T, H, D]; g, beta: [B, T, H]; initial_state: [B, H, V, K]
in the kernel's V-major state layout (here D = Dv = K = V, square).

ORACLE-SCOPE (tier=formula_exact_manual): covers the dense,
equal-length, scalar-beta, non-KDA, no-L2norm recurrent gated-delta
forward. NOT covered (the kernel's optional branches): varlen
(cu_seqlens), continuous batching (ssm_state_indices), speculative
decoding (num_accepted_tokens), USE_QK_L2NORM_IN_KERNEL, IS_KDA
per-channel gate, headwise (vector) beta, and HV > H grouped value
heads.
"""
from __future__ import annotations

import torch


def _fused_recurrent_gated_delta_rule(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
    beta: torch.Tensor, scale: float,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kernel-faithful gated DeltaNet recurrent forward.

    Returns the per-token output ``o`` [B, T, H, Dv] and the final
    state ``S`` [B, H, V, K] (the kernel's V-major state layout). See
    the module docstring for the exact update order.
    """
    qf, kf, vf = (t.to(torch.float32) for t in (q, k, v))
    gf, bf = g.to(torch.float32), beta.to(torch.float32)
    B, T, H, D = qf.shape
    Dv = vf.shape[-1]
    # State S is [B, H, V, K] = [B, H, Dv, D] -- the kernel's b_h layout.
    if initial_state is not None:
        S = initial_state.to(torch.float32).clone()
    else:
        S = qf.new_zeros(B, H, Dv, D)
    o = qf.new_zeros(B, T, H, Dv)
    for t in range(T):
        # Decay the carried state FIRST (kernel: b_h *= exp(b_g)).
        decay = torch.exp(gf[:, t]).reshape(B, H, 1, 1)
        S = decay * S
        # kS uses the DECAYED state: sum over K (kernel: tl.sum(b_h*b_k, 1)).
        kS = torch.einsum("bhvk,bhk->bhv", S, kf[:, t])
        delta = (vf[:, t] - kS) * bf[:, t].unsqueeze(-1)
        # Rank-1 state update (kernel: b_h += b_v[:, None] * b_k[None, :]).
        S = S + torch.einsum("bhv,bhk->bhvk", delta, kf[:, t])
        # Output reads the UPDATED state (kernel: b_o = tl.sum(b_h*b_q, 1)).
        o[:, t] = torch.einsum("bhvk,bhk->bhv", S, qf[:, t]) * scale
    return o.to(q.dtype), S.to(q.dtype)


def run(inputs: dict) -> dict:
    o, s = _fused_recurrent_gated_delta_rule(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"],
        inputs["beta"], float(inputs.get("scale", 1.0)),
        inputs.get("initial_state"))
    return {"output": o, "final_state": s}
