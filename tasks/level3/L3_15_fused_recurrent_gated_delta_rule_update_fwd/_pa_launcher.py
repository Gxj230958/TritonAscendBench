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
