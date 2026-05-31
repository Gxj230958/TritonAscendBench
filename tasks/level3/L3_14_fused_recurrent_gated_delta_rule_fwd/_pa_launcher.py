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
