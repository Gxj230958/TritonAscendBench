import triton

import torch


# Authored launcher: drives the in-module @triton.jit `experts_combine_kernel`
# exactly as the upstream host wrapper `experts_combine_triton` does (SGLang
# sglang/srt/layers/elementwise.py, commit
# 409d350fb6f6a1e7c7546e39028f811092a8e489).
#
# Why a launcher rather than the registered op: meta.oracle.gpu_triton is
# `kernel.py::placeholder` (no resolvable callable), so the bench cannot
# synthesise a run() for the bare @triton.jit. The host wrapper itself is
# severable (its only module-scope deps are `is_hip` -- a constant backend gate
# -- and the inert `register_custom_op` shim), and it touches only plain dense
# tensors. We reproduce its invocation verbatim.
#
# Upstream invocation reproduced here (experts_combine_triton):
#   moe_hidden_states: (bs, combine_k, hidden_dim) when 3D, else (bs,
#     hidden_dim) "pre-combined" with combine_k = 1.
#   mlp_hidden_states: (bs, hidden_dim) -- defines bs, hidden_dim.
#   combine_k = moe_hidden_states.shape[1] if 3D else 1.
#   out_hidden_states = empty_like(mlp_hidden_states), shape (bs, hidden_dim).
#   config = {BLOCK_SIZE = next_power_of_2(hidden_dim),
#             num_warps = max(min(next_power_of_2(cdiv(hidden_dim,1024)),8),4)}.
#   grid = (bs,).
#   experts_combine_kernel[(bs,)](out, moe, mlp, combine_k, hidden_dim,
#                                 **config).
#   Kernel math (per row pid): moe_x = sum_{k<combine_k} moe[pid,k,:];
#     combined = (moe_x + mlp[pid,:]) / sqrt(2). Output key "output", matching
#     the oracle contract; out dtype == mlp dtype.
def run(inputs: dict) -> dict:
    moe_hidden_states = inputs["moe_hidden_states"].contiguous()
    mlp_hidden_states = inputs["mlp_hidden_states"].contiguous()

    assert moe_hidden_states.is_contiguous()
    assert mlp_hidden_states.is_contiguous()

    if len(moe_hidden_states.shape) == 2:
        combine_k = 1  # pre-combined
    else:
        combine_k = moe_hidden_states.shape[1]

    out_hidden_states = torch.empty_like(mlp_hidden_states)

    bs, hidden_dim = mlp_hidden_states.shape

    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 1024)), 8), 4
        ),
    }

    experts_combine_kernel[(bs,)](
        out_hidden_states,
        moe_hidden_states,
        mlp_hidden_states,
        combine_k,
        hidden_dim,
        **config,
    )

    return {"output": out_hidden_states}
