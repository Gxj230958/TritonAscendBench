import torch
import triton


# Authored launcher: drives the in-module @triton.jit `_selective_scan_update_kernel`
# (vLLM 18f6bf5, vllm/model_executor/layers/mamba/ops/mamba_ssm.py) on the base
# Mamba2 decode path -- HAS_Z / HAS_D / HAS_DT_BIAS true, DT_SOFTPLUS true, no
# state_batch_indices / spec-decoding / varlen / stochastic rounding. Inputs are
# already in canonical rank (4D state, 3D x/dt/z), so the wrapper's unsqueeze
# bookkeeping is unnecessary; this launcher reproduces the wrapper's kernel
# launch verbatim EXCEPT it drops the `torch.accelerator.device_index` context
# (a device-placement detour the runner already manages, and an API the severed
# host's torch need not ship). The kernel's @triton.heuristics derive HAS_*,
# IS_*, and BLOCK_SIZE_DSTATE from the runtime args.
#
# Math (matches oracle_pytorch): state = state*exp(A*dt) + (B*dt)*x ;
# out = sum(state*C) + x*D ; out *= z*sigmoid(z). `out` is candidate-allocated;
# `state` is mutated in place. Output key: "out".


def run(inputs: dict) -> dict:
    state = inputs["state"]              # [B, H, dim, dstate]
    x = inputs["x"]                      # [B, H, dim]
    dt = inputs["dt"]                    # [B, H, dim]
    A = inputs["A"]                      # [H, dim, dstate]
    B = inputs["B"]                      # [B, G, dstate]
    C = inputs["C"]                      # [B, G, dstate]
    D = inputs["D"]                      # [H, dim]
    z = inputs["z"]                      # [B, H, dim]
    dt_bias = inputs["dt_bias"]          # [H, dim]

    batch, nheads, dim, dstate = state.shape
    ngroups = B.shape[1]
    # ssm output buffer in fp32 (the kernel stores out.to(out_ptr.element_ty)).
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)

    # Hand-tuned launch config (verbatim from selective_state_update), keyed on
    # dstate; no autotune (it would overwrite the state).
    if dstate <= 16:
        BLOCK_SIZE_M, num_warps = 32, 4
    elif dstate <= 32:
        BLOCK_SIZE_M, num_warps = 16, 4
    elif dstate <= 64:
        BLOCK_SIZE_M, num_warps = 8, 4
    elif dstate <= 128:
        BLOCK_SIZE_M, num_warps = 4, 4
    else:
        BLOCK_SIZE_M, num_warps = 4, 8

    tie_hdim = (
        A.stride(-1) == 0 and A.stride(-2) == 0
        and dt.stride(-1) == 0 and dt_bias.stride(-1) == 0
    )

    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)

    _selective_scan_update_kernel[grid](
        state,
        None,            # rand_seed_ptr (no stochastic rounding)
        x,
        dt,
        dt_bias,
        A,
        B,
        C,
        D,
        z,
        out,
        None,            # state_batch_indices_ptr
        None,            # dst_state_batch_indices_ptr
        0,               # null_block_id (unused without state_batch_indices)
        None,            # num_accepted_tokens_ptr
        None,            # cu_seqlens_ptr
        batch,           # N
        nheads,
        dim,
        dstate,
        nheads // ngroups,
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        x.stride(0), x.stride(1), x.stride(2),
        dt.stride(0), dt.stride(1), dt.stride(2),
        dt_bias.stride(0), dt_bias.stride(1),
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        D.stride(0), D.stride(1),
        z.stride(0), z.stride(1), z.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        0, 0,            # state_batch_indices strides
        0, 0,            # dst_state_batch_indices strides
        True,            # DT_SOFTPLUS
        tie_hdim,        # TIE_HDIM
        BLOCK_SIZE_M,
        num_warps=num_warps,
        USE_RS_ROUNDING=False,
        PHILOX_ROUNDS=0,
    )

    return {"out": out}
