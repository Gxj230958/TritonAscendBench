import torch
import triton


# Drives the in-module @triton.jit ``_correct_attn_cp_out_kernel`` exactly as
# the upstream host wrapper ``correct_attn_out`` does (vLLM
# vllm/v1/attention/ops/common.py, commit 18f6bf5a). The launch config below is
# a verbatim mirror of that wrapper: same grid, same pointer/stride/constexpr
# argument order, same in-place output aliasing and same auxiliary lse buffer.
#
# Why a launcher rather than binding ``correct_attn_out`` directly: the wrapper
# signature is ``correct_attn_out(out, lses, cp_rank, ctx, is_lse_base_on_e)``;
# ``ctx`` is a required ``CPTritonContext`` (a JIT-recompile cache, not a
# tensor) with no default, so the strict by-name input adapter rejects the
# wrapper as shapes-incomplete. The kernel itself reads only plain dense
# tensors plus the two scalars, so we marshal the wrapper's exact launch from
# the dense oracle inputs. (Same pattern as the passed siblings
# L2_15_chunked_lora_shrink / L3_14 / L3_20.)
#
# Launch contract (mirrors correct_attn_out, kernel.py:124-192):
#   grid = (B, H, 1); program_id(0)=batch, program_id(1)=head. Each program
#   reduces lses over the N (rank) axis to the global log-sum-exp, writes it to
#   the auxiliary ``lse`` buffer, then reweights this rank's ``out[b,h,:]`` by
#   ``exp(lses[cp_rank,b,h] - global_lse)``. ``out`` is passed as BOTH
#   ``outputs_ptr`` and ``new_output_ptr`` (the upstream in-place correction);
#   we reproduce that by writing the corrected result back into ``out``.
#   HEAD_DIM=D, N_ROUNDED=N (=E, the rank count), IS_BASE_E=is_lse_base_on_e.
#   ``cp_rank`` selects which rank's partial output is being corrected and is
#   forwarded verbatim as the ``lse_idx`` runtime arg -- it is load-bearing:
#   the reweight factor exp(lses[cp_rank] - lse) depends on it.
#
# The ``lse`` aux buffer keeps the same B/H strides as a slice of ``lses`` so
# the kernel's ``vlse_ptr`` stores land correctly even when ``lses`` is a
# non-contiguous view (exactly as the wrapper allocates it).
def run(inputs: dict) -> dict:
    out = inputs["out"]
    lses = inputs["lses"]
    cp_rank = int(inputs.get("cp_rank", 0))
    is_lse_base_on_e = bool(inputs.get("is_lse_base_on_e", True))

    assert out.ndim == 3, f"expected out [B,H,D], got {tuple(out.shape)}"
    assert lses.ndim == 3, f"expected lses [N,B,H], got {tuple(lses.shape)}"

    B, H, D = out.shape
    N = lses.shape[0]

    # Correct in place into a fresh copy so the oracle's input tensor (shared by
    # the comparison harness) is never mutated; the kernel aliases outputs_ptr
    # and new_output_ptr, so both must point at this same buffer.
    out = out.clone()

    o_sB, o_sH, o_sD = out.stride()
    l_sN, l_sB, l_sH = lses.stride()

    # Auxiliary global-lse buffer with the same B/H strides as a slice of lses
    # (the kernel addresses vlse_ptr via lses_stride_B/H).
    lse = torch.empty_strided(
        (B, H), (l_sB, l_sH), device=lses.device, dtype=lses.dtype
    )

    grid = (B, H, 1)
    _correct_attn_cp_out_kernel[grid](
        out,
        out,
        lses,
        lse,
        o_sB,
        o_sH,
        o_sD,
        l_sN,
        l_sB,
        l_sH,
        cp_rank,
        HEAD_DIM=D,
        N_ROUNDED=N,
        IS_BASE_E=is_lse_base_on_e,
    )

    return {"output": out}
