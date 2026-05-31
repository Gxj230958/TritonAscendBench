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
