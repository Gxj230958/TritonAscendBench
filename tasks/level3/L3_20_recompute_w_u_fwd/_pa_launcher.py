import torch
import triton


# Authored launcher: drives the in-module @triton.jit
# `recompute_w_u_fwd_kernel` exactly as the upstream host wrapper
# `recompute_w_u_fwd` does on its NON-VARLEN path (vLLM
# vllm/model_executor/layers/fla/ops/wy_fast.py, commit
# 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19).
#
# Why a launcher rather than binding the host wrapper directly: the
# wrapper signature is `recompute_w_u_fwd(k, v, beta, g_cumsum, A,
# cu_seqlens, chunk_indices=None)`. `cu_seqlens` has NO default, so the
# strict by-name signature adapter rejects it as shapes-incomplete. That
# parameter is the varlen-path control; the oracle (oracle_pytorch.py)
# models ONLY the non-varlen path (cu_seqlens is None), where the kernel
# processes one fixed BT-sized chunk per (batch, head) on a dense
# [B, T, ...] layout. Adding cu_seqlens to shapes.yaml would instead
# exercise the unmodelled varlen path. So we drive the kernel with
# cu_seqlens=None / chunk_indices=None, reproducing the wrapper's
# non-varlen branch byte-for-byte, and never touch numeric logic.
#
# Upstream invocation reproduced here (recompute_w_u_fwd, non-varlen):
#   k: [B, T, Hg, K]   (grouped/GQA key heads; value head h reads key
#                       head h // (H // Hg))
#   v: [B, T, H,  V]
#   beta:     [B, T, H]
#   g_cumsum: [B, T, H]   (kernel applies exp(g) for w; u has no gate)
#   A: [B, T, H, BT]      (per-chunk transition; last axis is the LOCAL
#                          within-chunk column, block-diagonal in time)
#   B, T, Hg, K, V = *k.shape, v.shape[-1]; H = v.shape[-2];
#   BT = A.shape[-1]; NT = cdiv(T, BT); BK = BV = 64.
#   u = empty_like(v); w = k.new_empty(B, T, H, K).
#   grid = (NT, B * H). The kernel writes w and u in place; the wrapper
#   returns (w, u).
#
# Output keys (w, u) match the oracle contract exactly.
def run(inputs: dict) -> dict:
    k = inputs["k"]
    v = inputs["v"]
    beta = inputs["beta"]
    g_cumsum = inputs["g_cumsum"]
    A = inputs["A"]

    B, T, Hg, K = k.shape
    V = v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    NT = triton.cdiv(T, BT)
    BK = 64
    BV = 64
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)

    recompute_w_u_fwd_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
        cu_seqlens=None,
        chunk_indices=None,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return {"w": w, "u": u}
