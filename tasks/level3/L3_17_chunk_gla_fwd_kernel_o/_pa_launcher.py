import torch


# Authored launcher: drives the in-module @triton.jit
# `chunk_gla_fwd_kernel_o` exactly as the upstream host wrapper
# `chunk_gla_fwd_o_gk` does on its NON-VARLEN path (vLLM
# vllm/model_executor/layers/fla/ops, commit
# 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19).
#
# Why a launcher rather than binding the host wrapper directly: the strict
# by-name signature adapter unpacks `B, T, H, K, V = *q.shape, v.shape[-1]`
# in the wrapper, which requires a rank-4 q [B, T, H, K]; it also takes a
# pre-allocated `o` output and a `cu_seqlens` varlen control with no usable
# default. The oracle (oracle_pytorch.py) models ONLY the non-varlen path
# (cu_seqlens is None), where the kernel processes one BT-sized chunk per
# (batch, head) on a dense head-structured layout. We allocate `o`, pass
# cu_seqlens=None, and set chunk_size = A.shape[-1] so the kernel's BT
# matches A's local-column extent exactly — reproducing the wrapper's
# non-varlen branch without touching numeric logic.
#
# Upstream invocation reproduced here (chunk_gla_fwd_o_gk, non-varlen):
#   q: [B, T, H, K]      gated query
#   g: [B, T, H, K]      log-gate (kernel applies exp(g))
#   v: [B, T, H, V]      values
#   A: [B, T, H, BT]     per-chunk intra score; last axis is the LOCAL
#                        within-chunk column (block-diagonal in time)
#   h: [B, NT, H, V, K]  per-chunk state, read as a [V, K] tile
#   o: [B, T, H, V]      output (written in place)
#   B, T, H, K = q.shape; V = v.shape[-1]; BT = A.shape[-1];
#   NT = cdiv(T, BT); grid = (cdiv(V, BV), NT, B * H). The wrapper passes
#   T, H, K, V and BT=chunk_size; BK/BV come from the autotuner.
#
# Output key (output) matches the oracle contract exactly.
def run(inputs: dict) -> dict:
    q = inputs["q"]
    v = inputs["v"]
    g = inputs["g"]
    A = inputs["A"]
    h = inputs["h"]
    scale = float(inputs["scale"])

    BT = A.shape[-1]
    o = torch.empty_like(v)

    # chunk_gla_fwd_o_gk is resolved from the kernel.py module namespace
    # the launcher is exec'd into (see _pa_oracle_check_runner._load).
    chunk_gla_fwd_o_gk(  # noqa: F821
        q=q,
        v=v,
        g=g,
        A=A,
        h=h,
        o=o,
        scale=scale,
        cu_seqlens=None,
        chunk_size=BT,
    )
    return {"output": o}
