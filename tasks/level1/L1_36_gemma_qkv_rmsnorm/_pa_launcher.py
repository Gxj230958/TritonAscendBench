import triton


# Authored launcher: drives the in-module @triton.jit `_gemma_qkv_rmsnorm_kernel`
# (SGLang 409d350, sglang/srt/layers/gemma4_fused_ops.py) exactly as its host
# wrapper `gemma_qkv_rmsnorm` does on the full Q+K+V path (HAS_KV=True). The
# wrapper pulls a stray `from sglang.srt.utils import is_cuda` into the module;
# this launcher bypasses the wrapper and drives the @triton.jit kernel directly
# so the graded surface is just the kernel + plain triton.
#
# Math (matches oracle_pytorch): per token, per head, RMS-normalize each head's
# vector -- out = x * rsqrt(mean(x^2)+eps) * w -- with Q scaled by q_weight,
# K by k_weight, V by 1. The kernel mutates q, k, v in place; output keys mirror
# the input names so mutated-input comparison sees the post-norm tensors.
EPS = 1e-6


def run(inputs: dict) -> dict:
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    q_weight = inputs["q_weight"]
    k_weight = inputs["k_weight"]

    M = q.shape[0]
    head_dim = q_weight.shape[-1]
    num_q_heads = q.shape[1] // head_dim
    num_kv_heads = k.shape[1] // head_dim
    BLOCK = triton.next_power_of_2(head_dim)

    _gemma_qkv_rmsnorm_kernel[(M,)](
        q,
        k,
        v,
        q_weight,
        k_weight,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        eps=EPS,
        HAS_KV=True,
        BLOCK=BLOCK,
    )

    return {"q": q, "k": k, "v": v}
