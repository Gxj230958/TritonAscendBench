import torch


# Authored launcher: drives the in-module host wrapper
# `concat_and_cast_mha_k_triton`, which itself launches the @triton.jit
# `concat_and_cast_mha_k_kernel`, exactly as the upstream call site does
# (sglang/srt/layers/attention/utils.py + the sarvam_moe.py /
# forward_mha.py `_concat_and_cast_mha_k` host paths; SGLang commit
# 409d350fb6f6a1e7c7546e39028f811092a8e489).
#
# Why a launcher: meta.oracle.gpu_triton was the `placeholder` sentinel, so
# the harness had no callable to drive the frozen kernel ("original side did
# not run"). The host wrapper `concat_and_cast_mha_k_triton` lives in the same
# kernel.py and is pure grid/stride/scalar arithmetic (no vllm/sglang runtime
# closure), so calling it directly is both faithful and severable. The harness
# exec's this file into the kernel.py module namespace, so the wrapper symbol
# is in scope here.
#
# Upstream contract reproduced (DeepSeek-V3 MLA K assembly):
#   k_nope : [B, H, Dn]      per-head nope part (B = n_tokens, H = heads)
#   k_rope : [B, 1, Dr]      shared rope part, BROADCAST across the H heads
#   out_dtype : str          cache dtype to cast the assembled K to
#   k = k_nope.new_empty(B, H, Dn + Dr, dtype=out_dtype)
#   concat_and_cast_mha_k_triton(k, k_nope, k_rope)
# The kernel stores k[:, h, :Dn] = k_nope[:, h, :] for every head and
# k[:, h, Dn:] = k_rope[:, 0, :] for every head (the rope read carries no head
# term -> the same [B, 1, Dr] slice is replicated across all heads), implicitly
# casting f32 -> out_dtype on store. Output key matches the oracle contract: k.
_OUT_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def run(inputs: dict) -> dict:
    k_nope = inputs["k_nope"]
    k_rope = inputs["k_rope"]
    out_dtype = str(inputs.get("out_dtype", "bf16"))

    assert k_nope.ndim == 3, "k_nope must be [B, H, Dn]"
    assert k_rope.ndim == 3 and k_rope.shape[1] == 1, "k_rope must be [B, 1, Dr]"
    assert k_nope.shape[0] == k_rope.shape[0], "k_nope/k_rope token count mismatch"

    dtype = _OUT_DTYPES.get(out_dtype)
    if dtype is None:
        raise ValueError(f"unsupported out_dtype {out_dtype!r}")

    n_tokens, head_cnt, nope_dim = k_nope.shape
    rope_dim = k_rope.shape[-1]

    # The frozen kernel passes no last-dimension strides, so it assumes the
    # innermost dim is contiguous. Make the inputs contiguous and allocate the
    # output contiguous in [B, H, Dn + Dr].
    k_nope = k_nope.contiguous()
    k_rope = k_rope.contiguous()
    k = k_nope.new_empty(
        (n_tokens, head_cnt, nope_dim + rope_dim), dtype=dtype
    )

    concat_and_cast_mha_k_triton(k, k_nope, k_rope)
    return {"k": k}
