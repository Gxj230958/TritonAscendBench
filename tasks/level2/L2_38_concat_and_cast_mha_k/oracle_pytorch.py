"""Oracle reference for L2_38_concat_and_cast_mha_k_kernel.

Tier: formula_exact_manual
Tolerance class: elementwise
Category: elementwise

Independent eager PyTorch reference authored from the operation's
documented mathematical definition (no upstream code copy).
"""
from __future__ import annotations

import torch

_OUT_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def _concat_and_cast_mha_k(
    k_nope: torch.Tensor, k_rope: torch.Tensor, out_dtype: str = "bf16",
) -> torch.Tensor:
    """DeepSeek-V3 MLA K assembly.

    k_nope is the per-head nope part ``[B, H, Dn]``; k_rope is the SHARED
    rope part ``[B, 1, Dr]`` that is broadcast across all H heads. The
    assembled cache vector for head ``h`` is the concatenation
    ``[k_nope[:, h, :], k_rope[:, 0, :]]`` along the head-dim axis, cast to
    the cache dtype:

        k[:, h, :Dn] = k_nope[:, h, :]
        k[:, h, Dn:] = k_rope[:, 0, :]   # same rope slice for every head
    """
    dt = _OUT_DTYPES.get(out_dtype, k_nope.dtype)
    n_tokens, head_cnt, nope_dim = k_nope.shape
    rope_dim = k_rope.shape[-1]
    k = k_nope.new_empty((n_tokens, head_cnt, nope_dim + rope_dim), dtype=dt)
    k[..., :nope_dim] = k_nope.to(dt)
    # k_rope is [B, 1, Dr]; the assignment broadcasts it across the H heads.
    k[..., nope_dim:] = k_rope.to(dt)
    return k


def run(inputs: dict) -> dict:
    return {"k": _concat_and_cast_mha_k(
        inputs["k_nope"], inputs["k_rope"],
        str(inputs.get("out_dtype", "bf16")))}
