"""Task L1_35_concat_and_cast_mha_k_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `concat_and_cast_mha_k_kernel`
Source path: `sglang/srt/layers/attention/utils.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): 6bae800d0cb2f1b4a2abb448bec90e2c4664d0b882e8bbcdfbe06a3bc8834cc1
Frozen by: scripts/phase_a_freeze.py
"""
import torch
import triton
import triton.language as tl
from tasks._frozen_shim import is_cuda  # Phase A': platform/math predicate (B-inert-shim)
_is_cuda = is_cuda()
if _is_cuda:
    from tasks._frozen_shim import concat_mla_absorb_q  # Phase A': sibling-only CUDA absorb-concat (B-inert-shim)
from tasks._frozen_shim import is_arch_support_pdl  # Phase A': sibling-only PDL arch gate (B-inert-shim)

@triton.jit
def concat_and_cast_mha_k_kernel(k_ptr, k_nope_ptr, k_rope_ptr, head_cnt: tl.constexpr, k_stride0: tl.constexpr, k_stride1: tl.constexpr, nope_stride0: tl.constexpr, nope_stride1: tl.constexpr, rope_stride0: tl.constexpr, nope_dim: tl.constexpr, rope_dim: tl.constexpr):
    pid_loc = tl.program_id(0)
    head_range = tl.arange(0, head_cnt)
    k_head_ptr = k_ptr + pid_loc * k_stride0 + head_range[:, None] * k_stride1
    nope_offs = tl.arange(0, nope_dim)
    src_nope_ptr = k_nope_ptr + pid_loc * nope_stride0 + head_range[:, None] * nope_stride1 + nope_offs[None, :]
    dst_nope_ptr = k_head_ptr + nope_offs[None, :]
    src_nope = tl.load(src_nope_ptr)
    tl.store(dst_nope_ptr, src_nope)
    rope_offs = tl.arange(0, rope_dim)
    src_rope_ptr = k_rope_ptr + pid_loc * rope_stride0 + rope_offs[None, :]
    dst_rope_ptr = k_head_ptr + nope_dim + rope_offs[None, :]
    src_rope = tl.load(src_rope_ptr)
    tl.store(dst_rope_ptr, src_rope)

def concat_and_cast_mha_k_triton(k: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor):
    assert len(k.shape) == 3 and len(k_nope.shape) == 3 and (len(k_rope.shape) == 3), f'shape should be 3d, but got k.shape={k.shape!r}, k_nope.shape={k_nope.shape!r}, k_rope.shape={k_rope.shape!r}'
    assert k.shape[0] == k_nope.shape[0] and k.shape[0] == k_rope.shape[0], f'invalid shape, got k.shape={k.shape!r}, k_nope.shape={k_nope.shape!r}, k_rope.shape={k_rope.shape!r}'
    assert k.shape[1] == k_nope.shape[1] and 1 == k_rope.shape[1], f'invalid shape, got k.shape={k.shape!r}, k_nope.shape={k_nope.shape!r}, k_rope.shape={k_rope.shape!r}'
    assert k.shape[-1] == k_nope.shape[-1] + k_rope.shape[-1], f'invalid shape, got k.shape={k.shape!r}, k_nope.shape={k_nope.shape!r}, k_rope.shape={k_rope.shape!r}'
    nope_dim = k_nope.shape[-1]
    rope_dim = k_rope.shape[-1]
    grid = (k.shape[0],)
    concat_and_cast_mha_k_kernel[grid](k, k_nope, k_rope, k.shape[1], k.stride(0), k.stride(1), k_nope.stride(0), k_nope.stride(1), k_rope.stride(0), nope_dim, rope_dim)

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
