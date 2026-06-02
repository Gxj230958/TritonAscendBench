"""Oracle reference for L2_39_scaled_mm (scaled_mm_kernel).

Tier: formula_exact_manual
Tolerance class: int8_w8a8
Category: quantization

Independent eager reference for vLLM compressed-tensors W8A8 GEMM. The kernel
under test (frozen kernel.py) computes a row/column dequantized int8 matmul:

  acc[m, n]  = sum_k A[m, k] * B[k, n]          (int8 x int8 -> int32, exact)
  C[m, n]    = scale_a[m] * scale_b[n] * acc    (cast to out_dtype)
  C[m, n]   += bias[n]                          (added after the cast)

A is [M, K] int8, B is [K, N] int8, scale_a is [M, 1] and scale_b is [N, 1]
(per-token / per-channel), bias is [N] in out_dtype. out_dtype is bfloat16
here (W8A8 epilogue output); bf16's fp32 exponent range keeps the dequantized
product in range without overflow. No upstream code is copied.
"""
from __future__ import annotations

import torch

OUT_DTYPE = torch.bfloat16


def _scaled_mm(
    A: torch.Tensor, B: torch.Tensor,
    scale_a: torch.Tensor, scale_b: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    M, K = A.shape
    _, N = B.shape
    # Exact int32 accumulation (mirrors the kernel's int8 tl.dot -> int32).
    acc = (A.to(torch.int32) @ B.to(torch.int32)).to(torch.float32)  # [M, N]
    scaled = scale_a.reshape(M, 1).to(torch.float32) * acc
    scaled = scale_b.reshape(1, N).to(torch.float32) * scaled
    # Cast to out_dtype first, then add bias in out_dtype -- the kernel adds
    # bias only after `c = accumulator.to(out_dtype)`.
    c = scaled.to(OUT_DTYPE)
    c = c + bias.reshape(1, N).to(OUT_DTYPE)
    return c


def run(inputs: dict) -> dict:
    C = _scaled_mm(
        inputs["A"], inputs["B"],
        inputs["scale_a"], inputs["scale_b"], inputs["bias"],
    )
    return {"C": C}
