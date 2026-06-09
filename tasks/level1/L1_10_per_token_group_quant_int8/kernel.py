"""Task L1_10_per_token_group_quant_int8 — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_per_token_group_quant_int8`
Source path: `vllm/model_executor/layers/quantization/utils/int8_utils.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): 59ad2a38c34721814e8a3cd949b1d7f01476a069aac7572664f38f66dbae05be
"""
import functools
import json
import logging
import os
from typing import Any
import torch
from ascendbench.frozen.shim import current_platform  # Phase A': platform stub (B-inert-shim)
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
if current_platform.is_rocm():

    @triton.jit
    def round_int8(x):
        return tl.extra.hip.libdevice.round(x).to(tl.int8)
elif current_platform.is_xpu():

    @triton.jit
    def round_int8(x):
        return tl.extra.intel.libdevice.round(x).to(tl.int8)
else:

    @triton.jit
    def round_int8(x):
        return tl.extra.cuda.libdevice.round(x).to(tl.int8)

@triton.jit
def _per_token_group_quant_int8(y_ptr, y_q_ptr, y_s_ptr, y_stride, N, eps, int8_min, int8_max, BLOCK: tl.constexpr):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.

    This function converts the tensor values into int8 values.
    """
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / int8_max
    y_q = tl.clamp(y / y_s, int8_min, int8_max).to(y_q_ptr.dtype.element_ty)
    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)

import torch


# Authored launcher: drives the in-module @triton.jit
# `_per_token_group_quant_int8` exactly as the upstream non-CUDA host path
# does (per_token_group_quant_int8, vLLM
# vllm/model_executor/layers/quantization/utils/int8_utils.py, commit 18f6bf5).
#
# Why a launcher rather than the host wrapper: the wrapper
# `per_token_group_quant_int8` is oracle-consistent, but its
# `current_platform.is_cuda()` branch dispatches to the C++ op
# `torch.ops._C.per_token_group_quant_int8`, which is a compiled-CUDA-only
# extension (`_OpNamespace '_C' has no attribute ...` outside a CUDA build).
# That C++ path never exercises the triton kernel under test. This launcher
# reproduces the wrapper's triton fallback (lines 237-280) verbatim so the
# @triton.jit kernel is always the unit evaluated, on any platform.
#
# Upstream triton call site reproduced here:
#   * iinfo(int8): int8_max = 127, int8_min = -128
#   * x_q  = empty_like(x, int8)          (per_token_group_quant_int8)
#   * x_s  = empty(x.shape[:-1] + (D // group_size,), float32)
#   * M    = x.numel() // group_size       (one program per contiguous group)
#   * N    = group_size                    (columns scanned per program)
#   * BLOCK = next_power_of_2(N)
#   * y_stride argument = group_size       (groups are contiguous, so the
#       pointer advance between programs is exactly `group_size` elements)
#   * grid = (M,)
#
# Kernel math (matches oracle_pytorch): per group of `group_size` elements,
# _absmax = max(amax(|y|), eps); y_s = _absmax / int8_max;
# y_q = clamp(y / y_s, int8_min, int8_max) -> int8. Output keys: q, scale.
def run(inputs: dict) -> dict:
    x = inputs["x"]
    group_size = int(inputs["group_size"])
    eps = float(inputs.get("eps", 1e-10))

    assert x.shape[-1] % group_size == 0, (
        "the last dimension of `x` must be divisible by `group_size`"
    )
    x = x.contiguous()

    iinfo = torch.iinfo(torch.int8)
    int8_max = iinfo.max
    int8_min = iinfo.min

    x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )

    M = x.numel() // group_size
    N = group_size

    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    _per_token_group_quant_int8[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        N,
        eps,
        int8_min=int8_min,
        int8_max=int8_max,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return {"q": x_q, "scale": x_s}
