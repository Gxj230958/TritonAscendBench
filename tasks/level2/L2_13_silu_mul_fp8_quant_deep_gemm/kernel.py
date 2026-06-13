"""Task L2_13_silu_mul_fp8_quant_deep_gemm — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_silu_mul_fp8_quant_deep_gemm`
Source path: `vllm/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): aeb1706204c6c43622c6e60476d632ff1d1759415e65fdac53f49fd59870a42d
"""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import ascendbench.frozen.shim.modular_kernel as mk  # Phase A': off-path modular_kernel annotation shim (B-inert-shim)
from ascendbench.frozen.shim import get_forward_context, is_forward_context_available  # Phase A': off-path forward-context (B-inert-shim)
from ascendbench.frozen.shim import init_logger  # Phase A': inert no-op sink (B-inert-shim)
from ascendbench.frozen.shim import MoEActivation  # Phase A': off-path activation enum (B-inert-shim)
from ascendbench.frozen.shim import FusedMoEConfig, FusedMoEParallelConfig, FusedMoEQuantConfig  # Phase A': off-path fused_moe config (B-inert-shim)
from ascendbench.frozen.shim import TopKWeightAndReduceDelegate  # Phase A': off-path reduce helper (B-inert-shim)
from ascendbench.frozen.shim import _resize_cache  # Phase A': off-path workspace reshape (B-inert-shim)
from ascendbench.frozen.shim import QuantKey, get_fp8_min_max, kFp8Dynamic128Sym, kFp8Static128BlockSym  # Phase A': quant symbols, symbol-aware (B-inert-shim)
from ascendbench.frozen.shim import current_platform  # Phase A': platform stub (B-inert-shim)
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
from ascendbench.frozen.shim import DeepGemmQuantScaleFMT, fp8_m_grouped_gemm_nt_masked, get_mk_alignment_for_contiguous_layout, is_deep_gemm_e8m0_used, is_deep_gemm_supported  # Phase A': off-path deep_gemm accel surface (B-inert-shim)
from ascendbench.frozen.shim import cdiv, round_up  # Phase A': pure-math shim (B-inert-shim)





@triton.jit
def _silu_mul_fp8_quant_deep_gemm(
    # Pointers ------------------------------------------------------------
    input_ptr,  # 16-bit activations (E, T, 2*H)
    y_q_ptr,  # fp8 quantized activations (E, T, H)
    y_s_ptr,  # 16-bit scales (E, T, G)
    counts_ptr,  # int32 num tokens per expert (E)
    # Sizes ---------------------------------------------------------------
    H: tl.constexpr,  # hidden dimension (per output)
    GROUP_SIZE: tl.constexpr,  # elements per group (usually 128)
    # Strides for input (elements) ---------------------------------------
    stride_i_e,
    stride_i_t,
    stride_i_h,
    # Strides for y_q (elements) -----------------------------------------
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,
    # Strides for y_s (elements) -----------------------------------------
    stride_ys_e,
    stride_ys_t,
    stride_ys_g,
    # Stride for counts (elements)
    stride_counts_e,
    # Numeric params ------------------------------------------------------
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    ceil_ue8m0: tl.constexpr,
    # Meta ---------------------------------------------------------------
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    G = H // GROUP_SIZE

    # map program id -> (e, g)
    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    e = e.to(tl.int64)
    g = g.to(tl.int64)

    # number of valid tokens for this expert
    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int64)

    cols = tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < BLOCK

    base_input_offset = e * stride_i_e + g * GROUP_SIZE * stride_i_h
    base_gate_offset = base_input_offset + cols * stride_i_h
    base_up_offset = base_input_offset + H * stride_i_h + cols * stride_i_h
    base_yq_offset = e * stride_yq_e + g * GROUP_SIZE * stride_yq_h + cols * stride_yq_h
    base_ys_offset = e * stride_ys_e + g * stride_ys_g

    for t in tl.range(0, n_tokens, num_stages=NUM_STAGES):
        gate = tl.load(
            input_ptr + base_gate_offset + t * stride_i_t, mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(input_ptr + base_up_offset + t * stride_i_t, mask=mask, other=0.0)

        gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
        y = gate * up

        # Use multiply-by-reciprocal to match PyTorch's tensor/scalar
        # division precision (Triton GPU fast-division for constexpr
        # divisors can introduce 1-ULP error).
        y_s = tl.maximum(tl.max(tl.abs(y)), eps) * (1.0 / fp8_max)
        if ceil_ue8m0:
            y_s = tl.exp2(tl.ceil(tl.log2(y_s)))

        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + base_yq_offset + t * stride_yq_t, y_q, mask=mask)
        tl.store(y_s_ptr + base_ys_offset + t * stride_ys_t, y_s)





import torch
import triton


# Authored jit-only launcher: drives the in-module @triton.jit
# `_silu_mul_fp8_quant_deep_gemm` (vLLM 18f6bf5,
# vllm/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py)
# exactly as its host wrapper `persistent_masked_m_silu_mul_quant` does on
# the Triton-fallback path (kernel.py lines 235-278), for the FLOAT32 scale
# format (ceil_ue8m0=False) that the oracle covers.
#
# Why a launcher rather than the host wrapper: the wrapper's CUDA branch
# calls `torch.ops._C.persistent_masked_m_silu_mul_quant` (the C++ kernel),
# NOT the @triton.jit kernel -- so invoking the wrapper on CUDA would grade
# the C++ op, not the Triton unit. The wrapper only reaches the Triton
# kernel on its ROCm/non-CUDA fallback, gated by `current_platform`/device
# capability probes that are unvendorable here. This launcher reproduces the
# fallback body verbatim so the @triton.jit kernel is always the unit
# evaluated, on any CUDA device.
#
# Upstream fallback call site reproduced here (FLOAT32 scale format):
#   * y [E, T, 2*H] contiguous f32 ; tokens_per_expert [E] (per-expert
#     valid token count) cast to int32
#   * H = (2*H) // 2 ; group_size = 128 ; G = cdiv(H, group_size)
#   * y_q     = empty((E, T, H), fp8_e4m3fn)            contiguous
#   * y_s     = empty_strided((E, T, G), (T*G, 1, T), float32)  -- the
#     DeepGemmQuantScaleFMT.FLOAT32 layout from scales_shape_stride_dtype
#   * grid = (E * G,) ; a kernel-internal loop walks the token dim
#   * BLOCK = group_size ; NUM_STAGES = 4 ; num_warps = 1
#   * eps = 1e-10 ; fp8_max = finfo(e4m3fn).max (448.0) ; fp8_min = -fp8_max
#   * ceil_ue8m0 = False (FLOAT32 format)
#   * strides passed by *y.stride(), *y_q.stride(), then the y_s strides
#     (T*G, 1, T) and tokens_per_expert.stride()[0]
#
# Kernel math (matches the re-authored oracle_pytorch): per (expert e, hidden
# group g) and each valid token t < tokens_per_expert[e],
#   gate    = silu(y[e, t, g_cols])             # f32
#   y       = gate * up   (up = y[e, t, H + g_cols])
#   y_s     = max(max(|y|), 1e-10) * (1 / fp8_max)   # 1e-10 floors the AMAX
#   y_q     = clamp(y / y_s, fp8_min, fp8_max) -> fp8_e4m3fn
# Tokens t >= tokens_per_expert[e] are never written by the kernel; we
# zero-init both outputs so the invalid rows are deterministic 0 and match
# the oracle's masked invalid rows under the harness full-tensor compare.
# Valid rows are fully overwritten, so zero-init does not change them.
# Output keys: "output" (fp8 quant), "scale" (fp32 group scale).

import torch
import triton


# Authored jit-only launcher: drives the in-module @triton.jit
# `_silu_mul_fp8_quant_deep_gemm` (vLLM 18f6bf5,
# vllm/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py)
# exactly as its host wrapper `persistent_masked_m_silu_mul_quant` does on
# the Triton-fallback path (kernel.py lines 235-278), for the FLOAT32 scale
# format (ceil_ue8m0=False) that the oracle covers.
#
# Why a launcher rather than the host wrapper: the wrapper's CUDA branch
# calls `torch.ops._C.persistent_masked_m_silu_mul_quant` (the C++ kernel),
# NOT the @triton.jit kernel -- so invoking the wrapper on CUDA would grade
# the C++ op, not the Triton unit. The wrapper only reaches the Triton
# kernel on its ROCm/non-CUDA fallback, gated by `current_platform`/device
# capability probes that are unvendorable here. This launcher reproduces the
# fallback body verbatim so the @triton.jit kernel is always the unit
# evaluated, on any CUDA device.
#
# Upstream fallback call site reproduced here (FLOAT32 scale format):
#   * y [E, T, 2*H] contiguous f32 ; tokens_per_expert [E] (per-expert
#     valid token count) cast to int32
#   * H = (2*H) // 2 ; group_size = 128 ; G = cdiv(H, group_size)
#   * y_q     = empty((E, T, H), fp8_e4m3fn)            contiguous
#   * y_s     = empty_strided((E, T, G), (T*G, 1, T), float32)  -- the
#     DeepGemmQuantScaleFMT.FLOAT32 layout from scales_shape_stride_dtype
#   * grid = (E * G,) ; a kernel-internal loop walks the token dim
#   * BLOCK = group_size ; NUM_STAGES = 4 ; num_warps = 1
#   * eps = 1e-10 ; fp8_max = finfo(e4m3fn).max (448.0) ; fp8_min = -fp8_max
#   * ceil_ue8m0 = False (FLOAT32 format)
#   * strides passed by *y.stride(), *y_q.stride(), then the y_s strides
#     (T*G, 1, T) and tokens_per_expert.stride()[0]
#
# Kernel math (matches the re-authored oracle_pytorch): per (expert e, hidden
# group g) and each valid token t < tokens_per_expert[e],
#   gate    = silu(y[e, t, g_cols])             # f32
#   y       = gate * up   (up = y[e, t, H + g_cols])
#   y_s     = max(max(|y|), 1e-10) * (1 / fp8_max)   # 1e-10 floors the AMAX
#   y_q     = clamp(y / y_s, fp8_min, fp8_max) -> fp8_e4m3fn
# Tokens t >= tokens_per_expert[e] are never written by the kernel; we
# zero-init both outputs so the invalid rows are deterministic 0 and match
# the oracle's masked invalid rows under the harness full-tensor compare.
# Valid rows are fully overwritten, so zero-init does not change them.
# Output keys: "output" (fp8 quant), "scale" (fp32 group scale).
def run(inputs: dict) -> dict:
    y = inputs["y"]
    tokens_per_expert = inputs["tokens_per_expert"]
    group_size = int(inputs["group_size"])

    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2
    assert group_size == 128, "kernel requires group_size == 128"
    assert H % group_size == 0, "H must be divisible by group_size"
    assert H % 8 == 0, "H must be divisible by 8"
    G = H // group_size
    assert tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E

    y = y.contiguous()
    tokens_per_expert = tokens_per_expert.to(
        device=y.device, dtype=torch.int32
    ).contiguous()

    fp8_dtype = torch.float8_e4m3fn
    finfo = torch.finfo(fp8_dtype)
    fp8_max = finfo.max
    fp8_min = -fp8_max
    eps = 1e-10

    # Zero-init so tokens never written by the kernel (t >= tokens_per_expert
    # [e]) are deterministic 0, matching the oracle's masked invalid rows.
    y_q = torch.zeros((E, T, H), dtype=fp8_dtype, device=y.device)

    # DeepGemmQuantScaleFMT.FLOAT32 scale layout (scales_shape_stride_dtype):
    # shape (E, T, G), strides (T*G, 1, T), float32. empty_strided then
    # zero_ to preserve the exact (non-contiguous) layout the kernel writes.
    ys_strides = (T * G, 1, T)
    y_s = torch.empty_strided(
        (E, T, G), ys_strides, dtype=torch.float32, device=y.device
    )
    y_s.zero_()

    # Static grid over experts and H-groups; the kernel loops the token dim.
    grid = (E * G,)
    stride_i_e, stride_i_t, stride_i_h = y.stride()
    stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()
    stride_cnt_e = tokens_per_expert.stride()[0]

    _silu_mul_fp8_quant_deep_gemm[grid](
        y,
        y_q,
        y_s,
        tokens_per_expert,
        H,
        group_size,
        stride_i_e,
        stride_i_t,
        stride_i_h,
        stride_yq_e,
        stride_yq_t,
        stride_yq_h,
        ys_strides[0],
        ys_strides[1],
        ys_strides[2],
        stride_cnt_e,
        eps,
        fp8_min,
        fp8_max,
        False,  # ceil_ue8m0 (FLOAT32 scale format)
        BLOCK=group_size,
        NUM_STAGES=4,
        num_warps=1,
    )

    return {"output": y_q, "scale": y_s}
