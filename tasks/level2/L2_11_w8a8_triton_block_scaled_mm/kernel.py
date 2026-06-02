"""Task L2_11_w8a8_triton_block_scaled_mm — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_w8a8_triton_block_scaled_mm`
Source path: `vllm/model_executor/layers/quantization/utils/fp8_utils.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): f2a3db8cc995bb2cbb38111ee1f84991f97f002cc04beb61572cb171f57ca02a
Frozen by: scripts/phase_a_freeze.py
"""
import functools
import json
import os
from collections.abc import Callable, Sequence
from typing import Any
import torch
import tasks._frozen_shim.envs as envs  # Phase A': vllm.envs allowlist shim (B-inert-shim)
from tasks._frozen_shim import ops as ops  # Phase A': off-path custom-ops shim (B-inert-shim)
from tasks._frozen_shim import init_logger  # Phase A': inert no-op sink (B-inert-shim)
from tasks._frozen_shim import get_fp8_min_max  # Phase A': quant symbols, symbol-aware (B-inert-shim)
from tasks._frozen_shim import all_close_1d, per_tensor_dequantize  # Phase A': off-path w8a8 weight utils (B-inert-shim)
from tasks._frozen_shim import BlockQuantScaleParameter, ChannelQuantScaleParameter, PerTensorScaleParameter  # Phase A': off-path weight-loader params (B-inert-shim)
from tasks._frozen_shim import set_weight_attrs  # Phase A': off-path weight-attr setter (B-inert-shim)
from tasks._frozen_shim import current_platform  # Phase A': platform stub (B-inert-shim)
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
from tasks._frozen_shim import get_tma_aligned_size, is_deep_gemm_e8m0_used, transform_sf_into_required_layout  # Phase A': off-path deep_gemm accel surface (B-inert-shim)
from tasks._frozen_shim import direct_register_custom_op  # Phase A': pure predicate (B-inert-shim)

def _triton_per_token_group_quant_fp8_impl(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    return per_token_group_quant_fp8(x, group_size, column_major_scales=False, use_ue8m0=False)

def _triton_per_token_group_quant_fp8_fake(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    x_fp8 = torch.empty((M, N), dtype=current_platform.fp8_dtype(), device=x.device)
    out_bs = torch.empty((M, (N + group_size - 1) // group_size), dtype=torch.float32, device=x.device)
    return (x_fp8, out_bs)
direct_register_custom_op('triton_per_token_group_quant_fp8', _triton_per_token_group_quant_fp8_impl, fake_impl=_triton_per_token_group_quant_fp8_fake)

@triton.jit
def _per_token_group_quant_fp8(y_ptr, y_q_ptr, y_s_ptr, group_size, y_num_columns, y_row_stride, eps, fp8_min: tl.constexpr, fp8_max: tl.constexpr, use_ue8m0: tl.constexpr, BLOCK: tl.constexpr):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row
    y_ptr_offset = row.to(tl.int64) * y_row_stride + row_g_id.to(tl.int64) * group_size
    y_ptr += y_ptr_offset
    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset
    y_s_ptr += g_id
    cols = tl.arange(0, BLOCK)
    mask = cols < group_size
    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    scale_raw = _absmax * (1.0 / fp8_max)
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)

@triton.jit
def _per_token_group_quant_fp8_colmajor(y_ptr, y_q_ptr, y_s_ptr, group_size, y_num_columns, y_row_stride, y_s_col_stride, eps, fp8_min: tl.constexpr, fp8_max: tl.constexpr, use_ue8m0: tl.constexpr, BLOCK: tl.constexpr):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row
    y_ptr_offset = row.to(tl.int64) * y_row_stride + row_g_id.to(tl.int64) * group_size
    y_ptr += y_ptr_offset
    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset
    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    y_s_ptr_offset = scale_col.to(tl.int64) * y_s_col_stride + scale_row.to(tl.int64)
    y_s_ptr += y_s_ptr_offset
    cols = tl.arange(0, BLOCK)
    mask = cols < group_size
    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    scale_raw = _absmax * (1.0 / fp8_max)
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)

def per_token_group_quant_fp8(x: torch.Tensor, group_size: int, eps: float=1e-10, dtype: torch.dtype | None=None, column_major_scales: bool=False, tma_aligned_scales: bool=False, out_q: torch.Tensor | None=None, use_ue8m0: bool | None=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.
    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dtype of output tensor. Note that only `torch.float8_e4m3fn`
        is supported for now.
        column_major_scales: Outputs scales in column major.
        tma_aligned_scales: Outputs scales in TMA-aligned layout.
        out_q: Optional output tensor. If not provided, function will create.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
        scaling factor.
    """
    if use_ue8m0 is None:
        use_ue8m0 = is_deep_gemm_e8m0_used()
    dtype = current_platform.fp8_dtype() if dtype is None else dtype
    assert x.shape[-1] % group_size == 0, f'the last dimension of `x` {x.shape[-1]} must be divisible by `group_size` {group_size}'
    assert x.stride(-1) == 1, '`x` groups must be contiguous'
    fp8_min, fp8_max = get_fp8_min_max()
    assert out_q is None or out_q.shape == x.shape
    x_q = out_q
    if x_q is None:
        x_q = torch.empty(x.shape, device=x.device, dtype=dtype)
    if column_major_scales:
        if tma_aligned_scales:
            m = x.shape[-2]
            sf_k = x.shape[-1] // group_size
            tma_aligned_m = get_tma_aligned_size(m, 4)
            shape = x.shape[:-2] + (m, sf_k)
            stride = (1, tma_aligned_m) if x.dim() == 2 else (tma_aligned_m * sf_k, 1, tma_aligned_m)
            x_s = torch.empty_strided(shape, stride, device=x.device, dtype=torch.float32)
        else:
            shape = x.shape[:-2] + (x.shape[-1] // group_size, x.shape[-2])
            x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)
    if current_platform.is_cuda() and x.is_contiguous():
        torch.ops._C.per_token_group_fp8_quant(x, x_q, x_s, group_size, eps, fp8_min, fp8_max, use_ue8m0, column_major_scales, tma_aligned_scales)
        return (x_q, x_s)
    M = x.numel() // group_size
    N = group_size
    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    if column_major_scales:
        _per_token_group_quant_fp8_colmajor[M,](x, x_q, x_s, group_size, x.shape[1], x.stride(0), x_s.stride(1), eps, fp8_min=fp8_min, fp8_max=fp8_max, use_ue8m0=use_ue8m0, BLOCK=BLOCK, num_warps=num_warps, num_stages=num_stages)
    else:
        _per_token_group_quant_fp8[M,](x, x_q, x_s, group_size, x.shape[1], x.stride(0), eps, fp8_min=fp8_min, fp8_max=fp8_max, use_ue8m0=use_ue8m0, BLOCK=BLOCK, num_warps=num_warps, num_stages=num_stages)
    return (x_q, x_s)

@triton.jit
def _w8a8_triton_block_scaled_mm(A, B, C, As, Bs, M, N, K, group_n, group_k, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_As_m, stride_As_k, stride_Bs_k, stride_Bs_n, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and
    store the result in output tensor `C`.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

import torch
import triton


# Authored jit-only launcher: drives the in-module @triton.jit
# `_w8a8_triton_block_scaled_mm` (vLLM 18f6bf5,
# vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py)
# exactly as its host wrapper `w8a8_triton_block_scaled_mm` does on the default
# (un-tuned) config path (kernel.py lines 834-928).
#
# Why a launcher rather than the host wrapper: the wrapper unconditionally calls
# `get_w8a8_block_fp8_configs(N, K, block_n, block_k)`, whose first statement is
# `current_platform.get_device_name()` -- a platform query that only builds a
# per-device autotune-config JSON filename. No such JSON ships with the task, so
# the lookup always returns None and the wrapper falls to the DEFAULT config
# regardless of device name. The device-name probe is therefore a perf-config
# detour with no effect on numeric output, but on the severed bench host it
# leaks the real `vllm.platforms` import (interface.py get_device_name raises
# NotImplementedError). This launcher reproduces the wrapper's DEFAULT-config
# launch body verbatim so the @triton.jit kernel is graded directly, on any
# device, without the platform-gated config detour.
#
# Default config copied verbatim from the wrapper's `else` branch:
#   BLOCK_SIZE_M = 64
#   BLOCK_SIZE_N = block_size[0]   # divisible by block_n (block-wise quant)
#   BLOCK_SIZE_K = block_size[1]   # divisible by block_k (>= 32 for int8 tl.dot)
#   GROUP_SIZE_M = 32 ; num_warps = 4 ; num_stages = 2
# Tile sizes are launch metadata only: the kernel accumulates in fp32 over the
# full K with masked tail loads, so the result is independent of the tiling.
#
# Kernel math (matches the re-authored oracle_pytorch): B is [N, K], the GEMM is
# A @ B.T,
#   C[m, n] = sum_k A[m, k] * B[n, k] * As[m, k // block_k] * Bs[n // block_n, k // block_k]
# accumulated in fp32 and cast once to output_dtype (fp16 default) at the store.
# Output key: "C".
def run(inputs: dict) -> dict:
    A = inputs["A"]
    B = inputs["B"]
    As = inputs["As"]
    Bs = inputs["Bs"]
    block_size = inputs["block_size"]
    output_dtype = inputs.get("output_dtype", torch.float16)

    assert len(block_size) == 2
    block_n, block_k = int(block_size[0]), int(block_size[1])

    # Shape invariants the host wrapper asserts before launch.
    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    M = A.numel() // A.shape[-1]

    assert B.ndim == 2 and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    # Default (un-tuned) config -- verbatim from the wrapper's `else` branch.
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_stages": 2,
    }

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    _w8a8_triton_block_scaled_mm[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
    )

    return {"C": C}
