"""Task L1_15_act_and_mul_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `act_and_mul_kernel`
Source path: `sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): a62c6efe1b70521c9b39ce0358f094c933a4da5b27b8356badd19ff24be97640
Frozen by: scripts/phase_a_freeze.py
"""
from __future__ import annotations
import functools
from collections import OrderedDict
from typing import Any, Dict, List, Optional
import torch
import triton
import triton.language as tl
from tasks._frozen_shim import is_batch_invariant_mode_enabled  # Phase A': batch-invariant gate (B-inert-shim)
from tasks._frozen_shim import get_moe_padding_size  # Phase A': MoE padding query (B-inert-shim)
from tasks._frozen_shim import per_token_group_quant_fp8, scaled_fp8_quant, sglang_per_token_group_quant_fp8  # Phase A': fp8 quant helpers -> shim (B-inert-shim)
from tasks._frozen_shim import per_token_group_quant_int8, per_token_quant_int8, sglang_per_token_group_quant_int8  # Phase A': int8 quant -> loud-fail stub (off-path)
from tasks._frozen_shim import cpu_has_amx_support, get_bool_env_var, is_cpu, is_cuda, is_hip, is_sm90_supported  # Phase A': platform/math predicate (B-inert-shim)
try:
    from triton.tools.tensor_descriptor import TensorDescriptor
    _support_tensor_descriptor = True
except:
    _support_tensor_descriptor = False
_is_hip = is_hip()
_is_cuda = is_cuda()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
if _is_cuda:
    pass
elif _is_cpu and _is_cpu_amx_available:
    pass
elif _is_hip:
    pass

@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def _apply_activation(x, ACTIVATION_TYPE: tl.constexpr):
    """
    Apply activation function based on compile-time constant.

    Args:
        x: Input tensor (converted to float32 inside)
        ACTIVATION_TYPE: Compile-time constant string ("silu" or "gelu")

    Returns:
        Activated output in the same dtype as input
    """
    x = x.to(tl.float32)
    if ACTIVATION_TYPE == 'silu':
        return x * tl.sigmoid(x)
    elif ACTIVATION_TYPE == 'gelu':
        kAlpha = 0.7978845608028654
        return 0.5 * x * (1 + tanh(kAlpha * (x + 0.044715 * x * x * x)))
    else:
        raise ValueError(f'Unsupported activation: {ACTIVATION_TYPE}')

@triton.jit
def act_and_mul_kernel(gateup_output, down_input, hidden_size, expert_ids_ptr, expert_step: tl.constexpr, BLOCK_SIZE: tl.constexpr, ACTIVATION_TYPE: tl.constexpr, SWIGLU_LIMIT: tl.constexpr=0.0, HAS_SWIGLU_LIMIT: tl.constexpr=False):
    """
    Unified activation and multiply kernel that handles both sorted and unsorted routing,
    and both SiLU and GELU activations using compile-time constants.
    """
    InDtype = gateup_output.dtype.element_ty
    OutDtype = down_input.dtype.element_ty
    half_hidden_size = hidden_size // 2
    pid = tl.program_id(0)
    expert_id = tl.load(expert_ids_ptr + pid // expert_step)
    if expert_id == -1:
        return
    gateup_output_ptr = gateup_output + pid * hidden_size
    down_input_ptr = down_input + pid * half_hidden_size
    gate_output_ptr = gateup_output_ptr
    up_output_ptr = gateup_output_ptr + half_hidden_size
    for start_offset in tl.range(0, half_hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < half_hidden_size
        gate_output = tl.load(gate_output_ptr + offset, mask=mask)
        up_output = tl.load(up_output_ptr + offset, mask=mask)
        if HAS_SWIGLU_LIMIT:
            gate_output = tl.minimum(gate_output, SWIGLU_LIMIT)
            up_output = tl.maximum(tl.minimum(up_output, SWIGLU_LIMIT), -SWIGLU_LIMIT)
        gate_output_activated = _apply_activation(gate_output, ACTIVATION_TYPE)
        gate_output_activated = gate_output_activated.to(InDtype)
        act_mul_output = gate_output_activated * up_output
        act_mul_output = act_mul_output.to(OutDtype)
        tl.store(down_input_ptr + offset, act_mul_output, mask=mask)

import torch


# Authored jit-only launcher: drives the in-module @triton.jit
# `act_and_mul_kernel` exactly as the upstream host wrapper
# `act_and_mul_triton` does for the UNSORTED routing path (SGLang, commit
# 409d350, sglang/srt/layers/moe/.../fused_moe_triton_kernels.py).
#
# Why a launcher rather than the host wrapper: the module imports
# `sglang.srt.utils` / `sglang.srt.layers.moe.utils` / the fp8+int8 quant
# kernels -- an unvendorable closure pulled in only by the dead sibling
# `invoke_fused_moe_kernel` / `fused_moe_kernel`. None of that is needed to
# drive THIS kernel; the wrapper body for `act_and_mul_triton` is pure
# grid/scalar setup. This launcher reproduces it verbatim for the
# `down_moe_use_tma=False` (unsorted) path so the @triton.jit kernel is the
# unit evaluated. (Same pattern as the passed sibling
# L2_37_silu_and_mul_post_quant.)
#
# Upstream `act_and_mul_triton(gateup_output, down_input, config, topk_ids,
# ..., down_moe_use_tma=False, activation="silu", swiglu_limit=None)` call site
# reproduced here:
#   * grid          = (down_input.shape[0],)                  # one program/row
#   * hidden_size   = gateup_output.shape[1]                  # == 2*half
#   * expert_ids    = topk_ids.view(-1)                       # unsorted layout
#   * expert_step   = 1                                       # unsorted layout
#   * BLOCK_SIZE=512, ACTIVATION_TYPE=activation,
#     SWIGLU_LIMIT=0.0, HAS_SWIGLU_LIMIT=False (no swiglu clamp -- oracle scope)
#
# Kernel math (matches the re-authored oracle_pytorch): per row pid,
#   expert_id = topk_ids[pid]
#   if expert_id == -1: return            # dropped token -> row NOT written
#   gate = gateup_output[pid, :half]; up = gateup_output[pid, half:]
#   down_input[pid] = act(gate).to(in_dtype) * up
# topk_ids == -1 is the load-bearing routing mask: those rows are skipped. We
# zero-init down_input so the skipped rows are deterministic 0 and match the
# oracle's masked (zeroed) rows under the harness full-tensor compare; the
# valid rows are fully overwritten, so zero-init does not change them.
# Output key: "output" (the down-projection input the kernel writes).
def run(inputs: dict) -> dict:
    gateup_output = inputs["gateup_output"].contiguous()
    # topk_ids is the per-token expert assignment; upstream forwards it via
    # `topk_ids.view(-1)` preserving dtype. Keep the input dtype (int64),
    # only flatten + contiguous as the wrapper does.
    topk_ids = inputs["topk_ids"].view(-1).contiguous()
    activation = str(inputs.get("act", "silu"))

    assert gateup_output.ndim == 2, "gateup_output must be (num_tokens, 2*half)"
    num_tokens, hidden_size = gateup_output.shape
    assert hidden_size % 2 == 0, "hidden_size must be even (gate || up)"
    half_hidden_size = hidden_size // 2
    assert topk_ids.numel() == num_tokens, "topk_ids must have one id per token"

    # Zero-init so rows the kernel skips (topk_ids == -1) stay 0, matching the
    # oracle's masked rows. Output dtype follows the input dtype (the kernel
    # casts the activated gate to InDtype then stores as OutDtype == InDtype).
    down_input = torch.zeros(
        (num_tokens, half_hidden_size),
        dtype=gateup_output.dtype,
        device=gateup_output.device,
    )

    grid = (num_tokens,)
    expert_step = 1  # unsorted routing: expert id looked up per row (pid // 1)
    act_and_mul_kernel[grid](
        gateup_output,
        down_input,
        hidden_size,
        topk_ids,
        expert_step,
        BLOCK_SIZE=512,
        ACTIVATION_TYPE=activation,
        SWIGLU_LIMIT=0.0,
        HAS_SWIGLU_LIMIT=False,
    )

    return {"output": down_input}
