"""Task L1_13_fused_softcap_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `fused_softcap_kernel`
Source path: `sglang/srt/layers/logits_processor.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): 546412f45e85c187a4f5f687c8131f256ff5266855e9daba0fe3ea74ac9f3741
"""

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Logits processing."""

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl
from torch import nn
from triton.language.extra import libdevice

from ascendbench.frozen.shim import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather  # Phase A': sglang sibling surface (B-inert-shim)
from ascendbench.frozen.shim import envs  # Phase A': sglang env namespace (B-inert-shim)
from ascendbench.frozen.shim import DpPaddingMode, attn_tp_all_gather, attn_tp_all_gather_into_tensor, dp_gather_replicate, dp_scatter, get_attention_dp_rank, get_attention_dp_size, get_attention_tp_size, get_dp_device, get_dp_dtype, get_dp_hidden_size  # Phase A': sglang sibling surface (B-inert-shim)
from ascendbench.frozen.shim import InputLogprobsResult, compute_temp_top_p_normalized_logprobs, get_token_ids_logprobs_chunk, get_token_ids_logprobs_prefill, get_top_logprobs_chunk, get_top_logprobs_prefill  # Phase A': sglang sibling surface (B-inert-shim)
from ascendbench.frozen.shim import VocabParallelEmbedding  # Phase A': sglang sibling surface (B-inert-shim)
from ascendbench.frozen.shim import CaptureHiddenMode, ForwardBatch, ForwardMode  # Phase A': sglang sibling surface (B-inert-shim)
from ascendbench.frozen.shim import get_global_server_args  # Phase A': sglang sibling surface (B-inert-shim)
from ascendbench.frozen.shim import is_npu, use_intel_amx_backend  # Phase A': sglang sibling surface (B-inert-shim)










@triton.jit
def fused_softcap_kernel(
    full_logits_ptr,
    softcapping_value,
    ncols,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < ncols

    # Load values
    row_ptr = full_logits_ptr + row * row_stride
    x = tl.load(row_ptr + offsets, mask=mask)

    # Perform operations in-place
    x = x / softcapping_value
    x = libdevice.tanh(x)
    x = x * softcapping_value

    # Store result
    tl.store(row_ptr + offsets, x, mask=mask)


def fused_softcap(full_logits, final_logit_softcapping):
    if full_logits.is_contiguous():
        nrows, ncols = 1, full_logits.numel()
        row_stride = ncols
    else:
        assert full_logits.ndim == 2, "non-contiguous softcap requires 2D tensor"
        assert (
            full_logits.stride(1) == 1
        ), "non-contiguous softcap requires contiguous columns"
        nrows, ncols = full_logits.shape
        row_stride = full_logits.stride(0)

    BLOCK_SIZE = 1024
    grid = ((ncols + BLOCK_SIZE - 1) // BLOCK_SIZE, nrows)

    fused_softcap_kernel[grid](
        full_logits_ptr=full_logits,
        softcapping_value=final_logit_softcapping,
        ncols=ncols,
        row_stride=row_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return full_logits


# ── Phase A' synthesised standalone entrypoint (HOLE B) ─────────────
# The staged candidate MUST expose run(inputs)->dict so S2 can compare
# it against the UNMODIFIED upstream (which exposes the positional
# entrypoint 'fused_softcap', NOT run). This wrapper is a PURE dict->positional
# argument shuffle of the EXISTING upstream entrypoint -- it never
# reimplements numeric logic (changing values = an A' failure). The
# original side runs the same upstream callable via the identical
# signature adapter in _pa_subproc_runner._signature_adapter, so the
# two manifests are structurally comparable. Promotion of this file is
# gated on a genuine S2-gpu-authoritative PASS (never fabricated).
def run(inputs: dict) -> dict:  # noqa: D401
    """Standalone S2 entrypoint -- maps `inputs` onto the upstream
    `fused_softcap` signature STRICTLY BY NAME (never a silent
    skip+positional-shift; a required no-default param with no value is
    a hard BLOCKED(shapes-incomplete), NOT a mis-call), returns a dict
    (mutated args surfaced when the upstream entrypoint mutates in
    place and returns None). This mirrors
    _pa_subproc_runner._signature_adapter EXACTLY so the original and
    severed sides bind identically."""
    import inspect

    import torch as _torch

    _MARK = "BLOCKED(shapes-incomplete"
    _fn = fused_softcap
    _PNM = {}  # meta.interface.param_name_map (wrapper param -> task key)
    try:
        _sig = inspect.signature(_fn)
    except (TypeError, ValueError):
        _sig = None
    if _sig is None:
        _ret = _fn(inputs)
        _bound = []
    else:
        _kw = {}
        _bound = []
        for _n, _p in _sig.parameters.items():
            if _p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if _n in inputs:
                _kw[_n] = inputs[_n]
                _bound.append(_n)
            elif _n in _PNM and _PNM[_n] in inputs:
                # meta.interface.param_name_map (baked): same task tensor
                # under a different key. _bound records the TASK KEY so the
                # in-place output reconstruction indexes inputs correctly.
                _src = _PNM[_n]
                _kw[_n] = inputs[_src]
                _bound.append(_src)
            elif _p.default is not inspect.Parameter.empty:
                continue
            else:
                raise RuntimeError(
                    _MARK + ": missing param " + repr(_n)
                    + " for entrypoint " + repr('fused_softcap')
                    + " -- shapes.yaml / build_inputs provides no value "
                    "and the parameter has no default. Refusing to "
                    "silently skip+shift positional args. Fix: add the "
                    "param to the task's shapes.yaml inputs (or "
                    "meta.interface.param_name_map)."
                )
        _ret = _fn(**_kw)
    if _ret is None:
        _out = {
            p: inputs[p]
            for p in _bound
            if _torch.is_tensor(inputs.get(p))
        }
        return _out or {"_inplace": None}
    if isinstance(_ret, dict):
        return _ret
    if isinstance(_ret, (list, tuple)):
        return {f"_out{i}": v for i, v in enumerate(_ret)}
    return {"_out0": _ret}
