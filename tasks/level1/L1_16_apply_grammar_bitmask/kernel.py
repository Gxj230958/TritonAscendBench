"""Task L1_16_apply_grammar_bitmask_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_apply_grammar_bitmask_kernel`
Source path: `vllm/v1/worker/gpu/structured_outputs.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): ab2919f98f4f145936718f09660cf597d46183fcd74f95c72ac1b2c70e5c21c8
"""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
from ascendbench.frozen.shim import cdiv  # Phase A': pure-math shim (B-inert-shim)
from ascendbench.frozen.shim import async_copy_to_gpu  # Phase A': setup-only buffer helpers (B-inert-shim)
from ascendbench.frozen.shim import InputBatch  # Phase A': sibling batch-state stand-in (B-inert-shim)




# Adapted from
# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py
@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    logits_indices_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    bitmask_idx = tl.program_id(0)
    logits_idx = tl.load(logits_indices_ptr + bitmask_idx)

    # Load the bitmask.
    block_id = tl.program_id(1)
    bitmask_offset = (block_id * BLOCK_SIZE) // 32 + tl.arange(0, BLOCK_SIZE // 32)
    packed_bitmask = tl.load(
        bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
        mask=bitmask_offset < bitmask_stride,
    )
    # Unpack the bitmask.
    bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
    bitmask = bitmask.reshape(BLOCK_SIZE)

    # Apply the bitmask to the logits.
    block_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(
        logits_ptr + logits_idx * logits_stride + block_offset,
        -float("inf"),
        mask=bitmask & (block_offset < vocab_size),
    )

# Authored jit-only launcher (U5): drives _apply_grammar_bitmask_kernel from the
# upstream call site (StructuredOutputsWorker.apply_grammar_bitmask). The kernel
# masks logits in place; the oracle's output key is "logits".

# Authored jit-only launcher (U5): drives _apply_grammar_bitmask_kernel from the
# upstream call site (StructuredOutputsWorker.apply_grammar_bitmask). The kernel
# masks logits in place; the oracle's output key is "logits".
def run(inputs: dict) -> dict:
    logits = inputs["logits"]
    logits_indices = inputs["logits_indices"]
    bitmask = inputs["bitmask"]
    num_masks = bitmask.shape[0]
    vocab_size = logits.shape[-1]
    BLOCK_SIZE = 8192
    grid = (num_masks, triton.cdiv(vocab_size, BLOCK_SIZE))
    _apply_grammar_bitmask_kernel[grid](
        logits,
        logits.stride(0),
        logits_indices,
        bitmask,
        bitmask.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return {"logits": logits}
