"""Task L2_08_gumbel_sample_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_gumbel_sample_kernel`
Source path: `vllm/v1/worker/gpu/sample/gumbel.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): b96066c1482edd6b890a7ac17eb77188480d6b081b0054cefae25e54cee15e37
Frozen by: scripts/phase_a_freeze.py
"""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl


@triton.jit
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _temperature_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def tl_rand64(seed, offset, includes_zero: tl.constexpr):
    lo, hi, _, _ = tl.randint4x(seed, offset)
    lo = lo.to(tl.uint32, bitcast=True).to(tl.uint64)
    hi = hi.to(tl.uint32, bitcast=True).to(tl.uint64)
    r = (hi << 32) | lo

    # 1 / 2**64
    scale = 5.421010862427522170037e-20
    u = r.to(tl.float64) * scale
    if not includes_zero:
        u = tl.maximum(u, 2.2250738585072014e-308)  # float64 tiny
    return u


@triton.jit
def gumbel_block_argmax(
    logits,
    block,
    mask,
    token_idx,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    vocab_size,
    APPLY_TEMPERATURE: tl.constexpr,
):
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp != 0.0 and APPLY_TEMPERATURE:
        # Apply temperature.
        # NOTE(woosuk): Match the behavior of _temperature_kernel.
        # E.g., if the kernel uses tl.div_rn, we should use tl.div_rn here too.
        logits = logits / temp

    if processed_logits_ptr is not None:
        # Store the temperature-applied logits.
        if processed_logits_col_ptr is not None:
            col = tl.load(processed_logits_col_ptr)
        else:
            col = 0
        tl.store(
            processed_logits_ptr
            + req_state_idx * processed_logits_stride
            + col * vocab_size
            + block,
            logits,
            mask=mask,
        )

    logits = logits.to(tl.float64)
    if temp != 0.0:
        # Calculate the seed for gumbel noise.
        seed = tl.load(seeds_ptr + req_state_idx)
        pos = tl.load(pos_ptr + token_idx)
        gumbel_seed = tl.randint(seed, pos)

        # tl.rand returns fp32, so build a true fp64 uniform from 64 random
        # bits before applying the double-log transform.
        u = tl_rand64(gumbel_seed, block, includes_zero=False)
        gumbel_noise = -tl.log(-tl.log(u))

        # Apply gumbel noise.
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    value, idx = tl.max(logits, axis=0, return_indices=True)
    return value, idx


@triton.jit
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    value, idx = gumbel_block_argmax(
        logits,
        block,
        mask,
        token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seeds_ptr,
        pos_ptr,
        processed_logits_ptr,
        processed_logits_stride,
        processed_logits_col_ptr,
        vocab_size,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    output_processed_logits: torch.Tensor | None = None,
    output_processed_logits_col: torch.Tensor | None = None,
) -> torch.Tensor:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=torch.float64)
    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        output_processed_logits,
        output_processed_logits.stride(0) if output_processed_logits is not None else 0,
        output_processed_logits_col,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
    )
    # NOTE(woosuk): Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled

import torch
import triton


# Authored jit-only launcher: drives the in-module @triton.jit
# `_gumbel_sample_kernel` exactly as the upstream host wrapper
# `gumbel_sample` does (vLLM, commit 18f6bf5,
# vllm/v1/worker/gpu/sample/gumbel.py).
#
# Why a launcher rather than the host wrapper: the module imports
# `from vllm.triton_utils import tl, triton`; on a host with no active
# Triton driver the real upstream re-export hands out a TritonPlaceholder
# (no-op .jit), so calling `gumbel_sample` would not run the kernel. The
# bench injects the real-triton shim into `vllm.triton_utils` and this
# launcher drives the frozen @triton.jit `_gumbel_sample_kernel` directly,
# then reproduces the wrapper's pure-PyTorch post-reduction (argmax over the
# per-block partials + gather). The @triton.jit kernel IS the unit evaluated;
# the host-side block reduction is the same code the upstream wrapper runs.
#
# Kernel contract (matches the re-authored oracle_pytorch). For each token
# `t` and vocab block `b` the @triton.jit kernel computes, with
#   req = expanded_idx_mapping[t]; temp = temperature[req]:
#     logits_f32 = logits[t, block].to(f32)           # block = b*BS + arange
#     if temp != 0 and APPLY_TEMPERATURE: logits /= temp
#     logits_f64 = logits.to(f64)
#     if temp != 0:                                    # gumbel-max sampling
#         gseed = philox.randint(seed[req], pos[t])
#         u     = fp64 uniform from 64 philox bits at offset `block`
#         logits_f64 += -log(-log(u))                  # gumbel noise
#     value, idx = masked_argmax(logits_f64)           # tie -> first index
#     local_argmax[t, b] = b*BS + idx; local_max[t, b] = value
# then the wrapper reduces: sampled[t] = local_argmax[t, argmax_b local_max[t]].
#
# Load-bearing inputs: `temperature` selects greedy (==0) vs gumbel (>0) and
# scales the logits; `seed`/`pos` seed the per-(token,vocab) philox noise;
# `expanded_idx_mapping` routes each token to its request's temp+seed. A
# kernel ignoring any of these returns a different `sampled` (fault-injection
# proven). Output key: "sampled" (the argmax token id per token).
def run(inputs: dict) -> dict:
    logits = inputs["logits"]
    expanded_idx_mapping = inputs["expanded_idx_mapping"]
    temperature = inputs["temperature"]
    seed = inputs["seed"]
    pos = inputs["pos"]
    apply_temperature = bool(inputs.get("apply_temperature", True))

    assert logits.ndim == 2, "logits must be (num_tokens, vocab_size)"
    num_tokens, vocab_size = logits.shape
    assert expanded_idx_mapping.numel() == num_tokens, (
        "expanded_idx_mapping must have one request index per token"
    )
    assert pos.numel() == num_tokens, "pos must have one position per token"

    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=torch.float64)

    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        None,  # output_processed_logits (not part of the sampling contract)
        0,     # processed_logits_stride
        None,  # output_processed_logits_col
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
    )

    # Upstream wrapper post-reduction (pure PyTorch, int64 for indexing):
    # pick the winning block per token, then gather its token id.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return {"sampled": sampled}
