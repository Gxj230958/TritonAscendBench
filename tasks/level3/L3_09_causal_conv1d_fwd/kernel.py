"""Task L3_09_causal_conv1d_fwd_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_causal_conv1d_fwd_kernel`
Source path: `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): 7b70ef2305f03fe907a7baa30d72b852bfdc95c54a824a26e7826334d0cfec44
Frozen by: scripts/phase_a_freeze.py
"""
import numpy as np
import torch
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
from ascendbench.frozen.shim import NULL_BLOCK_ID, PAD_SLOT_ID  # Phase A': attention sentinel constants (B-inert-shim)

@triton.jit()
def _causal_conv1d_fwd_kernel(x_ptr, w_ptr, bias_ptr, initial_states_ptr, cache_indices_ptr, has_initial_states_ptr, query_start_loc_ptr, batch_ptr, token_chunk_offset_ptr, block_idx_first_scheduled_token, block_idx_last_scheduled_token, initial_state_idx, num_computed_tokens, o_ptr, dim: tl.constexpr, seqlen: tl.int32, num_cache_lines: tl.constexpr, stride_x_dim: tl.constexpr, stride_x_token: tl.int64, stride_w_dim: tl.constexpr, stride_w_width: tl.constexpr, stride_istate_seq: tl.constexpr, stride_istate_dim: tl.constexpr, stride_istate_token: tl.constexpr, stride_cache_indices: tl.constexpr, stride_o_dim: tl.constexpr, stride_o_token: tl.int64, stride_block_m: tl.constexpr, pad_slot_id: tl.constexpr, null_block_id: tl.constexpr, HAS_BIAS: tl.constexpr, KERNEL_WIDTH: tl.constexpr, SILU_ACTIVATION: tl.constexpr, IS_APC_ENABLED: tl.constexpr, HAS_NULL_BLOCK: tl.constexpr, NP2_STATELEN: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = KERNEL_WIDTH - 1
    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    if idx_seq == pad_slot_id:
        return
    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index
    B_size: tl.constexpr = stride_block_m * BLOCK_M
    if IS_APC_ENABLED:
        current_first_index = tl.load(block_idx_first_scheduled_token + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
        sequence_completed_index = tl.load(num_computed_tokens + idx_seq)
        sequence_completed_offset_token = sequence_completed_index % B_size
        seq_completed_offset = B_size - sequence_completed_offset_token
        seq_end_offset = (seqlen - seq_completed_offset) % B_size
        last_full_block_token_index = sequence_end_index - seq_end_offset
        if seq_end_offset == 0:
            last_full_block_token_index = last_full_block_token_index - B_size
        n_block_to_fill = current_last_index - current_first_index
        conv_state_init_index = tl.load(initial_state_idx + idx_seq)
    else:
        n_block_to_fill = 0
        current_last_index = 0
        conv_state_init_index = 0
        current_first_index = 0
        last_full_block_token_index = 0
    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)
    x_base = x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim
    conv_states_input_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_cache_indices + conv_state_init_index).to(tl.int64)
    if HAS_NULL_BLOCK:
        if conv_states_input_coord == null_block_id:
            return
    conv_states_base = conv_states_ptr + conv_states_input_coord * stride_conv_state_seq + idx_feats * stride_conv_state_dim
    w_base = w_ptr + idx_feats * stride_w_dim
    if chunk_offset == 0:
        load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        else:
            if KERNEL_WIDTH >= 2:
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:
                col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
        if state_len <= seqlen:
            idx_tokens_last = seqlen - state_len + tl.arange(0, NP2_STATELEN)
            x_ptrs = x_ptr + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None] + (idx_feats * stride_x_dim)[None, :]
            mask_x = (idx_tokens_last >= 0)[:, None] & (idx_tokens_last < seqlen)[:, None] & (idx_feats < dim)[None, :]
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)
            conv_states_output_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_cache_indices + current_last_index).to(tl.int64)
            conv_states_ptrs_target = (conv_states_ptr + conv_states_output_coord * stride_conv_state_seq + idx_feats * stride_conv_state_dim)[None, :] + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, loaded_x, mask)
        elif load_init_state:
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)
            conv_states_ptrs_source = conv_states_ptr + conv_states_input_coord * stride_conv_state_seq + (idx_feats * stride_conv_state_dim)[None, :] + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
            mask = (conv_states_input_coord < num_cache_lines) & (idx_tokens_conv + seqlen < state_len)[:, None] & (idx_feats < dim)[None, :]
            conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)
            VAL = state_len - seqlen
            x_ptrs = x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
            mask_x = (idx_tokens_conv - VAL >= 0)[:, None] & (idx_tokens_conv - VAL < seqlen)[:, None] & (idx_feats < dim)[None, :]
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            tl.debug_barrier()
            new_conv_state = tl.where(mask, conv_state, loaded_x)
            conv_states_ptrs_target = conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.store(conv_states_ptrs_target, new_conv_state, mask)
        else:
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)
            VAL = state_len - seqlen
            x_ptrs = x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
            mask_x = (idx_tokens_conv - VAL >= 0)[:, None] & (idx_tokens_conv - VAL < seqlen)[:, None] & (idx_feats < dim)[None, :]
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            conv_states_ptrs_target = conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.store(conv_states_ptrs_target, new_conv_state, mask)
    else:
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            conv_states_ptrs = prior_tokens
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
        if KERNEL_WIDTH == 3:
            conv_states_ptrs = prior_tokens
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 1 * stride_x_token
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 1 * stride_x_token
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 2 * stride_x_token
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
        if KERNEL_WIDTH == 5:
            conv_states_ptrs = prior_tokens
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 1 * stride_x_token
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 2 * stride_x_token
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 3 * stride_x_token
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
        if chunk_offset - 1 < n_block_to_fill:
            idx_tokens_last = last_full_block_token_index - (n_block_to_fill - chunk_offset) * B_size - state_len + tl.arange(0, NP2_STATELEN)
            x_ptrs = x_ptr + (idx_tokens_last * stride_x_token)[:, None] + (idx_feats * stride_x_dim)[None, :]
            mask_x = (idx_tokens_last >= 0)[:, None] & (idx_feats < dim)[None, :]
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)
            conv_states_output_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_cache_indices + current_first_index + (chunk_offset - 1)).to(tl.int64)
            conv_states_ptrs_target = (conv_states_ptr + conv_states_output_coord * stride_conv_state_seq + idx_feats * stride_conv_state_dim)[None, :] + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, loaded_x, mask)
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_base_1d = x_base + token_offset * stride_x_token
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + 0 * stride_w_width
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + 1 * stride_w_width
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + 2 * stride_w_width
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + 3 * stride_w_width
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload
        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            acc += matrix_x * matrix_w
        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x
        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (idx_feats < dim)
        o_ptrs = o_ptr + (sequence_start_index + token_offset + idx_token) * stride_o_token + idx_feats * stride_o_dim
        tl.store(o_ptrs, acc, mask=mask_1d)

import torch
import triton

# Authored launcher: drives the in-module @triton.jit `_causal_conv1d_fwd_kernel`
# (vLLM vllm/model_executor/layers/mamba/ops/causal_conv1d.py, commit 18f6bf5)
# directly from the flat oracle fixtures.
#
# Why a launcher rather than the host wrapper `causal_conv1d_fn`: that wrapper's
# signature requires a `conv_states` cache tensor plus `query_start_loc`,
# `cache_indices`, scheduler `batch_ptr` / `token_chunk_offset_ptr`, and the
# continuous-batching / prefix-cache bookkeeping. shapes.yaml provides only the
# plain depthwise-conv fixtures (x, weight, bias, activation), so the by-name
# bind to the wrapper is shapes-incomplete. The kernel's mathematical core --
# the per-channel causal-window dot product the oracle models -- is reached by
# the single-sequence, no-initial-state path, which we set up here exactly as
# `causal_conv1d_fn` does for a one-sequence batch.
#
# Math the kernel realises (per channel d, output token t), with width W =
# weight.shape[1] = KERNEL_WIDTH and left-pad W-1 zeros:
#     acc = bias[d] + sum_{j=0..W-1} x_pad[d, t-(W-1)+j] * w[d, j]
#     out[d, t] = acc / (1 + exp(-acc))          # SiLU
# This is the identical op the oracle computes via F.conv1d(pad(x,(W-1,0)),
# w[dim,1,W], groups=dim) + bias[dim], silu. The continuous-batching machinery
# (APC, multi-sequence segmentation, conv_state cache fill) is upstream KV-cache
# plumbing the oracle does not model; it is held inert here (IS_APC_ENABLED=False,
# has_initial_state=False), NOT removed -- the graded jit is invoked verbatim.
#
# Launch reconciliation (mirrors causal_conv1d_fn's metadata=None path):
#   * One sequence of length L: query_start_loc = [0, L]; the kernel derives the
#     per-program (idx_seq, chunk_offset) from batch_ptr / token_chunk_offset_ptr.
#   * batch_ptr[k] = 0 (the single sequence) and token_chunk_offset_ptr[k] = k
#     for k in [0, n_chunks), with n_chunks = cdiv(L, BLOCK_M=8) -- so chunks
#     beyond the first hit the `chunk_offset > 0` branch that reloads prior
#     tokens from x (exercised whenever L > 8).
#   * NULL_BLOCK_ID is 0 and the wrapper passes HAS_NULL_BLOCK = (null_block_id
#     is not None) = True, so the kernel early-returns when the cache coordinate
#     equals 0. We therefore allocate >= 2 cache lines and route the sequence to
#     cache coordinate 1 (cache_indices = [1]); coordinate 0 would make the
#     kernel write nothing. has_initial_state = [False] makes the prior-token
#     window zero (left-pad), matching the oracle; conv_states contents are not
#     read for the math, only the coordinate is read for routing.
#   * BLOCK_M=8, BLOCK_N=256, num_stages=2 are the upstream launch constants;
#     they tile the grid only and never change the result (masked stores).
#
# Block sizes / scheduler tensors are neutral launch metadata. weight, bias and x
# are forwarded verbatim, so all three are load-bearing: zeroing the weight,
# permuting its taps, dropping the bias, or perturbing x all change the output.
def run(inputs: dict) -> dict:
    # NULL_BLOCK_ID / PAD_SLOT_ID resolve from the kernel module namespace this
    # launcher is exec'd into: kernel.py's top-level
    # `from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID`
    # is the binding on the original side, and the severed staging rewrites that
    # same line to `from ascendbench.frozen.shim import NULL_BLOCK_ID, PAD_SLOT_ID`.
    # Referencing them as module globals (NOT a fresh `from vllm import` inside
    # run(), which would resolve vllm at call time and leak it onto the severed
    # import-audit surface) keeps both sides byte-identical sentinel ints
    # (PAD_SLOT_ID=-1, NULL_BLOCK_ID=0) with zero upstream import here.
    null_block_id = NULL_BLOCK_ID  # noqa: F821 (injected into kernel.py module)
    pad_slot_id = PAD_SLOT_ID  # noqa: F821 (injected into kernel.py module)

    x = inputs["x"]
    weight = inputs["weight"].contiguous()
    bias = inputs.get("bias")
    activation = inputs.get("activation")

    dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)
    device = x.device

    # channel-last layout (stride_x_dim == 1), as the kernel's validate_data path
    # asserts ("Need to run in channel-last layout"). The oracle is layout
    # agnostic, so we materialise the contiguous channel-last view explicitly.
    x_cl = x.t().contiguous().t()  # shape [dim, seqlen], stride (1, dim)
    assert x_cl.stride(0) == 1

    out = torch.empty_like(x_cl)

    # one sequence spanning the whole token axis
    query_start_loc = torch.tensor([0, seqlen], dtype=torch.int32, device=device)

    BLOCK_M = 8
    BLOCK_N = 256

    # scheduler metadata for a single sequence split into ceil(L / BLOCK_M) chunks
    n_chunks = (seqlen + BLOCK_M - 1) // BLOCK_M
    MAX_NUM_PROGRAMS = max(1024, n_chunks)
    batch_ptr = torch.full(
        (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device
    )
    token_chunk_offset_ptr = torch.full(
        (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device
    )
    batch_ptr[:n_chunks] = 0
    token_chunk_offset_ptr[:n_chunks] = torch.arange(
        n_chunks, dtype=torch.int32, device=device
    )

    # >= 2 cache lines; route the sequence to coordinate 1 (coordinate 0 ==
    # NULL_BLOCK_ID would trigger the kernel's null-block early return).
    num_cache_lines = 2
    conv_states = torch.zeros(
        (num_cache_lines, dim, state_len), dtype=x_cl.dtype, device=device
    )
    cache_indices = torch.tensor([1], dtype=torch.int32, device=device)
    has_initial_state = torch.zeros((1,), dtype=torch.bool, device=device)

    bias_t = bias.contiguous() if bias is not None else None

    stride_x_dim = x_cl.stride(0)
    stride_x_token = x_cl.stride(1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    stride_istate_seq = conv_states.stride(0)
    stride_istate_dim = conv_states.stride(1)
    stride_istate_token = conv_states.stride(2)
    stride_cache_indices = cache_indices.stride(0)
    stride_o_dim = out.stride(0)
    stride_o_token = out.stride(1)

    def grid(META):
        return (n_chunks, triton.cdiv(dim, META["BLOCK_N"]))

    _causal_conv1d_fwd_kernel[grid](  # noqa: F821 (injected into kernel.py module)
        x_cl,
        weight,
        bias_t,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        batch_ptr,
        token_chunk_offset_ptr,
        None,  # block_idx_first_scheduled_token -> IS_APC_ENABLED=False
        None,  # block_idx_last_scheduled_token
        None,  # initial_state_idx
        None,  # num_computed_tokens
        out,
        # matrix dims
        dim,
        seqlen,
        num_cache_lines,
        # strides
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_cache_indices,
        stride_o_dim,
        stride_o_token,
        BLOCK_M // BLOCK_M,  # stride_block_m = block_size_to_align // BLOCK_M
        # others
        pad_slot_id,
        null_block_id,
        # meta
        HAS_BIAS=bias_t is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_APC_ENABLED=False,
        HAS_NULL_BLOCK=null_block_id is not None,
        NP2_STATELEN=np2_statelen,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=2,
    )

    return {"output": out}
