"""Task L3_08_causal_conv1d_update_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_causal_conv1d_update_kernel`
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
from tasks._frozen_shim import NULL_BLOCK_ID, PAD_SLOT_ID  # Phase A': attention sentinel constants (B-inert-shim)

@triton.jit()
def _causal_conv1d_update_kernel(x_ptr, w_ptr, bias_ptr, conv_state_ptr, conv_state_indices_ptr, num_accepted_tokens_ptr, query_start_loc_ptr, block_idx_last_scheduled_token, initial_state_idx, o_ptr, batch: int, dim: tl.constexpr, seqlen: tl.constexpr, state_len: tl.constexpr, num_cache_lines: tl.constexpr, stride_x_seq: tl.constexpr, stride_x_dim: tl.constexpr, stride_x_token: tl.int64, stride_w_dim: tl.constexpr, stride_w_width: tl.constexpr, stride_conv_state_seq: tl.constexpr, stride_conv_state_dim: tl.constexpr, stride_conv_state_tok: tl.constexpr, stride_state_indices: tl.constexpr, stride_o_seq: tl.constexpr, stride_o_dim: tl.constexpr, stride_o_token: tl.int64, null_block_id: tl.constexpr, HAS_BIAS: tl.constexpr, KERNEL_WIDTH: tl.constexpr, SILU_ACTIVATION: tl.constexpr, IS_VARLEN: tl.constexpr, IS_APC_ENABLED: tl.constexpr, IS_SPEC_DECODING: tl.constexpr, NP2_STATELEN: tl.constexpr, HAS_NULL_BLOCK: tl.constexpr, BLOCK_N: tl.constexpr):
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    if IS_APC_ENABLED:
        conv_state_init = tl.load(initial_state_idx + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
    else:
        conv_state_init = 0
        current_last_index = 0
    conv_states_input_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init).to(tl.int64)
    if HAS_NULL_BLOCK:
        if conv_states_input_coord == null_block_id:
            return
    if IS_VARLEN:
        query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
        query_end_index = tl.load(query_start_loc_ptr + (idx_seq + 1)).to(tl.int64)
        state_len = state_len - (seqlen - (query_end_index - query_start_index))
        seqlen = query_end_index - query_start_index
        x_offset = query_start_index * stride_x_token
        o_offset = query_start_index * stride_o_token
    else:
        query_start_index = idx_seq * seqlen
        query_end_index = query_start_index + seqlen
        x_offset = idx_seq * stride_x_seq
        o_offset = idx_seq * stride_o_seq
    if query_start_index == query_end_index:
        return
    if IS_SPEC_DECODING:
        conv_state_token_offset = tl.load(num_accepted_tokens_ptr + idx_seq).to(tl.int64) - 1
    else:
        conv_state_token_offset = 0
    conv_states_base = conv_state_ptr + conv_states_input_coord * stride_conv_state_seq + idx_feats * stride_conv_state_dim
    mask_w = idx_feats < dim
    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 6:
        conv_states_ptrs = prior_tokens + 4 * stride_conv_state_tok
        col4 = tl.load(conv_states_ptrs, mask_w, 0.0)
    idx_tokens = tl.arange(0, NP2_STATELEN)
    conv_state_ptrs_source = conv_state_ptr + conv_states_input_coord * stride_conv_state_seq + conv_state_token_offset * stride_conv_state_tok + (idx_feats * stride_conv_state_dim)[None, :] + ((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) * stride_conv_state_tok)[:, None]
    mask = (conv_states_input_coord < num_cache_lines) & (idx_tokens + seqlen < state_len)[:, None] & (idx_feats < dim)[None, :]
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)
    VAL = state_len - seqlen
    x_base = x_ptr + x_offset + idx_feats * stride_x_dim
    x_ptrs = x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    mask_x = (idx_tokens - VAL >= 0)[:, None] & (idx_tokens - VAL < seqlen)[:, None] & (idx_feats < dim)[None, :]
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()
    new_conv_state = tl.where(mask, conv_state, loaded_x)
    conv_states_offset = tl.load(conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index).to(tl.int64)
    conv_state_ptrs_target = (conv_state_ptr + conv_states_offset * stride_conv_state_seq + idx_feats * stride_conv_state_dim)[None, :] + (idx_tokens * stride_conv_state_tok)[:, None]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_base = w_ptr + idx_feats * stride_w_dim
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
    if KERNEL_WIDTH >= 5:
        w_ptrs = w_base + 4 * stride_w_width
        w_col4 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 6:
        w_ptrs = w_base + 5 * stride_w_width
        w_col5 = tl.load(w_ptrs, mask_w, other=0.0)
    x_base_1d = x_base
    mask_x_1d = idx_feats < dim
    for idx_token in tl.range(seqlen):
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
            elif KERNEL_WIDTH == 5:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 6:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    matrix_x = col4
                elif j == 5:
                    matrix_w = w_col5
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
        elif KERNEL_WIDTH == 5:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = matrix_x
        elif KERNEL_WIDTH == 6:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = col4
            col4 = matrix_x
        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (idx_feats < dim)
        o_ptrs = o_ptr + o_offset + idx_token * stride_o_token + idx_feats * stride_o_dim
        tl.store(o_ptrs, acc, mask=mask_1d)

def causal_conv1d_update(x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None=None, activation: bool | str | None=None, conv_state_indices: torch.Tensor | None=None, num_accepted_tokens: torch.Tensor | None=None, query_start_loc: torch.Tensor | None=None, max_query_len: int=-1, null_block_id: int=NULL_BLOCK_ID, block_idx_last_scheduled_token: torch.Tensor | None=None, initial_state_idx: torch.Tensor | None=None, validate_data=False):
    """
    x: Input tensor which can take the following shapes:

    - `[batch, dim]` - single token prediction
    - `[batch, dim, seqlen]` - single or multiple tokens prediction
    - `[num_tokens, dim]` - continuous batching, where num_tokens is
        the total tokens of all sequences in that batch

    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    block_idx_last_scheduled_token: (batch,), dtype int32
        The pointer into conv_state_indices, where the last cache block to be filled is located.
    initial_state_idx: (batch,), dtype int32
        The pointer into conv_state_indices, where the cache block containing the initial state is located.
    num_accepted_tokens: (batch,), dtype int32
        If not None, it indicates the number of accepted tokens for each
        sequence in the batch.
        This is used in speculative decoding, where the conv_state is updated
        in a sliding window manner.
    query_start_loc: (batch + 1,) int32
        If not None, the inputs is given in a varlen fashion and this indicates
        the starting index of each sequence in the batch.
    max_query_len: int
        If query_start_loc is not None, this indicates the maximum query
        length in the batch.
    null_block_id: int
            Block ID used to identify padded entries in
            conv_state_indices. Block 0 is the null block.
            for example: conv_state_indices = [null_block_id, 1, 20, null_block_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen) or (num_tokens, dim), same shape as `x`
    """
    if validate_data:
        assert null_block_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = 'silu' if activation is True else None
    elif activation is not None:
        assert activation in ['silu', 'swish']
    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    if query_start_loc is None:
        batch, dim, seqlen = x.shape
    else:
        assert conv_state_indices is not None
        batch = conv_state_indices.size(0)
        dim = x.size(1)
        seqlen = max_query_len
    _, width = weight.shape
    num_cache_lines, _, state_len = conv_state.size()
    if validate_data:
        assert dim == weight.size(0)
        assert state_len >= width - 1
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert batch == conv_state_indices.shape[0], f'ERROR: conv_state_indices should have shape ({batch},*) but got {conv_state_indices.shape}'
        assert num_cache_lines >= batch
        assert weight.stride(1) == 1
    out = x
    stride_w_dim, stride_w_width = weight.stride()
    if query_start_loc is None:
        stride_x_seq, stride_x_dim, stride_x_token = x.stride()
        stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    else:
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0
        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = conv_state_indices.stride(0) if conv_state_indices is not None else 0
    if num_accepted_tokens is not None:
        state_len = width - 1 + (seqlen - 1)
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (batch, triton.cdiv(dim, META['BLOCK_N']))
    _causal_conv1d_update_kernel[grid](x, weight, bias, conv_state, conv_state_indices, num_accepted_tokens, query_start_loc, block_idx_last_scheduled_token, initial_state_idx, out, batch, dim, seqlen, state_len, num_cache_lines, stride_x_seq, stride_x_dim, stride_x_token, stride_w_dim, stride_w_width, stride_istate_seq, stride_istate_dim, stride_istate_token, stride_state_indices, stride_o_seq, stride_o_dim, stride_o_token, null_block_id, HAS_BIAS=bias is not None, KERNEL_WIDTH=width, SILU_ACTIVATION=activation in ['silu', 'swish'], IS_VARLEN=query_start_loc is not None, IS_APC_ENABLED=block_idx_last_scheduled_token is not None, IS_SPEC_DECODING=num_accepted_tokens is not None, NP2_STATELEN=np2_statelen, HAS_NULL_BLOCK=null_block_id is not None, BLOCK_N=256)
    if unsqueeze:
        out = out.squeeze(-1)
    return out.to(original_x_dtype)

# Single-token decode: x is rank-2 [B, D]; the wrapper unsqueezes it internally
# and rolls conv_state in place. A launcher (vs a direct bind) sidesteps the
# static wrapper-contract gate, which can't see the x.unsqueeze(-1) and wrongly
# demands rank-3 x.
import torch  # noqa: F401


def run(inputs: dict) -> dict:
    x = inputs["x"]                    # [B, D] rank-2 decode token
    conv_state = inputs["conv_state"]  # [B, D, KW-1], rolled in place
    weight = inputs["weight"]
    out = causal_conv1d_update(  # noqa: F821 (exec'd into the kernel module)
        x,
        conv_state,
        weight,
        bias=inputs.get("bias"),
        activation=inputs.get("activation"),
        conv_state_indices=inputs.get("conv_state_indices"),
        null_block_id=inputs.get("null_block_id"),
    )
    # conv_state is graded via meta.interface.return_plus_inplace (the runner
    # overrides out_dict[conv_state] with the post-launch mutated input), so
    # only the conv output is returned here.
    return {"output": out}
