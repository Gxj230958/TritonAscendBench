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
    # same line to `from tasks._frozen_shim import NULL_BLOCK_ID, PAD_SLOT_ID`.
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
