# Phase A': sliced to the graded entrypoint's module-level closure
# (whole-module vendored kernel; dead upstream siblings + their
# upstream imports dropped -- the kept defs are byte-identical).
import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_eagle_inputs_kernel(last_token_indices_ptr, eagle_current_draft_step_ptr, eagle_input_ids_ptr, eagle_positions_ptr, eagle_query_start_loc_ptr, eagle_seq_lens_ptr, target_input_ids_ptr, target_positions_ptr, idx_mapping_ptr, last_sampled_ptr, next_prefill_tokens_ptr, num_sampled_ptr, num_rejected_ptr, query_start_loc_ptr, seq_lens_ptr, max_num_reqs, BLOCK_SIZE: tl.constexpr):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx)
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    query_len -= num_rejected
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        next_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        next_token = tl.load(next_prefill_tokens_ptr + req_state_idx)
    for i in range(1, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        input_ids = tl.load(target_input_ids_ptr + query_start + block, mask=mask)
        tl.store(eagle_input_ids_ptr + query_start + block - 1, input_ids, mask=mask)
    last_token_index = query_start + query_len - 1
    tl.store(last_token_indices_ptr + req_idx, last_token_index)
    tl.store(eagle_input_ids_ptr + last_token_index, next_token)
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        target_pos = tl.load(target_positions_ptr + query_start + block, mask=mask)
        tl.store(eagle_positions_ptr + query_start + block, target_pos, mask=mask)
    tl.store(eagle_query_start_loc_ptr + req_idx, query_start)
    tl.store(eagle_seq_lens_ptr + req_idx, seq_len)
    if req_idx == num_reqs - 1:
        tl.store(eagle_current_draft_step_ptr, 0)
        for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs + 1
            tl.store(eagle_query_start_loc_ptr + block, query_end, mask=mask)
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(eagle_seq_lens_ptr + block, 0, mask=mask)
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(last_token_indices_ptr + block, 0, mask=mask)

import torch


# Authored launcher: drives the in-module @triton.jit
# `_prepare_eagle_inputs_kernel` exactly as the upstream host wrapper
# `prepare_eagle_inputs` does (vLLM
# vllm/v1/worker/gpu/spec_decode/eagle/speculator.py, commit
# 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19).
#
# Why a launcher rather than the host wrapper: the real entry
# `prepare_eagle_inputs(...)` reads its tensors off live `InputBuffers` /
# `InputBatch` objects whose construction reaches into the vllm runtime
# (KV-cache buffers, idx_mapping bookkeeping, cudagraph state). None of
# that bookkeeping is the unit under test: the kernel is a pure per-request
# token-shift + metadata-prep pass. We reconstruct exactly the flat tensors
# the host wrapper forwards, straight from the dense oracle inputs, and call
# the kernel on the same grid `(num_reqs,)` with the upstream `BLOCK_SIZE=1024`
# launch literal.
#
# Layout reconciliation (dense fixtures -> upstream kernel contract):
#   The fixtures pack `num_reqs` requests contiguously with a constant raw
#   query length `Q = query_len` per request, so the per-request token spans
#   are `[r*Q, (r+1)*Q)` with no gaps and total token count `num_reqs * Q`.
#   We rebuild `query_start_loc = arange(0, num_reqs + 1) * Q` (the int32
#   indptr the kernel reads via `query_start_loc_ptr`) and pass the
#   target_input_ids / target_positions / num_sampled / num_rejected /
#   idx_mapping / seq_lens fixtures straight through. `last_sampled` and
#   `next_prefill_tokens` are indexed by `req_state_idx = idx_mapping[req_idx]`
#   inside the kernel, so they are forwarded unmodified at length
#   `max_num_reqs`. `num_rejected[r] in [0, Q)` by fixture construction, so the
#   kernel's truncated query length `Q - num_rejected[r]` stays >= 1.
#
#   Output buffers are zero-initialised exactly like the upstream
#   `InputBuffers` pre-launch state, so the rejected-tail slots the kernel
#   never writes keep value 0 (the oracle reproduces this). The cudagraph pad
#   tail of `eagle_query_start_loc` / `eagle_seq_lens` / `last_token_indices`
#   (`[num_reqs : max_num_reqs ...]`) is sized to `max_num_reqs` and graded so
#   the kernel's final-program padding logic is exercised.
#   `eagle_current_draft_step` is initialised to a nonzero sentinel so the
#   kernel's reset-to-0 store is actually verified by the bit-exact compare.
#
#   dtypes mirror the upstream EagleSpeculator / InputBuffers: int32 for
#   input_ids / query_start_loc / seq_lens / target_input_ids / last_sampled /
#   next_prefill_tokens, int64 for positions, last_token_indices and the
#   draft-step scalar.
def run(inputs: dict) -> dict:
    Q = int(inputs["query_len"])
    B = int(inputs["num_reqs"])
    M = int(inputs["max_num_reqs"])

    device = inputs["target_input_ids"].device
    total = B * Q

    target_input_ids = inputs["target_input_ids"].to(torch.int32).contiguous()
    target_positions = inputs["target_positions"].to(torch.int64).contiguous()
    idx_mapping = inputs["idx_mapping"].to(torch.int32).contiguous()
    num_sampled = inputs["num_sampled"].to(torch.int32).contiguous()
    num_rejected = inputs["num_rejected"].to(torch.int32).contiguous()
    last_sampled = inputs["last_sampled"].to(torch.int32).contiguous()
    next_prefill_tokens = inputs["next_prefill_tokens"].to(torch.int32).contiguous()
    seq_lens = inputs["seq_lens"].to(torch.int32).contiguous()

    # Contiguous per-request indptr: request r owns the token span [r*Q, (r+1)*Q).
    query_start_loc = (
        torch.arange(0, B + 1, device=device, dtype=torch.int32) * Q
    ).contiguous()

    # Zero-initialised eagle output buffers (the upstream InputBuffers
    # pre-launch state). The kernel writes only each request's truncated span;
    # rejected tails and unused cudagraph-pad slots stay at 0.
    eagle_input_ids = torch.zeros(total, dtype=torch.int32, device=device)
    eagle_positions = torch.zeros(total, dtype=torch.int64, device=device)
    eagle_query_start_loc = torch.zeros(M + 1, dtype=torch.int32, device=device)
    eagle_seq_lens = torch.zeros(M, dtype=torch.int32, device=device)
    # int64, matching the upstream EagleSpeculator.last_token_indices buffer.
    last_token_indices = torch.zeros(M, dtype=torch.int64, device=device)
    # Nonzero sentinel: the kernel's final program must reset this to 0.
    eagle_current_draft_step = torch.full((), 7, dtype=torch.int64, device=device)

    # The in-module @triton.jit kernel is exec'd into this namespace by the
    # loader; reference it by name. Both S2 sides share the same symbol.
    _prepare_eagle_inputs_kernel[(B,)](
        last_token_indices,
        eagle_current_draft_step,
        eagle_input_ids,
        eagle_positions,
        eagle_query_start_loc,
        eagle_seq_lens,
        target_input_ids,
        target_positions,
        idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        query_start_loc,
        seq_lens,
        M,
        BLOCK_SIZE=1024,
    )

    return {
        "eagle_input_ids": eagle_input_ids,
        "eagle_positions": eagle_positions,
        "eagle_query_start_loc": eagle_query_start_loc,
        "eagle_seq_lens": eagle_seq_lens,
        "last_token_indices": last_token_indices,
        "eagle_current_draft_step": eagle_current_draft_step,
    }
