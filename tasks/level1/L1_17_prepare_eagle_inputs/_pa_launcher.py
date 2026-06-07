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
