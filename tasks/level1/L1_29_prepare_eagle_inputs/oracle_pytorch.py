"""Oracle reference for L1_29_prepare_eagle_inputs_kernel.

Tier: formula_exact_manual
Tolerance class: sampling_rng
Category: spec_decode

The kernel under test is the GPU triton implementation vendored in
kernel.py (``_prepare_eagle_inputs_kernel``); this file authors an eager
PyTorch reference from the operation's documented mathematical definition
(no upstream code copy, never invokes the kernel).

Operation (vLLM EAGLE speculative-decoding draft input prep,
``vllm/v1/worker/gpu/spec_decode/eagle/speculator.py``). The kernel runs
one program per request over a packed token buffer. Per request ``r`` with
token span ``[start, end)`` where ``start = query_start_loc[r]`` and
``end = query_start_loc[r + 1]``:

  * truncated query length ``Lt = (end - start) - num_rejected[r]``;
  * ``next_token = last_sampled[idx_mapping[r]]`` if ``num_sampled[r] > 0``
    else ``next_prefill_tokens[idx_mapping[r]]`` (chunked-prefill case);
  * ``eagle_input_ids`` over the truncated prefix is ``target_input_ids``
    shifted left by one — ``eagle_input_ids[start + k - 1] =
    target_input_ids[start + k]`` for ``k`` in ``[1, Lt)`` — with the final
    truncated slot overwritten by ``next_token``;
  * ``last_token_indices[r] = start + Lt - 1``;
  * ``eagle_positions[start : start + Lt] = target_positions[start : start + Lt]``;
  * ``eagle_query_start_loc[r] = start``; ``eagle_seq_lens[r] = seq_lens[r]``;
  * the final program zeros ``eagle_current_draft_step`` and pads the
    cudagraph tail: ``eagle_query_start_loc[num_reqs : max_num_reqs + 1] =
    end_last`` (the raw, un-truncated end of the last request, i.e. the
    total token count), ``eagle_seq_lens[num_reqs : max_num_reqs] = 0`` and
    ``last_token_indices[num_reqs : max_num_reqs] = 0``.

Slots of ``eagle_input_ids`` / ``eagle_positions`` in a request's rejected
tail ``[start + Lt, end)`` are left at the pre-launch (zero-init) buffer
contents — the reference reproduces that exactly.

The fixtures pack the requests contiguously with a constant raw query
length ``Q`` per request (``query_start_loc = arange(0, num_reqs + 1) * Q``,
total tokens ``= num_reqs * Q``); the reference derives that same span
layout from the ``query_len`` / ``num_reqs`` knobs, never reading a kernel
pointer.
"""
from __future__ import annotations

import torch


def _prepare_eagle_inputs(
    target_input_ids: torch.Tensor,
    target_positions: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    query_len: int,
    num_reqs: int,
    max_num_reqs: int,
) -> dict[str, torch.Tensor]:
    device = target_input_ids.device
    Q = int(query_len)
    B = int(num_reqs)
    M = int(max_num_reqs)
    total = B * Q

    tii = target_input_ids.to(torch.int32)
    tpos = target_positions.to(torch.int64)
    idx = idx_mapping.to(torch.long)
    nsamp = num_sampled.to(torch.long)
    nrej = num_rejected.to(torch.long)
    lsamp = last_sampled.to(torch.int32)
    nprefill = next_prefill_tokens.to(torch.int32)
    slens = seq_lens.to(torch.int32)

    eagle_input_ids = torch.zeros(total, dtype=torch.int32, device=device)
    eagle_positions = torch.zeros(total, dtype=torch.int64, device=device)
    eagle_query_start_loc = torch.zeros(M + 1, dtype=torch.int32, device=device)
    eagle_seq_lens = torch.zeros(M, dtype=torch.int32, device=device)
    # int64, matching the upstream EagleSpeculator.last_token_indices buffer.
    last_token_indices = torch.zeros(M, dtype=torch.int64, device=device)

    for r in range(B):
        start = r * Q
        # num_rejected[r] < Q by fixture construction, so Lt in [1, Q].
        Lt = Q - int(nrej[r].item())
        req_state_idx = int(idx[r].item())
        if int(nsamp[r].item()) > 0:
            next_token = int(lsamp[req_state_idx].item())
        else:
            next_token = int(nprefill[req_state_idx].item())

        if Lt - 1 > 0:
            # Shift the target ids left by one over the truncated prefix.
            eagle_input_ids[start : start + Lt - 1] = tii[start + 1 : start + Lt]
        last_idx = start + Lt - 1
        last_token_indices[r] = last_idx
        eagle_input_ids[last_idx] = next_token

        eagle_positions[start : start + Lt] = tpos[start : start + Lt]
        eagle_query_start_loc[r] = start
        eagle_seq_lens[r] = slens[r]

    # Final-program cudagraph tail padding. end_last is the raw (un-truncated)
    # end of the last request == total token count.
    end_last = total
    eagle_query_start_loc[B : M + 1] = end_last
    # eagle_seq_lens[B:M] and last_token_indices[B:M] stay at the zero-init pad.
    eagle_current_draft_step = torch.zeros((), dtype=torch.int64, device=device)

    return {
        "eagle_input_ids": eagle_input_ids,
        "eagle_positions": eagle_positions,
        "eagle_query_start_loc": eagle_query_start_loc,
        "eagle_seq_lens": eagle_seq_lens,
        "last_token_indices": last_token_indices,
        "eagle_current_draft_step": eagle_current_draft_step,
    }


def run(inputs: dict) -> dict:
    return _prepare_eagle_inputs(
        inputs["target_input_ids"],
        inputs["target_positions"],
        inputs["idx_mapping"],
        inputs["num_sampled"],
        inputs["num_rejected"],
        inputs["last_sampled"],
        inputs["next_prefill_tokens"],
        inputs["seq_lens"],
        int(inputs["query_len"]),
        int(inputs["num_reqs"]),
        int(inputs["max_num_reqs"]),
    )
