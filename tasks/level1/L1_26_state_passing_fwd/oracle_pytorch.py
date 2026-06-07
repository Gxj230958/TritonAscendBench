"""Oracle reference for L1_26_state_passing_fwd_kernel.

Tier: formula_exact_manual
Tolerance class: recurrent_chunk
Category: ssm

The kernel under test is the GPU triton implementation vendored in
kernel.py; this file authors an eager PyTorch reference from the
operation's documented mathematical definition (no upstream code copy).

Contract matches kernel.py::_state_passing_fwd (the frozen host wrapper):
  states:             [nchunks, nheads, dim]
  dA_cumsum:          [nheads, nchunks, chunk_size]  (per-head, per-chunk
                      cumulative decay; the inter-chunk decay is the
                      chunk-FINAL value dA_cumsum[h, c, -1], see
                      kernel.py:79-82 + the recurrence at kernel.py:107)
  last_chunk_indices: [batch]  inclusive end-chunk index per sequence;
                      sequence b spans chunks
                      (last_chunk_indices[b-1]+1 .. last_chunk_indices[b]).
  initial_states:     optional [batch, nheads, dim]; None => zero state.
  output:             [nchunks, nheads, dim]

Per head h, within each sequence's chunk span, the recurrence is
  S_c = exp(dA_cumsum[h, c, -1]) * S_{c-1} + states[c, h]
with S reset to initial_states (or zero) at each sequence boundary.
"""
from __future__ import annotations

import torch


def _state_passing_fwd(
    states: torch.Tensor, dA_cumsum: torch.Tensor,
    last_chunk_indices: torch.Tensor,
    initial_states: torch.Tensor | None = None,
) -> torch.Tensor:
    nchunks, nheads, dim = states.shape
    sf = states.to(torch.float32)
    # Inter-chunk decay is the chunk-final cumsum slice (kernel.py:79-82).
    decay = torch.exp(dA_cumsum.to(torch.float32)[:, :, -1])   # [nheads, nchunks]
    out = torch.zeros_like(sf)
    lci = last_chunk_indices.to(torch.long).tolist()

    chunk_start = 0
    for b, last in enumerate(lci):
        chunk_end = int(last) + 1
        if initial_states is not None:
            prev = initial_states.to(torch.float32)[b]            # [nheads, dim]
        else:
            prev = torch.zeros(nheads, dim, dtype=torch.float32,
                               device=states.device)
        for c in range(chunk_start, chunk_end):
            cur = decay[:, c][:, None] * prev + sf[c]             # [nheads, dim]
            out[c] = cur
            prev = cur
        chunk_start = chunk_end
    return out.to(states.dtype)


def run(inputs: dict) -> dict:
    return {"output": _state_passing_fwd(
        inputs["states"], inputs["dA_cumsum"], inputs["last_chunk_indices"],
        inputs.get("initial_states"))}
