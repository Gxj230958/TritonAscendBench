"""Oracle reference for L2_20_causal_conv1d_update_kernel.

Tier: formula_exact_manual
Tolerance class: vector_reduce
Category: convolution

Authors an eager PyTorch reference for the vLLM `causal_conv1d_update`
single-step decode state update from the operation's mathematical
definition (no upstream code copy). The graded entrypoint is the host
wrapper `causal_conv1d_update`, which launches the frozen
`_causal_conv1d_update_kernel`; this oracle mirrors that kernel's exact
behaviour for the single-token (seqlen == 1), non-varlen,
non-speculative-decode path the task drives.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    conv_state_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step decode causal conv1d state update with cache gather.

    Shapes:
      x:                  [batch, dim]                  -- new decode token
      conv_state:         [num_cache_lines, dim, KW-1]  -- rolling history cache
      weight:             [dim, KW]                     -- depthwise conv taps
      bias:               [dim]                         -- optional
      conv_state_indices: [batch] int                  -- cache line per batch row

    For batch row ``i`` selecting cache line ``c = conv_state_indices[i]``:
      old = conv_state[c]                         # [dim, KW-1]
      window = concat(old, x[i, :, None], dim=-1) # [dim, KW]
      y[i] = sum_k window[:, k] * weight[:, k] (+ bias)   then optional SiLU
      conv_state[c] <- window[:, 1:]              # rolled left, append new token

    Cache lines not named by any index are passed through unchanged (the
    kernel only launches programs for idx_seq < batch and only writes the
    indexed target line). Returns (output [batch, dim], updated full cache).
    """
    x = x.to(torch.float32)
    conv_state = conv_state.to(torch.float32)
    w = weight.to(torch.float32)                              # [dim, KW]
    kw = w.shape[-1]
    idx = conv_state_indices.to(torch.long)                  # [batch]

    old = conv_state[idx]                                    # [batch, dim, KW-1]
    window = torch.cat([old, x.unsqueeze(-1)], dim=-1)       # [batch, dim, KW]
    y = (window * w.unsqueeze(0)).sum(dim=-1)                # [batch, dim]
    if bias is not None:
        y = y + bias.to(torch.float32)
    if activation in ("silu", "swish"):
        y = F.silu(y)

    new_cache = conv_state.clone()
    if kw > 1:
        # Rolled-left state: drop the oldest history column, append the new
        # token. Scatter back into the selected cache lines (index_copy so a
        # repeated index would deterministically keep the last write, matching
        # the kernel's last-writer-wins store; the task's cyclic permutation
        # has no repeats).
        rolled = window[..., 1:]                             # [batch, dim, KW-1]
        new_cache.index_copy_(0, idx, rolled)

    return y.to(torch.float32), new_cache.to(torch.float32)


def run(inputs: dict) -> dict:
    y, st = _causal_conv1d_update(
        inputs["x"],
        inputs["conv_state"],
        inputs["weight"],
        inputs.get("bias"),
        inputs.get("activation"),
        inputs["conv_state_indices"],
    )
    return {"output": y, "conv_state": st}
