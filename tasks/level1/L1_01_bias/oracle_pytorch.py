"""Oracle reference for L1_01_bias_kernel.

Tier: simple_pytorch_ok
Tolerance class: elementwise_exact_or_inf_safe
  (additive bias is exact in fp32; -inf masking is exact)

In-place semantics: `apply_logit_bias_oracle` mutates `logits` to match
what the GPU kernel would produce. Returned dict contains the post-state
`logits` tensor for runner comparison.
"""
from __future__ import annotations

import torch


def apply_logit_bias_oracle(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    pos: torch.Tensor,
    num_allowed_token_ids: torch.Tensor,
    allowed_token_ids: torch.Tensor,
    num_logit_bias: torch.Tensor,
    logit_bias_token_ids: torch.Tensor,
    logit_bias: torch.Tensor,
    min_lens: torch.Tensor,
    num_stop_token_ids: torch.Tensor,
    stop_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch reference, in-place on `logits` then returned."""
    num_tokens, vocab_size = logits.shape
    neg_inf = float("-inf")

    for t in range(num_tokens):
        r = int(expanded_idx_mapping[t].item())

        # Stage 1: allowed token mask
        n_allowed = int(num_allowed_token_ids[r].item())
        if n_allowed > 0:
            allowed = allowed_token_ids[r, :n_allowed].long()
            saved = logits[t, allowed].clone()
            logits[t, :] = neg_inf
            logits[t, allowed] = saved

        # Stage 2: additive bias
        n_bias = int(num_logit_bias[r].item())
        if n_bias > 0:
            tids = logit_bias_token_ids[r, :n_bias].long()
            logits[t, tids] = logits[t, tids] + logit_bias[r, :n_bias]

        # Stage 3: min-length stop suppression
        n_stop = int(num_stop_token_ids[r].item())
        p = int(pos[t].item())
        m = int(min_lens[r].item())
        if n_stop > 0 and p < m:
            sids = stop_token_ids[r, :n_stop].long()
            logits[t, sids] = neg_inf

    return logits


def run(inputs: dict) -> dict:
    """Standard oracle entrypoint."""
    logits = inputs["logits"].clone()
    out = apply_logit_bias_oracle(
        logits,
        inputs["expanded_idx_mapping"],
        inputs["pos"],
        inputs["num_allowed_token_ids"],
        inputs["allowed_token_ids"],
        inputs["num_logit_bias"],
        inputs["logit_bias_token_ids"],
        inputs["logit_bias"],
        inputs["min_lens"],
        inputs["num_stop_token_ids"],
        inputs["stop_token_ids"],
    )
    return {"logits": out}
