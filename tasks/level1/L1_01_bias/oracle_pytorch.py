"""Oracle reference for L1_01_bias_kernel.

Tier: simple_pytorch_ok
Tolerance class: elementwise_exact_or_inf_safe
  (additive bias is exact in fp32; -inf masking is exact)

The upstream kernel ``apply_logit_bias`` mutates ``logits`` in place via
``tl.store`` and returns ``None``. The phase-A' oracle-consistency
runner detects the None return and falls back to surfacing every
bound parameter as a named output (so the candidate manifest has
one entry per kernel arg). The oracle therefore returns the SAME
structure: the rw_in_place ``logits`` tensor carrying the three-stage
transform (allowed-mask -> bias add -> stop-mask) plus the read-only
tables echoed back unchanged. This makes the comparison's structural
arity match: oracle and candidate both publish 11 flattened tensors,
so the anchor's STRUCTURE check does not fire on a structurally-valid
in-place operator. Per the runner's pre-call input snapshot, this
oracle is invoked on the PRE-mutation tensor mapping; the candidate
mutates its own clone of those inputs.
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
    """Pure-PyTorch reference equivalent to ``apply_logit_bias``.

    Mutates the supplied ``logits`` tensor in place to match the
    upstream kernel semantics and returns it for the caller to surface
    as a named output. The three stages -- allowed-token allowlist,
    additive bias, and min-length stop-token suppression -- are applied
    per token in the same order the kernel writes them.
    """
    num_tokens, _vocab_size = logits.shape
    neg_inf = float("-inf")

    for t in range(num_tokens):
        r = int(expanded_idx_mapping[t].item())

        # Stage 1: allowed-token mask. Save the allowed slice, fill the
        # entire row with -inf, then restore the saved slice. This is
        # equivalent to the kernel's full-vocab LOGITS_BLOCK_SIZE loop
        # followed by a restore at the allowed indices.
        n_allowed = int(num_allowed_token_ids[r].item())
        if n_allowed > 0:
            allowed = allowed_token_ids[r, :n_allowed].long()
            saved = logits[t, allowed].clone()
            logits[t, :] = neg_inf
            logits[t, allowed] = saved

        # Stage 2: additive bias on the selected logit-bias token ids.
        # The kernel reads the post-stage-1 logits and adds bias in
        # place, so accumulating ``bias`` after the mask reproduces the
        # kernel's behaviour (allowed-but-biased tokens carry both the
        # original value and the bias; non-allowed-but-biased tokens
        # stay at -inf because -inf + bias = -inf in IEEE 754).
        n_bias = int(num_logit_bias[r].item())
        if n_bias > 0:
            tids = logit_bias_token_ids[r, :n_bias].long()
            logits[t, tids] = logits[t, tids] + logit_bias[r, :n_bias]

        # Stage 3: min-length stop-token suppression. When pos[t] is
        # below the request's min_lens, force every stop-token logit
        # back to -inf so the sampler cannot emit a stop token before
        # the minimum length is reached.
        n_stop = int(num_stop_token_ids[r].item())
        p = int(pos[t].item())
        m = int(min_lens[r].item())
        if n_stop > 0 and p < m:
            sids = stop_token_ids[r, :n_stop].long()
            logits[t, sids] = neg_inf

    return logits


def run(inputs: dict) -> dict:
    """Oracle entrypoint.

    The kernel returns None and mutates ``logits`` in place; the
    runner's None-fallback then surfaces EVERY bound tensor parameter
    as a named candidate output. To pass the anchor's structural arity
    check the oracle echoes the same 11 keys back: the rw_in_place
    ``logits`` carries the transformed values, and the 10 read-only
    tables are passed through unchanged (cloned so the oracle never
    mutates the caller's tensors -- the read-only tables MUST remain
    identical between candidate and oracle, and a shared reference
    would break that invariant if anything downstream wrote to them).
    """
    logits = inputs["logits"].clone()
    transformed = apply_logit_bias_oracle(
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
    return {
        "logits": transformed,
        "expanded_idx_mapping": inputs["expanded_idx_mapping"].clone(),
        "pos": inputs["pos"].clone(),
        "num_allowed_token_ids": inputs["num_allowed_token_ids"].clone(),
        "allowed_token_ids": inputs["allowed_token_ids"].clone(),
        "num_logit_bias": inputs["num_logit_bias"].clone(),
        "logit_bias_token_ids": inputs["logit_bias_token_ids"].clone(),
        "logit_bias": inputs["logit_bias"].clone(),
        "min_lens": inputs["min_lens"].clone(),
        "num_stop_token_ids": inputs["num_stop_token_ids"].clone(),
        "stop_token_ids": inputs["stop_token_ids"].clone(),
    }
