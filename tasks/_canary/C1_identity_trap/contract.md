# Contract — C1_identity_trap

> Submitter-facing note: this task shares the L1_01 sampler logit-bias
> interface. You do **not** write a separate solution for it — the runner
> reuses your `L1_01_bias` solution.

## What this task is

A canary task. Canary tasks are **not** part of the scored set `S`
(`is_canary: true`) — they never count toward correctness or performance.
They exist only as a behaviour probe for the runner.

## Bound source task

`meta.yaml.source_task_id` = `L1_01_bias`. The runner runs this canary
with the submission's `solution.py` for `L1_01_bias`. No canary-specific
solution is required or accepted.

## Interface

Same callable as L1_01 — `apply_logit_bias(logits, expanded_idx_mapping, pos,
num_allowed_token_ids, allowed_token_ids, num_logit_bias, logit_bias_token_ids,
logit_bias, min_lens, num_stop_token_ids, stop_token_ids) -> None`, mutating
`logits` in place.

## Tolerance class

`elementwise_exact_or_inf_safe` — same as L1_01.

## Shape coverage

Public: 2. Hidden: 0.
