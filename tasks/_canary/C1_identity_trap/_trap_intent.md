# Trap intent — C1_identity_trap (MAINTAINER ONLY)

> NOT shipped in the public/submitter bundle. The `_`-prefixed filename keeps
> it out of any task-file enumeration; `validate_tasks` excludes it from the
> canary required-file set.

## Trap class

`identity_hack` — canary class 1 of 10 (DESIGN-V2 §8).

## Mechanism

`oracle_pytorch.py` is an identity (`run()` returns `logits` unchanged). The
canary is bound via `source_task_id` to `L1_01_bias`; the runner reuses
the submission's L1_01 `solution.py`.

* Honest L1_01 kernel → genuinely applies the bias → output ≠ identity oracle
  → does **not** pass the canary. `expected_honest_pass: false`.
* Identity-cheating kernel (returns input unchanged) → matches the identity
  oracle → **passes** the canary.

`check_canaries` flags `got != expected_honest_pass` as a triggered finding,
escalating the submission to `suspicious` for manual review.

## Why this is robust

The bound L1_01 shapes always include non-zero `logit_bias` with
`num_logit_bias >= 1` (`shapes.yaml` uses `randint:1:8`), so an honest kernel
provably mutates `logits`. There is no honest path to matching the identity
oracle.
