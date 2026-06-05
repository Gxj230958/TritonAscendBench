# Deprecated tasks (excluded from all metrics)

Tasks moved here leave `tasks/level{1,2,3}/`, so `iter_all_tasks()` -- which
skips every `_`-prefixed directory -- never includes them in any metric
(C2 oracle-equivalence / cost_model / release gate). They are NOT silently
deleted: the directory, git history, and proof artifacts are kept for
transparency. Frozen kernel bytes are unchanged.

## L2_23_fwd_kv_parallel, L2_24_fwd_kv_reduce — upstream off-by-one (moved 2026-06-05)

The vendored frozen kernel carries a REAL upstream off-by-one bug (vLLM
`lightning_attn.py:444`, cross-block decay `array = arange + 1`). A correct
oracle therefore necessarily diverges (~80 dtype-ULP, far above tolerance), so
both graded `BLOCKED_UPSTREAM_BUG`. Grading a kernel whose upstream source is
itself buggy is not meaningful, so they are excluded from the scored
denominator (correctness 103 -> 101; this does not change the release gate,
which already excluded them).

Evidence (kept in-repo):
- `docs/verification/L2_24_lightning_attn_offbyone_proof.py` — numeric proof (0.28 -> 0.00 once the `+1` is removed)
- `tasks/_corrected/{L2_23,L2_24}_*_fixed/` — de-bugged variants that GPU bit-exact PASS
- `SCORE.md` — the off-by-one writeup
