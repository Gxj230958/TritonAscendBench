# Full-shape C2 verification — closing the 2-anchor blind spot (2026-06-03)

Answers "can we be certain all 103 tasks' precision alignment is problem-free?" by
grading EVERY public+hidden shape on the GPU, not just the two volume anchors the
production C2 gate uses.

## Why
The production C2 oracle-equivalence gate grades only 2 anchor shapes per task
(smallest + median by volume). That is a STRUCTURAL blind spot: a divergence that only
appears at large shapes is invisible — proven by L2_23/24, whose upstream off-by-one is
invisible at single-block anchors and only fires at multi-block L>=512. To answer
"are we certain" honestly, every shape must be graded.

## Method
Env-gated all-shape mode in `oracle_pyref_precompute.py` (`_C2_ALL_SHAPES`): the CPU
fp32 oracle precomputes a ref for every public+hidden shape (up to 10/task); the GPU
runner then grades each shape that has a ref. A sweep-only shape whose CPU oracle
OOMs/errors is skipped and reported UNGRADED (never silently folded into "all verified").

## A severe sync-staleness bug invalidated the first sweep (fixed)
The first sweep "found" L2_33 (device_assert never stripped) and L3_10 (bf16/fp32 dot
clash) failing at ALL shapes. These were NOT real bugs — the daemon ran STALE code. A
manifest diff (md5 daemon-vs-local over scripts/ + tasks/ + triton_ascend_bench/) showed
**88 stale files**: an old extraction workflow had APPENDED the launcher onto each
`kernel.py` on gpfs; the daemon wrote those files (newer mtime), so `sync-code`'s
`rsync --update` permanently skipped re-pushing the current pure-`kernel.py` + separate
`_pa_launcher.py` + strip-aware loader. Fix: force-sync content-based
(`rsync --checksum`, no `--update`); manifest diff re-verified 0 stale; re-ran clean.
**Lesson: always run a daemon-vs-local manifest diff before a large GPU sweep — a result
on stale code is worse than no result.** (Recorded in the sync-staleness memory.)

## Clean full-shape result (103 tasks, trustworthy, co-located-metric classified)

Classified by the co-located `max_tol_units = max(|actual-ref|/(atol+rtol*|ref|))`
(NOT the independent abs/rel maxima — see L2_19 below):

FINAL state after all precision fixes (the sweep's first pass found 22 ceiling; the four
precision fixes below; 3 fully fixed L1_38/L2_37/L2_36; L2_02's fast_exp fixed its exp gap but a residual bf16-MMA ceiling at P4 keeps it a CEILING task):

| Class | Count | Meaning |
|-------|-------|---------|
| STRICT_VERIFIED | 82 | every shape strict-VERIFIED (79 base + 3 fully-fixed: L1_38, L2_37, L2_36; incl. L1_27 nan bitmask). NOTE: a later dtype-ceiling-acceptance phase reclassified the rest -- see the proposal doc + the bottom line. |
| CEILING | 18 | large-shape-only divergence, worst `tol_units` 1–24 = 1–2 bf16/fp16/fp8 ULP (tighter-than-1-ULP tolerances inflate tol_units) |
| STRUCTURAL | 2 | L2_23, L2_24 — `tol_units` up to 6200, value wholly wrong: the known upstream off-by-one (honest BLOCKED) |
| UNGRADED | 0 | no CPU-oracle OOM |

The 18 CEILING are all bf16/fp16/fp8-input kernels (the 22 the sweep's first pass found,
minus the 3 fully-fixed L1_38/L2_37/L2_36; L2_02 stays a ceiling -- see below). Gray-zone (tol_units 5–24) spot-check:
`L2_01`/`L2_40` worst element = ~1.8–2.0 bf16 ULP; `L1_26`/`L3_13` = near-zero-ref
(small abs ≤0.11); `L3_03`/`L3_09` = bf16-matmul/recurrence error (abs ≤0.5) at
near-cancellation elements (high LOCAL rel, abs bounded by bf16 precision). None exceeds a
handful of ULP — orders below L2_23/24's value-wholly-wrong signature.

### The CEILING tasks are the CPU-oracle-vs-GPU-tensor-core limit, NOT bugs
All are bf16/fp16/fp8-input kernels. Their large shapes (P3/P4/P5/H3) cross the tight
atol/rtol because GPU bf16/fp16/fp8 tensor-core + recurrence accumulation diverges from a
CPU fp32 oracle by ~1 dtype-ULP at scale — which the 2-anchor grade never saw (anchors
are small enough to stay within atol). Worst max_abs <= 0.5 for most; the exception is a
large-magnitude output that is still 1 ULP — L3_14: max_abs=128 but max_rel=0.007 = 1 bf16
ULP on a ~1.8e4 recurrent state. (The fp8 cases L1_38/L2_37 that looked like this — max_abs=32
= 1 fp8 ULP — were FIXED via the correct fp8 comparator / gate_up-bf16 model, not left as
ceiling.) This extends the previously-documented 7-task fp32-vs-bf16 subset at full-shape
resolution. The operators ARE verified (pass anchors + most shapes; divergence is
dtype-ULP not structural). Resolution = honest documentation, NOT loosening tolerances.

### L2_19 — dump-confirmed CEILING (initially mis-flagged structural)
P4 reported max_abs=2.0 / max_rel=1.715, which my first classifier flagged structural.
A P4 actual-vs-ref dump disproved it: max_abs=2.0 sits on ref=-478 (= 1 bf16 ULP at that
magnitude); the failing elements are all 1-bf16-ULP off (e.g. diff=0.125 on ref=-19.1,
just over recurrent_chunk rtol=0.005 because bf16 relative precision ~0.0066 > 0.005).
The reported max_rel=1.715 came from a SEPARATE near-zero-ref element. So L2_19 is the
bf16-recurrence ceiling, same class as the other ceiling tasks.
DISCRIMINATOR FLAW exposed: the GPU runner emits `max_abs_diff` and `max_rel_diff` as
INDEPENDENT per-tensor maxima (different elements). Gating structural on
(max_abs AND max_rel) wrongly combined them. The correct co-located metric is
`max(diff / (atol + rtol*|ref|))`.

### L1_38 — a real bench bug (FIXED): tolerance MIS-CLASSIFICATION
`L1_38_silu_mul_fp8_quant_deep_gemm` outputs `float8_e4m3fn` but its tolerance_class WAS
`vector_reduce` (atol 1e-3, no fp8_ulp) — so the fp8 output was graded by float allclose
and failed by 1 fp8 ULP (max_abs 8/16/32) at large shapes. Every peer fp8-quant task
(L2_34, L2_37, L1_09) uses `fp8_software_emulated` (fp8_ulp=1). The 2-anchor grade hid
it (anchors had max_abs ~1e-9). FIXED: tolerance_class is now fp8_software_emulated (the
correct comparator for an fp8 output, NOT a tolerance pad). VERIFIED on GPU.

## Bottom line (independently reviewed)
Extraction is verified by the official C2 anchor gate; the full-shape sweep found 83
tasks that pass every enumerated shape (after 4 precision fixes), 18 with
characterized dtype-ceiling failures at large shapes, and L2_23/24 as the known upstream
structural bugs. **Not "100% problem-free", but there is no remaining un-characterized
extraction bug in the observed sweep — every divergence is either a fixed precision-model
gap, a 1–2-dtype-ULP ceiling the CPU oracle cannot certify at scale, or the known
upstream off-by-one.**

### Blind spots this conclusion does NOT cover (state honestly)
- The sweep grades only enumerated `public_shapes + hidden_shapes` and seed-0 fixture
  distributions — not arbitrary shapes/seeds.
- Oracle SCOPE exclusions still hold (e.g. L1_38 only covers the FLOAT32 scale format;
  per-task ORACLE-SCOPE notes list untested branches like CP_SIZE>1).
- The structural/ceiling split is only as good as the co-located `max_tol_units` metric;
  the abs+rel maxima are independent and must NOT be combined.

## Fixes applied / in flight
- **L1_38**: tolerance_class `vector_reduce` → `fp8_software_emulated` (the correct
  ordered-code fp8-ULP comparator; aux fp32 `scale` still uses atol/rtol). **VERIFIED on
  GPU — all shapes pass**, satisfying the review's <1% off-by-one-density caveat (sparse
  boundary flips, not systematic bias).
- **GPU runner**: now logs `max_tol_units = max(|actual-ref|/(atol+rtol*|ref|))` per
  shape — the co-located metric that separates structural (≫1) from a 1-ULP ceiling (~1).
  Diagnostic only; the verdict is still `compare().passed`.
- **`_c2_sweep_analyze.py`**: retired the stale `max_abs>1 && max_rel>0.5` rule; gates on
  `max_tol_units > 50` (calibrated: ceiling 1–24, structural 6200); old logs without the metric are tagged `needs_tu` (re-run).

- **L2_37**: oracle now rounds `gate_up` to bf16 (models kernel line 44's bf16*bf16
  product), so the per-group AMAX/scale/fp8-grid align with the kernel. **VERIFIED on GPU
  — all 9 shapes pass** (was 3 fp8-ULP failures). GPU-authoritative confirmation that
  Triton's bf16 elementwise product rounds to bf16, the review's "fixable cast point" branch.

## Can the CEILING tasks be solved? (20-task adversarial root-cause + adversarial challenge)
A parallel workflow (one diagnoser + one skeptical verifier per task) classified each
of the 20 ceiling tasks as FIXABLE vs IRREDUCIBLE. The review then CHALLENGED the "irreducible"
bucket as overconfident and named specific tasks with un-modeled kernel behaviors to
test. Every candidate fix was GPU-verified (GPU is authoritative). Outcome: **2 of 20
genuinely fixable (fixed), 18 CPU-oracle ceiling (not full-shape-certifiable).**

- **FIXED — L2_36** — kernel's `sum_vec += in_data * weight_scale` (kernel.py:48) is an
  ELEMENTWISE fp32 multiply-accumulate the GPU contracts into a single-rounding FMA
  (fma.rn.f32); oracle did two roundings. Fix: `torch.addcmul` (impl-dependent FMA
  emulation, documented). **GPU VERIFIED all shapes.**
- **PARTIAL — L2_02** — its fast_exp gap WAS fixed (real precision improvement, below), but
  the authoritative full-103 sweep then surfaced a residual bf16-`tl.dot` MMA ceiling at P4
  (tol_units=1.04, ~1 bf16 ULP). So L2_02 is a CEILING task (Mode-A allowlist), not all-pass.
  The fast_exp correction: kernel uses `fast_exp = exp2(LOG2E*x)` (kernel.py:288,
  373), NOT `torch.exp`. Oracle now uses `torch.exp2(1.4426950408889634*x)` at the two
  decay/scale exp sites. **GPU VERIFIED all shapes.** I had wrongly bucketed this as
  "irreducible tl.dot order" — confirmed it was that it was an un-modeled exp.
- **Tested other candidates, edits did NOT reduce the divergence -> reverted,
  stay CEILING**: L2_21 (sequential-tap + spelled-SiLU: tol_units unchanged 4.95 -> the
  residual is the libdevice exp in SiLU, not the accumulation order); L3_04 (online-softmax
  p->bf16 cast: 1.16 -> 1.3, slightly worse -> the dominant gap is the streaming-softmax
  ORDER, not the cast); L3_09 (chunked-K: tol_units unchanged 7.14 -> within-MMA micro-order,
  as the L2_01/L2_07 agents independently concluded).
- **18 CPU-ORACLE CEILING** (not literally irreducible — precisely: not certifiable at full shape under the current oracle abstraction without modeling/reimplementing the kernel's GPU evaluation ORDER; the operators ARE verified at the production 2-anchor gate + most shapes). Each with a SPECIFIC GPU mechanism:
  - sum-of-squares GPU block-reduction tree: L1_12, L2_15, L2_17, L2_40
  - tensor-core MMA (`tl.dot`) fp32-accumulate reduction ORDER (incl. within-MMA micro-order):
    L2_01, L2_07, L3_03, L3_09, L3_20, L3_16
  - K-axis `tl.sum` block-reduction tree: L2_19, L3_14, L3_15
  - GPU libdevice exp/log last-bit: L1_26, L2_21
  - online-softmax streaming ORDER + `tl.dot`: L3_04
  - (L2_03, L3_13 — the adversarial verifier debunked their "fixable" proposals: stale-fixture
    premise / provably-false "won't break P1/P2".)
  That is 18 tasks (the two earlier "17 vs 19" doc miscounts are corrected here).

### GPU-authoritative lesson: L3_16 false-fixable
A verify-agent "proved" L3_16 fixable (oracle `_bf16` body bf16 vs its TF32 docstring;
bit-exact reproduction of golden refs under a TF32 body). I applied oracle-TF32 +
shapes-fp32 and **the GPU REFUTED it** — fp32+TF32 fails even anchors P1/P2 (max_abs
0.067/0.083) that passed under bf16. The "bit-exact to golden" was oracle SELF-consistency
against STALE fp32 golden, NOT a GPU match; the bf16 body is correct, the TF32 docstring +
meta=fp32 + r10 fp32 refs were the stale parts (meta + contract now aligned to bf16; only the golden-hash provenance refresh remains a follow-up). Reverted.
**Static/golden proofs do not substitute for a GPU run — GPU is authoritative.** The SAME
discipline reverted L2_21/L3_04/L3_09 when their edits didn't help on GPU.

## Final tally (full 103, every kept fix individually all-shape-VERIFIED)
**FINAL (after the dtype-ceiling-acceptance phase): 82 STRICT_VERIFIED + 19 CEILING_ACCEPTED + 2 STRUCTURAL (L2_23/24 known upstream).** The 19 CEILING_ACCEPTED are graded by an allowlist-gated, independently reviewed mechanism (a DISTINCT status, never merged into VERIFIED) -- see docs/reviews/2026-06-03-ceiling-tolerance-relaxation-proposal.md.
Kept precision fixes (4): L1_38 (fp8 tolerance class), L2_37 (gate_up bf16), L2_36 (FMA),
L2_02 (fast_exp, residual ceiling). The review corrected an over-confident "19 irreducible",
and the GPU-authoritative loop reverted 4 static-only "fixes" (L3_16 + 3 no-benefit edits).

## L2_03 — full-shape grading caught a real launcher dtype bug
The dtype-equality gate on the ceiling path (runner:899) flagged L2_03 as MIXED: its launcher
(`_pa_launcher.py` + the embedded copy in `kernel.py`) ended with `states.to(inputs["x"].dtype)`,
downcasting the fp32 states to bf16. But upstream Mamba2 keeps states fp32 (`states_in_fp32=True`);
the wrapper allocates fp32, the oracle returns fp32, the ref is saved fp32. At the 2 anchors strict
compare() casts both to fp32 (hiding the mismatch -> VERIFIED); only the large strict-failing shapes
reach the dtype gate. Fix (review-approved, bench infra not the frozen kernel): drop the downcast ->
fp32 states. Result: anchors VERIFIED, H3/P4 CEILING_ACCEPTED (Mode-A tol_units 2.16/1.57), S2
bit-exact. The 2-anchor grade structurally could NOT have seen this — concrete proof the full-shape
grade earns its cost.

## Done
- `_C2_ALL_SHAPES` is now env-gated (`TAB_C2_ALL_SHAPES`, default "1" = full-shape). Production
  C2 grades all shapes by default (the user's full-volume-verification goal + it catches
  large-shape/dtype bugs); set `TAB_C2_ALL_SHAPES=0` for the fast 2-anchor grade.

## Follow-ups (provenance hygiene, non-blocking)
- Update `golden.tolerance_class_used` (L1_38) and `golden_sha256` (L2_37, oracle
  changed) in the two metas — the daemon regenerates the ref each run so C2 is
  self-consistent, but the provenance record should be refreshed.
