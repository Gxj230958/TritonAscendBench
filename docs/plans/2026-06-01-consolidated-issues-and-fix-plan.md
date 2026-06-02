# Consolidated issue inventory + unified fix plan (TAB)

**Status: PLAN ONLY. No task modified, no GPU run. Date 2026-06-01.**
Sources: this session's audits — `docs/reviews/2026-06-01-dtype-per-task-audit.md`,
`docs/plans/2026-06-01-dtype-alignment-OVERALL.md`, codex reviews.

## Part 1 — Problem inventory (by severity)

### P0 — Foundational verification gaps (affect whether results MEAN anything)

- **P0.1 Oracle correctness not systematically proven.** 85/103 tasks use an
  independent PyTorch oracle as the precision reference (only 18 use an
  upstream-kernel golden). If an oracle is wrong, that task's PASS is meaningless
  (kernel "matches" a wrong reference) or a false FAIL. To date oracles are
  spot-checked + codex-reviewed on a few, NOT all 85 cross-checked against an
  independent third source. This underpins the validity of the headline number.
  Evidence the method works: spot-checks show our `silu_and_mul` / `rms_norm`
  oracles match vLLM/SGLang's OWN test references; L2_23/24 oracle-vs-kernel
  disagreement found a real upstream bug.

### P1 — Incomplete work

- **P1.1 The 3 new tasks (L2_39 scaled_mm, L2_40 gemma_qkv_rmsnorm,
  L2_41 selective_state_update) are NOT GPU-verified.** Authored + frozen +
  schema-validated + oracle-smoke + severance-clean locally; still need daemon
  S2 (bit-exact + oracle anchor) + faithfulness audit. (catalogue is 103.)

### P2 — Fidelity gaps (real, but NOT correctness bugs; payoff at the 910B NPU stage)

- **P2.1 dtype misalignment.** 71/103 fixtures are fp32 where production is
  bf16/fp16. fp32 already verifies GPU extraction + logic (audit conclusion); the
  gap is that the GPU run does not exercise the production-dtype path, which matters
  only when the 910B stage compares NPU output against production-dtype goldens.
  Split by reference: **18 upstream-golden** (fixture→prod-dtype + `--force`
  recapture = direct "align with vLLM/SGLang") vs **55 oracle** (need
  production-track machinery).
- **P2.2 Infra needed for full dtype alignment.** (a) no dual-fixture support — one
  `shapes.yaml` drives one dtype per eval, so keeping the fp32 logic gate AND adding
  a bf16 production gate needs new capability; (b) no ULP comparison mode in
  `tolerance.py`; (c) golden recapture needs `--force` after dtype change;
  (d) some kernels (AWQ L2_12) accumulate `tl.dot` in the OUTPUT dtype, so the
  "bf16 inputs are gate-safe" shortcut is not universal — per-task check needed.

### P3 — Minor / cosmetic

- **P3.1 Metadata drift.** `meta.shape_coverage: public:3/hidden:1` is stale — the
  grader (`_shape_ids_for`) runs ALL 9 shapes; the field is unused by grading. Also
  the audit's "in-place norms" label should read "output-dtype-tied".

### Out of scope (already decided — NOT "to fix")

- The 3 BLOCKED tasks (L2_23/24 vLLM lightning_attn off-by-one; L2_33 sglang
  Triton-unsupported) are proven UPSTREAM issues, honestly BLOCKED under the
  DUAL_METRIC policy (`SCORE.md`). Not reopened here.

## Part 2 — Priority, dependencies, GPU-need

| id | severity | local or GPU | depends on |
|---|---|---|---|
| P0.1 oracle validation | **highest** | local (A-layer) + GPU (B-layer for no-ref) | — |
| P1.1 GPU-verify 3 new | high | **GPU** | — |
| P2.1 dtype align (18 golden) | medium | local edit + **GPU recapture** | P0.1 (oracle sanity) |
| P2.2 infra (dual-fixture+ULP) | medium | local | — (enabler for 55-oracle track) |
| P2.1 dtype align (55 oracle) | medium | local + **GPU** | P2.2 |
| P3.1 metadata | low | local | — |

## Part 3 — Unified fix plan (phased; local-first, GPU-last, conservative)

### Phase 0 — Validate the 85 oracles (foundational; do FIRST)

The most important phase: turn "trust my oracle" into "85 tasks each have a third
-source check." Per oracle task:
1. find the upstream's OWN reference implementation (in `infer/{vllm,sglang}` tests
   or `forward_native`-style methods) — the A-layer gold standard;
2. compare its math to our oracle → **MATCH / MISMATCH / NO-UPSTREAM-REF**;
3. for NO-UPSTREAM-REF: fall back to B-layer (GPU: run frozen kernel, confirm oracle
   agrees within tolerance) and C-layer (codex re-derive from spec);
4. every MISMATCH is escalated (our oracle bug OR an upstream bug like L2_23/24).
Mostly LOCAL (A/C); only NO-REF tasks need a GPU cross-check batch. codex reviews all
MISMATCH + a sample of MATCH. Output: an 85-row oracle-validation table.

### Phase 1 — GPU-verify the 3 new tasks (parallel with Phase 0)

Submit L2_39/40/41 to the daemon (sync-code → submit batch → wait → collect); confirm
S2 bit-exact + oracle anchor PASS; faithfulness-audit each; update SCORE.md/catalogue.
Needs GPU; independent of the rest.

### Phase 2 — dtype alignment (after Phase 0; the fidelity-to-production work)

1. **2.A — 71-row production-dtype mapping** (the "find"): per tensor, fix the real
   production dtype with evidence (kernel/wrapper assert > upstream test > call-site
   trace > role); conservative — unconfirmed → leave fp32. Handle AWQ-style
   `tl.dot`-accumulates-in-output-dtype cases.
2. **2.B — the 18 upstream-golden tasks**: set fixture to prod dtype, `--force`
   recapture the golden on GPU → golden = original vLLM/SGLang kernel's real
   production-dtype output (the direct "align with vLLM/SGLang"). No dual-fixture
   infra needed.
3. **2.C — infra**: dual-fixture support + a `max_ulps` comparison mode + tests
   (keep the fp32 logic gate AND add a production-dtype gate).
4. **2.D — the 55 oracle + 7 output-tied tasks**: set fixture to prod dtype; oracle
   returns `fp32_result.to(out_dtype)`; compare at the 1-ULP policy; GPU re-grade.

### Phase 3 — metadata cleanup (anytime, local)

Fix/retire stale `shape_coverage`; correct labels. Trivial.

## Part 4 — Effort + honest framing

- **Phase 0** (oracle validation): ~half–1 day (mostly local A/C; small GPU batch for
  NO-REF). **Highest value: certifies the existing 97 PASSes are meaningful.**
- **Phase 1** (3 new tasks GPU): 1 daemon batch + audit, ~half day.
- **Phase 2** dtype alignment:
  - 2.A mapping ~half day · 2.B 18 golden ~half day + 1 GPU recapture ·
    2.C infra ~half day · 2.D 55+7 oracle track: multi-session + multiple GPU iterations.
- **Phase 3**: ~1h.

Honest framing: Phase 0 + Phase 1 are the do-now items (validity + finish the new
tasks). **Phase 2 (dtype alignment) is the fidelity-to-production work — sequenced
SECOND, after the oracles it aligns to are validated (Phase 0).** None of the 71 is a
wrong PASS today (fp32 already verifies GPU extraction + logic); dtype alignment makes
the golden/oracle comparison happen at the real bf16/fp16, the prerequisite for a
meaningful 910B NPU-migration comparison.

**Recommended order: Phase 0 + Phase 1 now → Phase 2 (2.A→2.B→2.C→2.D) → Phase 3.**

**No code changed; no run started. Next: codex review of THIS consolidated plan, then
execute Phase 0 (local) + Phase 1 (GPU) on your go-ahead.**
