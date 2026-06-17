---
date: 2026-06-17
topic: shape-definition-approach
---

# How TAB should define operator shapes (vs industry prior art)

## Problem Frame

Our `shapes.yaml` parameterises every operator by a generic single-letter **knob grid**
(`B`/`L`/`D`/`N`/`T`/`H`/`E`/`C`/`G`/`Dv`…). A reader cannot tell what a knob means or which ones an
operator actually uses (median **60%** of a task's declared knobs are dead template boilerplate, and
the meaning lives only implicitly in `inputs[].dims`). The narrow question was "document/clean the
knobs"; the real question is **whether the knob-grid shape model itself is the right design**, judged
against how established kernel benchmarks define shapes. This affects every consumer (`build_inputs`,
manifest hash, `param_constraints`, `cost_model`, generation) so it is an architecture decision, not
a cosmetic one.

## Prior art (researched)

| Benchmark | How shapes are defined | Self-documenting? | Sweep | Cross-tensor constraints / analytic cost |
|---|---|---|---|---|
| **NPUKernelBench** (local, we already borrow geometry from it) | per-kernel JSON, one line per case = explicit **named tensors** with literal shapes (`x:[1,128,4096]`, `normalized_shape:[4096]`), real LLM dims × seqlen × dtype | yes | enumerated real dims | no |
| **KernelBench** (Stanford) | Python `get_inputs()`/`get_init_inputs()` returning **concrete tensors**; **one fixed shape per problem** | yes | no (one size/problem, by design) | no |
| **TritonBench** (Meta) | `get_input_iter()` iterator; shape families + cross-tensor coupling done **in Python** (verified: flash_attention/decoding_attention operators) | yes | yes (seqlen sweeps, ragged) | imperative (Python), not declarative |
| **SOL-ExecBench** (NVIDIA, 2026) | 235 problems = **subgraphs extracted from 124 real models**; graded vs analytic **Speed-of-Light roofline** bounds | yes (real provenance) | real workloads | SOL bound (roofline) |
| **TAB (ours)** | generic **knob-grid** parameterisation | **no** (cryptic + template-bloated) | yes (single-axis ladder + grid) | **declarative** (`param_constraints` predicates + `cost_model` formulas) |

**Findings (corrected after codex supervision — earlier draft over-claimed in our favour):**
1. The industry norm is **explicit, named, self-documenting** shapes. No established benchmark uses a
   generic single-letter knob grid like ours.
2. **Explicit shapes do NOT preclude constraints/cost/sweep** — they just need a generator/linter
   layer. TritonBench sweeps and couples tensors in Python; KernelBench uses *fixed concrete cases
   with no within-problem sweep* (a design choice, not an inability). So the honest reason to keep
   knobs is **compatibility with our current manifest/scoring/oracle consumers, not that knobs are
   intrinsically superior.** Our genuine differentiator is only that constraints/cost are
   **declarative** (in YAML) vs imperative (in Python) — a modest, not decisive, edge.
3. SOL-ExecBench grades vs a roofline SOL bound; we have roofline *code* (`gpu-rank`) but per
   `README.md` performance is still deferred and roofline efficiency is **not wired end-to-end on
   910B** — so we are aligned in **aspiration/prototype, not shipped scoring.** (Do not over-claim.)
4. **Provenance gap.** NPUKernelBench/SOL source shapes from **real model configs**; our shapes are
   mostly **synthetic ladders** (`extend_task_shapes` pow2 + LLM-ish dims) with only ~2 tasks
   actually borrowing NPUKernelBench geometry. We are weaker on real-model provenance and currently
   record no per-row provenance at all.

## The real tension (reframed)

The principled answer is the industry one: **explicit, self-documenting, per-op, provenance-tagged
shapes.** Knobs are not the enemy of constraints/cost/sweep, but our knob grid is unreadable. So the
decision is really: **migrate toward explicit-first shapes, and how far** — full rewrite (B) vs a
staged hybrid that becomes explicit-first over time (C). "Keep knobs because they're better" is
rejected as self-serving.

## Directions

| Dir | Description | Pros | Cons |
|---|---|---|---|
| **A. Patch knobs** | glossary + structured `knob_doc:` + prune dead knobs (the earlier proposal) | cheapest; keeps engine | still single-letter knobs; surface stays knob-first |
| **B. Adopt explicit per-tensor shapes** | move to NPUKernelBench/KernelBench-style named literal shapes | matches industry; maximally readable | loses constraints / analytic cost / single-axis sweep; full rewrite + re-grade |
| **C. Explicit-first hybrid (recommended *only with the invariants below*)** | keep knob engine underneath for consumer compatibility, but make the surface explicit-first: lean active knobs + a **generated materialised explicit-shape view** + provenance tags | self-documenting like peers *and* keeps declarative constraints/cost/sweep; stageable cheap→deep | needs a generator/linter; **degenerate risk** — if it ends up as just `knob_doc`+comments on the bloated template it has NOT fixed the root cause and B becomes the honest choice |

**Hard invariants for C to count as a real fix (codex):** (1) **no dead public knobs** — a public
row's knob set == its referenced set, enforced by a validator; (2) a **materialised concrete-shape
view** per row, generated faithfully from `inputs[]` (must scan `dims`+`dist`+`derive`+`scalar`, not
just `dims`); (3) **hidden rows stay sealed** (no leak into the candidate sandbox view); (4) an
**explicit dtype policy**; (5) **per-row provenance** (synthetic / NPUKB-borrowed / model-sourced /
edge-case / hidden).

## Recommendation

**Direction C, but framed honestly as a staged migration toward explicit-first — not as "knobs are
better".** Phase 1 (no re-grade): central glossary + structured `knob_doc` + a generated materialised
explicit-shape view + per-row provenance tags + the dead-public-knob validator. Phase 2 (gated on a
planned re-grade): prune dead knobs and/or descriptive renames, moving the *authoritative* surface
toward explicit. If we are unwilling to commit to the explicit-first invariants, the evidence favours
**Direction B** (a clean explicit-shape format with a thin generator that re-derives constraints/cost)
over a cosmetic knob patch. The root cause is the **global template**, not missing docs — Direction A
alone only decorates it.

## Decision

**Direction B chosen** (user, 2026-06-17): migrate to an explicit-shape authoring surface, with a
thin generator/linter that re-derives the constraint/cost/sweep capability so nothing is lost. The
knob grid is retired as the *authoring* surface; whatever knob-derived machinery remains internally
must be justified by consumer compatibility, not preserved by default.

## Requirements (Direction B)

**Authoring surface**
- R1. The authoritative shape artifact is **explicit per-tensor rows**: each input tensor listed with
  its literal `shape: [...]`, `dtype`, and human-named axes — self-documenting with no external
  decoder (NPUKernelBench/KernelBench parity).
- R2. **Named axes** (`batch`, `seqlen`, `head_dim`, …) replace single-letter knobs; **no global
  template** — each operator declares only the tensors/axes it actually uses (kills the 60%-dead-knob
  problem at the root).

**Preserve the engine via a generator/linter (so B loses nothing)**
- R3. A thin **per-op generator** emits the explicit rows (the single-axis ladder + coprime-grid +
  edge sweeps we built becomes generation, not hand-authoring); deterministic, regenerable.
- R4. A **linter** enforces cross-tensor consistency that knobs gave implicitly (e.g. q/k/v share
  `head_dim`) over the explicit rows.
- R5. **`param_constraints` re-expressed** over named axes; **`cost_model` formulas re-based** on
  named axes (no knob basis). This is the deepest migration item — verify every task.

**Honesty / provenance**
- R6. Each row carries **provenance** (synthetic / NPUKB-borrowed / model-sourced / edge-case /
  hidden); "real-model-sourced" becomes a tracked, improvable coverage metric, not an asserted claim.
- R7. **Hidden rows stay sealed** from the candidate sandbox view; **dtype policy explicit** per row.

**Migration**
- R8. Convert all 101 tasks knob-grid → explicit; this changes the manifest, so a **full
  S2/C2/golden/roofline re-grade** is required — staged so the corpus stays green per batch.
- R9. **Down-payment (optional, no re-grade):** the generated explicit-shape *view* + provenance tags
  can land first as a non-authoritative annotation before the authoritative cutover, giving the
  readability win immediately while R5/R8 are planned.

## Success Criteria
- A newcomer reads any row and knows what each dimension is **without** an external decoder.
- Zero loss of `param_constraints`, `cost_model`, single-axis sweep, or hidden-row sealing after
  migration (proven by a green full regression on the new format).
- Every row carries honest provenance; real-model-sourcing is measured.

## Scope Boundaries
- Not changing the grading pipeline (S2/C2/golden/roofline) — only the shape *authoring surface*.
- Not adopting SOL-ExecBench's model-subgraph extraction (we are operator-level by design); only
  borrowing its "real-model-sourced + roofline-graded" principle, which we largely already follow.

## Outstanding Questions

### Resolve Before Planning
- *(resolved)* Direction = **B** (explicit shapes).

### Deferred to Planning
- [Affects R5][Technical — load-bearing] How are `cost_model` formulas re-based off knobs onto named
  per-tensor axes for all 101 tasks, and how is each re-based formula re-validated? This is the
  single biggest migration risk.
- [Affects R3/R4][Technical] What is the minimal per-op spec the generator consumes, and how does the
  linter encode cross-tensor consistency previously implicit in shared knobs?
- [Affects R1][Needs research] Faithful materialisation must scan `dims` + `dist` + `derive` +
  `scalar` (the active-knob detector currently misses `dist` — fix first).
- [Affects R8][Technical] Re-grade sequencing so the corpus stays green per batch during the cutover
  (reuse the S2→quarantine→C2/golden→roofline pipeline already built).
- [Technical] Where the authoritative explicit rows live (`shapes.yaml` replacement vs new file) and
  the back-compat window for the old knob format.

## Next Steps
→ `/ce:plan` for structured implementation planning (direction resolved; load-bearing unknowns are
technical and belong in planning).
