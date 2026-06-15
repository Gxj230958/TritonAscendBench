---
title: "feat: broaden TAB shape coverage (mirror NPUKernelBench breadth)"
type: feat
status: active
date: 2026-06-15
revised: 2026-06-15
---

# feat: broaden TAB shape coverage (mirror NPUKernelBench breadth)

> **Revision note (post codex gpt-5.5:xhigh review, verdict PLAN-FLAWED → revised):** the first
> draft made three unverified assumptions that independent checks then refuted — (a) the verify
> pipeline is NOT fail-closed on new shapes (silent-skip path), (b) the universal generator is NOT
> the source of truth (current shapes are family-specific; running it would CLOBBER verified
> shapes), (c) the "61 dtype-locked / 22 tight" feasibility numbers were derived from `meta.dtype`
> not the kernel and are wrong (e.g. L2_20 meta says fp32 but the kernel has bf16/fp16/fp32 configs).
> This revision fixes the approach (per-task, not universal generator), makes the pipeline
> fail-closed + coverage-gated, and replaces asserted numbers with a reproducible feasibility report.

## Overview

Each TAB task currently exercises ~9 shapes (5 public P1–P5 + 4 hidden H1–H4, **not uniform** —
some tasks have fewer; corpus total ≈ 905 shape points) at **one fixed dtype** in
`tasks/<T>/shapes.yaml`. The reference bench `AscendOpGenAgent/benchmarks/NPUKernelBench` exercises
~50–60 cases/op across 3 dtypes (fp16/fp32/bf16) with realistic LLM dims, odd non-power-of-2 sizes,
edge cases, and swept sequence length. TAB under-samples the shape/dtype space, so a kernel correct
only on its 9 shapes can pass. This plan broadens TAB's per-task shape sets (and, where the kernel
truly supports it, dtype) **without changing any frozen kernel, without clobbering the verified
shapes, and only after the verification pipeline is made fail-closed so a new shape cannot be
silently ungraded.**

## Problem Frame

TAB is NOT NPUKernelBench: its kernels are frozen upstream snapshots with hard dtype/shape
constraints, and every shape drives a verified `r10_pyref` → S2 → C2 → golden pipeline. The first
draft assumed "add a shape and it just works" — false. Two structural facts must be fixed FIRST,
or broadening is unsafe:

1. **Silent-skip (anti-self-deception, the core risk).** `scripts/oracle/precompute_oracle_reference.py`
   skips a sweep-extra shape whose `build_inputs`/`mod.run` raises (`continue` at the `_extra`
   branches), and `scripts/oracle/run_oracle_equivalence_gpu.py` grades **only the
   `ref_*.safetensors` that exist** (glob), not `shapes.yaml`. So adding `P6` whose ref was never
   generated → C2 grades P1–P5 only → P6 is silently ungraded yet the task still "passes."
2. **No shape-set freshness.** `phase_a_prime` records only a kernel-hash; there is no shape-set
   hash. A consistency checker (`validate_corpus_consistency.py`: `equivalence_per_shape` ==
   `shapes.yaml`) exists but the release gate does not call it. So expanding `shapes.yaml` leaves
   the old PASS metadata as **stale proof** over the new shapes unless S2/C2 are re-run and
   coverage-gated.

## Requirements Trace

- **R1.** Increase shapes/task (denser sampling + realistic LLM dims + edge cases) with **every new
  shape valid under ALL of the task's constraints** (`param_constraints.dims/dtypes/dists/scalars/
  tensor_invariants` + launcher asserts) — verified by an actual build_inputs + oracle dry-run, not
  just a dims predicate.
- **R2.** dtype sweep ONLY where the **kernel** supports it (read its tl dtype configs, not
  `meta.dtype`), with a parametric oracle + contract tolerance.
- **R3.** **Preserve existing P1–P5/H1–H4 byte-for-byte** (append-only). Never run the universal
  generator over the corpus as the expansion authority — it would clobber the family-specific
  verified shapes.
- **R4.** Make the pipeline **fail-closed + coverage-gated** BEFORE adding shapes: a declared shape
  that can't build/oracle is an ERROR (not skipped); the release gate asserts declared shapes ==
  graded refs == C2 verdicts for ALL tasks (and == goldens only for `upstream_oracle_required`
  tasks); shape-manifest changes force S2/C2 re-run.
- **R5.** No `kernel.py` byte change. The 27 just-cleaned kernels stay frozen; the 101-task
  release-gate-green state is re-established over the expanded shape set, not assumed.
- **R6.** Honest, reproducible reporting: feasibility comes from a checked-in script that reads
  kernel + constraints, not from hand-asserted numbers; coverage claims state real per-tier counts.

## Scope Boundaries

- **Not** changing any `kernel.py`. Only per-task `shapes.yaml` (append), `meta.yaml`
  (`runner_checks.shape_coverage`, `equivalence_per_shape`), regenerated `r10_pyref/`/`golden/`,
  `contract.md` (count text), the pipeline fail-closed fixes, and the 2 test fixtures.
- **Not** running `scripts/corpus/generate_shapes_and_contract.py --all` over the corpus (it emits
  generic `PUBLIC_ROWS` that differ from the verified family-specific shapes → clobber).
- **Not** a blind 50×3 product — gated by per-task feasibility + the storage budget (see Cost).

## Context & Research (verified against source)

### Pipeline reality (fixes required first)

- Verify scripts enumerate shapes **dynamically** BUT not fail-closed: precompute silently skips
  failing extra shapes (`precompute_oracle_reference.py` `_extra` `continue` branches), C2 grades
  the existing `ref_*.safetensors` glob (`run_oracle_equivalence_gpu.py`), S2 + golden do enumerate
  `shapes.yaml` dynamically (`standalone_freeze_verifier.py`, `capture_golden_baselines.py`).
- **Generator is NOT the source of truth.** `generate_shapes_and_contract.py` has global
  `PUBLIC_ROWS`/`HIDDEN_ROWS` (e.g. P1 = {B:1,L:4,D:8}), but the verified corpus is family-specific
  (L2_20 P1 = {B:16,L:16,D:32,N:64}; L1_12 norm dims). Running it would overwrite verified shapes.
- **Constraint sources are multiple:** `param_constraints.dims` (predicate evaluator exists at
  `ascendbench/input_spec/param_constraints.py` / `expr.py`), plus `dtypes/dists/scalars/
  tensor_invariants`, plus launcher asserts (e.g. L2_29 `_pa_launcher.py` output/group-shape
  asserts, L1_06 tensor invariants). A dims-only filter is insufficient.
- **Existing drift:** L3_22 `contract.md` says H1–H4 but `shapes.yaml` has only H1 — `contract.md`
  count text must be handled, and current drift fixed.

### Cost (corrected)

- Storage is the binding constraint: `r10_pyref` already totals **13G**; **L3_29 alone is 8.7G**,
  L3_16 1.8G, L3_19 912M. Per-shape ref size scales with the operator's tensors — large for
  attention/SSM tasks. Doubling shapes on the big L3 tasks could add many GB. Per-shape adds also
  cost: GPU cost-model cross-check per public shape (`scripts/perf/cross_check_cost_model_gpu.py`),
  eval roofline per result shape (`ascendbench/engine/eval.py`), and a golden per shape for the 18
  `upstream_oracle_required` tasks (`eligibility.py`).

### NPUKernelBench (the borrow source)

- ~50–60 cases/op, ~1/3 each fp16/fp32/bf16; LLM dims (1536–8192), odd non-pow2, edge (0D/empty/
  singleton, 1D–9D), swept sequence length. Direct op-overlap to borrow shapes from:
  LayerNorm→L2_01/L1_12, SiLU→L2_04, GELU→L2_05, BatchMatmul→L2_20/L2_21, SwigluQuant→L2_13/L2_29.

## Key Technical Decisions

- **Decision: fail-closed pipeline FIRST (Phase 0), then broaden.** Until a declared shape that
  can't be graded is an ERROR and the release gate coverage-gates declared==graded, broadening is
  unsafe (silent-ungraded shapes). This is the precondition, not an afterthought.
- **Decision: per-task / per-family shape sets, append-only — NOT the universal generator.** Each
  task's new shapes are derived from ITS family's dims (scale the existing values + borrow
  NPUKernelBench dims for overlap ops + add edge cases), validated against all its constraints by a
  real build_inputs+oracle dry-run. Existing P1–P5/H1–H4 bytes are untouched.
- **Decision: feasibility from KERNEL source, not meta.** A reproducible script reads each
  `kernel.py`'s tl dtype configs (e.g. L2_20's `{bf16,fp16,fp32}` config dict) + `param_constraints`
  + tensor_invariants + launcher asserts to classify dtype- and shape-flexibility. The plan asserts
  NO feasibility counts; it ships the report.
- **Decision: shape-manifest freshness gate.** Add a `phase_a_prime.shape_manifest_sha256` — hash
  of the **canonical declared-shape manifest**: each shape's id + full knob payload + dtype +
  public/hidden role, order-normalized. (An id-set hash alone would miss a changed dim/dtype under
  an existing id, e.g. editing P3's D.) Any `shapes.yaml` change thus invalidates the S2/C2 PASS
  exactly like a kernel byte change does, forcing re-verify. Wire `validate_corpus_consistency`
  (declared==graded) into the release gate.
- **Decision: storage-aware rollout.** Skip/limit shape adds on the multi-GB tasks (L3_29 8.7G,
  L3_16, L3_19) until a storage budget + possible ref compression is decided; lead with the small
  elementwise/norm/matmul tasks where refs are KB–MB.

## Open Questions

### Resolved During Planning (verified)

- *Is verify fail-closed on new shapes?* No — silent-skip + glob-graded. Must fix (Phase 0).
- *Is the generator the source of truth?* No — current shapes are family-specific; don't run it.
- *Are the feasibility numbers reliable?* No — meta-derived, wrong; re-derive from kernel.

### Deferred to Implementation

- Per-tier target shape count — set after Phase-0 + the pilot's measured ref-size/verify cost.
- Per-family valid-lattice generators for tight tasks (block-quant, AWQ, expert/head, chunk).
- Storage policy for the multi-GB L3 tasks (cap shapes, compress refs, or exclude).

## Implementation Units

- [ ] **Unit 0 (Phase 0, BLOCKING): make the pipeline fail-closed + coverage-gated**
  - **Files:** `scripts/oracle/precompute_oracle_reference.py` (declared shapes must build —
    restrict the `_extra` silent-skip to an explicit opt-in sweep set, not declared shapes),
    `scripts/oracle/run_oracle_equivalence_gpu.py` / `apply_oracle_equivalence_verdict.py` (assert
    graded refs == declared shapes), `ascendbench/release_gate/eligibility.py` +
    `scripts/corpus/validate_corpus_consistency.py` (wire into the gate: declared shapes == graded
    refs == C2 verdicts for ALL tasks, **and == goldens only for `upstream_oracle_required` tasks**
    — non-UOR tasks have no goldens), `scripts/verify/standalone_freeze_verifier.py` (write
    `phase_a_prime.shape_manifest_sha256`, the full-payload manifest hash).
  - **Test scenarios:** add a shape to a test task without its ref → release gate goes RED
    (coverage gap caught); a declared shape that can't build → ERROR not skip; changing shapes.yaml
    flips `fresh_s2_pass` to false until re-verify.
  - **Verification:** a task with declared P6 but no ref_P6 → release-gate RED; after re-verify →
    GREEN. This unit ships BEFORE any shape is added.

- [ ] **Unit 1: reproducible feasibility report (replaces asserted numbers)**
  - **Files:** new `scripts/corpus/shape_feasibility_report.py`; output `docs/reviews/2026-06-15-shape-feasibility.md`.
  - **Approach:** per task, read `kernel.py` tl dtype configs (multi-dtype config dicts vs locked
    tl.bfloat16/fp8/int8), `param_constraints` (dims/dtypes/scalars/tensor_invariants), launcher
    asserts → classify (dtype-flexible? shape-loose/medium/tight?). Emit a checked-in table.
  - **Verification:** the report is reproducible (`--regen`) and its per-task verdicts are
    spot-checkable against source (e.g. L2_20 → dtype-flexible despite meta fp32).

- [ ] **Unit 2: per-task shape-extension tool (append-only, constraint-validated)**
  - **Files:** new `scripts/corpus/extend_task_shapes.py`; per-op borrow table from NPUKernelBench;
    test.
  - **Approach:** for a task, generate candidate new shapes from its family dims + NPUKernelBench
    overlap + edge cases; **validate each candidate by actually running build_inputs + oracle**
    (fail-closed dry-run) and the constraint predicates; append only the survivors as P6…/H5…/edge…
    Preserve P1–P5/H1–H4 bytes. Update `meta.shape_coverage` + `contract.md` count text.
  - **Test scenarios:** loose task gets many new shapes; tight task (L2_18 block-quant) gets only
    constraint-valid ones; a candidate that fails the oracle dry-run is dropped (asserted), never
    appended.

- [ ] **Unit 3: regenerate refs + GPU re-verify (per-shape, storage-aware)**
  - **Files:** uses precompute (r10_pyref) + the daemon (S2/C2/golden). Storage check before big-task adds.
  - **Approach:** precompute refs for the new shapes (fail-closed now), then daemon S2 + C2 + golden
    + cost-model + the new coverage gate. A new shape failing C2 = quarantined, never tuned-to-pass.
  - **Verification:** every declared shape has a verdict; release-gate GREEN under the coverage gate;
    existing P1–P5 verdicts unchanged; storage delta within budget.

- [ ] **Unit 4 (Phase 4, optional): dtype sweep on the kernel-verified-flexible subset**
  - **Files:** the dtype-flexible tasks' `shapes.yaml` + `oracle_pytorch.py` (parametrize) + tolerance.
  - **Approach:** per the Unit-1 report (kernel-verified multi-dtype), add fp16/bf16/fp32 variants
    with the oracle cast + contract tolerance. Locked-dtype/quant tasks untouched.

## Phased Delivery

- **Phase 0 (BLOCKING):** Unit 0 — fail-closed + coverage gate + shape-set hash. No shapes added yet.
- **Phase 1 (pilot, small-ref overlap tasks):** L2_01 layer_norm, L2_04 silu, L2_05 gelu, L2_20/21
  matmul — borrow NPUKernelBench dims, regenerate refs, GPU-verify under the new gate, **measure
  ref-size + verify cost** to size rollout. Confirm zero regression on P1–P5.
- **Phase 2 (small/medium-ref tasks):** broaden where refs are KB–MB.
- **Phase 3 (tight + quant/special):** per-family constraint-valid lattices.
- **Phase 4 (optional dtype sweep on kernel-flexible tasks).**
- **Phase 5:** full 101 regression + coverage-gated release-gate green; ship the feasibility +
  before/after coverage report.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| New shape silently ungraded yet task "passes" (core self-deception) | Phase-0 Unit 0: declared-shape build is fail-closed; release gate coverage-gates declared==graded==verdict (==golden for `upstream_oracle_required` tasks only) |
| Running the universal generator clobbers verified family-specific shapes | Per-task append-only tool (Unit 2); never run `generate_shapes_and_contract.py --all`; preserve P1–P5/H1–H4 bytes |
| Stale PASS proof over expanded shape set | `phase_a_prime.shape_manifest_sha256` (full payload: ids+knobs+dtype+role) forces S2/C2 re-verify on ANY shapes.yaml change; consistency check in release gate |
| Inflated/wrong feasibility (meta-derived) | Unit 1 reproducible report from KERNEL source; plan asserts no counts |
| Storage blow-up (r10_pyref 13G; L3_29 8.7G) | Storage-aware rollout; lead with small-ref tasks; budget + possible ref compression before big-task adds |
| New shape violates a constraint not in `dims` (launcher assert, tensor invariant) | Unit 2 validates by REAL build_inputs+oracle dry-run, not a dims predicate alone |
| Regression on existing P1–P5/H1–H4 | Append-only byte-preserving; Phase-1 confirms current verdicts unchanged; full regression Phase 5 |
| dtype gap under-served | Unit 1 re-derives kernel dtype-flexibility (more feasible than meta suggested); Phase 4 sweeps it |
| contract.md count drift (L3_22 already stale) | Unit 2 updates contract count text; fix existing drift as a pre-req |

## Sources & References

- NPUKernelBench: `/data1/code/agentic-rl/kernel/AscendOpGenAgent/benchmarks/NPUKernelBench`
- Pipeline (verified): `scripts/oracle/precompute_oracle_reference.py` (silent-skip `_extra`),
  `scripts/oracle/run_oracle_equivalence_gpu.py` (glob-graded), `standalone_freeze_verifier.py`,
  `capture_golden_baselines.py`, `scripts/corpus/validate_corpus_consistency.py`,
  `ascendbench/release_gate/eligibility.py`, `ascendbench/input_spec/param_constraints.py`/`expr.py`
- Generator (do NOT run corpus-wide): `scripts/corpus/generate_shapes_and_contract.py`
- Counterexamples (feasibility-from-kernel): L2_20 `kernel.py` multi-dtype configs vs meta fp32;
  L2_18 output-dtype-flexible / int8-input-locked
