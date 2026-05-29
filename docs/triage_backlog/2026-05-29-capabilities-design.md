# Path-to-100 capability map + build design (for codex review)

Authoritative classification of the 80 non-PASS tasks (using the
resolver's own `enumerate_candidates` + name-matching the candidate to the
oracle operator, cross-referenced with GPU resolve `primary_blocker`).
20/100 currently PASS, all via host-wrappers.

## Capability map (the scan — what's actually missing)

| | tasks | what unlocks it | reach |
|---|---|---|---|
| **A. host-wrapper exists, can't invoke** | 26 | import/invocation unblocking | partial |
| ↳ A1 vllm import wall | ~10 | shim/vendor (proven, = L2_13/14 lever) | +5-8 |
| ↳ A2 triton-API drift (no wall) | 8 | triton-compat shim (NEW) | +? |
| ↳ A3 sglang import wall | 8 | **bench-team arch decision** (no sound shim) | parked |
| **B. bare @triton.jit, no host-wrapper** | 28 | launcher synthesis | partial |
| ↳ B1 simple (req ≤ 6, 1D elementwise) | 3 | auto launcher synth | +3 |
| ↳ B2 medium (req 7-12) | 9 | semi-auto + light recipe | +5-9 |
| ↳ B3 complex (req 14-52, attn/moe/conv) | 16 | per-task hand launcher | heavy tail |
| **C. host-wrapper, shapes-incomplete** | 16 | per-task `shapes.yaml` authoring | +? deep |
| **D. host-wrapper, oracle-mismatch** | 6 | comparison policy / right-variant | partly bench-team |

> Correction to the earlier "bare-jit majority / ceiling ~30" claim: a
> finer classification shows **52 tasks already have a name-matching
> host-wrapper** (blocked by import/shapes/oracle, NOT launcher); only
> **28** are bare-jit. So the launcher is one of *several* needed
> capabilities, and not the largest.

## Bucket E — wrapper/contract mismatch (added per codex review)

The "52 host-wrapper" count is an UPPER BOUND, not authoritative:
`enumerate_candidates` counts any top-level launcher as a host wrapper,
resolution only tries the top-6, and candidate choice + `primary_blocker`
are explicitly heuristic. Confirmed example: L1_15's matched
`act_and_mul_triton` is the MoE-flavored wrapper (routing/config
contract), NOT the simple task oracle. So some FORM-A are really:
- **E. wrapper exists but is the WRONG operator/contract** — the right
  operator is a different (often bare-jit) function or absent. The 6
  `oracle-mismatch` FORM-A are confirmed E; an unknown fraction of the 16
  `shapes-incomplete` + 26 `upstream-not-invocable` may be E too.
Implication: do NOT drive ROI ordering off the 52 number; reclassify by
ACTUAL oracle-consistency before committing effort. A task is genuine
FORM-A only if its matched wrapper oracle-VERIFIES (or is plausibly
shapes-completable into a verify); otherwise it's E and needs the right
candidate (→ B) or is genuinely out of reach.

## Anti-fake reality (corrected per codex — NOT an absolute invariant)

Earlier draft claimed a wrong launcher/shim/shape can NEVER fake-PASS.
That is **too strong**. Two real false-PASS vectors:
1. The oracle-consistency anchor is **sampled + tolerance-based** — a
   wrong candidate can coincide with the oracle on the sampled shapes
   within tolerance (the resolver code says so explicitly).
2. S2 proves `severed == original under the SAME launch semantics`. If a
   **wrong launcher is applied to BOTH sides**, they agree (both wrong
   the same way) and S2 passes even though both diverge from true
   semantics.
=> A launcher recipe must be **correct**, not merely consistent.

**CRITICAL (codex 2nd-pass FAIL): once BOTH S2 sides use the same
launcher seam, S2 stops being an independent witness of launch
correctness** — severed==original is automatic even if the launcher is
wrong. The sampled/tolerant anchor and "cross-check a sibling wrapper"
(usually absent for true bare-jits) and "codex-review the recipe"
(process, not validation) do NOT close this. A recipe-launched bare-jit
task therefore needs an **independent launch witness** before PASS.

The sound way to get that witness — and the REVISED Capability B primary
path: **do not invent a launcher; recover the REAL upstream launch.** The
host-wrapper/callsite that launches this jit EXISTS in upstream (it was
stripped when the task was extracted). So:
- **B-primary = find + vendor the real upstream host-wrapper/callsite**
  (provenance-recorded: upstream file:line), via the existing U2 vendoring.
  Correct-by-construction (it IS what upstream runs) → turns bucket B into
  bucket A, no invented launcher, no fake-PASS vector.
- recipe still carries the FULL contract (grid, constexprs, arg_order,
  outputs **+ init + alias_map + mutated_inputs**) — but it is DERIVED
  from the upstream callsite, not guessed.
- **B-fallback (hand recipe with no upstream callsite found): PROVISIONAL
  / non-PASS** until an independent witness validates it (e.g. a trusted
  golden capture), AND harden the oracle anchor to non-sampled. Never let
  a guessed-and-self-consistent recipe reach PASS.

## Build design

### Capability B — bare-jit launcher (the primary ask)

**PRIMARY PATH (revised after 2nd-pass FAIL): recover + vendor the real
upstream host-wrapper, don't invent a launcher.** For a bare-jit task the
launching wrapper/callsite exists in upstream (stripped at extraction);
find it, vendor it (U2, provenance = upstream file:line), redirect the
entrypoint to it. This makes the launch correct-by-construction (it IS
upstream's launch) — closing the independent-witness hole.

**PROJECTION/ADAPTATION GATE (required, 3rd-pass FAIL fix):** provenance
proves "this is upstream's launch," NOT "this is exactly the task's
operator boundary." A real wrapper often computes MORE than the task
oracle (extra pre/post-process, a larger composite output — e.g. L1_35's
upstream wrapper returns (reorder_topk_ids, src2dst, seg_indptr) while the
oracle wants only src2dst). So vendoring alone must NOT auto-promote
B → A. A vendored-wrapper task promotes to A **only if**:
  (1) the wrapper's observable outputs already MATCH the task oracle
      (direct bit-exact), OR
  (2) a **reviewed, provenance-backed projection** maps the wrapper's
      outputs/mutations onto the task contract (e.g. select the src2dst
      element; the projection is recorded + codex-reviewed, not guessed).
Otherwise the task is **bucket E** (wrong abstraction/contract) — left
non-PASS, not force-fit. This gate is what makes the bit-exact verify
meaningful (it must compare the task's operator, not a composite superset).

**SEAM (only for the residual: no findable upstream wrapper):** if no
upstream callsite exists, a launcher seam is needed. It is a RUNNER-LEVEL
invocation seam (codex Finding 1), NOT a standalone-rewrite artifact: the
two runners `_pa_oracle_check_runner` (anchor invoke, ~L365) and
`_pa_subproc_runner._resolve_entrypoint` (S2 severed+original invoke,
~L555) do `dict→kwargs` and can't launch a bare jit. The seam must fire on
ALL invocation sides (anchor + S2 severed + S2 original) — a standalone-
only launcher leaves the original baseline un-launchable → no comparison.
(`phase_a_cost_xcheck` imports `_resolve_entrypoint`, covered transitively.)
BUT note: once both S2 sides share the seam, S2 is no longer an independent
witness (see Anti-fake) → seam-launched tasks stay PROVISIONAL until an
independent witness validates them. So the seam is a LAST resort; vendoring
the real wrapper is strongly preferred.

A bare `@triton.jit` can't be called with tensors; the seam needs a
launcher that (1) classifies each param as pointer / scalar /
`tl.constexpr`, (2) allocates output buffers, (3) computes the grid,
(4) picks constexpr values, then launches `kernel[grid](...)` — uniformly
on all invocation sides.

**B-auto (generic synthesizer — NARROW, codex Q1):** only a very small
B1 subset is safely auto-synthesizable. REFUSE (fail-closed → route to
B-recipe) on ALL of: multi-axis `program_id`, reductions, indirect
indexing, nontrivial stride synthesis, multiple plausible extents,
alias/mutation, or no explicit tail mask. Realistic B-auto reach ≈ the 3
SIMPLE tasks, maybe a couple more.
A `synthesise_jit_launcher(kernel_ast, shapes, oracle_outputs)`
that handles the *deterministically inferable* pattern (and EMITS a
recipe consumed by the same seam, so auto + manual share one execution
path):
- param classification from the jit signature: `tl.constexpr`-annotated →
  constexpr; params whose name matches a shapes input → tensor pointer;
  remaining int params → derived scalars (e.g. `n = <tensor>.numel()`).
- output buffers: allocate from the **oracle's return shapes/dtypes**
  (the oracle is the source of truth for output shape) — `candidate_inplace`.
- grid: infer dimensionality from `tl.program_id(axis=k)` uses in the
  kernel body; size each axis as `cdiv(<mapped-dim>, <BLOCK>)`. Only emit
  a launcher when the grid is unambiguously 1-D over a single shapes dim
  (the simple/elementwise case); otherwise REFUSE (fail-closed → the task
  stays BLOCKED, routed to B-recipe).
- constexpr values: pin to a single explicit value (BLOCK default, e.g.
  1024 / next_pow2(dim)); no autotune sweep.
Output: a synthesized `run(inputs)->dict`. Then normal S2-gpu verify.

**B-recipe (structured per-task, unlocks B2/B3 without guessing):**
Replace the `NotImplementedError` `JIT_LAUNCHER_TEMPLATE` stub with a
meta-driven recipe. Add to meta.yaml an optional:
```yaml
launch_recipe:
  kernel: <jit name>
  grid: "<expr over shapes knobs, e.g. (B, H, cdiv(M, BLOCK_M))>"
  constexprs: {BLOCK_M: 64, BLOCK_N: 64, ...}
  arg_order: [q, k, v, out, stride_qb, ..., B, H, M, N, D, BLOCK_M, ...]
  outputs:            # candidate_inplace buffers, FULL contract (codex F3)
    out: {dims: [...], dtype: ..., init: zeros|full:<expr>}
  alias_map: {...}    # which output aliases which input (in-place)
  mutated_inputs: [...]  # inputs the kernel writes through (S2 checks these)
```
A human authors the recipe (the launch knowledge that lived in the
missing host-wrapper); the seam binds shapes/strides/constexprs/outputs
and launches, on ALL invocation sides. Declarative-first with a Python
escape hatch for B3 kernels whose launch can't be expressed declaratively
(codex Q2 — pure schema would otherwise accrete hidden ad-hoc logic). A
recipe is treated as **correctness-critical code** (codex-reviewed, not a
config tweak) because a shared-wrong recipe can fake-PASS (see Anti-fake).
Complex kernels (B3) all go through this.

### Capability A — make host-wrappers invocable (co-build, higher near-term ROI)

- **A1 (vllm import-wall shim/vendor):** extend the proven shim/rewriter
  pattern (the L2_13/14 lever) to the ~10 remaining vllm walls, per-symbol
  inert-classified + codex-reviewed, loud-fail on un-vetted Class-S.
- **A2 (triton-API-drift compat):** a small compat shim for the
  version-drift the frozen kernels hit under the bench's triton (e.g.
  `_AutotunedKernel.arg_names`). Investigate the exact drift on the 8
  no-wall tasks (4 FLA: L3_13/16/18/20; 4 misc: L1_09/10/31, L2_20),
  then a targeted compat layer. Inert (compat only, no math).
- **A3 (sglang):** NOT shimmable (re-exports side-effectful Class-A);
  flagged for bench-team. Parked.

### Capability C/D — per-task shapes + comparison policy
- C: per-task `shapes.yaml` (the missing-input / output-buffer / varlen
  cases). Deep, one-by-one, with the false-PASS guard being the bit-exact
  verify.
- D: fp8 dequant-compare + variable-vs-fixed output-length comparison are
  **golden/tolerance policy** → propose to bench-team, do not change
  unilaterally.

## Proposed build order (REVISED per codex Q3)

0. **Reclassify out bucket E** (cheap) — re-scan FORM-A by ACTUAL
   oracle-consistency so ROI ordering isn't driven by the heuristic 52.
1. **A1 vllm import-wall shim extension** — proven lever (= L2_13/14),
   inert + verified, **no false-PASS vector**, ~5-8. Safest first win.
2. **A2 triton-API-drift compat** — investigate the shared drift on the 8
   no-wall tasks (4 FLA + 4 misc); targeted compat shim.
3. **B-primary: recover + vendor real upstream host-wrappers** — for the
   28 bare-jit tasks, find the upstream launch wrapper/callsite and vendor
   it (U2, provenance-backed) → bucket B becomes bucket A,
   correct-by-construction. The sound, no-fake-PASS path; do this BEFORE
   any seam/recipe.
4. **B-seam (all-side runner seam) + recipe** — ONLY for residual bare-jit
   tasks with no findable upstream wrapper. Seam-launched tasks stay
   PROVISIONAL (no independent witness) until validated; last resort.
5. **C shapes authoring** — per-task tail.
6. **A3 sglang + D comparison-policy** — bench-team decisions.

Realistic infra-reachable target after A1+A2+B-primary(vendor real
wrappers) ≈ **mid-30s** (codex Q4: "~40" is optimistic — bucket E
uncertainty eats part of the A/C inventory). B3-complex without a findable
upstream wrapper + C + E + bench-team policy are the tail beyond that.
**100% full auto remains out of reach** — bounded by host-wrapper coverage
(in-tree OR vendorable-from-upstream) + a structured-but-manual tail.

## Open questions for codex
1. Is the B-auto grid-inference (1-D-only, fail-closed otherwise) sound,
   or does even the simple case have a trap (e.g. masked tails, multi-dim
   elementwise)? Where exactly should it refuse?
2. B-recipe: is a declarative meta schema the right shape, or should the
   per-task launcher stay free-form Python (current stub intent)? Trade-off
   reviewability vs expressiveness.
3. Build order: is A2/A1 (host-wrapper unblocking) really higher ROI than
   B, given B is the user's stated focus? Your honest ranking.
4. Any capability I MISSED in the scan (a 5th bucket)?
5. False-PASS: any path where B-auto/B-recipe could produce a wrong-but-
   passing result that the oracle-anchor + bit-exact verify would NOT catch?
