# Shape Coverage Policy

How `shapes.yaml` is populated, validated, and bounded. The *schema* of a shape row lives in
[`task_api_contract.md`](task_api_contract.md) (`public_shapes` / `hidden_shapes` / `size_knobs` /
`inputs` / `param_constraints`); this document is the *policy* — what shapes a task should carry, how
each one earns its place, and why not every operator reaches the same count.

## 1. Goal: broad shape coverage for migration robustness

Each operator should be exercised across a **wide, realistic range of shapes** (target ~50, up from
the original baseline of 9) so the GPU→NPU-migrated kernel's correctness *and* performance are robust
across the shapes a real workload hits — not just one tiny/medium/large triple.

Generation is done by [`scripts/corpus/extend_task_shapes.py`](../../scripts/corpus/extend_task_shapes.py):

- **Value ladder** per active knob — powers of two **plus** realistic non-power-of-2 LLM dims
  (96, 192, 384, 640, 768, 1280, 1536, …) plus a few odd/scaled points, spanning the knob's `min`
  up to ~4× its existing max, capped by `DIM_CAP` (32768) and a per-tensor `ELEMENT_CAP` (16M), and
  respecting each knob's `min` + `mod` constraint.
- **Single-axis sweep** — each knob walked across its full ladder (others held at a median base) so
  every dimension's whole range is covered. → public shapes.
- **Coprime-stride multi-axis grid** — a low-discrepancy sample over the ladder *product* (each axis
  advances by a stride coprime to its ladder length, so no axis collapses onto a sub-cycle) for
  interaction coverage (small-B/large-D, …) without a blind full product. → public shapes.
- **Edges** — all-min, all-max, aspect extremes, singleton. → hidden shapes (sealed A′-validation).

Generation is deterministic (no RNG): the same task + targets always yields the same candidate set.

## 2. Every declared shape must be REAL — two-stage validity gate

A shape is only allowed into `shapes.yaml` if it passes **both** stages. No shape is declared on
faith.

1. **CPU pre-filter** (at generation time): `build_inputs` materialises the row, the task oracle runs
   on it, and `param_constraints` hold. A row that the oracle can't even run is dropped before it is
   ever written.
2. **GPU authoritative grade** (on the eval daemon — a real CUDA GPU, *not* the dev box):
   - **S2 severance** — the frozen kernel must compile, run, and be **bit-exact** across the
     severed/original boundary for that shape.
   - **Correctness** — for a normal task, **C2 oracle-equivalence** (kernel output aligns with
     `oracle_pytorch` within the task's tolerance class). For an `upstream_oracle_required` (UOR)
     task, whose simplified oracle is not trustworthy as ground truth, the verification is
     **golden capture** (the kernel's deterministic output is captured as the reference) instead.

The dev box is triage only; the daemon verdict is always authoritative
(see [`task_api_contract.md` §6](task_api_contract.md)).

## 3. No fake coverage — you cannot pad, drop, or tune to a number

- A declared-but-ungraded shape forces `SHAPE_COVERAGE_GAP`, and a task with that status is **never
  VERIFIED**. So `shapes.yaml` cannot be padded with shapes that do not actually grade.
- Tolerances are **never** relaxed, and hard shapes are **never** silently dropped, to inflate a pass
  rate. A shape the frozen kernel rejects is **quarantined** (removed) *honestly and visibly*, with
  the reason recorded — it does not get hidden or tuned past.
- `phase_a_prime.shape_manifest_sha256` hashes the parsed shape set (id + dtype + knobs, order-
  preserving). **Any** shape edit — add, remove, change a knob/dtype, reorder `inputs` — changes the
  hash and invalidates a stale S2/C2 PASS, forcing a fresh re-grade. Freshness is enforced, not
  assumed.

## 4. Honest ceiling — not every operator reaches the same count

The number of *valid, verified* shapes an operator can carry is bounded by reality, not by effort:

- **Frozen-kernel input contract.** The kernels are byte-locked upstream snapshots and must not be
  edited. Many require power-of-2 dims for `tl.arange`, channel-last layout asserts (`stride==1`),
  or block-size divisibility. Shapes that violate these simply will not compile/run — there is no
  legal way to host them.
- **Intrinsic shape space.** A 1–2-knob elementwise op (e.g. `silu_and_mul`) has no 50 *distinct,
  meaningful* shapes; manufacturing them would be near-duplicate padding, not coverage.
- **Numerical reality.** Non-deterministic kernels (MoE top-k, some atomics) fail S2 bit-exactness on
  some shapes; large bf16/fp8 reductions diverge from the oracle beyond tolerance.

The policy is therefore: **as many valid + verified shapes as the kernel honestly admits, up to
~50** — never "exactly 50 by any means". The current corpus reflects this (most ops 40–50; ops with
narrow kernel contracts or small shape spaces top out lower).

## 5. Quarantine taxonomy — why a specific shape is removed

| Cause | Surfaces as | Action |
|-------|-------------|--------|
| non-pow2 dim feeding `tl.arange` | S2 `original-no-run` CompilationError | remove the shape |
| layout / stride assert | S2 `original-no-run` AssertionError | remove the shape |
| non-deterministic kernel for that shape | S2 `compare` NOT bit-exact | remove the shape |
| large-shape oracle divergence beyond tol | C2 `FAILED` (task → `MIXED`) | remove the shape |
| large-shape dtype-ULP divergence on an **allowlisted** task | C2 `CEILING_ACCEPTED` | **keep** — graded as a ceiling, never counted as an exact pass (see `ascendbench/policies/ceiling_acceptance.yaml`) |

The daemon's batched S2 aborts at the first compile-rejected shape **without naming it**;
[`scripts/corpus/gpu_shape_probe.py`](../../scripts/corpus/gpu_shape_probe.py) re-runs each shape
through the same S2-original machinery in isolation to recover the offending shape ids, and
[`scripts/corpus/quarantine_shapes.py`](../../scripts/corpus/quarantine_shapes.py) removes exactly
those rows (byte-surgical) and rewrites the meta `shape_coverage` counts.

## 6. End-to-end flow

```
extend_task_shapes (generate + CPU-validate + append)   # corpus RED
  → S2 grade        → probe + quarantine kernel-rejected → re-S2  (until all PASS)
  → C2 / golden     → quarantine oracle-divergent shapes → re-S2 → re-C2  (until VERIFIED/CEILING)
  → validate_corpus_consistency GREEN + release-gate PROCEED
  → commit shapes.yaml + daemon-written meta verdicts FROM the GPU box   # corpus GREEN
```

> A Chinese mirror of `docs/reference/` lives under `docs/zh/reference/`; add `docs/zh/reference/shape_policy.md` when translating.
