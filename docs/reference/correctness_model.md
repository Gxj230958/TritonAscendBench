# Correctness Model

How TritonAscendBench decides an operator's output is **correct** — the
precision-alignment standard a ported kernel is graded against.

This is the mechanism companion to [`tolerance_policy.md`](tolerance_policy.md)
(which is the per-class cheat-sheet) and to [`ARCHITECTURE.md`](../ARCHITECTURE.md)
(which is the end-to-end data flow). The authoritative implementation is
`ascendbench/correctness/tolerance.py`; the per-class numbers live in
`ascendbench/policies/tolerance_policy.yaml`.

---

## 1. Two correctness layers (the most-confused point)

TAB judges correctness on **two distinct axes**, with **two distinct standards**:

| Layer | Question | Standard | Where |
|-------|----------|----------|-------|
| **C2 — oracle-equivalence** | Does the kernel compute the right math vs the PyTorch oracle? | a **tolerance class** (numeric tolerance) | every submitter grade; provider GPU cross-check |
| **S2 — severance** | Is the *frozen* standalone `kernel.py` faithful to the original upstream call? | **bit-exact** (values **and** aliasing **and** mutated-input behaviour) | provider-side corpus construction only |

The single most important rule: **tolerance applies only to C2.** S2 is always
bit-exact — it compares the severed kernel against the real upstream call and
must match identically, including which inputs are mutated in place and which
output tensors alias which inputs. A submitter is never graded by S2; S2 is the
gate that guarantees the task's `kernel.py` is a true copy of production code.

The rest of this document is about **C2**, the precision standard for grading.

---

## 2. What is compared, and how shape verdicts aggregate

An oracle and a candidate both return `dict[str, Tensor]`. Grading is per
**output key** and per **shape row**:

- **Structure first.** Every key the **oracle** returns must be present in the
  candidate's output with a matching tensor shape; extra candidate keys are
  ignored. A shape mismatch is an immediate fail — tensors are never cast or
  broadcast to force a comparison.
- **Per shape.** Every graded shape row (public shapes in `self-reported` mode;
  public **and** sealed `hidden_shapes` in `verified` mode) is compared
  independently with `tolerance.compare`.
- **Task verdict.** A task's `correctness_status` is `"ok"` only when **every**
  graded shape passes (`all(r.correctness_passed for r in ran)`); any failing
  shape makes it `"fail"`. `coverage` separately reports the passing fraction.
- **A device kernel must actually run.** A candidate that produces **no**
  device/triton kernel event has silently fallen back to the generic torch
  path; it is forced to a correctness **fail** even if numerically identical —
  a migration must run a real kernel on the target, not eager PyTorch.

---

## 3. `compare()` — the dtype routing

`tolerance.compare(actual, expected, cls)` dispatches by dtype **before** any
numeric check, because the right notion of "close" differs by dtype. In order:

1. **Shape mismatch** → fail (never cast).
2. **Integer dtype** (`bool`/`uint8`/`int8`/`int16`/`int32`/`int64`):
   - `cls.bit_exact`, or the class declares no `int_atol` → **`torch.equal`**.
   - else → **integer ULP**: `|actual − expected| ≤ int_atol`.
   - An explicit `bit_exact` always wins over `int_atol`.
   - *Why never float-cast:* casting an int output to fp32 and running
     `allclose` silently rounds e.g. `int64(2²⁴+1)` to `2²⁴`, turning an
     off-by-one into a false pass.
3. **fp8** (the class sets `fp8_ulp`):
   - both sides fp8 → **ordered-code ULP** (see §4).
   - exactly one side fp8 → fail. An fp8 graded output must be compared *as*
     fp8; fp32-casting one side to reach a pass is forbidden.
4. **`cls.bit_exact`** (non-integer) → `torch.equal`.
5. **Otherwise (float)** → cast **both** sides to fp32, then:
   - a **sampled fast-fail** (only when `numel > 1024` and `sample_fraction < 1`):
     reject early if even a random sample fails, but
   - the **authoritative** verdict is always the **full** `allclose` over every
     element — a sample passing can never be the final answer (NaN/Inf or drift
     outside the sample window must still be caught).

---

## 4. The four comparators

| Comparator | dtype | Pass criterion |
|------------|-------|----------------|
| **float allclose** (`_check_full`) | fp16/bf16/fp32 | inf and NaN **positions** must match (`inf_must_match`/`nan_must_match`); then `torch.allclose(atol, rtol)` over the finite elements. |
| **integer ULP** (`_int_atol_compare`) | int* | dtypes must be **identical**; `\|a−e\|` computed in widened `int64` (the subtraction cannot overflow) `≤ int_atol`. |
| **fp8 ordered-code ULP** (`_fp8_ulp_compare`) | `float8_e4m3fn` | dtypes identical; view as `uint8`, map to the **ordered sign-magnitude e4m3 code** (adjacent representable values are distance 1), canonicalise ±0, NaN positions must match; pass iff `max_ordered_dist ≤ ulp` **and** the off-by-one **fraction ≤ 1%**. |
| **bit-exact** (`_bit_exact_compare`) | any | dtypes identical; `torch.equal`. |

Two design notes worth calling out:

- **fp8's two-part test.** A single off-by-one ordered-code step is the same
  boundary-rounding ambiguity as an int8 quant (a value within the
  `silu`/`exp` epsilon of an fp8 bucket midpoint can round either way). But a
  *large fraction* of off-by-one elements is not boundary noise — it is a
  systematic rounding-mode bias (a real bug), so the `off_frac ≤ 1%` cap rejects
  it even when every individual step is only 1 ULP.
- **dtype equality is enforced by the integer / fp8 / bit-exact comparators**,
  not by the float path. The float path casts **both** sides to fp32, so a
  `fp32`-vs-`bf16` *float* output difference is not caught by `compare` alone —
  a strict `VERIFIED` float compare does not itself assert the output dtype. It
  is caught elsewhere: the severance anchor's dtype check (§7), and — only when a
  strict-failing shape is then considered for ceiling acceptance — the **ceiling
  dtype gate** (§6), which fail-closes a `CEILING_ACCEPTED` candidate whose output
  dtype differs from the reference.

---

## 5. Tolerance classes and tolerance source

Each task pins `meta.yaml.tolerance_class` to one of the classes in
`tolerance_policy.yaml`. A class carries `atol`, `rtol`, `inf_must_match`,
`nan_must_match`, `sample_fraction`, and the optional `bit_exact` / `int_atol` /
`fp8_ulp` switches that select the routing in §3. The full class table is in
[`tolerance_policy.md`](tolerance_policy.md).

The grading path resolves the class by name: `eval.py` uses
`tolerance.load_policy().get(tolerance_class)`, and the provider GPU cross-check
(`run_oracle_equivalence_gpu.py`) uses `_tolerance_for(tolerance_class)`, which
falls back to a permissive `vector_reduce`-style default if the class name is not
in `tolerance_policy.yaml`.

A separate helper, `tolerance.for_task`, derives a *tolerance source* used by the
**release-gate** accounting, not by grading: when a task's
`provenance.golden.tolerance_source` is `upstream_test_explicit` with an `atol`/
`rtol` pair, those are pinned from the upstream test (while still cloning the
class's `int_atol`/`fp8_ulp` so a pinned integer task keeps its ULP tolerance).
`migration_drift_budget` is recognised but is not release-ready and is reported
for visibility only.

---

## 6. Ceiling acceptance — scoped large-shape relaxation

Some large shapes of a correctly-aligned kernel sit a hair outside strict
tolerance purely from a dtype accumulation ceiling (e.g. a bf16 `tl.dot` MMA),
not from a real divergence. These are graded `CEILING_ACCEPTED` — a status
**distinct from `VERIFIED`, never merged into the strict pass count, and never
allowed to promote a `golden_sha256`**. It is applied **after** strict
`tolerance.compare` fails, by `_ceiling_verdict` in
`run_oracle_equivalence_gpu.py`, against the allowlist in
`ascendbench/policies/ceiling_acceptance.yaml`. `tolerance.py` itself never
relaxes; `release_gate/eligibility.py` only *reads* the recorded status.

The allowlist is **task-scoped** and uses a co-located discriminator:

```
max_tol_units = max( |actual − ref| / (atol + rtol·|ref|) )
```

This is the **same band** as strict `compare`, so a near-zero (near-cancellation)
reference stays bounded by `atol` and cannot be hidden. Two modes:

- **Mode A — units cap.** A task listed in `tol_units_tasks` is accepted iff it
  has a non-empty failing-key set, no compare error, correct dtype, all stats
  finite, and `max_tol_units ≤ tol_units_cap` (currently **8.0**). A clean dtype
  ceiling sits a few units above 1; the genuine upstream bug in `L2_23/24`
  off-by-one sits at 1000–6200 — over 100× past the cap — so it stays strict by
  construction.
- **Mode B — physical-scale bound.** For the rare task whose worst element is a
  legitimate near-cancellation point (near-zero ref, `atol`-dominated), listed in
  `physical_tasks`. Accepted iff **every** failing output key satisfies
  `max_abs ≤ physical_factor · max|ref[key]| · ulp_rel`.

Everything not on the allowlist stays strict. `validate_corpus_consistency`
enforces that only allowlisted tasks carry the ceiling status, with full shape
coverage, `golden_source = dtype_ceiling_accepted`, and no `golden_sha256`.

---

## 7. Provider-side oracle-alignment safeguards

Three mechanisms keep the C2 oracle a faithful reference for a GPU/NPU kernel
without weakening the tolerance:

- **TF32 emulation.** An unpinned fp32 `tl.dot` runs in TF32 on the GPU (10
  mantissa bits) while a CPU PyTorch oracle would compute full fp32 — a false
  mismatch. Affected tasks emulate it **in their own `oracle_pytorch.py`** via a
  `_to_tf32` helper that masks off the low 13 mantissa bits of the matmul
  operands, so the oracle matches the kernel's real arithmetic. This is
  per-task and explicit (not a global tolerance change).
- **Narrow-float anchor rescue.** The standalone-severance anchor runs under
  `TRITON_INTERPRET=1` (CPU, GPU-free); the interpreter truncates narrowing
  float stores (fp8 / fp16 / bf16) toward zero instead of round-to-nearest-even
  (`100.3 → 100.0`). The anchor flags such outputs (`narrow_float_output`); the
  parent then re-runs the **same** anchor once on a real GPU — which rounds
  correctly — with **unchanged** tolerance, and adopts that verdict. If no GPU is
  available there is no rescue: it is **fail-closed**, never projected to a pass.
- **No silent torch fallback** (§2): a candidate producing no real device kernel
  event fails even when numerically correct.

---

## 8. Where grading sits in the data flow

For one shape, in the trusted parent process:

```
build_inputs(shapes, shape_id, seed)
   → run candidate in the isolated subprocess (correctness pass)
   → _resolve_golden(task tier × mode) picks the reference
   → tolerance.compare(actual, golden, load_policy().get(tolerance_class))
   → per-shape correctness_passed
```

Per-shape results aggregate per §2. The isolation guarantees (no golden leak,
safetensors-only IPC, audit hook, correctness/timing pass separation) and the
`_resolve_golden` routing (oracle tier × mode) are described in
[`ARCHITECTURE.md` §3](../ARCHITECTURE.md). This document covers only the
precision standard applied at the `compare` step.
