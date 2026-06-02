# Strict plan — align bench fixture dtypes to vLLM/SGLang before re-running

**Status: PLAN ONLY. No task modified, no GPU run started.**
**Date: 2026-06-01. Basis: codex-arbitrated 103-task audit
(`docs/reviews/2026-06-01-dtype-per-task-audit.md`).**

## 0. Definitions (be precise about "aligned")

"dtype aligned with vLLM/SGLang" = the bench fixture declares, for EVERY tensor,
the SAME dtype the kernel receives in production:
- activations (q/k/v/x/hidden/gate/up/B/C/dt/z) → **bf16** (model compute dtype)
- SSM state / state-passing / tril intermediates → **fp32**
- quant weights/acts → **int8 / fp8_e4m3**; scales → **fp32**; indices → **int32/64**
- the OUTPUT dtype also matches production (bf16 where the op writes bf16).

"Input-only aligned" = inputs match production but the graded OUTPUT is kept fp32.
This is a WEAKER alignment (the production narrowing-store is not exercised).

## 1. Current state (codex-arbitrated)

| bucket | n |
|---|---:|
| ALIGNED (exact) | 29 |
| FORCED_FP32_TOLERANCE (production IS fp32) | 3 |
| REAL_BUG (raw-dtype `tl.dot`) | 33 |
| INPUT_BF16_GAP (upcast loads) | 31 |
| FORCED_FP32_INPLACE (in-place norm) | 7 |

**Already aligned / faithful: 32.  Need alignment: 71.**

## 2. The hard constraint that governs everything

The oracle gate is full-tensor **element-wise `torch.allclose(actual, expected,
atol, rtol)`** (`tolerance.py:239`). A **bf16 OUTPUT** carries ~1 bf16 ULP =
`2^-7 ≈ 0.0078` relative quantization; at a rounding boundary the kernel and oracle
differ by a full ULP. So a bf16 output only passes if **rtol ≥ ~0.0078** (plus the
kernel's own accumulation error → realistically **rtol ≈ 0.01–0.02**).

Current tolerance classes and whether they can pass a bf16 OUTPUT:

| class | rtol | bf16 output OK? |
|---|---|---|
| int8_w8a8 / int4_w4a8 | 0.01 | yes |
| fp8_software_emulated | 0.05 | yes |
| recurrent_chunk | 0.005 | **NO** (0.005 < 0.0078; fails on boundary / large magnitude) |
| attention_flash / cube_matmul_fp16 | 0.001 | no |
| vector_reduce / elementwise / cumsum_scan | 0.001 | no |
| atomic_scatter | 0.0001 | no |

**Consequence: NO bf16-production task (they all use recurrent_chunk / attention_flash
/ vector_reduce / elementwise / atomic_scatter) can have a bf16 OUTPUT under the
current tolerances.** Full dtype alignment is therefore IMPOSSIBLE without
introducing bf16-appropriate tolerances. (bf16 INPUTS are unaffected — both kernel
and oracle upcast the same bf16 bits to fp32, so inputs can switch freely.)

## 3. Per-task feasibility (the 71 not-aligned)

- **Input-only-alignable, gate-safe now (64 of 71)**: switch the bf16 activation
  INPUTS to bf16, keep fp32 OUTPUT. No tolerance change. Covers all
  INPUT_BF16_GAP (31) + REAL_BUG (33). For the 33 `tl.dot` tasks this also makes the
  kernel run the production bf16 dot, while the oracle's `a.float()@b.float()` of the
  same bf16 inputs matches it (bf16 tensor-core products are fp32-exact) — so the
  EXISTING tight tolerance still passes. (Confirm per task on GPU.)
- **Output-dtype-tied (7 of 71)**: `L1_01, L2_15, L2_16, L2_17, L2_20, L2_40, L3_07` —
  the output dtype is tied to an input (via `empty_like(x)` or `out` passed as both
  input and output, e.g. L2_15 `empty_like` / L3_07 out-as-both). So bf16 input ⇒ bf16
  output ⇒ needs the §4 cast+1-ULP policy (not a loose tolerance) to align.
  (Label corrected from "in-place norms" per codex; the 7-task set is confirmed.)
- **Full alignment (bf16 in AND out) for ALL 71**: requires bf16-appropriate
  tolerance classes (§4).

## 4. To fully align: oracle-output-cast + 1-ULP comparison (codex-refined)

**DO NOT add loose 0.01–0.02 bf16 tolerance classes** (codex REFINE, 2026-06-01).
The comparison core already upcasts both tensors to fp32 before `torch.allclose`
(`tolerance.py:348`). So the correct, tight mechanism is:

- the oracle computes in **fp32 internally** (unchanged true reference), then **casts
  its OUTPUT to the production dtype** (bf16/fp16) — `expected = oracle_fp32.to(out_dtype)`;
- the kernel already produces a bf16 output;
- the comparison is then **bf16-vs-bf16** (both upcast to fp32 inside the gate), so the
  two values are either identical or differ by **≥1 bf16 ULP** — compare with a
  **1-ULP-tight policy**, NOT a loose tolerance.

| compare | policy |
|---|---|
| kernel bf16 vs oracle fp32→bf16 (preferred) | `max_ulps = 1` (exact ULP count) |
| attention / recurrent exception | `max_ulps = 2` — ONLY after GPU evidence shows online-softmax / recurrent needs it |
| allclose fallback, 1 ULP | `rtol = 0.0078125` (= 1 bf16 ULP), `atol ≈ 1e-7` (NOT 0.0078) |
| fp16 production output | `rtol = 0.000977` (1 fp16 ULP), `atol ≈ 1e-7` |

Why this is tight, not loose: a correct kernel's bf16 output rounds to the SAME bf16
codepoint as the oracle's fp32→bf16 (or ±1 ULP at a boundary) → PASS. A real bug that
perturbs the fp32 result by **> 1 bf16 ULP** changes the bf16 codepoint → FAIL. A bug
**smaller** than 1 bf16 ULP is unrepresentable in bf16 output anyway (it does not exist
in production), so missing it is correct, not a gap. So this catches EVERY bf16-visible
bug with no loose-tolerance bug-miss risk. The L2_23/24 off-by-one (~5%) is ≫1 ULP →
still BLOCKED. (Implement `max_ulps` as a small comparison mode for exactness; the
allclose `rtol=1-ULP` fallback is acceptable but slightly over-accepts near binade
boundaries.)

`atol` note (codex): use a tiny absolute floor (~1e-7), NOT `atol≈rtol` — a fixed
`atol=0.01` would let `expected=0` accept `0.01`, which is not bf16 rounding.

**Residual:** the bf16 track still cannot see sub-bf16-ULP bugs; the **unchanged fp32
logic track (0.001)** is the mitigation — that is the whole point of the hybrid.

## 5. Recommended approach (rigorous)

**Hybrid two-track per bf16-production task** — keep the fine gate AND gain dtype
fidelity:
1. **fp32 logic track (scored, unchanged)**: the existing fp32 fixture + 0.001
   tolerance stays as the precise correctness gate. (This is what already verifies
   the 103 today; per the audit it is logically complete for the GPU stage.)
2. **bf16 production track (new, dtype-faithful)**: a second fixture with bf16 in +
   bf16 out + the matching `bf16_*` tolerance, exercising the exact production dtype
   path (incl. the narrowing store). Run on GPU; it is the dtype-aligned run the user
   asked for.

If a single track is preferred instead, switch the 71 to bf16 in+out + `bf16_*`
tolerances and retire the fp32 fixtures — simpler, but loses the 0.001 logic gate.

## 6. Strict execution sequence (codex-refined; do NOT run before each step is done)

1. **Track decided: HYBRID** (keep fp32 scored gate + add bf16/fp16 production track).
2. **Implement the production-dtype comparison policy** (ONE mechanism, not 5 loose
   classes): oracle output cast to the task's production output dtype + a 1-ULP
   comparison. Preferred = a small `max_ulps` mode in `tolerance.py`; fallback = a
   `prod_dtype_1ulp` class with `rtol = 1 ULP` (bf16 0.0078125 / fp16 0.000977),
   `atol ≈ 1e-7`. Unit-test it (correct→pass, 2-ULP shift→fail, expected-0 case).
3. **Resolve each task's real production OUTPUT dtype** (bf16 / fp16 / fp8) from the
   call site — NOT assume bf16 for all 71 (some are fp16/quant).
4. **Per task, author the production-track fixture**: bf16/fp16 inputs + production
   output dtype; oracle returns `fp32_result.to(out_dtype)`; assign the 1-ULP policy.
   Keep the existing fp32 fixture + 0.001 tolerance as the scored logic track.
5. **Local validate** each (schema + oracle smoke + severance). NO GPU yet.
6. **codex review** the comparison-policy impl + a sample of the production fixtures
   (batch infra + tolerance-policy change → mandatory review).
7. **GPU run** the production track as one batch; confirm every task PASSES at its
   production dtype. Only after this is the run "dtype fully aligned".
8. **Audit**: a task that newly FAILS the bf16 track but PASSES the fp32 track = a real
   bf16-precision-path issue to investigate (genuine bench finding), not a fixture bug.

## 7. Scope / effort

| step | tasks | effort |
|---|---|---|
| tolerance classes | — | ~1 file + tests, 30 min |
| input-only align (if chosen) | 64 | meta+shapes each, ~2–3 h |
| full bf16 align (in+out+tol) | 71 | meta+shapes+tol+output each + per-task GPU confirm, multi-session |
| in-place norms (7) | 7 | need bf16_reduce; output becomes bf16 |
| GPU re-grade | ~71 | 1–2 daemon batches |

## 8. Honest bottom line (codex-refined)

- "完全对齐 dtype" (production in **and** out) for the 71 is achieved by the **HYBRID**:
  keep the fp32 + 0.001 fixtures as the scored logic gate, AND add a production-dtype
  track per task (bf16/fp16 in+out; oracle output cast to production dtype; **1-ULP**
  comparison).
- The earlier idea of "add loose 0.01–0.02 bf16 tolerance classes" is **dropped**
  (codex): instead the oracle casts its output to the production dtype and we compare
  bf16-vs-bf16 with a **tight 1-ULP** policy — no loose tolerance, no bug-miss beyond
  what bf16 itself cannot represent. The unchanged fp32 track covers sub-ULP logic.
- bf16 INPUTS are gate-safe even for the `tl.dot` kernels (codex confirmed fp32
  accumulation), so they can move now; the OUTPUT cast + 1-ULP policy is the new piece.
- The 32 already-aligned/faithful tasks need nothing. Not all 71 are bf16 — resolve
  each task's real production output dtype (some fp16/quant).

**No code changed; no run started. Plan is codex-reviewed (REFINE applied). Awaiting
your go-ahead to start §6 step 2 (implement the 1-ULP comparison policy).**
