# Dtype-fidelity review — do the 100 tasks use fp32 where upstream is bf16?

**Status: review + plan only. No task was modified.**
**Date: 2026-06-01.**

## The question

While authoring 3 new L2 tasks we noticed one (gemma_qkv_rmsnorm) was authored
with `float32` inputs although SGLang runs it on `bfloat16`. The question: do the
existing 100 tasks share this gap, and if so, what is the fix?

## Finding (data)

103 scored tasks (100 + 3 new). Input-dtype profile from `meta.param_constraints.dtypes`:

| Profile | Count | Meaning |
|---|---:|---|
| `int/quant` (int8/fp8 + fp scales) | 57 | quantization kernels — dtype already matches upstream W8A8/W4A8 |
| `ONLY_fp32` | 42 | every input is fp32 |
| `has bf16/fp16` | 2 | already mixed/bf16 (`L1_09_quantize_k_cache_fast`, new `L2_41`) |
| mixed/other | ~2 | — |

The 42 `ONLY_fp32` tasks by category:

| Category | n | Tolerance class(es) |
|---|---:|---|
| ssm | 13 | recurrent_chunk |
| normalization/norm | 7 | vector_reduce |
| attention | 6 | attention_flash |
| elementwise | 5 | elementwise |
| moe | 3 | atomic_scatter |
| gemm | 2 | cube_matmul_fp16 |
| quantization | 2 | int8_quant_round / vector_reduce |
| convolution / sampling / other | 4 | mixed |

So yes — ~40 tasks use fp32 where the production kernel is invoked on bf16/fp16
activations.

## The crucial distinction: this is a COVERAGE gap, not a CORRECTNESS bug

Every one of these Triton kernels **upcasts its inputs to fp32 on load**
(`tl.load(...).to(tl.float32)`) and accumulates in fp32. So an fp32-input fixture
exercises the kernel's **core numerical logic faithfully** — the oracle-vs-kernel
comparison still catches any logic error (off-by-one, wrong reduction axis, bad
masking, etc.). The 42 fp32 tasks are **not silently wrong**.

What an fp32 fixture does NOT exercise:
1. the **bf16→fp32 upcast LOAD** path, and
2. the **fp32→bf16 narrowing STORE** of the output.

Item 2 is the one that actually matters for GPU→NPU migration (910B's bf16
rounding / mixed-precision accumulation differ from CUDA), so the gap is real —
but it is a *fidelity / path-coverage* gap, not a correctness defect.

## Why fp32 is largely FORCED, not a shortcut — the tolerance gate

The oracle-consistency gate is an **element-wise `torch.allclose(actual, expected,
atol, rtol)`** over the full tensor (`triton_ascend_bench/tolerance.py:239`). It
fails if **any** element exceeds `atol + rtol*|expected|`. The per-class budgets:

| Tolerance class | atol | rtol | bf16 output (≈0.4% rel noise) viable? |
|---|---|---|---|
| int8_w8a8 | 0.05 | 0.01 | **yes** (1% > 0.4%) |
| recurrent_chunk | 0.005 | 0.005 | borderline (fails at large magnitude: 1 bf16 ULP at \|x\|≈350 is ≈2 > allowed ≈1.8) |
| attention_flash | 0.005 | 0.001 | no |
| cube_matmul_fp16 | 0.005 | 0.001 | no |
| vector_reduce | 0.001 | 0.001 | no |
| elementwise | 0.001 | 0.001 | no |
| cumsum_scan | 0.001 | 0.001 | no |
| atomic_scatter | 0.0001 | 0.0001 | no |
| int8_quant_round | 1e-5 | 1e-5 | no (int-exact) |

bf16's relative quantization is ≈2^-8 ≈ 0.39%. That **exceeds rtol for every class
except `int8_w8a8`**. So a bf16 **output** physically cannot pass the current gates
(outside int8_w8a8): the kernel would be correct yet `allclose` fails on the
boundary elements. **fp32 output is required by the tight-tolerance + element-wise
design, not a corner cut.**

For an **in-place** op (norms: input buffer *is* the output buffer), input dtype ==
output dtype, so fp32 is forced for **both**. That is exactly why all existing norm
tasks (`L2_15/16/17`, `L1_12/13/14`) are fp32 — and why the new `L2_40_gemma_qkv`
must stay fp32 too.

## What CAN be improved (the bounded, real gap)

For a **non-in-place** op (separate output buffer) whose upstream inputs are bf16
activations, the **INPUTS can be bf16 while the OUTPUT stays fp32**:

- bf16 inputs do **not** cause oracle-vs-kernel divergence — the fixture builds one
  bf16 tensor, handed to both kernel and oracle, and both upcast it to fp32
  identically. So the gate is unaffected.
- This exercises the bf16→fp32 **load** path (item 1 above) and aligns the fixture
  to the production activation dtype.

This is exactly the pattern used for the new `L2_41_selective_state_update`:
bf16 activations (`x, dt, z, B, C`) + fp32 parameters/state/output (`A, D, dt_bias,
state, out`), graded under `recurrent_chunk`.

Candidates for this cheap input-fidelity upgrade: the **13 ssm** tasks
(`L2_02/03/18/19`, `L1_40`, `L3_13/14/15/16/17/19/21/23`), the **2 gemm** tasks
(`L2_13/14`, if a looser-but-justified tolerance is used), and several **attention**
and **elementwise** tasks where the op is not in-place.

## The deeper tradeoff (the actual decision)

There is a genuine tension the bench has implicitly resolved toward precision:

- **Tight tolerance + fp32 fixtures** → *precise* logic verification (rtol 1e-3
  catches subtle kernel bugs), but **zero** bf16-path / narrowing-store coverage.
- **bf16 fixtures + bf16-aware tolerance** → *realistic* dtype coverage, but a
  looser gate (rtol ~1e-2) that could miss subtle logic bugs — and loosening
  tolerance is exactly what the bench forbids elsewhere.

The narrowing store is fundamentally a **910B-stage** concern (the NPU's bf16
rounding is what differs); gating it on a CUDA daemon with a relaxed tolerance has
limited value and real downside (false PASS).

## Options

**Option A — input-fidelity only (recommended, low-risk).**
Keep fp32 outputs (forced). For non-in-place tasks whose upstream inputs are bf16,
switch **inputs** to bf16; outputs stay fp32. No tolerance change, no gate risk;
exercises the bf16-load path; aligns fixtures to production activation dtype.
Scope: ~13 ssm + a few gemm/attention/elementwise ≈ 15–20 tasks. Each edit is
`meta.param_constraints.dtypes` + `shapes.yaml` (no oracle/kernel/tolerance change),
then re-validate + one GPU re-grade batch.

**Option B — production-dtype second track (ambitious).**
Add, per applicable task, a SECONDARY bf16-in/bf16-out fixture with an explicitly
justified bf16-aware tolerance (e.g. rtol 1e-2), kept **non-scored** (like the
`_corrected/` variants). Tests the narrowing store. Cost: per-class tolerance
design + double fixtures + risk of tolerance-shopping. Higher effort; the payoff
(narrowing-store coverage on CUDA) is partial because the real divergence is on NPU.

**Option C — status quo + explicit policy doc (zero change).**
Declare fp32 the bench's deliberate *verification* convention (a precise logic gate)
and document that bf16 narrowing-store fidelity is owned by the 910B migration
stage, not the GPU oracle gate. No code change.

## Recommendation

**A + C**: do the cheap, zero-gate-risk input-fidelity upgrade (Option A) on the
non-in-place bf16-activation tasks, and write down the dtype policy explicitly
(Option C) so the fp32 outputs and the in-place fp32 norms are documented as a
deliberate, tolerance-driven choice rather than an oversight. Defer Option B unless
narrowing-store coverage becomes a stated requirement.

This keeps every existing precise gate intact, improves fidelity where it is free,
and is honest about what the GPU oracle can and cannot gate.

## Consistency note for the 3 new tasks

- `L2_39_scaled_mm`: int8 in / fp32 scales / bf16 out — already matches vLLM W8A8
  (int8_w8a8 absorbs the bf16 output). No change.
- `L2_41_selective_state_update`: already authored to the Option-A pattern (bf16
  activations + fp32 params/state/out). If Option A is adopted, the 13 existing ssm
  tasks would be aligned to this; if the bench keeps strict fp32, `L2_41` would be
  reverted to all-fp32 to match its siblings.
- `L2_40_gemma_qkv_rmsnorm`: in-place norm under vector_reduce → fp32 forced,
  consistent with `L2_15/16/17`. No change regardless of option.

## Scope / effort

- Option A: ~15–20 fixture edits (meta + shapes), ~5 min each, + 1 GPU re-grade
  batch. No oracle/kernel/tolerance edits. Reversible.
- Option B: per-class tolerance design + ~N double fixtures — multi-session.
- Option C: ~1 policy doc.

**No task files were modified by this review.**
