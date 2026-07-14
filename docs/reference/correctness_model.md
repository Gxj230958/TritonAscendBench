# Correctness Model

> 中文版：[../zh/reference/correctness_model.md](../zh/reference/correctness_model.md)

How TritonAscendBench decides an operator's output is **correct** — the
precision-alignment standard a ported kernel is graded against, and the error
models that justify each tolerance.

This is the mechanism companion to [`tolerance_policy.md`](tolerance_policy.md)
(the per-class cheat-sheet) and to [`ARCHITECTURE.md`](../ARCHITECTURE.md) (the
end-to-end data flow). The authoritative implementation is
`ascendbench/correctness/tolerance.py`; the per-class numbers live in
`ascendbench/policies/tolerance_policy.yaml`.

A note on what is *proven* vs *reasoned*: the policy file states each class's
`atol`/`rtol` bound and a one-line note — it does **not** carry a derivation.
The error-model rationales below (e.g. "why bf16 GEMM is 5e-3") are standard
numerical-analysis reasoning that explains *why a bound of that magnitude is
appropriate*; they are not bounds the code computes or asserts. Where a figure
is an external/standard fact rather than something the code checks, it is called
out as such.

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
  candidate's output with a matching tensor shape. The submitter grade
  (`eval.py`) iterates the oracle keys and ignores any *extra* candidate keys;
  the stricter provider oracle-equivalence cross-check
  (`run_oracle_equivalence_gpu.py`) instead requires the kernel and reference
  key-sets to match exactly. A shape mismatch is an immediate fail — tensors are
  never cast or broadcast to force a comparison.
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

## 3. Numeric dtype foundations

The tolerance standard rests on the IEEE-754 (and finite fp8) bit layouts. The
comparison code does **not** derive machine epsilon itself — float dtypes are
graded by `torch.allclose(atol, rtol)` with budgets from `tolerance_policy.yaml`.
Only the two dtype-aware ULP paths (`_int_atol_compare`, `_fp8_ulp_compare`)
manipulate raw bits.

### Bit layouts

| dtype | sign | exponent | mantissa (stored) | total | encoding |
|-------|------|----------|-------------------|-------|----------|
| fp32 | 1 | 8 | 23 | 32 | IEEE-754 binary32 |
| bf16 | 1 | 8 | 7 | 16 | truncated binary32 (same 8-bit exponent) |
| fp16 | 1 | 5 | 10 | 16 | IEEE-754 binary16 |
| fp8 e4m3fn | 1 | 4 | 3 | 8 | finite-only ("fn"), no Inf encoding |
| int8 | — | — | — | 8 | two's complement, range −128..127 |

bf16 shares fp32's 8-bit exponent (same dynamic range) but keeps only 7 mantissa
bits; fp16 trades exponent range (5 bits) for 10 mantissa bits.

### Two epsilons — keep them straight

Error magnitudes below use two related quantities; the distinction is a 2×
factor that otherwise causes confusion:

- **Machine epsilon `ε` = `2^-(stored mantissa)`** — the gap between 1.0 and the
  next representable value (one ULP at the 1.0 binade).
- **Unit roundoff `u` = `ε/2` = `2^-(stored+1)`** — the *worst-case relative
  error of a single correctly-rounded operation*. Error budgets are reasoned in
  terms of `u`.

| dtype | stored mantissa | `ε = 2^-m` | `u = 2^-(m+1)` |
|-------|-----------------|------------|----------------|
| fp32 | 23 | 1.19e-7 | 5.96e-8 |
| bf16 | 7 | 7.81e-3 | 3.91e-3 |
| fp16 | 10 | 9.77e-4 | 4.88e-4 |
| fp8 e4m3 | 3 | 0.125 | 0.0625 |

(These are standard IEEE-754 / OCP-fp8 values, computed independently; the code
does not assert them.) The coarse bf16/fp16 grids are why the float classes
carry loose `atol`/`rtol`, and why fp8 — where `ε` at 1.0 is a full 0.125 — is
graded on representable *codes*, not values.

### Integer ULP

For integer outputs, "1 ULP" is an absolute integer difference.
`_int_atol_compare` casts both tensors to **int64 first**, then computes
`(a − e).abs()`, so the subtraction cannot overflow the narrow input dtype.
(Concretely, int8 `(−128) − (127)` wraps to `+1` in-dtype but is `−255` after
widening — the widening is what makes `|a−e| ≤ int_atol` correct.)

### fp8 e4m3 ordered sign-magnitude code

`_fp8_ulp_compare` reinterprets each fp8 byte as `uint8 → int16` and builds a
signed ordered code so adjacent representable values are exactly distance 1:

- `mag = b & 0x7F` keeps the low 7 bits (exponent+mantissa). Because the exponent
  occupies the higher bits and the mantissa the lower bits, this 7-bit value is
  **monotonic in |x|** (a property of the IEEE bit ordering).
- `code = −mag if sign-bit else mag`; `mag == 0 → 0` canonicalises **+0 (0x00)
  and −0 (0x80)** to the same code so signed zeros compare equal.
- The single pattern `(b & 0x7F) == 0x7F` is **NaN** — e4m3fn is finite-only,
  there is no Inf encoding. NaN positions must match exactly.

**Worked 1-ULP step:** `1.0` encodes to byte `0b00111000` (= 56); the next
representable value `1.125` encodes to `0b00111001` (= 57). Ordered codes 56 and
57 differ by 1, and `1.125 − 1.0 = 0.125` is exactly one fp8 ULP at the exponent
of 1.0 (`2^0 · 2^-3`). A near-midpoint quant result that rounds to either
neighbour therefore shows up as ordered-code distance 1 — the fp8 analogue of
`int_atol = 1`.

---

## 4. `compare()` — the dtype routing

`tolerance.compare(actual, expected, cls)` dispatches by dtype **before** any
numeric check, because the right notion of "close" differs by dtype. In order:

1. **Shape mismatch** → fail (never cast).
2. **Integer dtype** (`bool`/`uint8`/`int8`/`int16`/`int32`/`int64`):
   - `cls.bit_exact`, or the class declares no `int_atol` → **`torch.equal`**.
   - else → **integer ULP**: `|actual − expected| ≤ int_atol`.
   - An explicit `bit_exact` always wins over `int_atol`.
   - *Why never float-cast:* casting an int output to fp32 and running `allclose`
     silently rounds e.g. `int64(2²⁴+1)` to `2²⁴`, turning an off-by-one into a
     false pass.
3. **fp8** (the class sets `fp8_ulp`):
   - both sides fp8 → **ordered-code ULP** (§3 / §4 below).
   - exactly one side fp8 → fail. An fp8 graded output must be compared *as* fp8;
     fp32-casting one side to reach a pass is forbidden.
4. **`cls.bit_exact`** (non-integer) → `torch.equal`.
5. **Otherwise (float)** → cast **both** sides to fp32, then:
   - a **sampled fast-fail** (only when `numel > 1024` and `sample_fraction < 1`):
     reject early if even a random sample fails, but
   - the **authoritative** verdict is always the **full** `allclose` over every
     element — a sample passing can never be the final answer (NaN/Inf or drift
     outside the sample window must still be caught).

---

## 5. The four comparators

| Comparator | dtype | Pass criterion |
|------------|-------|----------------|
| **float allclose** (`_check_full`) | fp16/bf16/fp32 | inf and NaN **positions** must match (`inf_must_match`/`nan_must_match`, both default true); then `torch.allclose(atol, rtol)` over the finite elements. |
| **integer ULP** (`_int_atol_compare`) | int* | dtypes must be **identical**; `\|a−e\|` in widened `int64` `≤ int_atol`. |
| **fp8 ordered-code ULP** (`_fp8_ulp_compare`) | `float8_e4m3fn` | dtypes identical; ordered e4m3 code (§3); pass iff `max_ordered_dist ≤ ulp` **and** the off-by-one **fraction ≤ 1%** (`max_frac`). |
| **bit-exact** (`_bit_exact_compare`) | any | dtypes identical; `torch.equal`. |

Two design notes:

- **fp8's two-part test.** A single off-by-one ordered-code step is genuine
  boundary-rounding ambiguity (a value within the `silu`/`exp` epsilon of an fp8
  bucket midpoint can round either way). But a *large fraction* of off-by-one
  elements is a systematic rounding-mode bias (a real bug), so the
  `off_frac ≤ 1%` cap rejects it even when every individual step is only 1 ULP.
- **dtype equality is enforced by the integer / fp8 / bit-exact comparators**,
  not by the float path. The float path casts **both** sides to fp32, so a
  `fp32`-vs-`bf16` *float* output difference is not caught by `compare` alone — a
  strict `VERIFIED` float compare does not itself assert the output dtype. It is
  caught elsewhere: the severance anchor's dtype check (§10), and — only when a
  strict-failing shape is then considered for ceiling acceptance — the **ceiling
  dtype gate** (§8).

---

## 6. Per-class error models

`tolerance_policy.yaml` defines **14 classes**. The grading path resolves one by
name (`eval.py` → `load_policy().get(tolerance_class)`; the provider GPU
cross-check → `_tolerance_for(...)`, with a permissive `vector_reduce`-style
fallback if the name is absent). The rationales below explain *why* each bound
sits where it does; they are error-model reasoning, not bounds the policy proves.

**Pointwise — `elementwise` (1e-3 / 1e-3).** silu/gelu/bias and similar. One
output rounding plus the transcendental's own library approximation (libdevice
vs torch). A single fp16 rounding is `≈ u_fp16 ≈ 4.9e-4`; 1e-3 covers it with
~2× margin. No accumulation → the tightest non-exact tier. Examples:
`L2_03_act_and_mul`, `L2_05_gelu_and_mul`, `L1_13_fused_softcap`.

**Logit-add — `elementwise_exact_or_inf_safe` (1e-6 / 0, inf+nan must match).**
A single additive op on logits is essentially exact in fp32, so `rtol=0` plus a
near-machine `atol` is right; the load-bearing constraint is that `−inf` mask
positions match **exactly** (a wrong-masked logit changes the softmax support).
This is the only class that *names* the inf/nan requirement, though `_check_full`
enforces it for every float class. Examples: `L1_01_bias`, the `C1_identity_trap`
canary.

**Reduce / softmax — `vector_reduce` (1e-3 / 1e-3).** RMSNorm/LayerNorm/softmax
reduce over the hidden dim in an fp32 accumulator (`u_fp32 ≈ 6e-8`, accumulation
jitter negligible); the dominant error is the final output downcast plus `exp`
library divergence. Same tier as pointwise. Examples: `L1_12_rms_norm`,
`L1_28_fused_rmsnorm`. (`cumsum_scan` and `mrope`, both 1e-3/1e-3, sit here by the
same single-rounding logic; see §7 for `cumsum_scan`'s zero-task status.)

**Matmul / GEMM — `cube_matmul_fp16` (5e-3 / 1e-3).** K-dim products accumulate
in fp32 (near-exact), but the **output downcast to bf16** carries `≈ u_bf16 ≈
3.9e-3`; 5e-3 sits just above that and also absorbs cube-unit / TF32-style
accumulation-order differences. This is the looser GEMM tier (vs 1e-3 pointwise)
specifically because of bf16 output granularity. (The class name says `fp16` but
its note specifies an fp32 accumulator and most assigned tasks use bf16 I/O; if a
task's output is genuinely fp16, `u_fp16 ≈ 4.9e-4` and 5e-3 is even looser.)
Examples: `L2_19_awq_gemm`, `L2_16_sgemm_lora_a`.

**Flash attention — `attention_flash` (5e-3 / 1e-3).** Same numbers as GEMM
because it *is* two fp32-accumulator matmuls (QKᵀ, PV) plus a bf16 output, with
extra fp ops from the online-softmax rescale (see §7). `rtol` stays at 1e-3
because softmax self-normalises. Examples: `L3_13_kernel_unified_attention_3d`,
`L3_16_fwd`.

**Int quant — `int8_w8a8` (5e-2 / 1e-2), `int4_w4a8` (5e-2 / 1e-2),
`int8_quant_round` (1e-5 / 1e-5, `int_atol 1`).** The W8A8/W4A8 dequant tier is
loose (`rtol 1e-2 = 1%`) because the int32 K-accumulation is exact but the
**per-token/per-group fp scale multiply over a wide dynamic range** amplifies
absolute error. `int8_quant_round` is the file's cleanest documented derivation:
its int8 `q` output is graded by `int_atol = 1` because fp32 banker's-rounding
(`torch.round`) vs Triton round-half-away-from-zero **differ by at most one step
on ties**; the `1e-5` atol/rtol grade only the auxiliary fp32 `scale` output
(`≈ max(|x|)/127`, a near-exact reduction).

**fp8 — `fp8_software_emulated` (1e-1 / 5e-2, `fp8_ulp 1`).** The 910B has no fp8
cube (§9), so it is modeled in the Roofline as vector + HBM. The fp8 e4m3 tensor
is graded on **ordered-code distance ≤ 1** (the fp8 analogue of `int_atol = 1`,
§3) with the 1% systematic-bias cap; the loose value `atol/rtol` grade only the
auxiliary fp32 scale outputs, reflecting e4m3's coarse `ε ≈ 12.5%`. Examples:
`L2_13_silu_mul_fp8_quant_deep_gemm`, `L1_09_quantize_k_cache_fast`.

**Recurrent / chunk — `recurrent_chunk` (5e-3 / 5e-3).** Mamba/GDN/KDA chunk
pipelines carry hidden state chunk-to-chunk; error accumulates across chunks
(§7). The **only non-quantized float class to widen `rtol`** above 1e-3, because
the carried, un-normalised state grows both absolute and relative error.
Examples: `L1_27_chunk_cumsum_fwd`, `L3_01_chunk_scan_fwd`.

**Sampling — `sampling_rng` (0 / 0, `bit_exact`).** When the seed matches, the
Philox/Triton draw stream must be **identical**, so it routes through
`torch.equal`; any nonzero tolerance would mask a wrong RNG stream. Examples:
`L2_08_gumbel_sample`, `L2_11_rejection_sample`.

**Atomic-scatter — `atomic_scatter` (1e-4 / 1e-4, `rerun_count 3`).** Atomic-add
scatter has non-deterministic summation order → fp non-associativity gives
run-to-run jitter; the tight 1e-4 reflects that individual atomic contributions
are small. Most tasks here emit **integer index** outputs, which route to
bit-exact regardless of the 1e-4 — the float tolerance only bites on float
scatter outputs. `non_deterministic`/`rerun_count` are metadata, not consumed by
`compare()` itself. Examples: `L1_24_compute_seg_indptr_triton`,
`L2_28_moe_align_block_size_stage2`.

---

## 7. Error accumulation and numerical stability

A single pointwise op is graded at the tightest float band (1e-3). Pipelined ops
get **wider** bands because they chain many rounding-injecting steps. The
compounding-error arguments here are standard numerical reasoning — the policy
widens the band, but no code computes an N-step bound.

| Class | atol | rtol | story |
|-------|------|------|-------|
| `elementwise` (baseline) | 1e-3 | 1e-3 | one rounding step |
| `cumsum_scan` | 1e-3 | 1e-3 | linear prefix sum — no widening |
| `attention_flash` | **5e-3** | 1e-3 | online softmax + 2 fp32 GEMMs |
| `recurrent_chunk` | **5e-3** | **5e-3** | state carried across chunks |

**Online softmax (`attention_flash`).** Flash attention never materialises the
full score row: it streams K/V blocks while keeping a running max `m` and
denominator `l`, rescaling the V-accumulator by `exp(m_old − m_new)` at every
block boundary. The oracle is the naive single-pass fp32 softmax; the kernel's
streaming reorder produces a different rounding sequence for the identical math.
One task's contract states a heuristic drift model — `≈ ε_fp16 × sequence_length`
(`L3_16_fwd/contract.md`), an author-stated estimate, not a code-proven bound.
`atol` widens to the GEMM band (two fp32-accumulated matmuls), but `rtol` stays
1e-3 because softmax is self-normalising (the output is a convex combination of V
rows, no relative blow-up). *Scope note:* several `attention_flash`-labelled L1
tasks (`L1_20`/`L1_21` kv-index builders, `L1_22` page-table transform) are
**integer paging/index kernels** that inherit the class by serving-domain, not
because they accumulate softmax error.

**Chunked recurrence (`recurrent_chunk`).** Mamba/GDN/KDA pipelines carry a
hidden state across chunks, so each chunk's rounding error feeds the next and is
re-scaled — error compounds per chunk. The oracles model this directly (e.g.
`L1_26_state_passing_fwd`: `S_c = exp(dA_cumsum[…,−1]) · S_{c−1} + states[c]`
looped over the chunk span) and inject the **same per-chunk reduced-precision
rounding the kernel uses** (`L3_01` rounds operands to bf16; `L3_22` drops the low
13 mantissa bits via `_to_tf32`, §10). Because the recurrence is multiplicative
and the carried state is not normalised, both absolute and relative error grow
chunk-over-chunk — hence the lone float class to widen `rtol` to 5e-3.

**Prefix scan (`cumsum_scan`).** A pure prefix scan is a *linear* fp32
accumulation, so running error grows only additively (`~N · u_fp32`) and stays
far under the bf16/fp16 output-rounding floor — hence it keeps the 1e-3 pointwise
band. In practice **no task uses `cumsum_scan`**: the bench's chunked-cumsum
kernels (`L1_27`, `L2_35`) are graded under `recurrent_chunk` because their
per-chunk `cumsum` is not a terminal output but feeds an `exp()`-decay
recurrence, inheriting the wider chunk band.

---

## 8. Ceiling acceptance — scoped large-shape relaxation

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

- **Mode A — units cap.** A task in `tol_units_tasks` is accepted iff it has a
  non-empty failing-key set, no compare error, correct dtype, all stats finite,
  and `max_tol_units ≤ tol_units_cap` (currently **8.0**). A clean dtype ceiling
  sits a few units above 1; the genuine upstream bug in `L2_23/24` off-by-one
  sits at 1000–6200 — over 100× past the cap — so it stays strict by construction.
- **Mode B — physical-scale bound.** For the rare task whose worst element is a
  legitimate near-cancellation point (near-zero ref, `atol`-dominated), listed in
  `physical_tasks` (currently `L3_22`). Accepted iff **every** failing output key
  satisfies `max_abs ≤ physical_factor · max|ref[key]| · ulp_rel`.

Everything not on the allowlist stays strict. `validate_corpus_consistency`
enforces that only allowlisted tasks carry the ceiling status, with full shape
coverage, `golden_source = dtype_ceiling_accepted`, and no `golden_sha256`.

---

## 9. GPU → NPU migration precision specifics

This is a migration benchmark: it grades GPU-Triton kernels on a CUDA device but
*targets* an Ascend 910B. Two device facts drive the precision design.

**910B numeric characteristics.** The 910B splits compute across a **Cube
(matrix) unit** and a **Vector (SIMD) unit**. The Roofline peak table reflects
exactly this: Cube `fp16`/`bf16` and `int8` classes, a single `vector` class, and
**deliberately no `fp8_cube` class**. `NpuBackend.dtype_supported` returns
`False` for `float8_e4m3fn`/`e5m2` with the comment *"910B does not have FP8 cube
arithmetic; software emulation only"* — so on the NPU an fp8 kernel runs on the
Vector / software-cast path, not a tensor-core fp8 MAC. (This is a
hardware-capability claim grounded in the project's own code comment + the
peak-table shape, not an external datasheet.) Cube matmuls accumulate in fp32
(hence `cube_matmul_fp16` at 5e-3); fp8 has no such hardware accumulator, which is
why `fp8_software_emulated` is wide and the fp8 tensor is graded on ordered-code
ULP (§6). `dtype_supported = False` is a hardware flag, **not** a denominator
escape — fp8 tasks stay graded under that class (eligibility is governed solely
by `eligibility.py`).

**The provider GPU cross-check is a PROXY, not a guarantee.** The reference
grade runs on a **CUDA GPU, not a 910B** (the CUDA peak table is labelled
"self-check, not for 910B scoring"). So GPU oracle-equivalence proves the
extracted GPU-Triton kernel runs and matches its PyTorch oracle **on CUDA** —
faithful extraction plus correct fp32-model logic. It does **not** prove
reproduction on Ascend silicon. A separate 910B stage owns that. What still
differs on the real NPU: fp8 arithmetic (software-emulated vs CUDA hardware fp8),
Cube-vs-Vector rounding and accumulator behaviour, `triton-ascend` JIT codegen,
and absolute throughput vs the (academic-spec, provisional) vector peak. The NPU
runtime seams exist but are stubbed off-device — `NpuBackend.event_timer` returns
`None` without `torch_npu`, and the NPU profiler raises *"only available on the
910B machine"* — so kernel-event / non-default-stream signals are observable only
on real hardware.

---

## 10. Provider-side oracle-alignment safeguards

The CPU-side oracle must reproduce what the *frozen GPU kernel actually computes*,
not the mathematically ideal result. Two mechanisms solve opposite-signed
problems: TF32 truncation is the **correct** emulation of the hardware
tensor-core `tl.dot`; narrow-float truncation is the **wrong** behaviour of the
CPU interpreter that the anchor must route around.

### TF32 emulation — the bit math

An fp32 value has 23 mantissa bits; the TF32 tensor-core format keeps the full
8-bit exponent but only the top **10** mantissa bits. The oracle helper drops
exactly the low 13 (`23 − 13 = 10`):

```python
def _to_tf32(x):  # in the task's own oracle_pytorch.py
    return (x.contiguous().view(torch.int32) & ~((1 << 13) - 1)).view(torch.float32)
```

`~((1<<13)-1)` clears mantissa bits 0–12, leaving bits 13–22 (= 10 bits). It
never touches sign or exponent, so this is **truncation toward zero**, not RNE.
The truncation choice is load-bearing: Triton's default `tl.dot` truncates the
low mantissa bits while cuBLAS rounds RNE, and only truncation reproduces the
measured hardware (one task's oracle records a GPU-anchor match at `max_rel ≈
21.4` under truncation, where RNE would have been ~3× too small). Tolerance is
**unchanged** — TF32 emulation aligns the oracle with the kernel's real
arithmetic so the residual gap is only fp32 accumulation-order noise. `_to_tf32`
is wrapped around **both** operands of any matmul whose kernel runs an *unpinned*
fp32 `tl.dot` (no `allow_tf32=False`, no `input_precision="ieee"`); it lives
per-task in the oracle (confirmed examples: `L2_14`, `L2_19`, `L2_21`, `L3_21`,
`L3_22`, `L3_27`).

### Narrow-float anchor — interpreter mis-rounding

The standalone-severance anchor runs under `TRITON_INTERPRET=1` (CPU, GPU-free).
The interpreter mis-rounds every **narrowing** float store: `fp32 → bf16/fp16`
**truncates toward zero** instead of RNE (`100.3 → 100.0` instead of `100.5`), and
`fp32 → fp8` mis-rounds both ways. So for any task whose oracle output is
fp8/fp16/bf16, the interpret verdict is untrustworthy *whether it passed or
failed* — a correct kernel can false-fail and a wrong kernel can false-pass by
truncating into the oracle value. The runner flags this as
`narrow_float_output: True` (it detects an oracle output dtype string containing
`float8` or `float16`, the latter also matching `bfloat16`).

When the flag is set **and a GPU is available**, the parent re-runs the **same**
anchor **once** on the real device with the task's **unchanged** tolerance and
adopts the GPU verdict (the interpreter result is discarded; `interpret=False`
cannot re-enter the branch, so the rescue fires at most once). The rescue
corrects only the **rounding model**, never the comparison — a genuinely wrong
operator still fails at the same atol/rtol, and a real value/dtype/structure
mismatch returns a fail without ever reaching the rescue path.

*Host dependency, stated honestly:* the rescue is gated on a GPU being present.
In deployment the oracle-consistency anchor runs on the CUDA daemon, so the
rescue engages. On a GPU-free box the flag cannot be resolved by re-run — the
code returns the raw interpret verdict rather than fabricating a CPU result, and
the genuinely fail-closed path (`consistent: false`, "no CUDA device") belongs to
the explicit `interpret=False` request when CUDA is absent. The correctness of a
narrow-float task therefore depends on grading on real hardware, by deployment
convention — not on the interpreter.

### No silent torch fallback

A candidate that produces no real device kernel event fails even when numerically
correct (§2) — a migration must run a kernel on the device.

---

## 11. Output sentinels — declared, not yet wired

`tolerance_policy.yaml` carries a per-dtype `sentinels:` table (its own comment
says "output buffer poisoning detection"): float → `NaN`, int8 → `−128`,
int16 → `−32768`, int32 → `−2147483648`, bool → `0xAA`. The intended design is
the standard "poison the destination first" pattern — pre-fill the output region
with a value the kernel would never write, run it, and flag any region left
holding the sentinel (closing the "read a stale / lucky-zero buffer as a pass"
hole).

**It is not consumed by any code today.** `load_policy()` reads only the
`classes` block and ignores `sentinels`; no module imports a sentinel constant.
There is also no buffer to poison: the kernel contract is **return-based, not
fill-based** (`npu_triton_impl(inputs) -> dict[str, Tensor]`) — the candidate
allocates and returns its own tensors, and the runner never hands it a
pre-allocated destination. So an unwritten-region sentinel scheme has nothing to
attach to as the harness stands.

What the runner relies on against a kernel that skips a region is instead:
shape-mismatch rejection, the strict ULP / bit-exact value comparison over the
returned data, and the NaN/Inf **position** gate in `_check_full` (`inf_must_match`
/`nan_must_match` default true, run before the finite `allclose`). The sentinel
table is best read as design intent for a future buffer-write harness (e.g. an
Ascend-side run that writes into a caller-provided buffer), not a check the
current grader performs.

---

## 12. Where grading sits in the data flow

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
