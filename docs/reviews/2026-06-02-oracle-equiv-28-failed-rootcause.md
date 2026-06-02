# Oracle-equiv (C2) — root-cause of the 22 FAILED + 6 MIXED

GPU batch `bc2full1`: 62 VERIFIED, 22 FAILED, 6 MIXED, 13 schema-gaps.
This doc root-causes the 28 FAILED/MIXED. Evidence cited to file:line.

Harness facts that frame everything:
- R10.a (`oracle_pyref_precompute`) runs the **oracle on CPU** (exact fp32).
- R10.b (`oracle_equivalence_gpu_runner`) runs the **kernel on GPU**.
- Compare is CPU fp32 `torch.allclose(atol,rtol)` per `tolerance_class`
  (`tolerance.py:308` `compare`), integer/bool dtypes go bit-exact
  (`tolerance.py:336`).

## Buckets

### H — Harness bug: R10.b ignores `interface.output_projection`
`oracle_equivalence_gpu_runner.py:692` maps kernel returns to ref keys
**positionally** and `:678` hard-requires `len(output_key_order)==len(kernel_out)`.
A kernel that returns graded output + aux tensors (mean/rstd) therefore yields
`output_keys_kernel != output_keys_ref` → structure-mismatch FAILED (`:760`)
even when the graded tensor is within tolerance.

- **L1_14_layer_norm_fwd_1pass** — `interface.output_projection {arity:3,
  map:{output:0}}` (meta), ref={output} only, diff P1 9.5e-7 / P2 1.4e-6 both
  **inside** vector_reduce (atol 1e-3) yet FAILED. [confirmed]
- Fix: R10.b must select graded outputs via `output_projection.map`
  (kernel_out[idx]→name), ignoring un-mapped aux returns — mirror C1
  (`_pa_oracle_check_runner`). Benefits every multi-output kernel.

### M — Meta bug: wrong `tolerance_class` (changes grading bar → codex sign-off)
GEMM kernels (`tl.dot` present) tagged a non-GEMM class:
- `vector_reduce` (1e-3): L2_01_bmm_chunk_fwd, L2_04_chunked_lora_expand,
  L2_05_chunked_lora_shrink, L2_07_gate_up_lora_b, L2_08_sgemm_lora_a,
  L2_09_sgemm_lora_b, L2_13_bmm
- `atomic_scatter` (1e-4, tightest): L2_30_fused_moe_router_tensorcore,
  L3_09_fused_moe, L3_11_expert_triton, L3_12_moe_mmk
- `atomic_scatter`, W4 quant: L3_10_fused_moe_kernel_gptq_awq → int4_w4a8
- `elementwise` (1e-3): L3_20_recompute_w_u_fwd
Correct class = `cube_matmul_fp16` (atol 5e-3, "GEMM with FP32 accumulator")
or `int4_w4a8`/`int8_w8a8` for quantized MoE. [confirmed tl.dot present;
each task's op-type to be codex-confirmed before reclass]

Non-GEMM mis-class causing false-FAILED:
- **L1_23_penalties_and_temperature** — deterministic fp32 logits tagged
  `sampling_rng` (bit_exact) → 1-ULP (abs 1.9e-6) false-FAILED. → `elementwise`.
- **L1_09 / L2_34 / L2_37** — `float8_e4m3fn` output tagged `vector_reduce`
  (1e-3). → `fp8_software_emulated` (atol 1e-1). L1_09 abs 32 ≈ 1 fp8-ULP near
  top-of-range (±448); may remain a hard fp8 ceiling after reclass. [confirmed
  fp8 output via ref safetensors]

Reclass is a correction to the *published* class for the op type, NOT a new
looser tolerance and NOT editing `tolerance_policy.yaml`. Each must be
independently confirmed as that op type (no masking of real bugs).

### F — Fixture/oracle fidelity (locally fixable)
- **L2_02_chunk_scan_fwd** (abs 591) — builder uses
  `torch.Generator(device=x.device)` + `device=x.device` randn
  (`oracle_pytorch.py:145`, `_pa_launcher.py:122`). R10.a(CPU) and R10.b(GPU)
  draw **different** derived `cb/dt/dA_cumsum/states` for the same seed → kernel
  graded against an oracle built on different inputs. Fix: generate on CPU then
  `.to(x.device)` in both byte-identical copies. Only task with this pattern.
  [confirmed]
- **L1_24_rejection_sample** (abs 0, FAILED) — oracle returns `output_token_ids`
  int64 (`oracle_pytorch.py:52` `.to(torch.long)`); kernel emits int32 → bit_exact
  dtype-mismatch (`tolerance.py:261`), abs computed 0 on the float view. Fix:
  oracle returns the kernel's real dtype (align bench to kernel, never edit
  kernel). [likely — confirm kernel output dtype on GPU]

### P — Precision regime: fp32 fixture → GPU-TF32 vs CPU-exact-fp32 (Phase-2 dtype)
`tl.dot` on fp32 inputs uses TF32 tensor cores on GPU; the CPU oracle is exact
fp32. Residual abs (1.5e-2…5e-2, amplified over recurrences) exceeds even the
correct GEMM/recurrent tolerance. Production runs these on fp16/bf16, where
`cube_matmul_fp16`/`recurrent_chunk` tolerances were calibrated to fit.
Residual after M-reclass: L2_05/07/08/13, L3_09/10/11/12/16, L3_13/L3_17 (deep
recurrence), L3_03, L3_20, L2_30-P2. Status: **NEEDS-PHASE2-DTYPE** (deferred per
owner). Cheap confirm: rerun one GEMM task with an fp16 fixture → expect PASS.
[mechanism confirmed; per-task pass-after-dtype likely]

### E — Errored / no-shape (R10.b launch/import failure; needs GPU error log)
R10.a refs exist → R10.b failed before per-shape compare. Need a verbose GPU
rerun to capture the actual exception (suspected import-time upstream dep or
launch-signature issue).
- L1_01_bias, L1_22_min_p, L1_25_merge_attn_states, L3_04_fwd. [needs GPU]

## Action order
1. **H** output-projection fix in R10.b (codex-review; unblocks L1_14 + any
   multi-output kernel).
2. **M** tolerance reclass (codex-review each op-type; grading bar).
3. **F** L2_02 RNG-device, L1_24 dtype (local).
4. GPU rerun fixed tasks + 4 errored (verbose) — confirm + capture E errors.
5. Cheap fp16 confirm for **P**; mark P Phase-2-deferred honestly.

Not kernel bugs, not oracle-math bugs except where noted; no vendored edits, no
tolerance-value edits, denominator stays 100.

## Codex review refinements (gpt-5.5:xhigh, 2026-06-02)
Overall REFINE. Adopted:
- **L2_30 split out of GEMM/TF32.** It is a router/top-k kernel returning
  `topk_weights` + integer `topk_ids` (top-k *after* `tl.dot`). P2 abs=6.0 is
  **topk_ids identity drift**, a hard integer surface — NOT a float residual.
  Do NOT reclass to cube_matmul, do NOT defer to Phase-2. → NEEDS-INVESTIGATION.
- **L1_09 stays FAILED after fp8 reclass.** `fp8_software_emulated` (atol 0.1,
  rtol 0.05) allows ~22.5 near ±448 < abs 32 → real fp8 ceiling or divergence,
  not auto-fixed. Investigate on GPU (1-ULP at top-of-range vs genuine).
- **L3_13 (abs 13.5) not deferred blindly.** Needs an fp16/bf16 or TF32-emulated
  rerun to confirm TF32 vs a real recurrence divergence before Phase-2 parking.
- H fix is low-risk: currently-VERIFIED projection tasks (L1_31, L2_15) map ALL
  tuple outputs, so projection leaves their verdicts unchanged (regression-check
  both in the rerun).

Applied now (codex safe-now): H output_projection in R10.b; L2_02 CPU-RNG;
L1_24 oracle int32; L1_23→elementwise; fp8 class for L1_09/L2_34/L2_37; GEMM
class for L2_01/04/05/07/08/09/13, L3_09/11/12/20, L3_10→int4_w4a8.
Verdict changes confirmed only by GPU rerun; meta edits do not assert PASS.

## GPU verification results (bc2fix1..5)
Baseline 62 VERIFIED → **83+** after fixes. Newly VERIFIED on real GPU:
- Harness/meta/dtype (bc2fix1-3): L1_14, L1_23, L1_24 (H/reclass/dtype);
  L2_04, L2_09, L3_10, L3_20 (reclass); L1_01, L1_22, L1_25, L3_04 (None-fill
  in-place harness fix — all 4 errored tasks); L1_31, L2_15 (projection
  regression intact).
- TF32 `_to_tf32` oracle emulation (bc2fix3-4): L2_01, L2_05, L2_07, L2_08,
  L2_13, L3_09, L3_11, L3_12 (single-matmul); L2_02 (591→0.1→pass, RNG+TF32),
  L3_17 (pinned-dot exclusion correct).

Two new harness fixes (both GPU-validated, zero regression):
1. R10.b honors `interface.output_projection.map` (multi-output kernels).
2. R10.b treats a `None` return as a fully-in-place kernel and builds the
   candidate dict from the mutated input buffers named by the ref keys.

Remaining tail (genuine anti-mechanization):
- **L3_16 P2**: 8.3e-2 → 1.08e-2 after `_to_tf32` (P1 passes); 2× over
  recurrent_chunk 5e-3 — deeper-shape fp32 accumulation-order residual.
- **L3_13**: 13.5 → ~1.2 after `_to_tf32`, but P1≈P2 (not depth-scaling) →
  a SYSTEMATIC recurrence-contract residual beyond TF32; needs deeper analysis
  (candidate hard-ceiling). codex-review before declaring BLOCKED.
- **fp8 trio (L1_09/L2_34/L2_37 P2)**: abs 32 = 1 fp8-ULP near ±448; exceeds
  fp8_software_emulated (~22.5 near saturation). Fixture-saturation vs real
  ceiling — under investigation.
- **L2_30 P2**: topk_ids identity drift (abs 6.0) — integer top-k flip from a
  TF32 router-logit near-tie; under investigation.

## Codex implementation review (gpt-5.5:xhigh, 2026-06-02) + dispositions
Overall REFINE → all TF32 edits + output_projection CONFIRMED at source (L3_17
pinned-dot, L3_03 decay-fold, L2_30 router dot all correct). Actions taken:
- **None-path tightened (must-fix, applied)**: R10.b now fail-closes a `None`
  return unless `interface.inplace_outputs` (or mutability) declares an in-place
  key covering a ref key — a no-op kernel echoing read-only inputs can no longer
  trivially pass. Added `inplace_outputs` to L1_01/L1_22/L3_04 (L1_25 had it).
- **L2_30 (VERIFIED) kept**: codex suggested argmax-on-logits to match the
  kernel's tie-break; softmax is monotonic so selection is identical except exact
  ties (not hit by the fixtures), and it already passes — not redone to avoid
  regressing a pass; tie-break is a documented latent edge.

Hard-tail dispositions (codex-confirmed mechanisms):
- **L3_13 — HARD-CEILING-BLOCKED**: snapshot/slice/gk contract correct; residual
  ~1.2 is recurrent `tl.exp`/MMA accumulation order, not a contract bug.
- **L3_16-P2 — HARD-CEILING-BLOCKED**: a TF32-dot accumulator feeds a second
  unpinned dot; the dense oracle can't reproduce the tensorcore reduction tree.
- **fp8 trio L1_09/L2_34/L2_37-P2 — HARD-CEILING-BLOCKED (raw-fp8 compare)**:
  per-group scale normalizes max to 448, so abs-32 = one E4M3-ULP at saturation;
  reducing magnitude won't help. Faithful fix is a contract change (compare
  dequantized `q*scale`, or a GPU fp8-cast oracle) — NOT a tolerance loosening.
- **L3_03-P2 — FIXED → VERIFIED**: reauthored the oracle to mirror the
  launcher's chunked state-passing (per-chunk-local decay + carry recurrence)
  instead of a single dense contraction. CPU-verified algebraically equal to the
  dense form (6e-8), and the chunked order matches the kernel → GPU VERIFIED.

## Running total: 85/103 VERIFIED
Genuine hard-ceilings (codex + independently re-verified, honest BLOCKED — not
faked, denominator stays 100):
- L3_13 (~1.2): TF32 tensorcore reduction-tree order compounded through the
  recurrence; state is fp32 in registers (no bf16 round-trip — ruled out), so
  not a dtype miss; a dense oracle cannot replicate the reduction tree.
- L3_16-P2 (1.08e-2): TF32-dot accumulator feeds a second unpinned dot.
- fp8 trio L1_09/L2_34/L2_37-P2 (abs 32): one E4M3-ULP at the 416→448 boundary;
  even dequantized compare = 32/448 = 7.1% > fp8 rtol 5% → only a bit-matched
  GPU fp8-cast oracle could close it.
## Schema-gap tasks (13) — workflow-diagnosed, all bench-side
Diagnosed in parallel (13-agent workflow), each validated locally via the CPU
R10.a audit (ALIGNED_READY) before GPU. Categories + fixes:
- **NEEDS_BRIDGE config-constant** (oracle read a key the kernel hardcodes):
  L1_11/L2_12 (group_size derived from tensor shapes), L1_17 (approximate pinned
  "none"), L1_30 (fp8_max pinned 448.0). Audit is `oracle_inputs - kernel_params`
  (oracle_pyref_precompute.py:352) — shapes-add does NOT help; must drop the key
  from the oracle's reads.
- **SIGNATURE_MISMATCH / launcher**: L2_17 (rename fixture w/b → weight/bias);
  L1_15/L2_24/L2_23 (gpu_triton placeholder → _pa_launcher.py::run activates the
  launcher so R10.a sets kernel_required=[]); L2_20 (new launcher avoids the AST
  rank false-positive, repoint gpu_triton).
- **RUNTIME_ERROR**: L1_08 (fp8 has no CPU index kernel → uint8-view gather).
- **output_projection**: L1_17 also returns (out, out_scales) → arity-2 map{output:0}.
- **TF32**: L2_12 fp32 quant-GEMM → `_to_tf32` + cube_matmul_fp16 reclass.
- **num_compute_units platform** (L1_13, L2_14): un-severed kernels import
  vllm.utils.platform_utils.num_compute_units, which raises on the daemon. Added
  `install_vllm_num_compute_units_stub()` (grid-sizing only, → 128, fail-safe) to
  the runner hygiene, before kernel load. NOT a real platform block.

GPU-verified (bc2gap2-5): L1_08, L1_11, L1_15, L1_17, L1_30, L2_12, L2_14, L2_17,
L2_20, L1_13 → VERIFIED. (L2_14 also needed _to_tf32 + cube_matmul reclass once
num_compute_units let it load; the num_compute_units stub had to move from main()
into R10.b's _import_kernel_function so it runs before the C2 _load — caught by
the GPU run, not assumed.)

Honest BLOCKED (codex + source-confirmed, NOT faked):
- **L2_23, L2_24** (lightning-attn): the repoint unmasked the real cross-block
  decay off-by-one — `exp(-slope*(t-s-1))` realized vs `exp(-slope*(t-s))`
  documented — a correct oracle MUST diverge → upstream-kernel-bug ceiling.
- **L2_33** (grouped-rope): frozen kernel uses a Python `if` on a [BLOCK_H] tensor
  → Triton 3.6.0 raises UnsupportedLanguageConstruct at compile; no bridge can
  make a non-compiling kernel pass.

## Final: 95/103 VERIFIED, 8 honest BLOCKED
Baseline 62 → 95 (+33: 23 in the C2 precision wave, 10 in the schema-gap wave).
8 BLOCKED = 5 precision ceilings (L3_13, L3_16-P2, L1_09/L2_34/L2_37 fp8) +
3 upstream/platform (L2_23, L2_24 off-by-one, L2_33 won't-compile on Triton 3.6).
Denominator stays 100; every VERIFIED is a real GPU run within the LOCKED op-class
tolerance; no faked passes, no relaxed tolerance values, no vendored kernel edits.
VERIFIED == element-wise allclose within op-class tolerance (NOT bit-exact); the
_to_tf32 oracle emulation makes the reference match the kernel's real TF32 path, so
GEMM/recurrent residuals are fp32 accumulation-order noise well inside tolerance.
