# Phase 2 — production-dtype alignment

The fp32 C2 stage (98/103 VERIFIED) verifies extraction + logic at the most
precise dtype. It does NOT verify production-dtype (bf16/fp16) fidelity: ~82/103
fixtures are fp32 while vllm/sglang run most ops at bf16. The tolerance classes
are literally named `cube_matmul_FP16`, `flash_attention_FP16`, … — calibrated
for fp16/bf16 — so production dtype is the intended fixture; fp32 was the scaffold
default. This phase aligns the fixtures to production dtype.

## Proven mechanism (L2_08 POC → VERIFIED at bf16)
- Flip the fixture float32 → bf16 (the activation/weight inputs).
- **`_to_tf32` is a no-op at bf16 and fp16** (≤10 mantissa bits → the low 13 bits
  the helper drops are already zero), so GEMM oracles need NO change — the
  existing `x.to(fp32) @ w.to(fp32)` already equals the kernel's bf16 `tl.dot`
  (fp32 accumulate; bf16×bf16 products are exact in fp32).
- `cube_matmul_fp16` (5e-3) fits bf16 (it was calibrated for it).
- GPU-verify; fix per-task output-dtype-contract mismatches.

## Per-task contract to check
The kernel's output dtype must follow the (now-bf16) input, else the oracle's
`.to(input.dtype)` (bf16) mismatches a fp32/fp16 kernel output. Most GEMMs do
`acc.to(input.dtype.element_ty)` (output follows input → clean). Watch:
- MoE (L3_09/11/12): output uses `compute_type` — verify it follows input.
- Recurrent/SSM: bf16 I/O but fp32 STATE carried across chunks.
- Attention: bf16 Q/K/V, fp32 online-softmax.
- Norm: bf16 I/O, fp32 internal reduction (the oracle already computes fp32).

## Families (~82 fp32 tasks)
- **GEMM/LoRA/bmm** (~12): clean flip. Done: L2_08; batch: L2_05/07/09; rest:
  L2_01/13, L3_20, L2_12 (awq: activation bf16, weights stay int4), L2_14.
- **MoE** L3_09/11/12, L2_30: flip activations; check compute_type/topk dtypes.
- **Recurrent/SSM** L2_02, L3_03/13/16/17: bf16 I/O, fp32 state — careful.
- **Attention** L1_25, L3_01-05…: bf16 Q/K/V, fp32 softmax.
- **Norm/elementwise** L1_13/14, L2_15/17…: bf16 I/O, fp32 internal.
- **Quant** (already int8/fp8): no change.
- **Index/scatter/sampling/no-float** (~12): no float dtype to align — leave.

## Risk
A bf16 flip may surface a REAL production-fidelity divergence (a kernel that's
fine at fp32 but diverges at bf16, or a non-following output dtype). These are
genuine new findings, not regressions — triage each like the C2 wave.

## Approach
Per family, batch-flip the fixtures, GPU-verify, fix contract mismatches,
codex-review the non-obvious dtype contracts. Keep the denominator at 100; no
faked passes. Dual-fixture (keep fp32 logic + add bf16 fidelity) is an option if
preserving the logic-verification matters, but the fp16-named tolerances suggest
a single production-dtype fixture is the intended design.
