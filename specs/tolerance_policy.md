# Tolerance Policy

> **v3.2 ERRATA — AscendC removed.** This is a GPU-triton → NPU-triton *migration* benchmark; there is no AscendC native baseline, no `ascendc_baseline.yaml`, no `in_s_native`/`oracle.ascendc`. Correctness is the scored axis; performance is raw NPU latency (deferred to 910B). AscendC-specific text below is **superseded** and kept only as historical context.


> v0 — per v4.13 §10.2 + R18-S2
> Machine-readable form: `triton_ascend_bench/tolerance_policy.yaml` (loaded by `eval.py`)

Per-task `meta.yaml.tolerance_class` references one of the classes below. Runner uses `torch.allclose(out_c, out_o, atol=cls.atol, rtol=cls.rtol)` for correctness check.

## Tolerance Classes

| Class | dtype | atol | rtol | Notes |
|-------|-------|------|------|-------|
| `elementwise` | FP16/BF16 | 1e-3 | 1e-3 | Default for pointwise ops (silu, gelu, bias) |
| `vector_reduce` | FP16/BF16 | 1e-3 | 1e-3 | RMSNorm / LayerNorm / softmax (in-row reduce) |
| `cube_matmul_fp16` | FP16/BF16 → FP32 acc | 5e-3 | 1e-3 | GEMM with FP32 accumulator |
| `int8_w8a8` | INT8 → FP16 dequant | 5e-2 | 1e-2 | W8A8 INT8 matmul, expected quantization error |
| `int4_w4a8` | INT4 → INT8 dequant | 5e-2 | 1e-2 | W4A8 grouped matmul |
| `fp8_software_emulated` | FP8 ↔ FP16 cast | 1e-1 | 5e-2 | **Performance not in primary score** (910B has no FP8 cube) |
| `recurrent_chunk` | FP16/BF16 | 5e-3 | 5e-3 | Mamba / GDN / KDA chunk pipelines |
| `sampling_rng` | int / float | 0 | 0 | Bit-exact required when seed matches |
| `atomic_scatter` | FP32 | 1e-4 | 1e-4 | Non-deterministic; rerun 3x for consistency |
| `cumsum_scan` | FP16/BF16/INT | 1e-3 | 1e-3 | Prefix scan accumulation |
| `attention_flash` | FP16/BF16 | 5e-3 | 1e-3 | Online softmax + matmul fused |
| `mrope` | FP16/BF16 | 1e-3 | 1e-3 | mRoPE position embedding |

## Special Rules

- **NaN/Inf consistency** (per v4.13 §5b.5 子项 ⑤):
  positions of NaN/Inf in `out_c` must match `out_o` exactly. If `out_o[i]` is NaN, `out_c[i]` must also be NaN.

- **Per-dtype sentinels** (per v4.13 R12-P1-3):
  - float dtypes: `NaN`
  - int8: `INT8_MIN` (-128)
  - int16/32: corresponding MIN
  - bool: alternating mask `0xAA`

- **Tolerance escalation** (per v4.13 §5b.27 cross-oracle drift):
  if PyTorch / GPU triton / AscendC three-way diff fails at standard tolerance, task does NOT auto-escalate. Owner must investigate and decide whether to:
  1. Tighten/loosen tolerance (with documented justification in `contract.md`)
  2. Fix oracle / wrapper
  3. Move task out of `S_native` (only score_all secondary)
  4. Skip task entirely (`skip_registry.yaml`)

## Authority

- Source of truth: `triton_ascend_bench/tolerance_policy.yaml` (this file's machine-readable counterpart).
- Each task's `meta.yaml.tolerance_class` MUST be one of the keys above. New classes require:
  1. Justification in PR description
  2. Code review from at least one Phase 2 owner
  3. Update to both this `.md` and the `.yaml`
