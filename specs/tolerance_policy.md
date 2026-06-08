# Tolerance Policy

> Authoritative scoring model: `docs/DESIGN.md §9`. GPU-Triton → 910B NPU-Triton migration; no AscendC baseline. Tolerance classes here gate **correctness** only; S2 equivalence uses strict bit-exact, not these classes.


> v0 — per v4.13 §10.2 + R18-S2
> Machine-readable form: `ascendbench/policies/tolerance_policy.yaml` (loaded by `eval.py`)

Per-task `meta.yaml.tolerance_class` references one of the classes below. Runner uses `torch.allclose(out_c, out_o, atol=cls.atol, rtol=cls.rtol)` for the correctness check. The table is illustrative; `ascendbench/policies/tolerance_policy.yaml` is the complete authoritative class list (it also defines e.g. `elementwise_exact_or_inf_safe`).

## Tolerance Classes

| Class | dtype | atol | rtol | Notes |
|-------|-------|------|------|-------|
| `elementwise` | FP16/BF16 | 1e-3 | 1e-3 | Default for pointwise ops (silu, gelu, bias) |
| `vector_reduce` | FP16/BF16 | 1e-3 | 1e-3 | RMSNorm / LayerNorm / softmax (in-row reduce) |
| `cube_matmul_fp16` | FP16/BF16 → FP32 acc | 5e-3 | 1e-3 | GEMM with FP32 accumulator |
| `int8_w8a8` | INT8 → FP16 dequant | 5e-2 | 1e-2 | W8A8 INT8 matmul, expected quantization error |
| `int4_w4a8` | INT4 → INT8 dequant | 5e-2 | 1e-2 | W4A8 grouped matmul |
| `fp8_software_emulated` | FP8 ↔ FP16 cast | 1e-1 | 5e-2 | 910B has no FP8 cube → modeled in the Roofline via vector compute + HBM (no FP8-cube π); score-eligibility is unaffected (defined solely by `eligibility.py`) |
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

- **Tolerance escalation** (cross-oracle drift):
  if the PyTorch oracle vs frozen GPU-Triton kernel diff fails at standard tolerance, the task does NOT auto-escalate. The owner must investigate and decide whether to:
  1. Tighten/loosen tolerance (with documented justification in `contract.md`)
  2. Fix oracle / wrapper
  3. Mark the task `BLOCKED` — a hard release block; it stays in `|S|` and is **not** removed from the denominator (no subset escape, `docs/DESIGN.md §4.4`)

## Authority

- Source of truth: `ascendbench/policies/tolerance_policy.yaml` (this file's machine-readable counterpart).
- Each task's `meta.yaml.tolerance_class` MUST be one of the keys in `ascendbench/policies/tolerance_policy.yaml` (the complete authority). New classes require:
  1. Justification in PR description
  2. Code review from at least one Phase 2 owner
  3. Update to both this `.md` and the `.yaml`
