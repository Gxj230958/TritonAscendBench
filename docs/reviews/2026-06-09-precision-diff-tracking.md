# 精度差异追踪 (TODO)

数据来源: `reorgfull0609` C2 grading (H200 daemon, commit 6685992). per-shape `max_abs_diff` /
`max_rel_diff` 取自每题 r10 verdict (`results/bc2reorgfull0609/r10/*.json`). 全 101 题 VERIFIED /
CEILING_ACCEPTED (均在 dtype-aware 容差内 PASS) — 下列是"容差内的最大偏差", 不是失败。

## 两组对齐机制 (truth 来源不同, 不可混比)

| 组 | 题数 | 对齐 truth | 说明 |
|---|---|---|---|
| **PyTorch-oracle** | 83 (75 formula_exact_manual + 8 simple_pytorch_ok) | kernel vs `oracle_pytorch` 实时输出 | 和 PyTorch 对齐 |
| **upstream-golden** | 18 (upstream_oracle_required) | kernel vs golden (上游 frozen kernel GPU 输出) | torch 复现已知偏离上游, 改用上游真实输出; kernel 本身=同一 severed 上游 kernel, 故 diff 趋近 0 |

## Top 精度差异

### PyTorch-oracle 组 (83) — top by max_abs
| task | tolerance_class | max_abs | max_rel | status | 解读 |
|---|---|---|---|---|---|
| L1_26_state_passing_fwd | recurrent_chunk | **384** | 0.0008 | VERIFIED | SSM 递归累积大 scale; **rel 仅 0.08%**, 非精度损失 |
| L3_23_fused_recurrent_gated_delta_rule_fwd | recurrent_chunk | 128 | 0.0135 | CEILING_ACCEPTED | recurrent final_state |
| L2_13_silu_mul_fp8_quant_deep_gemm | fp8 | 32 | 0.125 | VERIFIED | fp8 量化 ULP 级 |
| L1_09_quantize_k_cache_fast | 量化 | 32 | 0.1 | VERIFIED | 量化 ULP 级 |
| L2_18_w8a8_triton_block_scaled_mm | int8 | 8 | 0.0009 | VERIFIED | int8 ULP |

### upstream-golden 组 (18) — top by max_abs
| task | tolerance_class | max_abs | max_rel | status | 解读 |
|---|---|---|---|---|---|
| L2_30_silu_and_mul_post_quant | fp8_software_emulated | 32 | 0.5 | VERIFIED | fp8 e4m3 大值区相邻量化级 (±1 ULP) |
| L2_29_silu_mul_static_tensorwise_quant | fp8 | 32 | 0.125 | VERIFIED | fp8 ULP |
| L2_17_w8a8_block_int8_matmul | int8 | 8 | 0.0009 | VERIFIED | int8 ULP |
| L3_07_fused_sigmoid_gating_delta_rule_update | - | 2 | 8.73 | CEILING_ACCEPTED | rel 8.73 = ÷≈0 退化 |
| L3_19_fused_moe | - | 0.5 | 0.989 | CEILING_ACCEPTED | |
| (其余 13 个) | | ≤0.045 | | VERIFIED | 接近 bit-exact; L2_25/L2_08/L2_03 = 完全 0 |

## 指标陷阱 (必须在报告精度时注意)
- **`max_abs` 跨 dtype/scale 不可比**: L1_26 的 abs=384 (SSM 大 scale) vs fp8 的 abs=32 不是一回事; 看 rel 才知道 L1_26 实际仅 0.08%。
- **`max_rel` 在 `|oracle|≈0` 处退化**: L3_07 rel=8.73, 之前看到的 L3_16 rel=4e8 都是 `|k-o|/|o|` 当 o≈0 爆炸, **不是真精度差** (L3_16 实际 abs 仅 0.0078)。
- **量化 dtype 的 abs (fp8=32, int8=8) 是 ULP 级整数差**, 按 `fp8_ulp`/`int_atol` 判, 非连续精度差。

## TODO 追踪项
- [ ] **rel ÷≈0 退化处理**: 报告/分析精度时, 对 `|oracle| < eps` 的位置排除或改用 abs, 否则 rel 指标 (L3_07 8.73 / L3_16 4e8) 误导。
- [ ] **提取每题 tolerance margin** (`实际 diff / 该 class 容差上限` 的比值), 排序找"精度最吃紧"(最接近容差边界)的题 — 这才是比 raw max diff 更有意义的真指标。
- [ ] **复核 CEILING_ACCEPTED 题** (L3_23 / L3_07 / L3_19 / L3_15 / L1_15 / L3_16 / L2_23 等) 的 ceiling 容差放宽是否仍合理 (对照 `policies/ceiling_acceptance.yaml` allowlist)。
- [x] 18 upstream-golden 组 diff 极小 (量化 ULP + 3 题 bit-exact) — 设计如此 (绕开 torch 偏离上游), 无需追踪。
