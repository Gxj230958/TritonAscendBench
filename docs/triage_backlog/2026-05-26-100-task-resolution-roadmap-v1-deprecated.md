---
date: 2026-05-26
topic: 100-task-resolution-roadmap
status: actionable plan
based-on:
  - HEAD `b27b0a8`(post-revert clean,0 SKIPPED_BY_DESIGN)
  - `docs/triage_backlog/2026-05-26-truth-inventory.md`(provenance-first inventory)
audience: bench team / parallel sessions / reviewers
---

# 100-Task Resolution Roadmap

把 truth inventory 给出的 5 B-class + 95 C-class、12 family lane 的真实分布,转成可执行的 5-phase 路线图,**10-15 工作日** 把 100 task 全推到 PASS。

## 0. Executive Summary

- **总分母不变**:全 100 task 进 release-gate,不剔分母,不用 `release_scope.excluded` 偷剔。
- **路径**:lane-based 修法 → 简单 lane 先做,经验带到难 lane;**不**重写 phase_b schema(truth inventory 证 `D.unresolvable = 0`,schema 重写必要性大降)。
- **节奏**:Phase A(19,2-3 d)→ Phase C(31,3-4 d)→ Phase D(50,5-7 d),并行 Phase B(5 GPU sign-off,1 d),收尾 Phase E(release-gate,0.5 d)。
- **协调**:2 个 session 按 lane ownership 严格分,同一 lane 永远归同一人,避免文件冲突。

## 1. 战略层判定

### 1.1 既定 SETTLED(不翻炒)

| 约束 | 内容 |
|---|---|
| 分母完整 | 全 100 task 进 release-gate,**禁** `release_scope.excluded` 偷剔 |
| 双 oracle 路径 | **≥80%** PyTorch oracle 对齐 + **≤20%** provider GPU 预跑 golden |
| corpus 冻结 | `kernel.py`(SPDX header)是 frozen upstream,**禁改** |
| bench 约束 | English-only 代码/注释/docstring;无 process noise;stdlib logging;0 hardcode 外部项目名 |
| Phase 顺序 | 简单 lane 先做(经验积累 + 快速反馈),不要一上来攻最难 lane |

### 1.2 truth inventory 关键发现 → 战略推断

| 发现 | 战略含义 |
|---|---|
| `D.unresolvable = 0/100`(全部 kernel function 真在源码树找得到) | **跳过 P3 phase_b schema 重写**;直接 P4 family lane 实施 |
| 5 B-class(meta 不一致但 resolver 已 RESOLVED) | 一组小 fix(meta yaml 对齐)+ GPU sign-off,可与 Phase A 并行 |
| 95 C-class(`oracle.gpu_triton: kernel.py::placeholder`)| 单点工艺缺失,**不**是设计错位 — 每个 task ≤30 min meta + oracle 修可 RESOLVED |
| resolver 失败 92% 集中在 shapes-incomplete + upstream-not-invocable | 批量 fix 可行,**lane 内同质性高**(同 lane 用相同 reference 算式) |
| 12 lane,最大 lane moe(16),最小 lane gemm(3) | lane 划分自然成 phase 单元 |

### 1.3 为什么 lane-based 而不是 task-based?

- **同 lane task 共享数学**:gemm 全是 `@` 矩阵乘;block_table 全是 `torch.gather` index;lora 全是两层 `F.linear`。一个 lane 做透,其它 task 复用 oracle 模板。
- **修法批量化**:同 lane meta 修法雷同,~10 min/task 不是 30 min。
- **GPU 资源协调**:同 lane 的 GPU sign-off 可一次 batch 跑,不浪费排队。
- **认知负载小**:一个 session 集中一个 lane,不上下文切换。

## 2. 100 task 完整 5-phase 分布

### 2.1 总分布表

| Phase | Lane | Task 数 | 累计 | 估时 | 路径 | GPU |
|---|---|---|---|---|---|---|
| **A** | gemm(3)+ block_table(5)+ flashinfer_indices(5)+ lora(6) | **19** | 19/100 | 2-3 d | PyTorch oracle | 不需 |
| **B 并行** | 5 B-class GPU sign-off | **5** | (并行)| 1 d | GPU verify | 必需 |
| **C** | elementwise(13)+ norm(6)+ sampling(6)+ quant(6) | **31** | 50/100 | 3-4 d | PyTorch oracle | 不需 |
| **D** | paged_attention(13)+ moe(16)+ ssm_mamba(11)+ delta_rule(10) | **50** | 100/100 | 5-7 d | PyTorch 80% + GPU 20% | 部分需 |
| **E** | release-gate sign-off | 100 | sign-off | 0.5 d | full verify | 必需 |
| **合计** | 12 lane | **100** | 100/100 | **10-15 d** | | |

### 2.2 lane → phase 分配理由

| Lane | Phase | 数学复杂度 | 选择依据 |
|---|---|---|---|
| gemm | A | 最低(`@` / `F.linear` 一行)| `simple_pytorch_ok`/`formula_exact_manual`,无 attention 上下文 |
| block_table | A | 低(纯 index 计算 + `torch.gather`)| 100% `formula_exact_manual`,无浮点精度难题 |
| flashinfer_indices | A | 低(index transform)| 100% `formula_exact_manual` |
| lora | A | 低(2 层 `F.linear` 复合)| 100% `formula_exact_manual`,vllm sgmv 数学清晰 |
| elementwise | C | 中(`F.silu` / `F.gelu` / `softcap` 等)| 算式直白但函数多样,需逐 task 写 |
| norm | C | 中(layer_norm / rms_norm 公式)| `F.layer_norm` 可直接复用 |
| sampling | C | 中(softmax + clamp + masked_fill)| 多步组合 |
| quant | C | 中(`amax` / `scale` / `int8 round`)| per-token 算式有 corner case |
| paged_attention | D | 高(KV cache layout)| 13 task,需懂 paged-attn 数据结构 |
| moe | D | 高(dispatcher + experts)| 16 task,最大 lane,dispatch 逻辑复杂 |
| ssm_mamba | D | 高(状态空间模型)| `chunk_cumsum` / `chunk_scan` 等数学最难 |
| delta_rule | D | 高(gated delta rule recurrence)| 状态更新 + chunk-wise,与 ssm 同档难度 |

## 3. Phase A — 立即起手(19 task,2-3 工作日)

**全 19 task,均 `oracle_tier ∈ {simple_pytorch_ok, formula_exact_manual}`,纯 dev 机即可完成。**

### 3.1 Phase A 完整 task list

| Lane | Count | Task IDs |
|---|---|---|
| `gemm` | 3 | `L2_12_awq_gemm` / `L2_13_bmm` / `L2_14_matmul_kernel_persistent` |
| `block_table` | 5 | `L1_02_compute_slot_mappings` / `L1_03_convert_req_index_to_global_index` / `L1_04_copy_page_indices` / `L1_05_gather_block_tables` / `L1_34_transform_index_page_table_decode`(B-class)|
| `flashinfer_indices` | 5 | `L1_32_create_flashinfer_kv_indices_triton` / `L1_33_create_flashmla_kv_indices_triton` / `L1_35_compute_src2dst_triton` / `L1_36_compute_seg_indptr_triton` / `L1_37_compute_problem_sizes_w4a8` |
| `lora` | 6 | `L2_04_chunked_lora_expand` / `L2_05_chunked_lora_shrink` / `L2_06_qkv_lora_b` / `L2_07_gate_up_lora_b` / `L2_08_sgemm_lora_a` / `L2_09_sgemm_lora_b` |

### 3.2 per-task 修法模板(每 task ≤30 min)

每 task 流程:

```
1. Read tasks/<lvl>/<id>/kernel.py     -> locate real upstream entrypoint name
2. Read meta.yaml.upstream.{repo,path} -> confirm upstream file location
3. grep upstream source for signature + input semantics
4. Edit meta.yaml:
   - oracle.gpu_triton: kernel.py::<real-fn>           (replace placeholder)
   - oracle.entrypoint kwargs                          (fill if missing)
   - phase_a_prime.upstream_callable.{module, function} (resolver entrypoint)
5. If oracle_tier == simple_pytorch_ok and oracle_pytorch.py missing
   -> write ~10-20 line PyTorch reference
6. Single-task verify:
   python3 scripts/resolve_upstream_entrypoints.py --task <id> --refresh-cache
   -> expect verdict == PASS or entrypoint != null
7. pytest tests/unit_tests/<related module>            (if oracle_pytorch.py changed)
8. commit + push                                       (message <= 50 words)
```

### 3.3 lane 内 oracle 模板(可批量复用)

**gemm lane**(纯 matmul):

```python
# oracle_pytorch.py (simplified template)
import torch
def oracle(A: torch.Tensor, B: torch.Tensor, **kw) -> dict:
    out = A @ B  # or torch.bmm(A, B) for batched variant
    return {"out": out}
```

**block_table lane**(index gather):

```python
def oracle(block_table: torch.Tensor, seq_lens: torch.Tensor, **kw) -> dict:
    # Compute slot mapping / page indices from block_table per seq_lens
    indices = ...  # torch.gather / torch.repeat_interleave
    return {"slot_mapping": indices}
```

**flashinfer_indices lane**(index transform):

```python
def oracle(req_pool_indices: torch.Tensor, seq_lens: torch.Tensor, **kw) -> dict:
    # cumulative seq_lens -> kv_indptr
    kv_indptr = torch.cumsum(seq_lens, dim=0)
    return {"kv_indices": ..., "kv_indptr": kv_indptr}
```

**lora lane**(双 linear 复合):

```python
def oracle(x: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, scaling: float, **kw) -> dict:
    # x @ A^T @ B^T * scaling  (chunked / per-seq variants share structure)
    out = (x @ lora_a.transpose(-1, -2)) @ lora_b.transpose(-1, -2) * scaling
    return {"out": out}
```

### 3.4 Phase A acceptance

- 全 19 task `python3 scripts/resolve_upstream_entrypoints.py --task <id>` 真 PASS / RESOLVED
- `resolve_all.json` aggregated RESOLVED 计数严格递增 ≥19
- `pytest tests/` 不破(baseline 1944+)
- `git grep -nP '[\x{4e00}-\x{9fa5}]' -- 'tasks/level1/L1_*/oracle_pytorch.py' 'tasks/level2/L2_*/oracle_pytorch.py'` 命中 0
- 0 `phase_a_prime.skip_invocation: true` 新增
- 0 `release_scope.excluded: true` 新增

## 4. Phase B — 5 B-class GPU sign-off(并行 Phase A,~1 工作日)

**这 5 个 task resolver 已 RESOLVED,只缺 GPU 侧 verify 真跑通。**

### 4.1 Phase B 完整 task list

| Task | Lane | 现状 | 修法 |
|---|---|---|---|
| `L1_01_bias` | elementwise | meta SPDX 缺,resolver 已 PASS | dev 机 fix SPDX header → GPU 跑 `phase_a_prime --task L1_01_bias --verify` |
| `L1_12_rms_norm` | norm | resolver PASS | GPU sign-off only |
| `L1_16_silu_and_mul` | elementwise | resolver PASS | GPU sign-off only |
| `L1_26_merge_state` | elementwise | resolver PASS | GPU sign-off only |
| `L1_34_transform_index_page_table_decode` | block_table | meta `kernel.py::placeholder` 但 resolver auto-discovered 真函数 | dev 机 fix meta(oracle.gpu_triton 写真函数名)→ GPU sign-off |

### 4.2 Phase B acceptance

- GPU 跑 `bash scripts/automation/runners/tab_quickstart.sh verify-all --tier all`,这 5 task 全 PASS
- 5 task 由 `B-class` → `A.frozen-clean`(maturity 升级)

## 5. Phase C — 中等复杂(31 task,3-4 工作日)

### 5.1 Phase C 完整 task list

| Lane | Count | Task IDs |
|---|---|---|
| `elementwise` | 13 | L1_01(若 B 期未做)/ L1_15_act_and_mul / L1_16(若 B 期未做)/ L1_17_gelu_and_mul / L1_18_fused_softcap / L1_20_log_softmax / L1_25_merge_attn_states / L1_26(若 B 期未做)/ L1_27_apply_grammar_bitmask / L1_28_pack_bitmatrix / L1_29_prepare_eagle_inputs / L1_31_triton_mrope_forward / L2_38_concat_and_cast_mha_k |
| `norm` | 6 | L1_12(若 B 期未做)/ L1_13_layer_norm_fwd / L1_14_layer_norm_fwd_1pass / L2_15_fused_dual_residual_rmsnorm / L2_16_fused_rmsnorm / L2_17_layer_norm_gated_fwd |
| `sampling` | 6 | L1_19_topk_log_softmax / L1_21_gumbel_sample / L1_22_min_p / L1_23_penalties_and_temperature / L1_24_rejection_sample / L2_18_fused_gdn_gating |
| `quant` | 6 | L1_10_per_token_group_quant_int8 / L1_11_awq_dequantize / L1_30_act_quant / L1_38_silu_mul_fp8_quant_deep_gemm / L2_10_w8a8_block_int8_matmul / L2_11_w8a8_triton_block_scaled_mm |

**注**:与 Phase B 重合的 task(L1_01 / L1_12 / L1_16 / L1_26)若 Phase B 期已升 A-class,Phase C 不再算。Phase C 净新增 31 - 4 = **27 task**(若 Phase B 全做)。

### 5.2 per-lane oracle 模板

**elementwise lane**:

```python
# silu_and_mul: out = silu(x[..., :half]) * x[..., half:]
def oracle(x: torch.Tensor, **kw) -> dict:
    half = x.size(-1) // 2
    return {"out": torch.nn.functional.silu(x[..., :half]) * x[..., half:]}
```

**norm lane**:

```python
# rms_norm: x_norm = x / sqrt(mean(x^2, dim=-1) + eps) * weight
def oracle(x, weight, eps=1e-6, **kw) -> dict:
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return {"out": x / rms * weight}
```

**sampling lane**(min_p):

```python
def oracle(logits, p, **kw) -> dict:
    probs = logits.softmax(dim=-1)
    pmax = probs.max(dim=-1, keepdim=True).values
    mask = probs < (p * pmax)
    return {"logits": logits.masked_fill(mask, float("-inf"))}
```

**quant lane**(per-token int8):

```python
def oracle(x: torch.Tensor, group_size=128, **kw) -> dict:
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = amax / 127.0
    x_q = torch.round(x / scale).to(torch.int8)
    return {"x_q": x_q, "scale": scale.squeeze(-1)}
```

### 5.3 Phase C acceptance

- 累计 50/100 RESOLVED(若 Phase B 全做,实际 49-50)
- pytest 不破
- 0 中文 / 0 SKIPPED 新增

## 6. Phase D — 高复杂(50 task,5-7 工作日)

### 6.1 Phase D 完整 task list

| Lane | Count | Task IDs |
|---|---|---|
| `paged_attention` | 13 | L1_06_reshape_and_cache_kernel_flash / L1_07_dequantize_k_cache_fast / L1_08_dequantize_k_cache_paged / L1_09_quantize_k_cache_fast / L2_23_fwd_kv_parallel / L2_24_fwd_kv_reduce / L2_33_fwd_grouped_kernel_stage1_rope / L3_01_kernel_unified_attention_3d / L3_02_kernel_paged_attention_2d / L3_04_fwd / L3_05_fwd_kernel_alibi / L3_06_fwd_grouped_kernel_stage1 / L3_07_correct_attn_cp_out |
| `moe` | 16 | L2_25_fwd_kernel_ep_gather / L2_26_fwd_kernel_ep_scatter_1 / L2_27_deepep_permute_triton / L2_28_deepep_post_reorder_triton / L2_29_fused_moe_router_cudacore / L2_30_fused_moe_router_tensorcore / L2_31_moe_align_block_size_stage2 / L2_32_experts_combine / L2_34_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe / L2_35_pre_reorder_triton_kernel_for_cutlass_moe / L2_36_post_reorder_triton_kernel_for_cutlass_moe / L2_37_silu_and_mul_post_quant / L3_09_fused_moe / L3_10_fused_moe_kernel_gptq_awq / L3_11_expert_triton / L3_12_moe_mmk |
| `ssm_mamba` | 11 | L1_39_state_passing_fwd / L1_40_chunk_cumsum_fwd / L2_01_bmm_chunk_fwd / L2_02_chunk_scan_fwd / L2_03_chunk_state_fwd / L2_20_causal_conv1d_update / L2_21_causal_conv1d_fwd / L2_22_linear_attn_decode / L3_03_chunk_state_varlen / L3_16_chunk_fwd_kernel_o / L3_22_chunk_local_cumsum_vector |
| `delta_rule` | 10 | L2_19_fused_sigmoid_gating_delta_rule_update / L3_13 / L3_14 / L3_15 / L3_17 / L3_18 / L3_19 / L3_20 / L3_21 / L3_23 |

### 6.2 Phase D 路径选择(per-task 判断)

Phase D 是 oracle 数学复杂区,**每个 task 落实时选 PyTorch 80% 还是 GPU 20% 路径**:

| 判定条件 | 路径 | 工作量 |
|---|---|---|
| `oracle_tier ∈ {simple_pytorch_ok, formula_exact_manual}` 且 PyTorch reimplement < 50 行 | **PyTorch oracle** | 30-60 min/task |
| 上游函数已可 import + GPU 跑得通 | **provider GPU 预跑 golden**(phase_b_capture)| 10 min/task(GPU)+ 调 schema |
| 数学过复杂(`upstream_oracle_required` 且 PyTorch > 100 行)| **GPU 预跑** | 同上 |

**SETTLED 比例约束**:全 corpus **≤20** task 走 phase_b GPU 预跑;其余 **≥80** 走 PyTorch。前 50 个 task(Phase A + C)100% PyTorch,Phase D 50 个 task 里最多 20 个走 phase_b。

### 6.3 Phase D 内部 sub-phase

| Sub-phase | Lane | 估时 | 顺序理由 |
|---|---|---|---|
| D.1 | paged_attention(13)| 2 d | 数据结构最清晰,KV cache layout 易 reference |
| D.2 | ssm_mamba(11)+ delta_rule(10)= 21 | 2-3 d | 状态空间模型,数学密集但 chunk-wise 模板可批量 |
| D.3 | moe(16)| 1-2 d | 最大 lane,dispatcher 模式可批量;放最后,前面经验全用上 |

### 6.4 Phase D acceptance

- 累计 100/100 RESOLVED
- phase_b_capture 实际启用 ≤20 task(SETTLED 比例守住)
- pytest 不破

## 7. Phase E — release-gate sign-off(0.5 工作日)

### 7.1 流程

```bash
# Final verification on GPU machine
bash scripts/automation/runners/tab_quickstart.sh verify-all --tier all
bash scripts/automation/runners/tab_quickstart.sh release-gate
bash scripts/automation/runners/tab_quickstart.sh pass-all

# Inspect
ls results/release_gate_report.json
grep -c '"status": "PASS"' results/release_gate_report.json   # expect ~99-100
```

### 7.2 Phase E acceptance(release-ready)

- 100/100 task verdict = PASS(允许 ≤1 个 known-flaky 由 `meta.flaky: true` 标注;不剔分母)
- `release_gate_report.json` 100 行 PASS
- pytest 全 PASS 0 fail 0 skip
- `tab_quickstart.sh pass-all` 单条命令 exit 0
- bench tarball `make dist` 生成 ≤1.5 GB,文件清单 0 hardcode 外部项目名

## 8. 协调机制(2 session 并行)

### 8.1 ownership matrix(严格不重叠)

| Session A(我) | Session B(另一 session) |
|---|---|
| `gemm` (3) | `block_table` (5) |
| `lora` (6) | `flashinfer_indices` (5) |
| **Phase A subtotal: 9** | **Phase A subtotal: 10** |
| `elementwise` (13) | `sampling` (6) |
| `norm` (6) | `quant` (6) |
| **Phase C subtotal: 19** | **Phase C subtotal: 12** |
| `paged_attention` (13) | `moe` (16) |
| `delta_rule` (10) | `ssm_mamba` (11) |
| **Phase D subtotal: 23** | **Phase D subtotal: 27** |
| **A 总 own: 51 task** | **B 总 own: 49 task** |

**Phase B GPU sign-off** 5 task 由 A 全 own(协调 GPU 机调度)。

### 8.2 同步点

| 同步点 | 触发 | 双方动作 |
|---|---|---|
| S0(T+0)| 本 plan 通过 | A 起 walk-through,B 等 A 推 walk-through 完 |
| S1(T+0.5 d)| A walk-through 完 | B 拿模板起 block_table |
| S2(T+3 d)| Phase A 完 + Phase B GPU sign-off 完 | 双方 sync resolver 数据,确认 24-25/100 RESOLVED |
| S3(T+7 d)| Phase C 完 | 双方 sync,50-55/100 RESOLVED |
| S4(T+14 d)| Phase D 完 | 双方 sync,100/100 RESOLVED |
| S5(T+15 d)| Phase E 完 | release-gate PASS,bench tarball 发 |

### 8.3 冲突预案

| 风险 | 概率 | 预案 |
|---|---|---|
| 同 file edit 冲突 | 低(lane 不重叠)| ownership 严格不跨 lane;`scripts/` 共享文件 edit 前 announce |
| `resolve_all.json` 同时刷新 | 中 | 双方约定 commit 前先 pull rebase,refresh 后立即 push |
| `phase_b_capture` schema 改动 | 低 | 真碰到再 sync;默认 Phase D 才考虑 |
| GPU 机排队 | 中 | A 协调 GPU,B 期间继续 PyTorch oracle 路径 |

## 9. acceptance criteria(总体)

每 phase 结束 + final 都过下列闸:

```bash
# 1. Real RESOLVED count from resolver
python3 scripts/resolve_upstream_entrypoints.py --all --refresh-cache --json results/resolve_all.json
python3 -c "import json; d=json.load(open('results/resolve_all.json')); \
  print(sum(1 for v in d['verdicts'] if v.get('resolved')), '/', len(d['verdicts']))"
# Phase A expects >=19/100; Phase C expects >=50; Phase D expects >=100

# 2. 0 new SKIPPED_BY_DESIGN
grep -lE '^  skip_invocation:\s*true\b' tasks/level*/L*/meta.yaml | wc -l  # expect 0

# 3. 0 excluded from denominator
grep -lE 'release_scope:\s*$' -A2 tasks/level*/L*/meta.yaml | \
  grep -E 'excluded:\s*true' | wc -l  # expect 0

# 4. pytest all PASS, 0 fail 0 skip
cd /data1/code/agentic-rl/kernel/triton-ascend-bench
pytest tests/ -q  # expect 0 failed 0 skipped

# 5. 0 Chinese in code
git grep -nP '[\x{4e00}-\x{9fa5}]' -- 'tasks/*/L*/*.py' 'scripts/*.py'  # expect 0 hits

# 6. 0 process noise in changed files
git grep -nIE '\b(codex|claude|round[- ]?[0-9]|sanity findings|[RHDFP][0-9]+\b|V[1-9])' \
  -- '*.py' '*.md'  # expect 0 hits
```

## 10. 时间表(基线)

| 节点 | 日期 | 累计 | 完成什么 |
|---|---|---|---|
| T0 | 2026-05-26(今天)| 0/100 | revert + truth inventory + roadmap 完成 |
| T+0.5 d | 2026-05-26 晚 | 1/100 | L2_13_bmm walk-through 完(模板)|
| T+3 d | 2026-05-29 | 24/100 | Phase A 19 + Phase B GPU sign-off 5 完 |
| T+7 d | 2026-06-02 | 55/100 | Phase C 31 完(扣除 4 B 重合)|
| T+14 d | 2026-06-09 | 100/100 | Phase D 50 完 |
| T+15 d | 2026-06-10 | release | Phase E release-gate PASS |

**总估时:10-15 工作日。** 若 Phase D 部分 task 卡 GPU 排队或 lane 内复杂度超预期,缓冲 +2-3 d。

## 11. 风险 + 应对

| 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|
| Phase A 19 task PyTorch oracle 与 GPU triton 不对齐(精度漂)| 中 | 单 task 卡 1-2 h | 修 oracle / 加 atol/rtol 容差;若上游 triton 本就 fp16/bf16 不精确,放宽到 1e-2 |
| Phase B 5 task GPU 跑出错(driver / sys.path / etc.)| 中 | 1 day 延 | 临时降级为 dev 机 phase_a_prime --dryrun,后补 GPU |
| Phase D moe / ssm 数学超复杂,PyTorch reimplement > 100 行 | 高(预计 5-10 task) | 累计 2-3 d 延 | 落入 phase_b GPU 预跑通道(≤20 task 配额内) |
| `kernel.py` 改动需求(SPDX 缺 / freeze 漂)| 低 | 个例 | 跟 upstream PR 对齐补 SPDX,不改 kernel 逻辑 |
| corpus 演化新增 task(L3 新)| 中 | 总量增 | truth inventory 重跑 → roadmap 增量 |

## 12. 立即下一步(今天工作日内)

### 12.1 第一步:Phase A walk-through

挑 `L2_13_bmm`(最简,纯 batched matmul)做完整修复,作为后续 18 task 的**模板**:

```bash
# 1. Inspect kernel + meta
cat tasks/level2/L2_13_bmm/kernel.py | head -50
cat tasks/level2/L2_13_bmm/meta.yaml

# 2. Edit meta.yaml: oracle.gpu_triton -> real function name,
#    plus phase_a_prime.upstream_callable.{module, function}.
#    Write oracle_pytorch.py with a few lines of torch.bmm(A, B).

# 3. Resolver verify
python3 scripts/resolve_upstream_entrypoints.py --task L2_13_bmm --refresh-cache

# 4. Commit
git add tasks/level2/L2_13_bmm/
git commit -m "task(L2_13_bmm): unfreeze placeholder, add PyTorch bmm oracle"
```

预估 ~30-60 min。完后 push,作为模板给 B session 看。

### 12.2 第二步:批 gemm + lora(我)+ 通知 B session 起 block_table + flashinfer_indices

- 我:`L2_12_awq_gemm` + `L2_14_matmul_kernel_persistent` + lora 6 个 = 8 task,~3-5 hr
- B session:`block_table` 5 + `flashinfer_indices` 5 = 10 task,~3-5 hr

**今晚目标:Phase A 19 task 全 RESOLVED。**

### 12.3 何时启 Phase B GPU sign-off

Phase A 19 完时,统一 batch 跑 GPU(L1_01 + L1_12 + L1_16 + L1_26 + L1_34 共 5 task),~1 hr GPU 时间。

## 13. 文档引用

- 源数据:[2026-05-26-truth-inventory.md](2026-05-26-truth-inventory.md)
- 历史调研:[2026-05-25-bench-shim-layer-analysis.md](2026-05-25-bench-shim-layer-analysis.md) / [2026-05-25-parallel-execution-plan.md](2026-05-25-parallel-execution-plan.md) / [2026-05-25-blocked-task-resolution-plan.md](2026-05-25-blocked-task-resolution-plan.md)
- 架构 spec:[docs/architecture/automation-llm-repair-architecture.md](../architecture/automation-llm-repair-architecture.md)
- 既定 SETTLED:`/data1/code/agentic-rl/CLAUDE.md`(no-subset-escape)

---

**Roadmap 结束。本文档自含;无需补充信息即可起手 Phase A。**
