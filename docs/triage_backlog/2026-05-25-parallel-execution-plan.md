---
date: 2026-05-25
status: parallel execution plan for dev-machine BLOCKED task repair
audience: bench team — two parallel Claude sessions ("session A" = user-driven, "session B" = parallel agent)
based-on:
  - HEAD 2833a67 (commit chain since b534449 loader bridge)
  - docs/triage_backlog/2026-05-25-blocked-task-resolution-plan.md (40 task recipes)
  - docs/triage_backlog/2026-05-25-bench-shim-layer-analysis.md (shim layer analysis + top-3 leverage table)
  - deep review on shim leverage (settled, archived in previous reviews)
scope: dev-machine repair work; final release-gate GPU sign-off out-of-scope
---

# Dev 机 BLOCKED Task 修复并行执行方案

## 0. 阅读须知

- 文档落在 `docs/triage_backlog/`,被 `.gitattributes` 的 `docs/triage_backlog/` 通配 `export-ignore` 覆盖,不进入 release tarball;视作 bench 团队内部协调文档。
- 全部参考数据基于:**HEAD `2833a67`** + **`results/resolve_all.json` 100 verdicts**(verdict snapshot 早于近期一连串 fix commit,真实剩余 BLOCKED 须由 fresh resolver 重跑后填充,见 §3 表 B 的 update hook)。
- 文档以中文叙述,所有 yaml / python / shell 片段仍是英文(repo-wide convention)。
- 严格 task ownership:每个 task / 文件**只一边 own**,避免双方并行 edit 同文件造成 merge conflict 或互相覆盖。
- **本 doc 内术语豁免**(operator-facing 协调文档):允许在本 plan doc 主体使用 `codex`(指 review skill)/ `humanize:ask-codex`(指 review 通道)/ `Mx` / `Px`(指 shim-leverage 分析章节编号或 P3/P4 verdict-class cluster 简称)/ `Ax` / `Bx`(指本 doc 内的 session 任务编号)等术语作技术指代,这些**不**视作 "过程痕迹"。**代码 / commit message / 任务 yaml / 测试代码**中仍严格遵守 CLAUDE.md "No process noise" 禁词清单。

### 0.1 路径约定(sibling checkout 布局)

本 plan / 所有 shell 命令都假设以下目录布局,并默认从 **bench repo 根** 跑命令:

```
<workspace>/                          # e.g. /data1/code/agentic-rl/kernel/
├── triton-ascend-bench/               # this bench repo (command cwd)
│   ├── scripts/_pa_subproc_runner.py  # at runtime: REPO_ROOT.parent / "infer" / "vllm"
│   ├── triton_ascend_bench/
│   └── ...
└── infer/                             # upstream sibling checkout (outside bench repo)
    ├── vllm/                          # ../infer/vllm relative to bench repo root
    │   └── vllm/triton_utils/__init__.py
    └── sglang/python/                 # ../infer/sglang/python relative to bench repo root
        └── sglang/__init__.py
```

来源凭据:`scripts/_pa_subproc_runner.py:57` 写明 `INFER_ROOT = REPO_ROOT.parent / "infer"`,即 bench repo 父目录下的 `infer/` 而**不是**bench repo 内部的 `infer/`。

引用规则:
- 文档主体引用 upstream 文件路径,**统一写** `../infer/vllm/...` 和 `../infer/sglang/python/...`(以 bench 根为参考)
- 例外:`PYTHONPATH=...` 字符串、Python 字符串字面量(如 `spec_from_file_location` 第二参数)、verification 命令中的相对路径均以 bench 根为 cwd,因此 `../infer/...` 与其它引用一致
- `scripts/_pa_subproc_runner.py` 等代码内是以 `Path(__file__)` 起算的绝对路径解析,不依赖 cwd,与本约定无冲突

## 1. Executive Summary

近一周已 push 上 main 的 11 个 commit(详见 §2)覆盖了 shim 层 leverage 的大部分批量收益:loader bridge / sglang LoRA env / vllm-sglang transitive deps / auto_shapes_filler / 'struct:'+'ref:' input forms。**fresh resolver 跑完确认**(`results/resolve_all.json` 2026-05-25 17:31,100 verdicts):**剩余 BLOCKED = 96**(PASS = 4)。分布如下:

- `shapes-incomplete` = **71**(比早期预测的 ~10 显著多;auto-filler 解了 conventional 命名,但复杂 paged-attn / KV cache / MoE / mamba SSD / EP 类的 task-specific 命名占大头)
- `upstream-not-invocable-here` = **22**(包含 P3 forward shim 漂移类 ~5-10 个 + P4 残留缺包 / sglang vendored module shadow / 其它边角 ~12-17 个)
- `oracle-mismatch` = **2**(L1_01 + L1_09,A2 修)
- `smoke-fail` = **1**(L2_10,A3 判定)

剩余 BLOCKED 按修复性质明显分两类:

- **systemic / 跨 task**:一处改动一次解多个 task。代表是 P3 forward shim(`vllm.triton_utils` placeholder 漂移),根因是 vllm 端 `TritonPlaceholder` 把 `triton.next_power_of_2 / autotune / Config` 等公有 API 替换成 dummy,bench 端 JIT 重新求值时缺这些符号。预估 5-10 个 P3 task 一次性解决(从 22 个 upstream-not-invocable-here 中切出)。
- **per-task / 机械化**:每个 task 单独 walk-through(改 shapes.yaml / meta.yaml / 偶尔改 oracle_pytorch.py),没有共享 leverage。代表是剩余 71 个 shapes-incomplete 的复杂 op(paged-attn / MoE / KV cache / mamba SSD / EP)和剩余 ~12-17 个 P4 残留缺包。

两类工作负载性质完全不同,**并行模型**自然成立:**session A**(user-driven session)做 systemic / 复杂 / 跨 task 的修复(A own = 3 task + shim 子树);**session B**(并行 Claude session)做 per-task 机械化 walk-through 和归档(B own = **93 task** = 71 shapes-incomplete + 22 upstream-not-invocable-here)。两者通过严格 task ownership 划分(§6),不接触对方 own 的文件,通过 push-pull 协议(§7)同步进度。

最终验收目标(§10):本轮工作完成后,**总 BLOCKED → 0**(全 RESOLVED + GPU sign-off 待 phase B)**或**剩余真必须 phase B 的归档项 < 5,且全部已 meta.yaml 加 `phase_b_capture_required: true`;全套 pytest 0 fail 0 skip;M0 release boundary 不破。

## 2. 已 commit 进度盘点

按 commit 时序(自 `b534449` loader bridge 起)列已 push 上 main 的 11 个 commit(表中实际行数为准):

| Commit | 内容 | 解决类 |
|---|---|---|
| `b534449` | `fix(loader): resolve kernel.py relative imports against upstream sibling tree` | P2 relative import,loader 能正确解析 kernel.py 的 sibling 导入 |
| `0ef5377` | `test: cover upstream-sibling loader bridge in both runners` | 测试覆盖 `b534449` 的 loader bridge,确保 `_pa_oracle_check_runner` 和 `_pa_subproc_runner` 行为一致 |
| `212f46f` | `fix(L2_04-09): disable sglang custom triton kernel cache in oracle subprocess` | P3 sglang LoRA,subprocess env 关掉 sglang 自定义 triton kernel cache |
| `d2578b8` | `fix(L2_04-L2_09): mark sglang LoRA kernels skip_invocation, defer oracle to Phase B` | 6 个 LoRA task `phase_a_prime.skip_invocation` + phase_b_capture |
| `10f8529` | `feat(resolve): add deterministic auto-filler for shapes-incomplete tasks` | shapes auto-filler,批量解 conventional param naming 的 shapes-incomplete |
| `10818c4` | `fix(resolve): cachetools/cpuinfo leaves, relative-import loader, three task contract bugs, JSON evidence plumbing` | P4 残留缺包(cachetools / cpuinfo)+ 3 个 task contract bug |
| `38ad475` | `fix(tasks): apply auto_shapes_filler + L1_03 codex refinements + partial-json-parser leaf` | L1_03 NUM_TOPK_TOKENS / BLOCK_N + partial-json-parser leaf |
| `b37ec5f` | `fix(auto_shapes_filler): extend conventions for cascade-revealed param names` | auto-filler 扩展 cascade 揭示出来的更多 param 命名 |
| `251baae` | `fix(infer): add vllm/sglang transitive runtime deps to [infer] extra` | pyproject `[infer]` extras 补充 vllm/sglang transitive 依赖 |
| `fb9489d` | `feat(inputs): support dataclass-shaped inputs via 'struct:' form (Path X)` | inputs 支持 dataclass-shaped 输入,LoRA 类 input fixture 解锁 |
| `2833a67` | `feat(inputs): add 'ref:' form to struct fields + apply to all 6 LoRA tasks` | inputs 支持 'ref:' 引用 struct 字段,LoRA 6 个 task 一并 apply |

> 上表覆盖了 shim 分析 §4 的 M-A / M-B / M-C 大半,只剩 P3 forward shim(M-A 的硬骨头)和 oracle-mismatch 精修两块未做。

## 3. 真实剩余 BLOCKED 状态

### 3.1 表 A — 早期预测(历史,已被 §3.2 fresh data 取代)

> **历史记录**:本表是 fresh resolver 跑之前基于 commit chain 推算的预测;§3.2 fresh data 已经取代它。保留作为方案演进对比(prediction vs reality)。

`resolve_all.json` 早期 snapshot 时(在 11 个 fix commit 之前)的 verdict 分布:

| 状态 | snapshot 数 | 早期预测 fix 后剩 | fresh resolver 实测(§3.2) | prediction gap |
|---|---|---|---|---|
| `PASS` | 3 | **3 + Δ_p**(显著上升) | **4**(+1) | Δ_p 只有 +1,不及预期 |
| `BLOCKED(shapes-incomplete)` | 56 | **5-10** | **71** | 实测大幅高于预测,auto-filler 覆盖率比预想低 |
| `BLOCKED(upstream-not-invocable-here)` | 38 | **3-9 P3 + 3-5 P4 残留** | **22** | 实测低于 38 snapshot,fix chain 解了部分 |
| `BLOCKED(oracle-mismatch)` | 2 | **2** | **2** | 准 |
| `BLOCKED(smoke-fail)` | 1 | **1**(静态分析,可能 phase B) | **1** | 准 |

合计 fresh 剩余 BLOCKED:**96**(并非早期预测的 10-20;以 §3.2 表 B 为权威)。

### 3.2 表 B — fresh resolver 实测(2026-05-25 17:31)

> **数据来源**:`results/resolve_all.json`(2026-05-25 17:31 fresh snapshot,100 verdicts)。本表为权威剩余 BLOCKED 计数,session B 工作量基于本表估算。

| 状态 | fresh resolver 计数 | 影响 session |
|---|---|---|
| `PASS` | 4 | — |
| `BLOCKED(shapes-incomplete)` | 71 | session B(B1) |
| `BLOCKED(upstream-not-invocable-here)` — P3 forward shim 类 | ~5-10(预估;A1 shim 解之) | session A(A1)起手解 |
| `BLOCKED(upstream-not-invocable-here)` — 其它(P4 残留 / sglang vendored / 边角) | ~12-17(预估;A1 shim 解完后剩) | session B(B2 + B3) |
| `BLOCKED(upstream-not-invocable-here)` — 合计 | 22 | A1 + B2 / B3 |
| `BLOCKED(oracle-mismatch)` | 2 | session A(A2) — L1_01 / L1_09 |
| `BLOCKED(smoke-fail)` | 1 | session A(A3) — L2_10 |
| **总 BLOCKED** | **96** | A 3 task + B 93 task |

> A1 shim **预估** leverage 5-10 个 P3 task;真实切分要等 A1 跑完才知道。剩余 22 - 实测 leverage = 真 B own upstream-not-invocable-here。本 plan 把全部 22 个 upstream-not-invocable-here 都列在 §3.4 B own task list 中(因为 A1 解的 task 也属 B 不主动改 oracle / shapes 的范畴 — 这些 task 在 shim 接管后会自动 RESOLVE,session B 跳过即可)。

### 3.3 fresh resolver 重跑命令(参考)

> **CLI 校准(codex r1 MF5)**:resolver 真实 CLI 是 `--refresh-cache`(隐含 `--all` 并写入 canonical cache,见 `scripts/resolve_upstream_entrypoints.py:1993-2010`)+ `--workers <int>`(默认 1,**不**支持 `auto`,见 `:1985-1991`)。**没有** `--output` 参数(用 `--json <PATH>` 或 `--refresh-cache` 写 canonical cache)。

```bash
cd /data1/code/agentic-rl/kernel/triton-ascend-bench

# Step 1: back up the current canonical cache for old-vs-new diff.
# Canonical path is results/resolve_all.json (see CANONICAL_RESOLVE_CACHE
# in scripts/resolve_upstream_entrypoints.py:97). The repo-root resolve_all.json
# is a legacy/stale snapshot and must NOT be used as the data source.
cp results/resolve_all.json results/resolve_all.json.bak

# Step 2: run fresh resolver. --refresh-cache implies --all and writes back
# to canonical cache (results/resolve_all.json).
python3 scripts/resolve_upstream_entrypoints.py --refresh-cache --workers 4

# Step 3: diff old vs new (verdict_class distribution change).
PYTHONPATH=scripts python3 -c "
from automation.lib.resolve_loader import load_resolve_artifact
from collections import Counter
new = load_resolve_artifact('results/resolve_all.json')
old = load_resolve_artifact('results/resolve_all.json.bak')
print(f'fresh: {len(new)} verdicts')
print(f'stale: {len(old)} verdicts')
print('verdict_class delta (old -> new):')
nc = Counter(v['verdict_class'] for v in new)
oc = Counter(v['verdict_class'] for v in old)
for k in sorted(set(nc) | set(oc)):
    print(f'  {k:35} {oc.get(k,0):3} -> {nc.get(k,0):3}')

# Per-task status flip detail.
old_by_tid = {v['task_id']: v['verdict_class'] for v in old}
new_by_tid = {v['task_id']: v['verdict_class'] for v in new}
print()
print('Per-task flips:')
for tid in sorted(set(old_by_tid) | set(new_by_tid)):
    o = old_by_tid.get(tid, '-')
    n = new_by_tid.get(tid, '-')
    if o != n:
        print(f'  {tid:60} {o:25} -> {n}')
"
```

### 3.4 Session B own — 完整 task list(fresh resolver,2026-05-25 17:31)

> 数据来源:`results/resolve_all.json`。下表把 session B own 的全部 93 个 task 按 verdict_class 分组完整列出,session B 直接照清单逐批走 §5 B1 / B2 / B3 流程,**不需要**再自己跑命令。若需要重新生成(例如 fresh resolver 再跑一遍),命令在 §3.4.3。

#### 3.4.1 shapes-incomplete (71 task — session B own)

> 按 §5 B1 流程逐批 walk-through;简单 elementwise 类 5-7 个一个 commit,复杂 paged-attn / KV cache / MoE 类 1-3 个一个 commit。

- L1_02_compute_slot_mappings
- L1_04_copy_page_indices
- L1_05_gather_block_tables
- L1_06_reshape_and_cache_kernel_flash
- L1_07_dequantize_k_cache_fast
- L1_08_dequantize_k_cache_paged
- L1_10_per_token_group_quant_int8
- L1_11_awq_dequantize
- L1_15_act_and_mul
- L1_17_gelu_and_mul
- L1_18_fused_softcap
- L1_19_topk_log_softmax
- L1_20_log_softmax
- L1_21_gumbel_sample
- L1_24_rejection_sample
- L1_27_apply_grammar_bitmask
- L1_28_pack_bitmatrix
- L1_29_prepare_eagle_inputs
- L1_30_act_quant
- L1_31_triton_mrope_forward
- L1_32_create_flashinfer_kv_indices_triton
- L1_33_create_flashmla_kv_indices_triton
- L1_35_compute_src2dst_triton
- L1_36_compute_seg_indptr_triton
- L1_39_state_passing_fwd
- L1_40_chunk_cumsum_fwd
- L2_02_chunk_scan_fwd
- L2_03_chunk_state_fwd
- L2_11_w8a8_triton_block_scaled_mm
- L2_12_awq_gemm
- L2_13_bmm
- L2_14_matmul_kernel_persistent
- L2_15_fused_dual_residual_rmsnorm
- L2_16_fused_rmsnorm
- L2_17_layer_norm_gated_fwd
- L2_18_fused_gdn_gating
- L2_19_fused_sigmoid_gating_delta_rule_update
- L2_20_causal_conv1d_update
- L2_21_causal_conv1d_fwd
- L2_22_linear_attn_decode
- L2_23_fwd_kv_parallel
- L2_24_fwd_kv_reduce
- L2_25_fwd_kernel_ep_gather
- L2_27_deepep_permute_triton
- L2_28_deepep_post_reorder_triton
- L2_29_fused_moe_router_cudacore
- L2_30_fused_moe_router_tensorcore
- L2_31_moe_align_block_size_stage2
- L2_32_experts_combine
- L2_33_fwd_grouped_kernel_stage1_rope
- L2_34_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe
- L2_35_pre_reorder_triton_kernel_for_cutlass_moe
- L2_36_post_reorder_triton_kernel_for_cutlass_moe
- L2_38_concat_and_cast_mha_k
- L3_01_kernel_unified_attention_3d
- L3_02_kernel_paged_attention_2d
- L3_03_chunk_state_varlen
- L3_04_fwd
- L3_05_fwd_kernel_alibi
- L3_06_fwd_grouped_kernel_stage1
- L3_07_correct_attn_cp_out
- L3_11_expert_triton
- L3_12_moe_mmk
- L3_14_fused_recurrent_gated_delta_rule_fwd
- L3_15_fused_recurrent_gated_delta_rule_update_fwd
- L3_17_chunk_gla_fwd_kernel_o
- L3_19_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter
- L3_20_recompute_w_u_fwd
- L3_21_solve_tril_16x16
- L3_22_chunk_local_cumsum_vector
- L3_23_merge_16x16_to_32x32_inverse

#### 3.4.2 upstream-not-invocable-here (22 task — session B own,A1 shim 解后剩余真 B 工作)

> A1 shim 接管后**预估 5-10 个 task 自动 RESOLVE**(P3 forward shim leverage,见 §A1.6);真实切分等 A1 跑完后看 fresh resolver re-run。session B **不**主动改这 22 个 task,先 stage 在 backlog,A 推完 shim 后 B 拉新代码 re-run resolver,余下 ~12-17 个再按 B2(P4 残留缺包)/ B3(归档)流程处理。

- L1_03_convert_req_index_to_global_index
- L1_13_layer_norm_fwd
- L1_14_layer_norm_fwd_1pass
- L1_22_min_p
- L1_23_penalties_and_temperature
- L1_25_merge_attn_states
- L1_37_compute_problem_sizes_w4a8
- L1_38_silu_mul_fp8_quant_deep_gemm
- L2_01_bmm_chunk_fwd
- L2_04_chunked_lora_expand
- L2_05_chunked_lora_shrink
- L2_06_qkv_lora_b
- L2_07_gate_up_lora_b
- L2_08_sgemm_lora_a
- L2_09_sgemm_lora_b
- L2_26_fwd_kernel_ep_scatter_1
- L2_37_silu_and_mul_post_quant
- L3_09_fused_moe
- L3_10_fused_moe_kernel_gptq_awq
- L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64
- L3_16_chunk_fwd_kernel_o
- L3_18_chunk_scaled_dot_kkt_fwd

> **小计:71 + 22 = 93 task,session B own**(A own 的 3 个 task `L1_01_bias / L1_09_quantize_k_cache_fast / L2_10_w8a8_block_int8_matmul` 不计入)。

#### 3.4.3 重新生成命令(只在 fresh resolver 再跑后用)

```bash
cd /data1/code/agentic-rl/kernel/triton-ascend-bench

PYTHONPATH=scripts python3 -c "
from automation.lib.resolve_loader import load_resolve_artifact
from collections import defaultdict
r = load_resolve_artifact('results/resolve_all.json')
A_OWN = {'L1_01_bias', 'L1_09_quantize_k_cache_fast', 'L2_10_w8a8_block_int8_matmul'}
by_class = defaultdict(list)
for v in r:
    if v['verdict_class'] != 'PASS' and v['task_id'] not in A_OWN:
        by_class[v['verdict_class']].append(v['task_id'])
for cls in sorted(by_class):
    print(f'### B own {cls} ({len(by_class[cls])} task)')
    for tid in sorted(by_class[cls]):
        print(f'- {tid}')
    print()
"
```

## 4. Session A 任务清单(user-driven session,我做)

Session A 处理 systemic / 复杂 / 跨 task 修复,共 **A1 / A2 / A3 三块**。

### A1. `vllm.triton_utils` forward shim(P3 一次解 3-9 task)

#### A1.1 Goal

把 vllm 端 `TritonPlaceholder` 替换的 dummy Triton API 重新映射回真正的 public Triton,同时提供 interpreter-aware autotune fallback。一处改动,一次解决全部 P3 `'function' object is not subscriptable / 'triton' has no attribute 'next_power_of_2'` 类失败。

#### A1.2 Files 改动清单

**注入机制(取代 sys.path 覆盖 — codex r1 MF1)**:不**新建**空的 `vllm/__init__.py` package shim(会遮蔽真 `vllm.utils.math_utils / vllm.logger` 等真子模块,实测 `import vllm.utils.math_utils` 抛 ModuleNotFoundError);改为在 runner 启动早期用 **`sys.modules['vllm.triton_utils'] = <shim_module>`** **显式注入**:先 `import vllm`(让真 vllm package 完成 `__init__`),再用 `sys.modules` 注册替换条目。这样精准只替换 `vllm.triton_utils` 子模块,其它子模块照常从真上游 import,不破坏真 vllm package。

| File | 类型 | 说明 |
|---|---|---|
| `triton_ascend_bench/upstream_shims/__init__.py` | new | 空 `__init__.py`,确保是 Python 包(只起 namespace 作用,不被 `import vllm.*` 走到) |
| `triton_ascend_bench/upstream_shims/vllm_triton_utils_shim.py` | new | 实际 shim module 实现(独立文件;命名故意不含 `vllm/`,避免被误当作 package shim 遮蔽真 vllm)。文件提供 `triton / tl / tldevice / HAS_TRITON / LOG2E / LOGE2 / KernelProxy` |
| `triton_ascend_bench/upstream_shims/_inject.py` | new | 提供 `register()` 函数:`import sys; from . import vllm_triton_utils_shim as _s; sys.modules['vllm.triton_utils'] = _s;`(同时也 register `vllm.triton_utils.importing` 子条目如果需要)。在 runner subprocess 启动早期调用 |
| `scripts/_pa_oracle_check_runner.py` | modify | 在 `import vllm` 后立即 `from triton_ascend_bench.upstream_shims._inject import register; register()`;不动 `sys.path` 顺序 |
| `scripts/_pa_subproc_runner.py` | modify | 同款,与 oracle runner 行为一致 |
| `tests/unit_tests/test_upstream_shims.py` | new | 至少 12 个 unit test 覆盖每个 API + 3 个端到端 mock load |

**备选方案(方案 b)**:若必须用 package shim(例如未来 vllm 重组目录),则 shim `vllm/__init__.py` 必须 `__path__ = [<shim_dir>, str(VLLM_REAL_ROOT / 'vllm')]` 把真 vllm package `__path__` 接续进来,而非置空。优先用 `sys.modules` 注入,**不**用方案 b(精度差,易破其它子模块)。

#### A1.3 Algorithm 细节

`KernelProxy.__getitem__(grid)` 提供 subscriptable launcher:

- 输入 `kernel[grid]`,bind 到真 `triton.JITFunction`;
- 返回 partial-applied callable,call 时 forward 到真 `triton.JITFunction.run(grid=grid, ...)`;
- 兼容 vllm 习惯写法 `kernel[grid](args, ...)`。

Interpreter-aware `autotune` fallback:

- 检测 `os.environ.get("TRITON_INTERPRET") == "1"` 或 dev/CPU 环境(无 active triton driver);
- **driver 检测不能直接 access `triton.runtime.driver.active`** — 在 dev 机直接 access 会抛 `RuntimeError: 0 active drivers`(codex r1 MF2 已实测)。改成参照 `../infer/vllm/vllm/triton_utils/importing.py` 用 `triton.backends.backends` 数 active drivers,并裹 try/except:

  ```python
  def _has_active_triton_driver() -> bool:
      """Return True only when a real Triton driver is active. Mirrors
      vllm's importing.py pattern; never raises on a CPU-only dev box."""
      try:
          import triton
          from triton.backends import backends
          active = [
              x.driver for x in backends.values()
              if getattr(x, "driver", None) and x.driver.is_active()
          ]
          return len(active) >= 1
      except (RuntimeError, AttributeError, ImportError):
          return False
  ```

  （等价的、最小但够用的 fallback 也行:`try: return len(triton.runtime.driver.active.utils.devices) > 0; except (RuntimeError, AttributeError, ImportError): return False` — 关键是必须裹 try/except 兜住 `RuntimeError: 0 active drivers`。）
- 命中 fallback 分支时(无 driver / `TRITON_INTERPRET=1`):不做 benchmark,直接拿 `configs[0]` 作 chosen config,wrap kernel 为 JIT 直跑;
- 未命中(真 GPU 环境):透传到 `triton.autotune` 原始实现。

必须覆盖的 vllm placeholder 替换的 API:

```
# Module-level (mirror upstream __all__)
HAS_TRITON          # bool, True iff a real triton driver is active (see _has_active_triton_driver above)
triton              # real triton module re-export
tl                  # real triton.language re-export
tldevice            # real triton.language.extra.libdevice re-export (vllm alias)
LOG2E               # constant 1.4426950408889634
LOGE2               # constant 0.6931471805599453

# triton.* attributes the shim must expose
triton.jit
triton.heuristics
triton.autotune              # wrapped with interpreter-aware fallback (§A1.3 above)
triton.Config
triton.cdiv
triton.next_power_of_2

# tl.* attributes
tl.constexpr
tl.float32 / tl.float16 / tl.bfloat16
tl.int32 / tl.int64
tl.arange / tl.load / tl.store / tl.zeros / tl.full / tl.where
tl.dot / tl.exp / tl.log / tl.maximum / tl.minimum
tl.program_id / tl.num_programs

# KernelProxy (bench-internal helper, not in upstream __all__)
KernelProxy         # subscriptable launcher wrapper, exposed at vllm.triton_utils.KernelProxy
```

**`__all__` mirror 要求(codex r1 MF3)**:shim module 必须 export `__all__` 至少包含 upstream `../infer/vllm/vllm/triton_utils/__init__.py.__all__`(目前为 `["HAS_TRITON", "triton", "tl", "tldevice", "LOG2E", "LOGE2"]`),在 `tests/unit_tests/test_upstream_shims.py` 加一条 assertion:

```python
def test_all_mirrors_upstream():
    """shim __all__ must be a superset of upstream's, so bench callers that
    do `from vllm.triton_utils import *` see every symbol upstream exposes."""
    import importlib, sys
    # Load upstream module via direct path (NOT through the shim injection)
    spec = importlib.util.spec_from_file_location(
        "_upstream_vllm_triton_utils",
        "../infer/vllm/vllm/triton_utils/__init__.py",
    )
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    from triton_ascend_bench.upstream_shims import vllm_triton_utils_shim as shim
    assert set(shim.__all__) >= set(upstream.__all__), (
        f"shim missing: {set(upstream.__all__) - set(shim.__all__)}"
    )
```

#### A1.4 Test 用例(`tests/unit_tests/test_upstream_shims.py`)

| Case | 验证 |
|---|---|
| `test_re_export_triton_jit` | `from vllm.triton_utils import triton; assert triton.jit is real_triton.jit` |
| `test_kernel_proxy_subscriptable` | 装一个 dummy `@triton.jit` kernel,`kernel[(1,)](...)` 不抛 `'function' object is not subscriptable` |
| `test_autotune_interpreter_fallback` | `TRITON_INTERPRET=1`,`@triton.autotune(configs=[Config(...)])` 装饰的 kernel 能 invoke,不会 benchmark |
| `test_autotune_real_gpu_passthrough` | mock `len(triton.runtime.driver.active.utils.devices) > 0`,autotune 走原 path |
| `test_next_power_of_2` | `triton.next_power_of_2(7) == 8`,值匹配真 triton |
| `test_cdiv` | `triton.cdiv(10, 3) == 4` |
| `test_tl_constexpr` | `tl.constexpr` 与真 triton 同对象 |
| `test_tl_dtype_aliases` | float32 / float16 / bfloat16 / int32 / int64 都是真 triton 的引用 |
| `test_heuristics_passthrough` | `@triton.heuristics({...})` 装饰能正常生效 |
| `test_e2e_L1_22_min_p_load` | mock load `triton_ascend_bench/frozen/snapshots/L1_22_min_p_kernel_18f6bf5.py`,resolve 入口到 `apply_min_p`,不抛 subscriptable 错 |
| `test_e2e_L1_14_layer_norm_fwd_1pass_load` | 类似,attribute `next_power_of_2` 解析成功 |
| `test_e2e_L2_01_bmm_chunk_fwd_load` | 类似,`tl.dot` 等属性可访问 |

#### A1.5 Verification 命令

```bash
cd /data1/code/agentic-rl/kernel/triton-ascend-bench

# Direct import smoke test -- primary path uses sys.modules injection (see plan).
# bench package (incl. upstream_shims) must come first on PYTHONPATH; the real
# vllm sibling tree follows. Order is not strictly required by the injection
# scheme but kept stable so vllm submodule lookup remains predictable.
PYTHONPATH=triton_ascend_bench/upstream_shims:../infer/vllm:../infer/sglang/python python3 -c "
import sys
# 1) Load the real vllm package first.
import vllm
# 2) Inject the shim to replace vllm.triton_utils.
from triton_ascend_bench.upstream_shims._inject import register
register()
# 3) Verify the shim is active.
from vllm.triton_utils import triton, tl, tldevice, HAS_TRITON, LOG2E, LOGE2
mod = sys.modules['vllm.triton_utils']
assert mod.__name__.startswith('triton_ascend_bench.upstream_shims'), \
    f'shim NOT active, mod is {mod.__name__}'
print('shim active:', mod.__name__)
print('triton.next_power_of_2:', triton.next_power_of_2)
print('triton.autotune (wrapped):', triton.autotune)
print('HAS_TRITON:', HAS_TRITON)
print('LOG2E, LOGE2:', LOG2E, LOGE2)
# 4) Other real-vllm submodules must still import normally (not shadowed).
from vllm.utils.math_utils import cdiv  # must NOT raise
print('vllm.utils.math_utils.cdiv resolved OK:', cdiv)
"

# Single-task verify (representative P3 cases).
python3 scripts/resolve_upstream_entrypoints.py --task L1_22_min_p
python3 scripts/resolve_upstream_entrypoints.py --task L1_14_layer_norm_fwd_1pass
python3 scripts/resolve_upstream_entrypoints.py --task L2_01_bmm_chunk_fwd

# Unit tests.
pytest tests/unit_tests/test_upstream_shims.py -v
```

> **注意(codex r1 MF4)**:之前的旧设计想让 `sys.path` 优先级翻转(把 shim root 放到 `../infer/vllm` 之前)实现替换。这做法**根本不解决** `vllm.triton_utils` 子模块替换问题(且会附带遮蔽真 `vllm.utils.*` 子模块,见 MF1)。最终方案改为 **`sys.modules` 显式注入**(§A1.2),`sys.path` 顺序不再是 correctness 依赖(这一段为历史叙述,仅说明为何被废弃)。

#### A1.6 Expected Leverage

**fresh resolver 校准后预估 5-10 个 P3 task RESOLVED**(基于 §3.4.2 的 22 个 upstream-not-invocable-here task,reason 含 `next_power_of_2 / not subscriptable / has no attribute` 关键词)。A1 shim 跑完后真解 task 数 = fresh re-run resolver 实测,**剩余** 22 − 实测 ≈ 12-17 个 P4 残留 / sglang vendored / 边角进 B2 / B3。代表受益 task:

- L1_14_layer_norm_fwd_1pass(`triton has no attribute 'next_power_of_2'`)
- L1_22_min_p(`'function' object is not subscriptable`)
- L2_01_bmm_chunk_fwd(`triton.language has no attribute ...`)
- L1_13_layer_norm_fwd(同 L1_14 模式)
- L1_23_penalties_and_temperature / L1_25_merge_attn_states(候选,看 actual kernel)

#### A1.7 Commit 模板

```
feat(shim): add vllm.triton_utils forward shim for P3 placeholder drift

Replace vllm-end TritonPlaceholder dummies with re-exports of real public
Triton API. KernelProxy exposes __getitem__ for subscriptable launcher
syntax (kernel[grid](...)) used in vllm Triton kernels. autotune falls
back to first config under TRITON_INTERPRET=1 to avoid benchmark on CPU.

Resolves <N> P3 BLOCKED tasks (next_power_of_2 / subscriptable failures).
```

### A2. Oracle-mismatch 2 task 精修(L1_01 / L1_09)

#### A2.1 L1_01_bias

**Reason snapshot**:`meta points at kernel.py::apply_logit_bias but it does NOT pass the CURRENT hardened oracle-consistency anchor (oracle-mismatch)`。

**Files**:
- `tasks/level1/L1_01_bias/oracle_pytorch.py`(modify)
- `tasks/level1/L1_01_bias/meta.yaml`(modify,如 tolerance class 改动则触及;尽量不动)

**Walk-through(3-stage rewrite)**:
1. Stage 1:read `apply_logit_bias` kernel.py — 抽出 logical pseudocode,注意 allowed-token mask / bias add / stop-token mask 三阶段处理 + 返回值结构(单 logits tensor 还是带 stop mask 的 tuple)。
2. Stage 2:重写 `oracle_pytorch.py`,严格按 3 阶段 PyTorch 等价:
   - allowed mask:`logits = logits.masked_fill(~allowed_mask, -float('inf'))`
   - bias add:`logits = logits + bias_tensor`(注意 broadcast 维度)
   - stop mask:`logits = logits.masked_fill(stop_mask, -float('inf'))`
3. Stage 3:确保 oracle 只返单 logits tensor(若 contract 要求);若 kernel 多输出则 oracle 也要 wrap 成同样 tuple/dict 结构。

**Verification**:
```bash
python3 scripts/resolve_upstream_entrypoints.py --task L1_01_bias
# Expected: status: PASS
```

#### A2.2 L1_09_quantize_k_cache_fast

**Reason snapshot**:`BLOCKED(no-verified-upstream-entrypoint: oracle-mismatch): tried ['_quantize_k_cache_fast', '_quantize_k_cache_fast_separate']; ... upstream-not-invocable-here [AssertionError`(实际是混合 oracle-mismatch + assertion 路径)。

**Files**:
- `tasks/level1/L1_09_quantize_k_cache_fast/oracle_pytorch.py`(modify)
- `tasks/level1/L1_09_quantize_k_cache_fast/meta.yaml`(modify,可能改 `tolerance_class`)

**两选一方案**:

| 方案 | 操作 | 风险 |
|---|---|---|
| 方案 1 — tolerance bump | `meta.yaml: tolerance_class: quant_specific_loose` | 简单;但要确认这个 class 已在 `triton_ascend_bench/tolerance_policy.yaml` 定义,且 quantization 类 task 普遍适用 |
| 方案 2 — oracle 精确化 | 改 `oracle_pytorch.py` 精确匹配 kernel 的 rounding mode(banker / round-half-up / round-toward-zero)+ saturating int8 clip + scale factor 准 mantissa | 工作量大但严谨;对照 kernel 中 `tl.maximum/tl.minimum` 与 PyTorch `clamp` 是否一致 |

**推荐**:**先 试方案 1**(成本低),不通过再走方案 2。

**Verification**:同 §A2.1。

#### A2.3 A2 整体 Commit 模板

```
fix(tasks): resolve oracle-mismatch on L1_01_bias and L1_09_quantize_k_cache_fast

L1_01_bias: rewrite oracle_pytorch.py to match kernel's 3-stage logits
processing (allowed-mask -> bias add -> stop-mask) and return single logits
tensor matching the kernel contract.

L1_09_quantize_k_cache_fast: <bump tolerance | tighten rounding mode>.
```

### A3. L2_10 smoke-fail 静态分析

#### A3.1 Reason snapshot

`meta points at kernel.py::w8a8_block_int8_matmul but ... REAL S2-smoke now smoke-fail [S2-gpu NOT equivalent (phase=original-no-run): original side did not run (rc=2): CompilationError: at 57:23: accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)`。

#### A3.2 Files

- `tasks/level2/L2_10_w8a8_block_int8_matmul/kernel.py`(read-only,**不可改 frozen**)
- `tasks/level2/L2_10_w8a8_block_int8_matmul/meta.yaml`(若判定必须 phase B,**A 自己**落 phase_b_capture 归档块;不交给 B)
- `tasks/level2/L2_10_w8a8_block_int8_matmul/shapes.yaml`(若判定为候选 1 dev-fixable,A 自己改;不交给 B)
- 本 doc(A own task 全生命周期;§5 B3 不收 L2_10 条目)

#### A3.3 静态分析步骤

1. Read `tasks/level2/L2_10_w8a8_block_int8_matmul/kernel.py`(line 50-80),理解 `accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)` 的上下文(已确认是标准 matmul accumulator 初始化)。
2. Diagnosis 候选:
   - **候选 1**:`BLOCK_SIZE_M` / `BLOCK_SIZE_N` 未在 autotune `configs` 提供(constexpr 未 bind)→ 改 meta.yaml 补 param_constraints 或 shapes.yaml `inputs:` 补 BLOCK constants。
   - **候选 2**:dtype `tl.float32` 与硬件不兼容(某些 NPU backend 不支持 fp32 accumulator)→ 必须真 GPU 跑才能确认,标 phase B。
   - **候选 3**:Triton 版本与 kernel.py 写法不兼容(triton 3.x 新 API 与 2.x 老 kernel 差异)→ 看 vendored hash + upstream commit,可能改 oracle path 走 phase B。
3. 输出 **静态分析报告**(直接写到本 §A3.5,不写额外 .md)说明候选和判定。

#### A3.4 期望产出

> **Ownership 锁定**:L2_10 在 candidate 1 / 2 / 3 任一分支下,**A own task 全生命周期**(static analysis 报告 + meta.yaml / shapes.yaml 实际改写 + commit)。session B 不接任何 L2_10 改动,§5 B3 不收 L2_10 条目。

- 若候选 1(可 dev 修):**A 自己**给 meta.yaml / shapes.yaml diff 并落盘,自走 P3 / shapes-incomplete fix path,**不归 smoke-fail**。
- 若候选 2 / 3(必须 GPU):**A 自己**按 §B3.2 schema 写完整 meta.yaml 归档块(`phase_a_prime.skip_invocation + status:SKIPPED_BY_DESIGN + reason`,`phase_b_capture_required: true`,`phase_b_capture` 含 `upstream_callable.{module, function}` 两字段、`status: PENDING`、`reason`)并自己 commit。**不**移交给 session B。

#### A3.5 分析结论(占位,A 实际执行时填充)

_待 session A 执行 A3 步骤后填充:候选编号 + 判定 + 必要的 diff 草稿或归档条目。_

### A4. Session A 验证 + commit + push

#### A4.1 验证 checklist

```bash
cd /data1/code/agentic-rl/kernel/triton-ascend-bench

# Full pytest suite (skip is not allowed).
# Use set -o pipefail + PIPESTATUS to capture pytest's real exit code (the
# trailing | tail would otherwise swallow the failure status; cf. plan §10.3).
set -o pipefail
python3 -m pytest tests/ -v 2>&1 | tee /tmp/pytest_A.log | tail -40
PYTEST_RC=${PIPESTATUS[0]}
if [[ "$PYTEST_RC" -ne 0 ]]; then
    echo "[FAIL] pytest rc=$PYTEST_RC (see /tmp/pytest_A.log)"
    exit 1
fi
echo "[OK] pytest"
# Expected: 0 failed, 0 skipped.

# ruff (exit code is direct, no pipe needed).
ruff check triton_ascend_bench/upstream_shims/ tests/unit_tests/test_upstream_shims.py
ruff format --check triton_ascend_bench/upstream_shims/ tests/unit_tests/test_upstream_shims.py

# Chinese-character check (repo-wide convention).
git grep -nP '[\x{4e00}-\x{9fa5}]' -- '*.py' ':!*/3rdparty/*' ':!docs/'
# Expected: 0 hits.

# Process-noise check on code files. This plan doc allows the technical terms
# `codex / Mx / Px / Ax / Bx` (see plan section 0); negative pathspec excludes
# this plan and its two referenced review docs from the scan, while every
# other *.py and *.md file remains in scope.
git grep -nIE '\b(codex|claude|round[- ]?[0-9]|sanity findings|[RHDFP][0-9]+\b|V[1-9])' \
    -- '*.py' '*.md' \
    ':!docs/triage_backlog/2026-05-25-parallel-execution-plan.md' \
    ':!docs/triage_backlog/2026-05-25-blocked-task-resolution-plan.md' \
    ':!docs/triage_backlog/2026-05-25-bench-shim-layer-analysis.md'
# Expected: only legitimate exceptions (API paths v1/v2, hardware names like A100).
# The plan doc plus the two referenced review docs are explicitly excluded via
# negative pathspec because they are operator-facing coordination docs that
# allow technical terms such as `codex / Mx / Px` (see plan section 0).

# Release boundary unchanged.
pytest tests/unit_tests/test_release_boundary.py -v
# Expected: 9 passed.

# Fresh-resolver single-task re-check.
for tid in L1_14_layer_norm_fwd_1pass L1_22_min_p L2_01_bmm_chunk_fwd L1_01_bias L1_09_quantize_k_cache_fast; do
    python3 scripts/resolve_upstream_entrypoints.py --task "$tid"
done
```

#### A4.2 Commit 分批

3 个 commit:

1. `feat(shim): add vllm.triton_utils forward shim for P3 placeholder drift`(A1)
2. `fix(tasks): resolve oracle-mismatch on L1_01_bias and L1_09_quantize_k_cache_fast`(A2)
3. 视 A3 结论:`fix(tasks): patch L2_10 ...`(可 dev 修) **或** `chore(tasks): defer L2_10 smoke-fail to phase B golden capture`(必须 GPU)

#### A4.3 Push 前 codex review 闭环

按 CLAUDE.md `Codex 调用只用 humanize skill` 规则,通过 `humanize:ask-codex` 走 review-fix loop 直到 ALL PASS,再 push。**不**用 Bash 直接调用 `codex exec`。

## 5. Session B 任务清单(并行 Claude session)

Session B 处理 per-task 机械化 / 归档,共 **B1 / B2 / B3 / B4 四块**。

### B1. 剩余 shapes-incomplete 复杂 op walk-through

#### B1.1 Goal

fresh resolver 跑完后,列出 auto_shapes_filler 没解决的 task(预估:复杂 paged-attn / KV cache / MoE / mamba SSD 类),逐个走标准 walk-through。

#### B1.2 Task list(fresh resolver 实测)

> **完整 71 个 task 已列在 §3.4.1**。本节只点几个代表性子类型,帮 B 规划批次顺序。

按复杂度建议执行顺序(从易到难,先攒一批简单 elementwise PASS,再啃复杂 op):

- **简单 elementwise / scalar reduction**(批量 5-7 个 commit):L1_15_act_and_mul / L1_17_gelu_and_mul / L1_18_fused_softcap / L1_20_log_softmax / L1_21_gumbel_sample / L1_28_pack_bitmatrix / L1_30_act_quant / L2_29_fused_moe_router_cudacore / L2_30_fused_moe_router_tensorcore 等。
- **KV cache / paged-attn**(per-task 1-3 个 commit):L1_02_compute_slot_mappings / L1_04_copy_page_indices / L1_05_gather_block_tables / L1_06_reshape_and_cache_kernel_flash / L1_07_dequantize_k_cache_fast / L1_08_dequantize_k_cache_paged / L1_32_create_flashinfer_kv_indices_triton / L1_33_create_flashmla_kv_indices_triton / L3_01_kernel_unified_attention_3d / L3_02_kernel_paged_attention_2d / L3_03_chunk_state_varlen。
- **Mamba SSD / chunk scan**:L1_39_state_passing_fwd / L1_40_chunk_cumsum_fwd / L2_02_chunk_scan_fwd / L2_03_chunk_state_fwd / L2_17_layer_norm_gated_fwd / L2_22_linear_attn_decode / L3_14 / L3_15 / L3_17 / L3_19 / L3_20 / L3_22。
- **MoE / EP**:L2_25_fwd_kernel_ep_gather / L2_27 / L2_28 / L2_31 / L2_32 / L2_33 / L2_34 / L2_35 / L2_36 / L3_11_expert_triton / L3_12_moe_mmk。
- **Conv1d / causal conv**:L2_20_causal_conv1d_update / L2_21_causal_conv1d_fwd。
- **Quant / gemm**:L1_10_per_token_group_quant_int8 / L1_11_awq_dequantize / L2_11_w8a8_triton_block_scaled_mm / L2_12_awq_gemm / L2_13_bmm / L2_14_matmul_kernel_persistent / L2_38_concat_and_cast_mha_k。

> auto_shapes_filler 已经接管 conventional 命名(N / D / H / num_tokens / num_heads / head_dim);上述 task 主要卡在 task-specific 命名(num_seqs / num_blocks_per_seq / block_size / topk / num_experts / num_chunks 等),需要 per-task 改 shapes.yaml。

#### B1.3 单 task walk-through 流程(10-15 min / task)

```bash
cd /data1/code/agentic-rl/kernel/triton-ascend-bench

TASK_ID=<task_id>  # e.g. L1_06_reshape_and_cache_kernel_flash

# 0. Use find to resolve the task directory: shell does not expand quoted globs.
TASK_DIR=$(find tasks/level1 tasks/level2 tasks/level3 -maxdepth 1 -name "$TASK_ID" -type d 2>/dev/null | head -1)
if [[ -z "$TASK_DIR" ]]; then
    echo "FATAL: task $TASK_ID not found under tasks/level{1,2,3}/" >&2
    exit 1
fi
echo "Task dir: $TASK_DIR"

# 1. Read reason from the canonical resolver cache to see which params are missing.
# Canonical path is results/resolve_all.json (CANONICAL_RESOLVE_CACHE in resolver).
PYTHONPATH=scripts python3 -c "
from automation.lib.resolve_loader import load_resolve_artifact
verdicts = load_resolve_artifact('results/resolve_all.json')
hit = next((v for v in verdicts if v['task_id'] == '$TASK_ID'), None)
if not hit:
    print('no verdict for $TASK_ID -- resolver not run yet?')
else:
    print('STATUS:', hit.get('verdict_class'))
    print('REASON:', hit.get('reason'))
    print('TRIED:', hit.get('tried'))
"

# 2. Inspect kernel.py @triton.jit signature to confirm missing constexpr params.
grep -nE 'def |@triton.jit' "$TASK_DIR/kernel.py" | head -20

# 3. Inspect existing shapes.yaml and meta.yaml param_constraints.
cat "$TASK_DIR/shapes.yaml"
echo '---'
grep -A 20 'params:\|param_constraints:' "$TASK_DIR/meta.yaml" || true

# 4. Edit shapes.yaml to supply missing params (three forms: literal / pyexpr / 'struct:' + 'ref:').
#    Edit meta.yaml param_constraints if needed.

# 5. Single-task verify.
python3 scripts/resolve_upstream_entrypoints.py --task "$TASK_ID"
# Expected: status: PASS
```

#### B1.4 Commit 粒度

- 简单 elementwise 类(L1_15 / L1_17 / L1_18 / L1_20 / L1_21 / L1_25 / L1_27 类)**每 5-7 个 task 一个 commit**,标题 `fix(tasks): apply shapes for <N> elementwise tasks`。
- 复杂 paged-attn / KV cache / MoE 类**每个 task 一个 commit**(或 2-3 个同主题打包),标题 `fix(tasks): supply shapes for <task_id>`。

### B2. P4 残留缺包清理

#### B2.1 Goal

fresh resolver 跑完后,找仍 `ModuleNotFoundError: No module named X` 的 task,逐个判断:

| 缺包类型 | 处理 |
|---|---|
| 纯 Python dep(可 pip 装) | pip install + 加到 `pyproject.toml [infer]` extras |
| 跨 GPU runtime(deep_gemm / 硬件 specific) | 标 `phase_a_prime.skip_invocation: true`,defer 到 phase_b_capture |

#### B2.2 估算清单

fresh resolver 实测 `upstream-not-invocable-here` 22 个(§3.4.2 完整列表)。其中 A1 shim 预估解 5-10 个 P3 forward shim 类,剩 ~12-17 个属 B2(缺包)/ B3(归档)。A1 跑完后 fresh re-run resolver 确认真实切分。已知 fp8 / deep_gemm / sglang LoRA 类已归档到 §B3.3 候选列表;典型剩 B2 入口的:L1_03_convert_req_index_to_global_index / L1_37_compute_problem_sizes_w4a8 / L2_26_fwd_kernel_ep_scatter_1 / L2_37_silu_and_mul_post_quant 等需逐个看 reason 判断。

#### B2.3 Commit 模板

```
fix(infer): add <pkg> transitive dep to [infer] extras

Resolves <task_id> ModuleNotFoundError on importing kernel.py vendored
tree.
```

或 defer 类:

```
chore(tasks): mark <task_id> skip_invocation due to <reason>

<task_id>: imports <pkg> at module level which is GPU-runtime only and
not installable on dev. phase_b_capture entry added.
```

### B3. 极少数必须 phase B 的归档

#### B3.1 Goal

FP8 量化 / 硬件特异性 Triton compile 错 / FlashMLA / DeepEP 类 task 改 meta.yaml 加完整 phase_b_capture 块,等 GPU 机后续跑 `phase_b_golden_capture.py`。**L2_10 不在 B3 范围**(A own 全程,见 §A3 / §6.1)。

#### B3.2 Meta.yaml 归档模板

> **Schema 校准(codex r1 MF8)**:`specs/task_api_contract.md §6a.1` 强制 `skip_invocation: true` **必须**搭配 `status: SKIPPED_BY_DESIGN` 与非空 `reason`(否则 §6a.5 invariant 1 违反);`§6a.3-6a.5` 规定 `phase_b_capture.upstream_callable` 是**两字段嵌套结构** `{module, function}`,**不**是 `"module.path::function"` 单字符串语法。模板按规范如下:

```yaml
phase_a_prime:
  skip_invocation: true
  status: SKIPPED_BY_DESIGN
  reason: "<one-line reason: GPU-only runtime / FP8 unsupported on dev / Triton compile requires NPU>"

phase_b_capture_required: true

phase_b_capture:
  status: PENDING
  upstream_callable:
    module: vllm.v1.attention.backends.flashinfer   # canonical dotted Python path (no '::')
    function: convert_req_index_to_global_index     # single Python identifier
  reason: "<same as phase_a_prime.reason>"
```

字段语义:

- `phase_a_prime.skip_invocation` + `phase_a_prime.status = SKIPPED_BY_DESIGN`:bench 端跳过 phase A oracle invocation,把 truth-source 委托给 phase B golden capture(`specs/task_api_contract.md §6a.2`)。
- `phase_b_capture_required: true`:eligibility gate(`triton_ascend_bench/eligibility.py::release_status_pass`)在 `phase_b_capture.status != DONE` 时把 task 标 `release_status = BLOCKED`(`§6a.4`)。
- `phase_b_capture.upstream_callable.module`:**dotted Python module path**(`vllm.v1.attention.backends.flashinfer`),不允许 `::`(`§6a.3`)。
- `phase_b_capture.upstream_callable.function`:**单 Python identifier**(`convert_req_index_to_global_index`),不允许 dotted(多段 dotted 名必须落进 `module`,见 `§6a.3` 最末段)。
- 后续 phase B 在真 GPU 机跑 `scripts/phase_b_golden_capture.py` 时,把 `status` 从 `PENDING` 翻成 `DONE`,并补 `captured_at` ISO 8601 UTC timestamp。

#### B3.3 候选 task

- L1_38_silu_mul_fp8_quant_deep_gemm(FP8 + deep_gemm GPU-only)
- L3_09_fused_moe / L3_10_fused_moe_kernel_gptq_awq(GPU MoE)
- 部分 L3 chunk_gated / fused_recurrent(NPU 适配未跟进)

> **不含 L2_10**:`L2_10_w8a8_block_int8_matmul` 由 session A 在 A3 中全权处理(dev-fixable 时 A 自己改 meta/shapes,必须 GPU 时 A 自己落 phase_b 归档块),**不**进入 B3 list。

### B4. Session B 验证 + commit + push

#### B4.1 验证(每 commit 前)

```bash
# Single-task verify (run right after editing one task).
python3 scripts/resolve_upstream_entrypoints.py --task <task_id>

# Before a batch commit, run the full resolver and inspect the distribution
# delta. --workers takes an int (default 1); 'auto' is not supported (see §3.3).
python3 scripts/resolve_upstream_entrypoints.py --refresh-cache --workers 4

# Full pytest suite (skip is not allowed).
# Use set -o pipefail + PIPESTATUS to capture pytest's real exit code (the
# trailing | tail would otherwise swallow the failure status; cf. plan §10.3).
set -o pipefail
python3 -m pytest tests/ -v 2>&1 | tee /tmp/pytest_B.log | tail -40
PYTEST_RC=${PIPESTATUS[0]}
if [[ "$PYTEST_RC" -ne 0 ]]; then
    echo "[FAIL] pytest rc=$PYTEST_RC (see /tmp/pytest_B.log)"
    exit 1
fi
echo "[OK] pytest"

# Release boundary (exit code is direct, no pipe needed).
pytest tests/unit_tests/test_release_boundary.py -v
```

#### B4.2 Commit 粒度规则

不允许把 20 个 task 塞 1 个 commit。粒度:

- 同主题 5-7 个 task 一批(B1 简单 elementwise)
- 复杂 task 1-3 个一批(B1 复杂 op)
- B2 缺包按主题 1-3 个一批
- B3 归档按主题 1-3 个一批

#### B4.3 Push 前 review

同 A4.3:`humanize:ask-codex` review-fix loop 闭环。

## 6. 严格 Task Ownership

### 6.1 Session A own(我)

| 范围 | 文件 / task |
|---|---|
| **新建模块** | `triton_ascend_bench/upstream_shims/` 子树:`__init__.py`(空 namespace) + `vllm_triton_utils_shim.py`(shim 主体) + `_inject.py`(`register()` helper);**不**新建 `vllm/__init__.py` 或 `vllm/triton_utils/*` 子目录(避免 package shim 遮蔽真 vllm 子模块,见 §A1.2 MF1) |
| **测试** | `tests/unit_tests/test_upstream_shims.py`(new) |
| **Runner sys.modules 注入 hook** | `scripts/_pa_oracle_check_runner.py` / `scripts/_pa_subproc_runner.py`(在 `import vllm` 后调 `triton_ascend_bench.upstream_shims._inject.register()`;**不**动 `sys.path` 顺序) |
| **Task** | `L1_01_bias`(A2 oracle-mismatch) |
| **Task** | `L1_09_quantize_k_cache_fast`(A2 oracle-mismatch) |
| **Task** | `L2_10_w8a8_block_int8_matmul`(A3 smoke-fail 静态分析;**A own task 全生命周期**:dev-fixable 时 A 落 meta.yaml/shapes.yaml,必须 GPU 时 A 自己落 phase_b 归档块;**绝不**移交给 B) |
| **Doc** | 本 plan doc(`docs/triage_backlog/2026-05-25-parallel-execution-plan.md`)的所有更新 |

### 6.2 Session B own(并行 session)

| 范围 | 文件 / task |
|---|---|
| **所有其它 BLOCKED task** 的 `tasks/level*/<task_id>/{meta.yaml, shapes.yaml, oracle_pytorch.py}` | 除 A 显式 own 的 3 个 task 外的全部 |
| **pyproject** | `pyproject.toml [infer]` extras 补包(只 B 改) |
| **测试** | `tests/unit_tests/`(新增覆盖 B 改动的 test 文件;但**不**碰 `test_upstream_shims.py`) |

> **`triton_ascend_bench/frozen/snapshots/*` 是 frozen upstream snapshot,双方禁改(codex r1 MF9)**。Vendored 树由 `scripts/phase_a_freeze.py` 在专门的 re-freeze PR 中维护(参见 `docs/triage_backlog/2026-05-25-blocked-task-resolution-plan.md`)。bench 端 P2(relative import)修复**不**靠改 `_vendored`,而是利用 loader bridge(`b534449` 已 commit)在解析 kernel.py 时把 relative import 重定向到真 upstream sibling tree。session A / B 都**不**写 `triton_ascend_bench/frozen/snapshots/`。

### 6.3 共享(只读 / 单方写)

| 文件 | 读 | 写 |
|---|---|---|
| `scripts/_pa_oracle_check_runner.py` | 双方 | **只 A**(A1 加 `register()` 调用,sys.modules 注入) |
| `scripts/_pa_subproc_runner.py` | 双方 | **只 A**(A1 加 `register()` 调用,sys.modules 注入) |
| `pyproject.toml` | 双方 | **只 B**(`[infer]` extras);A 不动 |
| `results/resolve_all.json` | 双方 | **只 A 在 fresh resolver hook 时 update doc 表;不直接编辑 JSON** |
| `triton_ascend_bench/tolerance_policy.yaml` | 双方 | 双方都不动(若需新 class,先沟通) |
| `tasks/level*/<task_id>/kernel.py` | 双方读取分析 | **frozen,任何一方都不改** |
| 本 plan doc | 双方 | **只 A**(B 只读) |

### 6.4 Ownership 穷举完整性自检

修复涉及的所有可写文件类型在 §6.1–§6.3 中:

- ✓ shim module(A own)
- ✓ A 的 3 个 task(L1_01 / L1_09 / L2_10)
- ✓ B 的剩余全部 task(meta.yaml / shapes.yaml / oracle_pytorch.py)
- ✓ 2 个 runner(A 加 `register()` sys.modules 注入 hook 时改;不动 sys.path)
- ✓ pyproject(B own)
- ✓ tests(各自新增,不重叠)
- ✓ docs/triage_backlog 本 plan(A own)
- ✓ frozen kernel.py(双方禁改)
- ✓ frozen `triton_ascend_bench/frozen/snapshots/*`(双方禁改,见 §6.2)
- ✓ tolerance_policy(双方默认不动)

未列出的目录默认不在本次并行修复范围内(包括 `docs/diagrams/` SVG / `docs/release/` / 其它 `triton_ascend_bench/*.py` 模块)。如双方在执行时遇到需要触及此外文件,先在协调点(§7)沟通。

## 7. 协调点

### 7.1 同步点表

| 同步点 | 时机 | A 操作 | B 操作 |
|---|---|---|---|
| **S0**: fresh resolver 完成 | T+0(已达成) | `results/resolve_all.json` 2026-05-25 17:31;§3.2 + §3.4 已 update;本 plan commit + push | 拉 doc,看 §3.4 真实 BLOCKED list,起 B1 第一批 task |
| **S1**: A1 shim 完成 | T+1.5d → T+2d | push `feat(shim): ...`;通知 B | 拉新代码;若 B1 拟修的 task 与 §3.4.2 list 重合(例:L1_14 / L1_22 / L2_01 等 P3 task)则跳过,转下一批 |
| **S2**: A2 + A3 完成 | T+2d → T+2.5d | push A2 commit + A3 commit(A3 含 A 自己落的 L2_10 meta.yaml/shapes.yaml 全部改动,无 B 移交);通知 B | 拉新代码继续 B1 walk-through;**不**接 L2_10 任何文件 |
| **S3**: B 50% checkpoint | T+2d | 拉看 B 改的 task 是否影响 A2 / A3 verification(应不影响) | B1 累计 ~45 task,push 当前批次 commit |
| **S4**: B 主体收尾 | T+3.5d | 跟踪 B 余下批次 | push 剩余 B1 / B2 / B3 commit |
| **S5**: 总验证 + acceptance | T+4d | 跑 fresh resolver 终态;pytest 全套;release boundary;§10 验收 | 同上 |

### 7.2 同步协议(每次 push 后)

- 双方 push 前 **拉一次** main(`git fetch && git rebase origin/feat/cost-model-fill` 或对应分支)。
- Push 后在 commit message 或外部 channel(无 channel 则只靠 git log)显式标记完成的同步点(commit message 不带 round/V 等过程痕迹,但可写 "complete shim layer for P3 placeholder drift"):
- 任何一方在同步点之前若发现 ownership 冲突或拿不准的边界,**先停手**,等对方先 push 再判断。

## 8. 冲突解决预案

### 8.1 预期可能冲突

| 冲突场景 | 处理 |
|---|---|
| 双方同时改 `pyproject.toml` | A 不会改 pyproject;若 A 因 shim 需新增依赖(预估不需要,shim 只依赖标准 triton),先沟通 B 让其暂停 pyproject 写入 |
| 双方同时改 `_pa_oracle_check_runner.py` / `_pa_subproc_runner.py` | B 不会改这两个 runner;A 改完 push 后 B 才动其它 |
| Merge conflict(同一文件不同改动) | 谁先 push 谁 win;后者 `git pull --rebase`,resolve 冲突;若 resolve 不掉(例:逻辑互斥),后者 reset 改动重新设计 |
| Task 双方都改 | 不应发生(§6 已严格分配);若发生,**A own 优先**,B revert |
| fresh resolver 后发现新 task class(双方 ownership 都不覆盖) | 暂归 B(per-task / 机械化预设)。若属 systemic(类 A1 shim),归 A 评估后决定 |

### 8.2 不允许的操作

- B **绝对不**编辑 `triton_ascend_bench/upstream_shims/` 任何文件。
- A **绝对不**编辑 `pyproject.toml`。
- 双方**都不**编辑 `tasks/level*/<task_id>/kernel.py`(frozen)。
- 双方**都不**编辑 `triton_ascend_bench/frozen/snapshots/*`(frozen upstream snapshot;re-freeze 须走独立 PR via `scripts/phase_a_freeze.py`)。
- 双方**都不**修改 `triton_ascend_bench/tolerance_policy.yaml` 不商量。
- 双方**都不**push 到非自己分支(本协议默认双方都在同一 working branch,后续若 split branch 须双方协商分支名)。

## 9. 时间表 + 估时(fresh resolver 校准)

**B 工作量重估**(基于 §3.4 fresh data 实测 93 task):

- 71 shapes-incomplete × 10-15 min/task = **12-18 工时**
- 22 upstream-not-invocable-here × 5-30 min/task(部分被 A1 shim 解,B 跳过即可) = **2-11 工时**
- **B own 合计 14-29 工时 ≈ 2-4 工作日**(B 单干;A 并行不缩短 B 的关键路径)

**A 工作量**(不变):A1 shim ~1 工作日 + A2 oracle ~半天 + A3 L2_10 静态分析 ~2 小时 = **~1.5-2 工作日**。

并行最长关键路径 = max(A 路径, B 路径) = **B 路径 ≈ 2-4 工作日**(A 收尾后帮 B 收 review-fix loop 可微缩,但本质 bottleneck 在 B)。

| 阶段 | T+ | session A 工作 | session B 工作 | 累计 |
|---|---|---|---|---|
| **fresh resolver(已完成)** | T+0 | `results/resolve_all.json` 2026-05-25 17:31;§3.2 + §3.4 已 update | 待命读 doc | T+0 |
| **kickoff** | T+0 → T+0.5d | A1 shim 设计草稿(`KernelProxy` / autotune fallback);开第一个 commit 草稿 | B1 第一批简单 elementwise 5-7 个 task(§B1.2 推荐顺序) | 0.5d |
| **核心实施(平行)** | T+0.5d → T+1.5d | A1 shim 实施 + unit test;复跑 P3 task verify | B1 持续;一天 ~25-30 task | 1.5d |
| **A1 review-fix + A2 起手** | T+1.5d → T+2d | A1 `humanize:ask-codex` review-fix 闭环到 ALL PASS,push;A2 起 oracle-mismatch | B1 累计 ~50% (~45 task);开始 B2 / B3 识别 | 2d (**B 50% checkpoint**) |
| **A2 / A3 完成** | T+2d → T+2.5d | A2 commit;A3 L2_10 静态分析 + 自己落 meta/shapes(dev-fix)或 phase_b 归档(GPU-only),自己 commit;**不**移交 B | B1 收尾;B2 缺包处理 | 2.5d |
| **B 主体收尾 + 协同验证** | T+2.5d → T+3.5d | A 拉 B 各 push;复跑 §A4.1 验证 | B1 余下复杂 op + B3 归档(GPU-only) | 3.5d |
| **总验证 + acceptance** | T+3.5d → T+4d | 跑 fresh resolver 终态;pytest 全套;release boundary;§10 验收 checklist | 同左,余下 commit 全 push | **T+4d (B done)** |

> **关键 checkpoint**:T+2 → B 完成 ~50% (~45 task);T+4 → B 全部 done (93 task) + A 全部 done;终态 fresh resolver 0 BLOCKED 或剩 < 5 真必须 phase B 归档项。
> **缓冲**:若 T+4 仍剩,加 T+4.5d / T+5d 收尾;不允许压缩 walk-through 质量。

## 10. Acceptance Criteria

### 10.1 必满足项

- [ ] **总 BLOCKED 从 fresh 起点 96 收敛到 0**(全 RESOLVED + GPU sign-off 待 phase B)**或** 剩余真必须 phase B 的归档项 **< 5**,且全部已 meta.yaml 加 `phase_b_capture_required: true`(B3 归档完整)
- [ ] **session B 93 个 own task** 全部单独跑 `python3 scripts/resolve_upstream_entrypoints.py --task <tid>` 输出 `RESOLVED`(等价 PASS)**或**合规 `SKIPPED_BY_DESIGN` 含 `phase_a_prime.skip_invocation: true` + `phase_b_capture_required: true` + `phase_b_capture.status: PENDING / DONE`(meta.yaml schema 见 §B3.2)
- [ ] **session A 3 个 own task** (`L1_01_bias` / `L1_09_quantize_k_cache_fast` / `L2_10_w8a8_block_int8_matmul`) 同上判定:PASS 或合规 SKIPPED_BY_DESIGN
- [ ] session A 3 commit(A1 shim / A2 oracle-mismatch / A3 L2_10 处理)+ session B 多个 commit 全部 push 上 main
- [ ] `pytest tests/ -v` PASS,**0 failed 0 skipped**
- [ ] `ruff check` PASS;`ruff format --check` PASS。**扫描 paths(Python only)**:`scripts/automation/` + `triton_ascend_bench/` + `tests/` + `scripts/_pa_oracle_check_runner.py` + `scripts/_pa_subproc_runner.py`(A1 修改的 runner 必须覆盖,与 §10.3 `RUFF_PATHS` 数组对齐;`tasks/level*/` 是 yaml 不进 ruff 扫,见下一条 Chinese / 过程痕迹扫)
- [ ] `git grep -nP '[\x{4e00}-\x{9fa5}]' -- '*.py' 'tasks/level1/**/*.yaml' 'tasks/level2/**/*.yaml' 'tasks/level3/**/*.yaml'` 0 hit(除 docs 例外)。**扫描 paths 必须含**:`scripts/_pa_oracle_check_runner.py` + `scripts/_pa_subproc_runner.py`(A1 改的) + `tasks/level1/` + `tasks/level2/` + `tasks/level3/`(B 主写的 meta.yaml / shapes.yaml)
- [ ] 过程痕迹检查(`codex / round-N / [RHDFP][0-9]+ / V[1-9]`)在**代码 / commit message / 任务 yaml / 测试代码**上 0 hit(除合法例外)。**扫描 pathspec 必须含**:`'*.py'`(覆盖 `scripts/_pa_oracle_check_runner.py` + `scripts/_pa_subproc_runner.py` 等 A1 修改的 runner) + `'*.md'` + `'tasks/level1/**/*.yaml'` + `'tasks/level2/**/*.yaml'` + `'tasks/level3/**/*.yaml'`(B 主写的 meta.yaml / shapes.yaml)。本 plan doc(§0 已声明)在 operator-facing 文档语境下使用 `codex / Mx / Px / Ax / Bx` 作技术术语**是允许的**,不视作过程痕迹;`git grep` 命令必要时把本 doc 路径加 negative pathspec 排除以避开 false positive,但仍保留对全部代码文件的扫描。
- [ ] bench 仓 0 hardcode 外部项目路径(检查 `triton_ascend_bench/` 中没有 `/data1/code/agentic-rl/siiRL` / `slime` / `siirl-agentic` 等)
- [ ] `pytest tests/unit_tests/test_release_boundary.py -v` 9 passed(M0 release boundary 不破)
- [ ] `git archive HEAD | tar t | grep -E '(scripts/automation/|tab_automation/|docs/plans/|docs/release/internal-|docs/triage_backlog/|test_lib_|test_.*automation|test_fix_orchestrator|test_tier1_reverify)'` 0 hit(release tarball 不含 INTERNAL)

### 10.2 软目标(非阻塞)

- [ ] codex review-fix loop ALL PASS(`humanize:ask-codex` 在 A1 / A2 / B 批量 commit 上各跑过)
- [x] 本 plan doc 在 push 前补完 §3.2 表 B 真实数字(已完成,fresh data 2026-05-25 17:31)
- [ ] 本 plan doc 在最终 push 前补完 §A3.5 L2_10 静态分析结论(等 session A 跑 A3 时填充)
- [ ] 各 commit message ≤ 50 word,**单**主题(不混合 A 和 B 改动)

### 10.3 自动化预检 snippet

> **修订(codex r1 MF10)**:旧版用 `| tail -10` / `| wc -l` 把 pytest / grep 退出码吞掉,**失败也不退出**;改成显式 `set -euo pipefail` + `PIPESTATUS[0]` 捕获真退出码,每个阶段失败立刻 `exit <code>`。

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /data1/code/agentic-rl/kernel/triton-ascend-bench

echo "=== pytest ==="
python3 -m pytest tests/ -q --no-cov 2>&1 | tee /tmp/pytest.log | tail -10
PYTEST_RC=${PIPESTATUS[0]}
if [[ "$PYTEST_RC" -ne 0 ]]; then
    echo "[FAIL] pytest rc=$PYTEST_RC (see /tmp/pytest.log)"
    exit 1
fi
echo "[OK] pytest"

echo "=== ruff check ==="
RUFF_PATHS=(
    scripts/automation/
    triton_ascend_bench/
    tests/
    scripts/_pa_oracle_check_runner.py
    scripts/_pa_subproc_runner.py
)
ruff check "${RUFF_PATHS[@]}" || { echo "[FAIL] ruff check"; exit 2; }
echo "[OK] ruff check"

echo "=== ruff format --check ==="
ruff format --check "${RUFF_PATHS[@]}" || { echo "[FAIL] ruff format"; exit 2; }
echo "[OK] ruff format"

echo "=== 0 Chinese in code / task yaml ==="
CHINESE_PATHS=(
    scripts/automation/
    triton_ascend_bench/
    tests/
    scripts/_pa_oracle_check_runner.py
    scripts/_pa_subproc_runner.py
    tasks/level1
    tasks/level2
    tasks/level3
)
if grep -rIP '[\x{4e00}-\x{9fa5}]' "${CHINESE_PATHS[@]}" 2>/dev/null \
    | grep -v __pycache__ | grep -v '.egg-info' ; then
    echo "[FAIL] Chinese chars in code / task yaml"
    exit 3
fi
echo "[OK] 0 Chinese in code / task yaml"

echo "=== release boundary ==="
python3 -m pytest tests/unit_tests/test_release_boundary.py -q || { echo "[FAIL] boundary"; exit 4; }
echo "[OK] boundary"

echo "=== archive does not contain INTERNAL ==="
INTERNAL_HITS=$(git archive HEAD 2>/dev/null | tar t \
    | grep -cE '(scripts/automation/|tab_automation/|docs/plans/|docs/release/internal-|docs/triage_backlog/|test_lib_|test_.*automation|test_fix_orchestrator|test_tier1_reverify)' \
    || true)
if [[ "$INTERNAL_HITS" -ne 0 ]]; then
    echo "[FAIL] release tarball leaks $INTERNAL_HITS INTERNAL paths"
    exit 5
fi
echo "[OK] release tarball clean"

echo "=== ALL PRE-CHECKS PASS ==="
```

期望:**整脚本退出码 0**(任何一步失败立即 non-zero exit;pytest 0 fail / 0 skip,ruff 0 error,0 Chinese,9 release-boundary tests passed,0 INTERNAL paths in tarball)。

---

> **本 doc 落点**:`docs/triage_backlog/2026-05-25-parallel-execution-plan.md`,被 `.gitattributes` 的 `docs/triage_backlog/` 通配 `export-ignore` 覆盖,不进 release tarball。本 doc 仅由 session A 写;session B 只读。fresh resolver(2026-05-25 17:31)已 update §3.2 表 B 真实计数(BLOCKED = 96)和 §3.4 完整 B own 93 task list;§A3.5 L2_10 静态分析结论待 session A 跑 A3 后再 push 增量 update。
