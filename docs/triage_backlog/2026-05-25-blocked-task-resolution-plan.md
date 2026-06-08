---
date: 2026-05-25
status: triage plan for 40 BLOCKED tasks (L1_01 -- L2_03)
audience: bench team manual repair operators
based-on: resolve_all.json at repo root (100 verdicts, generated_at 2026-05-24T14:18:51Z)
scope: docs/triage_backlog/ is export-ignored (INTERNAL operator manual)
---

# BLOCKED Task 修复方案 — L1_01 -- L2_03 (2026-05-25)

## 0. 阅读须知

- 本文档是 **bench 团队内部操作手册**,目的是把 `resolve_all.json` 当前 BLOCKED 的 40 个 task(L1_01 -- L2_03 范围,跳过 3 个已 PASS)推到 RESOLVED 或 SKIPPED_BY_DESIGN。
- 文档落在 `docs/triage_backlog/`,被 `.gitattributes` `export-ignore` 通配覆盖,**不**进入 release tarball。
- 所有 task 数据(verdict_class / reason / 缺失参数 / 失败模式)源自仓库根目录的 `resolve_all.json`(provenance `generated_at: 2026-05-24T14:18:51Z`,`tool_version: 0.1.0`)。注意:`results/resolve_all.json` 是 resolver 的 canonical cache(`scripts/resolve_upstream_entrypoints.py` 里的 `CANONICAL_RESOLVE_CACHE`),而根目录的副本是本次评估的 snapshot 输入;两者可能因 cache 刷新时序而不同步。修复前先确认两者一致(或选择更 hardened anchor 那份作准)。
- 文档以中文叙述,**所有 yaml / python / 命令片段必须保持英文**(repo-wide convention)。
- 数字诚实:**只覆盖 40 个 task**(prompt 描述里写 "43" 是笔误,实际范围是 L1_01..L1_40 + L2_01..L2_03 减去 3 个 PASS = 40)。

## 1. Executive Summary

### 1.1 40 task 的分类计数([confirmed] 源自 `resolve_all.json`)

| 分类 | 40 范围内 | 全 100 task | 修法 |
|------|----------|------------|------|
| `shapes-incomplete` | **22** | 56 | 给 `shapes.yaml` 补缺失参数(逐个改) |
| `upstream-not-invocable-here` | **16** | 38 | `defer-to-golden`(批量加 `phase_a_prime.skip_invocation`) |
| `oracle-mismatch` | **2** | 2 | 三选一精修(tolerance bump / oracle 改写 / defer) |
| `smoke-fail` | 0 | 1 | 不在 40 范围(`L2_10_w8a8_block_int8_matmul`,未来处理) |
| `PASS`(跳过) | 3 | 3 | `L1_12_rms_norm` / `L1_16_silu_and_mul` / `L1_26_merge_state` |
| **合计** | **40 待修 + 3 PASS = 43 in user list** | 100 | — |

### 1.2 优先级 + 工估

按"批量收益高 → 单个收益高"排序:

| Step | 范围 | task 数 | 单 task 工时 | 累计 |
|------|------|--------|-------------|------|
| 1 | `upstream-not-invocable-here` 全部 defer | 16 | ~10 min | ~3 h(可半天完成) |
| 2 | `shapes-incomplete` 简单 elementwise(L1_15/L1_17/L1_18/L1_20/L1_21/L1_25/L1_27) | 7 | ~10 min | +1.5 h |
| 3 | `shapes-incomplete` paged-attn / KV cache(L1_02/L1_05/L1_06/L1_07/L1_08/L1_30) | 6 | ~20 min | +2 h |
| 4 | `shapes-incomplete` MoE / EP(L1_35/L1_36/L1_37) | 3 | ~25 min | +1.5 h |
| 5 | `shapes-incomplete` quant / softmax(L1_10/L1_11) | 2 | ~15 min | +30 min |
| 6 | `shapes-incomplete` flashinfer / FlashMLA(L1_32/L1_33) | 2 | ~15 min | +30 min |
| 7 | `shapes-incomplete` SSD / mamba(L1_39/L2_02) | 2 | ~25 min | +1 h |
| 8 | `oracle-mismatch`(L1_01 / L1_09)精修 | 2 | ~45 min | +1.5 h |
| **总计** | | **40** | | **~11 h ≈ 1.5 工作日**(单人,无 GPU 验证) |

> **说明**:工估**仅含开发机侧改 yaml / vendored / oracle_pytorch 的工时**,不含 GPU 机侧跑 `phase_a_prime` 验证的工时(需要等 GPU 队列,通常 +1-2 天)。也未包含可能的 multi-round iteration(改完发现还有遗漏)。保守估总周期 **3-4 工作日**(含验证回路)。

### 1.3 SETTLED 约束(memory `TAB baseline no-subset-escape`)

修复必须遵守:
- **禁** "剔精度难题改分母"。**全 100 task 都进 release 分母**,不允许把 BLOCKED 偷偷 `release_scope.excluded` 走捷径。
- `release_scope.excluded` 字段**本次不应使用**。每个 task 的 verdict 类应该走对应的合法路径:
  - `shapes-incomplete` → `patch-shapes`(改 shapes.yaml + meta.yaml)
  - `oracle-mismatch` → 三选一:`tolerance-bump`(改 meta.yaml.tolerance_class) / `rewrite-primitives`(改 oracle_pytorch.py) / `defer-to-upstream-oracle`(转 phase_b_capture)
  - `upstream-not-invocable-here` → `defer-to-golden`(写 phase_a_prime.skip_invocation + phase_b_capture + phase_b_capture_required)
- golden 双路径:`SKIPPED_BY_DESIGN` 状态必须配 `phase_b_capture_required: true`,确保最终 release gate 仍然依赖一个**真实捕获的 GPU golden**(不能空挂)。

### 1.4 不可改文件

| 文件 | 原因 |
|------|------|
| `tasks/level*/<task_id>/kernel.py` | **frozen upstream** by `scripts/phase_a_freeze.py`。文件头有 SPDX header(Apache-2.0)+ vendored file SHA-256。改动破坏 license attribution + 偏离 bench"验证 frozen upstream"的根本目标。 |
| `triton_ascend_bench/frozen/snapshots/<task_id>_<kernel>_<sha>.py` | 同上,这是 frozen 副本的源(`phase_a_freeze.py` 校对一致性)。 |
| `scripts/resolve_upstream_entrypoints.py` 及其依赖 | 是评估器本身;不可为绕过 BLOCKED 而放宽 anchor。 |
| `triton_ascend_bench/tolerance_policy.yaml` | 全局 tolerance class 表;改动影响所有 task,不是 per-task 修法。 |

### 1.5 可改文件

| 文件 | 用途 |
|------|------|
| `tasks/level*/<task_id>/meta.yaml` | task 元数据(`oracle.gpu_triton` / `param_constraints` / 新增 `phase_a_prime` / `phase_b_capture` / `phase_b_capture_required`)。 |
| `tasks/level*/<task_id>/shapes.yaml` | 测试形状 + `inputs:` 段(补缺失 param 的入口)。 |
| `tasks/level*/<task_id>/oracle_pytorch.py` | PyTorch reference 实现(`oracle-mismatch` 修法 2 改这里)。 |

---

## 2. 工作环境前置

### 2.1 路径锚点

```
REPO              = /data1/code/agentic-rl/kernel/triton-ascend-bench
TASKS_ROOT        = $REPO/tasks
VENDORED          = $REPO/triton_ascend_bench/frozen/snapshots
SPECS             = $REPO/specs/task_api_contract.md  # schema 权威
RESOLVE_CACHE     = $REPO/results/resolve_all.json    # verdict 输入
RESOLVER          = $REPO/scripts/resolve_upstream_entrypoints.py
```

### 2.2 单 task 验证命令(本机 CPU/TRITON_INTERPRET=1 跑)

```bash
cd /data1/code/agentic-rl/kernel/triton-ascend-bench

# 单 task 试跑(dry-run,不写 meta.yaml)
python3 scripts/resolve_upstream_entrypoints.py --task L1_18_fused_softcap

# 通过后写 oracle.gpu_triton 到 meta.yaml(只 shapes-incomplete 类需要)
python3 scripts/resolve_upstream_entrypoints.py --task L1_18_fused_softcap --apply
```

### 2.3 批量回归验证(改完 N 个 task 后)

```bash
# 备份基线(动手前先跑)
cp resolve_all.json /tmp/resolve_baseline_2026-05-24.json
cp results/resolve_all.json /tmp/resolve_canonical_baseline.json

# 全 100 task 重跑 resolver,刷新 canonical cache(results/resolve_all.json)
python3 scripts/resolve_upstream_entrypoints.py --refresh-cache 2>&1 | tee /tmp/resolve_new.log

# 与基线比较(基线 = 100 task: 3 PASS + 56 shapes-incomplete + 38 upstream-not-invocable + 2 oracle-mismatch + 1 smoke-fail
# 期望(本计划覆盖 40 范围全部修完)= 3 PASS + 22+34=34 shapes-incomplete + 38-16=22 upstream-not-invocable + 0 oracle-mismatch + 1 smoke-fail
# 即 RESOLVED 数从 3 上升至 3 + 22(shapes 修复) + 2(oracle-mismatch 修复) = 27
# SKIPPED_BY_DESIGN 数从 0 上升至 16(其中可能含 L1_38 走 patch-shapes 而非 defer))
```

### 2.4 Schema 速查

- `phase_a_prime.{skip_invocation,status,reason}` — `specs/task_api_contract.md §6a.1-6a.2`
- `phase_b_capture.{status,upstream_callable.{module,function},reason,captured_at}` — `§6a.3`
- `phase_b_capture_required` — `§6a.4`
- `release_scope.{excluded,reason,excluded_at,excluded_by}` — `§6a.5`(本计划不使用)
- 跨字段 invariants(写入会被 Layer-5 validator 拒)— `§6a.6`

---

## 3. 按 verdict_class 分组的修法 recipe

### 3.1 `shapes-incomplete`(22 个,recipe = `patch-shapes`)

#### 3.1.1 通用 4 步法

每个 task 按以下流程:

1. **从 `reason` 里抓"缺哪个参数 / 哪个入口"**。
   `resolve_all.json` 的 reason 包含明确指引:
   ```
   ShapesIncompleteError: BLOCKED(shapes-incomplete: missing param 'X'
     for entrypoint 'Y' -- shapes.yaml / build_inputs provides no value
     and the parameter has no default. ... Fix: add 'X' to the task's
     shapes.yaml inputs.)
   ```
   提取 `(entrypoint Y, missing param X)`。
2. **打开 `kernel.py`,定位入口 `Y`,读出函数签名**,推断参数 `X` 的:
   - **dtype**(通常 tensor → `float32` / `bfloat16` / `int64`,scalar → 直接 `scalar: <value>`)
   - **shape**(从 docstring / 同类参数推断;比如 `gateup_output` 与 `x` 同 shape 但 D/2)
3. **在 `shapes.yaml` 的 `inputs:` 段加 `X:` 块**,并同步更新 `meta.yaml` 的 `param_constraints.dtypes`(如需新增维度变量则也加 `dims:`)。
4. **写 `meta.yaml.oracle.gpu_triton` 为 `kernel.py::Y`**(从 `placeholder` 改为真实入口名),然后跑 `resolve_upstream_entrypoints.py --task <id>` 验证。

#### 3.1.2 子类 A:简单 elementwise(7 个,~10 min 每个)

| Task | Missing | Entrypoint | 估时 |
|------|---------|-----------|------|
| `L1_15_act_and_mul` | `gateup_output` | `act_and_mul_triton` | 10 min |
| `L1_17_gelu_and_mul` | `hidden_states` | `gelu_and_mul_triton` | 10 min |
| `L1_18_fused_softcap` | `full_logits` | `fused_softcap` | **walk-through 见 §3.1.7** |
| `L1_20_log_softmax` | (重定向到 `_log_softmax` 内的真入口) | — | 15 min(reason 显示 resolver 误试了 `rms_norm` / `bmm_batch_invariant`,需手动定位 `kernel.py` 真实 `log_softmax` 入口) |
| `L1_21_gumbel_sample` | `logits` | `gumbel_sample` | 10 min |
| `L1_25_merge_attn_states` | `output` | `merge_attn_states` | 10 min |
| `L1_27_apply_grammar_bitmask` | `logits_ptr` | `_apply_grammar_bitmask_kernel`(注意:`_ptr` 后缀说明是 `@triton.jit` 内核而非 host wrapper — 需要找它的 host launcher,而不是直接调 jit 函数) | 15 min |

> **共性陷阱**:resolver 报 missing `xxx_ptr` 时(如 L1_27 的 `logits_ptr`),通常是 resolver 误把 `@triton.jit` kernel 当成 host entrypoint。真正修法是去 `kernel.py` 里找包裹 `_apply_grammar_bitmask_kernel[(...)]` 的那个 host 函数,把 `meta.yaml.oracle.gpu_triton` 指过去。

#### 3.1.3 子类 B:paged-attention / KV cache(6 个,~20 min 每个)

| Task | Missing | Entrypoint | 备注 |
|------|---------|-----------|------|
| `L1_02_compute_slot_mappings` | `max_num_tokens` | `_compute_slot_mappings_kernel` | scalar int,从 shapes knobs `T` 或 `B*N` 推 |
| `L1_05_gather_block_tables` | `batch_idx_to_req_idx` | `_gather_block_tables_kernel` | int64 tensor [B] |
| `L1_06_reshape_and_cache_kernel_flash` | `kv_cache_dtype` | `triton_reshape_and_cache_flash` | scalar string,常用 `"auto"` 或 `"fp8"` |
| `L1_07_dequantize_k_cache_fast` | `quant_k_cache` | `_dequantize_k_cache_fast` | int8 tensor,shape 同 nope_q 但用 int8 dtype |
| `L1_08_dequantize_k_cache_paged` | `quant_k_cache` | `dequantize_k_cache_paged` | 同上,加 page_table 关联 |
| `L1_30_act_quant` | `page_table_1` | `get_valid_kv_indices` | int64 tensor [E, num_pages] |

> **共性陷阱**:这类 task 的 missing param 通常是 cache state(int8 量化缓存 / page table)。`dtype` 不是 `float32`,务必从 `kernel.py` 的 `tl.load(ptr).to(...)` 推 int8 或 int64,而不是默认 `randn` float。

#### 3.1.4 子类 C:MoE / Expert Parallel(3 个,~25 min 每个)

| Task | Missing | Entrypoint | 备注 |
|------|---------|-----------|------|
| `L1_35_compute_src2dst_triton` | `topk_ids` | `cutlass_w4_run_moe_ep_preproess` | int64 [B, top_k],resolver 误试多个 entrypoint,需手动选 `compute_src2dst_triton` 真入口 |
| `L1_36_compute_seg_indptr_triton` | `num_local_experts` | `moe_ep_deepgemm_preprocess` | scalar int |
| `L1_37_compute_problem_sizes_w4a8` | `problem_sizes1` | `compute_problem_sizes_w4a8` | int32 [E, 3] |

#### 3.1.5 子类 D:Quant / GEMM(2 个,~15 min 每个)

| Task | Missing | Entrypoint | 备注 |
|------|---------|-----------|------|
| `L1_10_per_token_group_quant_int8` | `A`(resolver 误试 `w8a8_block_int8_matmul`) | `_per_token_group_quant_int8` | 真入口在 `kernel.py` 里 — 需手动指明 |
| `L1_11_awq_dequantize` | `input`(resolver 试了 `awq_gemm_triton`) | `awq_dequantize` 或 `_awq_dequantize_kernel` host wrapper | 同 |

> **共性陷阱**:resolver 自动枚举 candidate 时,把同文件里的 GEMM/matmul 入口也作为试点。真正的修法是直接在 `meta.yaml.oracle.gpu_triton` 写明确入口(不让 resolver 瞎枚举),例如 `kernel.py::per_token_group_quant_int8`。

#### 3.1.6 子类 E:FlashInfer / FlashMLA / SSD(4 个,~15-25 min)

| Task | Missing | Entrypoint | 备注 |
|------|---------|-----------|------|
| `L1_32_create_flashinfer_kv_indices_triton` | `extend_seq_lens` | `seqlens_expand_triton`(resolver 试的)/ `create_flashinfer_kv_indices_triton`(真入口) | 真入口在 `kernel.py`,直接指定 |
| `L1_33_create_flashmla_kv_indices_triton` | 同上 | 同上,FlashMLA 变体 | 同 |
| `L1_39_state_passing_fwd` | `last_chunk_indices` | `_state_passing_fwd` | int32 [B, num_chunks] |
| `L2_02_chunk_scan_fwd` | `cu_chunk_seqlens` | `_chunk_scan_fwd` | int32 cumulative seq lens,从 shape `B*L` 拆 chunk |

#### 3.1.7 完整 walk-through:`L1_18_fused_softcap`

**当前状态**([confirmed] 从 `task_data_43.json`):
- `verdict_class`: `shapes-incomplete`
- `upstream_path`: `sglang/srt/layers/logits_processor.py`
- `oracle.pytorch`: `oracle_pytorch.py::run`
- `oracle.gpu_triton`: `kernel.py::placeholder`(**这是问题** — 没指真实入口)
- `param_constraints.dtypes`: `['x']`(对应 shape 里的 `x` 变量)
- `param_constraints.scalars`: `{'cap': '0 < cap <= 1e3'}`
- `tolerance_class`: `elementwise`

**reason 摘录**:
```
BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete):
  tried ['fused_softcap']; fused_softcap: shapes-incomplete
  [missing param 'full_logits' for entrypoint 'fused_softcap'
   -- Fix: add 'full_logits' to the task's shapes.yaml inputs.]
```

**kernel.py 真实签名**(line 1154):
```python
def fused_softcap(full_logits, final_logit_softcapping):
    ...  # host wrapper for fused_softcap_kernel @triton.jit
```

**当前 `shapes.yaml.inputs:`** (line 134-142):
```yaml
inputs:
  x:
    dims:
      - B
      - D
    dtype: float32
    dist: randn
  cap:
    scalar: 30.0
```

**修法 — 改 `shapes.yaml`**:

```diff
 inputs:
-  x:
+  full_logits:
     dims:
       - B
       - D
     dtype: float32
     dist: randn
-  cap:
+  final_logit_softcapping:
     scalar: 30.0
```

**修法 — 改 `meta.yaml`**(同步参数名 + 真入口):

```diff
 oracle:
   pytorch: oracle_pytorch.py::run
-  gpu_triton: kernel.py::placeholder
+  gpu_triton: kernel.py::fused_softcap

 param_constraints:
   scaffold: false
   dims:
     - D >= 1
     - B >= 1
   dtypes:
-    x: float32
+    full_logits: float32
   dists:
-    x:
+    full_logits:
       kind: randn
   scalars:
-    cap: 0 < cap <= 1e3
+    final_logit_softcapping: 0 < final_logit_softcapping <= 1e3
```

**也要改 `oracle_pytorch.py`**(参数名要对齐):查看现有 `run(...)` 签名,把 `x` / `cap` 参数名改为 `full_logits` / `final_logit_softcapping`。这是因为 runner 用 by-name binding,oracle 和 candidate 共享同一个 inputs dict。

**验证命令**:

```bash
cd /data1/code/agentic-rl/kernel/triton-ascend-bench
python3 scripts/resolve_upstream_entrypoints.py --task L1_18_fused_softcap
# 期望输出末尾:
#   L1_18_fused_softcap: RESOLVED (kernel.py::fused_softcap)
# 然后落盘:
python3 scripts/resolve_upstream_entrypoints.py --task L1_18_fused_softcap --apply
```

**期望 verdict 变化**:`BLOCKED(shapes-incomplete)` → `RESOLVED (kernel.py::fused_softcap)`。

---

### 3.2 `upstream-not-invocable-here`(16 个,recipe = `defer-to-golden`)

#### 3.2.1 通用 5 步法

1. **从 `reason` 抓首条异常**(`ImportError` / `ModuleNotFoundError` / `AttributeError` / `AssertionError` / `TypeError`)。
2. **判断"能否在本机/本验证环境绕过"**。决策矩阵:
   - `ModuleNotFoundError: cachetools / cpuinfo / ...` — 可以装,但这是上游不希望的 invasive change。倾向 **defer**,因为本 task 已经被 frozen,不应通过外部 dep 改变行为。
   - `ImportError: attempted relative import with no known parent package` — `kernel.py` 内有相对 import(`from ..helpers import ...`),作为独立文件无法解析。**defer**。
   - `AttributeError: module 'triton' has no attribute 'next_power_of_2'` — frozen kernel 用了上游 vllm 的 triton stub;本机 triton 版本不同。**defer**(不改 kernel.py)。
   - `TypeError: 'function' object is not subscriptable`(L1_22)— `from vllm.triton_utils import tl, triton` 在 standalone 加载时被 stub 化。**defer**。
   - `AssertionError: H must be divisible by 8`(L1_38)— shape 维度违反 kernel runtime 断言。**两种选项**:(a) 改 `shapes.yaml` 让 H 都满足 % 8 == 0(`patch-shapes` 变体);(b) 若现有 shapes 已合规仍 fail,则说明 candidate 与 oracle 的形状协议不一致,**defer**。优先 (a),失败再 defer。
3. **决定 `defer-to-golden`** ⇒ 改 `meta.yaml`,新增 3 个 block:
   - `phase_a_prime.{skip_invocation: true, status: SKIPPED_BY_DESIGN, reason: "..."}`
   - `phase_b_capture.{status: PENDING, upstream_callable: {module: "...", function: "..."}, reason: "..."}`
   - `phase_b_capture_required: true`
4. **`upstream_callable` 必须 canonical**(`§6a.3` 强校验):
   - `module`:`vllm.v1.worker.gpu.sample.min_p` 风格(纯点分,无斜杠 / 路径)
   - `function`:单 identifier(无点 / 空格,过 `str.isidentifier()`)
5. **跑 `resolve_upstream_entrypoints.py --task <id>` 验证**。verdict 不会变成 `RESOLVED`(defer 路径不写 `oracle.gpu_triton`),但 phase_a_prime 走到时会 `SKIPPED_BY_DESIGN`,eligibility gate 会接受。

#### 3.2.2 失败模式分桶

| 桶 | task 数 | 桶内 task ID | 共性 |
|----|--------|--------------|------|
| `cachetools_missing` | 4 | L1_19/L1_23/L1_24/L1_29 | 全是 vllm sample stack 的 task。`cachetools` 是 vllm 的 indirect runtime dep,本验证机不装。 |
| `relative_import` | 5 | L1_13/L1_28/L1_31/L1_40/L2_03 | frozen `kernel.py` 有 `from .helpers import ...` 类相对 import。 |
| `triton_attr_missing` | 2 | L1_14/L2_01 | frozen kernel 用了 vllm 自带 triton stub 里的 `next_power_of_2` / `tl.float32` 等本地版本不具备的 attr。 |
| `triton_function_not_subscript` | 1 | L1_22 | `from vllm.triton_utils import triton` 在 standalone 加载时 `triton` 是 stub function。 |
| `cpuinfo_missing` | 1 | L1_04 | `vllm` runtime 启动顺序触发 `import cpuinfo`。 |
| `assertion` | 3 | L1_03/L1_34/L1_38 | runtime assert 失败;L1_38 可能是 `patch-shapes` 子情况(见 3.2.4)。 |

#### 3.2.3 批量脚本(适用于 cachetools_missing / relative_import / triton_attr_missing / triton_function_not_subscript / cpuinfo_missing = 13 个 task)

```python
# /tmp/batch_defer_to_golden.py
# Apply phase_a_prime.skip_invocation + phase_b_capture + phase_b_capture_required
# to 13 tasks. Operator MUST review the upstream_callable.module / .function
# fields after running (script makes a best-guess from kernel.py headers).

import datetime
import re
import sys
from pathlib import Path

import yaml

REPO = Path("/data1/code/agentic-rl/kernel/triton-ascend-bench")

TARGETS = {
    # task_id : (upstream_module, upstream_function, reason_short)
    "L1_04_copy_page_indices":              ("vllm.v1.attention.copy_page_indices",       "copy_page_indices",                          "vllm runtime requires cpuinfo (not installable in verify env)"),
    "L1_13_layer_norm_fwd":                 ("vllm.model_executor.layers.layernorm",      "layer_norm_fwd",                              "kernel.py has relative imports that fail standalone"),
    "L1_14_layer_norm_fwd_1pass":           ("vllm.model_executor.layers.fla.layernorm",  "layer_norm_fwd_1pass",                        "frozen kernel uses upstream triton stub attrs (next_power_of_2) absent in verify env"),
    "L1_19_topk_log_softmax":               ("vllm.v1.sample.ops.topk_topp_sampler",      "topk_log_softmax",                            "vllm sample stack requires cachetools (not installable)"),
    "L1_22_min_p":                          ("vllm.v1.worker.gpu.sample.min_p",           "apply_min_p",                                 "vllm.triton_utils.triton is a stub in standalone; kernel cannot launch"),
    "L1_23_penalties_and_temperature":      ("vllm.v1.worker.gpu.sample.penalties",       "apply_penalties_and_temperature",             "vllm sample stack requires cachetools"),
    "L1_24_rejection_sample":               ("vllm.v1.sample.rejection_sampler",          "rejection_sample",                            "vllm sample stack requires cachetools"),
    "L1_28_pack_bitmatrix":                 ("vllm.v1.worker.gpu.structured_outputs",     "pack_bitmatrix",                              "kernel.py has relative imports that fail standalone"),
    "L1_29_prepare_eagle_inputs":           ("vllm.v1.spec_decode.eagle",                 "prepare_eagle_inputs",                        "vllm spec_decode stack requires cachetools"),
    "L1_31_triton_mrope_forward":           ("vllm.model_executor.layers.rotary_embedding", "triton_mrope_forward",                      "kernel.py has relative imports that fail standalone"),
    "L1_40_chunk_cumsum_fwd":               ("vllm.model_executor.layers.fla.ops.chunk_cumsum", "chunk_cumsum_fwd",                      "kernel.py has relative imports that fail standalone"),
    "L2_01_bmm_chunk_fwd":                  ("vllm.model_executor.layers.mamba.ops.ssd_bmm", "bmm_chunk_fwd",                              "frozen kernel uses upstream triton.language stub attrs absent in verify env"),
    "L2_03_chunk_state_fwd":                ("vllm.model_executor.layers.mamba.ops.ssd_chunk_state", "chunk_state_fwd",                   "kernel.py has relative imports that fail standalone"),
}


def level_dir(tid: str) -> str:
    return f"level{tid.split('_')[0][1:]}"


def patch(tid: str, module: str, function: str, reason: str) -> None:
    meta_path = REPO / "tasks" / level_dir(tid) / tid / "meta.yaml"
    assert meta_path.exists(), meta_path
    meta = yaml.safe_load(meta_path.read_text())

    # Schema invariants -- specs/task_api_contract.md §6a.6
    assert re.match(r"^[A-Za-z_][\w]*(\.[A-Za-z_][\w]*)*$", module), f"bad module {module!r}"
    assert function.isidentifier(), f"bad function {function!r}"

    meta.setdefault("phase_a_prime", {})
    meta["phase_a_prime"]["skip_invocation"] = True
    meta["phase_a_prime"]["status"] = "SKIPPED_BY_DESIGN"
    meta["phase_a_prime"]["reason"] = reason

    meta.setdefault("phase_b_capture", {})
    meta["phase_b_capture"]["status"] = "PENDING"
    meta["phase_b_capture"]["upstream_callable"] = {
        "module": module,
        "function": function,
    }
    meta["phase_b_capture"]["reason"] = (
        f"deferred to offline golden capture: upstream entrypoint "
        f"{module}.{function} requires runtime that this verify host "
        f"cannot provide ({reason})"
    )

    meta["phase_b_capture_required"] = True

    # Write back preserving original key order best-effort
    meta_path.write_text(yaml.dump(meta, sort_keys=False, allow_unicode=False))
    print(f"patched {tid}: defer-to-golden ({module}.{function})")


def main() -> int:
    for tid, (mod, fn, rs) in TARGETS.items():
        try:
            patch(tid, mod, fn, rs)
        except Exception as e:
            print(f"FAIL {tid}: {e}", file=sys.stderr)
            return 1
    print(f"\nDONE: patched {len(TARGETS)} tasks. "
          f"NOW manually review each meta.yaml's "
          f"phase_b_capture.upstream_callable.{{module,function}} "
          f"to confirm the dotted path actually exists in the upstream tree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**使用**:

```bash
cd /data1/code/agentic-rl/kernel/triton-ascend-bench
python3 /tmp/batch_defer_to_golden.py
# 然后人工 review 每个 meta.yaml,确认 upstream_callable.module 在
# infer/vllm/<...>.py 真的存在:
for tid in L1_04 L1_13 L1_14 L1_19 L1_22 L1_23 L1_24 L1_28 L1_29 L1_31 L1_40 L2_01 L2_03; do
  echo "=== $tid ==="
  grep -A3 "phase_b_capture" tasks/level*/${tid}_*/meta.yaml | head -10
done
```

> **⚠️ 关键人工 review**:脚本里 `TARGETS` 的 `(module, function)` 是**基于 task 名 + upstream path 的最佳猜测**,**未经验证**。bench 操作者必须打开 `infer/vllm/<module path>.py` 确认 `function` 名确实导出。如果导出名不一致(如 vLLM 内部命名为 `_apply_min_p` 而非 `apply_min_p`),需要更正。否则 phase_b golden capture 会拿不到东西。

#### 3.2.4 单独处理:assertion 桶(3 个,需手动判断)

| Task | Assertion | 推荐路径 |
|------|-----------|---------|
| `L1_03_convert_req_index_to_global_index` | `AssertionError:`(空消息)from `_pa_oracle_check_runner.py:147` | 先打开 `kernel.py` 看 assert,可能是 input shape 验证;若 shapes.yaml 可调,走 `patch-shapes`;否则 defer。 |
| `L1_34_transform_index_page_table_decode` | 同上 | 同上 |
| `L1_38_silu_mul_fp8_quant_deep_gemm` | `H must be divisible by 8` | **优先 `patch-shapes`**:检查 `shapes.yaml.hidden_shapes` 中 `H` 值,确保 H1/H2/H3/H4 全部 % 8 == 0。若现有 shapes 已合规仍 fail,说明 kernel 拿到的不是 `H` 这个 knob 而是另一维度,需 defer。 |

操作步骤(以 L1_38 为例):

```bash
cd /data1/code/agentic-rl/kernel/triton-ascend-bench
# 1) 检查现有 hidden_shapes 的 H 值
grep -A2 "H:" tasks/level1/L1_38_silu_mul_fp8_quant_deep_gemm/shapes.yaml | grep "H:" | sort -u
# 2) 找 kernel.py 里那个 assert
grep -n "assert.*divisible by 8\|H % 8" tasks/level1/L1_38_silu_mul_fp8_quant_deep_gemm/kernel.py
# 3) 决策:
#    - 若 assert 检查的就是 H knob -> 改 shapes.yaml,把 H1/H2 的 H 调整到 8 的倍数
#    - 若 assert 检查的是 hidden 维度(不是 knob) -> defer-to-golden
```

#### 3.2.5 完整 walk-through:`L1_22_min_p`

**当前状态**([confirmed] 从 `task_data_43.json`):
- `verdict_class`: `upstream-not-invocable-here`
- `upstream_path`: `vLLM sample`(meta 里写的简称)
- `kernel.py` imports: `from vllm.triton_utils import tl, triton`
- `reason`: `TypeError: 'function' object is not subscriptable` at `kernel.py:67` 行 `_min_p_kernel[(num_tokens,)](...)`

**根因**:
`vllm.triton_utils.triton` 在 vllm 全套加载后是真 triton 模块;在 standalone(单文件 import)时是 stub function(vllm runtime 没起来)。`_min_p_kernel[(num_tokens,)]` 期望 `triton.jit` 装饰返回的对象支持 `__getitem__`,但 stub function 不支持。

**修法**:`defer-to-golden`。kernel.py 不能动,所以靠 phase B 在真 GPU 机上跑 vllm runtime 时捕获 golden。

**改 `tasks/level1/L1_22_min_p/meta.yaml`**(在文件末尾追加):

```yaml
phase_a_prime:
  skip_invocation: true
  status: SKIPPED_BY_DESIGN
  reason: >-
    vllm.triton_utils.triton is a stub function in standalone import;
    _min_p_kernel[(grid,)] in kernel.py line 67 raises
    "TypeError: 'function' object is not subscriptable". Golden must be
    captured offline against vLLM's installed runtime where
    triton_utils.triton resolves to the real triton module.
phase_b_capture:
  status: PENDING
  upstream_callable:
    module: vllm.v1.worker.gpu.sample.min_p
    function: apply_min_p
  reason: >-
    Operator entrypoint apply_min_p(logits, expanded_idx_mapping, min_p)
    must be invoked under a real vLLM install on the GPU host. Verify
    host cannot satisfy this without installing the full vllm runtime
    (CUDA + cpuinfo + cachetools).
phase_b_capture_required: true
```

**人工核对**:
```bash
# 确认 vllm 树里这个模块和函数存在
ls /data1/code/agentic-rl/kernel/infer/vllm/vllm/v1/worker/gpu/sample/min_p.py
grep "def apply_min_p" /data1/code/agentic-rl/kernel/infer/vllm/vllm/v1/worker/gpu/sample/min_p.py
```

**验证命令**:
```bash
python3 scripts/resolve_upstream_entrypoints.py --task L1_22_min_p
# 期望:resolver 不再尝试 invoke,phase_a_prime 直接 SKIPPED_BY_DESIGN
```

---

### 3.3 `oracle-mismatch`(2 个)

#### 3.3.1 `L1_01_bias` — 决策:`rewrite-primitives`(改 `oracle_pytorch.py`)

**当前状态**:
- `verdict_class`: `oracle-mismatch`
- `tolerance_class`: `elementwise_exact_or_inf_safe`
- `oracle.gpu_triton`: `kernel.py::apply_logit_bias`(已正确指向)
- `oracle.pytorch`: `oracle_pytorch.py::run`

**reason 关键摘录**:
```
NOT within elementwise_exact_or_inf_safe (max_abs=1.509e+00 max_rel=7.722e-01);
STRUCTURE MISMATCH: candidate produced 11 flattened tensor outputs but only 1
matched the oracle's 1; extra unexplained outputs ['expanded_idx_mapping',
'pos', 'num_allowed_token_ids', 'allowed_token_ids']
```

**两层问题**:
1. **structure mismatch**:`apply_logit_bias` 是 in-place(`mutability: in_place_logits`,见 meta.yaml line 32),修改 `logits` 后还把别的 input 张量当 output 返回。oracle 应该只返回 `logits`。
2. **numerical mismatch**:`max_abs=1.509` 远大于 `elementwise_exact_or_inf_safe` 的容差(应该是 elementwise 精确等价或 -inf 安全)。说明 oracle 的实现和 kernel 行为不一致(例如:bias 累加顺序、mask 阈值)。

**修法**:

1. **打开 `oracle_pytorch.py::run`**,确认返回签名:
   ```bash
   cat /data1/code/agentic-rl/kernel/triton-ascend-bench/tasks/level1/L1_01_bias/oracle_pytorch.py
   ```
2. **要求 oracle 只返回 `{"logits": logits}` 一项**(in-place 语义)。如果当前返回多项 dict / tuple,删掉非 logits 项。
3. **对照 `kernel.py::_bias_kernel`**(line 22+ 的 @triton.jit)三阶段实现 oracle:
   - Stage 1:`num_allowed_token_ids[r] > 0` ⇒ 把不在 `allowed_token_ids[r, :num_allowed_token_ids[r]]` 列表里的所有 token 设为 `-inf`
   - Stage 2:`logits[t, logit_bias_token_ids[r, :num_logit_bias[r]]] += logit_bias[r, :num_logit_bias[r]]`
   - Stage 3:`pos[t] < min_lens[r]` ⇒ `logits[t, stop_token_ids[r, :num_stop_token_ids[r]]] = -inf`
4. **关键 elementwise_exact_or_inf_safe** 含义(`tolerance_policy.md`):每个元素 ① 完全相等 OR ② oracle 和 candidate 同时是 `-inf`。所以 oracle 也必须用 `float('-inf')` 而非 `-1e9` 类的占位。

**估时**:30-45 min(需对照 kernel 三阶段重写 oracle 的循环 / 索引)。

**验证命令**:
```bash
python3 scripts/resolve_upstream_entrypoints.py --task L1_01_bias
# 期望:
#   L1_01_bias: RESOLVED (kernel.py::apply_logit_bias)
```

#### 3.3.2 `L1_09_quantize_k_cache_fast` — 决策:`defer-to-upstream-oracle`(改 `oracle_pytorch.py` 直接 wrap upstream;若仍超容差则 defer-to-golden)

**当前状态**:
- `verdict_class`: `oracle-mismatch`
- `oracle.gpu_triton`: `kernel.py::placeholder`(尚未 resolve 入口)
- 涉及 int8 fp8 量化,数值精度敏感

**问题分析**:量化 op 的 PyTorch oracle 难以做到 bit-exact 等价(rounding mode / saturation / scale 精度都影响)。tolerance class 可能是 `vector_reduce`(查 meta.yaml 确认),允许 `1e-2 rel` 量级,但量化误差很容易超过。

**两步法**:

1. **第一步**:先把 `oracle.gpu_triton` 从 `placeholder` 改成真入口(参照 `kernel.py` 找 `quantize_k_cache_fast` host wrapper)。同时给 `oracle_pytorch.py::run` 实现一个简明的"scale + round to int8"逻辑。跑 resolver 看是否过 tolerance。
2. **第二步(若仍 mismatch)**:走 `defer-to-golden` —— 同 §3.2 推荐的 `upstream-not-invocable-here` 模板,但 reason 写明"PyTorch oracle 难以精确复现 int8 量化 rounding;golden 必须从上游真 GPU 跑产出"。

**估时**:30 min(若第一步通过)/ 45 min(若需 defer)。

---

## 4. 验证 + Acceptance Criteria

### 4.1 单 task 验证

改完任一 task 后:
```bash
cd /data1/code/agentic-rl/kernel/triton-ascend-bench
python3 scripts/resolve_upstream_entrypoints.py --task <task_id>
```

**期望返回**(stdout 末尾):
- `shapes-incomplete` / `oracle-mismatch` 修法 → `RESOLVED (kernel.py::<entrypoint>)`
- `upstream-not-invocable-here` defer 修法 → resolver 仍报 BLOCKED,但 `phase_a_prime.skip_invocation` 已写入。需要跑 `phase_a_prime.py` 验证它正确识别 SKIPPED_BY_DESIGN。

### 4.2 改 oracle.gpu_triton 后落盘

仅对 `shapes-incomplete` / `oracle-mismatch` 走 `patch-shapes` / `rewrite-primitives` 路径:

```bash
python3 scripts/resolve_upstream_entrypoints.py --task <task_id> --apply
```

`--apply` 只触动 `meta.yaml.oracle.gpu_triton` 一行,不动 kernel.py。

### 4.3 全量回归

改完所有 40 个 task 后:

```bash
# 备份现有 baseline
cp results/resolve_all.json /tmp/resolve_baseline_2026-05-24.json

# 重跑全 100 task
python3 scripts/resolve_upstream_entrypoints.py --refresh-cache 2>&1 | tee /tmp/resolve_new.log

# 对比统计
python3 - <<'PY'
import json
old = json.load(open("/tmp/resolve_baseline_2026-05-24.json"))
new = json.load(open("results/resolve_all.json"))
def class_of(v):
    s = v.get("status", "")
    return s.split(":")[-1].rstrip(")").strip() if "(" in s else s
from collections import Counter
co = Counter(class_of(v) for v in old["verdicts"])
cn = Counter(class_of(v) for v in new["verdicts"])
print("BEFORE:", dict(co))
print("AFTER :", dict(cn))
PY
```

**Acceptance**:
- `shapes-incomplete` count: 56 → 56 - 22 = **34**(只动了 40 范围内的 22,剩余 34 留给后续 release 计划)。若 L1_38 走 `patch-shapes` 而非 defer,则 count 还要 - 1 = **33**。
- `upstream-not-invocable-here` count: 38 → 38 - 16 = **22**(若 L1_38 走 patch-shapes 则 -15 = **23**)。
- `oracle-mismatch` count: 2 → **0**(两个都在 40 范围;若 L1_09 走 defer-to-upstream-oracle 路径,oracle-mismatch 仍归 0,只是 SKIPPED_BY_DESIGN 数 +1)。
- RESOLVED 数(resolver verdict):3 → ≥ 27(3 PASS 不变 + 22 shapes 修复 + 2 oracle-mismatch 修复)。注意 `RESOLVED` ≠ `PASS`,真 PASS 还需 GPU 机上 phase_a_prime 走过 S2 verify。
- SKIPPED_BY_DESIGN 数:0 → ~16(全部 defer 走完的 task),含 phase_b_capture.status=PENDING 等下一步 golden 捕获。
- `release_scope.excluded == true` 的 task 数:**必须 = 0**(不允许逃出分母)。

### 4.4 Phase A' / Phase B 后续触发

修复后下一步(**需要 GPU 机**,非本开发机):

```bash
# 在 GPU 机上
python3 scripts/phase_a_prime.py --all  # 对 RESOLVED 的 task 真跑 S2 verify
python3 scripts/phase_b_golden_capture.py --pending  # 对 status: PENDING 的 phase_b 触发 golden 捕获
```

完成后,`SKIPPED_BY_DESIGN + phase_b_capture.status = DONE` 的 task 才能进入 eligibility gate 的 PASS 状态。本计划只负责把 task 推到 resolver `RESOLVED` 或 `SKIPPED_BY_DESIGN` 这两种"可继续"状态。

---

## 5. 估时 + 节奏 + 责任分工

### 5.1 推荐执行顺序

| Day | 工作 | 产出 |
|-----|------|------|
| Day 1(上午) | §3.2.3 批量 defer 脚本运行 + 人工 review 13 个 `upstream_callable` | 13 个 task → `SKIPPED_BY_DESIGN` |
| Day 1(下午) | §3.2.4 处理 assertion 桶 3 个 task(L1_03/L1_34/L1_38);§3.1.2 简单 elementwise 7 个 | 10 个 task |
| Day 2(上午) | §3.1.3 paged-attn / KV cache 6 个 | 6 个 task |
| Day 2(下午) | §3.1.4 MoE 3 个 + §3.1.5 Quant 2 个 + §3.1.6 FlashInfer/SSD 4 个 | 9 个 task |
| Day 3(上午) | §3.3 oracle-mismatch 2 个精修(L1_01 / L1_09) | 2 个 task,全 40 收尾 |
| Day 3(下午) | 全量回归 `--refresh-cache` + diff 报告 + 提交 patch | 收口 |
| Day 4 | GPU 机上跑 phase_a_prime + phase_b_golden_capture(非本机工作) | golden 落盘 |

### 5.2 跨 task 复用建议

- **模板化 yaml 片段**:`defer-to-golden` 三 block 模板应直接复制(只换 `module` / `function` / `reason`),不要逐字重写。
- **相似 op 一致改**:L1_15 / L1_17(`act_and_mul` / `gelu_and_mul`)是同族 SGLang activation,修法几乎一致;L1_32 / L1_33(`create_flashinfer/flashmla_kv_indices`)同源,共享同 vendored 文件特征,修法一致。
- **批 commit**:每完成一个子类(如 §3.1.2 简单 elementwise 7 个全做完),一起 commit 一次,commit message 用 `task: resolve shapes-incomplete for simple elementwise (L1_15 L1_17 L1_18 ...)` 这种风格,**不要带 codex / round / V / Co-Authored-By 等过程词**(CLAUDE.md `No process noise` 规则)。

### 5.3 人工 review 不可省的环节

1. **batch_defer_to_golden.py 跑完后**:逐个 task 打开 `meta.yaml`,核对 `phase_b_capture.upstream_callable.{module,function}` 与 `infer/vllm/...` 实际导出名一致。**导出名错 = golden 永远拿不到**。
2. **shapes-incomplete 改完后**:跑 resolver 单 task 验证,看 verdict 是否真变成 `RESOLVED`。**没变 ≠ 还有别的 missing param** 或 oracle 不一致;不要假设修一项就万事大吉。
3. **oracle-mismatch 改完后**:`elementwise_exact_or_inf_safe` 容差极严,需要在本机用 `TRITON_INTERPRET=1` 跑 oracle vs candidate 两个 shape (P1 + H2) 都过。

### 5.4 风险与回退

- **风险 1**:`batch_defer_to_golden.py` 写错 `upstream_callable` → phase_b 跑不出 golden。
  **回退**:Layer-5 patch validator 拒绝写入(`§6a.3` canonical-path 校验),所以错的 module 路径在写文件时就会被拒,不会 silent corrupt。即便落地了,phase_b_golden_capture 会写 `status: FAILED + reason`,反向可见。
- **风险 2**:某个 `shapes-incomplete` task 改完后变成 `oracle-mismatch`(原来缺参数所以没跑到对比;补了参数才发现数值不一致)。
  **回退**:走 §3.3 oracle-mismatch 三选一路径。
- **风险 3**:某个 `shapes-incomplete` task 改完后变成 `upstream-not-invocable-here`(原来缺参数遮住了 import 错;补了参数后跑到 import 阶段才暴露)。
  **回退**:走 §3.2 defer-to-golden 路径。

---

## 附录 A — 全 40 task 速查表

| Task ID | Class | Repo | 修法 hint |
|---------|-------|------|----------|
| `L1_01_bias` | `oracle-mismatch` | vLLM | rewrite `oracle_pytorch.py::run`(3-stage,structure=1 logits) |
| `L1_02_compute_slot_mappings` | `shapes-incomplete` | vLLM | add `max_num_tokens` scalar |
| `L1_03_convert_req_index_to_global_index` | `upstream-not-invocable-here` | vLLM | assertion empty msg — inspect kernel.py assert,优先 patch-shapes 否则 defer |
| `L1_04_copy_page_indices` | `upstream-not-invocable-here` | vLLM | defer(cpuinfo missing in verify env) |
| `L1_05_gather_block_tables` | `shapes-incomplete` | vLLM | add `batch_idx_to_req_idx` int64 [B] |
| `L1_06_reshape_and_cache_kernel_flash` | `shapes-incomplete` | vLLM | add `kv_cache_dtype` scalar string |
| `L1_07_dequantize_k_cache_fast` | `shapes-incomplete` | SGLang | add `quant_k_cache` int8 tensor |
| `L1_08_dequantize_k_cache_paged` | `shapes-incomplete` | SGLang | add `quant_k_cache` int8 tensor |
| `L1_09_quantize_k_cache_fast` | `oracle-mismatch` | SGLang | 先 set real entrypoint + simple oracle;若超容差转 defer |
| `L1_10_per_token_group_quant_int8` | `shapes-incomplete` | vLLM | meta.oracle.gpu_triton 指明真入口(避免 resolver 误试 GEMM) |
| `L1_11_awq_dequantize` | `shapes-incomplete` | vLLM | 同上,指明 awq_dequantize host wrapper |
| `L1_13_layer_norm_fwd` | `upstream-not-invocable-here` | vLLM | defer(relative import) |
| `L1_14_layer_norm_fwd_1pass` | `upstream-not-invocable-here` | vLLM | defer(triton stub attr missing) |
| `L1_15_act_and_mul` | `shapes-incomplete` | SGLang | add `gateup_output` tensor |
| `L1_17_gelu_and_mul` | `shapes-incomplete` | SGLang | add `hidden_states` tensor |
| `L1_18_fused_softcap` | `shapes-incomplete` | SGLang | rename `x → full_logits`,`cap → final_logit_softcapping`(walk-through §3.1.7) |
| `L1_19_topk_log_softmax` | `upstream-not-invocable-here` | vLLM | defer(cachetools missing) |
| `L1_20_log_softmax` | `shapes-incomplete` | vLLM | resolver 误试 `rms_norm` — 在 meta 写明真入口 |
| `L1_21_gumbel_sample` | `shapes-incomplete` | vLLM | add `logits` tensor + 关联 inputs |
| `L1_22_min_p` | `upstream-not-invocable-here` | vLLM | defer(triton stub func not subscriptable;walk-through §3.2.5) |
| `L1_23_penalties_and_temperature` | `upstream-not-invocable-here` | vLLM | defer(cachetools missing) |
| `L1_24_rejection_sample` | `upstream-not-invocable-here` | vLLM | defer(cachetools missing) |
| `L1_25_merge_attn_states` | `shapes-incomplete` | vLLM | add `output` tensor(in-place out buffer) |
| `L1_27_apply_grammar_bitmask` | `shapes-incomplete` | vLLM | resolver 误把 `_apply_grammar_bitmask_kernel` 当 host entrypoint — 指 host wrapper |
| `L1_28_pack_bitmatrix` | `upstream-not-invocable-here` | vLLM | defer(relative import) |
| `L1_29_prepare_eagle_inputs` | `upstream-not-invocable-here` | vLLM | defer(cachetools missing) |
| `L1_30_act_quant` | `shapes-incomplete` | SGLang | add `page_table_1` int64 |
| `L1_31_triton_mrope_forward` | `upstream-not-invocable-here` | vLLM | defer(relative import) |
| `L1_32_create_flashinfer_kv_indices_triton` | `shapes-incomplete` | SGLang | add `extend_seq_lens`(指明 host wrapper) |
| `L1_33_create_flashmla_kv_indices_triton` | `shapes-incomplete` | SGLang | 同上 FlashMLA 变体 |
| `L1_34_transform_index_page_table_decode` | `upstream-not-invocable-here` | SGLang | assertion empty msg — inspect 后 defer |
| `L1_35_compute_src2dst_triton` | `shapes-incomplete` | SGLang | 在 meta 写明真入口 `compute_src2dst_triton` |
| `L1_36_compute_seg_indptr_triton` | `shapes-incomplete` | SGLang | add `num_local_experts` scalar |
| `L1_37_compute_problem_sizes_w4a8` | `shapes-incomplete` | SGLang | add `problem_sizes1` int32 [E, 3] |
| `L1_38_silu_mul_fp8_quant_deep_gemm` | `upstream-not-invocable-here` | vLLM | 优先 `patch-shapes`(H % 8 == 0)否则 defer |
| `L1_39_state_passing_fwd` | `shapes-incomplete` | vLLM | add `last_chunk_indices` int32 |
| `L1_40_chunk_cumsum_fwd` | `upstream-not-invocable-here` | vLLM | defer(relative import) |
| `L2_01_bmm_chunk_fwd` | `upstream-not-invocable-here` | vLLM | defer(triton.language stub attr missing) |
| `L2_02_chunk_scan_fwd` | `shapes-incomplete` | vLLM | add `cu_chunk_seqlens` int32 |
| `L2_03_chunk_state_fwd` | `upstream-not-invocable-here` | vLLM | defer(relative import) |

---

## 附录 B — 资料引用

- `specs/task_api_contract.md §6a.1-6a.6` — `phase_a_prime` / `phase_b_capture` / `release_scope` schema 权威。
- `specs/scoring_spec.md` line 59 — `release_status == PASS ⟺ phase_a_prime.status == PASS ∧ cost_model.status == PASS`(SKIPPED_BY_DESIGN 通过 eligibility gate 的 §6a.2 增量条款延伸到 PASS)。
- `specs/tolerance_policy.md` — tolerance class 数值约束。
- `scripts/resolve_upstream_entrypoints.py` — 评估入口,顶部 docstring 解释 dual anchor 设计。
- `scripts/automation/lib/file_blocks.py` — Layer-5 patch validator(写 `phase_a_prime` 之类字段时要满足的硬约束,line 576+)。
- `results/resolve_all.json` — 本计划的 verdict 输入(provenance `2026-05-24T14:18:51Z`,tool_version 0.1.0)。

---

**本文档不进入 release tarball。**`.gitattributes` 的 `docs/triage_backlog/ export-ignore` 规则保证它仅在内部 git 树中可见。
