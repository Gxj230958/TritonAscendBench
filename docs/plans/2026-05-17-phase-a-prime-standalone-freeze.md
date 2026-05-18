---
date: 2026-05-17
topic: phase-a-prime-standalone-freeze
status: codex-shipped
supersedes_assumption: "frozen kernel.py is self-runnable"
---

# Phase A′ — 把冻结 kernel.py 做成真正 standalone + 可复现

## 1. 问题陈述（为什么需要 Phase A′）

`triton-ascend-bench` 是 GPU-triton → 910B NPU-triton **迁移** 基准。迁移的
ground truth **定义上**是"被迁移的上游 GPU triton 生产 kernel 算出什么"。
当前 `phase_a_freeze.py` 采用**整文件 vendoring**（保留 Apache 头、简单），
后果用 `[confirmed]` 数据量化：

| 事实 | 数 / 101 | 后果 |
|---|---|---|
| kernel.py 仍含 live `from vllm…/sglang…` import | **87** | 不是 standalone；直接跑 ImportError；**且行为依赖运行时装的 vllm/sglang 版本 → 不是真冻结，golden 不可复现** |
| 仅 `@triton.jit`、无 host launcher | 14 | 无法直接 `run(inputs)` |
| 多-kernel pipeline（≥2 个 jit，需 host 编排） | 69（L1 24 / L2 27 / L3 18） | 端到端行为含 host 编排，不在单 launch |

**核心论点**：一个 `kernel.py` 偷偷 `from sglang.srt.distributed import …`，
其行为取决于环境里的 sglang 版本——这从 provenance 上就**不是冻结**。它的
golden 不可复现。这不只是"采集环境不便"，而是**直接打穿 baseline 精度/可复现
保证**（见 [[feedback-baseline-no-subset-escape]]：精度问题不许靠缩 scope 回避，
必须全量在 provenance 层立住）。Phase A′ 是 B′(ground-truth 采集) 能被定义的
**前置**，也是"冻结"二字真正成立的前提。

## 2. 目标与成功判据

**目标**：每个 `tasks/level*/<task>/kernel.py` 成为
**standalone + deterministic + version-pinned + 行为等价已验证** 的冻结快照，
使 ground-truth 采集 = "在纯 `torch+triton+CUDA` 盒子上 `run(inputs)→dict`"，
**不需要可运行 vLLM/SGLang 框架的机器**。

**成功判据（每条可机器校验）**：
- S1 101/101 kernel.py 在**仅装** `torch==2.5.1 triton==3.2.0`（+ Class-D 白
  名单 pkg）+ CUDA 13.0.0（`infra/version_matrix.toml` 的 `[gpu]`）的环境里
  `import` 成功、`run(inputs)→dict` 可执行，**无任何 vllm/sglang import 解析
  到 `infer/` 或 site-packages**。
- S2 **行为等价 gate（codex-r2 #1 修正：不用 tolerance_class）**：对每个
  task，`severed_kernel` vs `original_kernel`（original = 能 import
  `infer/vllm,sglang` 跑未切版）经 **S2-gpu-authoritative**（GPU 上，唯一
  PASS 来源）做**严格等价**：dtype/shape/strides/alias-in-place/被 mutate
  输入/全部输出 **bit-exact**（确定性 task 无容差；详见 §5）。**不用 task
  `tolerance_class`**（那是 candidate 评分用，过松）。不过 → BLOCKED
  （触发 S6，不降级、不剔除）。
- S3 确定性：同 (inputs, seed) 重复 N=5 次 **bit-exact**；仅 policy 显式
  标 `non_deterministic` 的 kernel 按其 `rerun_count` + 声明 band。
- S4 provenance：每个 standalone kernel.py 头部记录 upstream repo + commit_sha
  + 原文件 sha256 + 本 task 实际 vendoring 的传递依赖清单 + 每个被 stub 符号
  的"数值无关"判定依据；NOTICE 更新。
- S5 `infra/version_matrix.toml` 增 `[gpu_capture]` 段固化采集环境，偏离即
  invalidate（与现有 "Any deviation invalidates baseline" 一致）。
- **S6（硬发布 gate，codex-r1 FIX#7；codex-r2 #3 落地为可执行）**：只要存在
  **任一 score-eligible 任务** `phase_a_prime.status != PASS`，benchmark
  **不可发布**。**具体落地点（非口号）**：
  1. **存储**：`meta.yaml.phase_a_prime: {status: PASS|BLOCKED|UNRUN,
     reason, s2_gpu_run_id, classified_at}`（由 `phase_a_prime.py` 写）。
  2. **校验**：`scripts/validate_tasks.py` 新增检查——score-eligible 集
     （现有 cli.py L597 逻辑）∩ `{status != PASS}` 非空即 fail。
  3. **评分拒绝**：`tab analyze` 在 `cli.py` 计 score-eligible 后，若任一
     eligible task `phase_a_prime.status != PASS` → score_report 标
     `release_blocked: true` + 列阻塞 task，**拒绝产出可发布榜单**（仍可出
     诊断报告）。
  4. **CI**：`.github/workflows/ci.yml` 新增 `phase-a-prime-gate` step 跑
     上述 validate 检查（现 CI 仅 schema/version，需扩）。
  BLOCKED 是硬发布阻塞器，非"剔出分母"——这才让"BLOCKED ≠ 排除"有牙齿，与
  [[feedback-baseline-no-subset-escape]] 一致（否则 BLOCKED 变成新的分母黑洞）。

**非目标（Scope Boundary，明确不做）**：
- 不做 ground-truth 采集本身（= Phase B′）。
- 不改 scoring / score-eligible / 分母语义。
- 不在 910B 上跑任何东西。
- 不改 `oracle_pytorch.py` 语义（B′ 才把它降级为 cross-check）。
- 不"修正" upstream kernel 的算法——只做"等价地切依赖"，行为必须 byte/tol 不变
  （S2 强制）。改动 kernel 数值 = Phase A′ 失败，不是 Phase A′ 的权限。

## 3. 设计：依赖分类 + 处置规则（v2，codex-r1 FIX#1/#2/#5/#6 重写）

**关键修正（codex-r1 #1 [confirmed]）**：原"A 当且仅当在数据/控制依赖链上"
是**静态不可判定**的（装饰器、模块 import 副作用、`torch.ops`、env/文件/
平台读取、`@triton.autotune`、相对 import 都改变可达行为）。实证反例：
`L3_09` 用 device name + `VLLM_TUNED_CONFIG_FOLDER` 选 tuned JSON config，
`VLLM_BATCH_INVARIANT` 改默认 config——这些不是 scaffolding。
因此**改为保守可达闭包 + 默认归 A + Class-B 默认拒绝白名单**。

每个外部符号经**符号级可达闭包分析**（非模块名匹配）归入唯一一类：

### Class A — 默认类（**递归 vendoring**）
**默认归宿**。凡触及以下任一即 A：被 kernel/launcher 调用；装饰器/grid
lambda/autotune key 或 config；模块级常量；env / 文件 / 平台读取
（`current_platform`、`*_ENV`、tuned-config 文件夹）；相对 import；
`torch.ops.*`；custom-op 注册；任何静态不能证明无运行时数值/形状/调度影响
的符号。**未知动态行为 → A（绝不 B）**。
- 处置：从 `infer/<repo>@commit_sha` 抽符号 + 传递闭包，vendoring 到
  content-addressed 池（见 D1），改写 import 指向池，保留 Apache 头 + 记
  provenance（含可达闭包路径）。

### Class B — 仅"可证明数值惰性"符号（**极窄白名单，默认拒绝**）
**不再是"看起来像环境就 stub"**。一个符号入 B 必须有**正向证明**其运行时对
输出/形状/调度/dtype/stride/副作用**完全无影响**（如：纯 logging sink、
纯类型注解 import、确定性常量且值与上游默认一致且被记录）。证明依据逐符号
写入 S4 provenance。无法正向证明 → 退回 A。`triton_utils` 不整体 stub：
其 `import triton; tl` 再导出部分可 B，但 `HAS_TRITON`/autotune helper/
capability gate 属 A。`platform_utils.num_compute_units`（实证决定 `L1_13`
launch tiling）属 A，非 B。

### Class C — 可选加速 fallback（**显式语义 fork**，codex-r1 FIX#5）
`deep_gemm`、`flash_attn`、`torch.ops._C.*`、kernel 内 `if cuda_arch>=80:
accel else triton`（实证 `L1_38`）。
- 处置：迁移基准对象是 **triton 路径**，golden = triton 路径输出。**这是
  显式语义 fork**：(a) per-task `contract.md` + `meta.yaml.reference_path:
  triton_fallback` 记录"参考 = triton fallback，非生产 accel 默认路径"；
  (b) **S2 的 original 侧必须用同样 accel-disabled 契约运行**（否则比的是
  severed-triton vs original-accel，无效）；(c) B′ 采集与 candidate 评分同
  此 triton 契约。承认：这意味着 golden ≠ vLLM 生产实际跑的（生产默认 accel）
  ——这是 triton 迁移基准的有意定义，必须透明记录。
- **（codex-r2 #4 修正）no-triton-fallback ⇒ BLOCKED**：Class-C 仅当**存在
  可达的 triton fallback 且 S2 能跑它**时适用。实证 `L2_11` 有
  `torch.ops._C.per_token_group_fp8_quant_packed`(标 "CUDA kernel path
  only")——若某 task 是 **accel-only、无 triton 路径**，"强制 triton" 不可能
  → 该 task **BLOCKED**（触发 S6），**绝不**伪造/跳过（贯彻无 subset 逃逸）。

### Class S — 显式语义配置（env/device/file 驱动的 config 选择）
（codex-r1 scenario-1）`VLLM_TUNED_CONFIG_FOLDER` 选 tuned block config、
`VLLM_BATCH_INVARIANT` 改默认 config、device-name→config 映射。
- 处置：**不静默 stub**。pin 由**确定性规则**选定（codex-r2 #5 修正，非
  "通用 fallback" 任意值）：**= upstream 在 "tuned-folder 缺失 + 所有相关
  env 未设" 下的原生默认解析路径**（即上游自己的 default code path，可复现、
  非杜撰）。写入 `meta.yaml.pinned_semantic_config{rule, resolved_value,
  upstream_default_ref}` + contract；S2 断言 original/severed **两侧用同一
  pinned config**。**若该 pin 与生产 tuned config 不同，须像 Class-C 一样
  显式声明** "golden ≠ production tuned config"（透明，不掩盖）。

### Class D — 稳定第三方公开包（**pin，不 vendoring**）
`einops` 等。加入 `version_matrix.toml [gpu_capture].extra_pins`。存疑降 A。

> 仲裁：默认 A；B 需正向数值惰性证明（默认拒绝）；C/S 是显式语义决策且必须
> 双侧契约一致 + 记录；D 仅限 PyPI 稳定且与 vllm/sglang 无关。

## 4. 多-kernel pipeline（69 题）：per-task 最小 run() 契约（codex-r1 FIX#8）

**修正**：不"整体冻结 upstream host wrapper"（对 `L3_09` 这类
module-scope import config/modular_kernel/_custom_ops/`torch.ops.vllm.*` 的
深纠缠文件会拖进整个框架）。改为**每 task 提取最小 `run(inputs:dict)->dict`
harness**：
- 只 vendoring 从 `run` 入口可达的 host 逻辑闭包（按 §3 分类，逐符号）；
- 显式声明并比对**副作用**：in-place 写、aliasing、被 mutate 的输入张量都
  纳入 S2 等价断言（不只比返回值）；
- module-scope 框架 import / `torch.ops.vllm.*` 由 §3 闭包解析（vendoring 或
  Class-S 语义配置），**绝不整体拖框架**。**（codex-r2 #6 修正：删除"证明
  可安全移除"这个又溜回来的不可判定证明 / T3 人工不是开放式证明逃逸口）**
  module-scope 副作用 import 的处置**只有三种**：① 命中**机械白名单**的可证
  惰性 import（纯 logging/typing，规则同 Class-B 白名单）→ 可移除；②否则
  **保留并 vendoring** 其副作用闭包（按 §3）；③ 两者都不行 → **BLOCKED**
  （触发 S6）。**没有"人工判定它其实安全所以删掉"这一档**。
- 14 个 jit-only：统一确定性 launcher 模板（grid 由 shape 推导、autotune
  pin 单 `triton.Config`、单 stream、无分布式）。

## 5. 行为等价验证 Gate（S2，v2，codex-r1 FIX#2/#3/#4/#6 重写）

S2 是 **refactor 等价证明**，不是 correctness 检查——**不得用 task
`tolerance_class`**（其 FP8 类 atol=1e-1 等过松；且 `tolerance.compare` 把
float 转 fp32 再比会放过 dtype 错配）。

**S2 = S2-smoke（不可 PASS）+ S2-gpu-authoritative（PASS 必需）**：
- `S2-smoke`：CPU / `TRITON_INTERPRET=1`，仅做 import 解析 + 路径 + shape
  冒烟预筛。**永远不产生 PASS**（codex-r1 #8 [confirmed]：interpret 不能验
  CUDA custom op / GPU rounding / atomic / autotune 选择 / FP8）。
- `S2-gpu-authoritative`：**GPU 上**，PASS 的唯一来源。

**进程隔离 + import 审计（codex-r1 FIX#2，防 sys.modules 污染）**：
original 与 severed 各在**全新独立子进程**跑；severed 子进程做 **import
审计**，断言其解析的模块**零** vllm/sglang/site-packages 泄漏（只允许
torch+triton+Class-D+content-addressed 池）。审计不过即 BLOCKED。

**等价断言（FIX#4，严于 tolerance_class）**：逐 (输入 shape, seed) 断言
severed vs original：① dtype 完全一致 ② shape 一致 ③ strides/layout 相关处
一致 ④ alias/in-place 副作用与被 mutate 输入一致 ⑤ **全部**输出张量
**严格 bit-exact**。**（codex-r2 #2 修正：删除 ≤1–2 ULP 逃逸口）**——severing
是同代码的纯 import 改写 + 池 vendoring，**不应 reassociate**；任何确定性
task 的非 bit-exact 差异**不是"合法数值漂移"，而是 config/副作用未真正冻结
的证据**（import 顺序、triton autotune cache、dict 序、env/默认 config 解析）
→ 一律 **BLOCKED + investigate**（调查必须定位并冻结那个未冻结的
config/副作用，修好再跑，不得靠放宽容差通过）。**唯一例外**：policy 预先
显式声明 `non_deterministic` 的 kernel，用其 `rerun_count` + 声明 band。
`tolerance_class` **只**用于 candidate 评分，绝不用于 S2。

**采样（D2 修正，codex-r1 FIX#6 去循环）**：
`{small,medium,large} × seed{V0,V1,V2}` + **A′ 专属 provider-only
validation-hidden** `hidden×2 × seed{V3,V4}`，含非连续/转置/边界 dtype/
mask-boundary 输入。**A′ 的 validation-hidden 种子与 B′ 的 sealed
score-hidden 种子不相交**（A′ 绝不在 B′ 评分用的 sealed shape 上调 severing
决策）。Class-C/S 任务：original 与 severed **双侧同 accel-disabled +
同 pinned semantic config**（否则 S2 无效）。

**判定**：S2-gpu-authoritative 全过 + S3 确定性过 → 写 standalone
kernel.py + 池引用 + provenance + 更新 meta.yaml；任一不过 →
`phase_a_prime: BLOCKED`（**触发 S6 硬发布阻塞**，不冻结、不进 B′、
**不剔分母**）。

## 6. 工作分解与**诚实**估计（v2，codex-r1 FIX#6：按符号级闭包，非模块名）

**修正**：Tier 由**每 task 符号级可达闭包**实测决定，**非模块名**。codex-r1
实证模块名会骗人：`platform_utils.num_compute_units` 影响 tiling 却名似环境；
SGLang `cached_triton_kernel` 行为随 CUDA/env 变。**诚实预期：因默认归 A +
B 需正向证明，多数 task 闭包含 ≥1 个 A → T2/T3 占多数，T1 大概率远小于早期
估计、甚至坍塌。** 不承诺"60-70 可机械化"。

实现**第一步**跑 `phase_a_prime.py --classify --dry-run` 产出**真实**符号级
闭包直方图 → `docs/plans/phase_a_prime_classification.md`，**以实测为准**定
Tier。Tier 仅作调度假设：
- T0 闭包仅 torch+triton（少）→ 全自动 gate+冻结
- T1 闭包仅含可证明惰性 Class-B（**预期很少**）→ 脚本批处理
- T2 浅 Class-A 闭包（少量符号、收敛快）→ 半自动 vendoring + 人核
- T3 深纠缠（多 A 符号 / pipeline / `torch.ops.vllm.*` / 大闭包，如 L3_09）
  → **真 per-task 手工**，不可机械化长尾（[[project-tab-pertask-tail]]）

诚实交付：按实测 Tier 分批；**报告真实通过 S2-gpu-authoritative 的题数**；
T3 明标 per-task；**不 fake 全量**；只要有 score-eligible BLOCKED 即按 S6
声明 benchmark 未达发布条件（不掩盖）。

**交付物**：① `scripts/phase_a_prime.py`（符号级闭包分类器 + 池化 vendoring
引擎 + 子进程 S2 runner + import 审计 + S3）② content-addressed
`tasks/_frozen_deps_pool/`（D1）③ 极窄 `tasks/_frozen_shim/`（仅证明惰性
Class-B）④ per-task standalone kernel.py + 池引用 + meta.yaml
(`commit_sha/file_sha256/frozen_deps[]/classB_symbols[]+证明/reference_path/
pinned_semantic_config`) ⑤ `version_matrix.toml [gpu_capture]` ⑥
`docs/plans/phase_a_prime_classification.md`（符号→类 全表 + 证明，可审计）
⑦ per-task PASS/BLOCKED 报告 + S6 发布判定。

## 7. 风险与缓解（v2）

- R1 **B 误判改了数值**（采样不足）→ B 需正向数值惰性证明（默认拒绝）+ S2
  bit-exact + 非连续/dtype/mask-boundary 采样 + 子进程 import 审计；存疑一律 A。
- R2 **传递闭包爆炸** → 池化去重 + 深度上限 + 环检测；超阈值入 T3，记录闭包。
- R3 **upstream 不确定性**（autotune/atomic/RNG/tuned-config）→ autotune pin
  单 config、Class-S pin 语义配置并双侧一致、seed 固定、Class-C 强制 triton；
  ref 与 severed 同确定性 + 同 pinned config + 同 accel 契约。
- R4 **original 侧跑不起来**（`infer/vllm` 自身缺依赖）→ 先建最小可 import
  验证环境；跑不起 = 该 task BLOCKED + 记录（暴露其 provenance 风险，触发 S6）。
- R5 **A/B 误判** → 默认 A + B 正向证明 + S2 bit-exact 兜底；S2 fail 必复审
  分类，**绝不放宽容差**。
- R6（**修正 codex-r1 #8**）：**Phase A′ S2-gpu-authoritative 必须有 GPU**
  ——interpret 不能验 CUDA custom op/rounding/atomic/autotune/FP8，只能做
  S2-smoke 预筛、不产 PASS。GPU 同时服务 A′ S2-authoritative 与 B′ golden
  采集。"A′ 不需要 GPU"是错的，已纠正。

### 7.x Pass-S2-but-wrong 防御（codex-r1 四场景，逐一封口）
| 场景 | 防御 |
|---|---|
| shim 固定 `VLLM_BATCH_INVARIANT`/无 tuned-config，但生产 shape 会用 tuned config | Class-S：语义配置 pin 且记录入契约，**双侧同 pinned config**；不静默 stub |
| 错 dtype 因 compare 转 fp32 被放过 | S2 显式 dtype 断言（不经 tolerance.compare），bit-exact |
| original+severed 同进程致 sys.modules 污染 → S2 过但干净盒子挂 | 各自全新子进程 + severed 侧 import 审计断言零 vllm/sglang/site 泄漏 |
| 可选 accel task 仅因 original 隐式跑了 CPU/fallback 才过 | Class-C：original 侧强制同 accel-disabled 契约 + 记录 reference_path |

## 8. 已决议（自主拍板，不留给后续决策）

- **D1（原 Q1，已决）**：用 **content-addressed 共享池**
  `tasks/_frozen_deps_pool/<sha256[:12]>/<module>.py`。理由：跨 task 去重 +
  sha 本身即 pin + 可审计闭包；per-task `meta.yaml.frozen_deps[]` 记录引用的
  pool 条目 sha。kernel.py 的 import 改写为指向 pool 的稳定相对路径。
- **D2（原 Q2，已决；codex-r1 FIX#6 去循环更正）**：S2 含 hidden 抽样 =
  `{small,medium,large}×seed{V0..V2}` + **A′ 专属 provider-only
  validation-hidden** `hidden×2×seed{V3,V4}`。**A′ validation-hidden 种子与
  B′ sealed score-hidden 种子不相交**（A′ 绝不在 B′ 评分用的 sealed shape 上
  调 severing 决策——否则 severing 决策泄漏进评分集）。种子分裂规则记入
  实现。任一组 S2 fail → BLOCKED。
- **D3（原 Q3，已决）**：§6 的 Tier 题数是**估计**；实现**第一步**即跑
  `phase_a_prime.py --classify --dry-run` 产出真实 `import→Class` 直方图与
  Tier 实分配，写入 `docs/plans/phase_a_prime_classification.md`，**以实测直方
  图为准**校准后续批处理（这是定义好的校准步骤，非待决问题）。

## 9. 与既有设计/记忆的一致性

- 贯彻 [[feedback-baseline-no-subset-escape]]：BLOCKED ≠ 剔分母；精度全量立住。
- 遵 [[project-tab-pertask-tail]]：T3 长尾诚实标注，不 fake 全量。
- 与 v3.2 一致：correctness 由"就是冻结 kernel 的实际输出"定义为真；perf 仍
  deferred 到 910B（Phase A′ 不碰 perf）。
- 复用既有：`phase_a_freeze.py` 的 `resolve_source/_provenance_header`、
  `infra/version_matrix.toml` pin 机制、`tolerance.py` 分档容差、
  `tasks/_vendored/` 快照命名约定。

## 10. codex-r3 SHIP — 实现澄清（已采纳，非新改方案，防实现期歧义）

codex 三轮收敛（r1 8 must-fix → r2 6 → r3 SHIP）。r3 SHIP 附以下实现约束，
全部采纳为实现期 invariant：

- **C1 非确定性 guard（封 tolerance 后门）**：`non_deterministic` 必须在
  PASS run **之前** 由 policy **显式声明**，其 band 来源**只能是 original
  kernel 自身的 original-vs-original 重复实测**（frozen env 下），**绝不**
  来自 severed-vs-original 差异、也绝不用 `tolerance_class`。"跑出来发现不
  确定 → 临时加 non_deterministic 标 → 放宽通过" 是**禁止**的后门：发现未
  声明的不确定性 = BLOCKED+investigate（要么定位并冻结不确定源，要么由
  owner 显式 policy 声明并附 original-repeat 证据，再重跑）。
- **C2 release_blocked 与诊断评分共存（已确认 coherent）**：score doc 可为
  **诊断**存在（含 perf_status=unavailable）；`release_blocked:true` 时仅
  **拒绝可发布榜单**，二者不矛盾，需在 score_report 明示二态。
- **C3 Class-S device-dependent 默认**：若 upstream 默认仍读 device/arch，
  `resolved_value` 必须**字面 pin**（不在运行时按 ambient device 重解析），
  且 `[gpu_capture]` 记录采集设备 identity/capability/FP8 语义；pin 是该设备
  上的字面解析值。
- **C4 C/S 重叠**：一个 kernel 同时有 accel-fork 与 tuned-config 时，**两个
  契约都施加**（accel-disabled 路径 + pinned semantic config），S2 双侧均按
  此双契约跑。
- **C5 无残留不可判定**：实现层最终规则——unknown → A；
  unvendorable / unauditable / no-triton-path / 未冻结副作用 → BLOCKED。
  `phase_a_prime.py` 不做任何"静态证明它安全"的不可判定判断。

> 状态：codex-shipped。本方案作为 /ce:plan 的输入冻结。

---
附录 A：87 题外部 import Top 频次（实测，分类草案见 §3）见
`scripts/phase_a_prime.py --classify --dry-run` 产出的完整 import→Class 表。
