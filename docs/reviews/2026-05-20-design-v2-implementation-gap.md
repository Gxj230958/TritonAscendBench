# DESIGN-V2.md 目标态 ↔ 代码实现差距诊断

**Date:** 2026-05-20
**对象:** `triton-ascend-bench`(GPU Triton → 昇腾 910B NPU Triton 迁移评测基准)
**核心问题:** `docs/DESIGN-V2.md`(最终目标设计规范)与当前代码实现的差距还有多大?

## 盘点方法

- 目标态来源:`docs/DESIGN-V2.md`(present-tense 目标设计,刻意省略实现进度)、`specs/scoring_spec.md`、`specs/task_api_contract.md`、`specs/reward_hack_defense.md`。
- 代码盘点:逐一读 `triton_ascend_bench/{scoring,eligibility,dataset,eval,inputs,process_isolation,adversarial,static_checker,timing,cli}.py`、`scripts/{validate_tasks,phase_a_prime,phase_b_oracle}.py`、`infra/version_matrix.toml`、`tasks/level{1,2,3}/*/{meta,shapes,oracle_pytorch}.{yaml,py}`、`tasks/_canary/`。
- 交叉核实:实现树 grep;100 个 task 的 meta/shapes/oracle 字段统计;论断**经 codex 独立交叉核实六轮修正** —— 撤回初版"差距收敛单一链"的 overclaim,修正 B6 事实错误,补入 hidden-shape 密封、Layer E、golden 双路径、fallback 归一化、profiler/sanity、oracle-body 审计、eval `import torch` bug 等条目,删除 B7 固定分母误分类。

> **重要前提**:DESIGN-V2.md 是目标态规范,"代码落后目标"本身不是 DESIGN 的缺陷 —— 它正是本诊断要量化的对象。三层中 `DESIGN-V2.md` 与 `specs/` 均为目标态,仅代码层旧。

## 一句话结论

差距分布在**两条轴 + 一个 bug**:

1. **性能轴 —— 整条未实现(差距主体)**:从 `meta.cost_model` 数据 → 计时管道 → Roofline 评分,6 个环节当前 **0 个完成**,性能分完全不产出(`scoring.aggregate_perf()` 恒返回 `perf_status="unavailable"`)。强耦合链,须整条补齐才有产出。
2. **正确性/防御轴 —— 9 个未完成项**:hidden shape 既未注入也未对 candidate 密封、canary 题库为空且运行时防御(metamorphic / canary / Layer E 审计 / profiler+sanity)未接入评测、19 个 `upstream_oracle_required` 任务缺 provider 离线 golden、fallback→correctness-fail 归一化未实现、validator 完整性不足。彼此相对独立,可并行补。
3. **实现 bug(E1)**:`eval.py` 的 correctness grading 编排缺 `import torch`,gradeable 任务的 public-shape 比对跑到 `torch.is_tensor` 即 `NameError`,被宽 `except` 静默吞掉 —— 当前 dev 机上 correctness 评测实际不产出有效结果。1 行级修复,应**最优先**做。

冻结链(`phase_a_prime`/severing/入口双锚)、5 文件 loader、`eligibility`、`inputs`/`tolerance`/静态防御 A-D/F 模块、固定 `|S|` 评分分母、100 任务目录已对齐目标态。代码落后 DESIGN/specs 约一个完整迭代。

## 差距全景表

### A. 已对齐(无 gap / 小 gap)

| # | DESIGN-V2.md 机制 | 代码状态 | 证据 | 置信度 |
|---|---|---|---|---|
| A1 | 符号 severing + 等价校验(`phase_a_prime`) | ✅实现 | `phase_a_prime.py`:standalone freeze + GPU 权威等价验证 + A/B/C/S/D 符号分类 | [confirmed] |
| A2 | 整文件 vendoring / 入口双锚 / 确定性校验 | ✅实现 | `phase_a_freeze.py` / `resolve_upstream_entrypoints.py` / `_pa_subproc_runner.py` / `_pa_oracle_check_runner.py` 均在 | [likely]※未细查内部逻辑 |
| A3 | 5 文件契约 loader(`dataset.py`) | ✅实现 | `load_task`(`dataset.py:64-105`)解析 meta/shapes/contract/kernel 四文件;`oracle_pytorch.py` 不在 loader 执行 —— `TaskDef.oracle_module_name`(`dataset.py:40`)记模块名常量,由 eval 评测期从 `task_dir` 重建路径加载(与 DESIGN §6.1"第 5 文件以路径形态纳入 TaskDef"字面有微小差异:常量 vs 显式 path 字段,功能等价) | [confirmed] |
| A4 | score-eligible 集(`eligibility.py`) | ✅实现 | `is_score_eligible`(`eligibility.py:36-52`)non-stub oracle ∧ non-PHASE-C-MANUAL;cli 与 validate_tasks 共用单一来源 | [confirmed] |
| A5 | size-knob `inputs` / 分级 `tolerance` / 静态防御 `static_checker` **Layer A-D/F** 模块 | ✅模块实现 | `inputs.py` / `tolerance.py` / `static_checker.py` 模块本身齐备且实现(Layer E 见 C5,不含在此)。**注**:把它们串起来的 `eval.py` grading 编排存在 E1 bug —— 模块单体实现 ≠ 端到端可跑通 | [likely]※模块单体核实,编排见 E1 |
| A6 | 固定 `\|S\|` benchmark 评分分母 | ✅实现 | `cli.py:615` `s_native_size=len(score_eligible_ids)`、`cli.py:715` analyze 以 score-eligible 集为分母(§9 no-subset-escape);eval 单任务 `aggregate_score`/`coverage` 是 task-local 诊断(§6.7),不是 benchmark 分母 | [confirmed] |
| A7 | 100 任务目录 | ✅达标 | level1=40 / level2=38 / level3=22 = **100**;`oracle_simplified` 100/100 已迁移新枚举,旧 `oracle_tier` 0 命中 | [confirmed] |

### B. 性能轴 —— 整条未实现(差距主体)

| # | DESIGN-V2.md 机制 | 代码状态 | 证据 | 置信度 |
|---|---|---|---|---|
| B1 | `meta.yaml.cost_model`(per-compute-class `flops {c:W_c}` / per-resource `bytes {r:M_r}` / `gpu_xcheck_tol`) | ❌未实现 | **0/100** meta.yaml 含 `cost_model` 字段;`validate_tasks` 的 `REQUIRED_META_TOP_LEVEL` 不要求它 | [confirmed] |
| B2 | cost-model GPU 交叉校验(`phase_a_prime` 内一步,产出 `cost_model.status`) | ❌未实现 | `phase_a_prime.py` grep `cost_model` 零命中;`update_meta_phase_a_prime`(`L1039`)写的 meta 字段不含 cost-model | [confirmed] |
| B3 | 计时管道(产出 $t_{npu}$:warmup+trials host 计时 + host/event 交叉仲裁,两进程 correctness/timing 拆分) | ❌未接入 | `timing.py` host 计时原语 `measure()`(warmup+n_repeat)已实现,但 `eval` 从不调用 —— `eval.py` 构造 `ShapeRunResult` 时 `measured_ns`/`baseline_ns` 恒传 `None`(`eval.py:140,165`);`timing.py:46-48` event-arbitration(`use_event`)路径 `raise NotImplementedError`,host/event 交叉仲裁不工作;`process_isolation.py:20-23` docstring 明示两进程拆分属 GPU 机器层 | [confirmed] |
| B4 | Roofline 评分(`scoring.py`:`t_ref=max(max_c W_c/π_c, max_r M_r/β_r)`,`e=min(P_cap,t_ref/t_npu)`) | ❌未实现 | 实现树(`triton_ascend_bench/`+`scripts/`)`t_ref/roofline/compute_class/cost_model` **0 命中**;`scoring.py:11,112` 仍是旧 `baseline_ns/measured_ns` speedup 比值;`cli.py:687-710` `analyze` 路径把 submission 的 `aggregate_score` 塞进 `baseline_ns`、硬设 `measured_ns=1.0`(schema-shim、非真实计时);`aggregate_perf()` 对当前任务全返回 `perf_status="unavailable"` | [confirmed] |
| B5 | 复合发布门 `release_status = phase_a_prime.status ∧ cost_model.status` | ❌部分实现 | `run_release_gate`(`validate_tasks.py:203-274`)只查 `phase_a_prime.status != "PASS"`(`L256`);`cli.py:623-628` `release_blocking` 同样仅用 `phase_a_prime.status` 谓词 —— 两条代码路径均缺 `cost_model.status` | [confirmed] |
| B6 | `infra/` 910B 按 compute-class/resource 的 $\pi_c$/$\beta_r$ 峰值表 + 接线 scoring | ❌粒度不足且未接线 | `version_matrix.toml:89-92 [hardware.910b]` 已有粗粒度标量 `peak_fp16_tflops=256`/`peak_int8_tops=512`/`hbm_bandwidth_gbps=1600`,但**未达** DESIGN §5.6/§9 要求的 per-compute-class(fp16/bf16 cube、int8、vector)/per-resource(HBM、通信)粒度,且 scoring 未读取 | [confirmed] |

### C. 正确性 / 防御轴 —— 9 个未完成项

| # | DESIGN-V2.md 机制 | 代码状态 | 证据 | 置信度 |
|---|---|---|---|---|
| C1 | verified 模式注入密封 hidden shapes(防探测) | ❌未实现 | `eval.py:121` `pub, _ = _inputs.shape_ids(shapes_doc)` —— hidden 段被 `_` 丢弃,评测只跑 public shapes | [confirmed] |
| C2 | candidate 净化挂载剔除 `shapes.yaml` 的 `hidden_shapes` 段(§3.3 "非原样挂载整个任务目录") | ❌未实现 | `process_isolation.py:137,148` 把**完整 `shapes.yaml`**(含 `hidden_shapes` 段)路径传给 candidate 子进程;100/100 `shapes.yaml` 内联 `hidden_shapes`,仓内 0 个 `shapes_hidden.yaml` —— hidden shapes 对 candidate 完全可见 | [confirmed] |
| C3 | canary 反作弊题库(`tasks/_canary/` 10 类陷阱题) | ❌未实现 | `tasks/_canary/` **0 文件**;`adversarial.py:8` 注释假设 canary corpus 住在此目录 | [confirmed] |
| C4 | 运行时防御接入评测(metamorphic + canary 在 eval 流程内执行) | ❌未实现 | `adversarial.run_metamorphic_checks` / `check_canaries` **零调用方**;`ShapeRunResult.metamorphic_results` 字段定义但 eval 从不填充 | [confirmed] |
| C5 | Layer E 运行时审计(`sys.addaudithook` 静态层运行时回退防线) | ❌未接线 | 实现树 `addaudithook` **0 命中**;`static_checker.py:9` 仅注释 "Layer E — Runtime audit hook (installed at runner time, not here)",无任何安装它的代码 | [confirmed] |
| C6 | golden 来源双路径(§5.7:`upstream_oracle_required` 题 golden 由 provider 离线 GPU 跑冻结上游 kernel 采集) | ❌单路径 | `oracle_simplified` 分布 = 73 `formula_exact_manual` + 8 `simple_pytorch_ok` + **19 `upstream_oracle_required`**;`eval.py:135` 对所有任务一律 `omod.run(...)` 用现跑 PyTorch oracle 当 golden;`triton_ascend_bench/` 无离线 golden artifact/loader 通道 —— 19 题真值来源缺失 | [confirmed] |
| C7 | fallback→correctness-fail 归一化(资格门谓词 `correctness=ok ∧ ¬fallback`) | ❌未实现 | `scoring.py:240-244` `native_coverage` 仅 `sum(... if correctness_status=="ok")`,不含 `¬fallback`;`eval.py` 的 `TaskEvalResult.fallback_detected`(`eval.py:53`)定义后从不被设值;DESIGN-V2.md:637/§6.7 要求 fallback 评分前归一为 correctness fail | [confirmed] |
| C8 | validator 完整性(spec `task_api_contract.md:133,137-143` 要求的校验) | ❌部分实现 | `validate_tasks.py:120-183` 做了 5 文件存在 / `ast.parse` 语法 / `@triton.jit` 存在 / hash 完整性,`L100-103` 仅查 `oracle_simplified` enum;但**未**校验:① kernel.py 导出 `meta.oracle.gpu_triton` 入口、`oracle_pytorch.py` 导出 `run`、`contract.md` 与 `meta.yaml` 一致性、`cost_model.{flops,bytes}` schema(spec L137-143);② `upstream_oracle_required` 题 oracle body 不得含 `F.silu`/`F.gelu`/`F.softmax`/`torch.softmax` 等高层 op(spec L133)—— 已核实 **≥8 个**此类任务 `oracle_pytorch.py` 含高层 op(`L1_15`/`L2_17`/`L2_19`/`L2_33`/`L2_37`/`L3_01`/`L3_02`/`L3_04`),validator 未拦截(精确违例数取决于 spec "high-level ops" 口径) | [confirmed] |
| C9 | 运行时 sanity / profiler 防御(profiler 检测 NPU kernel-event 存在性防空 kernel、performance sanity) | ❌未实现 | 实现树 `profiler` 仅出现在 `static_checker.py:66-67` 黑名单(禁 solution 调用),无评测侧 profiler kernel-event 检测;`eval.py` 评测主路径从不设 `performance_status`/`performance_flags`(`eval.py:185` 注释 "performance_status intentionally left non-ok") | [confirmed] |

### D. 已实现但数据未产出

| # | 机制 | 代码状态 | 证据 | 置信度 |
|---|---|---|---|---|
| D1 | `phase_a_prime` 已在任务上跑出冻结凭证 | ⚠️未执行 | **0/100** meta.yaml 含 `phase_a_prime` 字段 → 发布门现在跑必然全 BLOCK | [confirmed] |

### E. 实现 bug(已写代码存在缺陷)

| # | 模块 | bug | 证据 | 置信度 |
|---|---|---|---|---|
| E1 | `eval.py` correctness grading 编排 | 缺 `import torch` → `NameError` → 宽 `except` 吞掉 → gradeable 任务 grading 静默 skip | `eval.py` 全文无 `import torch`,但 `eval.py:151` 用 `torch.is_tensor`;`eval.py:175` `except Exception` 仅 `logger.debug` 吞掉 —— 任何 gradeable 任务跑到 `:151` 即 `NameError`,`correctness_status` 停在默认 `"error"`,public-shape 比对实际不产出 | [confirmed] |

## 差距的结构

```
性能轴 B(强耦合链,须整条补齐才产出):
  meta.cost_model 字段(B1) + GPU 交叉校验(B2) + π/β 峰值表(B6)
    → dataset 解析 cost_model → 计时管道(B3) → Roofline t_ref/e(B4) → 复合发布门(B5)

正确性/防御轴 C(相对独立,可并行):
  hidden-shape 完整性:未注入(C1) + 未对 candidate 密封(C2)
  防作弊运行时:canary 题库空(C3) + metamorphic/canary 未接入(C4) + Layer E 审计未接线(C5) + profiler/sanity 未接入(C9)
  真值来源:upstream_oracle_required 19 题离线 golden 缺失(C6)
  资格门计分:fallback→correctness-fail 归一化缺失(C7)
  validator 完整性(C8)

实现 bug E1(与轴划分正交,1 行级修复,须最优先):eval.py 缺 import torch
```

性能轴任一环节缺失,整轴不产出。正确性/防御轴 9 项彼此独立;C1/C2 同属 hidden-shape 完整性、C3/C4/C5/C9 同属"防作弊在评测期生效",各宜成组补。E1 不修复则 correctness 评测全程静默 skip,优先级最高。

## 工作量画像(按补齐成本,置信度标注)

| 环节 | 性质 | 成本画像 | 置信度 |
|---|---|---|---|
| **E1 eval `import torch` bug** | 加一行 import + 收窄宽 `except` | **微改(1 行级)** —— 应最优先 | [confirmed] |
| B4 Roofline `scoring.py` + `cli.py` analyze | 评分公式改写 + analyze 数据装配去 shim | 中改 —— 公式 DESIGN/specs 已定义 | [likely] |
| B5 复合发布门 / `dataset.py` 解析 cost_model | 字段 + 条件 | 各小改 | [likely] |
| B3 计时管道 | eval 接入 `timing.measure` + event-arbitration + 两进程拆分(NPU 实测部分须 910B 机) | 中改 | [likely] |
| B2 GPU 交叉校验脚本 | 新脚本,需 GPU 训练机插桩实测 FLOPs/bytes | 中改 | [likely] |
| B6 `infra/` π/β 峰值表细化 + 接线 | 粗粒度标量 → per-class/per-resource 表 | 小-中改 | [likely] |
| **B1 100 题 cost_model 公式** | **逐题人工推导 FLOPs/bytes** | **抗机械化长尾** —— 每个算子 W_c/M_r 的 size-knob 公式需算子级知识 | [confirmed] |
| C1 hidden shape 注入 / C2 candidate 挂载净化 | eval 加 verified 分支 + process_isolation 剔除 hidden 段 | 各小-中改 | [likely] |
| C4 metamorphic/canary 接入 eval / C5 Layer E addaudithook 接线 / C9 profiler+sanity 接入 | 接入 + 状态传递(NPU profiler 部分须 910B 机) | 各中改 | [likely] |
| C6 golden 离线通道 | eval 加 golden 来源分支 + 离线 golden 存储/loader(19 题离线采集需 GPU 批处理) | 中改 + GPU 批处理 | [likely] |
| C7 fallback→correctness-fail 归一化 | `native_coverage` 加谓词 + eval 实现 fallback 检测并设 `fallback_detected` | 小-中改 | [likely] |
| C8 validator 完整性 | 加 5 类校验(含 oracle-body 审计) | 小-中改 | [likely] |
| **C3 canary 10 类陷阱题** | **逐类设计陷阱任务** | **抗机械化长尾** —— 10 类(identity/zero-out/cache-reuse/...)各需构造可信陷阱 | [confirmed] |

→ E1 是 1 行级 bug;代码骨架(B2-B6、C1/C2/C4/C5/C6/C7/C8/C9)是有界中小改;两个长尾是 **B1(100 题 cost-model 公式)** 与 **C3(10 类 canary 陷阱题)**,二者是工期主体。

## 修复方案候选(诊断阶段仅列出,不在此展开)

0. **最优先修 E1**:`eval.py` 补 `import torch`、收窄宽 `except`(否则 correctness 评测全程静默 skip,后续所有正确性验证都建立在跑不通的链路上)。
1. **性能轴先打通骨架再填数据**:`infra/` π/β 表 → B1 字段 schema → dataset 解析 → B3 计时管道 → B4 scoring → B5 发布门 → B2 GPU 交叉校验脚本;先用 1–2 个样板任务的手写 cost_model 端到端跑通,再铺开 100 题。
2. **cost_model 公式与 golden 双路径对齐分批**:可机械近似的算子用 size-knob 通式批量起草,融合算子逐题人工 —— 与 §5.7 golden 80/20 切分天然对齐。
3. **正确性/防御轴并行推进**:C1/C2(hidden-shape)、C7(fallback 归一化)、C8(validator)是有界改可先做;C3(canary 题库)+ C4/C5/C9(运行时接入)合为一个子项,canary 题库设计是另一条长尾。
4. **GPU 批处理合并立项**:D1(phase_a_prime 在 100 题实跑 freeze)、B2(cost-model GPU 交叉校验)、C6(19 个 `upstream_oracle_required` 任务离线 golden 采集)均需 GPU 机,建议合并到一次 GPU 批处理。

## 局限性

- `process_isolation.py` 完整 Docker/seccomp/cgroup/NPU 设备白名单部署层,DESIGN §3.3 明定属 verified runner 的 GPU/NPU 机器层 —— **不计为 dev 代码 gap**;B3 仅指两进程 correctness/timing 拆分结构 + 计时管道接入,C5 的 `addaudithook` 是纯 Python、不属该机器层。
- A2 / A5 标 [likely]:A2 脚本文件存在但未细查内部逻辑;A5 三模块单体已实现,但端到端编排存在 E1 bug。
- `upstream.function_hash` 与代码实际比对的 `upstream.file_sha256`(`validate_tasks.py:160`)存在 spec/impl 偏差(spec `task_api_contract.md:143` 规定对 `function_hash`);当前 100/100 任务两字段数值相等、行为等价,故非阻塞;若未来两者分化(函数级 hash ≠ 整文件 hash)则应并入 C8。
- hidden-shape 密封形态:`specs/task_api_contract.md:131` 提独立密封文件 `shapes_hidden.yaml`,与 DESIGN §5.2/§3.3 的"`shapes.yaml` 内 `hidden_shapes` 段 + 挂载时剔除"表述不一致;C1/C2 以 DESIGN 为准,最终密封形态待方案阶段统一。
- 本诊断重点盘点评分 / cost-model / golden / loader / 发布门 / 隔离 / 防御链;`manifest.py`、`backends/`、`trusted_shim/` 未逐一盘点 —— 若其中存在与 DESIGN 的实质偏差,本报告未覆盖。
