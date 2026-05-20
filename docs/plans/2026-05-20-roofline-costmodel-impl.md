# 实施方案:Roofline + cost-model 链与防御轴补齐

**Date:** 2026-05-20
**对象:** `triton-ascend-bench`
**依据:** `docs/reviews/2026-05-20-design-v2-implementation-gap.md`(差距诊断,经 codex 7 轮核验收敛)
**目标态:** `docs/DESIGN-V2.md` + `specs/{scoring_spec,task_api_contract,reward_hack_defense,timing_arbitration,process_isolation}.md`

## 1. 目标

让代码追上 DESIGN-V2.md 目标态。诊断确认差距 = 1 个阻断 bug(E1)+ 性能轴整条(B1–B6)+ 正确性/防御轴 9 项(C1–C9)+ 1 个未执行项(D1)。本方案给出落地顺序、每项的文件级改动与验证标准(DoD),供 `/ce:work` 执行。

## 2. 关键约束:进程隔离边界

candidate 只在 `process_isolation.run_candidate_isolated()` 的隔离子进程内执行。当前 trusted-parent ↔ candidate-child IPC 接口很窄:
- **下行(runner→child)**:`repo_root` / `shapes_yaml` 路径 / `solution` / `shape_ids` / `seed`;child 自行 `build_inputs`。
- **上行(child→runner)**:`status.json` = `{shapes:{sid:{ran,error,keys}}, load_error}` + `{sid}.safetensors`。

性能轴与防御轴的几乎所有新信号都须穿越此边界。**协议扩展(W1.0)是性能轴与防御轴的共同前置基础**,必须先做。

## 3. 范围与边界

| 类别 | 内容 | 执行位置 |
|---|---|---|
| 代码骨架 | E1、W1.0 协议、B1 schema/B3/B4/B5/B6、C1–C9 全部代码 | **dev 机**(`/ce:work`) |
| 长尾内容 | B1 的 100 题 cost_model 公式、C3 的 10 类 canary 陷阱题 | **dev 机**(框架+样板先行,逐题/逐类铺开) |
| 数据产出 | D1 freeze 实跑、B2 cost-model GPU 实测交叉校验、C6 19 题离线 golden 采集 | **GPU 机**(方案给脚本,执行单列为 Phase 4) |

**计时说明**:B3 计时管道的代码结构在 dev 机实现并以 CPU 计时占位自测;真实 910B `t_npu` 数值须在 NPU 机产出。dev 机阶段验证管道连通性、上行链路、双沙箱隔离链,不验证数值真实性。

## 4. 阶段总览与依赖

```
Phase 0  E1 修复(阻断 bug,1 行级)
   │
Phase 1  W1.0 协议扩展(基础)──┬─ W1.1 π/β 峰值表
   │     (W1.3/1.4/2.1/2.2/2.4 │ ─ W1.2 cost_model schema
   │      的共同前置)          ├─ W1.3 计时管道 + 双沙箱(依赖 W1.0)
   │                           ├─ W1.4 Roofline scoring + producer schema(依赖 W1.0/1.1/1.2/1.3)
   │                           └─ W1.5 复合发布门(依赖 W1.2)
   │     └─ M1 里程碑:1–2 样板任务性能轴端到端跑通(CPU 计时占位)
Phase 2  W2.1 hidden-shape(依赖 W1.0) · W2.4 运行时防御(依赖 W1.0) · W2.2 fallback(依赖 W2.4)
   │     W2.3 validator · W2.5 canary 框架(此二项独立,可在 W1.0 前先做)
   │     W2.6 存量 corpus 收敛(依赖 W2.3,须在 Phase 4 前完成)
Phase 3  W3.1 cost_model×100 · W3.2 canary×10(依赖 Phase 1/2 的 schema 与框架)
Phase 4  W4.1 D1 · W4.2 B2 · W4.3 C6(依赖 Phase 1/3,需 GPU 机)
```

严格有序:Phase 0 → W1.0 → 其余。W2.1/W2.4 依赖 W1.0,W2.2 依赖 W2.4(fallback 信号生产方);W2.3/W2.5 无此依赖;W2.6 依赖 W2.3,须在 Phase 4 / release-gate 前完成。

## 5. gap → W 项覆盖映射

| 诊断 gap | W 项 | 诊断 gap | W 项 |
|---|---|---|---|
| E1 | W0.1 | C1 | W2.1 |
| B1 | W1.2(schema)+ W3.1(100 题数据) | C2 | W1.0 + W2.1 |
| B2 | W4.2 | C3 | W2.5(框架)+ W3.2(10 类) |
| B3 | W1.0 + W1.3 | C4 | W2.4 |
| B4 | W1.4 | C5 | W2.4 |
| B5 | W1.5 | C6 | W4.3 |
| B6 | W1.1 | C7 | W2.2(消费)+ W2.4(生产 fallback 信号) |
| | | C8 | W2.3(校验)+ W2.6(存量收敛) |
| | | C9 | W2.4 |
| | | D1 | W4.1 |

## 6. 详细 Work Breakdown

### Phase 0 — 修复 E1(阻断 bug)

**W0.1** | gap E1 | 文件 `triton_ascend_bench/eval.py`
- 改动:顶部加 `import torch`;`eval.py:175` 宽 `except Exception` 收窄 —— grading 编排的程序性错误(`NameError`/`ImportError`/`AttributeError`)须冒泡或显式标 `correctness_status="error"` 并记 reason,不得静默 `logger.debug` 吞掉(仅 oracle/candidate 运行期异常可降级为 per-shape fail)。
- DoD:跑 1 个 gradeable 任务,`correctness_status` 产出 `ok`/`fail` 而非恒 `error`;新增 `tests/unit_tests/test_eval_grading_smoke.py` 端到端验证 grading 链路跑通。

### Phase 1 — 性能轴(W1.0 基础 + B 轴)

**W1.0 — child↔parent 协议扩展(基础)** | 文件 `process_isolation.py`、`inputs.py`
协议扩展三部分:
- **(下行)inputs 注入 + 净化最小集**:**所有 shape(public 与 hidden)的 inputs 一律由 runner 物化、经下行通道注入** —— 张量值存 `safetensors`、非张量值(`scalar`/`scalar_knob`)存 JSON sidecar manifest;correctness pass 注入单份,timing pass 注入 per-trial 序列(`t=0..N-1`,`seed=meas_seed+t`)。child **不调 `build_inputs`、不再获得 `repo_root`**,只在净化最小集(`kernel.py` + `meta.yaml` + stripped 公开 shape 视图 + 注入的 inputs + `solution.py`)内运行 —— 与 DESIGN §3.3「非原样挂载整个任务目录」「oracle/golden/hidden 永不挂载 candidate」一致(统一由 runner 物化:既规避 child 无 hidden 定义、`build_inputs` 需完整 `shapes_doc` 的问题,也彻底收口 candidate 视图)。
- **(上行)diagnostics schema 扩展**:`status.json` per-shape 加 `durations` / `kernel_event_present` / `stream_nondefault` / `fallback_signal` / `sampled_outputs` 引用,schema 版本化;`CandidateRunResult`/`ShapeCandidateResult` 同步加字段。
- DoD:child 从 `safetensors`+sidecar 重建含 `scalar` 的 mixed inputs 并跑通;hidden shape 的 timing per-trial fresh-input 路径跑通;`status.json` 含 diagnostics + schema 版本号。

**W1.1 — `infra/` π/β 峰值表** | gap B6 | 文件 `infra/`(新增 `peak_table_910b.toml` 或扩 `version_matrix.toml`)
- 改动:按 compute-class 给 $\pi_c$(fp16 cube / bf16 cube / int8 / vector)、按 resource 给 $\beta_r$(HBM / 跨设备通信);标注数据来源与单位;提供 loader 返回 `{class:π}`/`{resource:β}`。
- DoD:峰值表含 §5.6 全部 class/resource 条目;loader 有单测。

**W1.2 — cost_model schema + dataset 解析** | gap B1 | 文件 `specs/`、`dataset.py`、`scripts/validate_tasks.py`
- 改动:① 固化 `meta.yaml.cost_model` schema:`flops` 与 `bytes` **各支持标量(单 class/resource 退化形,如 `flops: "2*B*L*D"`)与映射(`{c:W_c}`/`{r:M_r}`)两形态**(DESIGN §5.6,与 `task_api_contract.md:106-109` 示例一致),validator/dataset 接受两形态、内部统一归一化为 map;含 `gpu_xcheck_tol`;② **统一 formula evaluator environment**:cost_model 公式除 size-knobs(B/L/D…)外还用派生符号(如 `dtype_size`,见 `task_api_contract.md:108`),须定义一个共享 evaluation environment(`knobs` + 固定 derived symbols + helpers),`dataset.py`/`validate_tasks.py`/W4.2 verbatim 复用同一 evaluator,杜绝符号集漂移;③ `TaskDef` 加 `cost_model` 字段、`load_task` 解析 —— **loader 对 missing `cost_model` 向后兼容**(`TaskDef.cost_model=None`,不报错),使 W3.1 前 `iter_all_tasks`/`cli analyze`/`phase_a_prime` 等共享路径不被破坏;④ `validate_tasks` `REQUIRED_META_TOP_LEVEL` 加 `cost_model` 并校验公式可解析 —— **red→green 过渡仅由 validator 层承载,loader 层始终可用**。**注**:Phase 1 仅用 `--task <id>`/样板 fixture 验证 validator 逻辑;全语料 validator green 推迟到 W3.1 后。
- DoD:schema 文档化(含两形态);`dataset` 解析并暴露归一化 `cost_model`,**对 missing `cost_model` 返回 `None` 不报错**(单测);`validate_tasks --task` 对样板任务合法(标量、映射各一)/非法 `cost_model` 分别 pass/报错;统一 formula evaluator 有单测且三处复用。

**W1.3 — 计时管道 + 双沙箱** | gap B3 | 文件 `process_isolation.py`、`timing.py`、`backends/npu.py`、`eval.py`
- 改动:
  1. **双沙箱**:`process_isolation` 提供 **correctness pass 与 timing pass 两次独立 candidate 子进程调用**,二者之间 `SIGKILL` 进程组 + 清 `/tmp` scratch / 编译缓存 / 设备内存(切断 correctness→timing 预计算→查表污染,DESIGN §3.3/§6.6)。
  2. timing pass child 内 trusted-shim 段:warmup(warm_inputs)+ measured(每 trial fresh inputs,经 W1.0 下行通道由 runner 物化注入)+ runner-secret RNG 抽样 `sampled_outputs` 上行;host `CLOCK_MONOTONIC_RAW` 权威 + NPU event 交叉,按 `timing_arbitration.md` 仲裁 —— `host_ns` 非有限/≤0、或 host-event 漂移 >10% 且 profiler 检出 non-default stream ⇒ `runtime_status="invalid_timing"`;`host_ns<1μs` ⇒ `performance_status="quarantined"`;`performance_flags` 仅承载 `invalid_stream`/drift 细项(落位用现有独立结果位 `runtime_status`/`performance_status`,**不新造 `timing_status`**)。
  3. `timing.py` `measure` 重构支持 per-trial input factory(取代固定 args 重复),event-arbitration hook 落 `backends/npu.py`。
  4. `eval` 只编排两 pass、记录回传 `durations`/`sampled_outputs`,不在 parent 侧计时。
  5. **sampled-output 校验闭环**:runner(parent)对 timing pass 上行的 `sampled_outputs` 重新与 golden `tolerance.compare`,mismatch ⇒ `performance_status` 失效 + `performance_flags.append("measured_outputs_drift")`(闭合 correctness-pass→timing-pass 作弊路径,timing_arbitration.md "for correctness verification")。
- DoD:correctness pass 与 timing pass 是独立子进程、中间 `SIGKILL`+scrub 隔离链可验证;timing child per-trial fresh inputs(含 hidden)跑通;`sampled_outputs` 上行并经 parent 校验,mismatch 时 `performance_status` 失效可验证;计时仲裁失败路径(`runtime_status="invalid_timing"`/`performance_status="quarantined"`/`performance_flags` 含 `invalid_stream`)有单测;`eval` 不含 parent 侧 wall-clock 计时。

**W1.4 — Roofline scoring + producer schema** | gap B4 | 文件 `eval.py`、`scoring.py`、`cli.py`
- 改动:
  1. **producer schema**:`ShapeRunResult` 加 per-shape `t_ref`/`t_npu`/per-class `W`/per-resource `M` 与 `durations` 派生的 `median`/`p99`/`cv`/`trials`(DESIGN §6.6);`TaskEvalResult` 加 task-level `s_t`。
  2. **聚合点明确**:per-shape `e=min(P_cap,t_ref/t_npu)`(`t_ref=max(max_c W_c/π_c,max_r M_r/β_r)`)在 `scoring` 计算;**per-shape `e` → task-level `s_t` 的几何均值聚合在 `eval` 完成**(`s_t=(∏_s e)^(1/n)`,DESIGN §9),per-shape 明细仅作诊断;`TaskMeasurement` 加 `s_t` 字段;`scoring.capped_geomean_native` 消费 task-level `s_t` 做跨 task 几何均值。
  3. **round-result 计时字段钉死**:emit 中 `baseline_ns=t_ref`、`measured_ns=t_npu`(DESIGN §9 provenance),per-shape 披露 `median`/`p99`/`cv`/`trials`;`cli run`/`run-suite` emit task-level `s_t` + per-shape 明细;`cli.py:687-710` analyze 去除 `measured_ns=1.0` shim、从 `s_t` 装配,**不再**用 task-level baseline/measured 比值。
  4. **schema 迁移完整性**:`tab validate-result` 的 result schema 同步更新接受新字段(含 per-shape `t_ref`/`t_npu`/`median`等、task-level `s_t`,以及 W4.3 新增的独立 `task_status` 字段 `evaluated`/`unsupported`/`blocked`);相关测试更新。
  5. **`task_status!="evaluated"` 的 null 字段契约**:`unsupported`/`blocked` 任务天然无有效性能数据 —— `s_t`/`baseline_ns`/`measured_ns`/per-shape timing 明细允许 `null`/缺省;`validate-result` 对此类任务只校验 `task_status` 合法、不要求数值字段;`scoring`/`analyze`/duplicate-merge 对 `task_status!="evaluated"` 的任务统一跳过数值聚合(**不发明哨兵值**,不当 fail 压分)。
- DoD:`scoring` 单测算出正确 per-shape `e`;`eval` 单测:多 shape 任务 `s_t` = per-shape `e` 几何均值;`cli run` 输出 JSON 含 `s_t`/`baseline_ns=t_ref`/`measured_ns=t_npu`/per-shape `median`等;`tab validate-result` 接受新 schema 且对 `task_status!="evaluated"` 任务接受 null 数值字段;`scoring`/`analyze` 正确忽略该类任务的 null 值(单测);`aggregate_perf` 在样板任务产出 `perf_status="ok"`。

**W1.5 — 复合发布门** | gap B5 | 文件 `scripts/validate_tasks.py`、`cli.py`、`scripts/phase_a_prime.py`
- 改动:发布门谓词改为 `phase_a_prime.status==PASS ∧ cost_model.status==PASS`,**三处 release-status authority 同步**:① `run_release_gate`(`validate_tasks.py:256`);② `cli.py:623-628` `release_blocking`;③ `phase_a_prime.py` 的 "S6 release verdict" 报告(`phase_a_prime.py:2920-2997`,该报告明示与 `--release-gate` blocker count parity,须同步纳入 `cost_model.status`)。
- DoD:三路径谓词一致;单测覆盖 `cost_model.status=BLOCKED` ⇒ 发布门拦截;`phase_a_prime` S6 报告 blocker count 与 `--release-gate` 一致。

> **M1 里程碑**(Phase 1 末):2 个样板任务(1 elementwise + 1 reduce)从 cost_model → 双沙箱计时 → t_ref → per-shape e → task s_t → geomean 端到端跑通(CPU 计时占位)。

### Phase 2 — 正确性/防御轴(C 轴)

**W2.1 — hidden-shape 注入与密封** | gap C1+C2 | 文件 `eval.py`、`cli.py`、`process_isolation.py`、`specs/task_api_contract.md`
- 改动:① `evaluate()` 签名加 `mode`(`self-reported`/`verified`)参数;`cli.py` 给 **`tab run` 与 `tab run-suite` 均加 `--mode`** 并贯通到 `evaluate()`(`--mode` 当前仅在 run-suite parser `cli.py:54`;单任务 `tab run` 是 bring-up/smoke 入口,须能跑 verified 以支持 W2.1/W4.3 单任务测试);② verified 模式:runner 侧物化 hidden shapes 的 inputs,经 W1.0 下行通道注入 child;③ `process_isolation` 给 candidate 的是 **W1.0 净化最小集** —— 剔除 `shapes.yaml` 的 `hidden_shapes` 段,且**不暴露 `repo_root` / 原始 `task_dir` / `golden/` / `oracle_pytorch.py` / `contract.md`**,candidate 无法遍历 `tasks/` 访问其它任务或 provider-only artifact。**hidden 密封形态最终裁定**:`shapes.yaml` 的 `hidden_shapes` 段为 hidden shapes **唯一真相源**;candidate 净化挂载剔除该段;`cli --shapes-overlay`(`cli.py:56`)的"独立 hidden 文件"语义**废弃**(flag 移除或标 deprecated 不再生效)——verified 模式 hidden shapes 直接取自 task 自带 `shapes.yaml.hidden_shapes`(runner 持完整文件、candidate 视图剔除)。**spec 同步**:`specs/task_api_contract.md:131` 当前仍写运行时 `shapes_hidden.yaml` augmentation —— 本 W 项须同步改为 `shapes.yaml.hidden_shapes` 段、删 overlay 语义,使 specs 对齐 DESIGN §3.3/§5.2(DESIGN 已是内联段),CLI 文档一并更新。
- DoD:verified 模式经 `cli→evaluate→process_isolation` 贯通跑 hidden shapes;self-reported 模式不注入 hidden;candidate 子进程读到的 shapes 视图不含 hidden 段(单测断言);`--shapes-overlay` 已废弃且测试确认无残留语义;`specs/task_api_contract.md` 已无 `shapes_hidden.yaml` 残留语义。

**W2.2 — fallback→correctness-fail 归一化** | gap C7 | 文件 `eval.py`、`scoring.py`
- 改动:① `eval` 据 W2.4 上行的 `fallback_signal` 设 `TaskEvalResult.fallback_detected=True` 并**强制 `correctness_status="fail"`**(fallback 评分前归一为 correctness fail,DESIGN §6.3/§6.7);② `scoring.native_coverage` 谓词 `correctness=="ok" ∧ ¬fallback_detected`(双保险)。
- DoD:单测:`fallback_signal` 触发 ⇒ `fallback_detected=True` ∧ `correctness_status=="fail"` ∧ 不计入 coverage 分子;emit 的 task result 不会在 fallback 后报 correctness `ok`。

**W2.3 — validator 完整性** | gap C8 | 文件 `scripts/validate_tasks.py`
- 改动:加校验(均 **phase-aware**,与 `validate_tasks` 现有 phase-aware hash 校验一致)——
  ① `oracle_pytorch.py` 导出 `run`、`contract.md` 与 `meta.yaml` 一致、`cost_model.{flops,bytes}` schema、`upstream_oracle_required` 题 oracle body 不含高层 op(spec L133)、`meta.runner_checks.metamorphic_checks` 声明的 check 名均在 W2.4 registry(未知 ⇒ 报错);
  ② **kernel-export(phase-aware)**:`meta.oracle.gpu_triton` 初始为 `kernel.py::placeholder`,由 resolver/`phase_a_prime` 填真实 entrypoint —— `gpu_triton` 仍 placeholder 时只校验字段格式;仅当 `gpu_triton` 已非 placeholder(或 `phase_a_prime.status==PASS`)时,才强制 `kernel.py` 真导出该 callable(否则当前 94/100 placeholder 题会大面积误报);
  ③ **golden 完整性(phase-aware)**:对 `upstream_oracle_required` 题,W4.3 执行后校验 `golden/<shape_id>.safetensors` 布局完整(覆盖全部 public+hidden eval shape);W4.3 前不 enforce。此校验属 `validate_tasks` schema 模式 —— 发布流程跑完整 `validate_tasks` 即拦截缺 golden 的 verified 题,**不改 DESIGN §4.4 的二元 `release_status` 谓词**(`phase_a_prime.status ∧ cost_model.status`)。
- DoD:对现存 ≥8 个违例任务(`L1_15`/`L2_17`/…)validator 报错;未注册 metamorphic check 报错;placeholder 题不因 kernel-export 误报;W4.3 后缺 golden 的 `upstream_oracle_required` 题被 validator 拦。

**W2.4 — 运行时防御接入** | gap C4+C5+C9 | 文件 `process_isolation.py`、`eval.py`、`adversarial.py`、`static_checker.py`、`backends/npu.py`、`cli.py`
- 改动:
  1. **child-side 检测**(child trusted-shim 段):profiler 检测 NPU kernel-event 存在性(`kernel_event_present`)、non-default stream(`stream_nondefault`)、**fallback 信号**(`fallback_signal` —— 无 triton/NPU kernel-event 即判 candidate 回退 torch 通用路径),均经 W1.0 上行;profiler NPU 实测部分留 `backends/npu.py` hook。
  2. `eval` 据上行诊断设 `runtime_status`/`performance_status`/`performance_flags`(计时仲裁失败的 `invalid_timing`/`quarantined`/`invalid_stream` 落位见 W1.3)。
  3. **metamorphic 与 canary 分离**:`run_metamorphic_checks`(对真实任务跑变换,填 per-task `metamorphic_results`)与 `check_canaries`(对 canary 陷阱任务判定,产出**独立的 canary findings**,schema 区别于 `metamorphic_results`)是两条路径。
  4. **metamorphic registry**:transformations registry 置于 `adversarial.py`,Phase 2 至少支持现有任务 `meta.yaml.runner_checks.metamorphic_checks` 已声明的全部 check 名(如 `differential_perturbation_general`);未知 check 的 validator 拒收规则在 W2.3 实现(`validate_tasks` 报错,非 runtime 静默跳过)。
  5. **canary 升级链**:`cli run-suite`/`analyze` 做 submission 级 canary 聚合,`canary_caught>0 ⇒` submission 标 `suspicious`(触发人工 review,DESIGN §6.8/§8)。
  6. `cli.py` `tab run`/`run-suite` emit 扩展:`round_result` JSON 序列化 `shape_results`、`metamorphic_results`、canary findings。
  7. Layer E `sys.addaudithook` 由 `static_checker` 暴露安装函数、`process_isolation` child 启动时安装。
- DoD:child 诊断信号(fallback/stream/kernel-event)端到端到达 `performance_status`/`performance_flags`;`metamorphic_results` 与 canary findings 各自独立进 `round_result` JSON;`canary_caught>0` 时 submission 标 `suspicious`(单测);现有 `meta` 全部 check 名在 registry 可解析、未知 check 被 validator 拒;addaudithook 安装函数有单测。

**W2.5 — canary 题库框架** | gap C3(框架) | 文件 `tasks/_canary/`、`adversarial.py`、`dataset.py`、`cli.py`、`scripts/validate_tasks.py`
- 改动:① 建 `tasks/_canary/` 目录结构与 1 个样板 canary(`identity` 类),固化 canary 任务 schema;② **canary discovery 路径** —— `iter_all_tasks()` 按设计跳过 `_` 前缀目录,canary 须由专用枚举器(`dataset`/`adversarial` 内)发现并加载,经独立路径喂 `check_canaries`,**不进入 score-eligible 集**(canary 不计主指标,§6.8);③ **canary→solution 绑定**:canary 不是独立提交项 —— 每个 canary 在其 `meta.yaml` 声明 `source_task_id`(指向一个真实 level1/2/3 任务),`run-suite`/`check_canaries` 复用 submission 中对应 `source_task_id` 的 `solution.py` 跑该 canary(观测 submitter 在陷阱上的行为),submitter 无需为 canary 单独写 solution;该映射入 canary schema,由 `dataset`/`adversarial` 解析、`cli` 串接;④ **`validate_tasks` 增 `_canary` 专用校验入口** —— 主校验只扫 `level1/2/3`,malformed canary 否则缺静态门禁,故为 canary schema(含 `source_task_id` 有效性)设独立静态校验。
- DoD:1 个样板 canary 经专用枚举器被发现、经 `source_task_id` 绑定到真实任务的 submitter solution、被 `check_canaries` 用该 solution 识别触发;断言 canary 未进 score-eligible 集;`validate_tasks` 对 malformed canary(含无效 `source_task_id`)报错。

**W2.6 — 存量 corpus 收敛(remediation)** | gap C8(收敛) | 文件 `tasks/level*/`、视裁定 `meta.yaml` / `oracle_pytorch.py`
- 改动:W2.3 的 validator 新规则会抓出存量坏数据 —— 已核实 ≥8 个(`L1_15`/`L2_17`/`L3_04` 等)`upstream_oracle_required` 任务的 `oracle_pytorch.py` 含 `F.silu`/`F.gelu`/`F.softmax` 等高层 op,违反 spec L133。须**逐题收敛**(不能停在「validator 报错」):每题判定 —— ① 改 oracle、展开高层 op 为底层算子;② 调整 `oracle_simplified` 分级;③ 若发现 spec L133 与 W4.3 离线 golden 设计存在张力(`upstream_oracle_required` 题真 golden 来自离线、其 `oracle_pytorch.py` 仅作 helper)则上报裁定后统一处理。排在 W2.3 之后、Phase 4 / release-gate 之前。
- DoD:W2.3 validator 对该规则全语料 0 报错;每题收敛方式有记录。

### Phase 3 — 长尾填充

| ID | gap | 内容 | 策略 | DoD |
|---|---|---|---|---|
| W3.1 | B1 | 100 题 `meta.yaml.cost_model` 公式 | 可机械近似算子(elementwise/reduce,~73 `formula_exact_manual` 主体)用 size-knob 通式批量起草;融合算子(flash/chunk-recurrent,~19 `upstream_oracle_required`)逐题人工推导 | 100/100 `meta.yaml` 含通过 `validate_tasks`(全语料)校验的 `cost_model`;分批提交,每批附人工 spot-check |
| W3.2 | C3 | `tasks/_canary/` 10 类陷阱题 | 逐类设计:identity / zero-out / cache-reuse / non-default-stream / torch-fallback / shape-route / lookup-table / async-no-sync / partial-compute / alibi-attention | 10 类齐备;`check_canaries` 对每类"作弊解"判定为 caught |

### Phase 4 — GPU 批处理(GPU 机)

**W4.1 — phase_a_prime freeze 实跑** | gap D1 | GPU 机
- 改动:在 100 题上实跑 `phase_a_prime` freeze,promote 后回写 `meta.phase_a_prime.{status,standalone_sha256}`。
- DoD:100/100 `meta.yaml` 含 `phase_a_prime` 字段;发布门可正常评估。

**W4.2 — cost-model GPU 交叉校验** | gap B2 | `scripts/phase_a_prime.py`(GPU 机)
- 改动:cost-model GPU 交叉校验**作为 `phase_a_prime.py` 内的一步**(非独立 sidecar,与 DESIGN §6.10/§7.2 一致):插桩 GPU-Triton 参考实测 FLOPs/bytes,与 `meta.cost_model` 解析值(复用 W1.2 统一 evaluator)逐 shape 比对,超 `gpu_xcheck_tol` ⇒ 写 `meta.cost_model.status=BLOCKED`,否则 `PASS`。
- DoD:`phase_a_prime` 产出每题 `cost_model.status` 并回写 meta;偏差任务正确 BLOCK。

**W4.3 — `upstream_oracle_required` 离线 golden + `task_status` carrier** | gap C6 | GPU 机 + `dataset.py`/`eval.py`/`scoring.py`/`cli.py`
- 改动:
  1. **golden 采集与契约**:19 个 `upstream_oracle_required` 任务离线在 GPU 跑冻结上游 kernel 采集 golden。**artifact 契约**:`golden/<shape_id>.safetensors`(public 与 hidden 同构、hidden 部分密封不进 public bundle);`dataset.py` 发现并纳入 `TaskDef`;golden 仅 runner(parent)读,**绝不进 candidate 净化挂载**(W1.0/W2.1);布局完整性由 W2.3 ③ 静态校验。
  2. **新增独立 `task_status` 字段**:task 级,取值 `evaluated` / `unsupported` / `blocked`,与 per-task `correctness/runtime/performance_status` **正交**(后者描述「评测了、结果如何」,`task_status` 描述「该任务在本模式下是否可评测」)。改动面:`eval.py` 设值;`scoring.py` 据它计分(`evaluated` 正常计,`unsupported`/`blocked` 不当 fail 压分);`cli.py` `validate-result` 校验该字段、`analyze` 据它处置;`analyze` 的 duplicate-merge worst-case 逻辑纳入 `task_status`;配套测试。
  3. **mode-aware golden 来源**:`eval.py` 按 `oracle_simplified`×`mode` 选 golden —— `upstream_oracle_required` 题 **verified 模式**加载离线 golden、缺 `golden/` ⇒ `task_status=blocked`(fail-closed,且经 W2.3 ③ 静态拦截);**self-reported 模式**无 provider golden ⇒ 该 19 题 `task_status=unsupported`;其余分级现跑 oracle。
  4. **analyze 分母 authority**:`analyze` 以 trusted suite JSON **顶层 `mode`** 为唯一 authority(不按 results 内容推断);**self-reported 模式分母 = repo-derived score-eligible 集 − 固定的 `upstream_oracle_required` 集合**(二者均由 repo `meta` 派生、写死,不按 results 里出现哪些 `unsupported` 动态扣减,杜绝结果内容驱动分母);**verified 模式分母 = 固定 `\|S\|`**(§9,不受 `task_status` 影响 —— verified 下这 19 题有 golden、正常评测)。
- DoD:19 题离线 golden 按契约布局齐备且不出现在 candidate 挂载视图(单测);verified 缺 golden ⇒ `task_status=blocked` 且被 W2.3 ③ 静态拦;self-reported 该 19 题 `task_status=unsupported` 且被 `analyze` 按固定公式排除出该模式分母;`validate-result` 接受 `task_status`;两模式均有测试。

## 7. 测试策略

- 每个 W 项配单元测试或端到端 smoke test;**0 skipped**(缺依赖须装齐)。
- M1 里程碑后跑全量 `pytest`,确认性能轴样板任务端到端绿。
- Phase 1-2 期间全语料 `validate_tasks` 预期 red(cost_model 未填满),用 `--task` 验证逻辑;W3.1 后转 green。
- Phase 4 后跑完整 `validate_tasks --release-gate`,目标 0 BLOCK。

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| dev 机无 NPU,B3 真实计时不可验证 | Phase 1 用 CPU 计时占位验证双沙箱隔离链 + per-trial 计时 + 上行链路;数值真实性 Phase 4 在 910B 机验证 |
| W1.0 协议改动影响面广(下游 5 个 W 项依赖) | W1.0 先做、先冻结协议 schema 并版本化;下游 W 项基于固定协议开发 |
| B1 100 题 cost_model 公式工作量大、易错 | 先样板任务端到端跑通(M1)再铺开;批量起草后 B2 GPU 交叉校验兜底纠错 |
| C3 canary 10 类设计需对抗思维 | 每类先写"作弊解"再写陷阱,确保 `check_canaries` 真能 caught |
| Phase 1/2 多个 W 项改 `eval.py`/`process_isolation.py` | 相关 W 项串行编辑 |

## 9. 交付物清单

- 代码:`process_isolation.py`/`inputs.py`/`eval.py`/`scoring.py`/`dataset.py`/`timing.py`/`backends/npu.py`/`static_checker.py`/`adversarial.py`/`cli.py`/`validate_tasks.py` 改动
- 新增:`infra/` 峰值表、`scripts/phase_a_prime.py` 内的 cost-model GPU 交叉校验步骤(**非独立 sidecar 脚本**)、`tasks/_canary/` 10 类、各 W 项测试
- 数据:100 题 `meta.cost_model`、100 题 `phase_a_prime` 字段、19 题离线 golden
- 文档:cost_model schema(标量/映射两形态)与 child↔parent 协议 schema 固化进 `specs/`
