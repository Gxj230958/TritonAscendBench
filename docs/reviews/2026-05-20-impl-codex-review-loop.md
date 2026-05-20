# Roofline + cost-model 实现 — Codex Review Loop (converged)

**Date:** 2026-05-20
**Target:** `feat/roofline-costmodel` 分支的实现(实施方案 `docs/plans/2026-05-20-roofline-costmodel-impl.md` 的 13 个 W 项 W0.1 / W1.0–W1.5 / W2.1–W2.6)
**Method:** 13 W 项由 serial/parallel subagent 实现(`/ce:work`);随后 codex 逐轮独立审实现 → 找问题 → subagent 批量修 → 再审。Claude 对每条 codex 反馈先评审合理性(项目决策/已知取舍优先于 codex)再采纳。收敛于 **R13 = ALL PASS**。

## 轮次记录

| Round | MUST-FIX | 摘要 |
|------:|:--------:|------|
| R1 | 4 | `performance_status` 无人设 ok / 标量 cost_model 接不上 scoring / event 计时通道未接 / validator 未 require cost_model |
| R2 | 4 | timing-pass 失败被忽略 / NPU event API 参数反 / `dtype_size` 未传入 Roofline / `analyze` 分母被 result 内容驱动 |
| R3 | 5 | `task_status` 恒 evaluated(W4.3 纯代码部分未做)/ analyze 无法派生 blocked / 抽样 seed 可预测 / 进程组未真 SIGKILL / profiler 绑 warmup 可规避 |
| R4 | 3 | offline-golden 重开 timing 作弊路径 / eval↔eligibility golden 判据不一致 / golden validator phase-gate 错 |
| R5 | 3 | profiler 对同一 input 额外跑一次 candidate(correctness+timing 两路)/ map-key 未静态校验 / provenance pair 跨 shape 拼接 |
| R6 | 3 | event hook 仍重复调用 candidate / child 进程 import 不到本 branch trusted helper(Layer E 静默失效)/ status precedence |
| R7 | 3 | offline 路径仍 import oracle / validate_tasks golden 判据未用共享谓词 / metamorphic 路径空转(有名无实) |
| R8 | 2 | verified-blocked 任务逃出固定分母 / sampled-output 校验 fail-open |
| R9 | 2 | `aggregate_perf` 残留 unavailable 短路 / 整表 `require_verified` 让无关 `[unverified]` 连累所有 task |
| R10 | 1 | data-gap 判定漏看 `roofline_data_complete`,重开 subset-escape |
| R11 | 3 | W2.6 corpus 收敛不彻底(R9-3 强化 validator 后抓到 W2.6 漏掉的 `tanh`/`softplus`) |
| R12 | 1 | oracle-op 检测漏 `torch.nn` module-form(`torch.nn.SiLU()`) |
| **R13** | **ALL PASS** | **收敛** |

总:13 轮,~37 个 genuine MUST-FIX(全部采纳并修复)。全量 pytest 一路绿:525 → 542 → 608 → 676 → 694 → 729 → 770 → 787 → 799 → 818 → 835 → 844 → 866 → 876 → 893 → 898 → 904(0 failed / 0 skipped 全程保持)。

## review 抓住的整合缝隙类型

W 项由独立 subagent 实现,跨 W 项的契约缝隙是 codex review 的主要价值:`performance_status` 无 W 项负责设 ok、标量 cost_model 与 scoring 接不上、event 计时通道未端到端接通、W4.3 的 task_status 三态纯代码未做、profiler/event 额外重复调用 candidate(重开 stateful 作弊)、Layer E audit hook 因 import 时序从未真正生效、metamorphic 防御空转、verified-blocked 逃分母 —— 这些都是"单 W 项功能上接通了、但跨 W 项有效性/安全性有缺口"。

## 采纳透明度

逐条评审:R11 的 `tanh`/`softplus` —— codex 选项含"放宽 validator",Claude 裁定与 W2.6 既定方向一致(禁所有复合激活、展开到基础 primitive),改 oracle 而非放宽 validator。其余均为真实 bug / 整合缺口,直接修复。

## Outcome

`feat/roofline-costmodel` 实现内部一致、防作弊闭环完整(profiler/计时/sampled-output/进程隔离/Layer E)、跨 W 项契约无缝隙,全量 pytest 904 passed。诊断(codex 7 轮)→ 方案(codex 11 轮)→ 实现(13 W 项)→ 实现 review(codex 13 轮)的完整闭环收敛。

**范围说明**:本批次 = Phase 0+1+2 的 dev 机代码骨架。Phase 3 长尾(100 题 cost_model 公式、10 类 canary)、Phase 4 GPU 数据产出(phase_a_prime freeze、cost-model GPU 交叉校验、19 题离线 golden)按计划在本批次范围外;另有 Task #19(task 去 `_kernel` 改名)、Task #20(算子参数合法性 gap)为待立项工作。
