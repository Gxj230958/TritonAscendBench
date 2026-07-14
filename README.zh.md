# TritonAscendBench (TAB)

> English: [README.md](README.md)

> 一个用于将 GPU Triton kernel 通过 `triton-ascend` 迁移到华为 Ascend 910B 的**迁移基准**（Migration Benchmark）。

[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE)
[![Tasks](https://img.shields.io/badge/Tasks-101-blue)](tasks/)

## 是什么

TritonAscendBench 评估一个 NPU triton-ascend 实现能否在华为 910B（Atlas A2 系列）上，在**正确性和性能**两个维度上，匹配来自 vLLM/SGLang 推理路径的 **GPU Triton 参考实现**。

与 [KernelBench](https://github.com/ScalingIntelligence/KernelBench)（PyTorch → CUDA 生成）和 [TritonBench](https://github.com/thunlp/TritonBench)（PyTorch → Triton 生成）不同，本项目是一个**迁移基准**：起点是来自 vLLM / SGLang 的*现有生产级 GPU Triton 代码*，目标是 910B 上的 NPU triton-ascend。

## 为什么

- **生成式基准**（KernelBench、TritonBench）测试的是"LLM 能否从零写出一个 kernel？"
- **迁移基准**（本项目）测试的是"一个真实的生产 kernel 能否在无回归的情况下迁移到 NPU？"
- 真实的工业场景：vLLM / SGLang 已经在 H100/A100 上以成熟的 Triton kernel 运行。将它们迁移到 910B 需要的是对上游语义的忠实，而非创造性。

## 快速开始

### 作为提交者（移植一个 kernel）

```bash
pip install triton-ascend-bench

# 1. Scaffold a solution dir for a specific task. The task_id is positional.
tab init-solution L1_12_rms_norm_kernel --out my_solutions/L1_12_rms_norm_kernel

# 2. Write your NPU triton implementation in
#    my_solutions/L1_12_rms_norm_kernel/solution.py
#    Also drop a manifest.yaml at the root (see ascendbench.manifest.SolutionManifest).

# 3. Run one task locally (positional task_id).
tab run L1_12_rms_norm_kernel \
    --solution my_solutions/L1_12_rms_norm_kernel/solution.py \
    --backend gpu \
    --out round_result.json

# 4. Run the whole suite (one solution.py per task expected at <root>/<task_id>/solution.py).
tab run-suite \
    --solutions-root my_solutions \
    --backend gpu \
    --out suite_results.json

# 5. Package and submit. `package` warns if manifest.yaml is missing.
tab package my_solutions/L1_12_rms_norm_kernel --out submission.tar.gz
tab validate-submission submission.tar.gz                     # schema-check the manifest
tab submit submission.tar.gz --endpoint https://leaderboard.example.com
```

### 作为基准维护者

任务包由一条分阶段、幂等的流水线构建：

```bash
# Phase 0: validate every task against the spec schema.
python scripts/corpus/validate_tasks.py --strict
python scripts/provenance/verify_version_matrix.py

# Phase 0.5: scaffold the 101-task tree from docs/kernel-select.md.
python scripts/corpus/generate_task_scaffolds.py        # 96 scaffolds (+5 PoC)

# Phase A: vendor the upstream GPU triton kernel.py (Apache-2.0/BSD-3
# attribution preserved; raw snapshot kept in ascendbench/frozen/snapshots/).
python scripts/verify/freeze_upstream_sources.py --limit 9999    # 93 frozen

# Phase B: author the PyTorch ground-truth oracle_pytorch.py.
python scripts/corpus/generate_oracle_from_kernel.py                 # 96 working refs

# Phase C: emit size-knob shapes.yaml + contract.md; the oracle input
# bridge is ascendbench.input_spec.inputs.build_inputs.
python scripts/corpus/generate_shapes_and_contract.py                   # 92 (+5 PoC migrated)
```

候选实现的**正确性评分器可在无 GPU/NPU 的环境下运行**：
`ascendbench.integrity.process_isolation` 在一个隔离子进程中执行提交的
`npu_triton_impl(inputs)`（仅用 safetensors 做 IPC；候选实现永远看不到 oracle 输出），
并由 `tolerance.compare` 按 shape 将其与进程内的 oracle 逐一对比评分。计时/性能
需要 910B 机器——参见下文 **Maturity**。

## 任务等级

| Level | Tasks | Type | Hardware |
|-------|-------|------|----------|
| **L1** | 36 | Single-op / elementwise / reduce / quant / sample | NPU Vector |
| **L2** | 35 | Fused / Cube matmul / LoRA / MoE EP / Mamba step | NPU Cube + Vector |
| **L3** | 30 | Full Flash Attention / Fused MoE / Mamba chunk pipeline | NPU full pipeline |

选取理由：参见 [选型理由](docs/zh/kernel-select.md)。

## 打分

这是一个 GPU-triton → NPU-triton 的**迁移**基准。**正确性是实时计分的轴**
（按 shape 用 `tolerance.compare` 对比 PyTorch oracle）。这里**没有 AscendC 原生基线**——
迁移要回答的问题是"该迁移是否相对 GPU-triton 原版保留了正确性/性能"，
而不是"triton-ascend 是否和手写原生一样快"。

性能指标（暂缓——需要 Ascend 910B）：

$$\text{FastP}(p) = \frac{|\{t \in \text{catalogue} : \text{correct} \land t_{\text{npu}} \le p \cdot t_{\text{baseline}}\}|}{|\text{catalogue}|}$$

- `t_baseline` 是该任务声明的性能参考（当前未实测；当硬件可用时报告原始 NPU 延迟）
- `FastP_native@1.0/1.5`、`capped_geomean_native`——保留函数/键名是为了输出
  schema 的稳定性；`_native` = NPU-triton 赛道（不是 AscendC）。上限封顶 5×，下限 ε=0.01。

完整计分规范：[计分规范](docs/zh/reference/scoring_spec.md)。

## 双层排行榜

| Tier | Verification | Use case |
|------|-------------|----------|
| **v0.1 self-reported** | Submitter's own measurements + hash check | Iteration, sandbox |
| **v1.0 verified** | Official 910B runner 100% re-execution + hidden shapes + sandbox + audit | Public ranking |

## 奖励作弊防御

横跨 7 个层面（Process / Code / Correctness / Sanity / Submission / Test-set / Cross-cutting）的 29 条设计内建（by-design）防御原则，覆盖 31 类攻击类别。

## 文档

按意图入门:

- **怎么用** → [使用指南](docs/zh/USAGE.md) — scaffold → 实现 → 评分 → 打包 → 提交,含 worked example。
- **原理** → [架构](docs/zh/ARCHITECTURE.md) → [设计](docs/zh/DESIGN.md) · [选型理由](docs/zh/kernel-select.md)。
- **深度参考**([docs/zh/reference/](docs/zh/reference/)):
  - *契约* — [Task API 契约](docs/zh/reference/task_api_contract.md)
  - *正确性* — [正确性模型](docs/zh/reference/correctness_model.md) · [容差策略](docs/zh/reference/tolerance_policy.md)
  - *计分* — [计分规范](docs/zh/reference/scoring_spec.md) · [时序仲裁](docs/zh/reference/timing_arbitration.md)
  - *完整性* — [奖励作弊防御](docs/zh/reference/reward_hack_defense.md) · [进程隔离](docs/zh/reference/process_isolation.md) · [AST 拒绝表](docs/zh/reference/ast_denylist.md)
  - *溯源* — [依赖模型](docs/reference/dependency_model.md)(英文;运行时为何不需 vLLM/SGLang) · [制品溯源](docs/zh/reference/artifact_provenance.md)

## 成熟度

**正确性**评分器在一台普通的 CPU 开发机上即可运行（无需 GPU/NPU），并已在
整个语料库上得到验证：一份提交会流经 static-check → 隔离子进程 →
`tolerance.compare` 对比 PyTorch oracle。**性能**轴（Roofline 效率）需要真实的
Ascend 910B 硬件，目前尚未端到端打通；在此之前，性能仅以原始延迟形式报告，
并被排除在发布门（release gate）之外。这里没有 AscendC 基线——这是一个针对上游
GPU-Triton 参考实现计分的迁移基准。

## 许可证

Apache 2.0 — 参见 [LICENSE](LICENSE)。

Vendored / inspired 来源：
- [KernelBench](https://github.com/ScalingIntelligence/KernelBench) (Apache 2.0) — eval / timing 模式
- vLLM (Apache 2.0) — 任务源 kernel
- SGLang (Apache 2.0) — 任务源 kernel

完整署名：[NOTICE](NOTICE)。

## 贡献

这是一个**仅提供方（provider-only）**的仓库。我们维护任务包、评测框架（harness）、基线和排行榜。我们**不**编写 NPU triton kernel 实现（那是提交者的职责）。

欢迎通过 GitHub issues 提交 bug 报告和改进建议。
