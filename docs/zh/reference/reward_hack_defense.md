# 设计层面的奖励作弊(reward hacking)防御

> English: [../../reference/reward_hack_defense.md](../../reference/reward_hack_defense.md)

> 原则编号(`§5b.N`)是本目录自有的稳定编号。

## 核心原则

奖励作弊(reward hacking)防御必须**内建于基准设计之中**,而不是作为运行时检测器事后加挂。运行时检查是兜底层,而非主防线。

## 7-Layer Defense

| Layer | Purpose | Principles |
|-------|---------|-----------|
| L1 Process / Sandbox | 将候选实现与 oracle/golden/计时隔离开 | §5b.1 Oracle isolation, §5b.1a Safe output ABI, §5b.2 Runner-owned timing |
| L2 Code Enforcement | AST + import +产物溯源 | §5b.3 AST denylist, §5b.4 Artifact provenance |
| L3 Correctness Hardening | 超越裸 allclose 的 5 项子检查 | §5b.5 (input hash / sentinel / strict output / differential / NaN consistency) |
| L4 Sanity / Anomaly | 用绝对边界捕获明显的作弊 | §5b.6 Sanity bound, §5b.7 Profiler verdict, §5b.8 HW telemetry |
| L5 Submission Policy | 限流 / canary / 相似度 / 可复现性 | §5b.9–§5b.12 |
| L6 Test-Set Design | 隐藏封存 + 轮换 + 3-random 生成器 | §5b.13 |
| L7 Cross-Cutting | 缓存清洗 / 延迟披露 / 审计 | §5b.14–§5b.16 |
| **Addendum — Simplification Safety** | 安全的 CI 快速失败 + L1 oracle 分级 + 跨 oracle 漂移 + stdout 排除出裁决 | §5b.25–§5b.29 |

## 31-Category Attack Matrix

类别:

- **A1–A8** 正确性作弊(import oracle、identity、硬编码、缓冲区复用、NaN 滥用、容差拟合、输入篡改)
- **B1–B9** 性能作弊(async 不 sync、流切换、JIT 缓存滥用、预热预计算、空 kernel、计时 API 篡改、shape 路由、部分计算、频率操纵)
- **C1–C5** 提交 / Schema(哈希篡改、symlink、路径穿越、清单不一致、沙箱(sandbox)逃逸)
- **D1–D4** 测试集过拟合(多次提交探测、隐藏计时推断、对抗式过拟合、隐藏维度生日攻击)
- **E1–E5** LLM 特有(复制 kernel.py、专挑简单题、注释注入、跨任务污染、记忆背题)
- **F1–F10** Runner 边界(oracle 协同进程污染、原生产物洗白、有状态调用顺序作弊、CPU 回退、跨任务泄漏、sybil、侧信道、供应链、prompt 注入、输出契约滥用)

## 主动缓解映射（按攻击类型）

关键示例:

| Attack | Mitigation |
|--------|-----------|
| A1 Import oracle | Oracle 隔离:oracle_pytorch.py 绝不挂载到候选实现的沙箱(sandbox)中 |
| A3 Identity hack | 对 3 个随机化输入做差分(differential)+ canary 恒等陷阱任务 |
| A4 Hardcode lookup | Runner 保密的 RNG 输入 + canary shape 路由陷阱 |
| A5 Buffer reuse | 字节模式 sentinel 检测(软标记) |
| A8 Input tampering | 前后输入哈希比对 |
| B1 Async no-sync | 受信 shim,设备级 `torch.npu.synchronize()` |
| B2 Non-default stream | host_ms 为权威 + event 交叉校验,漂移 > 10% → invalid_timing |
| B5 Empty kernel | Profiler 检查 NPU kernel event 是否存在 |
| B7 Shape routing | 隐藏对抗集 + 隐藏生成器 3-random |
| D1 Multi-submit probe | 提交限流(3/week/task)+ sybil 检测 |
| F1 Oracle co-process pollution | 进程隔离:候选实现 / oracle / 比对分别在独立进程中 |
| F2 Native artifact whitewashing | 仅经 broker 的产物溯源 |

## 简化安全约束

在应用受 KernelBench/TritonBench 启发的简化时:

1. **CI 快速失败必须使用采样的数值 allclose,而非 stdout 文本**(§5b.25)
2. **L1 oracle 简化按 `oracle_simplified` 枚举进行**,而非一刀切(§5b.26)
3. **跨 oracle 漂移绝不自动把 PyTorch 裁定为真相**(§5b.27)
4. **隐藏生成器同时随机化 shape、value 与 edge**(§5b.28)
5. **候选实现的 stdout 仅作调试日志,绝不进入裁决**(§5b.29)

违反上述任何一条都会重新引入奖励作弊(reward hacking)的攻击面。
