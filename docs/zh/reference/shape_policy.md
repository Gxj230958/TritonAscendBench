# Shape 覆盖策略

`shapes.yaml` 如何被填充、验证、以及为何有上限。单个 shape 行的*字段 schema* 见
[`task_api_contract.md`](task_api_contract.md)(`public_shapes` / `hidden_shapes` / `size_knobs` /
`inputs` / `param_constraints`);本文是*策略* —— 一个 task 应该带哪些 shape、每个 shape 凭什么留下、
以及为什么不是每个算子都能到同样的数量。

## 1. 目标:为迁移鲁棒性提供广覆盖

每个算子都应在**广而真实的 shape 范围**上被执行(目标 ~50,从原始 baseline 的 9 提升),这样
GPU→NPU 迁移后的 kernel 在真实负载会遇到的各种 shape 上,**正确性和性能**都鲁棒 —— 而不只是
一个 tiny/medium/large 三元组。

生成由 [`scripts/corpus/extend_task_shapes.py`](../../../scripts/corpus/extend_task_shapes.py) 完成:

- **Value ladder**(每个活跃 knob)—— 2 的幂 **加上** 真实的非 2 幂 LLM 维度
  (96、192、384、640、768、1280、1536…)再加少量奇数/缩放点,覆盖该 knob 的 `min` 到约
  4× 现有最大值,受 `DIM_CAP`(32768)和每张量 `ELEMENT_CAP`(16M)封顶,并遵守每个 knob 的
  `min` + `mod` 约束。
- **单轴 sweep** —— 每个 knob 在其完整 ladder 上遍历(其余固定在中位 base),让每个维度的整个范围
  都被覆盖。→ public shapes。
- **Coprime-stride 多轴网格** —— 在 ladder *笛卡尔积*上的低差异采样(每个轴按与其 ladder 长度互质
  的步长前进,因此没有轴会坍缩到子循环上),覆盖**交互**(小 B/大 D…)而不做盲目全积。→ public shapes。
- **Edges** —— all-min、all-max、长宽比极端、singleton。→ hidden shapes(密封的 A′ 验证)。

生成是确定性的(无 RNG):同一 task + 同一目标数永远产出同一候选集。

## 2. 每个 declared shape 必须真实 —— 双阶段真实性门

一个 shape 只有**同时**通过两阶段才允许进入 `shapes.yaml`。没有任何 shape 是凭信任声明的。

1. **CPU 预过**(生成时):`build_inputs` 物化该行、task oracle 在其上运行、`param_constraints`
   成立。oracle 连跑都跑不了的行,在写入前就被丢弃。
2. **GPU 权威 grade**(在一台真实 CUDA GPU 上):
   - **S2 severance** —— frozen kernel 必须能编译、能运行,且在 severed/original 边界上对该 shape
     **bit-exact**。
   - **正确性** —— 普通 task 用 **C2 oracle-equivalence**(kernel 输出在 task 容差类内与
     `oracle_pytorch` 对齐)。对 `upstream_oracle_required`(UOR)task(其 simplified oracle 不可作
     ground truth),验证改用 **golden capture**(把 kernel 的确定性输出捕获为参考)。

GPU grade 永远是权威;任何仅 CPU 的运行只是 triage(见 [`task_api_contract.md` §6](task_api_contract.md))。

## 3. 不许造假 —— 不能注水、不能剔难题、不能调参凑数

- declared-but-ungraded 的 shape 会触发 `SHAPE_COVERAGE_GAP`,带此状态的 task **永不 VERIFIED**。
  所以 `shapes.yaml` 不能用"实际过不了 grade"的 shape 注水。
- **绝不**为抬高通过率而松容差,**绝不**悄悄剔除难 shape。frozen kernel 拒绝的 shape 会被
  **quarantine**(移除),且**诚实可见**地记录原因 —— 不藏、不调参绕过。
- `phase_a_prime.shape_manifest_sha256` 对 parsed shape 集(id + dtype + knobs,保序)做哈希。
  **任何** shape 改动 —— 增、删、改 knob/dtype、重排 `inputs` —— 都改变哈希、使陈旧的 S2/C2 PASS
  失效,强制重新 grade。新鲜度是被强制的,不是假设的。

## 4. 诚实天花板 —— 不是每个算子都能到同样数量

一个算子能承载的*有效、已验证* shape 数,由现实而非投入决定:

- **Frozen kernel 输入契约。** kernel 是 byte-locked 的上游快照,不可编辑。很多要求 `tl.arange`
  的维度是 2 的幂、channel-last layout assert(`stride==1`)、或 block 整除。违反这些的 shape
  根本编译不过/跑不了 —— 没有合法办法承载它们。
- **内在形状空间。** 1–2 个 knob 的 elementwise 算子(如 `silu_and_mul`)根本不存在 50 个*互异、
  有意义*的 shape;硬造只是近似重复的注水,不是覆盖。
- **数值现实。** 非确定性 kernel(MoE top-k、部分 atomics)在某些 shape 上过不了 S2 bit-exact;
  大 bf16/fp8 reduction 与 oracle 的偏差超出容差。

因此原则是:**kernel 能诚实承载多少有效+已验证的 shape,就做多少,上限 ~50** —— 绝非"不择手段
凑 50"。当前 corpus 正反映这一点(多数算子 40–50;kernel 契约窄或形状空间小的算子上限更低)。

## 5. Quarantine 分类法 —— 某个 shape 为何被移除

| 原因 | 表现为 | 处理 |
|------|--------|------|
| 非 pow2 维度喂 `tl.arange` | S2 `original-no-run` CompilationError | 移除该 shape |
| layout / stride assert | S2 `original-no-run` AssertionError | 移除该 shape |
| 该 shape 上 kernel 非确定性 | S2 `compare` 非 bit-exact | 移除该 shape |
| 大 shape oracle 偏差超容差 | C2 `FAILED`(task → `MIXED`) | 移除该 shape |
| **allowlist 内** task 的大 shape dtype-ULP 偏差 | C2 `CEILING_ACCEPTED` | **保留** —— 按 ceiling grade,从不算精确 pass(见 `ascendbench/policies/ceiling_acceptance.yaml`) |

daemon 的批量 S2 在第一个被编译拒绝的 shape 处中止、**不报具体是哪个**;
[`scripts/corpus/gpu_shape_probe.py`](../../../scripts/corpus/gpu_shape_probe.py) 用同一套 S2-original
机制逐个隔离重跑,定位出问题的 shape id;
[`scripts/corpus/quarantine_shapes.py`](../../../scripts/corpus/quarantine_shapes.py) 精确(byte-surgical)
删除这些行并重写 meta 的 `shape_coverage` 计数。

## 6. 端到端流程

```
extend_task_shapes(生成 + CPU 验证 + 追加)              # corpus RED
  → S2 grade        → probe + quarantine kernel 拒绝的 → re-S2  (直到全 PASS)
  → C2 / golden     → quarantine oracle 发散的 shape   → re-S2 → re-C2  (直到 VERIFIED/CEILING)
  → validate_corpus_consistency GREEN + release-gate PROCEED
  → 从 GPU 机 commit shapes.yaml + daemon 写的 meta verdict  # corpus GREEN
```

> 英文权威版见 [`docs/reference/shape_policy.md`](../../reference/shape_policy.md)。
