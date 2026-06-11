# TritonAscendBench — 设计原理与打分模型

> English: [../DESIGN.md](../DESIGN.md)

> TAB 衡量什么，以及为什么这样衡量。面向机器的打分细节参见
> [`docs/reference/scoring_spec.md`](reference/scoring_spec.md)；本文档
> 是该 spec 所实现的设计原理。

TritonAscendBench（TAB）是一个**迁移基准**。每个 task 给你一个
取自真实 vLLM、SGLang 或 flash-linear-attention 推理路径的现有生产级 GPU Triton kernel，
并提出一个问题：*这个 kernel 能否经由 `triton-ascend` 移植到华为 Ascend 910B NPU，
而不损失正确性、也不让性能回退？* 整个语料库共 101 个 task，分布于三个
难度等级（L1=36，L2=35，L3=30）。

---

## 1. TAB 是什么 — 迁移命题

大多数 kernel 基准测试考察**生成**：给定一个 PyTorch 参考实现，模型
能否*从零写出*一个快速 kernel？

- [KernelBench](https://github.com/ScalingIntelligence/KernelBench) — PyTorch → CUDA 生成。
- [TritonBench](https://github.com/thunlp/TritonBench) — PyTorch → Triton 生成。

TAB 转而考察**迁移**。起点不是一个空白文件 —— 而是
*已经发布并已经正常工作的真实生产级 GPU Triton 代码*。vLLM 和
SGLang 如今已在 H100/A100 上运行着成熟、经过调优的 Triton kernel；
其工业级问题是：这些一模一样的 kernel 能否迁移到另一种加速器上
而不出现行为差异。因此被打分的对象是**保真度，而非
创造力**：当一份提交的 910B 移植版复现了上游
kernel 的数值行为、并跑满了 NPU 的 Roofline 时，它就是好的 —— 而不是当它发明出某种
巧妙的新算法时。

这一框定驱动了其他每一个设计决策：

- 参考标准是**上游 kernel 的可观测行为**，被捕获为一个
  PyTorch ground-truth oracle（参考实现），而不是一份手写的“理想”实现。
- **不存在 AscendC 手写基线**。比较的是“移植版 vs
  GPU-Triton 原版”，而非“triton-ascend vs 原生汇编”。（打分符号上的 `_native`
  后缀（例如 `capped_geomean_native`）仅出于
  输出 schema 稳定性而保留，它表示的是 *NPU-triton 赛道*，而非 AscendC。）
- 性能针对 **910B Roofline** 做归一化，因为源 GPU
  和 910B 在峰值能力上不同，且原始延迟不可跨硬件
  比较（§4）。

---

## 2. Task 选择与等级

全部 101 个 task 都是从生产推理路径中提取的真实 `@triton.jit` kernel
—— 绝非为本基准构造的合成 kernel。完整的选择
理由、逐 task 的来源表，以及 910B 硬件适配标准见
[`docs/kernel-select.md`](kernel-select.md)；要点如下：

**选择原则。** 仅限生产路径（训练反向的 `_bwd_*` /
`*_backward` kernel 被排除 —— 目标场景是推理 / agentic-RL
rollout）；相关算子的成对覆盖（GELU↔SiLU、ALiBi↔RoPE、
quantize↔dequant、vLLM↔SGLang 双源 state merge）；以及对
近乎相同的 fused-op 阶段变体做去重。

**910B 硬件适配。** 选择受限于 `triton-ascend` 实际
能表达什么、以及 910B 实际能*测量*什么。FP16 / BF16 / INT8
在 cube 和 vector 单元上都是原生支持的，因此保留。FP8 在 910B 上没有 cube/vector
算术（仅存储 + 软件转换），因此 FP8-cube-算术
算子被排除 —— 它们的性能无法在目标硬件上被诚实地测量 ——
仅保留单个 fused FP8 软件转换算子作为
正确性/压力测试代表。其他约束（192 KB Unified Buffer、
`tl.dot` 收缩维对齐、无 BF16 `tl.atomic_add`、无 async-copy /
pipeline / TMA）决定了哪些 fused 和 attention kernel 是可处理的。

**等级**跟踪迁移复杂度：

| Tier | Tasks | Shape | 910B units |
|------|------:|-------|-----------|
| **L1** | 36 | 单算子：bias / norm / activation / sampling / KV-index / quant | Vector |
| **L2** | 35 | Fused / cube matmul：LoRA matmul、W8A8 INT8 GEMM、MoE routing+permute、Mamba conv1d、gated-delta gating | Cube + Vector |
| **L3** | 30 | 完整子系统：Flash / Paged / Unified Attention、Fused MoE、Mamba / GDN / KDA chunk pipeline | Full pipeline |

难度随等级上升：L1 大多是 `medium`/`low`，L3 几乎全是
`high` —— chunk-recurrent + online-softmax fused kernel 是
`triton-ascend` 迁移真正的硬骨头。

---

## 3. 正确性模型

一个 task 只有在通过**两个独立的证明**时才算*正确*，每个证明
回答一个不同的问题。两者都是真实的、由代码强制执行的门 —— 而非
启发式判断。

### 3.1 Oracle 等价性 —— “kernel 算出的数学对不对？”

每个 task 都附带一个 PyTorch ground-truth oracle（`oracle_pytorch.py`）。
候选实现的输出由
`ascendbench.correctness.tolerance.compare` 按 shape 与 oracle 比较，
使用该 task 声明的容差类。容差策略（完整列表见
[`docs/reference/tolerance_policy.md`](reference/tolerance_policy.md)）是
dtype 感知的，而非一刀切：

- 浮点类先转换为 fp32 再运行 `torch.allclose(atol, rtol)`，先做一次
  抽样快速失败，随后是一次**权威的全张量检查**，使得 NaN /
  Inf 或落在抽样窗口之外的漂移无法蒙混过关
  （`tolerance.compare`）。
- 整数 / bool / RNG-sampling 输出默认为**逐位精确（bit-exact）** 的 `torch.equal`
  （`_bit_exact_compare`）—— 它们绝不能被 fp32 转换，否则会悄悄
  把例如 `int64(2^24+1)` 舍入为 `2^24`。
- 在桶边界处合法地向任一方向舍入的量化输出
  使用显式 ULP 容差：int8/int32 舍入用 `int_atol`
  （`_int_atol_compare`），fp8 e4m3 用 ordered-code ±1 ULP
  （`_fp8_ulp_compare`，它还会对 off-by-one 元素的*比例*设上限，
  从而仍能抓住系统性的舍入模式偏差）。

这道门以 `provenance.golden.oracle_kernel_equivalence` 记录在每个
task 上，并由谓词
`is_provenance_eligible`（`ascendbench/release_gate/eligibility.py`）遵守。当该状态为
`VERIFIED` 时一个 task 具有 oracle 等价性资格 —— 或为 `CEILING_ACCEPTED`，
即一个正确对齐的 task 在大尺寸下其精确容差结构上不可达、
从而进入 allowlist 的大 shape dtype-ULP 上限；或为一个其
提供方侧 `gpu_kernel_capture_status` 为 `DONE` 的 task —— **并且**其
`tolerance_source` 不是被推迟的 `migration_drift_budget` 路径。

oracle 锚点对自身的强度是诚实的：它是在一组抽样 shape 上
针对上游 kernel 的 wrapper 验证的，源代码
本身将其记录为*一个强经验锚点，而非形式化的不可能性
证明*。因此输入的区分能力（真正能把
正确移植与貌似正确但实则错误的移植区分开的 shape 分布）是语料库
设计的一部分，而非事后补丁。

### 3.2 逐位精确（bit-exact）切断验证 —— “提取出的 kernel 是否忠实于上游？”

每个 task 的 `kernel.py` 是上游源码的逐字节冻结快照。为了让它
可独立运行（脱离完整的 vLLM/SGLang 运行时），构建
pipeline 重写其 import 并切片其依赖闭包 —— 但它必须
**绝不**触碰数值逻辑。切断验证恰恰证明这一点：
它在 GPU 上并排运行*原始的*接入上游的 kernel 和*切断后的*独立
kernel，并断言它们是**逐位相同**的 ——

- 相同的输出 key 集合，且每个输出有相同的 `dtype` / `shape` / `stride` /
  `storage_offset`；
- `torch.equal` 逐位精确相等（无 fp32 转换逃逸，无 ULP 容差）；
- 相同的别名关系（哪些输出共享一个存储缓冲区）以及相同的
  in-place 输入变更；
- 确定性：切断侧被重跑数次，必须逐位复现
  自身的输出。

判定结果记录为 `phase_a_prime.status`。通过需要一次**真实的 GPU
运行** —— CPU / 解释器预筛绝不可能产生 pass，而一台没有
CUDA 的主机会返回诚实的“needs GPU”，而非伪造的成功。如果
切断后的 kernel 引入哪怕一位偏差，或泄漏了一个上游
运行时 import，验证就会 block。

它证明了什么、没证明什么，在设计上是显式的：它是一个强
保证，即*切断在所列举的 shape、别名关系和变更面上没有引入可观测的变化*
—— 这是抵御移植悄悄
偏离上游的最难一道门。它**不是**对上游
kernel 本身正确性的独立证明；那是上面 oracle 等价性门的工作。
两者是有各自标准的独立阶段，一个 task 必须同时满足两者。

### 3.3 “PASS” 意味着什么

一个 task 当且仅当它在 oracle 对齐的合格集上同时通过*两个*正确性证明时，
才是**可发布（release-ready）** 的。复合谓词是
`release_ready_with_provenance` =
`release_status_pass`（切断 / `phase_a_prime.status == PASS`）**AND**
`is_provenance_eligible`（oracle 等价性）。关键在于，**`cost_model.status`
不是这道门的一部分**（§4.3）—— 可发布性是一个*正确性*
陈述。

### 3.4 独立的、非 GPU 的正确性路径

候选正确性评分器在**没有任何 GPU 或 NPU** 的情况下运行。
`ascendbench.integrity.process_isolation` 在一个隔离子进程中执行
提交的 `npu_triton_impl(inputs)` —— 仅用 safetensors 的 IPC，子进程从不
看到 oracle、oracle 输出、`golden/` 或
`hidden_shapes` 区段 —— 而 `tolerance.compare` 将每个 shape 与
进程内 oracle 比较打分。这使得正确性信号在一台普通开发
机上可复现，独立于硬件可用性。

硬件权威性叠加在该路径之上：**对提交移植版的提供方侧
验证运行在 910B NPU 上**（真正的迁移
目标，正确性与性能在那里都被确认），而 GPU 仅
用于**离线**，以建立 oracle 等价性基线和
锚定语料库的切断交叉校验。开发机或 GPU 的判定
绝不替代 NPU 的判定。

---

## 4. 打分模型

打分在一个**固定**的可打分集 `S`（单一权威：
`ascendbench/release_gate/eligibility.py`）上有两个目标。`|S|` 绝不会为了
美化一份提交而缩小 —— 失败的 task 留在分母里（无子集逃逸）。

### 4.1 目标 1 —— 正确性覆盖门

```
native_coverage = #{ t ∈ S : correctness_status_t == ok ∧ ¬fallback_t } / |S|
```

排名以 `native_coverage ≥ 0.70` 为门
（`MIN_COVERAGE_FOR_RANKING = 0.70`，`scoring.is_rank_eligible`）。一个被检测到的
PyTorch fallback 会在打分*之前*被归一化为一次正确性失败，因此一个
“正确但暗中退回到 eager PyTorch”的解无法通过该
门（`native_coverage` 保留 `¬fallback` 这个合取项作为第二道
防线）。覆盖率上没有 ε：失败或缺失的 task 对分子贡献 0，但
**不会**从 `|S|` 中移除。

### 4.2 目标 2 —— Roofline 归一化的性能（被排名的指标）

因为源 GPU 和 910B 在峰值能力上不同，绝对延迟
不可跨硬件比较。性能转而是一个**相对 910B
Roofline 的效率**。对每个 shape，逐 task 的成本模型
（`ascendbench/perf/cost_model.py`）给出按计算类的 FLOPs `W_c` 和
按资源的字节数 `M_r`；打包的 peak table
（`ascendbench/policies/peak_table_910b.toml`）给出峰值算力 `π_c` 和峰值
带宽 `β_r`。理论延迟下界是标准的 Roofline 下
界（`scoring.compute_t_ref`）：

```
t_ref = max( max_c W_c/π_c ,  max_r M_r/β_r )
```

每个 shape 的效率是 `e = min(P_CAP, t_ref / t_npu)`
（`scoring.shape_efficiency`）。理论将 `e` 界定在 `(0, 1]`，因为 `t_ref` 是一个
下界；`e > 1` 意味着测得的时间落到了理论下界之下（一个
成本模型或计时假象），并被钳制在 `P_CAP = 5.0`。task 级的
分数是其各 shape 效率的几何平均，任何失败都以 ε 填充：

```
s_t = (∏_s e(t,s))^(1/n_t)   if all statuses ok and no fallback
s_t = ε = 0.01               otherwise
```

而语料库的总体头条指标是固定 `|S|` 上的 capped 几何平均：

```
capped_geomean_native = exp( (1/|S|) · Σ_{t∈S} log s_t )
```

`compute_t_ref` 是**fail-closed** 的：一条引用了 peak table 未覆盖的计算类或
资源、或引用了非正 / 非有限峰值的公式
会 raise，而不是在坏数据上打分。成本模型公式是经由
一个共享的 AST 白名单求值器求值的尺寸-knob
表达式 —— 没有 `eval`，没有属性访问，没有任意调用（`cost_model.evaluate_formula`）。

`s_t ≈ 1` 意味着 910B Roofline 被跑满。`RooflineP@τ`（`S` 中
`e ≥ τ` 的比例）是一个诊断量，而非排名键。示意性的 910B 峰值
（HBM ≈ 1.6 TB/s，FP16 cube ≈ 256 TFLOPS，INT8 ≈ 512 TOPS）权威地存于
打包的 peak table 中。

**设备侧计时。** 为了让 `e` 成为一个良定义的 Roofline 效率，`t_npu`
必须反映真实的设备工作。权威的计时器是 host 侧的
`time.clock_gettime(CLOCK_MONOTONIC_RAW)`，每次被测调用都由一个 backend stream 的 `synchronize()`
包夹（`ascendbench/perf/timing.py`）：
`torch.npu.Event.elapsed_time` 只看到它自己的 stream 而错过跨 stream
的间隙，因此 NPU event 计时器被保留为**次级交叉校验**，并相对
host 时间做仲裁（`arbitrate_timing`），绝不用作主计时。`n_repeat`
次被测试验中的每一次都运行在一个**全新**的逐次输入上（warmup 试验
排除在外），因此候选实现无法跨重复缓存一个答案，而 warmup 吸收
JIT 编译开销。host 与 event 之间的大幅漂移连同
非默认 stream 信号会将该次试验标记为无效。原理见
[`docs/reference/timing_arbitration.md`](reference/timing_arbitration.md)。

### 4.3 成本模型只是证据，而非门

成本模型存在的**唯一**目的是构建 `t_ref`。它与
一个被插桩的 GPU-Triton 参考实现的逐 shape 一致性（`cost_model.status`）是 ncu 证据支撑的
性能/访问元数据 —— *而非*发布阻断器。一个可移植的解析式字节
公式在结构上无法在每个 shape 上命中一个紧的逐 shape 容差：
条件访问（例如 `HAS_Z` 分支）、L2 复用、tile 几何，以及 HBM
突发下界都会让真实流量偏离闭式计数。把
`cost_model.status` 强行塞入发布门，要么会因为一个
性能模型的技术细节而把发布钉成红色，要么会推向不诚实的逐 shape 拟合。
因此它的状态*作为元数据被记录并披露*，而正确性门
（§3）独立成立。

### 4.4 发布资格 —— 可打分 ∩ oracle 对齐

可发布集由**一个**权威计算，
`compute_release_ready_eligible` =
`compute_score_eligible(tasks) ∩ { t : is_provenance_eligible(t.meta) }`。
`cli analyze` 和语料库发布门都调用它，因此合格集不会在
不同调用点之间漂移。

- `compute_score_eligible` 丢弃那些无法被公平打分的 task —— 一个 stub
  oracle（`raise NotImplementedError`）、一个 `PHASE-C-MANUAL` shapes 文件，或一个
  被算子标记为 `release_scope.excluded` 的 task —— 因此不可评分的 task 不会
  压低有效的提交。语料库仍然包含它们；只有*被打分的
  分母*减去它们。
- 与 `is_provenance_eligible` 的交集加上 oracle 等价性
  要求。

在该集合内，一个 task 若其复合 `release_status != PASS` 则阻断发布，
且一个被阻断的 task **不能**通过从 `|S|` 中被丢弃来逃避门 —— 它
留在分母里并打 ε 分。两条模式感知的分母规则保持这一点
诚实（`scoring.py`）：

- 一个在 self-reported 模式下没有提供方 golden 的 `upstream_oracle_required`
  task 是 `unsupported`：完全在被打分集之外（从分子*和*
  分母中都丢弃），因为不存在公平的比较。
- 一个 verified 模式下其离线 golden 缺失或损坏的 task 是 `blocked`：
  它**留在**固定的 `|S|` 中并作为失败打分（ε）。Blocked 是
  发布门的目标，绝不能逃出分母。

---

## 5. 反作弊姿态

因为分数正是提交者会去优化的对象，所以测试框架假设一个
对抗性提交者并进行纵深防御。在高层面上：

- **进程隔离。** 候选实现运行在一个单独的子进程中，该进程从不
  与受信任的 oracle 共享解释器状态或模块缓存，仅经由 safetensors
  通信，且从不接收 oracle、golden 输出、task
  目录或 hidden-shape 区段。正确性与计时在两个*独立*的
  子进程 pass 中评分，第一个 pass 的进程组被杀掉，并在两者之间
  清洗 `/tmp` + Triton 编译缓存，因此没有预计算
  通道能存活（`ascendbench.integrity.process_isolation`）。
- **静态 import / AST 检查。** 提交的 `solution.py` 通过一道分层静态
  防御 —— import 黑名单、一个正向的 AST 节点类型白名单、
  call/attribute 点分名列表、一个安装在隔离子进程内部的运行时审计 hook，以及
  一个源字符串黑名单（`ascendbench.integrity.static_checker`）—— 因此一个
  解无法触及 oracle、文件系统或网络。
- **存在 canary。** 语料库包含被设计用来抓住一份
  在玩弄 runner 而非计算算子的提交的 canary task；一个
  被触发的 canary 会升级整份提交。检测规则是
  有意**不**在此处记录的。

§4 中固定分母、无子集逃逸的纪律本身就是一条反
作弊属性：提交者无法通过让难 task 消失来改善
头条数字。

---

## 6. 路线图（尚未实现）

以下内容是**已设计但尚未构建**的，在此列出以免它们
被误认为是已发布的行为：

- **加固的 verified-runner 沙箱。** 已发布的进程隔离模型
  在一台开发机上强制执行 host 无关的、安全关键的属性。
  完整的加固模型 —— 容器 digest pinning、seccomp/cgroup 约束，以及
  910B 验证机上的 NPU 设备隔离 —— 是未来工作。
- **公开排行榜与提交 broker。** 两级
  （self-reported / verified）公开排行榜、提交端点，以及
  将仲裁官方 910B 重新执行的签名工件 broker 都尚未
  运转。
- **更强的正确性锚点。** 多 seed 和基于属性的输入扫描，
  以及为 oracle 锚点提供逐 task 披露的强度标签（所用 shape、
  容差类、逐位精确性），都计划用来把输入区分能力
  提升到当前单 seed 物化之上。

---

## 另见

- [中文 README](../../README.zh.md) — 快速开始、项目状态。
- [`docs/kernel-select.md`](kernel-select.md) — task 选择与完整语料库表。
- [`docs/reference/scoring_spec.md`](reference/scoring_spec.md) — 面向机器的打分细节。
- [`docs/reference/tolerance_policy.md`](reference/tolerance_policy.md) — 容差类。
- [`docs/reference/correctness_model.md`](reference/correctness_model.md) — 精度对齐打分标准。
- [`docs/reference/task_api_contract.md`](reference/task_api_contract.md) — task bundle schema。
- [`docs/reference/timing_arbitration.md`](reference/timing_arbitration.md) — 设备计时方法论。
