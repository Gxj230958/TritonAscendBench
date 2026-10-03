# TritonAscendBench — 架构

> English: [../ARCHITECTURE.md](../ARCHITECTURE.md)

TAB 是如何构建的，以及一次评测的流程：这是
[README](../../README.zh.md) 的"工作原理"配套文档。

TAB 是一个 **迁移基准**。每个任务取一个真实的生产环境 GPU Triton
内核（来自 vLLM / SGLang / flash-linear-attention），并提出这样的问题：它能否
在不损失正确性或性能的前提下，通过 `triton-ascend` 移植到华为 Ascend 910B NPU？该语料库共有
**101 个任务** —— L1 有 36 个（单算子 / elementwise /
reduce / quant），L2 有 35 个（融合 / matmul / MoE / 状态空间步进），L3 有 30 个
（完整 Flash Attention / 融合 MoE / 分块流水线）。

有两个事实塑造了整个设计：

1. **正确性是实时、计分的维度。** 提交的内核会按 shape（形状）逐一
   针对一个可信的 PyTorch oracle（参考实现）在一套容差策略下评分。这条路径
   在 **无 GPU 且无 NPU** 的情况下运行 —— 它是任何人都能在本地复现的部分。
2. **两台机器，两种角色。** 提交者与离线正确性评分器
   在普通商用硬件上运行。*Verified*（已验证）的性能在 910B NPU 上测量；
   GPU 仅由提供方用于离线的 oracle 等价性
   交叉核对，绝不作为迁移目标。

---

## 1. 仓库布局

```
triton-ascend-bench/
├── tasks/                  # the corpus — level1/ level2/ level3/, one dir per task
├── ascendbench/            # the Python package (harness, grader, scoring, loaders)
├── scripts/                # provider-side corpus-construction pipeline
├── docker/                 # Dockerfile.gpu (oracle cross-check) / Dockerfile.npu (910B)
└── docs/                   # this file, reference specs, kernel-select rationale
```

### 1.1 任务目录：五文件解剖

每个 `tasks/level{1,2,3}/<task_id>/` 恰好是五个计分文件。schema
校验器（`scripts/corpus/validate_tasks.py`，`REQUIRED_FILES`）检查这套文件
是否齐全；加载器（`ascendbench/corpus/dataset.py::load_task`）将
`meta.yaml` / `shapes.yaml` / `contract.md` / `kernel.py` 解析为一个 `TaskDef`
（`oracle_pytorch.py` 单独加载，在 runner 侧，绝不交给候选实现）。

| 文件 | 角色 |
|------|------|
| `kernel.py` | **冻结的上游 GPU Triton 参考实现** —— 迁移的 *源*。一个 vendored（内嵌）的 `@triton.jit` 内核，保留了原始的 Apache-2.0/BSD 署名。以 SHA 锁定（`TaskDef.kernel_py_sha256`）。它是提交者阅读的移植目标；在候选实现评分期间它 **不会** 被执行。 |
| `meta.yaml` | 所有任务元数据：`task_id` / `level`，`upstream` 出处（repo、`commit_sha`、`file_sha256`、license）、`interface` 别名映射与可变性契约、`tolerance_class`、`oracle_simplified` 层级、`cost_model`、`param_constraints`（算子输入契约）、`runner_checks`（静态检查、变形检查、shape 覆盖），以及基线信任链记录（`provenance.golden`、`phase_a_prime`）。 |
| `shapes.yaml` | `size_knobs` 默认值、`public_shapes`、`hidden_shapes`，以及一个 `inputs:` 块。每个输入是一个由 knob 参数化的张量规格（`dims` / `dtype` / `dist`）或一个非张量标量。一个 inputs 块覆盖每一个 shape 行。由 `ascendbench.input_spec.inputs.build_inputs` 消费。 |
| `contract.md` | 人类可读的契约：待实现的 `solution.py` 签名、禁止的捷径、输入表、必需语义、容差类，以及变形不变量。 |
| `oracle_pytorch.py` | **PyTorch 标准答案**：`run(inputs: dict) -> dict[str, Tensor]`。与设备无关（在 CPU 上运行）；既不依赖 triton，也不依赖 CUDA 或 NPU。 |

例如，`tasks/level1/L1_12_rms_norm/` 移植了 vLLM 的 `_rms_norm_kernel`：其
`oracle_pytorch.py::run` 以 float32 计算 `x / sqrt(mean(x²)+eps) * weight`，
`shapes.yaml` 在 knob `D`（hidden_size）上声明了 5 个 public + 4 个 hidden shape 行，
而 `meta.yaml` 钉定 `tolerance_class: vector_reduce`（atol/rtol
1e-3）外加两项变形检查（weight-identity、row-independence）。

Canary 陷阱任务存放在 `tasks/_canary/` 下。加载器的 `iter_all_tasks()`
按设计会跳过每一个以 `_` 为前缀的目录，因此 canary 永远无法进入
计分合格的分母；它们只能通过专门的
`iter_canary_tasks()` 枚举器被列出。

### 1.2 `ascendbench/` 包

| 子包 | 内容 |
|------------|---------------|
| `corpus/` | `dataset.py` —— `TaskDef`、`load_task`、`iter_all_tasks`、canary 发现；`provenance.py` —— 基线信任链的类型化视图。 |
| `input_spec/` | `inputs.py` —— `build_inputs`（从 `shapes.yaml` 物化一个 shape 的输入）、用于 IPC 的 `split_inputs`/`merge_inputs`；`expr.py` —— `eval_numeric`，一个 AST 白名单化的 knob 公式求值器（绝不裸用 `eval`）；`param_constraints.py` —— 算子输入契约 + `resolve_task_status`。 |
| `integrity/` | `process_isolation.py` —— 隔离的候选实现 runner；`static_checker.py` —— AST 允许/拒绝扫描 + 运行时审计钩子；`adversarial.py` —— canary 与变形检查；`wrapper_contract.py`。 |
| `correctness/` | `tolerance.py` —— `load_policy` + `compare`；`golden_store.py`；`determinism.py`。 |
| `engine/` | `eval.py` —— `evaluate()`，逐任务编排器；`cli.py` —— `tab` 命令行接口。 |
| `perf/` | `scoring.py` —— `capped_geomean_native`、`compute_t_ref`、Roofline 效率运算；`cost_model.py`；`peak_table.py`；`timing.py`。 |
| `backends/` | `get_backend(...)` → `GpuBackend` / `NpuBackend`：设备初始化、事件计时器、内核事件 profiler、编译预热策略。 |
| `release_gate/` | `eligibility.py` —— `compute_release_ready_eligible`、`is_provenance_eligible`、`release_status_pass`；`sanity_gate.py`。 |
| `frozen/` | vendored 的上游 `snapshots/`，外加用于让冻结的 `kernel.py` 独立运行的 `deps/` 与 `shim/`。 |
| `policies/` | `tolerance_policy.yaml`（容差类）与 `version_matrix.toml`（锁定的数值栈）。 |

此外还有 `manifest.py`（`SolutionManifest`）、`triton_compat.py`，以及
`trusted_shim/`、`upstream_shims/`、`_util/` 中的辅助模块。

### 1.3 `docker/`

- `Dockerfile.gpu` —— 离线 oracle 等价性交叉核对主机（H100/A100）。
  基础镜像是一个必需的 `--build-arg`，这样构建就不会悄无声息地落在
  错误的数值栈上。
- `Dockerfile.npu` —— verified 的 910B runner（Ascend CANN 基础镜像；锁定的
  `torch-npu` + `triton-ascend`）。

---

## 2. 语料库构建（提供方侧，离线）

构建一个任务是一条分阶段、幂等的流水线。它只在
基准被 *编写* 时运行 —— 绝不在提交者评测期间运行。

```
freeze upstream kernel ─▶ standalone-severance verify ─▶ author PyTorch oracle
        ─▶ generate shapes + contract ─▶ schema-validate ─▶ oracle-equivalence
        cross-check ─▶ capture golden
```

**1. 冻结上游内核。** `scripts/verify/freeze_upstream_sources.py`
将生产环境的 `@triton.jit` 内核 vendor 进 `kernel.py`，并记录
上游 repo、commit、文件 SHA 与 license。原始快照也保存在
`ascendbench/frozen/snapshots/` 下。

**2. 使其独立运行，然后逐位精确地验证。**
`scripts/verify/standalone_freeze_verifier.py` 执行符号级
可达闭包分析：冻结内核触及的每一个外部符号都会被
分类（vendor 它、将其改写为公开的 Triton import、用一个可证明无副作用的 shim 替换它，或者 —— 如果它无法变得确定且独立 ——
**BLOCKED**，这是一道硬门，绝不是悄无声息地丢弃）。随后会对切断后的内核
针对原始上游调用验证 **逐位精确(bit-exact)等价性**
（数值，加上别名与被改写输入的行为）。该验证的
权威通过需要真实 GPU；在纯 CPU 的机器上它会诚实地报告
"needs GPU"，而非声称通过。结果记录在 `meta.yaml`
的 `phase_a_prime` 下（状态、GPU 运行 id，以及独立内核的 SHA）。

**3. 编写 PyTorch oracle。**
`scripts/corpus/generate_oracle_from_kernel.py` 对冻结内核做 AST 解析，
提取真实的（非指针、非 constexpr）张量参数列表，并
用一个类型化的 `run(inputs)` 脚手架生成 `oracle_pytorch.py`。对于定义明确的
数学（norm / activation / gemm 系列），它直接写出参考实现并
将任务标记为 `oracle_simplified: simple_pytorch_ok`。三个 oracle 层级：

| `oracle_simplified` | 含义 | Golden 来源 |
|---------------------|---------|---------------|
| `simple_pytorch_ok` | 一个直接的 PyTorch 参考实现在容差内匹配。 | 进程内 oracle |
| `formula_exact_manual` | 一个手写的闭式参考实现。 | 进程内 oracle |
| `upstream_oracle_required` | 不存在闭式 oracle；标准答案只能来自运行冻结的上游内核。 | 离线 `golden/` 制品（仅 verified 模式） |

**4. 生成 shapes 与 contract。**
`scripts/corpus/generate_shapes_and_contract.py` 输出 knob 网格、
public/hidden shape 行、`inputs:` 分布 DSL，以及人类可读的
`contract.md`。`inputs.build_inputs` 桥接器使 `shapes.yaml` 可执行 ——
oracle 由此可以在任意 shape 行上运行。

**5. 校验。** `scripts/corpus/validate_tasks.py --strict` 对照
[`docs/reference/task_api_contract.md`](reference/task_api_contract.md) 中的 schema 检查每个任务：
五个文件齐全、必需的 `meta.yaml` 字段、一个已知的 `tolerance_class`、
一个可解析的 `@triton.jit` 内核、一个导出 `run` 的 `oracle_pytorch.py`、
cost-model 公式可求值，以及 `param_constraints` 形态。数据集
加载器保持宽松（这样畸形任务仍能加载并抵达门）；
**齐全性与 schema 在此处强制**，而非在加载器中。

**6. Oracle 等价性交叉核对 + golden 采集。** 在 GPU 主机上，
`scripts/oracle/` 运行冻结内核并在任务的容差类下将其与 PyTorch oracle
比较，将每个 shape 的判定
（`VERIFIED` / `CEILING_ACCEPTED`）记入 `meta.yaml` 的 `provenance.golden`。
`scripts/provenance/`（`capture_golden_baselines.py`、`write_golden_manifest.py`、
`verify_upstream_commit_chain.py`、`verify_version_matrix.py`）为 `upstream_oracle_required` 任务采集
离线 golden 制品，并写出 evaluator 后续会重新核验的
SHA + 签名清单。

---

## 3. 评测流程

`ascendbench/engine/eval.py::evaluate(task_id, solution_path, backend, mode)` 是
逐任务编排器。一次调用对一个 `(task, solution, backend)` 三元组评分。

```
load_task ─▶ static checks ─▶ build inputs ─▶ run candidate in isolation
          ─▶ grade vs oracle / tolerance ─▶ score
```

**解析任务。** `load_task` 将五个文件解析为一个 `TaskDef`。`mode`
选择评测 shape 集合：`self-reported` 只对 public shapes 评分；
`verified` 对 public **和** hidden shapes 评分。可信父进程始终持有
完整的 `shapes.yaml`；候选实现从不持有（见下方隔离），因此
对 hidden shapes 评分永远不会泄漏隐藏的 knob。

**静态检查。** `static_checker.check_solution` 对提交的源码运行任务声明的
检查（AST 拒绝名单、import 允许名单、constexpr 匹配）。一个被拦截的解
在正确性上失败，且根本不会被运行。

**构建输入。** 可信父进程通过
`inputs.build_inputs(shapes_doc, shape_id, seed)` 物化每个计分 shape 的输入。输入按
shape/trial **惰性逐个构建**（`run_two_pass_lazy` 在每次隔离扫描内部流式执行
构建 → 序列化 → 释放），因此父进程的主机峰值内存保持在 O（单个 shape 的
一份输入）而非 O（全部计分 shape × trial 数）—— 大 shape 任务不再因物化
阶段撑爆主机 memcg 而 OOM。正确性扫描得到
每个 shape 一个 dict（seed 0）；计时扫描得到一个逐 trial 序列 —— 预热
trial 之后是被测 trial，每个都是一个在不同 seed 下的 **全新** 输入，
这样候选实现就无法在多个 trial 之间缓存某一个答案。

**在隔离中运行候选实现。** `integrity/process_isolation.py` 绝不
在可信进程中执行候选代码。提交的
`npu_triton_impl(inputs) -> dict` 在一个独立子进程中运行，并具备以下
与主机无关的保证：

- **两次独立扫描。** 一次 *correctness* 扫描，随后 —— 在对整个进程组执行 SIGKILL
  外加清洗 Triton 编译缓存、`/tmp` scratch
  与设备内存之后 —— 一次独立的 *timing* 扫描。该清洗关闭了
  correctness→timing 的预计算通道。
- **仅 safetensors 的 IPC。** 输入向下流、输出向上流，均为 safetensors
  载荷外加 JSON 旁路文件。**无 pickle**，因此加载候选实现的输出
  无法执行代码。
- **无 golden 泄漏。** 子进程只收到 `solution.py` 外加
  runner 物化的输入 —— 绝不收到任务目录、oracle、`contract.md`、
  `golden/` 制品或 shapes 文档。由于它拿到的是
  物化的输入而非 shapes 文件，hidden shapes 在
  候选实现的视野中结构性地缺席。
- **运行时审计钩子。** 在子进程内，候选模块加载之后，一张
  `sys.addaudithook` 网（`static_checker.install_runtime_audit_hook`）拦截
  静态 AST 扫描看不到的动态 import / `exec` / subprocess / `ctypes` 逃逸。
  可信辅助模块在钩子生效 *之前* 就已被 import 并缓存；
  一个短暂的 "compile" 阶段允许一次性的 Triton JIT 预热。
- **以主机为权威的计时 + 抽样重核验。** 计时扫描用
  `CLOCK_MONOTONIC_RAW` 为每个被测 trial 计时。一个 runner 私有的 RNG
  （每次评测从 `secrets` 重新播种，与输入 seed 不同）
  挑选一个 trial 子集，其输出被传送上来；父进程在那些 trial 的
  输入上重新运行 oracle 并比较 —— 由此关闭
  "在正确性上诚实、在计时上快但错"的作弊路径。

**针对 oracle 评分。** 对每个 shape，父进程获得 golden
输出，并用 `tolerance.compare(actual, expected, tol_cls).passed` 将候选实现的
correctness 扫描输出与之比较，其中 `tol_cls` 来自
`tolerance.load_policy()[task.tolerance_class]`（完整的 dtype 分派
以及 C2/S2/ceiling 标准见
[`docs/reference/correctness_model.md`](reference/correctness_model.md)）。golden 来源由
`_resolve_golden`（oracle 层级 × mode）路由：

| 任务层级 | Mode | Golden | `task_status` |
|-----------|------|--------|---------------|
| `simple_pytorch_ok` / `formula_exact_manual` | 任一 | 进程内 PyTorch oracle | `evaluated` |
| `upstream_oracle_required` | self-reported | 无 | `unsupported` |
| `upstream_oracle_required` | verified，golden 完整 | 离线 `golden/<shape>.safetensors`（SHA/签名链重新核验） | `evaluated` |
| `upstream_oracle_required` | verified，golden 缺失 | 无（fail-closed） | `blocked` |

一个悄悄回退到通用 torch 路径（不产生任何 NPU
内核事件）的候选实现会被归一化为正确性 **fail**，即使数值上
正确 —— 一次迁移必须真正在设备上运行一个内核。

**评分。** 正确的 shape 喂入 `perf/scoring.py::capped_geomean_native`。性能
模型（`compute_t_ref` / Roofline 效率）需要 verified 的 910B
峰值，且是 **仅作证据用** 的 —— 见 §3.1 与
[`docs/reference/scoring_spec.md`](reference/scoring_spec.md)。计时仲裁
（host 与 device-event 漂移、亚微秒隔离）遵循
[`docs/reference/timing_arbitration.md`](reference/timing_arbitration.md)。

### 3.1 发布门是正确性 —— 不是成本

让一个任务 **可发布** 的是正确性，由
`release_gate/eligibility.py::compute_release_ready_eligible` 计算：

```
release-ready = score_eligible ∩ { tasks where is_provenance_eligible(meta) }
```

两个正确性条件：

- **Oracle 等价性**（`is_provenance_eligible`）：基线信任链已
  建立 —— `provenance.golden.oracle_kernel_equivalence` 为 `VERIFIED` 或
  一个白名单内的大 shape dtype-ULP `CEILING_ACCEPTED`，**或者** GPU 采集为
  `DONE` —— 且容差来源不是一个 deferred 回退。
- **逐位精确(bit-exact)切断**（`release_status_pass`）：冻结内核通过了
  切断验证（`phase_a_prime.status == PASS`，或
  `SKIPPED_BY_DESIGN`，仅当离线 golden 采集为 `DONE` 时）。

`cost_model.status` 是 **刻意不作为** 发布阻断项的。cost model 是
一个以 ncu 为证据的性能/访问模型，不是正确性证明 —— 条件性
内存访问、L2 复用、tile 几何，以及 HBM 突发下限，使得一个可移植的
解析式字节公式在结构上无法在每个 shape 上都命中一个紧的逐 shape 容差。它的逐 shape 证据与 PASS/BLOCKED 状态会作为
元数据保留；以它为发布门要么会因为一个 perf-model 技术细节把发布钉红，
要么会推向不诚实的逐 shape 拟合。所有三个
release-status 权威（`validate_tasks.py`、`tab analyze` 报告器，以及
切断验证器）调用 **同一套** eligibility 函数，因此
阻断集合在构造上是一致的。

---

## 4. 各部分在何处运行

| 角色 | 硬件 | 运行 |
|-------|----------|------|
| **Submitter** | 任意（不需要 GPU/NPU） | `tab run` / `tab run-suite`：静态检查 → 隔离子进程 → 针对进程内 PyTorch oracle 的 `tolerance.compare`。这是独立、可本地复现的正确性路径。 |
| **Provider（离线）** | GPU（H100/A100） | oracle 等价性交叉核对 —— 冻结的 `kernel.py` 对比 `oracle_pytorch.py`，在容差类下 —— 外加 golden 采集与切断验证。GPU 是 *交叉核对* 主机，绝不是迁移目标。 |
| **Provider（verified）** | Ascend 910B NPU | 权威的 verified 运行：移植后的内核在 910B 上对 public + hidden shapes 重新执行，并测量延迟。迁移在这里被真正证明。 |

正确性评分器经过工程设计，使得与主机无关、对安全至关重要的
属性（进程隔离、仅 safetensors 的 IPC、无 golden 泄漏、审计钩子）
即使在纯 CPU 的开发机上也成立。重量级设备沙箱叠加在其之上，
在 GPU/NPU 机器上层叠加载。

---

## 5. 路线图（尚未实现）

以下是已设计但 **未** 交付的部分；上文没有任何内容依赖它们。

- **加固的 verified runner。** `process_isolation.py` 当下强制执行
  与主机无关的子集（子进程隔离、组杀、清洗、审计钩子、
  safetensors IPC）。完整的设备沙箱 —— Docker + seccomp + cgroup 限制、
  一个 NPU 设备允许名单，以及 trusted-shim 的别名 / `data_ptr` 检查 —— 已为 910B 机器层
  规定，但尚未接入 runner。
- **公开排行榜 + 提交 broker。** `tab submit` 端点与
  两层（self-reported / verified）排行榜，外加一个为可信 golden 分发
  服务的签名制品 broker，均在计划中。CLI 表面已存在；
  托管服务尚不存在。
- **Roofline 计分的性能排名。** 逐任务的效率运算
  （`compute_t_ref`、`capped_geomean_native`）已实现，但排名仍
  受制于 verified 的 910B 峰值表；在那些落地之前，性能仅
  作为原始延迟报告，并留在发布门之外。

当前状态见 [README](../../README.zh.md)。
