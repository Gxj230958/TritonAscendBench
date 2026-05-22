# cost_model 公式填充方案

**Date:** 2026-05-21
**对象:** `triton-ascend-bench`
**依据:** `docs/DESIGN.md §5.6/§9`、`specs/task_api_contract.md §5.1`、`specs/scoring_spec.md`、`triton_ascend_bench/cost_model.py`、`infra/peak_table_910b.toml`

## 1. 背景与目标

双目标 Roofline 评分的**性能目标**需要每题 `cost_model` 提供 `W_c`(FLOPs)/`M_r`(bytes),用于 Roofline 基线 `t_ref = max(max_c W_c/π_c, max_r M_r/β_r)`(`DESIGN §5.6/§9`)。

**现状**:100 题仅 **2 题**(`L1_01_bias`、`L1_12_rms_norm`)有 `cost_model`,98 题缺。`validate_tasks` 把 `cost_model` 列为 `REQUIRED_META_TOP_LEVEL` ⇒ 98 题 validate red;且 `cost_model.status` 进 P7 硬发布门(`release_status == PASS ⟺ phase_a_prime.status == PASS ∧ cost_model.status == PASS`)⇒ 缺 `cost_model` 则 benchmark 不可发布。

**目标**:给**全部 non-canary real task** 填 `cost_model`(`flops`/`bytes` size-knob 公式 + compute-class/resource 标注)——`validate_tasks` 把 `cost_model` 列为 required 是对**所有非 canary task**(不只 score-eligible),故 scope 是全部 100 题(`tasks/_canary/` 豁免)。使 `cost_model` 维度 `validate_tasks` 转 green、`cost_model.status` 可 PASS。

## 2. `cost_model` schema(回顾)

`meta.yaml.cost_model` 三键(`specs/task_api_contract.md §5.1`):

| key | 含义 |
|---|---|
| `flops` | per-compute-class FLOPs `W_c`,size-knob 公式 |
| `bytes` | per-resource bytes `M_r`,size-knob 公式 |
| `gpu_xcheck_tol` | analytic-vs-instrumented 容差,`(0,1)`,默认 `0.05` |

`flops`/`bytes` 各两形式:
- **scalar** — 单公式字符串 + 必填伴随键 `flops_class`(∈ `fp16_cube`/`bf16_cube`/`int8`/`vector`)、`bytes_resource`(∈ `hbm`/`interconnect`)
- **map** — `{compute_class: 公式}` / `{resource: 公式}`,算子混多类时用

公式语言:size-knob 算术(`B`/`L`/`D`/… 与 `shapes.yaml` 同参数)+ 派生符号 `dtype_size`(`cost_model.py::DERIVED_SYMBOL_DEFAULTS`,默认 2.0);经 `expr.py::eval_numeric` 求值。

## 3. 填充方法论

逐题:读 `kernel.py`(冻结上游)+ `oracle_pytorch.py` 理解算子 → 算 FLOPs(按 compute class)+ bytes(按 resource)→ 选形式 → 校验。

### 3.1 FLOPs `W_c`
数算子浮点运算量,写成 size-knob 公式。**compute class 判定**:
- **cube 类**(矩阵乘累加,走 Cube 单元):`fp16_cube`/`bf16_cube`/`int8` —— matmul/GEMM、attention 的 `QK^T`·`PV`、LoRA。FLOPs 以 `2*M*N*K`(2 = 乘+加)为基型。
- **vector 类**(逐元素/归约/激活,走 Vector 单元):`vector` —— elementwise、norm、softmax、reduce、激活函数。FLOPs 以 `常数 * 元素数` 为基型(如 RMSNorm-on-`[B,D]` 约 `4*B*D`)。
- 混合算子(matmul + bias epilogue、attention 的 cube+vector 段)用 **map 形式**分类写,例:`flops: {fp16_cube: "2*M*N*K", vector: "M*N"}`。

**cube 子类按 dtype 精确选**(schema 要求精确枚举名 `cost_model.py::VALID_FLOPS_CLASSES`):

| 操作数/累加 dtype | `flops_class` / map key |
|---|---|
| `float16` 矩阵乘 | `fp16_cube` |
| `bfloat16` 矩阵乘 | `bf16_cube` |
| `int8` 量化 GEMM(如 `L2_10_w8a8_block_int8_matmul`) | `int8` |
| 逐元素 / 归约 / 激活(任意 dtype) | `vector` |

### 3.2 bytes `M_r`
数访存字节:读 input 张量 + 写 output 张量之和。

**mixed-dtype 规则(关键)**:`dtype_size` 派生符号是 scoring 从**第一个 declared tensor input** 取的**单一代表值**(`eval.py` 的 representative dtype_size),只反映 primary tensor family。corpus 已有 mixed-dtype task(`L1_01_bias`:float32 logits + int64 index 表;`L2_10`:int8 操作数 + float32 scales)——对它们「读写全 `* dtype_size`」会系统性误算、W4.2 大概率 fail。故:
- primary tensor family 的字节用 `* dtype_size`;
- **non-primary dtype 的张量**(index/表/scale 等)用**字面量字节宽**(`int64`⇒`*8`、`int32`⇒`*4`、`float32`⇒`*4`、`int8`⇒`*1`);
- 须数齐 **auxiliary/index 张量**与 kernel 实际触碰的**重复 pass**(不止主张量一读一写)。

**resource 判定**:
- `hbm` — 片上 HBM 访存。本 benchmark 100 题均为**单 kernel / 单设备**语义(含 EP/DeepEP gather/scatter/reorder 题 —— 它们在 bench 里是**本地 tensor 重排**,非跨卡通信),故绝大多数乃至全部 task 用 `hbm`。
- `interconnect` — **仅**真正的跨 rank / allreduce / all2all 通信 kernel 的通信字节;bench 题若无此语义则不用。

### 3.3 形式选择
单一 compute class + 单一 resource → scalar(配 `flops_class`/`bytes_resource`);混合 → map。

### 3.4 符号语义审计(强制)
`validate_tasks` 只验公式 **parse + 对 knob universe eval-ability**,**不验语义** —— 一个引用了该题 `shapes.yaml` 碰巧声明、但算子并不使用的 knob 的公式仍会 validate green(「自洽 ≠ 忠实」陷阱,同 Task #20 WB.1 教训)。故每题填 `cost_model` 后**强制 audit**:公式里每个 size-knob 符号,必须对应 `kernel.py`/`oracle_pytorch.py` 里该算子**真实操作的 tensor 维度或 loop extent**,不得用 validator 碰巧接受的无关 knob。
**反例(须连带修正)**:现有样例 `L1_12_rms_norm` 的 `cost_model` 用了 `L`(`flops: {vector: 4*B*L*D}`、`bytes: {hbm: (2*B*L*D + D)*dtype_size}`),但其张量声明为 `[B, D]`、oracle `reshape(-1, D)` 后逐行操作 —— `L` 非该算子真实维度。填充 98 题时须连同**核实并修正现有 2 个样例**(`L1_01`、`L1_12`)的 `cost_model` 忠实性。

## 4. 分类策略(可机械化 vs 抗机械化长尾)

按算子族 + compute-class 倾向分三档(算子族认知复用 Task #19/#20 的 13 族分类):
- **可机械化**(elementwise / reduce / norm / 激活 / 量化):vector 类为主,FLOPs/bytes 公式规律性强(`常数*元素数` + 张量读写和),批量起草。
- **半机械化**(GEMM / matmul / LoRA grouped-GEMM):cube 类,公式 `2*M*N*K` 基型规律,逐题对 `M/N/K` 与 epilogue。
- **抗机械化长尾**(attention / SSM-chunk / recurrent / MoE 融合):多 compute class、需逐题对照 `kernel.py` 的循环/分块结构与 oracle 数学,FLOPs 含多段(cube QK·PV + vector softmax)。

## 5. 验收与校验(两层)

- **第一层 — analytic 自洽(dev 机可做,本方案交付)**:`validate_tasks::_validate_cost_model` —— `parse_cost_model` + `validate_formulas`:公式 parse + 对该题 `shapes.yaml` knob universe 求值通过。**只验 parse/eval-ability,不验数值对错**。
- **第二层 — GPU 交叉校验(W4.2,需 GPU 机,本方案不含)**:analytic FLOPs/bytes vs 插桩 GPU-Triton 实测,超 `gpu_xcheck_tol` ⇒ `cost_model.status = BLOCKED`(release-gate 硬阻塞)。数值正确性靠这层兜底。

## 6. 关键约束与边界

- **peak_table π/β 占位**:`infra/peak_table_910b.toml` 的 `vector` `π_c = 0.0`、`interconnect` `β_r = 0.0`(`[unverified]` 占位,注释明示「loader 对 `π_c≤0` 拒绝报错,须 910B 实机 microbenchmark 回填」)。故 **`cost_model` 公式填充 ≠ 立即能算出 Roofline 分** —— vector-bound 算子(占多数)的 `W_vector/π_vector` 在 `π_vector=0` 时不可求。peak_table 峰值回填是 **NPU 机器侧另一工作项**,不在本方案范围;本方案使 `cost_model` **公式就绪**,为峰值回填后立即可评分铺路。
- **analytic 估算 vs 实测**:dev 机填的是 analytic 公式(数学推导的 FLOPs/bytes 量级);`validate_tasks` 只验 parse/eval。真实数值精度由 W4.2 GPU cross-check(GPU 机)裁定 —— 公式力求 analytic 准确,但承认 GPU cross-check 是最终裁判。
- **抗机械化长尾**:融合算子 `cost_model` 需逐题对照 kernel 循环结构,抗机械化。按既定原则交付(方法论 + 批量机械化子集 + 诚实报告长尾),不 fake 全量。

## 7. 范围边界(非目标)

- **不做** peak_table 峰值回填(NPU 实机 microbenchmark)
- **不做** W4.2 GPU cross-check 实跑(GPU 机)
- **不改** `cost_model` schema / 评分代码(W1.2 已实现;本方案是纯数据填充)
- `L1_21_gumbel_sample`:`validate_tasks` 对它同样要求 `cost_model`(非 canary)。其 oracle 是 stub、`shapes.yaml` 为非 executable 占位(无 `size_knobs`)⇒ 无 size-knob 符号可写真实 shape-依赖公式。处置:填**常量型占位 `cost_model`**(常量公式 + 合法 `flops_class`/`bytes_resource`,同 Task #20 WA.5 给 scaffold task 的占位手法)——`validate_formulas` 对常量 / 仅-`dtype_size` 公式在空 knob 集上可通过,故 `L1_21` 同样 validate green。但其 `cost_model` 数值**非真实 FLOPs/bytes**,须显式标注为占位(待该题补 oracle + executable `shapes.yaml` 后填真实公式);因 `L1_21` non-score-eligible(stub-oracle)、不进评分分母,占位 `cost_model` 不影响评分正确性

## 8. 验收标准(DoD)

- **全部 100 个 non-canary task** 的 `meta.yaml` 含 `cost_model`(`flops`/`bytes` + class/resource 标注):其中 99 题为真实 shape-依赖公式,`L1_21` 为显式标注的常量型占位
- `validate_tasks` 的 `cost_model` 维度对**全部 100 题**转 green(`parse_cost_model` + `validate_formulas` 通过)
- 每题真实 `cost_model` 经 §3.4 符号语义审计 —— 公式符号语义忠实(非仅 validator-passing);现有 2 样例(`L1_01`/`L1_12`)的忠实性一并核实修正
- 抗机械化长尾如实统计报告,不瞎填充数
- 不破坏现有 pytest 套件
