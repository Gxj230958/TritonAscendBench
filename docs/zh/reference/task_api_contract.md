# 任务 API 契约

> English: [../../reference/task_api_contract.md](../../reference/task_api_contract.md)

> 权威打分模型：`docs/DESIGN.md §4`（双目标 —— 正确性门 + 910B Roofline 归一化性能）。GPU-Triton → 910B NPU-Triton 迁移；无 AscendC 基线。


> 唯一真相来源：每个任务的 `meta.yaml`。面向人的文档（本文件 + `contract.md`）只作解释，不能覆盖 `meta.yaml`。

## 1. 任务目录布局

每个任务都是 `tasks/level{1,2,3}/<task_id>/` 下的一个目录：

```
tasks/level1/L1_12_rms_norm/
├── kernel.py # GPU triton reference (pure operator, no oracle/meta pollution)
├── oracle_pytorch.py # PyTorch eager reference (runner-only, not mounted to candidate)
├── meta.yaml # Structured metadata + runner_checks (source of truth)
├── shapes.yaml # SHAPE_CONFIGS + ADVERSARIAL cases
└── contract.md # Human-readable semantic contract
```

（5 文件契约 —— 见 `docs/ARCHITECTURE.md §1.1`。辅助目录：`_frozen_deps_pool/`、`_frozen_shim/`、`_canary/`、`_vendored/`。）

## 2. `kernel.py` 契约

- 从 vLLM / SGLang 上游抽取出的纯算子实现
- 必须导出由双锚点解析器（dual-anchor resolver）验证的真实上游入口点，记录在 `meta.yaml.oracle.gpu_triton`（在解析器填入之前为 `kernel.py::placeholder`）
- 必须包含至少一个带 `@triton.jit` 装饰器的 kernel
- 可被提交者（submitter）作为参考导入；不得污染 oracle/meta-yaml
- 上游 commit SHA + 文件哈希锁定在 `meta.yaml.upstream`

## 3. `oracle_pytorch.py` 契约

- 通过双锚点（dual-anchor）与冻结的上游语义对齐的 PyTorch eager 参考实现（无 triton）；**不是独立的真相来源**
- 必须暴露 `run(inputs: dict) -> dict[str, Tensor]`
- 设备无关（张量保持在调用方的设备上）
- **绝不挂载到候选实现沙箱**
- 仅由 `Trusted Runner` 进程加载

## 4. 提交者 `solution.py` 契约

```python
def npu_triton_impl(inputs: dict) -> dict:
 """
 Required interface for all submissions.

 Args:
 inputs: dict with keys per `meta.yaml.interface.alias_map`
 (all tensors on NPU device)

 Returns:
 dict with keys matching `meta.yaml.oracle.gpu_triton` output schema
 (all tensors on NPU device; candidate-allocated)
 """
```

**`solution.py` 中禁止的内容**（静态 AST 黑名单）。权威的、详尽的黑名单是实现本身 ——
`ascendbench/integrity/static_checker.py`（`_DENYLIST_IMPORT_EXACT` /
`_DENYLIST_IMPORT_PREFIX` + Layer D 属性黑名单）。代表性
类别：
- `import oracle_pytorch` / `import ascendbench` / `import tasks.*`
- 高层 torch 算子：`torch.nn.functional.*`、`Tensor.matmul`、`@` 运算符
- 原生 AscendC 算子：`sgl_kernel_npu.*`、`vllm_ascend.npu_*`、`torch_npu.npu_*`
- 代码执行 / 反射：`exec`、`eval`、`compile`、`__import__`、`importlib`、`functools.partial`、`ctypes`、`cffi`、`torch.utils.cpp_extension`
- **反序列化（密封 golden 数据外泄）**：`safetensors`、`safetensors.torch.load_file`、`torch.load`、`torch.serialization`、`pickle` —— verified runner 将密封的 golden 张量保存在进程内存中；加载任何序列化的张量都会绕过 file-mode + 签名门
- IPC / 网络：`subprocess`、`socket`、`requests`、`urllib`、`http`
- 计时操纵：`torch.npu.Event`、`torch.cuda.Event`、`time`、`datetime`、`torch.profiler`、`LD_PRELOAD`
- 文件 / IO：`os`、`os.path`、`pathlib`、`mmap`、`shutil`、`tempfile`、`glob`、`fcntl`、`open`（超出 stdlib 范围）
- 线程：`threading`、`multiprocessing`、`asyncio`、`concurrent.futures`

## 5. 必填 `meta.yaml` 字段

机器可读的 schema 由 `scripts/corpus/validate_tasks.py`（`REQUIRED_META_TOP_LEVEL` + 嵌套结构检查）强制约束。必填项：

```yaml
schema_version: "1.0"
task_id: "L1_12_rms_norm"
level: "L1" # enum: L1, L2, L3, appendix, canary
idx: 12
name: "_rms_norm_kernel"
repo: "vLLM" # enum: vLLM, SGLang, FBGEMM, Bytedance_internal
dtype_class: "VV" # enum: VV, CV
category: "norm" # enum: elementwise, norm, reduce, sampling, kv_cache, lora, attention, moe, state_space
npu_difficulty: "medium" # enum: low, medium, high

upstream:
 repo: "vllm-project/vllm"
 path: "vllm/model_executor/layers/batch_invariant.py"
 commit_sha: "<frozen at Phase 1.2>"
 function_hash: "<sha256 of kernel source>"
 license: "Apache-2.0"

interface:
 alias_map:
 input: ro
 weight: ro
 output: wo_candidate_allocated
 mutability: "no_inplace"
 stride_layout: "input_2d.contiguous required"
 workspace_policy: "none"
 stream_sync_policy: "runner-managed"
 rng_state_policy: "deterministic"

oracle:
 pytorch: "oracle_pytorch.py::run"
 gpu_triton: "kernel.py::placeholder" # resolver fills the verified upstream entrypoint

tolerance_class: "vector_reduce" # one of the classes in ascendbench/policies/tolerance_policy.yaml

cost_model: # performance baseline — see docs/DESIGN.md §4.3
 flops: "2*B*L*D" # size-knob FLOPs formula (scalar form)
 flops_class: fp16_cube # scalar form: name the compute class
 bytes: "(B*L*D + B*D) * dtype_size" # HBM bytes (scalar form)
 bytes_resource: hbm # scalar form: name the resource
 gpu_xcheck_tol: 0.05 # analytic vs instrumented GPU-Triton mismatch tolerance
 # status (PASS / BLOCKED) is produced by the GPU cross-check, not authored here

oracle_simplified: "formula_exact_manual" # (simple_pytorch_ok / formula_exact_manual / upstream_oracle_required)

runner_checks:
 expected_kernel_count_range: [1, 5]
 min_dram_bytes: 65536
 min_flops: 524288
 metamorphic_checks:
 - "differential_perturbation_general"
 output_format: "safetensors"
 output_allocation: "candidate_allocated"

is_canary: false
is_hidden_only: false
```

### 5.1 `cost_model` schema（性能基线）

`meta.yaml.cost_model` 提供 Roofline 基线 `t_ref = max(max_c W_c/π_c, max_r M_r/β_r)`
所需的每任务 `W_c`（FLOPs）和 `M_r`（bytes）
（`docs/DESIGN.md §4.3 / §4.2`）。三个 key：

| key | meaning |
|-----|---------|
| `flops` | per-compute-class FLOPs `W_c`, as a size-knob formula |
| `bytes` | per-resource HBM/interconnect bytes `M_r`, as a size-knob formula |
| `flops_class` | **scalar form only, required** — the compute class the scalar `flops` formula belongs to (`fp16_cube` / `bf16_cube` / `int8` / `vector`) |
| `bytes_resource` | **scalar form only, required** — the resource the scalar `bytes` formula belongs to (`hbm` / `interconnect`) |
| `gpu_xcheck_tol` | analytic-vs-instrumented GPU-Triton mismatch tolerance, in `(0, 1)` (default `0.05`) |

`flops` 和 `bytes` 各自独立地接受**两种形式**：

- **scalar** —— 单个公式字符串，即单计算类 /
 单资源的退化形式。一个 scalar `flops` **必须**携带
 配套的 `flops_class`，scalar `bytes` 必须携带配套的 `bytes_resource`，
 用以命名该公式所属的峰值表（peak-table）类/资源：

 ```yaml
 cost_model:
 flops: "2*B*L*D"
 flops_class: fp16_cube
 bytes: "(B*L*D + B*D) * dtype_size"
 bytes_resource: hbm
 gpu_xcheck_tol: 0.05
 ```

- **map** —— 当算子混用多个计算类或数据资源时，使用
 `{compute_class: formula}` / `{resource: formula}`：

 ```yaml
 cost_model:
 flops:
 fp16_cube: "2*M*N*K" # matmul on the cube unit
 vector: "M*N" # epilogue bias on the vector unit
 bytes:
 hbm: "(M*K + K*N + M*N) * dtype_size"
 interconnect: "M*N * dtype_size" # cross-device collective traffic
 gpu_xcheck_tol: 0.05
 ```

 计算类 key 是 `ascendbench/policies/peak_table_910b.toml` 中的类
 （`fp16_cube` / `bf16_cube` / `int8` / `vector`）；资源 key 是它的
 资源（`hbm` / `interconnect`）。

验证器和数据集加载器接受任一形式并**在内部归一化为
map** —— scalar 形式变成一个以其配套 `flops_class` / `bytes_resource` 为 key 的单条目 map，
即一个真实的峰值表类/资源（绝不是合成 key）。每个下游消费者
都看到一致统一的结构，归一化后的 map 始终可以直接对照峰值表打分。
map 形式字段上若出现 `flops_class` / `bytes_resource` 会被拒绝 —— map
本身已经给类/资源做了 key。

**公式语言。** 公式是一个 size-knob 算术表达式，其
参数化方式与 `shapes.yaml` 相同（`B` / `L` / `D` / …），再加上一组固定的
*派生符号（derived symbols）* —— 目前为 `dtype_size`（任务主 dtype 的元素字节大小）。
打分时 `eval.py` 将**代表性 `dtype_size`** 推导为
`shapes.yaml.inputs` 中**第一个声明的张量输入**的字节宽度（按声明顺序，
跳过没有元素大小的非张量形式 `scalar` / `derive` / `scalar_seq` /
`scalar_knob`）—— 例如 `float32` 主操作数给出 `dtype_size = 4`，`int64`
给出 `8`。一个只含 scalar 的 `inputs` 块没有张量元素大小，因此使用
fp16 默认值（`2.0`）。对真正的混合 dtype 任务做逐输入精确处理属于长尾
工作；在此之前，这个单一的代表值会喂给每个 `flops` / `bytes` 公式。
验证器用 fp16 默认 `dtype_size` 来评估公式 —— 它只确认公式*能解析并求值*，
而非其数值。`cost_model.py` 精确地定义了存在哪些派生符号，并被
`dataset.py`、`validate_tasks.py` 和 GPU cross-check 逐字复用，因此符号
集合永不漂移。算术安全核心本身位于
`ascendbench/input_spec/expr.py`：`cost_model` 公式求值是该共享
AST 白名单求值器的 `eval_numeric` 入口点。数值模式下允许：
`+ - * / // ** %`、一元 `+` / `-`、数值字面量、
knob/派生名称，以及 `min` / `max` / `ceil` / `floor` / `sqrt` 调用。
其它任何东西（属性访问、下标、推导式、未知名称）都会被
拒绝 —— 公式通过 AST 白名单求值，绝不用 `eval`。

**`cost_model` 存在性 —— 分层处理（加载器宽松，验证器严格）。**
两个消费者*按设计*对缺失的 `cost_model` 块区别对待：

- **数据集加载器（`dataset.py`）** —— *向后兼容*。一个 `meta.yaml` 没有
 `cost_model` 块的任务可以正常加载，`TaskDef.cost_model = None`；但
 *格式错误*的块仍会抛错。这使得 pre-tasks（以及任何合成测试 fixture）
 可加载，并让 Roofline 路径在任务无 cost model 时直接跳过而非失败。
- **任务验证器（`scripts/corpus/validate_tasks.py`）** —— *严格*。
 `cost_model` 是 `meta.yaml` 的**必填**顶层 key：缺失块是
 硬验证错误。

简而言之：缺失 `cost_model` *不是加载错误*但*是发布错误*。当
块存在时，两者都对其严格验证。

### 5.2 `param_constraints` schema（算子输入契约）

`meta.yaml.param_constraints` 以机器可检查的形式声明算子的**完整输入契约**
—— 张量维度、dtypes、值分布、非张量字面量以及跨张量不变式。它的存在
使得 benchmark 任务再也不能携带机械式的占位参数：那些参数要么
跑不起来，要么更糟 —— 在一个无意义的参数点上评估 oracle 和候选实现，
使两者虚假地"通过"。

`param_constraints` 有**六个子节**：

| subsection | object | form |
|-----------|--------|------|
| `dims` | size-knob integers | list of **predicate** expression strings |
| `dtypes` | each tensor `inputs.*` | `{input_name: dtype_enum}` |
| `dists` | each tensor `inputs.*` | `{input_name: {kind, args}}` |
| `scalars` | non-tensor literals (`scalar` / `scalar_seq`) | `{name: predicate-or-derive}` |
| `tensor_invariants` | materialised tensors | list of `{type, args}` |
| `scaffold` | the block itself | bool — `true` = un-reconciled draft |

#### `dims` —— knob 约束谓词

一组针对 `shapes.yaml` size knobs 的谓词表达式字符串。
每个字符串是一个布尔表达式，针对每个 public / hidden
knob 行求值。示例：

```yaml
dims:
 - "D in [16, 32, 64, 128]" # closed knob set
 - "H % G == 0" # divisibility (constexpr-derived)
 - "is_pow2(D)" # power-of-two head dim
 - "L >= 16" # lower bound (BLOCK_SIZE tile floor)
```

这些覆盖了 kernel 的 `tl.constexpr` / `BLOCK_SIZE` 对 knob 全集
施加的约束。求值器（共享的 `eval_predicate`
入口点）支持比较运算（`== != < <= > >=`）、布尔运算符
（`and or not`）、`in`、列表 / 元组字面量，以及谓词辅助函数
`is_pow2` / `gcd` / `sqrt` / `len` / `sum`。schema-shape 层只验证
该子节的*形状*（即它是一个字符串列表）；语义求值
在语义 eval 门中进行。

#### `dtypes` —— 逐输入 dtype 约束

一个 `{input_name: dtype}` map；`shapes.yaml.inputs` 中声明的每个张量输入
对应一条。值是一个**封闭枚举** —— `build_inputs` dtype 别名表
（`ascendbench/input_spec/inputs.py`）的 key，该表是合法 dtype 集合的
**唯一权威**：

```
float32 float16 bfloat16 bf16 fp16 fp32
int8 int32 int64 long bool float8_e4m3fn
```

`validate_tasks` 和 `build_inputs` 共享这一张表，因此未知的
dtype 字符串（拼写错误）会被静态拒绝，而非悄悄回退到
`float32`。示例：

```yaml
dtypes:
 x: float32
 a: int8 # W8A8 quantised activation
```

#### `dists` —— 逐输入分布约束

一个 `{input_name: {kind, args}}` map，声明每个张量
输入必须以何种分布物化。`kind` 是下方 **dist-kind
表**中的一个值；`args` 是一个（可能为空的）结构化参数 map。
这会静态拒绝一个*受支持但错误*的分布 —— 例如一个
必须是行随机（row-stochastic）`softmax_rows` 矩阵的输入被声明为普通
`randn`。

```yaml
dists:
 q: {kind: randn}
 probs: {kind: softmax_rows}
 seqused_k: {kind: randint, args: {lo: 0, hi: "L"}}
```

**表示桥接。** `shapes.yaml.inputs.*.dist` 保留其既有的
字符串 DSL 形式（`randint:0:E`、`offsets:to=B*L`、`full:expr`）；
`param_constraints.dists` 块使用结构化的 `{kind, args}` 形式。
两者通过单个共享的 `parse_dist_spec` 归一化器
（`ascendbench/input_spec/inputs.py` —— 字符串 DSL ⇄ `{kind, args}`）协调，
因此不存在重复解析或漂移：`build_inputs` 物化、
`validate_tasks` dist 检查以及生成器全部经由它。
字符串 DSL 仍然是权威的 `shapes.yaml` 编写形式，不会被迁移。

**dist-kind 表**（封闭的 `kind` 枚举，绑定到 `build_inputs`
分布词汇表）：

| kind | meaning | `args` |
|------|---------|--------|
| `randn` | standard normal | — |
| `rand` | uniform `[0, 1)` | — |
| `ones` | constant 1 | — |
| `zeros` | constant 0 | — |
| `eye` | identity matrix | — |
| `tril` | well-conditioned lower-triangular | — |
| `randint` | uniform integers | `lo`, `hi` (knob exprs) |
| `softmax_rows` | row-stochastic (softmax over last dim) | — |
| `l2norm_rows` | randn rows L2-normalised over the last dim (unit-norm keys; the stable delta-rule key recipe) | — |
| `logsigmoid` | `logsigmoid(rand)` — strictly negative gate so `exp(g)` is a genuine decay in `(0, 1)` | — |
| `offsets` | monotonic **leading-zero** indptr int64 vector (`[0] == 0`, total in `[-1]`) | `step` *or* `to` (knob expr) |
| `cumcount` | monotonic **non-leading-zero** cumulative count vector (element `r` = running count through segment `r`, total in `[-1]`); honors the declared `int32`/`int64` dtype | `to` (knob expr) |
| `arange` | `0.. n-1` int64 | — |
| `cyclic_perm` | single-cycle permutation `perm[i] = (i+1) % n` — a guaranteed non-identity bijection of `[0, n)` (every position remapped for `n >= 2`); honors the declared `int32`/`int64` dtype | — |
| `full` | constant-filled int64 | `value` (knob expr) |
| `randn_inf` | randn with a fixed fraction of entries set to `+inf` (flash-decoding empty-split LSE sentinel) | — |
| `randn_inf_excl` | randn with `+inf` injected only where the referenced earlier tensor is finite (no both-empty positions) | `ref` (input name) |
| `randperm` | random permutation of `[0, n)` (MoE token reorder; may coincide with identity at small `n` — `cyclic_perm` guarantees non-identity) | — |
| `tril_chunk` | strict lower-triangular mask per `[BT, BT]` chunk in the `(T, BT)` plane; needs shape `[B, T, H, BT]` | — |
| `rope_cos_sin` | deterministic RoPE cos/sin cache `[max_pos, rotary_dim]` (`cos^2+sin^2 == 1`, half-row cos / half-row sin) | — |
| `routing_ids` | MoE per-token expert ids in `[0, E)` with a pinned `-1` drop sentinel (≥1 written + ≥1 skipped row per shape) | `E` (knob expr) |
| `moe_sorted_ids` | `moe_align_block_size`-style sorted_token_ids: distinct route-slot ids in `[0, B)`, sentinel `B` padding | `B` (knob expr) |
| `gumbel_routing` | per-token request ids in `[0, R)`; token 0 pinned greedy-request, token 1 noisy-request | `R` (knob expr) |
| `gumbel_temps` | per-request sampling temperature mixing pinned greedy (`0.0`) and noisy (`[0.3, 1.7)`) entries | — |
| `window_pos` | per-token local position `t % W` inside fixed draft windows tiling `[0, n)` | `W` (knob expr) |
| `window_req` | per-token request id `(t // W) % E`, constant within each length-`W` window; pairs with `window_pos` on the same `W` | `W`, `E` (knob exprs) |

#### `scalars` —— 非张量字面量约束

针对**非张量字面量**输入的约束 —— `shapes.yaml.inputs`
中以 `scalar` key（单个 JSON 标量）或
`scalar_seq` key（列表字面量）声明的条目。每个条目要么是：

- 一个**值域谓词** —— 一个 Python 兼容的布尔表达式
 （允许链式比较），作用于字面量名称**及
 `shapes.yaml` size knobs**，例如 `"0 < eps <= 1e-2"` 或
 引用 knob 的 `"sum(mrope_section) == D // 2"`。（只允许 Python 可解析的
 形式 —— *不是*数学区间记法 `(0, 1e-2]`。）
 谓词针对一个符号表求值，该符号表合并了 shape 行的
 knobs 和字面量自身的名称，与 `dims` 谓词求值
 镜像；当 knob/名称冲突时，字面量名称优先。
- 一个**derive 规则** —— 见 `shapes.yaml.inputs` 非张量形式一节（`{derive: "1/sqrt(D)"}`）。derive 规则会
 与 `shapes.yaml.inputs[name]` 协调：该输入本身必须是
 `derive` 形式，并携带一个结构等价的公式。任务不可以
 声明 `scale: {derive: "1/sqrt(D)"}` 而 `shapes.yaml` 又将 `scale` 钉死为
 固定的 `{scalar: 0.125}` —— 验证器会拒绝这种不匹配。

`scalar_seq` 谓词可以使用 `len`、`sum` 以及**常量整数
索引**（`block_size[0] == 8`、`mrope_section[-1] == 2`），从而把一个
列表字面量本身也纳入机器可检查的契约。索引
只接受常量整数字面量（允许负索引）；切片
或非常量索引会被拒绝，越界索引是
验证错误。

```yaml
scalars:
 eps: "0 < eps <= 1e-2"
 mrope_section: "sum(mrope_section) == D // 2" # scalar_seq, knob-referencing
 block_size: "len(block_size) == 2 and block_size[0] == 8" # scalar_seq, indexed
```

**`shapes.yaml.inputs` 非张量形式。** 物化层
（`ascendbench/input_spec/inputs.py`）用以下 key 中恰好一个来声明
一个非张量输入；值类型是 **schema 固定的**：

| key | meaning | value |
|-----|---------|-------|
| `scalar` | a literal constant | a single JSON scalar (`bool`/`int`/`float`/`str`/`None`) |
| `derive` | a per-shape-row derived scalar | a numeric knob expression string, e.g. `"1/sqrt(D)"`; `build_inputs` evaluates it through `eval_numeric` against the shape row's knobs |
| `scalar_seq` | a literal list | a list of JSON scalars (e.g. `L2_10` `block_size`, `L1_31` `mrope_section`) |
| `scalar_knob` | a derived integer | an integer-valued knob expression string (legacy form) |

`scalar` / `derive` 值**必须**是一个 JSON 标量 —— 它跨越 sidecar IPC 类型边界，而该边界不携带 pickle。因此列表字面量
*不能*在 `scalar` 下编写；它使用专门的 `scalar_seq`
key，IPC sidecar 将其作为 JSON 数组携带。`build_inputs` 会以
`TypeError` 拒绝非 JSON 标量的 `scalar` / `derive` 值。

#### `tensor_invariants` —— 物化张量不变式

一组 `{type, args}` 条目，每条都是一个**枚举式**不变式，针对
物化后（`build_inputs` 之后）的张量检查。`type` 是下方
封闭集合中的一个值；`args` 携带逐类型参数。这些
表达了 knob 表达式无法表达的契约 —— 例如 `cu_seqlens` 必须
单调，`len(seqused_k)` 必须等于某个 knob。

**枚举式不变式类型表：**

| `type` | meaning | `args` |
|--------|---------|--------|
| `monotonic_nondecreasing` | values are non-decreasing along the last axis | `input` |
| `first_eq` | first element equals a value | `input`, `value` (knob expr) |
| `last_eq` | last element equals a value | `input`, `value` (knob expr) |
| `length_eq` | `len(input)` equals a knob expression | `input`, `value` (knob expr) |
| `bounded_above_by` | every element `<=` a bound | `input`, `value` (knob expr) |
| `index_in_range` | every element is a valid index into `[0, value)` | `input`, `value` (knob expr) |

```yaml
tensor_invariants:
 - {type: monotonic_nondecreasing, args: {input: cu_seqlens}}
 - {type: first_eq, args: {input: cu_seqlens, value: 0}}
 - {type: length_eq, args: {input: seqused_k, value: "E"}}
```

#### `scaffold` —— 未协调草稿哨兵

一个显式布尔字段。`scaffold: true` 标记一个由 `phase_c_docs` scaffold
生成器发出、且**尚未针对算子真实契约手工
协调**的 `param_constraints` 块。它是
生成器、验证器和 eval 门共同查阅的**单一共享哨兵**
—— 一个显式 schema 字段，取代旧的
隐式文本标记，从而没有组件会漂移。`scaffold` **不**
参与打分资格（score-eligible 分母只看
stub / `PHASE-C-MANUAL`）—— `scaffold` 任务是 `blocked` 并留在
固定分母中，绝不从中移除。

#### `param_constraints` 作为 `task_status` 门

除了验证器之外，`param_constraints` 还是一个 **eval 门**：一个
块缺失 / `scaffold: true` / schema 非法 / 语义违规的任务
其 `task_status = "blocked"`。该裁决由单一的
仓库权威仲裁者 `cli analyze` 的 `_authoritative_task_status`
决定（`eval.evaluate` 镜像它，两者都经由共享的
`param_constraints.resolve_task_status`），遵循固定优先级：

1. `is_canary: true` ⇒ `evaluated`（豁免 `param_constraints` 门）；
2. self-reported 模式 `upstream_oracle_required` ⇒ `unsupported`（——
 `param_constraints` 门**不**将其拉回分母）；
3. verified 模式离线 `golden/` 不完整 ⇒ `blocked`（—— 优先于
 `param_constraints` 门）；
4. `param_constraints` 未通过共享 helper ⇒ `blocked` —— 任务留在
 **固定的** score-eligible 分母中，并打分为 ε（它**不**
 从集合中移除；移除它将构成子集逃逸 subset-escape）；
5. 其余 ⇒ `evaluated`。

因此一个 `blocked` 的 `param_constraints` 任务表现得与
Roofline 数据不完整的任务完全一致：它留在 `|S|` 中并计为失败。它
**不**改变 score-eligible 集合本身（仍只是 `stub` /
`PHASE-C-MANUAL`）。

#### `param_constraints` 存在性 —— 分层处理（加载器宽松，验证器严格）

与 `cost_model`（§5.1）类似，两个消费者*按设计*对
`param_constraints` 区别对待 —— 但加载器在这里**更**宽松：

- **数据集加载器（`dataset.py`）** —— *完全宽松*。`TaskDef.param_constraints`
 是一个默认为 `None` 的 `dict | None` 字段。缺失块产生
 `None`。一个**格式错误**的块（非法 schema、错误的约束表达式）
 被**原样保留为原始 dict** —— 加载器**不**验证
 也不抛错。这与 `cost_model` 有意不同，后者会急切解析
 并在格式错误的块上抛错：格式错误的 `param_constraints` 必须
 允许加载，以便它能到达 `task_status = blocked` 门，
 而非在 `load_task` 内部崩溃。加载器存储原始 dict；它
 绝不构建结构化对象。
- **任务验证器（`scripts/corpus/validate_tasks.py`）** —— *严格*。
 `param_constraints` 是 `meta.yaml` 的**必填**顶层 key；缺失
 块是硬验证错误。schema-shape 验证器层只检查
 **schema 形状**（六个子节存在且具有正确的容器
 类型）；五个子节的*语义*检查在 eval 门中运行。

**`is_canary` 豁免。** 一个 `is_canary: true` 的任务**不**被要求
携带 `param_constraints` —— canary 复用其源任务的 solution，
是一个刻意构造的陷阱，且从不参与打分资格。`validate_tasks`
（以及 eval 门）对任何 `is_canary` 任务跳过 `param_constraints`
要求，以免封死 canary 检测链。

简而言之：缺失或格式错误的 `param_constraints` *不是加载
错误*但*是发布错误*。

### 5d. `provenance.golden` schema（基线信任链）

`provenance:` 顶层块历史上携带*内容*元数据
（`added_in`、`authors`、`last_modified`）。一个
嵌套的 `golden:` 子块记录*基线信任链* ——
一个任务的参考值是如何产生、验证和签名的。
顶层 key 不受影响；嵌套块纯粹是增量添加。

```yaml
provenance:
 added_in: v0.1 # existing top-level metadata, untouched
 authors: [...] # existing
 last_modified: '2026-05-26' # existing
 golden: # nested baseline trust chain
 source: upstream_author_test | gpu_kernel_rerun
 tolerance_source: upstream_test_explicit | upstream_test_default | migration_drift_budget
 atol: 1.0e-5 # pinned from upstream test
 rtol: 1.0e-5
 ref_path: tasks/<id>/upstream_ref.py # source ① only — vendored author test ref
 test_file_path: <relative to infer/{repo}> # source ① only
 test_capture_sha: <40 hex> # source ① only — author test git SHA
 golden_sha256: <64 hex> # SHA of the captured ref artifact
 oracle_kernel_equivalence: VERIFIED | FAILED | MIXED | SKIPPED | NEEDS_MANUAL | LAUNCH_BLOCKED | WRAPPER_CONTRACT_MISMATCH | ORACLE_KERNEL_LAYOUT_MISMATCH | INPUT_DRIFT
 gpu_kernel_capture_status: PENDING | DONE | FAILED | BLOCKED_NON_DETERMINISTIC
 gpu_kernel_deterministic: true # N=3 bit-identity check
 upstream_closure_fingerprint: <64 hex> # cache invalidation key
 reviewers: [<name1>, <name2>] # reviewer attestation
 reconstructability_risk: low | medium | high # reconstructability risk score
```

**封闭枚举**（验证器严格；见 `ascendbench/corpus/provenance.py`）：

| Field | Allowed values |
|-------|----------------|
| `source` | `upstream_author_test`, `gpu_kernel_rerun` |
| `tolerance_source` | `upstream_test_explicit`, `upstream_test_default`, `migration_drift_budget` |
| `oracle_kernel_equivalence` | `VERIFIED`, `FAILED`, `MIXED`, `SKIPPED`, `NEEDS_MANUAL`, `LAUNCH_BLOCKED`, `WRAPPER_CONTRACT_MISMATCH`, `ORACLE_KERNEL_LAYOUT_MISMATCH`, `INPUT_DRIFT` |
| `gpu_kernel_capture_status` | `PENDING`, `DONE`, `FAILED`, `BLOCKED_NON_DETERMINISTIC` |
| `reconstructability_risk` | `low`, `medium`, `high` |

**哈希字段约束**（长度 + 十六进制）：

| Field | Required hex length |
|-------|--------------------|
| `golden_sha256` | 64 |
| `upstream_closure_fingerprint` | 64 |
| `test_capture_sha` | 40 |

**开放 / 运行态附加项**（由加载器逐字保留，**不**做
封闭枚举检查 —— 这些是由
`scripts/oracle/apply_oracle_equivalence_verdict.py` 及类似脚本写入的流水线产物）：
`verified_at`、`tolerance_class_used`、`equivalence_per_shape`、
`golden_source`。

**宽限期**：验证器在任何任务上接受缺失的 `golden:` 块
（旧版 meta + 尚未运行 pyref/golden 采集阶段的任务）。
这在 release-gate 报告中被记录为*警告关切（warning concern）*，
但不是阻塞发布的错误。一旦块存在，每个
封闭枚举和哈希长度值都必须符合 —— 一个格式错误的值
**是**阻塞发布的，因为 release gate、eval 边界和
GoldenStore 都直接读取这些字段。

**加载器契约**：`ascendbench.corpus.dataset.TaskDef` 通过
`provenance_golden: ProvenanceGoldenSpec | None` 携带嵌套块的
类型化视图（加载器宽松，与
`cost_model` / `param_constraints` 同一模式）；原始 YAML 结构通过
`TaskDef.meta['provenance']` 保持可达，以便向前
兼容。

## 6. 权威性 + 跨文件一致性

- `meta.yaml` 是所有机器可读字段的**唯一真相来源**。
- `contract.md` **仅供人阅读** —— 不得引入 `meta.yaml` 中没有的事实。
- `shapes.yaml` 在一个文件中同时携带 public 和 hidden eval shapes：一个 `public_shapes` 列表和一个密封的 `hidden_shapes` 列表（`docs/DESIGN.md`）。**没有**单独的 `shapes_hidden.yaml` overlay —— `shapes.yaml.hidden_shapes` 是隐藏 shape 的唯一真相来源。在 `verified` 模式下，trusted runner 物化 `hidden_shapes` 行的输入并对其打分；在 `self-reported` 模式下，隐藏 shape 不被物化。候选实现沙箱完全收不到 `shapes.yaml` 文档（它只得到 runner 物化的输入 —— 即进程隔离模型），因此 `hidden_shapes` 区段在候选实现视图中结构性缺失。
- 如果 `meta.yaml.cost_model` 缺失，或其分析的 FLOPs/bytes 与插桩的 GPU-Triton 参考相差超过 `gpu_xcheck_tol` → `cost_model.status = BLOCKED`。这是 **ncu 取证的性能元数据，而非 release-gate 阻塞项**：cost_model 是一个分析式访存模型，而条件访问 / L2 复用 / tile 几何 / HBM 突发下限使得一个可移植公式在结构上无法处处命中逐 shape 容差。release gate 锚定于两个真正的正确性证明 —— C2 oracle 等价性 + S2 切断验证（`docs/DESIGN.md §4.4`）；`asymptotic_hbm_xcheck` 在大/渐近 shape 上对 cost_model PASS 设门，并将小 shape 记录为 `FLOOR_OBSERVED`。
- 如果 `meta.yaml.oracle_simplified = "upstream_oracle_required"` 且 `oracle_pytorch.py` 主体包含 `F.silu`、`F.gelu` 等高层算子 → 静态检查拒绝。

## 6a. Bench 写入的 provenance 字段（只读）

一个任务的 `meta.yaml` 可能携带 bench 在语料构建期间写入的可选
provenance/状态字段。它们对**提交者而言是只读的** —— 你
绝不编写或修改它们 —— 且每一个都是可选的、隐式缺省（implicit-absent）
缺省值，因此没有它们的 `meta.yaml` 验证结果完全相同。它们不在
`REQUIRED_META_TOP_LEVEL` 中，且 `scripts/corpus/validate_tasks.py` 接受
缺失或格式良好的块。

| field group | what it records (read-only) |
|---|---|
| `phase_a_prime` | standalone-severance verdict (`status`) and a `skip_invocation` flag for tasks graded by captured golden instead of live invocation |
| `phase_b_capture` | provider-side golden-output capture metadata for `upstream_oracle_required` tasks |
| `phase_b_capture_required` | whether a task needs provider golden before it can be graded in verified mode |
| `release_scope` | corpus-management eligibility (e.g. an operator marked out of the scored set) |

这些字段喂给
[设计文档](../DESIGN.md) 中描述的正确性/资格门，以及
`ascendbench/release_gate/eligibility.py` 中的 release-gate 谓词。完整 schema 和写入规则是
bench 内部细节。

## 7. 验证器

`python scripts/corpus/validate_tasks.py`（逐任务 schema / 哈希验证；`--release-gate` 运行 S6 门）检查：
- 全部 5 个文件存在（`kernel.py` / `oracle_pytorch.py` / `meta.yaml` / `shapes.yaml` / `contract.md`）
- `meta.yaml` 匹配 schema；顶层 `cost_model` key 是**必填**的 —— 缺失块是硬错误。`cost_model.{flops,bytes}`（scalar 或 map 形式）在共享求值器下解析，且每个公式针对任务的 `shapes.yaml` knob 全集求值（§5.1）。（数据集*加载器*保持宽松 —— 缺失块产生 `TaskDef.cost_model = None`；只有验证器强制存在性。见 §5.1。）
- 顶层 `param_constraints` key 是**必填**的 —— 缺失块是硬错误，但 `is_canary` 任务豁免。运行两个验证器层：**schema-shape** 层（六个子节 `dims` / `dtypes` / `dists` / `scalars` / `tensor_invariants` / `scaffold` 存在且具有正确的容器类型）和**五子节语义**层（`dims` 谓词对每个 knob 行求值为真，`dtypes` / `dists` 与 `shapes.yaml.inputs` 一致，`scalars` 值域谓词成立，`tensor_invariants` 在物化的样本张量上成立）。一个 `scaffold: true` 块短路为单个 `scaffold` 失败（不运行语义检查）。两层都经由共享的 `ascendbench.input_spec.param_constraints` helper，因此验证器、`eval` 和 `cli analyze` 状态解析器无法漂移。（数据集*加载器*保持完全宽松 —— 缺失块产生 `TaskDef.param_constraints = None`，格式错误的块作为原始 dict 保留而不抛错。见 §5.2。）
- `kernel.py` 至少有一个 `@triton.jit` 并导出解析出的上游入口点（`meta.yaml.oracle.gpu_triton`）
- `oracle_pytorch.py` 导出 `run`
- `contract.md` 引用与 `meta.yaml` 匹配
- 磁盘上 `kernel.py` SHA 匹配 `meta.yaml.phase_a_prime.standalone_sha256`（字节锁 byte-lock）

§6a 中的只读 provenance 字段
（`phase_a_prime.skip_invocation`、`phase_a_prime.status`（含
`SKIPPED_BY_DESIGN`）、`phase_b_capture`、`phase_b_capture_required`、
`release_scope`）**不**加入 `REQUIRED_META_TOP_LEVEL`。它们
可选且带显式缺省值。这些字段的 schema-shape 和跨字段不变式
检查位于 patch 验证器
（一个 bench 内部的写入守卫层）中，它在写入到达磁盘之前
设门；这里的语料验证器忽略缺失或
已格式良好的块。
