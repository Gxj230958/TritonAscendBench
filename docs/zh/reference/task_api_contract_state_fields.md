# 任务状态字段 — bench 写入的 meta.yaml 状态机

> English: [../../reference/task_api_contract_state_fields.md](../../reference/task_api_contract_state_fields.md)

> bench 写入的 `meta.yaml` 状态字段的完整 schema + 写入方/状态机叙述。
> 任务契约（`task_api_contract.md` §6a）只承载精简的只读字段参考；
> 本文档保存维护者一侧的写入细节。

## 6a. 任务状态字段（维护者一侧写入）

下列 `meta.yaml` 字段是**可选的、维护者一侧的状态**，由 bench 自动化流水线
和各阶段 runner（`scripts/verify/standalone_freeze_verifier.py`、
`scripts/provenance/capture_golden_baselines.py`）写入。
它们驱动三判定流水线（`shapes-incomplete` / `oracle-mismatch`
/ unfixable）以及发布范围（release-scope）资格门。下面四个字段组
全部**可选且带显式默认值**：一份早于本节产生的任务 `meta.yaml`
（即更早编写的 100 题语料）加载和校验时的行为与之前完全一致 —— 默认值是
*隐式缺省*，绝不是错误。这些字段由自动化写入，从不手工编写。

这些字段**不会**被加入 `REQUIRED_META_TOP_LEVEL`（先例：现有的
`phase_a_prime` 块一直都是可选的）。这些字段存在时的 schema 形状检查，
位于 patch validator（一个 bench 内部的写入守护层）中 —— 该层会在
畸形写入落盘之前就将其拒绝。`scripts/corpus/validate_tasks.py`
会忽略缺省 / 格式良好的块。

### 6a.1 `phase_a_prime.skip_invocation`

| key | value |
|-----|-------|
| **type** | `bool` |
| **required** | optional |
| **default** | `false`（视为缺省） |
| **written by** | bench 自动化在 `oracle-mismatch` 判定下做出 `defer-to-golden` 决策时写入 |
| **read by** | `scripts/verify/standalone_freeze_verifier.py` `s2_verify_task` 提前返回；资格门（`ascendbench/release_gate/eligibility.py`） |

**语义。** 当为 `true` 时，phase A' GPU S2 verify 步骤**不会**
在隔离子进程中调用算子。取而代之，它记录
`phase_a_prime.status = SKIPPED_BY_DESIGN`，并将离线 golden
采集（phase B，见 §6a.3）作为唯一真值来源。这处理了
`upstream-not-invocable-here` 的情形 —— 即某个算子的上游入口点
无法在 GPU verify 主机上独立调用（例如需要分布式 launcher、
自定义 CUDA 构建，或一个私有依赖）。

`skip_invocation: true` **必须**同时附带
`phase_a_prime.status = SKIPPED_BY_DESIGN` 以及一个非空的
`phase_a_prime.reason`。一个同时带 `skip_invocation: true` 且
`status: PASS` 的任务是 schema 违例（跨字段不变式，§6a.6）。

### 6a.2 `phase_a_prime.status` —— `SKIPPED_BY_DESIGN` 枚举扩展

| existing values | `PASS` / `BLOCKED` / `UNRUN` |
|-----------------|------------------------------|
| **new value** | `SKIPPED_BY_DESIGN` |
| **written by** | `phase_a_prime.update_meta_phase_a_prime`（现有写入方） |
| **read by** | 资格门；`scripts/corpus/validate_tasks.py` golden 存在性检查 |

**语义。** `SKIPPED_BY_DESIGN` 是真值来源委派标记：
phase A' 有意地没有运行隔离调用；离线的 phase B
golden（§6a.3）才是权威。资格门将此状态视同 `PASS`，
*仅当*对应的 `phase_b_capture.status`
记录了一次成功的采集（`DONE`）时。

当 `phase_a_prime.status = SKIPPED_BY_DESIGN` 时，`phase_a_prime.reason`
（自由格式字符串）是**必需的**，且必须解释为何该算子
无法独立调用。

### 6a.3 `phase_b_capture` 块

针对单个任务的离线 phase B golden 采集的审计轨迹记录。

| key | type | required | default | semantics |
|-----|------|----------|---------|-----------|
| `phase_b_capture.status` | string 枚举 `PENDING` / `DONE` / `FAILED` | optional | 缺省（= 未尝试采集） | 采集状态机 |
| `phase_b_capture.upstream_callable.module` | string，规范的点分 Python 路径 | optional | 缺省 | 哪个上游模块导出了要采集的入口点 |
| `phase_b_capture.upstream_callable.function` | string，合法的 Python 标识符 | optional | 缺省 | 在该模块下调用哪个函数 |
| `phase_b_capture.reason` | string | optional；当 `status` 为 `FAILED` 时**必需** | 缺省 | 自由格式的失败 / 跳过说明 |
| `phase_b_capture.captured_at` | ISO 8601 UTC 时间戳（`YYYY-MM-DDTHH:MM:SSZ`） | optional；当 `status` 为 `DONE` 时**必需** | 缺省 | 采集成功的时刻 |

**写入方。**

- bench 自动化在一次 `defer-to-golden` patch 上写入初始的
  `{status: PENDING, upstream_callable: {module, function},
  reason: …}`。`upstream_callable.{module, function}`
  来源于双锚点解析器（dual-anchor resolver）的证据。
- `scripts/provenance/capture_golden_baselines.py` 在采集子进程成功时
  将 `status` 更新为 `DONE`（并设置 `captured_at`），或在采集失败时
  更新为 `FAILED`（并设置 `reason`）。`upstream_callable` 子块不会被
  采集脚本重写 —— 它是 orchestrator 编写的意图。

**读取方。** 资格门；`scripts/corpus/validate_tasks.py` golden
目录存在性检查（一次 `DONE` 采集意味着一个已填充的 `golden/`
目录）。

**`upstream_callable.module` 格式。** 一条规范的点分 Python 模块
路径：由单个点分隔的非空合法 Python 标识符序列，开头 / 结尾
不带点，内部不含空白或斜杠。示例 —— 合法：`vllm.model_executor.layers.layernorm`、
`sgl_kernel.activation`；非法：`vllm/layernorm`、`vllm..layernorm`、
`3vllm.layernorm`、`vllm. layernorm`。规范路径的强制校验
由 Layer 5 patch validator
（一个 bench 内部的写入守护层）执行。

**`upstream_callable.function` 格式。** 一个单独的合法 Python
标识符（不含点、不含空白，必须满足 `str.isidentifier()`）。
多段点分名称应归属于 `module`，而非 `function`。

### 6a.4 `phase_b_capture_required`

| key | value |
|-----|-------|
| **type** | `bool` |
| **required** | optional |
| **default** | `false`（视为缺省） |
| **written by** | orchestrator 在 `oracle-mismatch` 或 `upstream-not-invocable-here` 决策下做 `defer-to-golden` patch 时写入 |
| **read by** | 资格门（`ascendbench/release_gate/eligibility.py::release_status_pass`） |

**语义。** 当为 `true` 时，任务的发布资格要求
`phase_b_capture.status == DONE`（即 golden 已被采集且
该算子的权威 oracle 已落盘）。当为 `false`（默认）时，
资格判定谓词完全不查询 `phase_b_capture`。

这个布尔值是将离线 golden 显式提升进
发布门合取谓词的开关 —— 没有它，一个
`oracle-mismatch` 任务可能仅凭 phase A' 结果就被错误地标记为
`release_status = PASS`，而它的 golden 实际仍然缺失。

### 6a.5 `release_scope` 块

针对某个被 orchestrator 标记为 LLM patch 流程不可修复、并从发布范围
移除的任务（`--mark-unfixable` 流程）的显式排除记录。

| key | type | required | default | semantics |
|-----|------|----------|---------|-----------|
| `release_scope.excluded` | `bool` | optional | 缺省（= `false`） | sentinel —— 任务被排除在发布评分之外 |
| `release_scope.reason` | string | optional；当 `excluded` 为 `true` 时**必需** | 缺省 | 自由格式的、人类可读的排除原因 |
| `release_scope.excluded_at` | ISO 8601 UTC 时间戳（`YYYY-MM-DDTHH:MM:SSZ`） | optional；当 `excluded` 为 `true` 时**必需** | 缺省 | 记录该排除的时刻 |
| `release_scope.excluded_by` | string，标识执行者（例如 `fix_orchestrator/automation`、operator handle） | optional；当 `excluded` 为 `true` 时**必需** | 缺省 | 谁记录了该排除 |

已上线的写入方会原子性地构造这个四键块；部分写入
（例如有 `excluded: true` 而无 `reason`）不会由该 helper 产生，
并且会被 Layer 5 patch validator 拒绝。一个
`release_scope.excluded` 为 `true` 的任务会被
`is_score_eligible`（提前退出）从资格分母中剔除，因此它不会影响
发布通过率。

### 6a.6 跨字段不变式

这些不变式将四个字段组绑定在一起。它们在*写入*时由
patch validator（Layer 5）强制执行，在*门控*时由
资格判定谓词强制执行。validator（`scripts/corpus/validate_tasks.py`）
不会重新推导它们 —— 一个畸形的跨字段组合不可能落入
`meta.yaml`，因为该 patch 会在落盘之前于 Layer 5 被拒绝。

1. `phase_a_prime.skip_invocation == true` **⇒**
   `phase_a_prime.status == "SKIPPED_BY_DESIGN"` **且**
   `phase_a_prime.reason` 是一个非空字符串。
2. `phase_a_prime.status == "SKIPPED_BY_DESIGN"` **⇒**
   `phase_a_prime.skip_invocation == true` **且**
   `phase_a_prime.reason` 是一个非空字符串。
3. `phase_b_capture.status == "DONE"` **⇒**
   `phase_b_capture.upstream_callable.module` 是一条规范的点分
   Python 路径 **且** `phase_b_capture.upstream_callable.function` 是
   一个合法的 Python 标识符 **且** `phase_b_capture.captured_at` 是一个
   ISO 8601 UTC 时间戳。
4. `phase_b_capture.status == "FAILED"` **⇒**
   `phase_b_capture.reason` 是一个非空字符串。
5. `phase_b_capture_required == true` **且**
   `phase_b_capture.status != "DONE"` **⇒** 该任务为
   `release_status = BLOCKED`（而非 `PASS`）；由资格门强制执行
   这一点。
6. `release_scope.excluded == true` **⇒**
   `release_scope.reason` 是一个非空字符串 **且**
   `release_scope.excluded_at` 是一个 ISO 8601 UTC 时间戳 **且**
   `release_scope.excluded_by` 是一个非空字符串。该任务会被
   从资格分母中排除（`is_score_eligible` 中的提前退出）。

**ISO 8601 时间戳格式。** 本节中所有 `*_at` 时间戳都由
bench 自动化以精确如下的形式写入：`YYYY-MM-DDTHH:MM:SSZ`（UTC、
秒级分辨率、结尾带 `Z`）。`±HHMM` 数值偏移形式**不会**
由 bench 产生 —— 读取方应按照 ISO 8601 标准接受任一形式，
但 bench 只写入 `Z` 形式。
