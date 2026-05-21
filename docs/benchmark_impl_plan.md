# 910B NPU Triton Migration Benchmark — 实施方案 v4.13 ✅ FROZEN

> **v3.2 ERRATA — AscendC removed.** This is a GPU-triton → NPU-triton *migration* benchmark; there is no AscendC native baseline, no `ascendc_baseline.yaml`, no `in_s_native`/`oracle.ascendc`. Correctness is the scored axis; performance is raw NPU latency (deferred to 910B). AscendC-specific text below is **superseded** and kept only as historical context.


> **状态：✅ FROZEN — codex round 19 终极判定：整体 4.94/5，RH 子域 4.86/5，0 P0 + 0 P1**
> 起草：2026-05-14；冻结时间：2026-05-14（19 轮 review 完整迭代后）
> 历史轨迹：v1 (2.7) → v4.5 (4.95 整体 frozen 但 RH 仅 4.20) → v4.13 (4.94/4.86 ✅ 完整 frozen)
> 共 19 轮 codex review 累计 ~150 条 finding 全采纳，无反驳
> 定位锚：**benchmark provider**（类比 KernelBench），**不写 NPU 实现**

## codex round 19 终极判定原文

> "v4.13 已把 round 18 的 6 个实质 blocker 补齐，且补法方向正确：numeric fast-fail 不再依赖 stdout、L1 oracle 不再过度简化、cross-oracle drift 不再默认 PyTorch 为真理、candidate-allocated ABI 下 sentinel 统一为 soft signal、hidden generator 同时随机 shape/value/edge、candidate stdout 与 verdict 解耦。**可以判定 frozen，进入 Phase 0/实现落地。**"

## v4.13 相对 v4.12 的 6 条修订（codex round 18 全采纳）

| # | P 级 | 内容 |
|---|---|---|
| **R18-S1** | P1 [confirmed] | **CI fast-fail 改 `sampled numeric allclose`**（不是 stdout 文本对比）：抽样 raw element + `torch.allclose` 判；任一样本 fail → `FAIL_QUICK`；样本通过仍**必须**走 full `allclose`（fast-pass 不存在）|
| **R18-S2** | P1 [confirmed] | **L1 oracle 三档分类**：`meta.yaml.oracle_simplified ∈ {simple_pytorch_ok, formula_exact_manual, upstream_oracle_required}`；量化/RNG/topk/layout 必须 `upstream_oracle_required`；如 `silu_and_mul` 可用 `F.silu(x[..., :d]) * x[..., d:]`（依据 vLLM activation.py:137） |
| **R18-S3** | **P0** [confirmed] | **Cross-oracle drift 不预判谁错**：三方不一致 → 该题不进 `S_native` + 冻结 semantic contract + owner 调研；**不自动认 PyTorch ref 是真理**（避免与"迁移 vLLM/SGLang 真实 kernel"定位冲突） |
| **R18-S4** | P1 [confirmed] | **A5 output sentinel 口径统一为 soft signal**：v4.12 攻击面矩阵 "runner sentinel hard" vs §5b.5 "byte-pattern flag" 冲突；统一为 `performance_flags.append("suspected_buffer_reuse")` 不 hard fail |
| **R18-S5** | P1 [confirmed] | **Hidden generator 三随机 + 日志脱敏**：hidden adversarial 同时随机 shape / value / edge；hidden 失败日志只给 aggregate reason（不含 tensor 数值）|
| **R18-S6** | P1 [confirmed] | **Candidate stdout 不进 correctness verdict**：candidate 进程 stdout 仅作 capped ≤64KB debug log；hidden case 禁打印 expected/actual tensor |

## v4.12 相对 v4.11 的 1 条修订（codex round 16 全采纳）（原文保留）

（删除原 v4.12 标题，保留原历史描述）

# v4.12 历史

> **状态：v4.12 — 基于 codex round 16 客观采纳 1 条 P1 blocker（warm-up vs measured input 分离）**
> 起草：2026-05-14
> 历史：... → v4.5 (4.95 整体 frozen) → ... → v4.11 (4.90 / 4.78 RH) → **v4.12（修最后 1 P1）**
> codex round 16 明确："修完这一条，我会判 frozen"
> 共 15 轮 codex spec review + 1 轮 reward hacking 专项，客观采纳 ~144 条 finding
> 定位锚：本项目 = **benchmark provider**（类比 KernelBench），**不写 NPU 实现**

## v4.12 相对 v4.11 的 1 条修订（codex round 16 全采纳）

| # | 内容 |
|---|---|
| **R16-P1-1** | Timing pass **warm-up 与 measured 输入分离**：warm-up 用 `warm_inputs`（同 shape 不同随机值），measured 重生成 `meas_inputs`（fresh），防 B4 warm-up 预算 hack；measured 每 10 trial 抽样 output，runner 对比 `golden_meas_outputs` 验 correctness，防 measured 阶段返回 dummy 数据 |

## v4.11 相对 v4.10 的 3 条修订（codex round 15 全采纳，0 P0）

| # | 内容 |
|---|---|
| **R15-P1-1** | Artifact provenance 单一权威：active spec **完全删除两阶段 allowlist 授权流程**；verified runner 实现只信 broker provenance + ELF signature；v4.3-v4.7 含混描述标为历史档案 |
| **R15-P1-2** | Output ABI metadata 分离：trusted shim **序列化前**检查 device/dtype/shape/stride/storage_offset/data_ptr/is_contiguous/input_alias，通过 runner-controlled signed pipe 输出 metadata；safetensors 只承载数值 |
| **R15-P1-3** | Timing pass warm-up/measured 分两阶段：warm-up 10 trials 含 JIT 不计时 → synchronize → measured 100 trials；scrub cross-task cache 在 warm-up 前 |

## v4.10 相对 v4.9 的 7 条修订（codex round 14 全采纳，0 P0）

| # | 内容 |
|---|---|
| **R14-P1-1** | Sandbox bundle visibility 传导：明示 verified runner 按 `visibility: public/runner_only/hidden_commitment` 过滤；mount 时只挂 public + solution.py，不挂 runner_only oracle |
| **R14-P1-2** | `meta.yaml.interface.alias_map.output` 改 `wo_candidate_allocated`（v4.9 仍 `wo_pre_allocated` 与 candidate-allocated 矛盾） |
| **R14-P1-3** | Sentinel 改 byte-pattern heuristic（不 hard sentinel overwrite），与 candidate-allocated ABI 一致 |
| **R14-P1-4** | runner_checks **单源** `meta.yaml`（删 §5b.5/§5b.7 "contract.md 声明" 表述），`contract.md` 仅人类可读 |
| **R14-P1-5** | AST allowlist 补必备节点（`ast.Module/Expr/Import/ImportFrom/alias/keyword`），删 `ast.MatMult`（与 backstop "禁 Tensor.__matmul__" 矛盾，统一禁 `@` 用 `tl.dot`） |
| **R14-P1-6** | Artifact provenance **单一权威**：runner-controlled compiler broker + ELF signature（v4.3 R6-5 两阶段 allowlist + v4.7 broker 双轨清理为只用 broker；`solution_artifacts.json` 降级为 self-reported 日志） |
| **R14-P1-7** | B-P1-4 表"speedup < 0.1× → invalid"改 `performance_flags=["low_performance"]` 不 hard fail；Phase 4 v1.0 verified 增量从 "+4-6 周" 改 "+5-7 周" 同步 v4.6 改目录化后增量 |

## v4.9 相对 v4.8 的 8 条修订（codex round 13 全采纳）

| # | P 级 | 内容 |
|---|---|---|
| **R13-P0-1** | P0 | **§5b.3 Timing pass 架构图 propagation 漏改**：v4.8 改了 §5b.2 文字但忘改架构图（仍是旧 `event.elapsed_time` 测时）；v4.9 把图改为 `CLOCK_MONOTONIC_RAW` host_ms 权威 + event_ms cross-check，偏差>10% → `runtime_status=invalid_timing + performance_flags=["invalid_stream"]` |
| **R13-P1-1** | P1 | `invalid_stream` 不新增 enum，映射到 `invalid_timing` + `performance_flags` 复用 |
| **R13-P1-2** | P1 | `meta.yaml` schema：`schema_version: "1.0"` 加引号防 YAML float；`level: "L1"` 类型与 enum 一致 |
| **R13-P1-3** | P1 | `meta.yaml` 加 machine-readable `runner_checks` 字段：`expected_kernel_count_range / min_dram_bytes / min_flops / metamorphic_checks / output_format / output_allocation`（从 contract.md 迁来） |
| **R13-P1-4** | P1 | `task_manifest.json` files[] 加 `visibility: public \| runner_only \| hidden_commitment`；`oracle_pytorch.py` 标 `runner_only`；hidden shapes 用 `hidden_commitment` 仅含 sha256，不进 public bundle |
| **R13-P1-5** | P1 | Output sentinel API 二选一明确：`meta.yaml.runner_checks.output_allocation: "candidate_allocated"` → 不声称 sentinel overwrite hard check，只做 byte pattern 启发式 |
| **R13-P1-6** | P1 | AST 从 denylist 升级 **positive allowlist + 节点级**（含 `ast.MatMult` 处理 `@` / `IMPORT_ALLOW` / `CALL_ALLOW` / `ATTR_DENY_PATTERNS`）+ backstop 保留旧 denylist 双层 |
| **R13-P1-7** | P1 | §1 Phase 流程图 + 立即启动节 残留 "3-5 周/2-3 周" 旧数，统一为 4-6 周/3-4 周/5-7 周 |
| **R13-P2-1** | P2 | 编号 §13 重复（"9 轮 review loop" 与 v4.9 状态并存），清理为线性 |

## v4.8 相对 v4.7 的 16 条修订（codex round 12 全采纳）

### 新 P0（2 条致命）

| # | 内容 |
|---|---|
| **R12-P0-1** | **Pickle sandbox escape 致命漏洞**：v4.7 让 runner 读 candidate 写的 `output.pt` 是漏洞 — `.pt` 是 pickle，candidate 构造恶意 pickle 让 trusted runner `torch.load()` 反序列化时**任意代码执行**。修法：禁 `.pt/.pth/.pkl`，强制 **`safetensors`**（推荐）或 metadata.json + raw bytes + sha256（兜底）；最强方案 runner-owned shared memory。新增 §5b.1a |
| **R12-P0-2** | **Timing shim 仍漏计 non-default stream**：v4.7 `event_end.record(current)` 只测 current stream，candidate 用 non-default stream 偷计算会漏计。修法：**权威 timing 改 `CLOCK_MONOTONIC_RAW` host time + device-wide synchronize**，event 仅作 cross-check；偏差 > 10% 检查 stream 列表，发现非 runner-allocated stream → `runtime_status=invalid_stream` |

### P1 propagation 漏改（10 条）

| # | 内容 |
|---|---|
| **R12-P1-1** | `low_performance` 不入 `performance_status` enum（v4.7 矛盾）；改新加独立字段 `performance_flags: list[str]` |
| **R12-P1-2** | Similarity policy 各处统一为 flag + manual review，**不自动剔除**（v4.7 B-P1-5 表 + E1 攻击面 + §5b.11 三处口径不一致 → 全统一）|
| **R12-P1-3** | `meta.yaml` schema 严格化：`schema_version` 必须 str（避免 YAML float 解析）；`level` 类型与 enum 一致；`metamorphic_checks / expected_kernel_count_range / min_dram_bytes / min_flops` 等 runner 需要的字段从 `contract.md`（人类可读）**移到 `meta.yaml`/或新增 `contract.yaml`** machine-readable |
| **R12-P1-4** | `task_manifest.json` files[] 补 `visibility: public\|runner_only\|hidden_commitment` 字段；hidden shape/material 永不进 public bundle |
| **R12-P1-5** | Output sentinel API 二选一（v4.7 仍冲突）：选 1 — runner-preallocated outputs（API 显式传 outputs）才能做 sentinel overwrite；或选 2 — candidate-allocated outputs 不声称 sentinel overwrite hard check，改为 byte pattern 启发式 |
| **R12-P1-6** | AST denylist 升级 positive allowlist + AST 节点级（Call/Attribute/MatMult/Subscript）：`exec/eval/compile/__import__` 是 Call 不是 Import；`@` 运算符是 `ast.MatMult`；`getattr` 仅允许 whitelist 名 |
| **R12-P1-7** | Artifact provenance 补 OS 级权限边界：candidate 无写 `/runner_cache/...` 权限；JIT 通过 runner-controlled compiler broker；产物签名验证 |
| **R12-P1-8** | 全文资源数字同步：§2.2 合计 / §9 / §13 仍残留 v4.5 旧数（7-13 周 / 11-18 人月 / 12-18 周）；全部改为 **v0.1: 9-15 周 / 12.5-21.5 人月 / v1.0: 14-20 周 / 17.5-27.5 人月**；§2.1 Phase 1 行从 3-5 改为 4-7 人月 |
| **R12-P1-9** | §3.1 Must Have 写 "5 个 spec 文件"与 §11 总 11 个不一致；全篇统一 **11 个 spec**（5 基础 + 6 reward/security）|
| **R12-P1-10** | `ascendc_baseline.yaml` 必备性：Phase 1.8 + 目录注释 + Go/No-Go 仍写"仅 S_native"，与"所有 task 都有 `available: false`"矛盾；全篇统一所有 task 都有该文件 |

### P2（4 条）

| # | 内容 |
|---|---|
| **R12-P2-1** | v4.7 计数错：声称"14 条"实际 15 条（2 P0 + 10 P1 + 2 P2 + 1 隐性 P2），v4.8 正确数 |
| **R12-P2-2** | 历史 §10/§11/§12 表述区分清楚"历史 vs 当前活跃" |
| **R12-P2-3** | candidate sandbox mount 明示 kernel.py / meta.yaml / shapes.yaml 是 **public task metadata**；hidden shapes 不在其中 |
| **R12-P2-4** | profile verdict R11-P2-2 编号错（写成 R11-P2-4） |

## v4.7 相对 v4.6 的 14 条修订（codex round 11 全采纳）

### P0 致命修订（2 条）

| # | 内容 |
|---|---|
| **R11-P0-1** | **删除 candidate sandbox mount `golden_outputs.pt`**（v4.6 §5b.3 致命设计错误，等于把答案直接给 candidate）；oracle 计算 + allclose + verdict 都在 trusted runner 进程，candidate 只看 inputs |
| **R11-P0-2** | **重写 cross-process timing 架构**：v4.6 parent process fork 前后记 `torch.npu.Event` 不可靠包子进程 NPU work；改为 **trusted timing shim**（runner-controlled binary 包裹 candidate function，与 candidate 同 process 同 stream 记 event）+ runner-controlled profiler trace 作 cross-check；correctness/timing 分两个 candidate process 跑 |

### P1 修订（10 条）

| # | 内容 |
|---|---|
| **R11-P1-1** | **`meta.yaml` 完整 schema**：v4.6 没枚举 11 字段；v4.7 给完整 YAML schema（每字段类型 / 必填 / enum / 跨文件一致性），并锁定 `meta.yaml` 为 source of truth |
| **R11-P1-2** | **`task_manifest.json` 升级 per-file**：v4.6 仅 bundle SHA；v4.7 加 per-file role / sha256 / size / required / canary flag / hidden_only / in_s_native |
| **R11-P1-3** | **Output sentinel API 冲突解开**：v4.6 buffer poison 与返回式 API 矛盾；v4.7 改为 candidate-allocated output + runner 检查 sentinel byte pattern + per-dtype sentinel（float NaN / int MAX_MIN / bool mask） |
| **R11-P1-4** | **Artifact provenance 解 hidden shape JIT 冲突**：v4.6 预登记 `.so` 与 hidden shape JIT 冲突；v4.7 改为 runner-controlled compiler driver + provenance signature（IR hash + compiler version + sm_arch 加密签名注入 ELF section） |
| **R11-P1-5** | **`min_coverage_for_ranking` per-track 分母**：v4.6 写"99 题中 70 题"与 native track 分母 \|S_native\|≈30-40 冲突；v4.7 拆 `native_coverage ≥ 70% of \|S_native\|` + `all_track_coverage ≥ 70/99` |
| **R11-P1-6** | **`speedup < 0.1×` 解释修正**：v4.6 写"< 0.1× 反向 hack"错；v4.7 改为 `performance_status=low_performance` 标记不 hard fail（< 0.1× 是慢不是快）；空 kernel 实际是 > 10× 已被 quarantine 抓 + 无 NPU event hard fail |
| **R11-P1-7** | **Similarity policy 矛盾修正**：v4.6 §5b.11 "trivial_copy 不进 verified ranking" 与 B-P1-5 "仅 flag" 矛盾；v4.7 统一为 flag + manual review queue，不自动剔除 |
| **R11-P1-8** | **AST denylist 扩展 + 升级 semantic**：v4.6 漏 `torch.ops/Tensor.matmul/.item()/getattr(__import__)`；v4.7 分三层：Import denylist + AST semantic rules（含 `@` 运算符 / Tensor.cpu/.numpy/.item / torch.ops 等）+ runtime sys.audit hook |
| **R11-P1-9** | **Differential rule per-contract metamorphic**：v4.6 "output diff ≥ 0.5×" 阈值对 softmax/mask/quant 误伤；v4.7 改 contract.md 声明 `metamorphic_checks:` 字段，per-task 检查规则 |
| **R11-P1-10** | **资源 / 周期数字同步**：§11 写 v4.6 8-14 周 / 12-20 人月与 §2.1/§13 旧 7-13 / 11-18 不同步；v4.7 全部同步为 9-15 周 / 12.5-21.5 人月（v0.1）和 14-20 周 / 17.5-27.5 人月（v1.0） |

### P2 修订（2 条）

| # | 内容 |
|---|---|
| **R11-P2-1** | **Spec 文件数量统一**：v4.6 出现 5/6/7/11 不同数字；v4.7 统一为 5 基础 + 6 reward/security = 11 总 spec 文件 |
| **R11-P2-2** | **Profiler verdict 改 NPU event presence + bytes/FLOPs lower bound**：v4.6 "op count 与 oracle 偏离 5×" 不成立（PyTorch vs Triton op 结构天然不同）；v4.7 改为 NPU kernel event presence + task-specific expected kernel envelope + bytes/FLOPs lower bound |

### P2 隐性（1 条）

| # | 内容 |
|---|---|
| **R11-P2-3** | **`ascendc_baseline.yaml` 必备性统一**：v4.6 目录树固定列但 Go/No-Go 写"仅 S_native"；v4.7 改为所有 task 都有该文件，非 S_native 用 `available: false` |

## v4.6 相对 v4.5 的两大架构修订

### 修订 A — Task 文件改"目录 + 多文件分离"（用户提出"算子模型设计"原则）

v4.5 把 GPU triton kernel + PyTorch oracle + AscendC ref + shape configs + adversarial + 11 字段 META 全塞进单 .py，违反 single responsibility。v4.6 改成**目录结构**：

```
tasks/level1/L1_12_rms_norm/
├── kernel.py              ← 纯算子实现（@triton.jit + wrapper，可独立 import 复用）
├── oracle_pytorch.py      ← PyTorch reference 兜底（独立 oracle，runner 外层加载）
├── meta.yaml              ← 结构化元数据（11 字段 + license + commit SHA + tolerance class），跨语言可读
├── shapes.yaml            ← SHAPE_CONFIGS + ADVERSARIAL cases（配置与代码分离）
├── contract.md            ← Semantic contract（人类可读 + 11 字段）
└── ascendc_baseline.yaml  ← AscendC native baseline adapter 配置（所有 task 都有，非 S_native 用 `available: false`，v4.7 R11-P2 修订）
```

参照工程界主流做法（FlagGems / Liger-Kernel / FBGEMM / PyTorch native ops）：算子 vs 配置 vs 测试 vs 文档严格分离。**单文件混合**模式（KernelBench / TritonBench）仅适用于简单 task；本 benchmark 的 11 字段 contract + multi-shape + adversarial + AscendC ref 复杂度超出单文件。

**v4.7 R11-P1 — `meta.yaml` 完整 schema**（解决 v4.6 没枚举的问题）：

```yaml
# 必备字段（task validator enforce）
schema_version: "1.0"                        # str（v4.9 R13-P1 修：必须加引号防 YAML 解析成 float）
benchmark_version: "v0.1"                    # str, 必填
task_id: "L1_12_rms_norm"             # str, 必填, 必须与目录名一致

# Identity
level: "L1"                                  # str（v4.9 R13-P1 修：类型与 enum 一致；enum {L1,L2,L3,appendix,canary}，原 int+混合字符串 enum 矛盾）
idx: 12                                       # int, 必填, 与 task_id 后缀一致
name: "_rms_norm_kernel"                     # str, 必填
repo: "vLLM"                                  # str, 必填, enum {vLLM, SGLang, FBGEMM, 字节中台}
dtype_class: "VV"                             # str, 必填, enum {VV, CV}
category: "norm"                              # str, 必填, enum {elementwise, norm, reduce, sampling, kv_cache, lora, attention, moe, state_space, ...}
npu_difficulty: "medium"                     # str, 必填, enum {low, medium, high}

# Upstream freeze（Phase 1.2 写入）
upstream:                                     # dict, 必填
  repo: "vllm-project/vllm"                  # str
  path: "vllm/.../batch_invariant.py"        # str
  commit_sha: "abc123..."                    # str, sha
  function_hash: "def456..."                 # str, sha256 of kernel source
  license: "Apache-2.0"                      # str

# Interface contract（11 字段 — v4.4 R7-6 全字段）
interface:                                    # dict, 必填
  alias_map: {input: ro, weight: ro, output: wo_candidate_allocated}  # dict[str, enum]，v4.10 R14-P1 同步 candidate-allocated
  mutability: "no_inplace"                    # str, enum
  shape_feasibility: "all_configs_verified"  # str
  correctness_oracle: "pytorch + ascendc"    # str
  baseline_eligibility: "gpu_triton + ascendc + torch_eager"  # str
  skip_reason: null                           # str | null
  stride_layout_constraints: "contiguous required"  # str
  workspace_policy: "none"                   # str, enum {none, scratch_<size>MB}
  stream_sync_policy: "runner-managed"       # str, enum
  rng_state_policy: "deterministic"          # str, enum {deterministic, requires_seed, free}

# Oracle 引用
oracle:                                       # dict, 必填
  pytorch: "oracle_pytorch.py::run_oracle"   # str, 必填
  gpu_triton: "kernel.py::run_kernel"        # str, 必填
  ascendc:                                    # dict
    available: true                           # bool
    config_file: "ascendc_baseline.yaml"     # str if available

# Tolerance class（查 shared/tolerance_policy.yaml）
tolerance_class: "vector_reduce"              # str, 必填, enum 来自 tolerance_policy

# Flags（v4.7 新增）
is_canary: false                              # bool, 默认 false
is_hidden_only: false                         # bool, 默认 false
in_s_native: true                             # bool, 必填; if true 则 oracle.ascendc.available 必须 true

# Runner verdict 字段（v4.9 R13-P1 修：从 contract.md 移到 meta.yaml machine-readable）
runner_checks:
  expected_kernel_count_range: [1, 5]         # list[int]，profiler 验证 NPU kernel launch event 数量
  min_dram_bytes: 65536                       # int，基于 input+output 总大小推算的下界
  min_flops: 524288                           # int，基于算子语义推算的下界
  metamorphic_checks:                          # list[str]，引用 shared/metamorphic_lib.yaml 中的规则名
    - "softmax_sum_to_one"
    - "differential_perturbation_general"
  output_format: "safetensors"                 # str, enum {safetensors, raw_bytes_sha256}
  output_allocation: "candidate_allocated"     # str, enum {candidate_allocated, runner_preallocated}（v4.9 R13-P1 二选一）
```

**Authority**：`meta.yaml` 是 source of truth；`contract.md` 是人类可读补充（语义不变量、边界 case 描述），与 `meta.yaml` 字段冲突时以 `meta.yaml` 为准（v4.7 R11-P1 锁定）。v4.9 修订：runner 自动判定所需字段全部迁到 `meta.yaml`，`contract.md` 仅做说明。

**`task_manifest.json` schema 升级**（v4.7 R11-P1 修订：v4.6 仅 bundle SHA，不够）：

```json
{
  "schema_version": "1.0",
  "benchmark_version": "v0.1",
  "tasks": [
    {
      "task_id": "L1_12_rms_norm",
      "level": 1,
      "is_canary": false,
      "is_hidden_only": false,
      "in_s_native": true,
      "files": [
        {"name": "kernel.py",             "role": "kernel",   "sha256": "...", "size": 1234, "required": true, "visibility": "public"},
        {"name": "oracle_pytorch.py",     "role": "oracle",   "sha256": "...", "size": 567,  "required": true, "visibility": "runner_only"},
        {"name": "meta.yaml",             "role": "meta",     "sha256": "...", "size": 234,  "required": true, "visibility": "public"},
        {"name": "shapes.yaml",           "role": "shapes",   "sha256": "...", "size": 345,  "required": true, "visibility": "public"},
        {"name": "shapes_hidden.yaml",    "role": "shapes_hidden", "sha256": "<commitment>", "size": 0, "required": false, "visibility": "hidden_commitment"},
        {"name": "contract.md",           "role": "contract", "sha256": "...", "size": 4567, "required": true, "visibility": "public"},
        {"name": "ascendc_baseline.yaml", "role": "ascendc",  "sha256": "...", "size": 123,  "required": true, "visibility": "public"}
      ]
    },
    ...
  ]
}
```

`init-solution` CLI 打包的 task_bundle.tar.gz 包含上述 6 文件 + manifest fragment。

### 修订 B — Reward hacking 防御从 15 原则扩到 24 原则（codex 专项 review 揭示）

| # | P 级 | 修订内容 | 落地位置 |
|---|---|---|---|
| **B-P0-1** | P0 [confirmed] | **Process isolation contract**：candidate、reference/oracle、comparison、timing 分进程；candidate 只拿 tensor inputs，不 mount oracle/task source；parent 进程负责 allclose/hash/timing/profiler verdict | §Phase 2 evaluator 架构 + §5b §1 |
| **B-P0-2** | P0 [confirmed] | **Runner-owned timing/comparison**：candidate sandbox 内禁 `torch.npu.Event / time.perf_counter / LD_PRELOAD`；timing 由 runner 外层包 candidate；correctness/timing 阶段分开（candidate exit + kill process group 才进下一阶段，防 stateful hack）| §5b §2 |
| **B-P0-3** | P0 [confirmed] | **AST + import/runtime hook denylist**（具体规则）：禁 `exec/eval/compile/importlib/ctypes/cffi/cpp_extension/load_inline/subprocess/socket/requests/urllib/threading/multiprocessing/asyncio/timing monkey patch/torch.nn.functional.*/native AscendC ops/oracle imports`；AST 扫描 + import hook 双层 | §5b §3 + §5b AST denylist 全表 |
| **B-P0-4** | P0 [confirmed] | **Artifact provenance**：verified runner 只允许 runner-controlled triton-ascend JIT cache 产物；任意 `.so / C extension / prebuilt binary / cpp_extension / ctypes load` 不进 NPU Triton track allowlist；submitter 提供的 `solution_artifacts.json` **不被信任**（v4.4 R7-4 已含但需 enforce 拒绝任意 native artifact） | §5b §4 |
| **B-P0-5** | P0 [confirmed] | **Correctness hardening**（5 子项）：① input hash before/after（v4.5 已含）② output buffer poison（写 sentinel value 让 reference reuse 被 detect）③ output dtype/device/shape/stride/layout/storage alias 严格检查 ④ 3 组 randomized differential inputs，output 必须随 input 扰动合理变化 ⑤ NaN/Inf 处理一致性 | §5b §5 |
| **B-P1-1** | P1 [confirmed] | **`min_coverage_for_ranking`**：native track 主榜要求 `correctness_rate ≥ 70%`（70/(\|S_native\|) 题正确才进 verified ranking）；否则只进 self-reported 不进 verified | §5 scoring + §Phase 4 |
| **B-P1-2** | P1 [confirmed] | **Submission throttling + sybil 检测**：同 submitter 同 task ≤ 3 次/周；按 org identity（GPG key / email domain）汇总；多账号短期协同提交 pattern 自动 flag | §Phase 4.7 verified runner |
| **B-P1-3** | P1 [confirmed] | **Canary tasks 5-10 个**：埋设 identity hack / zero-out / cache reuse / non-default stream / fallback / shape-route 陷阱题；不进 score 公式，纯"诚信度量"；canary 高分自动 flag 整 submission 可疑 | §Phase 1 batch 5 新增 |
| **B-P1-4** | P1 [confirmed] | **Absolute sanity bound**（v4.10 R14-P1 同步）：① `runtime_status=invalid_timing` hard fail ② 无 NPU kernel event in profiler → hard fail ③ `hardware_utilization > 100%` → hard fail ④ speedup > 10× → quarantine + proof ⑤ **speedup < 0.1× → `performance_flags=["low_performance"]`，不 hard fail**（< 0.1× 是慢实现非 hack） | §5 + §5b §6 |
| **B-P1-5** | P1 [likely] | **Solution similarity check**（**v4.8 R12-P1 统一**：所有位置 flag + manual review，**不自动剔除**）：① solution.py 与 kernel.py byte-level / AST 相似度 > 95% → `trivial_copy` flag + 进 manual review queue（不剔除）② 跨 submitter solution 相似度高 → 协同 flag + 人工 audit | §Phase 4 verified runner |

### KernelBench 直接借鉴（v4.6 新增）

- **3 个对抗 kernel 改造为 NPU 版**：`test_kernels/{result_reuse_kernel, zero_out_kernel, non_default_stream_kernel}.py` → NPU 版作为 `test_eval_adversarial.py` 单元测试
- **AST 规则完整列表**（参考 KernelBench `kernel_static_checker.py`）：try/except fallback / pass bypass / 装饰器跳过 / torch high-level op / stream/thread injection / lazy tensor / timing monkey patch / precision downgrade — v4.6 §5b AST denylist 完整列出

---

## 0. 项目定位（v4 核心修正，绝对不能再搞错）

## v4.5 相对 v4.4 的 3 条修订（codex round 8 全采纳）

| # | P 级 | 修订内容 | 落地位置 |
|---|---|---|---|
| R8-P1-1 | P1 [confirmed] | **runtime_status enum 加 `invalid_timing`**：v4.4 §5 R7-3 写 `t_npu ≤ 0 / NaN → runtime_status=error`，但 enum 是 `{ok, timeout, oom, compile_fail}`，没有 `error` → JSON schema/validator 无法实现。改：enum 加 `invalid_timing`；不用泛化 `error`（避免与 `correctness_status=error` 混淆错误来源） | §5 status enum + t_npu 处理 |
| R8-P1-2 | P1 [confirmed] | **hidden logs 分两层**：v4.4 §4.8 hidden shapes 未公开（防 overfit）vs §4.10 完整 raw JSON 全公开（含 hidden shape 维度+timing） → 矛盾；首批提交后 hidden set 即泄露。改：public logs 公开全 timing；hidden logs 仅公开 aggregate + sealed hash commitment；hidden set 按 epoch 轮换 + retired 后解封 | §4.10 |
| R8-P3 | P3 [confirmed] | **周期统一为 7-13 周**：§0 写 v0.1 8-13 周；§2.2/§9 写 7-13 周。按 phase 加和 7-13 更自洽；统一到 7-13 | §0 承诺等级表 |

## v4.4 相对 v4.3 的 6 条修订（codex round 7 全采纳）

| # | P 级 | 修订内容 | 落地位置 |
|---|---|---|---|
| R7-1 | P1 [confirmed] | **§6 verified runner 流程 step 7 漏改修复**：v4.3 改了 §4.9 为 "100% 重跑"但忘改 §6 step 7（仍是 "top-K 100% / 非 top-K 抽样 30%"，与 R6-1 自相矛盾）；本版统一为 "所有 verified 100% 重跑；top-K 额外触发 audit-rerun；非 top-K 不 sampling 作为 ranking 数据来源" | §6 verified runner step 7 |
| R7-2 | P1 [confirmed] | **`signature.sig` 循环定义解开**：files[] 列出 tar 内 payload 文件 sha256，但若包含 signature.sig 自身就会循环（签名内容里有签名文件 hash）。明确：files[] **不含** manifest.json + signature.sig；tar 允许集 = `{manifest.json} ∪ {signature.sig} ∪ files[]` 一一对应 | §6 schema |
| R7-3 | P2 [likely] | **`t_npu_triton=0` / NaN / Inf 边界**：max(ε, min(5, t_a/t_n)) 前必须验 t_n 合法性；t_n ≤ 0 或 NaN → runtime_status=error；t_n=+∞ → runtime_status=timeout；t_n < 1μs → performance_status=quarantined | §5 公式前置检查 |
| R7-4 | P2 [confirmed] | **Triton artifact allowlist 信任边界**：v4.3 §5 让 submitter 包带 solution_artifacts.json 进 runtime_env/，但 verified runner 不能信任 submitter 提供的 allowlist；verified 模式 runner **必须自建** allowlist（no-network build sandbox 重新生成），不读 submitter 包内的；self-reported 模式可读 submitter 包内的 | §5 fallback detector |
| R7-5 | P2 [confirmed] | **§4 v1.0 资源数字不一致**：§0/§2.1 算出 v1.0 总 15-24（即 v0.1 11-18 + 4-6），但 §4 仍写 "+6-9 人月"，需统一为 "+4-6 人月"；§2.3 硬件表只写 1 张 910B，需拆 v0.1 1 张 / v1.0 2 张 | §4 v1.0 任务表 + §2.3 |
| R7-6 | P3 [confirmed] | **旧 API 残留清理**：Phase 1 task 文件示例还含 `def npu_triton_impl(x, bias, output)` 这种 task-内置 submitter 入口（v3.1 旧设计），与 v4.1 重设计的"task/solution 分离"冲突；§3.1 Must Have 还写 `run_and_check.py / run_full_benchmark.py` 旧 CLI 名 → 统一为 task bundle 只含 oracle/spec，solution.py 由 `npu-bench init-solution` 生成；CLI 用 6-CLI 新名 | Phase 1 task 模板 + §3.1 |

## v4.3 相对 v4.2 的 6 条修订（codex round 6 全采纳）

| # | P 级 | 修订内容 | 落地位置 |
|---|---|---|---|
| R6-1 | P1 [confirmed] | **Verified rerun 口径修正**：v4.2 §4 写"所有 verified 均重跑"但 §4.9 写"top-K full + 30%"自相矛盾。统一为 **"所有 verified 提交 100% 官方重跑"**；top-K full rerun / 第三方 audit 仅作**二次审计**，不替代 100% 重跑 | §Phase 4.9 |
| R6-2 | P1 [likely] | **`capped_geomean` 数学防御**：ok case 公式从 $\min(5, t_a/t_n)$ 改为 $\max(\epsilon,\ \min(5,\ t_a/t_n))$，防 $t_{npu}=0$ 或异常大值导致 log 溢出 | §5 公式 |
| R6-3 | P1 [confirmed] | **Submission manifest 改 canonical `files[]`**：列出 tar 内**所有**文件 path + sha256 + size；signature.sig 签 deterministic JSON（key 字典序），任何 tar 内文件不在 `files[]` → reject | §6 schema |
| R6-4 | P2 [likely] | **Sandbox candidate 严格 no network**：candidate namespace `--network=none`，结果上传移至 **runner 外层**（candidate exit 后由 sandbox 外读取 result 并上传） | §Phase 4.12 |
| R6-5 | P2 [likely] | **Triton artifact 两阶段 allowlist**：build 阶段隔离 sandbox 收集 triton-ascend 编译产物 sha256 → 登记 `solution_artifacts.json`；run 阶段仅允许已登记 artifacts + frozen runtime libs | §5 fallback detector |
| R6-6 | P3 [confirmed] | **清理两处不一致**：删除 `performance_status=capped`（状态 enum 保留 `ok/suspicious/quarantined`）；资源表统一为 v0.1 **11-18 人月** / v1.0 **15-24 人月** | §5 status + §0 + §2.1 |

## v4.2 相对 v4.1 的 4 条 P2 polish（codex round 5 全采纳）

| # | P 级 | 修订内容 | 落地位置 |
|---|---|---|---|
| P2-1 | submission artifact 安全 | `signature.sig` 明确签 `manifest.json + payload hash list`；`manifest.json` 加 `result_shas / log_shas / runtime_env_shas / schema_version`；**tar 解包安全**：禁 symlink/hardlink/path traversal、限制 tar 总大小（≤ 500 MB）、只允许白名单路径（`manifest.json / solutions/ / results/ / runtime_env/ / logs/ / signature.sig`）| §6 Submitter Artifact Schema 强化 |
| P2-2 | verified board 数据来源严格定义 | 写死：**"verified board 排名数据只来自官方 frozen runner 重跑"**，submitter 自报仅用于差异检测；top-K rerun / 第三方 audit 是二次审计，**不替代** "所有 verified 提交均官方重跑"；sandbox 显式列出 6 项：no network / read-only FS / seccomp syscall allowlist / device allowlist / cgroup memory limit / cgroup CPU limit | §Phase 4 v1.0 + 新 §11 verified runner spec |
| P2-3 | 数学精确性 | `capped_geomean_native` 公式：$s_t = \min(5, t_{ascendc}/t_{npu})$ if `status=ok`，else $s_t = \epsilon = 0.01$；最终 $\exp(\text{mean}(\log(s_t)))$；fallback detector 从 **blacklist 升级为 allowlist**：允许 NPU Triton runtime 与编译产物 .so，记录 loaded shared objects + profiler op events，不在 allowlist 的 op 即 fallback hack | §5 scoring spec 公式 + fallback detector |
| P2-4 | 测试覆盖 + 人月口径同步 | 新增 4 个 unit test：`test_submission_artifact_safe_unpack` / `test_cli_e2e_package_validate` / `test_compile_cache_key_isolation` / `test_profile_fallback_detector`；**Phase 2 人月正文 1.5-2.5 与资源总表的 1-1.5 不一致**，统一到 **1.5-2.5** | §Phase 2.20 + §2.1 资源表 |

## v4.1 相对 v4 的 7 条修订（codex round 4 全采纳）

| # | 严重度 | 修订内容 | v4 位置 → v4.1 章节 |
|---|---|---|---|
| E1 | P1 | 补完 **Provider Surface**：dataset.py / task_manifest.json / validate_task.py / make_hf_dataset.py / compile_cache.py / analysis.py / profile.py（NPU msprof） / leaderboard_validator.py + 完整 unit tests | §Phase 2（任务表扩） |
| E2 | P1 | **Submitter API 重设计**：task 文件 read-only + hash-verified；solution 独立 `solutions/<task_id>.py`；CLI 改为 `init-solution / run / run-suite / package / submit / validate-result`；result.json 含 task_bundle_sha + solution_sha + runner/container hash | §Phase 2 + 新 §8 |
| E3 | P0 | **两级 Leaderboard**：`self-reported`（接受 result.json）vs `verified`（官方 frozen runner 在 910B 上重跑）；提交格式 + sandbox + hidden adversarial + full rerun + raw logs 公布 + quarantine appeal | §Phase 4 + 新 §9 |
| E4 | P1 | **Score 防作弊硬化**：固定分母 `\|fixed_set\|`、失败/未提交计 0、主榜禁 `∞`、改 capped geomean + `FastP@1.0/@1.5`、NPU Triton track 禁 AscendC/torch fallback（hard fail 或 hybrid track） | §3.1 + 新 §10 scoring_spec |
| E5 | P2 | AscendC oracle reference **改名** `AscendC native baseline adapter`；S_native 进入条件：wrapper 与 PyTorch oracle 在所有 public + hidden shape 一致；每题 wrapper timebox 0.5-1 天 | §Phase 3.4 |
| E6 | P2 | **PoC Go/No-Go 加 submitter path 验证**：独立 solution artifact 加载 + task hash 篡改失败 + mock candidate 正例 + fallback/篡改负例 + result schema 校验 + NPU backend smoke（MockNpuBackend / torch_npu eager smoke 即可） | §Phase 0.6 + 0 Go/No-Go |
| E7 | P1 | **拆 self-reported vs verified 承诺等级**：v0.1 self-reported（10-17 人月 / 8-13 周 / 1 张 910B）；verified leaderboard（**16-24 人月 / 12-18 周 / 2 张 910B 或 rerun pool**） | §0 总览 + §2 资源 |

## 0. 项目定位（v4 核心修正，绝对不能再搞错）

### 0.1 本项目 = benchmark provider

类比 KernelBench：
- **KernelBench 项目本身**：提供 270 题 PyTorch ref + `eval.py` + `timing.py` + H100/A100 baseline 数据 + HuggingFace dataset
- **KernelBench 不做的**：不写 CUDA kernel。CUDA kernel 是 **LLM/工程师/研究者**（benchmark 用户）提交的，KernelBench 只评测

**本项目同理**：
- **本项目做**：99 题 GPU triton ref + Evaluator harness（接受外部提交的 NPU triton kernel）+ 三层 baseline 数据 + benchmark dataset 发布
- **本项目不做**：99 题 NPU triton-ascend 实现。NPU triton 实现是 **submitter**（华为 triton-ascend 团队 / vLLM-Ascend / sgl-kernel-npu / LLM agent / 学术研究者 / 客户）的活，本项目只评测他们提交的实现

### 0.2 角色分工

| 角色 | 职责 |
|---|---|
| **本项目 owner / 团队**（约 2-3 FTE） | 维护 benchmark 基础设施 |
| **Benchmark submitter**（华为 / vLLM-Ascend / sgl-kernel-npu / LLM agent / 研究者） | 写 NPU triton kernel 实现，提交到 benchmark，跑评测 |
| **Benchmark 用户**（决策者 / 选型方） | 看排行榜 / 报告 / 数据，做技术选型 |

### 0.3 使用场景（澄清）

```
用户对 benchmark 的使用：
1. 拿到 99 题 task 文件（每题含 GPU triton 原 kernel + PyTorch oracle + shape + adversarial）
2. 自己实现对应的 NPU triton kernel
3. 跑 evaluator：correctness 对齐 + 性能对比 GPU triton baseline
4. 评测标准：精度和性能不能低于当前 GPU triton 实现
5. 提交 result.json 到 benchmark 仓库或 leaderboard
```

### 0.4 与 v3.1 错误定位的对比

| 维度 | v3.1（错误） | v4（正确） |
|---|---|---|
| Phase 4 NPU 实现 | 6-14 周 / 17-29 人月（本项目要做） | **删除**（不是本项目活） |
| `migration_score` 评什么 | 我们写的 NPU 实现 | **submitter 写的 NPU 实现** |
| 5-7 kernel engineer | 写 NPU triton 实现 | 不需要 |
| 公开级承诺 | 40-60 人月 | **9-13 人月**（benchmark infra-only）|
| Leaderboard | 可选 | **核心交付物**（接受外部提交的关键载体） |
| 周期 | 22-36 周（满员/三线） | **8-12 周** |

## 1. 4 个 Phase（v4 简化）

```
Phase 0  →  Phase 1  ─┐
准备       Task 目录   │
1-2 周     4-6 周      ├→ Phase 3 → Phase 4
           Phase 2  ──┘   Baseline   开源化 + Leaderboard
           Harness         2-3 周    v0.1: 1-3 周 / v1.0: +5-7 周
           3-4 周
           (与 1 并行)
```
（v4.9 R13-P1 同步：v4.6 改目录化 + 加 process isolation/timing shim/AST 等后，Phase 1: 3-5→4-6 周，Phase 2: 2-3→3-4 周；v1.0 verified 增量 4-6→5-7 周）

**承诺等级**（v4.1 拆 self-reported vs verified）：

| 等级 | 总周期 | 总人月 | 硬件 | 范围 |
|---|---|---:|---|---|
| **v0.1 self-reported** | **9-15 周**（v4.7 R11-P1 同步：v4.5 7-13，v4.6 8-14，v4.7 +1 周 process isolation/timing shim/artifact provenance 实现）| **12.5-21.5** | 1 张 910B + 1 卡 GPU | task + harness + 3 层 baseline + self-reported leaderboard |
| **v1.0 verified** | **14-20 周** | **17.5-27.5** | 2 张 910B 或借机 rerun pool + 1 卡 GPU | + verified leaderboard 官方 runner 重跑 + sandbox + hidden adversarial + 第三方 audit |

---

### Phase 0：准备与决策（1-2 周）

**目标**：环境就绪 + RFC 决策 + 5 题 PoC 验证 evaluator 可行（**不再要 NPU triton kernel 跑通**，只验证 GPU triton 跑通 + harness 雏形可对它评分）。

**任务**：

| # | 任务 | 产物 | 工作量 |
|---|---|---|---|
| 0.1 | Frozen version matrix（CANN / driver / torch / torch_npu / triton-ascend / Python / OS / Docker digest） | `infra/version_matrix.toml` | 0.5 天 |
| 0.2 | GPU 实机（H100 / A100）+ 910B 实机（至少 1 张，CI 测试用） SSH + Docker 启动 | inventory | 1.5 天 |
| 0.3 | RFC_0001：开源 vs 内部、HF dataset、Leaderboard 三选一 | `RFC_0001_scope.md` | 1 天 |
| 0.4 | License 检查（KernelBench Apache 2.0 + NOTICE） | LICENSE / NOTICE | 0.5 天 |
| 0.5 | Upstream contact（vLLM-Ascend / sgl-kernel-npu / triton-ascend 维护者） | `infra/upstream_contacts.md` | 1 天 |
| 0.6 | **5 题 PoC**：验证 task 文件格式 + GPU triton 跑通 + harness 雏形评分（**不要求 NPU triton 实现**） | `tasks/poc/*.py` × 5 + smoke 报告 | 3-4 天 |
| 0.7 | CI skeleton（GitHub Actions 跑 PoC） | `.github/workflows/ci.yml` | 1 天 |

**5 题 PoC 选 5 个代表（v4 调整：不再 7 题，因为不测 NPU 实现）**：

| 能力族 | PoC 题 | 验证什么 |
|---|---|---|
| Elementwise | `L1-01 _bias_kernel` | 最小 task 文件 + GPU baseline |
| Reduce | `L1-12 _rms_norm_kernel` | 行内 reduce baseline |
| Cube matmul | `L2-10 _w8a8_block_int8_matmul` | Cube tile + INT8 量化 baseline |
| API semantics stress | `L2-22 _causal_conv1d_update_kernel` | in-place + multi-output + alias map |
| 完整 fused | `L3-04 _fwd_kernel`（Flash Attention） | UB 极限 / 完整 softmax+matmul baseline |

**人力**：1 lead + 1 NPU infra（part-time）+ 1 kernel engineer + 0.5 PM = 2.5-3 FTE × 1-2 周 = **2.5-4.5 人月**

**Go/No-Go**（v4.1 E6 增强，验证 submitter path 闭环）：
- ✅ 5/5 PoC 题：task 文件格式正确 + GPU triton 跑通 + harness 能输出 baseline timing
- ✅ RFC_0001 签字
- ✅ 至少 1 张 NPU + 1 卡 GPU 访问 OK
- ✅ **Submitter path 闭环**（v4.1 新增）：
  - 独立 `solutions/<task_id>.py` artifact 加载流程跑通
  - Task hash 篡改测试：篡改 task 文件 → harness 必须拒绝（hash mismatch）
  - Mock candidate 正例：用 PyTorch eager 包装成 mock NPU solution，跑通 evaluator 流程
  - Fallback 负例：solution 内部 `import torch; return torch.add(...)` → harness 必须 flag fallback hack
  - Result schema 校验：用 `validate-result` CLI 验证 result.json 格式
  - NPU backend smoke：`MockNpuBackend` 或 torch_npu eager 跑通 timing 路径（**不要求真 NPU triton 实现**）

---

### Phase 1：99 Task 目录生成（4-6 周，v4.6 改目录结构，可与 Phase 2 并行）

**目标**：从 `infer/vllm` + `infer/sglang` 抽取 99 题 GPU triton kernel，按 **v4.6 目录 + 多文件分离结构** 生成。每题锁 upstream commit SHA。

**v4.6 改用目录结构后的任务表**（工作量略增，因每题 6 文件）：

| # | 任务 | 产物 | 工作量 |
|---|---|---|---|
| 1.1 | `extract_kernel.py`：AST/import graph 抽取 → `kernel.py` | tool | 5 天 |
| 1.2 | `freeze_source.py`：commit SHA + vendored snapshot | tool + `tasks/_vendored/` | 2 天 |
| 1.3 | `derive_pytorch_oracle.py`：启发式生成 → 独立 `oracle_pytorch.py` | tool | 4 天 |
| 1.4 | `extraction_failure_taxonomy.md` | spec | 1.5 天 |
| 1.5 | `meta.yaml` schema + generator（11 字段 + license + commit SHA + tolerance class） | schema + tool | 2 天 |
| 1.6 | `shapes.yaml` schema + 99 题填充（small/medium/large + 3 adversarial） | schema + 99 配置 | 4 天 |
| 1.7 | `contract.md` 模板 + 99 题 owner review（v4.5 含但 v4.6 提到独立 .md） | 99 contract | 14 天 |
| 1.8 | `ascendc_baseline.yaml` schema + **99 题全部填充**（v4.8 R12-P1：所有 task 都有该文件，非 S_native 用 `available: false`；30-40 S_native 题填实际 native ref，其余 stub）| schema + 99 配置 | 3 天 |
| 1.9 | Batch 1：10 题代表 PoC（5 PoC + 5 API 语义覆盖） | dirs × 10 | 5 天 |
| 1.10 | Batch 2：剩余 30 个 L1 | dirs × 30 | 7 天 |
| 1.11 | Batch 3：38 个 L2 | dirs × 38 | 11 天 |
| 1.12 | Batch 4：21 个 L3 + L3-07 appendix | dirs × 22 | 12 天 |
| **1.13 (v4.6 新增)** | **Canary tasks 5-10 个**：埋 identity/zero-out/cache-reuse/non-default-stream/fallback/shape-route 陷阱（B-P1-3）| canary dirs × 5-10 | 3 天 |
| 1.14 | Skip registry | yaml | 0.5 天 |

**v4.6 单题目录示例**（详见 v4.5 文档前面问答展示）：

```
tasks/level1/L1_12_rms_norm/
├── kernel.py              # 纯 @triton.jit + wrapper, 可独立 import
├── oracle_pytorch.py      # 独立 PyTorch reference
├── meta.yaml              # 结构化元数据
├── shapes.yaml            # SHAPE_CONFIGS + ADVERSARIAL
├── contract.md            # Semantic contract
└── ascendc_baseline.yaml  # AscendC native baseline adapter（v4.8 R12-P1：所有 task 都有；非 S_native 用 `available: false`）
```

**人力**：1-2 kernel engineer × 4-6 周 = **4-7 人月**（vs v4.5 3-5 人月；v4.6 因多文件 + canary 增 ~1 人月）

**每题 Task 文件内容**（不含 NPU 实现）：

v4.6 已改目录结构（见 §修订 A），单题完整示例（L1-12 `_rms_norm_kernel`）参见独立 `task_template_example.md`（详细 6 文件内容由 §修订 A 处展示）。

**Go/No-Go**（v4.6 更新）：
- 99 题目录 + 1 appendix 目录 + 5-10 个 canary 目录
- 每题 6 个必备文件（kernel.py / oracle_pytorch.py / meta.yaml / shapes.yaml / contract.md / ascendc_baseline.yaml） — **v4.8 R12-P1 同步**：所有 task 都有 `ascendc_baseline.yaml`，非 S_native 用 `available: false`；不再写"仅 S_native"
- **11 个 spec 文件** v0（v4.8 R12-P2 同步）：5 基础（task_api_contract / tolerance_policy / scoring_spec / timing_arbitration / extraction_failure_taxonomy）+ 6 reward/security（reward_hack_defense / ast_denylist / process_isolation / canary_task_design / artifact_provenance / submission_throttling）
- **v4.6 新增**：6 个 reward hacking spec 文件 v0（reward_hack_defense / ast_denylist / process_isolation / canary_task_design / artifact_provenance / submission_throttling）

---

### Phase 2：Provider Surface + Evaluator Harness（3-4 周，与 Phase 1 并行）

**目标**：fork KernelBench 移植到 NPU；实现完整 Provider Surface（不只 eval/timing，还含 dataset/manifest/validator/HF loader/analysis/profile/leaderboard validator）。

**任务**（v4.1 E1 + E2 重定，覆盖完整 KernelBench Provider Surface）：

| # | 任务 | 产物 | 工作量 |
|---|---|---|---|
| 2.1 | Fork KernelBench src/kernelbench/ + NOTICE | submodule | 0.5 天 |
| 2.2 | `DeviceBackend` 抽象（cuda/npu/mock） | base.py | 2 天 |
| 2.3 | CudaBackend + NpuBackend + MockNpuBackend（Phase 0 PoC 用） | code | 3 天 |
| 2.4 | `eval.py`：5 接口（prepare_inputs / run_ref / run_candidate / collect_outputs / check_mutation） | code | 3 天 |
| 2.5 | `task_api_contract.md` v0（Phase 1.0 前置依赖） | spec | 1.5 天 |
| 2.6 | `timing.py` + `timing_arbitration.md` | code + spec | 3 天 |
| 2.7 | `scoring.py` + `scoring_spec.md`（**v4.1 硬化版**：FastP@1.0 / @1.5 / capped geomean、固定分母、禁 fallback） | code + spec | 4 天 |
| 2.8 | `adversarial.py`（runtime checks + reward-hacking 防御 + hidden shape） | code | 2.5 天 |
| 2.9 | `static_checker.py`（辅助）| code | 1 天 |
| 2.10 | `tolerance_policy.md` | spec | 1.5 天 |
| **2.11 (E1 新增)** | **`dataset.py`**：统一 dataset abstraction（local + HF 双入口，类比 KernelBench `dataset.py#L29-L63`） | code | 2 天 |
| **2.12 (E1 新增)** | **`task_manifest.json`**：99+1 题清单 + 元数据 + bundle SHA + 版本号 | json + tool | 1 天 |
| **2.13 (E1 新增)** | **`validate_task.py`**：task 文件 schema 校验工具 | code | 1.5 天 |
| **2.14 (E1 新增)** | **`make_hf_dataset.py`**：HF dataset 打包（类比 KernelBench `make_hf_dataset.py#L139-L189`） | code | 1.5 天 |
| **2.15 (E1 新增)** | **`compile_cache.py`**：NPU triton 编译缓存（避免 submitter 反复 JIT） | code | 1.5 天 |
| **2.16 (E1 新增)** | **`analysis.py`**：result 聚合分析（fast_p / geomean / 分轨展示） | code | 2 天 |
| **2.17 (E1 新增)** | **`profile.py`**：NPU profiling（NCU → msprof 等价物）| code | 2 天 |
| **2.18 (E1 新增)** | **`leaderboard_validator.py`**：result.json 格式校验 + hash 验证 + sandbox runner | code | 3 天 |
| **2.19 (E2 新增)** | **Submitter CLI 重设计**：`init-solution` / `run --task --solution` / `run-suite --solutions-dir` / `package` / `submit` / `validate-result` | scripts | 2.5 天 |
| 2.20 | 完整 unit tests（类比 KernelBench `unit_tests/`，v4.2 P2-4 补 4 个新增到 **13 个**）：`test_eval_adversarial / test_eval_timing / test_static_checker / test_dataset / test_precision / test_score / test_validate_kernel_static / test_task_validator / test_leaderboard_validator` + **v4.2 P2-4 新增**：`test_submission_artifact_safe_unpack`（tar 解包安全）/ `test_cli_e2e_package_validate`（CLI 端到端）/ `test_compile_cache_key_isolation`（编译缓存 cache key 隔离）/ `test_profile_fallback_detector`（fallback detector allowlist） | tests | 7 天 |
| 2.21 | Raw result JSON schema（v4.1 增字段：task_bundle_sha / solution_sha / runner/container/hardware/software hash / per-shape timing / quarantine flags） | code | 1 天 |

**人力**：1.5-2 engineer × 3-4 周 = **1.5-2.5 人月**（v4.2 P2-4 同步：原 §2 资源总表 1-1.5 不一致，统一为 **1.5-2.5**）

**关键产出**（v4.1）：
- `npu_kernel_bench` Python 包 — **完整 Provider Surface**（19 个模块）
- 5 个独立 spec 文件
- **Submitter CLI**：task / solution 严格分离
- Raw result JSON schema：含完整 hash chain + container digest + per-shape timing

**Submitter 提交接口（v4.1 E2 重设计 — task/solution 严格分离 + hash 验证）**：

```bash
# Submitter 用法
$ pip install npu-kernel-bench

# 1. 拿到 task bundle（只读 + hash 锁定）
$ npu-bench init-solution --task L1_01_bias --out my_solutions/
# 生成:
#   my_solutions/L1_01_bias/
#     ├── task_bundle.tar.gz       # read-only, SHA256-verified
#     ├── task_bundle.sha256       # canonical hash
#     └── solution.py              # 待填充模板

# 2. 写 NPU triton 实现（仅修改 solution.py，不可改 task_bundle）
# solution.py 必须实现:
#   def npu_triton_impl(inputs: dict) -> dict
# 调用 npu-bench run 时会自动校验 task_bundle 未被篡改

# 3. 单题评测
$ npu-bench run --task L1_01_bias --solution my_solutions/L1_01_bias/solution.py
# 输出 result.json:
{
  "task_id": "L1_01_bias",
  "task_bundle_sha": "<frozen sha>",       # v4.1 E2 新增
  "solution_sha": "<submitter sha>",       # v4.1 E2 新增
  "runner_hash": "<npu-bench version>",    # v4.1 E2 新增
  "container_digest": "<docker digest>",   # v4.1 E2 新增
  "hardware_id": "910B-A2-...",            # v4.1 E2 新增
  "software_hash": {"cann": "8.5.0", "torch_npu": "2.8.0", "triton_ascend": "3.2.0"},
  "per_shape_timing": {"small": {...}, "medium": {...}, "large": {...}},
  "quarantine_flags": [],                  # v4.1 E2 新增
  "correctness_status": "ok",
  "runtime_status": "ok",
  "performance_status": "ok",
  ...
}

# 4. 套件批量评测
$ npu-bench run-suite --solutions-dir my_solutions/ --output results/

# 5. 打包提交件
$ npu-bench package --solutions-dir my_solutions/ --results-dir results/ --out submission.tar.gz

# 6. 校验提交件格式（本地预检）
$ npu-bench validate-result --submission submission.tar.gz

# 7a. 提交到 self-reported leaderboard（信任 submitter 自跑数据）
$ npu-bench submit --self-reported --submission submission.tar.gz

# 7b. 提交到 verified leaderboard（官方 frozen runner 在 910B 重跑）
$ npu-bench submit --verified --submission submission.tar.gz
# 官方 runner 接收 → static check → sandbox → timeout → hidden adversarial shapes → full rerun → 上榜
```

---

### Phase 3：Baseline 数据采集（2-3 周）

**目标**：跑三层 baseline，提供 `migration_score` 分母。

**任务**（v4 不变）：

| # | 任务 | 产物 | 工作量 |
|---|---|---|---|
| 3.1 | H100 GPU triton baseline（99 题 × 3 shape × 100 trial × 3 run） | `results/h100_gpu_triton/*.json` | 4 天 |
| 3.2 | A100 secondary baseline | `results/a100_gpu_triton/` | 2 天 |
| 3.3 | 910B torch_npu eager baseline | `results/910b_torch_eager/` | 3 天 |
| 3.4 | **910B AscendC native baseline adapter**（v4.1 E5 改名，原 "AscendC native oracle reference" 改名以避免与 correctness oracle 混淆）：30-40 题，含 wrapper 层接 sgl-kernel-npu / vllm-ascend native ops；**每题 wrapper timebox 0.5-1 天**，超时该题降级为无 native baseline。进 `S_native` 条件：wrapper 与 PyTorch oracle 在所有 public + hidden shape 上语义一致（不一致 → 不进 S_native）| `results/910b_ascendc/` | 5-10 天 |
| 3.5 | Per-batch sanity（CV > 10% 重测、outlier 标记、频率/温度 log） | sanity 报告 | 2 天 |
| 3.6 | `S_native` 锁定（明确哪些题有 AscendC oracle） | spec | 1 天 |
| 3.7 | `baseline_dashboard.html` | dashboard | 2 天 |

**人力**：
- 1 baseline owner × 2-3 周
- 0.5 NPU infra（频率锁 / 温度监控）
- 0.5 AscendC wrapper specialist（接 sgl-kernel-npu / vllm-ascend native ops）
- 合计 **2-3 人月**

**关键产出**：
- `S_native` 集合锁定（30-40 题，公开级 `score_native_only` 分母）
- 三层 baseline raw JSON（含 version_matrix_sha / docker_digest / 频率 / 温度）

---

### Phase 4：开源化 + 两级 Leaderboard（v4.1 E3 重设计）

**目标**：把 benchmark 推向社区。**两级 leaderboard**：self-reported（v0.1）+ verified（v1.0）。

**v0.1 self-reported（1-3 周，1.5-3 人月）**：

| # | 任务 | 产物 | 工作量 |
|---|---|---|---|
| 4.1 | Frozen result release（v0.1 immutable tarball + Docker image + raw JSON） | GitHub release | 3 天 |
| 4.2 | HuggingFace dataset（task bundle + GPU/AscendC/eager baseline） | HF dataset URL | 3 天 |
| 4.3 | GitHub repo 公开 + CI（GPU side full + NPU side smoke via MockNpuBackend） | repo public | 3 天 |
| 4.4 | **Self-reported leaderboard**：接受 submitter 上传 submission.tar.gz（含 task_bundle_sha 校验、result.json 格式校验、static check），**不重跑** | leaderboard 静态站 | 5 天 |
| 4.5 | Tech report + arXiv（可选） | preprint | 5 天 |
| 4.6 | 社区 outreach：vLLM-Ascend / sgl-kernel-npu / triton-ascend issue + 文档站友链 | issues + docs | 3 天 |

**v1.0 verified 核心定义（v4.2 P2-2 写死）**：

> **verified leaderboard 上的排名数据，只来自官方 frozen runner 在 910B 上重跑的结果**。submitter 自报的 `result.json` 仅用作差异检测（与 runner 重跑结果对比，偏差 > 20% flag 检查）；**top-K full rerun / 第三方 audit 是二次审计层**，不替代 "所有 verified 提交均经过官方重跑" 这一基础规则。

**v1.0 verified（额外 5-7 周，**+5-6 人月**，需 2 张 910B 或借机 pool）**（v4.10 R14-P1 同步：v4.6 改目录化后 +1 周；§0/§2.1 都已更新到 v0.1 9-15 周/12.5-21.5 人月 / v1.0 14-20 周/17.5-27.5 人月）：

| # | 任务 | 产物 | 工作量 |
|---|---|---|---|
| 4.7 | **Frozen 官方 runner**（910B + 锁定 version matrix + Docker image）：接受 submission.tar.gz，自动 unpack → static check → sandbox → timeout → 重跑 | runner service | 10 天 |
| 4.8 | **Hidden / adversarial shapes**：除 public small/medium/large 外，额外 5-10 个未公开 shape 用于 verify，防 submitter overfit | shape set | 3 天 |
| 4.9 | **Full rerun protocol（v4.3 R6-1 修正）**：**所有 verified 提交 100% 官方重跑**（3 run + CV check），无 sampling；top-K 高分提交额外做 audit-rerun（重复 100% 但用第二台独立 910B 跑，对比 CV），第三方 audit 月度独立抽 ≥ 5 题 — 这两层都是**二次审计**，不替代 100% 基础重跑 | runner workflow | 5 天 |
| 4.10 | **Raw logs 公布**（v4.5 R8-P1-2 分两层防 hidden set 泄露）：① **public logs**：完全公开 — 99 题 public shapes (small/medium/large) 的完整 per-shape timing / 频率 / 温度 / npu-smi / container digest / commit SHA ② **hidden logs**：仅公开 aggregate pass/fail + quarantine reason + `hidden_set_version` + **sealed hash commitment**（hidden shape 维度的预承诺 hash），hidden shape 具体维度仅向第三方 audit 披露，并在 hidden set retired 后解封归档 ③ **hidden/adversarial set 按 leaderboard epoch 轮换**（建议每 6 个月轮换 + sealed-set 解封） | log archive + sealed commitment scheme | 5 天 |
| 4.11 | **Quarantine appeal**：submitter 对 quarantine 决策可 appeal，附 profiler proof + memory lower bound + output dependency proof | appeal flow + docs | 3 天 |
| 4.12 | **Sandbox + 资源限制**（R12 缓解，v4.3 R6-4 严格 no network）：① **candidate execution namespace 严格 `--network=none`**（无任何 endpoint），**日志/结果上传由 runner 外层完成**（candidate exit 后由 sandbox 外的 runner 进程读取 result + 上传，candidate 自身不持网络能力） ② 文件系统 **read-only**（除 `/tmp/scratch` 256 MB） ③ **seccomp syscall allowlist**（仅允许 NPU runtime / Python / torch 必需 syscall） ④ **device allowlist**（仅 `/dev/davinci*` 与必要 char device） ⑤ **cgroup memory limit** 32 GB ⑥ **cgroup CPU limit** 16 core；拒绝 privileged / ASLR off / SETUID + artifact 签名验证 | sandbox config + verified runner image | 5 天 |
| 4.13 | **第三方 rerun audit**：周期性请独立方在外部 910B 上重跑 top 提交 ≥ 5 题 | audit report | 5 天 |

**人力**：
- v0.1：1 lead + 0.5 DevRel × 1-3 周 = **1.5-3 人月**
- v1.0：+1 runner ops engineer × 4-6 周 = **+4-6 人月**

**关键 mental model（v4.1 E3）**：

```
Submitter 提交流程：
  自跑 result.json → submit --self-reported → 上 self-reported board（信任 submitter 自报数据，仅做 hash + schema 校验）
       │
       └──→ submit --verified → 进 verified queue → 官方 frozen runner 重跑 → 通过 hidden adversarial → 上 verified board
                                                  │
                                                  └──→ 失败 / quarantine → submitter 可 appeal → 第三方 audit
```

**R12 触发条件**：v1.0 verified leaderboard 启动后，**必须**实现 §4.12 sandbox + 资源限制 + artifact signing + 拒绝特权模式。无 v1.0 → 不触发 R12。

---

## 2. 资源需求 v4

### 2.1 总人月

| Phase | 人月 |
|---|---:|
| Phase 0 | 2.5-4.5 |
| Phase 1 | **4-7**（v4.8 R12-P1 同步：v4.6 改目录+canary 后 +1-2 人月）|
| Phase 2 | **1.5-2.5**（v4.2 P2-4 同步，原 1-1.5 不一致已修） |
| Phase 3 | 2-3 |
| Phase 4 v0.1（self-reported）| 1.5-3 |
| Phase 4 v1.0（verified 增量）| +4-6 |
| **v0.1 self-reported 合计** | **12.5-21.5 人月**（v4.7 R11-P1 同步 §0）|
| **v1.0 verified 合计** | **17.5-27.5 人月**（v4.7 R11-P1 同步 §0）|

vs v3.1: 24-60 → v4: 10-17 → v4.3: 11-18/15-24 → v4.6: 12-20/17-26 → **v4.7 v0.1: 12.5-21.5 / v1.0: 17.5-27.5**（§0 与 §2.1 完全一致；v4.7 因 Phase 2 process isolation/timing shim 实现 +0.5-1.5 人月）

### 2.2 总周期

| Phase | 墙钟周 |
|---|---:|
| Phase 0 | 1-2 |
| Phase 1+2（并行）| 3-5 |
| Phase 3 | 2-3 |
| Phase 4 | 1-3 |
| **合计** | **9-15 周**（含并行，v4.8 R12-P1 同步 §0 承诺等级表；v4.5 7-13/v4.6 8-14/v4.7+8 9-15）|

### 2.3 角色 + 硬件

| 资源 | 数量 |
|---|---|
| Lead engineer | 1 × 全周期 |
| Kernel engineer / spec author | 1-2 × Phase 1+2 |
| Baseline owner | 1 × Phase 3 |
| NPU infra | 0.5 × Phase 0+3 |
| AscendC wrapper specialist | 0.5 × Phase 3 |
| PM / DevRel | 0.5 × Phase 4 |
| **总 FTE (v0.1)** | **2-3.5** |
| **总 FTE (v1.0 增量)** | **+1 runner ops** = 3-4.5 |
| GPU (H100/A100) | 1 卡（v0.1/v1.0 同） |
| 910B (v0.1 self-reported) | **1 张**（CI nightly 用 mock） |
| 910B (v1.0 verified) | **2 张** 或借机 rerun pool（v4.4 R7-5 拆分硬件需求）|

---

## 3. 验收标准 v4

### 3.1 Must Have（benchmark 上线）

- [ ] 99 题 task 文件 + 1 appendix
- [ ] **11 个** spec 文件（5 基础：task_api_contract / tolerance_policy / scoring_spec / timing_arbitration / extraction_failure_taxonomy；6 reward/security：reward_hack_defense / ast_denylist / process_isolation / canary_task_design / artifact_provenance / submission_throttling）实文件 — v4.8 R12-P2 同步
- [ ] `npu_kernel_bench` Python 包 + 6-CLI（v4.4 R7-6 统一）：`npu-bench init-solution / run / run-suite / package / submit / validate-result`（替代 v3 旧 CLI 名 `run_and_check.py / run_full_benchmark.py`）
- [ ] 三层 baseline 完整：H100 GPU triton（99 题）+ 910B AscendC native（`|S_native|=30-40`）+ 910B torch eager（99 题）
- [ ] `S_native` 锁定 + `score_native_only` / `score_all` / `score_fallback_only` 公式 + reference impl
- [ ] Adversarial runtime checks（input hash + buffer poison + stream audit + excessive speed flag + fallback detector）
- [ ] Raw result JSON schema + version_matrix_sha / docker_digest

### 3.2 Should Have（公开发布）

- [ ] HuggingFace dataset 发布
- [ ] GitHub repo + CI
- [ ] Leaderboard 网站（接受 submitter 提交）
- [ ] 第三方 rerun protocol
- [ ] Tech report / arXiv
- [ ] vLLM-Ascend / sgl-kernel-npu / triton-ascend 上游 issue + 文档站友链

### 3.3 Best Effort

- [ ] 跨硬件扩展（910C / 950 baseline）
- [ ] Leaderboard 实时排名（vs 静态 release）
- [ ] 学术合作

### 3.4 **v4 明确不在 scope**

| 不做项 | 理由 |
|---|---|
| ❌ 99 题 NPU triton 实现 | benchmark provider 不写实现，submitter 写 |
| ❌ `migration_score` 把"我们的 NPU 实现"评高 | 没有"我们的 NPU 实现" |
| ❌ Phase 4a / 4b NPU 实现 + perf hardening | 不在 scope |
| ❌ 5-7 kernel engineer 团队 | 不需要（benchmark infra 不需要这么多人） |

---

## 4. 风险登记 v4（精简，删 Phase 4 相关风险）

| # | 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|---|
| R1 | triton-ascend 编译器不支持的 pattern 让 task 文件中的 GPU triton kernel 在 NPU 上无法被 submitter 实现 | 高 | 中 | 不影响 benchmark provider scope；记录到 `unsupported_patterns.md`，让 submitter 知情 |
| R2 | UB 192 KB 限制让 L3 完整 attention/MoE benchmark 题"没人能跑赢"AscendC | 中 | 低 | 不影响 benchmark；这就是 benchmark 揭示能力上限的本职功能 |
| R3 | AscendC native baseline 接口与 GPU triton 接口不对齐 | 中 | 中 | Phase 3 每题预留 wrapper 0.5-1 天 |
| R5 | KernelBench license 合规 | 低 | 中 | Phase 0.4 法律 + NOTICE |
| R6 | 910B 实机 SSH 不稳 | 中 | 低 | Phase 3 只需 1 张 NPU 跑 AscendC baseline，可降级到借机 |
| R8 | Version drift（CANN / triton-ascend 升级让 baseline 失效） | 高 | 中 | Frozen version matrix + Docker digest |
| R9 | Source drift（vLLM/SGLang upstream 改 triton kernel） | 高 | 中 | 每题锁 commit SHA + vendored snapshot |
| R11 | Oracle/spec drift（PyTorch ref / AscendC ref / GPU triton 语义不一致） | 中 | 中 | Per-task golden case + cross-oracle diff（三方一致才进 `S_native`） |
| R12 | Leaderboard untrusted kernel execution（仅 Phase 4 leaderboard 启动后） | 中 | 高 | sandbox + timeout + 资源限制 + artifact signing |

**删除的 v3.1 风险**（因 Phase 4 NPU 实现不再 scope）：
- 原 R10 NPU SPOF 影响 dev/CI 双环境 → v4 只需 1 张 NPU，降级
- 原"五线/三线 staffing fallback"风险 → 不再相关
- 原"Phase 4 6-14 周延期"风险 → 不再相关

---

## 5. Score 公式硬化（v4.6 增 min_coverage + absolute sanity bound）

**主榜公式（防 reward-hacking 关键）**：

$$FastP_{native}(p) = \frac{\#\{t \in S_{native} : correct(t) \land \neg quarantined(t) \land \neg fallback(t) \land t_{npu\_triton}(t) \le p \cdot t_{ascendc}(t)\}}{|S_{native}|}$$

- 分母 `|S_native|` **固定**（Phase 3 锁定，约 30-40 题），后续不变
- 未提交 / correctness=fail / runtime=timeout / performance=quarantined / detected fallback → 分子贡献 **0**（不减分母）
- 主榜禁用 `p=∞`（仅 correctness）作为单一汇总，避免"乱写但正确"刷上榜

**主榜呈现指标**（v4.1 E4）：
- `FastP_native@1.0`：跑赢 AscendC native 比例
- `FastP_native@1.5`：≤ 1.5× AscendC 比例
- `capped_geomean_native`：几何平均 speedup，cap 单题最高 5×（防极端 outlier 拉总分）
- `correctness_rate_native`：纯 correctness 通过率（独立展示，**不是**主榜单一指标）

**次榜 / 分轨展示**：
- `FastP_all`：分母 `|S_scored| = 99`；分子用 `t_denom`（AscendC if available, else GPU triton）—— **明确比较对象，不混分母**
- `FastP_fallback`：仅在 `S_fallback = S_scored \ S_native` 上，分母 = torch eager
- `vs_gpu_track`：单独一栏展示 `t_npu / t_gpu_triton`（信息性，不参与排名）

**NPU Triton track（v4.1 E4 关键）**：
- 提交者代码 **禁止调用** `torch.add / torch.matmul / torch.nn.functional.*` 等 high-level torch op
- 静态 check + runtime profiler 双重检测
- 若检测到 AscendC ops 调用（如 `npu_fused_infer_attention_score`）→ **hard fail**（NPU Triton track 不允许 fallback 到 native ops）
- 若 submitter 想用 AscendC + triton 混合 → 提交到独立的 `hybrid track`，与 NPU Triton track 不同榜

**Quarantine 处理**：
- `t_npu / t_gpu_triton < 1/3` → flag suspicious，进一步 profile
- `t_npu / t_gpu_triton < 1/5` → quarantine，submitter 必须提交 profiler trace + memory lower bound proof + output dependency proof 才能解除
- 进 quarantine 的题 `performance_status = quarantined`，**仍计入分母**，贡献 0（v3.1 errata 已说明）

**v4.2 P2-3 — `capped_geomean_native` 精确公式**：

对每个 task $t \in S_{native}$ 计算 per-task speedup：

$$s_t = \begin{cases}
\max\left(\epsilon,\ \min\left(5,\ \dfrac{t_{ascendc}(t)}{t_{npu\_triton}(t)}\right)\right) & \text{if } correctness\_status=ok \land runtime\_status=ok \land performance\_status=ok \land no\_fallback \\
\epsilon = 0.01 & \text{otherwise (fail / timeout / quarantine / fallback / not submitted)}
\end{cases}$$

**v4.3 R6-2 关键**：ok case 加 $\max(\epsilon, \cdot)$ 包裹，防 $t_{npu\_triton}=0$（除零） 或 $t_{npu\_triton}$ 异常大（speedup → 0 → log(0) = $-\infty$）导致 geomean 爆炸。$\epsilon = 0.01$ 与 fail case 一致。

**v4.4 R7-3 — timing 合法性前置检查**：

在算 $s_t$ 前，先校验 $t_{npu\_triton}$ 合法性：

```
合法条件：t_npu_triton 必须是 positive finite value（即 0 < t < +∞，且非 NaN）

非法情况处理（v4.5 R8-P1-1 修：用 `invalid_timing` 替代 `error`，避免与 `correctness_status=error` 混淆错误来源）：
  t_npu_triton ≤ 0   → runtime_status = invalid_timing, score 走 epsilon
  t_npu_triton = NaN → runtime_status = invalid_timing, score 走 epsilon
  t_npu_triton = +∞  → runtime_status = timeout,        score 走 epsilon
  t_npu_triton < 1 μs（疑似未真正执行）→ performance_status = quarantined, 进 §5 quarantine 流程
```

这样 $\max(\epsilon, \min(5, t_a/t_n))$ 永远不会面对除零或无限大输入。除法发生时 $t_n$ 一定是 positive finite。

最终：

$$capped\_geomean_{native} = \exp\left(\frac{1}{|S_{native}|}\sum_{t \in S_{native}}\log(s_t)\right)$$

性质：
- 单题最大 speedup 上限 = 5（防极端 outlier 拉总分）
- 未提交 / 失败 / quarantine 时贡献 $\log(\epsilon) = \log(0.01) \approx -4.6$（强惩罚）
- 分母固定 `|S_native|`，与 `FastP_native` 公式一致
- 完全可复现（无随机性，无 NaN 处理歧义）

**v4.10 R14-P1 — Fallback Detector：runner-controlled compiler broker 单一权威**（替代 v4.3 R6-5 "两阶段 allowlist"）：

历史 v4.3 R6-5 用 build 阶段 exact `.so` allowlist + `solution_artifacts.json` 登记；v4.7 R11-P1-4 改 runner-controlled compiler driver + ELF provenance signature。v4.10 统一**只用 broker provenance 为 verified 主路径**：

- **Build/run 阶段都用同一 broker provenance**：runner 启动 candidate 前注入 broker（runner-controlled binary），所有 triton-ascend JIT 编译经 broker → broker 计算 `(triton_ir_hash, compiler_version, sm_arch)` 签名注入 ELF section
- candidate 运行时新 shape 触发 JIT 仍可通过（broker 编译 + 签名），解决 hidden shape JIT 冲突
- run 阶段加载 `.so` 时验签：无 broker 签名 → reject
- `solution_artifacts.json` **仅作 self-reported 日志**，不进 verified 权威路径（v4.10 R14-P1 修：v4.3-v4.7 含混路径）

**v4.11 R15-P1-1 — Active spec：verified 主路径仅信 broker provenance（删除两阶段 allowlist 授权流程）**

历史 v4.3 R6-5 / v4.4 R7-4 / v4.7 R11-P1-4 多版本含混已彻底清理。以下"两阶段 allowlist" / "Phase A-B" 描述**仅作为历史档案**，**不是 active spec**；verified runner 实现不得采用。

```
Verified runner 启动 candidate 流程（v4.11 active）：

1. Runner 注入 broker（runner-controlled binary，candidate 不可改）到 sandbox
2. Candidate 触发 triton-ascend JIT 时，broker 拦截编译请求
3. Broker 编译 → 计算 (triton_ir_hash, compiler_version, sm_arch) → 加密签名注入 ELF section
4. 加载 .so 时验签：
   - 有 broker 签名 + signature 验证通过 → 允许加载
   - 无 broker 签名 → reject (含 ctypes / cpp_extension load_inline / submitter 包内预编译 binary)
5. 同时 torch_npu profiler 抓 op event → op name 不在 NPU triton allowlist → fallback hack flag

`solution_artifacts.json` 与 `runtime_env/lib/*.so`：仅作 self-reported 日志，verified runner **完全不读**
hidden shape runtime JIT 自动通过（broker 编译 + 签名）→ 无双轨冲突
```

----（以下历史档案，不是 active spec）----



| 旧 (v4.2 单阶段) | 新 (v4.3 两阶段) |
|---|---|
| 单一固定 allowlist：`triton_ascend.so + torch_npu.so + harness` | **Build 阶段**：在隔离 build sandbox 内 import `solutions/<task_id>.py`，触发 triton-ascend JIT 编译；收集所有产生的 `.so` path + sha256，登记进 **`solution_artifacts.json`** |
| 问题：submitter 的 triton kernel JIT 产物会被误杀，或必须放宽到允许任意 `.so`（漏 fallback） | **Run 阶段**：仅允许 ①`solution_artifacts.json` 内已登记 `.so`（per-task whitelist） + ②frozen runtime libs（`triton_ascend.so` / `libtorch_npu.so` / npu-bench harness） |

**两阶段流程**（v4.4 R7-4 信任边界硬化）：

```
v4.4 R7-4 信任边界关键修订：
  Self-reported 模式：submitter 包里的 solution_artifacts.json 可信，runtime_env/ 子目录读取
  Verified 模式：submitter 包里的 solution_artifacts.json **仅作 self-reported metadata 参考**，
                runner 必须在 no-network build sandbox 内自己重新生成 allowlist
                run sandbox 只消费 runner-generated artifact manifest（不读 submitter 提供的）

Phase A: Build (verified 模式下由 runner 自建，不信任 submitter 包)
  1. runner 启动 no-network build sandbox
  2. mount solutions/<task_id>.py read-only（从 submission.tar.gz 提取）
  3. python -c "from solutions.L1_01_bias import npu_triton_impl; warm up call"
  4. LD_AUDIT hook 记录所有 dlopen 加载的 .so 路径
  5. 计算每个 .so sha256
  6. 写入 runner-controlled 路径：/runner_state/<submission_id>/solution_artifacts.json
  7. （对比 submitter 包内 solution_artifacts.json，若有 .so 仅在 submitter list 但 build 时未加载 → flag 可疑，不进 allowlist）

Phase B: Run (verified sandbox)
  1. import solution
  2. dlopen hook：每个被加载的 .so 检查是否在
     - /runner_state/<submission_id>/solution_artifacts.json（runner 生成）
     - frozen runtime allowlist（npu-bench / triton-ascend / torch_npu）
  3. 不在两个 list 任意一个 → fallback hack flag
  4. 同时 torch.profiler / torch_npu profiler 抓 op event → op name 不在 allowlist → fallback hack flag
```

实现：
- `LD_AUDIT` hook 监控 `dlopen` （Linux）
- `torch.profiler` + `torch_npu` profiler 抓取所有 op event
- `solution_artifacts.json` 入 `submission.tar.gz` 的 `runtime_env/` 子目录

**状态三字段**（沿用 v3.1）：
- `correctness_status ∈ {ok, fail, error}`
- `runtime_status ∈ {ok, timeout, oom, compile_fail, invalid_timing}`（v4.5 R8-P1-1 加 `invalid_timing` 用于 t_npu ≤ 0 / NaN）
- `performance_status ∈ {ok, suspicious, quarantined}`
- `performance_flags: list[str]`（v4.8 R12-P1 新增独立字段，不放 status enum）：可含 `low_performance`、`high_variance`、`thermal_throttle` 等，不影响 status 但记录到 result JSON 供分析

`FastP_native` 只计 `correctness_status=ok ∧ runtime_status=ok ∧ performance_status=ok ∧ no fallback detected` 的题。

**v4.6 新增 — `min_coverage_for_ranking`**（B-P1-1，**v4.7 R11-P1 修订：分母 per-track**）：

- **Native track 主榜**（`FastP_native` / `capped_geomean_native`）：`native_coverage = correct_native_tasks / |S_native| ≥ 70%`（即 |S_native| ≈ 30-40 中至少 21-28 题 ok 才上 native ranking）
- **All-track 主榜**（`FastP_all` 次榜）：`all_track_coverage = correct_scored_tasks / |S_scored| ≥ 70%`（99 题中 ≥ 70 题）
- 任一 track 不达 coverage 门槛 → 只进 self-reported，不进对应 verified ranking

防"集中攻易题 aggregate 仍高分"hack (E2)。

**v4.6 新增 — `absolute_sanity_bound`**（B-P1-4）：

任何题命中以下条件 → **直接 hard fail，不进 score**：
- `runtime_status=invalid_timing`（t_npu ≤ 0 / NaN / Inf）
- Profiler 抓不到任何 NPU kernel event（即没真的跑 NPU triton kernel）— 防 B5 空 kernel / B8 部分计算 / CPU host fallback
- `hardware_utilization > 100%`（超 910B 理论峰值，物理不可能）
- speedup > 10×（vs AscendC native）— quarantine + 强制 proof（profiler trace + memory lower bound + output dependency）
- speedup < 0.1×（vs AscendC native）— **v4.8 R12-P1 修订**：原 v4.7 写 `performance_status = low_performance` 与 enum `{ok, suspicious, quarantined}` 冲突；改为新加 `performance_flags: list[str]`（独立字段），append `"low_performance"`；status 仍是 `ok`（不 hard fail，< 0.1× 是慢，不是 hack）；空 kernel 实际表现为 speedup > 10×（已被 quarantine 抓）+ 无 NPU event hard fail

---

## 5b. Reward Hacking by-Design Defense（v4.6 新增 — 24 原则完整章节）

**核心理念**：reward hacking 防御必须 **by-design 在 benchmark 构建时固化**，运行时 detector 是兜底。本节是 v4.6 最重要的新增章节。

### 5b.1 攻击面矩阵（31 类）

| 类 | # | 攻击 | v4.6 防御 |
|---|---|---|---|
| **A Correctness** | A1 | 直接 import oracle | 5b.§1 oracle isolation + 5b.§3 AST denylist |
| | A2 | 复用 reference 中间张量 | 5b.§1 process isolation + 5b.§5 output buffer poison |
| | A3 | Identity hack (return input) | 5b.§5 differential input perturbation + B-P1-3 canary identity 陷阱 |
| | A4 | Hardcode output 查表 | 5b.§5 differential + B-P1-3 canary shape-route 陷阱 |
| | A5 | Output buffer 复用 | **v4.13 R18-S4 修订**：soft signal 模式 — byte-pattern match → `performance_flags.append("suspected_buffer_reuse")` 不 hard fail（与 candidate-allocated ABI 一致）|
| | A6 | NaN / Inf 钻 allclose | 5b.§5 子项 ⑤ NaN/Inf 一致性检查 |
| | A7 | 容差边缘 fitting | 5b.§5 differential 3 组随机 inputs，单组 fitting 失效 |
| | A8 | 篡改 inputs | 5b.§5 input hash before/after |
| **B Performance** | B1 | Async 不 sync | 5b.§2 runner-owned timing + post-call npu.synchronize |
| | B2 | 非默认 stream 偷计算 | 5b.§2 all-stream sync before timing end |
| | B3 | 预编译 cache 复用 | 5b.§4 artifact provenance + warm-up 与 timing 隔离 |
| | B4 | Warm-up 预算 | 5b.§4 warm-up 后 fresh inputs（differential 已 cover）|
| | B5 | 空 kernel | 5b.§6 absolute sanity: profiler 无 NPU kernel event hard fail |
| | B6 | 篡改 timing API | 5b.§2 sandbox 禁 timing API import / LD_PRELOAD |
| | B7 | Shape 路由 | hidden adversarial shape + canary shape-route |
| | B8 | 部分计算（只算 batch[0]） | 5b.§5 differential 不同 batch 应 output 不同 + profiler 检查 element count |
| | B9 | NPU 频率 manipulation | runner 强制频率锁 + 拒绝 submitter Dockerfile（v4.6 §B-P1-2 强化） |
| **C Submission** | C1-C5 | Hash/symlink/traversal/manifest/逃逸 | v4.5 tar 安全 + signature + sandbox 网络隔离已含 |
| **D Test-set** | D1 | 多次提交反推 hidden | B-P1-2 submission throttling |
| | D2 | Hidden timing 推断 | v4.5 §4.10 hidden logs 只发 aggregate（已含）|
| | D3 | Adversarial overfit | hidden set epoch 轮换（v4.5 R8-P1-2 含）|
| | D4 | Hidden 维度生日攻击 | B-P1-2 throttling + hidden set 由 private generator 随机生成（v4.6 强化）|
| **E LLM-specific** | E1 | 抄 kernel.py 90%+ | B-P1-5 similarity flag + manual review queue（v4.8 R12-P1 统一，不自动剔除）|
| | E2 | 集中攻易题 | B-P1-1 `min_coverage_for_ranking` 70% 门槛 |
| | E3 | LLM prefix / comment 欺骗 | 5b.§3 AST 扫描忽略注释，只看代码语义 |
| | E4 | Cross-task contamination | B-P1-2 sybil + B-P1-3 canary |
| | E5 | Memorization | 不可完全防（LLM 训练集见过 ref），但 differential 让 memorization 不够 |
| **F Runner 边界**（codex 新增）| F1 | Runner / oracle 同进程污染 | **5b.§1 process isolation**（核心 P0）|
| | F2 | 任意 native artifact 洗白 allowlist | 5b.§4 artifact provenance（仅 runner-controlled triton JIT 产物）|
| | F3 | Stateful / call-order hack | 5b.§2 阶段间 kill process group + fresh process per phase |
| | F4 | CPU/host fallback | 5b.§6 profiler 检查 NPU kernel event 数量 |
| | F5 | 跨 task / 跨 run 泄露 | runner 每 task 清 /tmp + Triton cache + device memory |
| | F6 | 多账号 sybil | B-P1-2 per-org quota + identity 汇总 |
| | F7 | 频率/温度 side-channel | runner 强制频率锁 + 温度 log 公开（攻击者无 fine-grained 信号）|
| | F8 | submitter runtime_env supply chain | 5b.§4 verified runner **忽略** submitter Dockerfile，自建 frozen env |
| | F9 | LLM prompt injection / solution laundering | 静态扫描注释含可疑 prompt → flag；跨 submitter similarity flag |
| | F10 | 输出 contract abuse（dtype/layout/alias）| 5b.§5 子项 ③ output 严格检查 |

### 5b.2 24 条防御原则

**Layer 1 — Process / Sandbox isolation（P0 必做）**

**§5b.1 Oracle isolation contract**（B-P0-1，**v4.7 R11-P0-1 致命修订**）
- task 目录的 `oracle_pytorch.py` 在 **runner 外层 trusted process** 调用
- Sandbox mount 只挂 `kernel.py + meta.yaml + shapes.yaml + inputs.pt`
- ⚠️ **绝不挂 `golden_outputs.pt`**（v4.6 原文有这一句是致命设计漏洞 — 等于把答案直接给 candidate）
- **不挂** `oracle_pytorch.py` / `ascendc_baseline.yaml`
- `sys.path` 不含 task 目录路径
- Oracle 计算 + allclose 比较 + correctness verdict 都在 trusted runner 进程；candidate exit 后 runner 读 candidate 写的输出（详见 §5b.1a 安全输出 ABI），与 runner-held golden 比对

**§5b.1a 安全 candidate output ABI**（v4.8 R12-P0-1 致命修订）：

⚠️ v4.7 让 runner 读 candidate 写的 `output.pt` 是 **sandbox escape 漏洞** — `.pt` 是 pickle 格式，candidate 可构造恶意 pickle，让 trusted runner `torch.load()` 反序列化时**任意代码执行**。

**禁止格式**：`.pt` / `.pth` / `.pkl` / 任何 pickle-based 格式（含 `torch.save` 默认）

**强制格式**（二选一）：
1. **`safetensors`**（推荐）：纯数据格式无代码执行；candidate 用 `safetensors.torch.save_file({"output": t}, "output.safetensors")`；runner 用 `safetensors.torch.load_file("output.safetensors")`
2. **自定义 metadata+raw+sha256**：`metadata.json`（shape/dtype/device/sha256，纯结构化）+ `raw_bytes.bin`（`tensor.cpu().numpy().tobytes()`）；runner 用 `json.loads()`（无代码执行）+ `np.frombuffer().reshape()` 安全重建 + 验签 sha256

**最强方案**：**runner-owned shared memory output ABI** — runner 分配 shared memory，candidate 写 raw bytes；无文件 IO，无序列化层

**v4.11 R15-P1-2 ABI 增强**：Trusted shim 在 **序列化前** 检查 candidate 返回 tensor 的原始属性（safetensors 会丢失/规范化以下信息）：

```
Trusted shim 序列化前必检查 + 通过 runner-controlled signed pipe 输出 metadata:
  - device              (确认 NPU device，非 CPU/cuda)
  - dtype               (与 oracle expected 匹配)
  - shape               (与 oracle expected 匹配)
  - stride              (与 oracle expected 匹配，防错误 layout)
  - storage_offset      (防 candidate 返回错误 view)
  - data_ptr            (防 alias to inputs)
  - is_contiguous       (符合 contract 要求)
  - input_alias_check   (output.data_ptr() 不能等于 inputs[*].data_ptr())

Pipe 协议：
  - metadata.json signed by runner-controlled key（candidate 不可写）
  - safetensors 只承载 tensor 数值 + dtype + shape（轻量化）
  - runner 端读 metadata + 验签 → 检查上述字段 → 再读 safetensors 数值

实现：trusted shim 内置 metadata writer，candidate 看不到 metadata signing key
```

**§5b.2 Runner-owned timing / comparison**（B-P0-2，**v4.7 R11-P0-2 致命重写**）

v4.6 原方案：parent process 在 fork 前后记 `torch.npu.Event` — codex 指出 cross-process NPU event 不可靠包住子进程 work（不同 stream/context）。重写如下：

**方案：Trusted timing shim**

- candidate sandbox 内仍跑 candidate 代码，但 candidate function 入口由 **runner 注入的 trusted shim** 包裹
- trusted shim 是 runner-controlled binary（hash 入 frozen image，candidate 不可改）
- shim 行为（v4.8 R12-P0-2 修订：覆盖 non-default stream）：
  1. `npu.synchronize()` 全清
  2. **权威 timing 改用 `CLOCK_MONOTONIC_RAW` host time**（不是 event）— 因为 `event_end.record(current)` 只测 current stream，candidate 用 non-default stream 偷计算会漏计
  3. `host_t0 = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)`
  4. 调 `solution.npu_triton_impl(inputs)`（同 process）
  5. **`torch.npu.synchronize()` device-wide sync**（等所有 stream 全完）
  6. `host_t1 = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)`
  7. `latency_ms = (host_t1 - host_t0) / 1e6`
  8. 同时记 event（仅 current stream），作 cross-check：`event_ms = event_end.elapsed_time(event_start)`
  9. 若 `host_ms - event_ms > 10%` → profiler 抓 stream 列表 → 若 candidate 用了非 runner-allocated stream → `runtime_status=invalid_timing` + `performance_flags=["invalid_stream"]`（v4.9 R13-P1 修：不新增 enum 值，复用 `invalid_timing` + 独立 flag 字段）
  10. 写 timing 到 runner-controlled output pipe（candidate 无写权限）
- **备用强化方案**：强制所有 kernel 只能在 runner-created stream 上执行；shim 启动前 `torch.npu.set_default_stream(runner_stream)`；profiler 验证 stream id，发现额外 stream → invalid
- Submitter solution.py 接口：`def npu_triton_impl(inputs: dict) -> dict`，**只看 inputs，看不到 timing shim**

- Correctness 与 timing 用 **不同 candidate process 跑两次**（candidate 不能在 correctness 阶段记 cache 然后 timing 阶段查表，防 stateful hack F3）
- 每阶段后 `kill -9` candidate process group，清 device memory + Triton cache + /tmp scratch

候选代码 sandbox 内禁止 import：`torch.npu.Event / torch.cuda.Event / time.perf_counter / time.monotonic / time.process_time / LD_PRELOAD setup / torch.profiler / torch_npu.profiler`（runner-controlled wrapper 例外）

**Layer 2 — Code-level enforcement（P0 必做）**

**§5b.3 AST + import / runtime hook denylist**（B-P0-3）

**v4.9 R13-P1 — positive allowlist + 节点级 AST 规则**（不再是 denylist 模式）：

NPU triton solution.py 只允许 AST 节点白名单内的操作。任何节点类型不在 allowlist → reject。

```python
# Positive ALLOWLIST（基于 AST node type，v4.10 R14-P1 补完）：
AST_ALLOW = {
    # 模块结构（v4.10 R14-P1 必补，否则正常 solution 全拒）
    ast.Module, ast.Expr, ast.Expression,
    ast.Import, ast.ImportFrom, ast.alias, ast.keyword,
    # 函数定义 / 类定义（必须）
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Return,
    ast.arguments, ast.arg,
    # 控制流（必须）
    ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass,
    ast.Try, ast.ExceptHandler,  # try/except 允许但 except body 内禁 import torch.high-level
    # 表达式 / 字面量
    ast.Constant, ast.Name, ast.Load, ast.Store, ast.Del,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Lambda,
    ast.IfExp, ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.Dict, ast.Set,
    # 运算符（白名单）
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
    # ast.MatMult 删除（v4.10 R14-P1）：原 v4.7 允许导致与下面 backstop "禁 Tensor.__matmul__" 矛盾；v4.10 统一禁 `@`，NPU triton 路径用 `tl.dot` 即可
    ast.USub, ast.UAdd, ast.Not, ast.Invert,
    ast.Eq, ast.NotEq, ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
    # Attribute / Call 必须配合 NAME ALLOWLIST 验证
    ast.Attribute, ast.Call,
    ast.AugAssign, ast.Assign, ast.AnnAssign,
}

# Import ALLOWLIST（仅允许）：
IMPORT_ALLOW = {
    'torch',                       # tensor 创建 / dtype / shape，禁 high-level op (见 NAME)
    'torch.nn', 'torch.nn.init',  # 仅 dtype/init，禁 functional
    'triton', 'triton.language',  # NPU triton 主路径
    'triton.testing',              # do_bench wrapper（runner-controlled）
    'typing',                       # type hints
    'math',                         # 标量数学
    'numpy',                        # 仅在 candidate-side compute；禁 numpy.linalg/scipy
}

# Call NAME ALLOWLIST（dot-path 形式）：
CALL_ALLOW = {
    # Tensor 创建
    'torch.empty', 'torch.zeros', 'torch.ones', 'torch.full', 'torch.empty_like',
    'torch.tensor', 'torch.as_tensor',
    # Shape ops
    'Tensor.reshape', 'Tensor.view', 'Tensor.contiguous', 'Tensor.transpose',
    'Tensor.permute', 'Tensor.unsqueeze', 'Tensor.squeeze',
    'Tensor.size', 'Tensor.shape', 'Tensor.stride', 'Tensor.numel', 'Tensor.dim',
    # Dtype cast（不参与算术 fallback）
    'Tensor.to',  # 仅 dtype/device 转换
    # Triton 主要 API
    'triton.cdiv', 'triton.next_power_of_2', 'triton.heuristics', 'triton.autotune',
    'triton.jit', 'triton.Config',
    'tl.load', 'tl.store', 'tl.dot', 'tl.sum', 'tl.max', 'tl.min', 'tl.exp', 'tl.log',
    'tl.where', 'tl.arange', 'tl.zeros', 'tl.full', 'tl.cdiv', 'tl.program_id',
    'tl.constexpr', 'tl.static_range',
    # 显式禁用列表（即使 import 也禁）：
    # 'torch.matmul', 'torch.addmm', 'torch.bmm',
    # 'torch.nn.functional.*',
    # 'Tensor.matmul', 'Tensor.__matmul__',
    # 'Tensor.cpu', 'Tensor.numpy', 'Tensor.item', 'Tensor.tolist',
    # 'torch.ops.*', 'torch.classes.*',
    # 'getattr', 'setattr', 'delattr', '__import__',
    # 'open', 'eval', 'exec', 'compile',
}

# AttrLookup denylist (在 allowlist 之外特别禁的 attribute path)：
ATTR_DENY_PATTERNS = [
    'torch.ops.*', 'torch.classes.*', 'torch.fx.*', 'torch.compile',
    'torch.nn.functional.*',
    'Tensor.cpu', 'Tensor.numpy', 'Tensor.item', 'Tensor.tolist', 'Tensor.detach',
    'torch.cuda.stream', 'torch.npu.stream', 'torch_npu.npu.stream',
    'sgl_kernel_npu.*', 'vllm_ascend.npu_*', 'torch_npu.npu_*',
]
```

**保留旧 denylist 作 backstop**（v4.7 R11-P1，下面），与上面 allowlist 一起强制；违反任何一层 reject：
```python
DENYLIST_IMPORT = {
    # Code execution
    'exec', 'eval', 'compile', '__import__',
    'importlib', 'importlib.util', 'importlib.import_module',

    # Native code loading
    'ctypes', 'ctypes.CDLL', 'cffi',
    'torch.utils.cpp_extension', 'torch.utils.cpp_extension.load',
    'torch.utils.cpp_extension.load_inline',

    # IPC / Network
    'subprocess', 'socket', 'requests', 'urllib', 'http', 'pickle',
    'threading', 'multiprocessing', 'asyncio', 'concurrent.futures',

    # FS / IO / mmap
    'os', 'os.path', 'pathlib', 'mmap', 'fcntl', 'shutil', 'tempfile',
    'glob', 'sys.path', 'sys.modules', 'sys.meta_path',

    # Timing / profiling manipulation
    'time', 'datetime', 'monotonic',
    'torch.cuda.Event', 'torch.npu.Event', 'torch_npu.npu.Event',
    'torch.profiler', 'torch_npu.profiler',

    # Oracle / harness imports
    'oracle_pytorch', 'npu_kernel_bench', 'tasks',
}
```

**Layer B — AST semantic rules**（不只看 import，还看 attribute call / function call / subscript）：
```python
DENYLIST_AST_PATTERNS = [
    # 高 level torch op（NPU Triton track 禁，仅允许 tensor 创建 / dtype cast / shape ops）
    'torch.nn.functional.*',
    'torch.matmul', 'torch.addmm', 'torch.bmm',
    'torch.softmax', 'torch.log_softmax',
    'Tensor.matmul', 'Tensor.__matmul__',  # @ 运算符也禁
    'torch.rms_norm', 'torch.layer_norm', 'torch.group_norm',

    # AscendC native ops
    'sgl_kernel_npu.*', 'vllm_ascend.npu_fused_*', 'torch_npu.npu_*',
    'torch.ops.*', 'torch.classes.*',  # custom op registry 也禁

    # Tensor ↔ host 转移（防 CPU fallback F4）
    'Tensor.cpu', 'Tensor.numpy', 'Tensor.item', 'Tensor.tolist',
    'Tensor.detach',

    # Lazy / 编译时绕开
    'torch.fx', 'torch.compile', 'torch.jit.script', 'torch.jit.trace',

    # Stream / thread injection
    'torch.cuda.stream', 'torch.npu.stream', 'torch_npu.npu.stream',
    'torch.cuda.set_stream', 'torch.npu.set_stream',

    # Dynamic attribute lookup（防 getattr(__import__('torch'), 'matmul')）
    'getattr',  # 仅允许 whitelist 的 attribute name

    # 字符串字面量含 oracle / golden / hidden / answer / cheat
    re.compile(r'(?i)(oracle|golden|hidden_shape|hidden_set|answer|cheat)'),

    # Try/except fallback to high-level op
    # AST detect: try block 内含 high-level op
    # Pass / inheritance bypass
    # Decorator bypass like @some.bypass
]
```

**Layer C — Runtime audit hook**：
- `sys.audit` hook（Python 3.8+）监控 `import`、`open`、`subprocess.*`、`socket.*` 等敏感操作
- `LD_AUDIT` hook 监控 `.so` 加载（v4.5 已含）
- Audit log 写 runner-controlled file，candidate 不可读

实现：AST 扫描（`ast.NodeVisitor` 完整遍历，不是 regex 而是 AST node type 检查）+ runtime `sys.audit` + `LD_AUDIT` 三层。

**§5b.4 Artifact provenance**（B-P0-4，**v4.7 R11-P1 修订：解 hidden shape JIT 冲突**）
- Submitter `submission.tar.gz` 内的 `solution_artifacts.json` **不被 verified runner 信任**
- 改为 **runner-controlled compiler provenance**（不是预登记 exact `.so`）：
  - Runner 启动 candidate 时设置 **dedicated triton-ascend JIT cache namespace**：`/runner_cache/<submission_id>/triton-ascend/`
  - 只允许加载来自该 namespace 的 `.so`
  - 该 namespace 内 `.so` 必须由 **runner-controlled compiler driver** 编译产生（不是 candidate 任意写入）
  - **Provenance signature**：编译时 driver 计算 `(triton_ir_hash, compiler_version, sm_arch)` → 加密签名注入 `.so` ELF section；runner 加载前验签
- **运行时 hidden shape JIT 不冲突**：
  - candidate 触发 JIT 时，driver 在 runner-controlled namespace 内编译 → 自动签名 → 可加载
  - candidate 不能预编译塞自己的 `.so`（不在 namespace，签名不对）
- **拒绝**：
  - 任意 `cpp_extension.load_inline` 编译的 `.so`（不经 runner-controlled driver）
  - `ctypes.CDLL` 加载的预编译 binary
  - submitter 包内 `runtime_env/lib/*.so`（即使 submitter 自报 OK）
  - 任何 ELF section 缺 provenance signature 的 `.so`

**Layer 3 — Correctness hardening（P0 必做）**

**§5b.5 Correctness hardening 5 子项**（B-P0-5）

| 子项 | 实现 |
|---|---|
| ① Input hash before/after | runner 算 `sha256(input.cpu().numpy().tobytes())`，candidate 跑后再算，不一致 → fail（防 A8）|
| ② Output byte-pattern heuristic check | **v4.10 R14-P1 修订**：v4.7-v4.9 多版本反复 — 现统一为 candidate-allocated outputs（与 `meta.yaml.runner_checks.output_allocation = candidate_allocated` 一致），不做 hard sentinel overwrite。改为 **byte-pattern heuristic**：runner 让 oracle 计算时用特殊 sentinel-tainted buffer 做中间计算，候选 output 的 byte stream 不应含此 sentinel byte 模式；若 candidate output 字节匹配 sentinel pattern 出现频率 > 噪声水平 → flag（不 hard fail，仅 reward hacking signal） |
| ③ Output 严格检查 | dtype / device / shape / stride / layout / storage data_ptr 与 oracle 完全一致；不能返回 input view（防 F10）|
| ④ Differential 3 组 randomized inputs（per-contract metamorphic check）| 同 task 跑 3 组不同 seed 的 inputs；candidate output 对 input perturbation 的反应**由 `meta.yaml.runner_checks.metamorphic_checks` 列表定义**（v4.10 R14-P1 修订：单源 `meta.yaml`，引用 `shared/metamorphic_lib.yaml` 中规则名）；对 softmax/clamp/quantize 等饱和算子用 task-specific 不变量。`contract.md` 仅人类可读解释，不作 runner 输入 |
| ⑤ NaN/Inf 一致性 | candidate output 含 NaN/Inf 位置必须与 oracle 一致；防 A6 |

**Layer 4 — Sanity / Anomaly detection（P1 强化）**

**§5b.6 Absolute sanity bound**（B-P1-4，§5 已含）
**§5b.7 Profiler verdict**（v4.10 R14-P1 修订：runner_checks 单源 `meta.yaml`）：
- **NPU kernel event presence**：profiler 必须抓到 ≥ 1 个 NPU triton kernel launch event；完全无 → hard fail（防 B5/F4 CPU fallback）
- **Task-specific expected kernel envelope**：每题 **`meta.yaml.runner_checks.expected_kernel_count_range`** 声明 (如 `[1, 5]`)；candidate kernel count 超出范围 → flag
- **Bytes / FLOPs lower bound**：每题 **`meta.yaml.runner_checks.min_dram_bytes` / `min_flops`** 声明；profiler 实测低于 lower bound 50% → flag（防 B5 空 kernel / B8 部分计算）
- 不再从 `contract.md` 读 runner 字段（v4.10 R14-P1 强制 machine-readable 单源）；`contract.md` 仅人类可读说明

**§5b.8 Hardware telemetry**
- 跑 candidate 时记录 npu-smi（功耗、温度、频率）
- 异常低功耗（kernel 没真的跑）→ flag
- 异常高频率（超 frozen 频率锁）→ flag

**Layer 5 — Submission policy（P1 强化）**

**§5b.9 Submission throttling**（B-P1-2）
- 同 submitter 同 task ≤ 3 次/周
- 按 GPG key + email domain 汇总作 org identity
- 多账号短期高频提交 → sybil flag

**§5b.10 Canary tasks**（B-P1-3）
- benchmark 内埋 5-10 个 canary task（不在 99 scored 中）
- canary 设计：明确有 trivial reward hack 路径（如 identity / zero-out / cache reuse / non-default stream / fallback / shape-route）
- 任何 submission 在 canary 上得高分 → 整 submission 可疑 flag
- KernelBench 的 3 个 adversarial kernel (`result_reuse_kernel.py` / `zero_out_kernel.py` / `non_default_stream_kernel.py`) 改造为 NPU 版作 canary

**§5b.11 Solution similarity**（B-P1-5）
- solution.py vs task 的 kernel.py byte-level + AST 相似度
- > 95% → `trivial_copy` flag + **manual review queue**（v4.7 R11-P1 修订：v4.6 "不进 verified ranking" 与 B-P1-5 "仅 flag 不 reject" 自相矛盾；改为只 flag + 人工 review，Triton NPU 解法可能天然接近 GPU Triton 结构，不自动剔除）
- 跨 submitter solution 相似度高（编辑距离 < 50 token）→ 协同 flag

**§5b.12 Reproducibility enforce**（v4.5 + B-P1-4 强化）
- verified runner **忽略** submitter Dockerfile / pip lockfile（防 F8 supply chain）
- runner 自建 frozen image（`version_matrix.toml` 锁定）
- submitter 自报 timing vs runner 重跑偏差 > 20% → **reject**（v4.5 是 flag，v4.6 升级为 reject）

**Layer 6 — Test-set design（已有 + 加强）**

**§5b.13 Hidden set 密封 + 轮换 + 私有 generator**（v4.5 R8-P1-2 + B-P1-2 加强）
- v4.5: epoch 轮换 + sealed hash commitment
- v4.6 加强：hidden set 由 **private generator 随机生成**（不只是固定列表），每次 verified rerun 从 generator pool 抽 5-10 个 — 防 D4 生日攻击

**Layer 7 — Cross-cutting（P1）**

**§5b.14 Cross-task cache scrub**（F5 防护）
- 每个 task 跑前清：`/tmp/triton-ascend-cache/*` / device memory / Python module cache
- candidate process 用新 namespace 启动

**§5b.15 Public delayed disclosure**
- verified ranking 前 5 名的 solution.py source 公开延迟 2 周
- 防 F9 solution laundering 即时复用

**§5b.16-§5b.24 其他**（详见 reward_hack_defense.md spec file）：
16. Hash chain 完整性 verification（v4.5 已含）
17. Sandbox network isolation（v4.5 已含）
18. Tar safe extraction（v4.5 已含）
19. Manifest canonical signature（v4.5 已含）
20. Sandbox device allowlist（v4.5 已含）
21. cgroup resource limit（v4.5 已含）
22. Output entropy soft check（仅 flag 不 fail）— codex 建议作 soft signal
23. Memory lower bound proof（quarantine appeal 时要 submitter 提交）
24. Third-party audit cycle（v4.5 已含，月度独立 5+ 题）

### 5b.3 Process Isolation 架构图（v4.7 R11-P0-1/2 重写）

**Correctness pass**（candidate process #1）：

```
┌─────────────────────────────────────────────────────┐
│ Runner Process (trusted, parent)                    │
│ ├─ load oracle_pytorch.py (in-memory)               │
│ ├─ compute golden_outputs (在 runner 进程)            │
│ ├─ NOT serialize golden to candidate sandbox        │
│ ├─ compute input_hash_before                        │
│ │                                                    │
│ ├─ fork() → ┌──────────────────────────────────────┐│
│ │           │ Candidate Process #1 (correctness)   ││
│ │           │ - mount: kernel.py (ro)              ││
│ │           │ - mount: meta.yaml (ro)              ││
│ │           │ - mount: inputs.pt (ro)              ││
│ │           │ - mount: trusted_shim.so (ro)        ││
│ │           │ - NOT mount: golden_outputs (致命!)   ││
│ │           │ - NOT mount: oracle_pytorch          ││
│ │           │ - NOT mount: ascendc_baseline        ││
│ │           │ - --network=none / seccomp / cgroup  ││
│ │           │                                      ││
│ │           │ trusted_shim wraps:                  ││
│ │           │   out = solution.npu_triton_impl(in) ││
│ │           │   write output.safetensors (v4.8     ││
│ │           │     R12-P0-1: 禁 .pt/pickle，用       ││
│ │           │     safetensors 防 sandbox escape)   ││
│ │           └──────────────────────────────────────┘│
│ │ exit + kill -9 process group                     │
│ ├─ read candidate-written output.safetensors (安全) │
│ ├─ compute input_hash_after                         │
│ ├─ assert input_hash_before == input_hash_after     │
│ ├─ output dtype/device/shape/stride/storage check   │
│ ├─ per-dtype sentinel check (v4.7 R11-P2: 不只 NaN)  │
│ ├─ torch.allclose(out, golden, atol_by_tol_class)   │
│ ├─ differential 3 组 randomized inputs (重复上面)    │
│ └─ correctness verdict: ok / fail                   │
└─────────────────────────────────────────────────────┘
```

**Timing pass**（candidate process #2，与 #1 完全隔离，不共享 cache / state）：

```
┌─────────────────────────────────────────────────────┐
│ Runner Process (continue, fresh fork)               │
│ ├─ scrub cross-task: /tmp/triton-ascend-cache,      │
│ │   device memory, /tmp/scratch                     │
│ ├─ fresh inputs (with same shape configs)           │
│ ├─ fork() → ┌──────────────────────────────────────┐│
│ │           │ Candidate Process #2 (timing)        ││
│ │           │ - same mount as #1                   ││
│ │           │ - trusted_shim wraps (v4.12 R16-P1   ││
│ │           │   warm-up vs measured input 分离)：  ││
│ │           │   # Phase a: warm-up 用 warm_inputs   ││
│ │           │   #   (同 shape 不同随机值, 不计时)   ││
│ │           │   warm_inputs = gen_inputs(           ││
│ │           │       shape_config, seed=warm_seed)   ││
│ │           │   for _ in range(10):                 ││
│ │           │     solution.npu_triton_impl(         ││
│ │           │       warm_inputs)                    ││
│ │           │   torch.npu.synchronize()             ││
│ │           │   # Phase b: 重生成 fresh measured    ││
│ │           │   #   inputs（防 B4 warm-up 预算）   ││
│ │           │   meas_inputs = gen_inputs(           ││
│ │           │       shape_config, seed=meas_seed)   ││
│ │           │   torch.npu.synchronize()             ││
│ │           │   host_t0 = clock_gettime_ns(         ││
│ │           │            CLOCK_MONOTONIC_RAW)       ││
│ │           │   event_start = npu.Event(); record() ││
│ │           │   # broker 已编译 .so + 签名 →         ││
│ │           │   # 此后 100 trials 无新 JIT          ││
│ │           │   measured_outputs = []               ││
│ │           │   for trial in range(100):            ││
│ │           │     out = solution.npu_triton_impl(   ││
│ │           │             meas_inputs)              ││
│ │           │     if trial % 10 == 0:               ││
│ │           │       measured_outputs.append(out)    ││
│ │           │   event_end = npu.Event(); record()   ││
│ │           │   torch.npu.synchronize()  # device   ││
│ │           │                            # -wide    ││
│ │           │   host_t1 = clock_gettime_ns(         ││
│ │           │            CLOCK_MONOTONIC_RAW)       ││
│ │           │   # 权威: host_ms（cover all streams）││
│ │           │   host_ms = (host_t1-host_t0)/1e6     ││
│ │           │   # 仅 cross-check: event_ms          ││
│ │           │   event_ms = event_end.elapsed_time(  ││
│ │           │              event_start)             ││
│ │           │   # 偏差 > 10% → 查 stream list       ││
│ │           │   write host_ms + event_ms + sampled  ││
│ │           │     measured_outputs to pipe          ││
│ │           │ - candidate 看不到 shim 内 timing 代码 ││
│ │           └──────────────────────────────────────┘│
│ │ exit + kill process group + clear cache          │
│ ├─ read host_ms / event_ms / sampled outputs        │
│ ├─ 若 |host_ms - event_ms|/host_ms > 10% →          │
│ │   profiler 查 stream list；任何 non-runner stream │
│ │   → runtime_status=invalid_timing +              │
│ │     performance_flags=["invalid_stream"]         │
│ ├─ runner profiler trace cross-check                │
│ ├─ absolute sanity bound (§5b.6)                    │
│ ├─ NPU kernel event presence check                  │
│ ├─ fallback detector (LD_AUDIT + profiler op)       │
│ ├─ **v4.12 R16-P1: measured outputs correctness**   │
│ │   runner 对 meas_inputs 算 golden_meas_outputs    │
│ │   torch.allclose(sampled_measured, golden_meas)   │
│ │   → fail 则 perf_status=quarantined + flag       │
│ │     "measured_outputs_drift"（防 measure 阶段 dummy）│
│ └─ performance verdict: ok / quarantined / fail     │
└─────────────────────────────────────────────────────┘
```

---

### 5b.25 CI Fast-Fail Prefilter（v4.13 R18-S1 新增）

**用 `sampled numeric allclose`，不是 stdout 文本**：

```python
def fast_fail_prefilter(out_c, out_o, tolerance_class):
    """CI 加速：仅过滤明显不一致的 case，不替代严格 allclose"""
    tol = tolerance_policy.lookup(tolerance_class)
    flat_c, flat_o = out_c.flatten(), out_o.flatten()
    n = flat_c.numel()
    
    # 抽样 raw element（runner secret RNG 选 sample index，candidate 不可预测）
    sample_idx = runner_secret_rng.choice(n, size=min(1000, n), replace=False)
    sampled_c, sampled_o = flat_c[sample_idx], flat_o[sample_idx]
    
    # 用同一容差判抽样
    if not torch.allclose(sampled_c, sampled_o, **tol):
        return FAIL_QUICK  # fast-fail（不要等 full allclose）
    return PROCEED_TO_FULL_ALLCLOSE  # fast-pass 不存在
```

**关键约束**：
- 抽样用 raw element + `torch.allclose`，**不是** stdout 文本字符串对比
- Fast-FAIL 模式：fast-fail 后该 trial 失败，但若样本通过仍走 full allclose
- 输入由 runner secret RNG 生成（防 A4 hardcode）
- 仅减少明显失败 case 的耗时，**不简化通过 case 检查**

### 5b.26 L1 Oracle 简化三档（v4.13 R18-S2 新增）

`meta.yaml.oracle_simplified` 字段三档：

| 档位 | 适用 | 例子 | oracle 写法 |
|---|---|---|---|
| `simple_pytorch_ok` | 语义有 PyTorch native fallback 等价的算子 | `silu_and_mul` / `gelu_and_mul` / `add_bias` | `F.silu(x[..., :d]) * x[..., d:]`（一行） |
| `formula_exact_manual` | 数学公式直白但需手写 | `_rms_norm_kernel` / `layer_norm` / `softmax` | 显式分步 PyTorch eager（5-15 行）|
| `upstream_oracle_required` | 量化/RNG/topk/layout/bitmask 等无 PyTorch 等价 | `_per_token_group_quant_int8` / `_gumbel_sample_kernel` / `_topk_log_softmax_kernel` / `_triton_mrope_forward` / `pack_bitmatrix` | 必须手写完整 oracle（含 dtype cast / RNG / scale 时机），可能 30-50 行 |

**强制**：所有 sample / quant / mRoPE / bitmask / state-passing / scan 类题必须 `upstream_oracle_required`，不可简化。

### 5b.27 Cross-Oracle Drift 处理（v4.13 R18-S3 新增）

当三方 oracle（PyTorch ref / GPU triton / AscendC native）不一致时：

```
1. 不预判谁错（不自动认 "PyTorch 真理 + vLLM Triton 错"）
2. 该题不进 S_native（仅进 score_all 次榜）
3. 冻结 semantic contract + tolerance
4. owner 调研后再决定：
   - 调宽 tolerance？
   - 修 oracle / wrapper？
   - 报 vLLM upstream issue？
   - skip_registry？
5. 决定写入 contract.md "Cross-oracle resolution" section
```

**理由**：benchmark 目标是迁移 vLLM/SGLang **真实**生产 kernel；PyTorch ref 不一定是 upstream Triton 的"权威真理"。

### 5b.28 Hidden Generator 三随机 + 日志脱敏（v4.13 R18-S5 新增）

**Hidden adversarial 必须同时随机**：
- shape（防 B7 shape-route：固定 hidden shape 可被 hardcode 攻击）
- value distribution（防 A4 hardcode：随机 random/sparse/binary/extreme 值分布）
- edge cases（防 A7 容差边缘 fitting：随机 mask 比例 / NaN 注入 / 边界值）

**Hidden 失败日志脱敏**（防 D1 反推）：
```
日志只含：
- "L3_04 hidden case failed: numerical_diff_exceeded"
- "L1_12 hidden case failed: nan_inf_inconsistency"
- aggregate count: "12 hidden cases passed, 3 failed"

不含：
- expected/actual tensor 具体数值
- hidden shape 具体维度
- input random distribution 描述
```

### 5b.29 Candidate Stdout 不进 Verdict（v4.13 R18-S6 新增）

**candidate 进程的 stdout**：
- 仅作 **capped ≤64KB debug log**
- **不进入** `correctness_status` / `runtime_status` / `performance_status` 判定
- Hidden case 跑 candidate 时 **stdout 被丢弃**（不写入 debug log，防 hidden value 经由 candidate `print` 泄露）
- 防止 candidate `print(expected_tensor_string)` 伪造输出通道

---

## 6. Submitter Artifact Schema（v4.1 E2 详细规范）

**submission.tar.gz 强制结构（v4.3 R6-3 改 canonical files[] manifest）**：

```
submission.tar.gz
├── manifest.json   # v4.3 R6-3: canonical files[] manifest
│   {
│     "schema_version": "1.0",
│     "benchmark_version": "v0.1",
│     "submitter": "<organization or anonymous>",
│     "submission_id": "<uuid>",
│     "submitted_at": "<iso8601>",
│     "track": "npu_triton" | "hybrid" | "self_reported",
│     "tasks": ["L1_01_bias", ...],
│     "files": [    # v4.3 R6-3: 列出 tar 内每个文件（manifest.json 自身除外）
│       {"path": "solutions/L1_01_bias.py", "sha256": "<>", "size": 1234},
│       {"path": "results/L1_01_bias.json", "sha256": "<>", "size": 567},
│       {"path": "logs/L1_01_bias.log",      "sha256": "<>", "size": 8901},
│       {"path": "runtime_env/Dockerfile",          "sha256": "<>", "size": 2345},
│       {"path": "runtime_env/pip_lockfile.txt",    "sha256": "<>", "size": 6789},
│       {"path": "runtime_env/docker_digest.txt",   "sha256": "<>", "size": 65},
│       ...
│     ]
│   }
│   # v4.4 R7-2: files[] 的精确范围（解循环定义）：
│   #   - files[] 覆盖所有 payload 文件
│   #   - files[] **不包括** manifest.json 自身、不包括 signature.sig（否则签名文件 hash 进被签 manifest 形成循环依赖）
│   #   - tar 允许的完整集合 = {manifest.json} ∪ {signature.sig} ∪ files[]（一一对应，path 不重复）
│   #
│   # signature.sig 签的 = SHA256(canonical JSON serialization with sorted keys)
│   # canonical: json.dumps(manifest, sort_keys=True, separators=(',', ':'))
│   # 验签流程：
│   #   1. 解 manifest.json → 用 sorted-key 序列化重算 hash
│   #   2. 用 GPG/Sigstore 验 signature.sig 对应 manifest hash
│   #   3. 对 tar 内 payload 文件（即 files[]），sha256 必须与 manifest.files[*].sha256 匹配
│   #   4. tar 内 payload 文件不在 manifest.files[] → reject；files[] 列出但 tar 缺失 → reject
│   #   5. tar 内 path 不重复
├── solutions/
│   ├── L1_01_bias.py    # NPU triton 实现，hash 入 manifest
│   └── ...
├── results/
│   ├── L1_01_bias.json  # result.json per task
│   └── ...
├── runtime_env/
│   ├── Dockerfile              # 可复现的容器
│   ├── docker_digest.txt       # 提交时实际使用的 image digest
│   ├── pip_lockfile.txt        # 完整 pip freeze
│   ├── cann_version.txt        # CANN 8.5.0
│   ├── triton_ascend_version.txt
│   ├── npu_smi_snapshot.txt    # 提交机硬件信息
│   └── system_info.txt         # OS / kernel / Python / arch
├── logs/                       # 完整 raw timing logs（per task per shape per run）
└── signature.sig               # v4.3 R6-3: GPG / Sigstore 签名
                                # 签 SHA256(json.dumps(manifest, sort_keys=True, separators=(',', ':')))
                                # canonical manifest 覆盖整个 files[]，单点验签即覆盖全包
```

**v4.2 P2-1 tar 解包安全规则**（runner unpack 时强制）：

| 规则 | 拒绝条件 |
|---|---|
| 总大小限制 | tar 解包后总字节数 > 500 MB → 拒绝 |
| Symlink 禁止 | 任何 entry 是 symlink/hardlink → 拒绝 |
| Path traversal 禁止 | entry name 含 `..` 或绝对路径或 null 字节 → 拒绝 |
| 路径白名单 | entry 不在 `{manifest.json, solutions/*.py, results/*.json, runtime_env/*, logs/*, signature.sig}` 模式 → 拒绝 |
| 单文件大小限制 | 单 entry > 100 MB → 拒绝 |
| Entry 数量上限 | > 10000 entry → 拒绝 |
| Permission | entry 含 setuid/setgid bit → 拒绝 |

实现：fork python `tarfile` 时用 `tarfile.SafeExtractor`（Python 3.12+）或自实现 `safe_extract`。

**verified runner 校验流程**（v4.1 E3）：

```
1. Hash 验证：解 manifest.json → 检查每 task_bundle_sha 与 frozen task hash 是否一致
2. Static check：扫描 solutions/*.py，检测 forbidden import / torch high-level op / AscendC native call
3. Sandbox unpack：solutions 解到 sandbox，禁特权模式 / 禁网络 / cgroup 限制
4. Smoke test：跑 5 题 PoC（含 mock 正例 / 篡改负例 / fallback 负例）
5. Public shape 重跑：99 题 × small/medium/large × 3 run，对比 submitter 自报数据偏差 > 20% → flag
6. Hidden adversarial shape：5-10 个未公开 shape 重跑，验证非 overfit
7. **所有 verified 提交均 100% 官方重跑**（v4.4 R7-1 漏改修复：原 v4.2 旧口径"top-K 100% / 非 top-K 抽样 30%"已与 §4.9 R6-1 修订冲突，本步同步统一）；**top-K 触发额外 audit-rerun**（第二台 910B 独立重跑做 CV 对比）；**非 top-K 不做 sampling 作为 verified ranking 数据来源**
8. 第三方 audit：每月独立方在外部 910B 上重跑 top 5 提交至少 5 题（二次审计层）
9. 上榜 / quarantine / reject
```

## 7. v4 vs v3.1 总结

| 维度 | v3.1（错误定位） | v4（正确定位） |
|---|---|---|
| 项目本质 | 既做 benchmark 又做 NPU 实现 | 只做 benchmark |
| Phase 数 | 6（含 Phase 4a + 4b NPU 实现 + Phase 5 开源） | 5（Phase 0/1/2/3 + 开源） |
| 总人月 | 24-60 | **10-17** |
| 总周期 | 22-36 周 | **7-13 周** |
| Kernel engineer | 5-7（写 NPU 实现）| 1-2（写 task 文件 + harness） |
| NPU 硬件 | 2-3 张 | **1 张**（CI 用 mock） |
| 类比 KernelBench | 不直接 | **直接对标**（提供 task + harness + baseline，不写实现） |
| Leaderboard | 可选 | **核心交付**（接受 submitter 的载体） |
| migration_score 评什么 | "我们的 NPU 实现"（不存在） | **submitter 的 NPU 实现**（leaderboard 排名）|
| Phase 4a/4b | 6-14 周 NPU 实现 | **删除** |
| Risk R10 NPU SPOF | 高优 | 降级 |
| 满员 vs 三线 fallback | 复杂 | 不再相关 |

---

## 8. v4.1 vs v4 总结

| 维度 | v4 | v4.1 |
|---|---|---|
| 承诺等级 | 单一（10-17 人月 / 7-13 周） | 拆 self-reported (10-17 人月) vs verified (16-24 人月) |
| Provider Surface | eval/timing/scoring/adversarial/static_checker（5 模块） | + dataset/manifest/validator/HF/compile_cache/analysis/profile/leaderboard_validator（**13 模块**）|
| Submitter API | "填 task 里的 npu_triton_impl" | task/solution 严格分离 + hash 验证 + 重设计 CLI（6 命令）|
| Leaderboard | 单一 leaderboard | 两级：self-reported + verified |
| Score 公式 | 三档（score_native_only / score_all / score_fallback_only） | 主榜 FastP@1.0/@1.5 + capped geomean；禁 ∞ 主榜；NPU triton track 禁 fallback |
| AscendC oracle 命名 | "native oracle reference"（混淆 correctness 与 performance） | "native baseline adapter"（清晰）+ timebox 0.5-1 天/题 |
| Submission artifact | result.json 单文件 | submission.tar.gz 完整结构（manifest + solutions + results + runtime_env + logs + signature）|
| 硬件 | 1 张 910B | self-reported 1 张；verified 2 张或 rerun pool |
| PoC Go/No-Go | 5 题 GPU triton + harness 雏形 | + 独立 solution / hash 篡改 / mock 正例 / fallback 负例 / result schema / NPU smoke |
| R12 触发 | leaderboard 启动 | **v1.0 verified leaderboard** 才触发 |

## 9. 立即可启动的工作（v4.1）

v4.1 定位 + 交付边界全锁定后，**Phase 0 启动只需 3 件 sign-off**：

1. **RFC_0001**：开源 vs 内部；leaderboard 走 self-reported only (v0.1) 还是直接到 verified (v1.0)
2. **资源**：v0.1 = 1 张 910B + 1 卡 GPU + 2-3 FTE；v1.0 = 2 张 910B + 1 卡 GPU + 3.5-5 FTE
3. **License**：KernelBench Apache 2.0 fork + vLLM/SGLang/sgl-kernel-npu vendored snapshot 法务

签完字后启动 Phase 0（1-2 周）→ Phase 1 (4-6 周) + Phase 2 (3-4 周，并行) → Phase 3（2-3 周）→ Phase 4 v0.1（1-3 周） → 可选 Phase 4 v1.0（额外 5-7 周）= **v0.1 self-reported 9-15 周，v1.0 verified 14-20 周**（v4.8 R12-P1 同步）

## 10. v4.3 状态

codex round 6 在 v4.2 上给 **4.7/5**，明确 "**not frozen**"，提了 6 条 finding。

v4.3 把这 6 条全部硬化：
- **R6-1**：verified 所有提交 **100% 官方重跑**（非抽样 30%），top-K / 第三方仅二次审计
- **R6-2**：`capped_geomean` ok case 加 $\max(\epsilon, \cdot)$ 包裹，防 t_npu=0 / 异常大值导致 log 爆炸
- **R6-3**：canonical `files[]` manifest（列出 tar 内每个文件 path+sha256+size），signature.sig 签 deterministic JSON
- **R6-4**：sandbox candidate 严格 `--network=none`，上传由 runner 外层完成
- **R6-5**：triton artifact 两阶段 allowlist（build 阶段收集 + 登记，run 阶段仅允许已登记）
- **R6-6**：删 `performance_status=capped`（cap 仅 scoring 内部），资源表统一 v0.1 11-18 / v1.0 15-24

**v4.3 状态**：codex round 7 给 4.75/5，**not frozen**，6 条 finding 全部 [confirmed] 或 [likely]。

## 11. v4.6 状态 — 架构级升级

v4.5 codex round 9 整体 plan 给 4.95/5（spec/protocol 一致性 frozen）。但**专项 reward hacking review** 揭示子域仅 4.2/5，需要 v4.6 架构级升级。

v4.6 整合两大批架构修订：
- **修订 A**：Task 文件目录化（kernel/oracle/meta/shapes/contract/ascendc_baseline 6 文件分离），符合工程界主流"算子模型"设计
- **修订 B**：Reward hacking 防御从 15 原则扩到 24 原则（5b 完整章节），含 10 个 P0/P1 必做项 + KernelBench 3 个对抗 kernel 借鉴

v4.6/v4.7 资源 / 周期影响（v4.7 R11-P2 修订 Phase 2 工作量低估）：
- Phase 1 task 生成从 3-5 → **4-6 周**（多文件 + canary），人月 3-5 → **4-7**
- Phase 2 evaluator 加 process isolation + trusted timing shim + artifact provenance + AST semantic checker + profiler verdict + canary framework：**+1-2 周 / +0.5-1.5 人月**（v4.6 写 "多 2-3 天" 严重低估）
- 总周期 v0.1 self-reported: **9-15 周**（v4.5 原 7-13；v4.6 8-14；v4.7 因 Phase 2 +1 周 9-15）
- 总人月 v0.1: **12.5-21.5**（v4.5 原 11-18；v4.6 12-20；v4.7 +0.5-1.5）
- 总人月 v1.0: **17.5-27.5**（v4.5 原 15-24；v4.6 17-26；v4.7 +0.5-1.5）

**Spec 文件数量统一**（v4.7 R11-P2）：
- **基础 spec（5 个）**：task_api_contract / tolerance_policy / scoring_spec / timing_arbitration / extraction_failure_taxonomy
- **Reward / security spec（6 个）**：reward_hack_defense / ast_denylist / process_isolation / canary_task_design / artifact_provenance / submission_throttling
- **合计 11 个**（v4.6 文中出现 5/6/7 不同数字，v4.7 统一为 11）

**v4.6 状态**：codex round 11 给整体 4.70 / reward hacking 子域 4.55，**not frozen**，14 条 finding（含 2 致命 P0）。

## 12. v4.7 状态

codex round 12 给 v4.7 整体 4.74 / RH 4.58，**not frozen**，16 条 finding（含 2 新 P0：pickle escape + stream timing 漏计）。

## 13. v4.8 状态

codex round 13 给 v4.8 整体 4.82 / RH 4.66，**not frozen**，8 条 finding（1 P0 timing 图传导 + 7 P1/P2 propagation）。

## 14. v4.9 状态

codex round 14 给 v4.9 整体 4.86 / RH 4.72，**not frozen，0 P0 + 7 P1**（全部传导/口径统一问题）。

## 15. v4.10 状态

codex round 15 给 v4.10 整体 4.89 / RH 4.76，**not frozen，0 P0 + 3 P1**（artifact 双权威 / ABI metadata / warm-up JIT）。

## 16. v4.11 状态

codex round 16 给 v4.11 整体 4.90 / RH 4.78，**not frozen，仅 1 P1**（timing warm-up vs measured 同输入）。

## 17. v4.12 状态

codex round 17 给 4.90/4.78，**not frozen** (1 P1 measured input cache hack)；round 18 简化方案专项 review 又提 6 条 → v4.13。

## 18. v4.13 状态

v4.13 整合 round 18 6 条修正（sampled numeric fast-fail / L1 三档 / cross-oracle drift 不预判 / A5 soft signal / hidden 三随机 / stdout 不进 verdict）。新增 5b.25-5b.29 五条防御原则到 RH 章节（24 → 29 条）。

**预期 round 19 → ✅ FROZEN**。

## 12. 9 轮 review-iterate loop 完整回顾

| Round | Plan | 评分 | Δ | 关键变化 |
|---|---|---:|---:|---|
| 1 | v1 | 2.7 | — | 6 phase / 11-13 周 / 24 人月 |
| 2 | v2 | 3.8 | +1.1 | 7 phase / 能力栈五线 / 三档 score / R8/R9/R10 |
| 3 | v3.1 | 4.3 | +0.5 | 5 spec 抽出 / decision tables（但 **scope 错** — 把 NPU 实现纳入 provider） |
| 4 | v4 | 3.9 | -0.4 | **定位修正**：删 Phase 4 NPU 实现；总人月 24-60 → 10-17；总周期 22-36 → 7-13 |
| 5 | v4.1 | 4.6 | +0.7 | Provider Surface 5 → 13 模块；submitter API task/solution 分离；两级 leaderboard |
| 6 | v4.2 | 4.7 | +0.1 | signature 安全 / tar 解包 / capped_geomean 公式 / fallback allowlist |
| 7 | v4.3 | 4.75 | +0.05 | 100% rerun / canonical files[] / no network / triton artifact 两阶段 |
| 8 | v4.4 | 4.85 | +0.1 | §6 漏改 / signature 循环 / t_npu 边界 / allowlist 信任 / 资源同步 / 旧 API 清理 |
| 9 | **v4.5** | **4.95** | **+0.1** | **FROZEN** — runtime_status enum / hidden logs 分层 / 周期口径统一 |

**累计采纳 80+ 条 finding，无任何 finding 被反驳**。所有采纳决策都经过 CLAUDE.md "review external feedback critically — never auto-accept" 原则的客观评估。

## 13. Phase 0 启动只需 3 个 sign-off

1. **RFC_0001 决策**：开源 vs 内部；leaderboard 走 self-reported only (v0.1) 还是直接到 verified (v1.0)
2. **资源到位**：v0.1 = 1 张 910B + 1 卡 GPU + 2-3 FTE；v1.0 = 2 张 910B + 1 卡 GPU + 3.5-5 FTE
3. **License**：KernelBench Apache 2.0 fork + vLLM/SGLang/sgl-kernel-npu vendored snapshot 法务

签完字后立即启动 Phase 0（1-2 周）→ Phase 1 (4-6 周) + Phase 2 (3-4 周，并行) → Phase 3（2-3 周）→ Phase 4 v0.1（1-3 周）→ 可选 Phase 4 v1.0（额外 5-7 周）（v4.9 R13-P1 同步）

**v0.1 self-reported benchmark 9-15 周上线 / v1.0 verified leaderboard 14-20 周上线**（v4.8 R12-P1 同步）
