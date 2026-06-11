# 打分规范

> English: [../../reference/scoring_spec.md](../../reference/scoring_spec.md)

> 打分模型的面向机器的细节。权威模型见 `docs/DESIGN.md §4`；本规范不得与其产生偏离。

这是一个 GPU-Triton → 910B NPU-Triton **迁移**基准，采用**双目标**打分：正确性覆盖率是**排名准入门**；910B Roofline 归一化的性能效率是**被排名的指标**。两者都使用固定的可打分集合 `S`（单一来源：`ascendbench/release_gate/eligibility.py`）；`|S|` 是固定的，永不缩减（不允许子集逃逸）。

## 目标 1 — 正确性门

对每个任务 `t ∈ S`，每个 shape 通过 `tolerance.compare(candidate, golden, class)` 进行比对；结果汇总为状态组（`correctness_status` / `runtime_status` / `performance_status` + `fallback_detected` + `performance_flags`）。`fallback_detected` 在打分前会被归一化为一次正确性失败，因此门谓词明确为 `correctness_status == ok ∧ ¬fallback_detected`。

$$\text{native\_coverage} = \frac{\big|\{t \in S:\ \text{correctness\_status}_t = \textsf{ok}\ \wedge\ \neg\,\text{fallback}_t\}\big|}{|S|},\qquad \text{ranking eligibility}:\ \text{native\_coverage} \ge 0.70$$

覆盖率上没有 ε；缺失/失败的任务对分子贡献 0，但**不会**从 `|S|` 中移除。低于 0.70 → 仅进入 `self-reported` 榜，不进入 `verified` 排名。

## 目标 2 — Roofline 归一化性能（被排名的指标）

910B 与源 GPU 的峰值能力不同，因此绝对延迟无法跨硬件比较；性能相对 910B Roofline 进行归一化。对任务 `t`、shape `s`：

- `W_c(t,s)` = 计算类 `c` 的 FLOPs，`M_r(t,s)` = 数据资源 `r`（HBM / 跨设备通信）的字节数 —— 来自每任务的成本模型，并针对一次插桩的 GPU-Triton 参考运行做交叉校验（fail-closed；`task_api_contract.md §cost_model`）；
- `π_c` / `β_r` = 该类 / 资源对应的 910B 峰值算力 / 带宽（打包于 `ascendbench/policies/peak_table_910b.toml`）；
- **Roofline 理想延迟**（标准 Roofline 下界；假设资源理想重叠 → 最紧的理论下限；完全串行不重叠则应改为求和）：

$$t_{ref}(t,s) = \max\!\Big(\max_c \tfrac{W_c}{\pi_c},\ \max_r \tfrac{M_r}{\beta_r}\Big)$$

- 实测 NPU 延迟 `t_npu(t,s)`（host 权威计时；`timing_arbitration.md`）；
- 每 shape 的效率 `e(t,s) = min(P_CAP, t_ref / t_npu)`，理论上 `∈ (0,1]`（`t_ref` 是下界）；`> 1` 仅来自成本模型低估 / 计时噪声，会被 `P_CAP = 5.0` 钳制并标记待复核。

$$s_t = \begin{cases}\big(\textstyle\prod_s e(t,s)\big)^{1/n_t} & \text{all three statuses ok and no fallback} \\ \epsilon = 0.01 & \text{otherwise}\end{cases}\qquad \text{capped\_geomean\_native} = \exp\!\Big(\tfrac{1}{|S|}\textstyle\sum_{t \in S}\log s_t\Big)$$

三个状态：`correctness_status == ok`、`runtime_status == ok`、`performance_status == ok`；外加 `fallback_detected == false`。`baseline_ns = t_ref`，`measured_ns = t_npu`。ε 填充与固定的 `|S|` 分母在性能指标上的应用方式与门完全一致 —— 非 ok / fallback 的任务得分为 ε，且不会从 `|S|` 中移除（不允许子集逃逸）。

### 头部指标

- `native_coverage` —— 排名准入门（正确性）。
- `capped_geomean_native` —— 被排名的指标（Roofline 效率几何均值）；`s_t ≈ 1` 表示 910B Roofline 已被打满。
- `RooflineP@τ`（仅诊断用）= `S` 中 `e ≥ τ` 的占比（例如 τ ∈ {0.5, 0.8}）；不是排名键。

仅正确性（`p = ∞`）**不是**头部指标 —— 它是门，不是排名。

## 状态字段

```
correctness_status ∈ {ok, fail, error}
runtime_status     ∈ {ok, timeout, oom, compile_fail, invalid_timing}
performance_status ∈ {ok, suspicious, quarantined}
performance_flags  ∈ list of {low_performance, roofline_exceeded, invalid_stream,
                              measured_outputs_drift, suspected_buffer_reuse,
                              thermal_throttle, high_variance, ...}
fallback_detected  ∈ bool   # normalized to a correctness failure before scoring
```

`performance_flags` 是一个列表，不属于枚举的一部分。

## 发布门（S6）

可发布集合即与 C2 对齐的可打分集合（`compute_release_ready_eligible` = `compute_score_eligible` ∩ `is_provenance_eligible`，即 `oracle_kernel_equivalence ∈ {VERIFIED, CEILING_ACCEPTED}`）。在该集合之上，一个任务只有在 `release_status == PASS` 时才可发布，其中

```
release_status == PASS  ⟺  phase_a_prime.status == PASS
```

（S2 可复现的切断验证冻结）。`cost_model.status` 是有 ncu 证据支撑的性能 / 访存元数据（条件性访存、L2 复用、tile 几何、HBM 突发下限，使得一个可移植的解析式字节公式在结构上无法在所有地方命中每 shape 的容差）—— 它**不是**发布的阻断项。任何与 C2 对齐的任务若 `release_status != PASS` 都会阻断基准发布（`validate_tasks.py --release-gate`）。BLOCKED 是硬性的，无法通过把任务从 `|S|` 中移除来规避。处于 C2 对齐集合之外的任务（例如 `oracle_kernel_equivalence == BLOCKED_UPSTREAM_BUG`）会被排除在已发布的语料之外。

## 合理性边界（Sanity Bounds）

| 条件 | 处理 |
|-----------|--------|
| `runtime_status = invalid_timing`（t_npu ≤ 0 / NaN / Inf） | 硬失败 |
| Profiler 捕获到 0 个 NPU kernel 事件 | 硬失败（很可能是 CPU fallback / 空 kernel） |
| `hardware_utilization > 100%` | 物理上不可能 → 与 `e > 1` 同样处理：钳制 + 标记 `roofline_exceeded` 强制复核（成本模型 / peak-table / 测量伪影），而非静默硬失败 |
| `e > 1`（t_npu < t_ref） | 成本模型低估 / 噪声 → 在 `P_CAP` 处钳制，标记 `roofline_exceeded` 待复核 |
| `e < 0.1`（远低于 Roofline） | **不是**硬失败 —— 标记 `low_performance` |
| host 与 event 计时漂移超过阈值 | 按 `timing_arbitration.md`：`runtime_status=invalid_timing` + 标记 `invalid_stream` |
| 实测采样输出 ≠ golden_meas_outputs | `performance_status=quarantined` + 标记 `measured_outputs_drift` |

## 硬件利用率（辅助诊断）

$$\text{hardware\_utilization} = \max\!\Big(\frac{\text{achieved}_{HBM}}{\beta_{HBM}} \times 100,\ \frac{\text{achieved}_{FLOPS}}{\pi_{class}} \times 100\Big)\,\%$$

910B 峰值（`π_class`、`β`）来自打包的 peak 表（示例性地：HBM ≈ 1.6 TB/s、FP16 cube ≈ 256 TFLOPS、INT8 ≈ 512 TOPS —— 权威值存放在打包的表中）。这是解析式 `t_ref` 的经验对应物；属诊断用，不是排名键。

## 结果 JSON Schema（每任务每次运行）

```json
{
  "task_id": "L1_12_rms_norm",
  "task_bundle_sha": "<frozen sha>",
  "solution_sha": "<submitter sha>",
  "runner_hash": "tab-0.1.0+<git sha>",
  "container_digest": "sha256:...",
  "hardware_id": "910B-A2-card-0",
  "software_hash": {"cann": "8.5.0", "torch_npu": "2.8.0", "triton_ascend": "3.2.0"},
  "version_matrix_sha": "<from ascendbench/policies/version_matrix.toml>",
  "commit_sha_repo": "<bench repo>",
  "commit_sha_upstream": {"vllm": "...", "sglang": "..."},

  "correctness_status": "ok",
  "runtime_status": "ok",
  "performance_status": "ok",
  "performance_flags": [],
  "fallback_detected": false,

  "per_shape": {
    "small":  {"median_ms": 0.042, "p99_ms": 0.051, "cv": 0.03, "trials": 100,
               "t_ref_ns": 38000,   "roofline_efficiency": 0.90},
    "medium": {"median_ms": 0.318, "p99_ms": 0.389, "cv": 0.04, "trials": 100,
               "t_ref_ns": 290000,  "roofline_efficiency": 0.91},
    "large":  {"median_ms": 2.140, "p99_ms": 2.580, "cv": 0.05, "trials": 100,
               "t_ref_ns": 1950000, "roofline_efficiency": 0.91}
  },
  "runs": [
    {"run_id": 1, "summary": {}, "temp_c": 52, "freq_mhz": 1850},
    {"run_id": 2, "summary": {}, "temp_c": 53, "freq_mhz": 1850},
    {"run_id": 3, "summary": {}, "temp_c": 54, "freq_mhz": 1850}
  ],

  "adversarial": {
    "extreme_small":    {"correctness": true, "ms": 0.012},
    "long_seq":         {"correctness": true, "ms": 12.5},
    "non_align_hidden": {"correctness": true, "ms": 0.345}
  },

  "denominator_source": "roofline_ideal",
  "baseline_ns": 290000,
  "measured_ns": 318000,
  "s_t_value": 0.91,
  "cost_model_status": "PASS",
  "release_status": "PASS",
  "hardware_utilization_pct": 91.0,

  "reward_hack_flags": []
}
```
