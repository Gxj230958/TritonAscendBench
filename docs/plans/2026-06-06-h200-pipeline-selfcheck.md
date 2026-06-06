# H200/910B Roofline pipeline 自检 + 全题排名 — 最终实现 plan

状态: READY for implementation (codex 二轮 sign-off, 设计层 0 FAIL; 填 peak 全排名方向; 2026-06-06)
日期: 2026-06-06
硬件: daemon 实际是 **H200** (非 H100; version_matrix [gpu]/[hardware.h100] 型号记错, 配置 bug)。
      H200=H100 同 GH100 die, compute peak 同 (fp16=989/int8=1979/FP32 vector=67T/L2=50MB),
      唯一差 HBM: **H200 HBM3e beta=4800GB/s** vs H100 3350。version_matrix hbm=3350 需修。

## 0. 目标 (bench 底线, 不可妥协)

**所有 101 题必须在相应硬件 (H200 / 910B) 上拿到排名。** 做不到 = cost-model 失职。
当前只覆盖 13/101 (88 题引用 vector class, 910B vector peak=0 占位 -> 算不出 t_ref -> 被排除)。
本 plan 让 **101 题全部可排名**, 同时验证排名机制本身 work。

## 1. 问题陈述

Roofline 评分链 `cost_model 公式 -> t_ref -> e -> s_t -> 排名` **从没用真实 measured 端到端验证过**,
且当前数值就是错的:
**铁证 (亲验)**: `shape_efficiency` 里 `e=t_ref/t_npu` 无单位转换; `t_ref` 秒 (scoring.py:252,310),
`t_npu` 纳秒 (eval.py:730 `measured_ns=t_npu`); 差 1e9 -> 真实 `e≈1e-9`。真跑过排名全废、不可能没发现
-> 从没真跑过。当前 scoring 测试全合成数 (同单位才没暴露)。

## 2. 核心策略: 填 vector peak (别留 0) -> 所有题算 t_ref -> 全排名

`t_ref = max(W_c/pi_c, M_r/beta_r)`。88 题算不出 t_ref 的唯一原因是 `vector pi=0`。**给 vector peak
填一个有 provenance 的值, 所有 101 题立即可算 t_ref、可排名。**

**为什么排名仍对 (codex 仲裁 confirmed)**: `t_ref` 是 per-task 公共分母 (eval.py:720 每 shape 算一次,
不随候选变), 跨候选比时约掉: `score_A/score_B = geomean(t_npu_B/t_npu_A)`。=> **vector peak 不精确
不影响相对排名**, 只影响"达 roofline 百分之多少"的绝对解读。
**唯一代价 (codex confirmed) -- 两端饱和**: `e=min(P_CAP, t_ref/t_npu)` 有**两端**: ①peak 偏小 ->
t_ref 偏大 -> e 撞 **P_CAP** 上限 -> 候选都=P_CAP 不区分; ②e 任意小时 `s_t_value()` floor 到
**EPSILON** 下限 (scoring.py:375) -> 弱 kernel 也不区分。撞任一端那题就丢区分。
=> **填 confirmed SKU 真值** (见第3节), **两端饱和都要报告, 并由 C3 验证 good/bad 对不饱和** -- 不是
单靠"填高避 cap" (填高会把弱 kernel 推 EPSILON)。

## 3. vector peak 来源 + provenance 体系

| 硬件 | vector peak | provenance | 用途 |
|---|---|---|---|
| **H200** | FP32 FMA **67T** (CUDA-core); special-func 走 SFU ~4T | `nvidia-official` (公开规格) | **verified**: 排名 + 绝对效率% 都可用 |
| **910B** | FP16 **20T**(910B2C/48AIV) 或 **16.7T**(910B4/40AIV) / FP32 ~半 | `third-party-academic-spec` (arXiv 2505.14022 Table1 [confirmed] "910B2C Vector@FP16=20T" + CSET 支持; 微架构 48AIV×128×2×1.6GHz≈19.6T 一致。**注: USENIX DeepServe 不确认 vector=20T, 只给聚合 400T**) | **排名可用**, 绝对效率% **不可** (需实测) |
| 910B (最终) | on-device microbench 实测 | `on-device-measured` | verified: 排名 + 绝对% 都可用 |

**SKU 注意**: 20T 是 910B2C (48 AIV); 910B4 (40 AIV) ≈16.9T。**bench 机器 SKU 必须 `npu-smi` 确认**
后填对应值。华为官方从不单列 vector peak ([confirmed] 11 搜索+datasheet+DaVinci 论文)。
**按 confirmed SKU 填真值** (48 AIV->20T, 40 AIV->16.7T 线性缩放; `npu-smi` 确认 SKU)。
**不为避 cap 偏离 confirmed SKU 值** -- cap-hit 高时单独标记/处理, 不是故意填高覆盖真值 (codex 纠正)。

## 4. P0 前置 blocker (当前代码真 bug/gap, 全 [confirmed] 有行号; 没修完排名数值是错的)

- **P0-1 单位 bug (critical)**: t_ref 秒 vs t_npu 纳秒直除 (scoring.py:313 / eval.py:715,735)。
  **修法**: 保持 `compute_t_ref()` 返回秒, 在 eval/scoring 边界 `t_ref_s*1e9` 转纳秒喂
  `shape_efficiency()` (ns/ns), 保 `baseline_ns`/`measured_ns` schema (specs/scoring_spec.md:30)。
  **不要** t_npu 转秒。测试钉: 1e-6s->1000ns; t_npu=1000ns->e=1.0; 2000ns->e=0.5; emitted
  `baseline_ns==1000`; **覆盖 per-shape 和 task-level** (task_measurement 复用 r.t_ref eval.py:196)。
- **P0-2 peak 路径不可选 (到 eval + analyze)**: peak_table.py:29 默认 910b; `_apply_roofline`
  (eval.py:657) 无路径; `cli analyze`(cli.py:1067,1086) 也从默认表推 completeness。**修法**: peak
  路径可配 (env/参数), 传到 eval + CLI run + suite + analyze; 默认仍 910b。
- **P0-3 backend 路由 (不完整)**: `backend="gpu"` 没传到 `run_two_pass`/child; child 无条件
  `NpuBackend` (eval.py:916,1119 / process_isolation.py:320); inputs CPU 序列化 (inputs.py:1087)。
  **修法**: backend arg threading + child 侧 tensor move 到 CUDA 再 `fn(inp)`。
- **P0-4 timing 无 device sync**: host timing 只 wrap Python call (process_isolation.py:599)。spec 要
  measured 前 sync + event end 后 sync (timing_arbitration.md:26,41)。**修法**: 加 pre/post device
  sync; 对齐协议 10 warmup/100 trials (当前 5/30, process_isolation.py:164)。
- **P0-5 provenance trap + 双 flag (codex confirmed, critical for 填 peak)**: 当前只有**单一 authority**
  `roofline_data_complete` (scoring.py:126 / eval.py:668 / cli.py:1067), 语义="verified positive-finite
  peak" -> 填的 academic-spec peak 会被当 complete 然后**误读为 absolute-verified**。**修法**:
  ① peak_table schema 加显式 `provenance` 字段 (`nvidia-official | huawei-official |
  third-party-academic-spec | architecture-estimate | on-device-measured`), load 解析;
  ② **拆两个独立判定** (不能共用单一 roofline_data_complete):
  `ranking_usable` (任何有 provenance 的正有限 peak, 含 academic-spec/estimate -> 进排名) 与
  `absolute_verified` (仅 `{nvidia-official, huawei-official, on-device-measured}` -> 可绝对效率%声明)。
  cli analyze 不能把 academic-spec 当 absolute-verified。**`absolute_verified` 按 task 引用的每个
  peak 逐一判** (该 task 所有 referenced peak 都在 absolute_verified 集才算, 不用宽泛 "all cube" --
  codex: bf16_cube 当前标 unverified 但无 task 引用; per-referenced-peak 才不漏)。
- **顺手**: scoring.py:318 注释 sign 反 (e>1 是 over-estimate 不是 under)。

## 5. 红线 (codex 仲裁: 收窄不废)

- **估算/academic-spec peak 用于产生完整排名 = 可以** (相对排名 robust, t_ref 约掉; 除非撞 cap/floor)。
- **绝对效率%声明 = 必须 `absolute_verified` peak** = `{nvidia-official, huawei-official,
  on-device-measured}`。`third-party-academic-spec` / `architecture-estimate` 是 **ranking-only**,
  标注后只服务排名。(910B 的 cube/int8/hbm 是 huawei-official -> 这些题绝对% 可信; vector 是
  academic-spec -> vector-bound 题绝对% 待实测)。
- **报 cap-hit counts**: 每 task 报多少 shape/候选撞 P_CAP。高 cap-hit = 该 task peak 估偏低、区分度
  受损 -> 提示调高 vector peak (within reason) 或等实测。
- 不编造无任何依据的 peak; 不拿 H200 PASS 冒充 910B verified; 区分度对照不可造假。

## 6. 自检验收标准 (C1-C4; H200 上验证机制 work)

- **C1 全题连通**: **101 题** (填 H200 vector peak 67T 后) 端到端算出 e, **0 raise / 0 incomplete**。
  这直接验证"填 peak -> 全排名"的底线。
- **C2 范围自洽 (两端饱和)**: 每 `e in (0, P_CAP]`; **报 per-task `cap_hit_count` +
  `epsilon_floor_hit_count` + min/p5 e** (上下两端饱和都看); 无 e<=0 / NaN。
- **C3 区分度 (灵魂)**: 排名只看 t_npu, 必须证明机制真能区分快慢 kernel:
  - C3a 机制: 固定 t_ref 放大 t_npu k 倍 -> e ÷k (合成)。
  - C3b 真实: vendored 原 kernel e 落合理高位 (不撞 cap)。
  - C3c 对照: 先 `block_size`/`num_warps` variants (gap 预声明+重复稳定+非噪声); 再加**至少一个
    degraded 实现 per class** (mem-vector / cube-GEMM / write-absorb), 要求 `e_bad/e_good` 跟踪
    `t_good/t_bad` (tol 内); **且 good/bad 对两端都不饱和** (不撞 P_CAP 也不撞 EPSILON, 否则区分度是假的)。
- **C4 effective-HBM 自洽 (仅 absolute-claim 用; 概念层二轮判 0 FAIL)**: 绝对效率指标定义为
  "effective external-HBM 带宽利用率" (非 logical bytes), write-absorb 下 `e=(M_eff/beta)/t_npu`
  自洽。**profiler 子门** (仅 absolute claim, 不阻塞排名): ① 仅 HBM-active-max shape; ② 排除
  launch-overhead tiny shape; ③ write-absorb 要 per-candidate DRAM throughput 在 tol 内。
  注: **C4 是绝对效率%的守卫, 不是排名的前提** -- 排名不依赖 absolute t_ref scale, **除非 cap/floor
  饱和** (坏的 effective-HBM/t_ref 致 e 撞 P_CAP 或 EPSILON 时排名也受影响)。

## 7. H200 -> 910B 迁移 + 命名防自欺

H200 自检过 = 证明**机制** work + H200 上 101 全排名 (verified peak)。**不证明 910B 公式/timing**。
**命名 + artifact (写进产物 + 验收测试)**: 结果标 `H200_SELF_CHECK_PASS`, 永不 `PIPELINE_VERIFIED`;
产物带 `hardware_id` / `peak_table_path` / `peak_table_device` / `peak_provenance` /
`self_check_scope` / `not_valid_for_910b`。910B 上线: vector 填 academic-spec 即可**全排名**;
但 `910B_ABSOLUTE_EFFICIENCY_VERIFIED` 另需 `910B_PEAK_MEASURED` (实机 microbench) +
`910B_FORMULA_XCHECK_PASS` (实机 ncu)。
**验收测试**: H200 PASS 不满足 910B 的 measured/xcheck 门; academic-spec peak 标 provenance 后
`cli analyze` 不能把它当 verified-absolute (验 P0-5 provenance 解析)。

## 8. 实现步骤

1. 修 P0-1 单位 (ns canonical at boundary) + 测试钉 baseline_ns。
2. 修 P0-5 provenance: peak_table schema 加 provenance 字段; load 解析; verified vs ranking-usable。
3. peak 路径传播 (P0-2): eval + CLI run + suite + analyze。
4. backend 路由 (P0-3) + child tensor-to-CUDA。
5. timing pre/post sync (P0-4) + 10/100 协议。
6. 建 `peak_table_h200.toml` (全 nvidia-official: fp16=989/int8=1979/hbm=4800/vector=67T)。
7. `peak_table_910b.toml` vector 从 0 改 academic-spec 值 (SKU 确认后, 20T/16.9T, provenance=
   third-party-academic-spec), 不再留 0。
8. 跑 pilot (含 C3 区分度对照 + cap-hit 报告), 再全 101 连通 (C1)。

**最小子集 (8-12 task)**: 2 mem-bound vector + 2 cube/GEMM + 2 write-absorb + 1 floor + 1 mixed-bound
+ 1 high-pi vector (如 L2_25, 验高 pi_req 下 cap-hit 行为) + 1 special-function (SFU, 验 peak 取值)。

## 9. 诚实预期 (codex 对抗审计后, 非乐观)

- **排名覆盖: 101/101** (填 peak 后全部可算 t_ref)。这是底线, 达成。
- **绝对效率% 可信度分层** (不是全可信):
  - H200: 67T verified -> 101 题绝对效率% 都可信。
  - 910B: vector=academic-spec -> 排名可信, **绝对效率% 标 "based on academic-spec, pending
    on-device measurement"**; cube/int8/hbm verified 的题绝对% 可信。
  - 真正需实测精化的: 少数 compute-bound vector task (t_ref 实质依赖 vector peak) 的绝对%。
- **cost_model 公式量级 (层1 cross-check)**: 54 PASS/47 non-PASS 是测量物理限制, 本 plan 不解决,
  也不影响排名 (status 不 gate)。"公式 101 健康" = "都 scorable", 非 "已验证正确"。

## 10. 风险

- **SKU 不确认就填错 vector peak**: 必须 npu-smi 确认 910B SKU (40 vs 48 AIV)。
- **P0-3/P0-4 真实 device 延迟**: backend 路由+tensor-to-CUDA+sync 可能是隐藏大头。
- **C3 区分度**做不出 good/bad gap -> 机制未证明。
- **cap-hit 失控**: 若 peak 估太低致大量 task 撞 cap -> 排名分辨率塌 -> 调高 peak 或标记等实测。
- **GPU 反作弊诊断 daemon-side 待实现**: `GpuBackend._profile_cuda` 是 H200 seam, dev 返回
  全 `None` 诊断 (kernel-event / non-default-stream / fallback 信号)。timing + correctness 路径
  不受影响 (None 被 eval 忽略), 但 GPU 上的 reward-hacking 门暂关, 待真 CUDA profiler trace 解析
  在 daemon 上实现。**不在 dev 盲写 profiler parser** — 信号未经真机验证不可信。
