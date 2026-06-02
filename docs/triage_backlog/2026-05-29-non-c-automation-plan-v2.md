# 方案 v2:自动化攻克非 vllm._C 依赖的 task(codex review 后修订)

date: 2026-05-29
status: codex deep-reviewed (8 findings, 全部采纳), v2 incorporates all

## 修订动机:v1 被 codex 抓到 4 个硬伤

codex 深度 review v1 给 8 条 confirmed/likely findings(全部带源码引用),全部采纳:

| # | finding | 采纳决定 |
|---|---------|---------|
| 1 | Phase1 接入 overlap-gate 会破坏 D 的 cache fingerprint(kernel_resolver.py 不在 `_PROVENANCE_CODE_FILES`) | 采纳:Phase1 前置 = 先把 kernel_resolver.py 纳入 provenance hash |
| 2 | overlap>=50% 阈值会错分(L1_30 真入口 overlap 仅 0.33) | 采纳:废弃 50% 阈值 |
| 3 | overlap>50% 不证明真shapes(L2_14 漏 bias 语义) | 采纳:改用集合关系 + 候选完整性比较 |
| 4 | 我的 32/39 分诊用了被污染的 daemon bucket(任一候选 shapes-incomplete 即整题归 shapes) | 采纳:重新用 R10.a 集合关系分诊 |
| 5 | 已有更强判别器(R10.a SIGNATURE_MISMATCH/NEEDS_BRIDGE 集合包含) | 采纳:复用,不引入第三套启发式 |
| 6 | Phase2 6规则低估 inputs.py 语法(scalar_seq/struct/ref/null);sampling_metadata/target_logits 是上层构造不该补 | 采纳:明确 spec 子集白名单 |
| 7 | o/out->zeros 过于乐观(dtype/scale/alias/None) | 采纳:收紧 output buffer 规则 |
| 8 | 安全护栏不够(anchor 只跑 2 个 shape) | 采纳:验证跑全 public+hidden |

## 干净分诊(替代 v1 的 32/39,用 codex-endorsed 集合关系)

来源:`scripts/oracle_pyref_precompute.py --all`(集合包含关系,非阈值)。完整清单见
`docs/triage_backlog/2026-05-29-clean-set-relation-triage.md`。

```
26  ALIGNED_READY              (已能进 R10.b)
34  SIGNATURE_MISMATCH         wrapper required param 不在 shapes -> Phase2 自动补目标
31  NEEDS_BRIDGE               oracle 跟 wrapper 不匹配
     ├─ 16 resolver-returned-None   resolver 找不到入口 -> Phase1 目标
     └─ 15 oracle-extra-keys        oracle 读 wrapper 没有的 key -> 逐个判断
 7  WRAPPER_CONTRACT_MISMATCH  (rank unpack, 已有 auto_patch)
 1  RUNTIME_ERROR
 1  ORACLE_KERNEL_LAYOUT_MISMATCH
```

## v2 三条线(优先级排序)

### Phase 0(前置,必做):统一 resolver fingerprint
- 把 `triton_ascend_bench/kernel_resolver.py` 纳入 `resolve_upstream_entrypoints.py` 的 `_PROVENANCE_CODE_FILES` hash 列表(codex finding 1)
- 否则任何 resolver 改动不会让 cache stale,旧 verdict 被当 fresh = 破坏 fail-closed
- 这是 Phase 1 的硬前置

### Phase 1:统一 resolver 选入口逻辑(目标 16 resolver-returned-None)
- 把 kernel_resolver 的 meta.name-优先 + positive-overlap-gate 作为**可复用函数**暴露,resolve_upstream_entrypoints import 它做候选过滤
- 不直接替换 D 的 resolver(保留它的 cache/fail-closed),只共享"选对入口"判定核心
- codex finding 3:不只看单候选过阈值,要在多个候选里选**oracle-key 覆盖最完整**的(避免 L2_14 选 matmul_persistent 丢 bias 而非 addmm)
- 验证:16 resolver-None task 重跑,看多少选出正确入口;26 ALIGNED + 4 RESOLVED 不 regress

### Phase 2:自动补缺失 input(目标 34 SIGNATURE_MISMATCH 的子集)
codex finding 6/7:严格限定**可自动补的 spec 子集白名单**:
- ✅ 可自动补:`tensor`(同名 sibling 在 oracle,复用其 shape/dtype)、`scalar`(wrapper default 或 contract.md 明确值)、`constexpr BLOCK_SIZE_*`(autotune config 默认 + Triton 约束)
- ❌ 默认不自动补(surface 人工):`scalar_seq`、`struct`、`ref`(依赖 sibling 顺序+对象共享)、`null`、上层构造对象(`sampling_metadata`/`target_logits`/`input_buffers`)
- output buffer(`o`/`out`):收紧——必须确认 dtype + 无 alias/scale 配套才用 zeros,否则人工

### 安全护栏(codex finding 8,贯穿全程,最高优先)
- 任何自动补/选入口后,**必须跑该 task 全部 public + hidden shapes 的 oracle-vs-kernel 数值对比**,不只 anchor 2 个
- 只有全 shape 数值等价才标 RESOLVED;补了能 launch 但数值不对 = FAILED
- 绝不"能跑通就算 pass"

## 跟 D session 边界
- 不重写 loop/daemon/orchestrator
- 增量:Phase0 fingerprint + Phase1 共享 resolver 核心 + Phase2 扩 auto_patch
- 提交前跟 D session 对齐 resolve_upstream_entrypoints 改动

## 统一口径(消除两套 scoreboard)
- 收敛到 loop daemon + meta.yaml 单一 truth source,我的 R10.b 手工路径退役
- 每轮全量跑后 verdict 写回 meta.yaml provenance.golden

## 现实预期(codex 提醒 v1 的 30-45 偏乐观,下调)
- Phase 1(16 resolver-None):oracle 跟某真 wrapper 高重叠的能救——估 **8-14**
- Phase 2(34 SIGNATURE,白名单子集):tensor/scalar/constexpr 可补的——估 **12-18**
- 合计现实:从 4 提到 **24-36**(全量同环境,严格数值等价)
- 剩余:_C-class(需 venv _C)、segfault(LoRA)、oracle-mismatch、scalar_seq/struct/上层构造对象、真 NEEDS_BRIDGE,需人工或 Phase B golden

## 执行顺序
1. Phase 0 fingerprint(硬前置)
2. Phase 1 resolver(救 16,且不修 resolver 就补 shapes 会污染)
3. Phase 2 shapes 白名单自动补(救 SIGNATURE 子集)
4. 每 Phase 全量重跑 + 全 shape 数值等价验证 + 写回 meta.yaml
