# C2 精度对齐 — CEILING 对抗审计

目标:把 19 个 `CEILING_ACCEPTED` 逐个对抗审计,区分**真硬件 dtype 上限**(诚实保留)vs
**bench 侧可修 bug**(oracle 漏 emulate / launcher downcast / fixture dist 错 → 真修拉回 VERIFIED)。
不靠放松 tolerance、不 over-emulate oracle、kernel.py byte-frozen 不改。

## 当前真实全局状态(2026-06-05 全量盘点,103 tasks)

| 维度 | 状态 |
|---|---|
| S2 severance(抽取 + severed==original bit-exact) | **103/103 PASS** |
| C2 oracle-equivalence | **82 VERIFIED** + **19 CEILING_ACCEPTED** + **2 BLOCKED_UPSTREAM_BUG** |
| release gate | GREEN(101 release-ready PASS) |
| cost_model(perf metadata,非 release blocker) | 54 PASS / 49 honest BLOCKED(first-principles 极限) |

- 2 BLOCKED = `L2_23_fwd_kv_parallel` / `L2_24_fwd_kv_reduce` lightning-attn off-by-one **真上游 bug**
  (max_tol_units 1000–6200),正确 oracle 必然 diverge,诚实 BLOCKED,不可修。
- per-shape:848 VERIFIED + 55 CEILING_ACCEPTED + 18 FAILED(=L2_23/24 各 9 shape)。

## 19 CEILING 与现有机制(`triton_ascend_bench/ceiling_acceptance.yaml`)

- **Mode A**(18 task):co-located `max_tol_units ≤ 8.0`,GPU 实测,距真 bug ~125x — 证明放松 backward-stable。
- **Mode B**(L3_13):dump-proven per-key physical-scale bound(near-cancellation,max_tol_units 12.10)。

这套机制证明"放松安全",但**未回答**:这些 ceiling 能否通过修 bench 侧 oracle/launcher/fixture **消除**、
拉回真 VERIFIED?这正是本审计要答的(用户:"真实解决而非规避")。

## 关键样本(已读,支持"部分 ceiling 可真修")

L3_18/19/20 同 family(chunk scaled-dot,unpinned `tl.dot` 在 Ampere+ 跑 TF32 tensor core):

| task | oracle TF32 emulation | C2 |
|---|---|---|
| L3_18 | 对**所有** unpinned dot operand 做 `_to_tf32` 截断 | **9/9 VERIFIED** |
| L3_19 | 同样全 emulate | **9/9 VERIFIED** |
| L3_20 | 仅 emulate unpinned `w` dot(`u` dot IEEE-pinned 未 emulate) | 6 VERIFIED + **3 CEILING**(H3/P3/P4 大 K) |

→ oracle 是否忠实 emulate kernel 的真实 narrowing 直接决定 VERIFIED vs CEILING。L3_20 残差需深读
kernel+oracle+dump 判:是已 emulate 全部、剩 fp32 accumulation-order(真上限),还是漏了某 narrowing(可修)。

## 审计分类与红线

- **A 真上限**:oracle 已忠实 emulate kernel **literally 存在的**所有 narrowing,残差是 fp32/bf16
  accumulation-order 或 bf16-MMA,不可消除 without 复制 kernel exact tiling → 诚实 CEILING。
- **B 可修**:oracle 漏 emulate 某 narrowing / launcher dtype downcast / fixture dist 错 → 给具体 diff 提议。
- **C 需 dump**:需 GPU actual-output dump 才能判 A/B。

**红线(防 over-emulate = tune-to-pass 伪装)**:
1. kernel.py **byte-frozen**,绝不改。
2. 不动 tolerance / ceiling cap。
3. oracle emulation 必须 **1:1 对应 kernel.py 里 literally 存在的操作**(每个 emulation 指得到 kernel 源码行);
   不允许为凑 VERIFIED 往 oracle 塞 kernel 没有的 narrowing。
4. 改完必须 GPU daemon 重跑 C2,且**不能让任何已 VERIFIED shape 退化**。

## codex 对齐判据(2026-06-05,gpt-5.5:xhigh)

- **审计值得做,但是 source-grounded A/B/C,不是"把 19 CEILING 全变 VERIFIED 的运动"**。
- A vs B 充分判据:"literal kernel op" 必须含**完整 operand lineage**(fixture dtype → launcher `.to()`/cast → `tl.dot`/`tl.store`)、dot precision pin、operand folding order、mask、fast-math。
  - **A** = 无 source-derived op 漏掉,残差只是 reduction-order / accumulation / store-rounding,被现有 acceptance 界住。
  - **B** = 具体 kernel/launcher op 在 oracle 漏了或错了,且提议 emulation line-addressable + operand-exact。
  - **C** = source 分析无法唯一区分 backend 行为,或改动只在看了 pass/fail 数据后才显得好。
- 范例 anchor:**L2_03**(launcher fp32-cast + unpinned `tl.dot` + oracle plain einsum → 漏 TF32 = **B**);**L3_20**(kernel pin `u`、只 `w` unpinned,oracle 正确只 emulate `w` → 残差 pinned-u accumulation-order = **A**,别误判);**L3_18/19**(oracle 指名 unpinned dot 行 + operand-exact emulate = gold-standard)。
- 补充红线:no fitted constants / per-shape branches / error-derived correction;`allow_tf32=False`/`ieee` = TF32 emulation 的 **hard stop**;两个 emulation 都 pass → C(不选 nicer)。
- **GPU 验证防 self-fulfilling oracle**:普通 C2 rerun 只证 kernel-vs-new-oracle 一致,不能 catch tune-to-pass。须:① 改 oracle **前** freeze kernel actual dump;② source-derived candidate 只能改善被 cited op 影响的 output,pinned/control 路径(如 L3_20 的 `u`)不变;③ 独立高精度 ref(fp64/full-fp32)+ wrong-variant control(TF32 RNE vs truncate)作**诊断**,不作 grading target。

## 进度

- [x] 全量状态盘点
- [x] codex 对齐审计判据与红线
- [~] Workflow fan out 19 ceiling 对抗审计(进行中,38 agents:深读审计 → skeptic verify)
- [ ] 我 + codex review 审计结果,确认 B 类清单(防 over-emulate)
- [ ] 真修 confirmed-B(oracle/launcher/fixture,kernel byte-frozen)→ push + sync → GPU 重跑 + negative control 验证
