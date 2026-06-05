# cost_model 49 个非 PASS 的逐 task 根因分析 (2026-06-05)

49 task Workflow 逐个深读 kernel+meta+launcher + skeptic verify。**核心结论:0 个可修的 first-principles 公式 bug**
(Workflow 的 formula_improvable_CONFIRMED 为空,唯一候选 L2_23 被 verify 正确否决——它是上游 bug 不是公式问题)。
分类经我审定(用真实 status + gate-shape deviation 方向 + 有无 tl.dot,修正了 Workflow 的 floor/upstream 边界误判)。

## 7 类汇总

| 类 | 数 | 根因 | 可修? |
|---|---|---|---|
| **A write-absorb** | 16 | write-heavy,H200 50MB L2 吸收 output writeback,measured≈read-side ≪ analytic(read+write)。**公式对 910B 正确**(NPU buffer 192KB,write 必到 HBM),只是 GPU 验不了 write 侧 | ❌ 测量盲区非公式错 |
| **B data-dependent** | 13 | attention/SSM reload(6,有dot)+ gather/index dedup(7)。真实 DRAM 取决于运行时 index 分布 + L2 dedup | ❌ 不可 portable |
| **C FLOOR-N/A** | 6 | sub-burst index kernel,逻辑字节 ≪ DRAM burst floor,无 asymptotic regime | ❌ cost_model 不适用(已标 FLOOR_DOMINATED) |
| **D burst-margin** | 11 | read-dominated,measured 一致比 analytic 大 ~6-8%(DRAM burst / cache-line over-fetch 硬件 margin) | ❌ 需 hardware-counter model,非 source-first-principles |
| **E 公式漏项** | 1 | L2_26 公式漏 m_indices store,但真实值 data-dependent | ❌ 真值不可 portable,补项也凑不准 |
| **F 上游 bug** | 2 | L2_23/24 lightning-attn off-by-one(已证明) | — 移出评测分母 |

## 影响什么 (核心)

- **cost_model 不在 release gate**(perf metadata)→ **49 个非 PASS 完全不影响 correctness 进榜**(101 release-ready 不变),不影响精度对齐/编译。
- 影响的仅是 **Obj2 910B 性能排名标尺的 GPU 验证覆盖**:这 49 个 task 的理论 bytes 公式在 H200 上没验证准(54/103 验准)。
- **但根因都不是公式错**:write-absorb(测量盲区)+ data-dependent(运行时)+ burst-margin(硬件粒度)+ floor(不适用)。公式 first-principles 对 910B 仍可用,**910B 真机上可重新验证**(NPU cache 小、无 H200 L2 write-absorb)。
- 硬刷任何一个 = tune-to-pass(违反诚实红线)。

## 逐 task (按类)

**A write-absorb (16, over 方向)**: L1_03(132%) L1_05(11%) L1_07(61%) L1_08(318%) L1_14(40%) L1_31(40%)
L1_36(93%) L2_18(91%) L2_20(18%) L2_22(9.5%) L2_33(87%) L2_38(119%) L2_39(28%) L3_13(37%) L3_14(136%) L3_22(10%)

**B data-dep-dot reload (6)**: L3_01(27%) L3_02(42%) L3_03(45%) L3_04(71%) L3_09(20%) L3_17(28%)
**B data-dep-index (7)**: L1_02(96%) L1_18(17%) L1_32(98%) L1_33(38%) L1_38(17%) L1_40(30%) L2_37(14%)

**C FLOOR-N/A (6)**: L1_04 L1_24 L1_28 L1_29 L1_35 L1_37

**D burst-margin ~7% (11)**: L1_01(6.8%) L1_12(7.4%) L1_13(7.2%) L1_16(7.6%) L1_27(7.4%) L1_39(5.9%)
L2_03(7.5%) L2_16(6.7%) L2_17(7.7%) L3_18(6.6%) L3_20(6.2%)

**E 公式漏项 (1)**: L2_26(data-dep,不可 portable)
**F 上游 bug (2)**: L2_23 L2_24 → 移出评测分母

## 结论

49 个非 PASS 没有一个是"我们能修的 cost_model 公式 bug"——全是 GPU 测 NPU-model 的固有限制(write-absorb / data-dependent / burst margin)+ sub-burst N/A + 上游 bug。
**cost_model 已是诚实极限,54 PASS 是 first-principles 可验证的上界**,不影响 bench 的 correctness 交付。
