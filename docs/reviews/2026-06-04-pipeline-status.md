# TAB 自动化抽取+验证链路 — 全流程真跑状态

快照时刻：2026-06-04 ~21:00 CST。全量真跑(S2 severance + C2 精度)后的真实状态。

颜色含义：**绿=dev 侧已真跑闭环 · 黄=正在重验 · 灰=GPU/NPU 实测 gated(dev 做不了) · 红=上游 bug 阻断**。
关键区分：上半组是**自动化抽取+精度验证链路**(dev 可真跑、已闭环);下半组是 **release-certification**
(必须 GPU/NPU 真机实测才能认证,dev 做不了,by-design 未完成)。

```mermaid
flowchart TD
    U["冻结上游算子 frozen kernel<br/>byte-frozen provenance"]:::frozen

    subgraph extract["A. 抽取+精度链路 -- dev 真跑已闭环"]
        S1["① entrypoint resolve<br/>daemon resolve · 已定(cache)"]:::done
        S2["② S2 severance verify<br/>severed==original bit-exact<br/>等价 103/103 PASS · 单 run<br/>(promote 双-run bug + L3_10 stale run 已修, 重验通过)"]:::done
        S3["③ C2 oracle-equiv 精度<br/>GPU bf16 vs CPU-fp32 · full-shape 921 grades<br/>真跑完成"]:::done
        S4["④ 自洽门 validate_corpus<br/>GREEN · 101 aligned + 2 blocked"]:::done
        S6a["⑥a cost_model 公式<br/>schema 可求值 evaluable"]:::done
    end

    S3 --> outVer["VERIFIED · 82"]:::done
    S3 --> outCeil["CEILING_ACCEPTED · 19"]:::done
    S3 --> outBlk["BLOCKED_UPSTREAM_BUG · 2<br/>L2_23/24 上游 off-by-one"]:::blocked

    subgraph cert["B. release 认证 -- GPU/NPU 实测 gated"]
        S6b["⑥b cost_model.status<br/>需 GPU ncu 实测 HBM vs 公式<br/>RED · status 字段缺(无 CUDA 拒跑)"]:::gated
        S5["⑤ release-gate<br/>phase_a_prime PASS AND cost_model.status PASS<br/>RED · by-design until W4.3"]:::gated
        G1["⑦ golden 离线参考采集<br/>GPU-host artifact · 18 未采集"]:::gated
        G2["⑧ NPU 执行 / t_npu / peak table<br/>910B 真机 · hardware-gated"]:::gated
    end

    U --> S1 --> S2 --> S3
    outVer --> S4
    outCeil --> S4
    S4 --> S6a --> S6b --> S5 --> G1 --> G2

    classDef frozen fill:#ECEFF1,stroke:#90A4AE,color:#37474F
    classDef done fill:#E8F5E9,stroke:#66BB6A,color:#2E7D32
    classDef running fill:#FFF8E1,stroke:#FB8C00,color:#E65100
    classDef gated fill:#FAFAFA,stroke:#BDBDBD,color:#9E9E9E
    classDef blocked fill:#FFEBEE,stroke:#EF5350,color:#C62828
```

## 真实状态小结

**A 组 — 自动化抽取+精度链路(dev 真跑、已闭环、可信)**

| stage | 状态 |
|---|---|
| ① resolve | 已定(cache) |
| ② S2 severance | 等价 103/103 PASS · 单 run;此前 promote 对 65 题有"双-run + 丢 classB"bug + L3_10 stale fp32 run,已修(`_has_module_level_run` guard + fail-closed equality check + classB merge),**GPU 重跑通过** |
| ③ C2 精度 | 82 VERIFIED + 19 CEILING_ACCEPTED + 2 BLOCKED_UPSTREAM_BUG,full-shape 921 grades |
| ④ 自洽门 | GREEN(对抗审计确认是真门) |
| ⑥a cost_model 公式 | schema 可求值 |

**B 组 — release-certification(GPU/NPU 实测 gated,by-design 未完成)**

| stage | 为什么 gated |
|---|---|
| ⑥b cost_model.status | 需在 live CUDA 上用 Nsight Compute 实测 HBM traffic 与公式对账(无 CUDA 直接拒跑);dev 做不了 |
| ⑤ release-gate | = phase_a_prime PASS **AND** cost_model.status PASS;后者缺 → RED(代码注释:expected RED until W4.3) |
| ⑦ golden | GPU-host 离线参考,18 个 score-eligible 题未采集 |
| ⑧ NPU | 910B 真机执行 / t_npu 打分 / peak table,无 910B 不可达 |

**一句话**:**抽取等价 → full-shape 精度对齐 → 自洽** 这条核心链路 dev 侧已真跑闭环、可信;
**release 认证**(cost_model GPU 实测 / golden 采集 / NPU 执行)按设计需 GPU/NPU 真机,dev 做不了,尚未完成。
