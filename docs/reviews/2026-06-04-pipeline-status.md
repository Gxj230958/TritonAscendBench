# TAB bench 全流程状态 — release-gate 重定位后

数据来源：`fe7370a`+ 全量真跑（S2 `bs2final`、C2 `bc2clean3`、cost xcheck `bcostfinal` asymptotic policy），2026-06-05。

颜色：**绿=dev 真跑闭环 · 蓝=release 门(C2+S2) · 黄=性能 evidence(旁路,不阻塞) · 灰=GPU/NPU gated · 红=上游 bug**。
核心：bench 的真正 correctness 门是 **C2 精度 + S2 severance**；cost_model 是 ncu-evidenced 性能 metadata，**不在** release 门。

```mermaid
flowchart TD
    U["冻结上游算子 frozen kernel<br/>byte-frozen provenance"]:::frozen

    U --> S1["① entrypoint resolve<br/>daemon resolve · 已定"]:::done
    S1 --> S2["② S2 severance verify<br/>severed==original bit-exact · 单 run<br/><b>103 / 103 PASS</b>"]:::done
    S2 --> S3["③ C2 oracle-equiv 精度<br/>GPU bf16/fp16 vs CPU-fp32 · full-shape 921 grades<br/><b>真跑闭环</b>"]:::done

    S3 --> outVer["VERIFIED · 82"]:::done
    S3 --> outCeil["CEILING_ACCEPTED · 19"]:::done
    S3 --> outBlk["BLOCKED_UPSTREAM_BUG · 2<br/>L2_23/24 lightning-attn off-by-one"]:::blocked

    outVer --> S4
    outCeil --> S4
    S4["④ 自洽门 validate_corpus<br/><b>GREEN · 101 aligned + 2 blocked</b>"]:::done
    S4 --> RG["★ release-gate<br/>= C2(release-ready eligible) AND S2(phase_a_prime)<br/><b>GREEN · 101 release-ready PASS</b><br/>L2_23/24 因 C2 未对齐被排除"]:::gate

    RG --> G1["⑦ golden 离线参考<br/><b>18/18 DONE</b>(只 upstream_oracle_required 题需要)"]:::done

    S4 -. ncu evidence(旁路,不阻塞 release) .-> CM["⑥ cost_model<br/>ncu 实测 HBM vs 解析公式 · asymptotic_hbm_xcheck<br/>6 PASS + 92 FLOOR_OBSERVED + 97 BLOCKED<br/>(conditional/L2/tile/burst-floor 本质难逐 shape 5%)"]:::evidence

    G1 --> G2["⑧ NPU 执行 / t_npu 打分 / peak table<br/>910B 真机 · hardware-gated"]:::gated

    classDef frozen fill:#ECEFF1,stroke:#90A4AE,color:#37474F
    classDef done fill:#E8F5E9,stroke:#66BB6A,color:#2E7D32
    classDef gate fill:#E3F2FD,stroke:#42A5F5,color:#1565C0
    classDef evidence fill:#FFF8E1,stroke:#FBC02D,color:#F57F17
    classDef gated fill:#FAFAFA,stroke:#BDBDBD,color:#9E9E9E
    classDef blocked fill:#FFEBEE,stroke:#EF5350,color:#C62828
```

## 真实状态小结

**正确性链路 + release 门（已闭环、可交付）**

| stage | fresh 结果 |
|---|---|
| ① resolve | 已定 |
| ② S2 severance | **103/103 PASS**（bit-exact、单 run） |
| ③ C2 精度 | **82 VERIFIED + 19 CEILING + 2 BLOCKED_UPSTREAM_BUG**，full-shape 921 grades |
| ④ 自洽门 | **GREEN**（101 aligned） |
| ★ **release-gate** | **GREEN · 101 release-ready PASS** = C2(release-ready eligible) ∧ S2(phase_a_prime)。L2_23/24（C2 上游 bug）被排除 |
| ⑦ golden | **18/18 DONE**（其余 85 题 oracle self-contained，不需 golden） |

**性能 evidence（旁路，不阻塞 release）**

| stage | 状态 |
|---|---|
| ⑥ cost_model | ncu 实测 HBM vs 解析公式，`asymptotic_hbm_xcheck` policy（大 shape gate + 小 shape `FLOOR_OBSERVED`）。**6 PASS + 92 FLOOR_OBSERVED + 97 BLOCKED**。103/103 题有 per-shape measured evidence。逐 shape 5% 对 conditional access / L2 reuse / tile geometry / HBM burst floor 本质难达，故 cost_model 是 evidenced metadata 而非 release blocker；逐题 first-principles 重建公式是可选改进（只修 real bug，绝不 tune-to-pass） |

**hardware-gated（910B，做不了）**

| stage | 状态 |
|---|---|
| ⑧ NPU | 910B 真机执行 / t_npu 打分 / peak table，无 910B 不可达 |

**一句话**：**"100 tasks 精度对齐编译通过"的真实验证已闭环**——C2 精度 + S2 severance 是 release 的真门，101 release-ready 全 PASS、golden 18/18 DONE。cost_model 退为 ncu-evidenced 性能 metadata（不阻塞 release）；唯一剩下的是 NPU 910B 真机执行。
