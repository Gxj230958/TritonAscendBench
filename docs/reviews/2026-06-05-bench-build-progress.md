# TAB bench 构建进度 (2026-06-05)

bench = GPU-Triton → 910B NPU-Triton 迁移评测。构建流程分 6 步,两大目标:
**Obj1 correctness(精度对齐+编译=进榜资格)** 已闭环;**Obj2 性能排名** 的标尺(cost_model)已诚实定格,最终排名待 910B 真机。

```mermaid
flowchart TD
    U["上游 vllm / sglang<br/>103 个 Triton 算子"]:::src
    E["① 抽取 + 冻结 (dev)<br/>symbol-closure severance<br/>→ frozen kernel.py + _frozen_deps"]:::done

    subgraph O1["Obj1 correctness (已闭环 ✅)"]
        S2["② S2 severance (GPU)<br/>severed == original bit-exact + N=5 确定性<br/>103 / 103 PASS"]:::done
        C2["③ C2 精度对齐 (GPU)<br/>frozen kernel vs CPU-fp32 oracle, full-shape<br/>83 VERIFIED · 18 CEILING · 2 上游bug"]:::done
        GATE{"④ release gate = C2 ∧ S2<br/>GREEN · 101 release-ready"}:::gate
        S2 --> C2 --> GATE
    end

    subgraph O2["Obj2 性能 (标尺定格)"]
        CM["⑤ cost_model 标尺 (GPU ncu xcheck)<br/>理论 W_c / M_r vs 实测 DRAM<br/>54 PASS · 6 FLOOR_DOMINATED · 43 BLOCKED"]:::evidence
        ROOF["⑥ 910B Roofline 排名<br/>peak π_c/β_r · 实测 t_npu · 效率 e · capped_geomean<br/>⚠ 需 910B 真机"]:::gated
        CM --> ROOF
    end

    U --> E --> S2
    GATE -->|"101 进榜"| CM
    GATE -. "L2_23/24 上游 off-by-one bug(不可修)" .-> OUT["✗ 不进榜"]:::blocked

    classDef src fill:#ECEFF1,stroke:#90A4AE,color:#37474F
    classDef done fill:#E8F5E9,stroke:#66BB6A,color:#2E7D32
    classDef gate fill:#E3F2FD,stroke:#42A5F5,color:#1565C0
    classDef evidence fill:#FFF8E1,stroke:#FBC02D,color:#F57F17
    classDef gated fill:#FAFAFA,stroke:#BDBDBD,color:#9E9E9E
    classDef blocked fill:#FFEBEE,stroke:#EF5350,color:#C62828
```

## 当前在第 ⑤-⑥ 步之间

**①-④ 全绿(已闭环可交付)**:抽取→冻结→S2 编译/severance→C2 精度对齐→release gate。
**⑤ cost_model 标尺已诚实定格(黄)**:本轮工作主要在这。
**⑥ 910B 性能排名(灰)**:hardware-gated,eval daemon 是 CUDA GPU 不是 910B NPU,够不到。

## 本 session 的工作(都落在 ③⑤)

| 工作 | 步 | 结果 |
|---|---|---|
| cost_model 根因深挖 | ⑤ | 49 BLOCKED 主因不可 portable(GPU L2 write-absorb + data-dependent gather),codex confirmed |
| ncu profiler 修(3 真 infra bug) | ⑤ | L1_02/L1_31/L2_33 从 no-evidence → 有真 DRAM evidence,GPU 验证 |
| FLOOR_DOMINATED 重分类(6 个) | ⑤ | sub-burst index kernel 诚实归类(source-audited allowlist,L2_26 防误标) |
| **L2_03 oracle TF32 bug 真修** | ③ | CEILING → **VERIFIED**(补 oracle 漏的 TF32 truncation,GPU 验证全 9 shape) |
| L3_04 对抗审计 | ③ | codex 纠正:是真 ceiling(flash-vs-naive streaming drift),非 bug,不硬修 |

全程 codex 监督(7+ 轮),抓出 C2 sneak、L2_26 误标、L3_04 假 bug 等,防自嗨。

## 诚实定格的硬上限(不可在 dev+CUDA 消除)

- **C2 的 18 CEILING**:真 bf16/TF32 dtype 硬件上限(已对齐到极限);**2 上游 bug**(L2_23/24)不可修
- **cost_model 43 BLOCKED**:GPU L2 write-absorb + data-dependent gather,first-principles 不可 portable;硬刷=tune(红线)
- **⑥ 910B 排名**:需 910B 真机(peak table / t_npu)

**一句话**:dev + CUDA GPU 能做的(①-⑤ 全链)已做到诚实极限并可交付;唯一剩的 ⑥ 是 910B 真机阶段。
