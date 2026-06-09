验证确认了审计的全部地基事实(`tasks/level1=40 / level2=39 / level3=22`;`meta.yaml` 第 4 行 `level: L2` 无引号;`docs/kernel-select.md:5` 仍是 stale 的 `100 / L1 40·L2 38·L3 22`;`validate_tasks.py:101` 的 `VALID_LEVELS` 闭包枚举)。38 项变更的加减法也自洽(L1 40−13+9=36 / L2 39+13−12−9+4=35 / L3 22+12−4=30,合 101)。下面是完整可执行计划。

---

# Level 重分类执行计划 — TritonAscendBench(field-only / strategy A)

> 日期 2026-06-07 · 策略已批准:**field-only(A)**,38 题,终态 **L1=36 / L2=35 / L3=30**(含 `L2_41_selective_state_update → L3`)。
> 本文件本身将提交 codex 终审签字后方可推进。

---

## 1. 目标与最终态

### 1.1 策略定义(A:field-only)
只编辑 38 题 `meta.yaml` 的 `level:` 一行。**不移动目录、不重命名 task_id、不动 kernel.py / golden / provenance / contract.md**。
有意接受 **task_id 前缀(如 `L2_16`)与 `level` 字段(`L1`)发散**,且目录保持在 `tasks/level2/`。审计已证实:`level` 字段在生产路径只被读 1 处(`dataset.py:129` → `Task.level` int),该 int 只喂给两个 dev-side `cli.py` 点,**从不进入 scoring / golden / release-gate / daemon / ranking / provenance**。

### 1.2 "WHOLE bench completely fine" 的可验收定义
终态必须同时满足以下三组,缺一不可:

| 维度 | 终态判据 |
|------|---------|
| **A. dev 单测** | `pytest tests/unit_tests/` = **0 failed / 0 skipped**(含 23 个暂停的 pending fix 全部落地) |
| **B. 字段与校验** | 全 101 题 `meta.yaml` YAML 合法;`validate_tasks.py` 的**错误集与改前 baseline 完全相同**(dev 上有 **18 个 pre-existing `golden/ present but missing` 错误** —— golden safetensors 封存在 daemon/gpfs、dev 本地无,与 level 无关;改 level 不得增减任何错误。**exit 0 仅在有 golden 的 daemon 侧可达**,见 Gate 1 修正);`level` 字段分布精确 **L1=36 / L2=35 / L3=30**;`git diff` 仅 38 个 `meta.yaml`、每个恰好 1 行(`1 1`) |
| **C. 全量 GPU 回归** | C1/C2 oracle-equiv + phase_a_prime/S2 + cost_model_xcheck + golden + release-gate + gpu-rank 全 101 题重跑,逐 stage 与变更前 baseline **功能性 bit-identical(仅 38 个 meta.yaml 的 level 字节不同;golden bytes / resolve verdict / release-gate GREEN 101 全一致;gpu-rank 排名集合与顺序不变)** |

终态的本质:这是一次**零 GPU 语义影响**的元数据重标。GPU 全量回归不是用来"发现"变化,而是用来**证明**审计结论("level-agnostic 管线")成立 —— 期望 delta 为零。

### 1.3 38 项变更(按组,task_id 全列)
- **G1 — L1→L2(13)**:`L1_13_layer_norm_fwd` `L1_14_layer_norm_fwd_1pass` `L1_15_act_and_mul` `L1_16_silu_and_mul` `L1_17_gelu_and_mul` `L1_19_topk_log_softmax` `L1_20_log_softmax` `L1_21_gumbel_sample` `L1_22_min_p` `L1_23_penalties_and_temperature` `L1_24_rejection_sample` `L1_28_pack_bitmatrix` `L1_38_silu_mul_fp8_quant_deep_gemm`
- **G2 — L2→L3(12)**:`L2_02_chunk_scan_fwd` `L2_03_chunk_state_fwd` `L2_04_chunked_lora_expand` `L2_06_qkv_lora_b` `L2_07_gate_up_lora_b` `L2_09_sgemm_lora_b` `L2_19_fused_sigmoid_gating_delta_rule_update` `L2_20_causal_conv1d_update` `L2_21_causal_conv1d_fwd` `L2_30_fused_moe_router_tensorcore` `L2_33_fwd_grouped_kernel_stage1_rope` `L2_41_selective_state_update`
- **G3 — L2→L1(9)**:`L2_16_fused_rmsnorm` `L2_18_fused_gdn_gating` `L2_27_deepep_permute_triton` `L2_28_deepep_post_reorder_triton` `L2_32_experts_combine` `L2_35_pre_reorder_triton_kernel_for_cutlass_moe` `L2_36_post_reorder_triton_kernel_for_cutlass_moe` `L2_38_concat_and_cast_mha_k` `L2_40_gemma_qkv_rmsnorm`
- **G4 — L3→L2(4)**:`L3_07_correct_attn_cp_out` `L3_12_moe_mmk` `L3_21_solve_tril_16x16` `L3_22_chunk_local_cumsum_vector`

每题目录 = `tasks/level{task_id[1]}/{task_id}/`(目录由 task_id 前缀定位,**与新 level 无关**)。

### 1.4 重分类判定规则(rubric,供 kernel-select.md §3 引用)
可判定的三级口径,与上述分组一一吻合:
- **L1 — 单遍逐元素 / gather-scatter 向量算子**:无跨元素归约(或仅简单 per-row)、无 matmul、无递归。VV、low/medium。例(重标后):`fused_rmsnorm`、`fused_gdn_gating`、`deepep_permute/post_reorder`、`experts_combine`、`pre/post_reorder_for_cutlass_moe`、`concat_and_cast_mha_k`、`gemma_qkv_rmsnorm`。
- **L2 — 需多遍 / 数值敏感归约(softmax、layernorm、采样)或单次 matmul(CV,无递归 / 无 chunked-state)**:例:`layer_norm_fwd`、`act_and_mul`、`log_softmax`、`gumbel/min_p/rejection 采样`、`moe_mmk`、`solve_tril_16x16`、`chunk_local_cumsum_vector`、`correct_attn_cp_out`。
- **L3 — 有状态 / 递归 / chunked-state(SSM)、分组-segmented GEMM(LoRA-b)、多 stage attention、conv1d 序列**:跨 chunk 依赖或分组矩阵。例:`chunk_scan/chunk_state_fwd`、`selective_state_update`、`causal_conv1d_*`、`*_lora_b`、`fwd_grouped_kernel_stage1_rope`、`fused_sigmoid_gating_delta_rule_update`。

---

## 2. 阶段顺序(为何 unit-first)

```
阶段0  清 23 个 pending unit fix → 拿到 0-fail/0-skip 绿基线(独立 commit)
  └─ 同时:对全量 GPU 管线产物做 baseline 快照(resolve_all.json / golden manifest hash / ranking json / release-gate 输出)
阶段1  38× meta.yaml level 字段批量调整 + docs 对齐 + (可选) cli.py:257(独立 commit)
阶段2  全量回归:dev 单测 + 全 101 GPU 管线,逐 stage 与阶段0 baseline diff,断言"仅 level 字节不同"
```

**为何 unit-first(干净归因)**:23 个 pending fix 与 level 重标**无因果关系**。若把两者混在一个 commit / 一次回归里,阶段2 出现的任何单测红会无法判定是"level 改动引入"还是"pending fix 未清干净"。先把 baseline 打绿并单独 commit,阶段1 之后的任何回归即可**确定性归因**到 level 改动 —— 而审计已论证 level 改动不该引入任何单测红(§3f),所以阶段2 若出红,必是 baseline 没清干净或审计有盲点,二选一,无歧义。这也满足 CLAUDE.md "Test after each change" 与 "clean attribution"。

**baseline 快照(阶段0 末尾必做)**:在改任何 level 字节之前,从当前 GPU-PASS 101/101 状态导出/留存:① GPU 侧 `resolve_all.json`、golden manifest 的逐题 hash、gpu-rank 的 ranking json、release-gate 输出;② **dev 侧 `python3 scripts/validate_tasks.py 2>&1 | grep ERROR | sort > /tmp/vt_before.txt`**(18 个 pre-existing golden-missing 错误集,作 Gate 1 的 diff 基准 —— 见 §4)。没有 baseline 就无法证明 "bit-identical except level"。

---

## 3. 逐模块修正对齐清单

### 3a. 38× `meta.yaml` `level:` 字段(脚本化批量,只改 level: 行)

唯一必改项。用下述脚本保证**逐字节只动 `level:` 一行**(避免 ruamel/yaml 重排):

```python
#!/usr/bin/env python3
# scripts/oneoff/relabel_levels_field_only.py  (run from repo root)
import re
from pathlib import Path

ROOT = Path("tasks")
CHANGES = [  # (task_id, cur, new) — 38 total
 ("L1_13_layer_norm_fwd","L1","L2"),("L1_14_layer_norm_fwd_1pass","L1","L2"),
 ("L1_15_act_and_mul","L1","L2"),("L1_16_silu_and_mul","L1","L2"),
 ("L1_17_gelu_and_mul","L1","L2"),("L1_19_topk_log_softmax","L1","L2"),
 ("L1_20_log_softmax","L1","L2"),("L1_21_gumbel_sample","L1","L2"),
 ("L1_22_min_p","L1","L2"),("L1_23_penalties_and_temperature","L1","L2"),
 ("L1_24_rejection_sample","L1","L2"),("L1_28_pack_bitmatrix","L1","L2"),
 ("L1_38_silu_mul_fp8_quant_deep_gemm","L1","L2"),
 ("L2_02_chunk_scan_fwd","L2","L3"),("L2_03_chunk_state_fwd","L2","L3"),
 ("L2_04_chunked_lora_expand","L2","L3"),("L2_06_qkv_lora_b","L2","L3"),
 ("L2_07_gate_up_lora_b","L2","L3"),("L2_09_sgemm_lora_b","L2","L3"),
 ("L2_19_fused_sigmoid_gating_delta_rule_update","L2","L3"),
 ("L2_20_causal_conv1d_update","L2","L3"),("L2_21_causal_conv1d_fwd","L2","L3"),
 ("L2_30_fused_moe_router_tensorcore","L2","L3"),
 ("L2_33_fwd_grouped_kernel_stage1_rope","L2","L3"),
 ("L2_41_selective_state_update","L2","L3"),
 ("L2_16_fused_rmsnorm","L2","L1"),("L2_18_fused_gdn_gating","L2","L1"),
 ("L2_27_deepep_permute_triton","L2","L1"),("L2_28_deepep_post_reorder_triton","L2","L1"),
 ("L2_32_experts_combine","L2","L1"),
 ("L2_35_pre_reorder_triton_kernel_for_cutlass_moe","L2","L1"),
 ("L2_36_post_reorder_triton_kernel_for_cutlass_moe","L2","L1"),
 ("L2_38_concat_and_cast_mha_k","L2","L1"),("L2_40_gemma_qkv_rmsnorm","L2","L1"),
 ("L3_07_correct_attn_cp_out","L3","L2"),("L3_12_moe_mmk","L3","L2"),
 ("L3_21_solve_tril_16x16","L3","L2"),("L3_22_chunk_local_cumsum_vector","L3","L2"),
]
LINE = re.compile(r"^level:\s*(L[123])\s*$")

def meta_path(tid): return ROOT / f"level{tid[1]}" / tid / "meta.yaml"

def main():
    assert len(CHANGES) == 38
    for tid, cur, new in CHANGES:
        p = meta_path(tid)
        assert p.is_file(), f"missing {p}"
        lines = p.read_text().splitlines(keepends=True)
        hits = [i for i, l in enumerate(lines) if LINE.match(l.rstrip("\n"))]
        assert len(hits) == 1, f"{p}: {len(hits)} level lines"
        i = hits[0]
        assert LINE.match(lines[i].rstrip("\n")).group(1) == cur, f"{p}: not {cur}"
        eol = "\n" if lines[i].endswith("\n") else ""
        lines[i] = f"level: {new}{eol}"
        p.write_text("".join(lines))
        print(f"{tid}: {cur} -> {new}")
    print("OK 38/38")

if __name__ == "__main__":
    main()
```

脚本内置 cur 断言 → 若某题当前 level 与预期不符(被人改过)会**fail-closed**,不静默写错。`assert len(hits)==1` 防止 `meta.yaml` 里有第二处 `level:`(如嵌套 block)被误伤。`idx` 字段不动(审计:`idx` 仅在 `validate_tasks.py:71` required-key 列表里出现,无唯一性 / 无 level↔idx 不变量)。

### 3b. `docs/kernel-select.md`(手改,**禁止 phase_c_docs 正常模式**)

> 记忆 [phase_c_docs clobbers shapes]:`phase_c_docs.py` 正常模式会 clobber `shapes.yaml:973` 且从不产出本文档;`--refresh-mirror-only` 只重写 contract.md 的 param_constraints mirror(不含 level)→ 对本任务无用。**全部手改**。

| 位置 | 现状(已 stale) | 改为 |
|------|----------------|------|
| `:5` §1 headline | `100 个 kernel … L1 40 · L2 38 · L3 22` | `101 个 kernel … L1 36 · L2 35 · L3 30` |
| `:9-11` §1 侧栏 | `SGLang 39` / `VV 65 · CV 35` / `medium 70 · high 24 · low 6` | `SGLang 40` / `VV 66 · CV 35` / `medium 71 · high 24 · low 6`(纯 100→101 旧漂移,顺手对齐;level 改动**不影响** repo/dtype_class/difficulty) |
| `:21-29` §3 rubric | 口语化 L1/L2/L3 定义 | 替换为本计划 §1.4 的可判定 rubric |
| `:79-226` §7 逐题表 | 行按 `### L1/L2/L3` 分组,task_id 前缀编码旧 level;另缺行/废行 | (1) 把 38 行移到**新** level header 下(**保留旧 task_id 前缀**)或加显式 `level` 列;(2) 删废行 `L2_23` / `L2_24`,补 `L2_39_scaled_mm` / `L2_40_gemma_qkv_rmsnorm` / `L2_41_selective_state_update`;(3) 加注:**"task_id 前缀是历史选型编号;权威 level 以 meta `level` 字段为准,可能与前缀及 `tasks/levelN/` 目录不一致。"** |

### 3c. 其他文档

| 文件:行 | 处置 |
|---------|------|
| `README.md:88-90` | level 表 `L1 40 / L2 38 / L3 23`(adversarial 抓出,初版漏) → `L1 36 / L2 35 / L3 30`(meta 字段口径);右列描述顺手对齐新 rubric |
| `README.md:140` | `101 task directories (L1=40, L2=38, L3=23)` → 用 level **字段**分布 `L1=36 / L2=35 / L3=30`,并显式标注"这是 meta level 字段;目录布局 level1=40/level2=39/level3=22 单列说明(field-vs-dir 有意发散)" |
| `README.md:129` | `Selection rationale (100 tasks)` → `(101 tasks)` |
| `docs/benchmark_select.md:1,5,569` | **不重写 v3.1 正文**(历史/dated,非代码消费)。顶部加一行 2026-06-07 errata,指向 `docs/kernel-select.md` 为当前权威分布。**L3_07 口径(adversarial 抓出)**:benchmark_select:5 把 `L3_07_correct_attn_cp_out` 标为 "appendix"(`scored 100 + appendix 1`),但当前 `meta.level=L3` 且在 scored 101 内(kernel-select §7 + `dataset` 扫描均含),本计划按 scored **L3→L2** 处理;errata 注明 appendix 是 v3.1 历史身份,当前为 scored,权威以 meta 为准 |
| `docs/DESIGN.md / DESIGN-V2.md / ARCHITECTURE-DATAFLOW-V1.md` "分三级" | **false positive**(指 `oracle_simplified` 保真度档,非 L1/L2/L3)→ 不改 |
| `SCORE.md` | 不打印 level 分布,按稳定 task_id 引用(`L2_33` 等前缀不变)→ 不改 |
| `docs/reviews/2026-05-19-*.md` / `docs/plans/2026-05-17-*.md` | dated 审计快照,反映其当日 → 不改(可选加 "superseded" 一行) |

> 注:`docs/` 在本开发机 root-owned 只读(CLAUDE.md)。文档实际落盘需在有写权限处(或由有权账户)执行;本计划负责给出确切 diff 内容。

### 3d. `cli.py:257` init-solution 路径提示(决策点)

`cli.py:252-257` 把 `tasks/level{task.level}/<task_id>/contract.md` 写进生成的 `solution.py` docstring。对 22 个跨档题(如 `L2_16` field 现为 `L1`)会打印**错的路径提示** `tasks/level1/L2_16.../`。审计判定:cosmetic、dev-CLI scaffold 字符串、从不被打开、无 scoring/golden/daemon 影响,strategy A **有意接受**。

**推荐(可选,低优先)**:改 1 行 —— 把 dir 来源从 `task.level` 换成 **task_id 前缀**(`task.task_id[1]`),则提示在 A/B 任何策略下都指向真实在盘目录,且与 field-vs-dir 哲学正交。(口径更正,adversarial 指出:field→dir 路径**不止此一处** —— `generate_task_scaffolds.py:482` 也从 level 字段建目录,但它仅在**重新生成 scaffold** 时触发、不碰现存 101 题,故归 §3f SAFE;`cli.py:257` 是唯一**运行期 dev-CLI** 会输出错路径提示的点。)
**前置 guard**:改前先 `grep -rn "init-solution\|level{task.level}\|See tasks/level" tests/`,确认无单测断言该 hint 字符串(审计未提及有,`test_cli.py` 只覆盖 `run-suite --level` 与非法 level)。若无断言 → 安全改;若有 → 留给 codex 仲裁。
**默认立场**:倾向修(near-zero cost、移除已知 footgun、不引入 GPU);但因 A 明确接受,最终在 §6 codex gate 一并裁决,不擅自越过已批准决策。

### 3e. 需要更新的测试

**无。** 审计确认:无任何单测断言某真实题的具体 level 值,也无任何 per-level 分布/计数断言(全仓无 `40/39/22` 硬编码)。三类 level-touching 测试全部无害:
- (a) synthetic tmp_path/in-memory fixture 自带 hardcoded level(`test_validate_tasks_completeness` / `release_scope` / `param_constraints` synthetic / `file_blocks` / `fix_orchestrator` / `phase_b_golden_capture` / `prompt_*` / `review_fixes` 等)—— 与真语料解耦;
- (b) 经 `levelN/` **目录路径**访问改动题(`test_kernel_resolver` L1_20/L1_17/L3_07、`test_validate_tasks_param_constraints` L1_16、`test_dataset:69`、`test_wrapper_contract`、`test_corrected_variant`)—— 目录未变,通过;
- (c) 成员/计数断言全部满足:`test_dataset.py:36` `task.level in {1,2,3}`(L1_16→L2→2 仍在集)、`test_cli.py:85` `n_tasks>0`(level-1 字段口径变 36,仍 >0)、`test_phase_c_generators_wa5.py:432` 目录计数 `==101`(目录未增删)。

→ 阶段1 不动任何 `tests/`。若阶段2 出现单测红,即审计盲点信号,停下走 §6 codex 复盘,不就地打补丁。

### 3f. SAFE 模块清单(无需改动,附审计证据 —— 让用户看见覆盖面)

| 模块 | 读 field/dir/neither | 证据(file:line) | 为何 field-only 安全 |
|------|----------------------|------------------|----------------------|
| dataset loader | dir | `dataset.py:105-108,196-200` | scan `level1/2/3` + task_id 定位;目录+task_id 不变 |
| dataset 字段读取 | field | `dataset.py:129-134,181` | 唯一生产 field reader,只产 `Task.level` int,下游仅 cli.py;发散正是 A 接受的 |
| golden_store | dir | `golden_store.py:86-91 + locate()` | 按 task_id scan,golden-safe,无 field 读 |
| scoring | neither | `scoring.py` 全文 | per-shape e→s_t geomean,**无 per-level 分桶/加权**,level-agnostic |
| eval | neither | `eval.py` 全文 | 仅 docstring 提 level{N};无 field 读、无从 field 建路径 |
| eligibility | neither | `eligibility.py` | release-ready 独立于 level |
| manifest / golden_manifest | neither | `manifest.py` + `scripts/golden_manifest.py` | 无 field 读、无 field 建路径 |
| validate_tasks 枚举门 | field | `validate_tasks.py:101,1173-1174` | 38 新值全 ∈ `{L1,L2,L3,...}`;**无 level↔dir、无 level↔task_id、无 idx 唯一性门** |
| validate_tasks contract 门 | neither | `validate_tasks.py:560-568` | short_id 取自 task_id 前两段(`L2_16`),非 level |
| validate_tasks 主扫描 | both | `validate_tasks.py:1488-1497` | walk 目录但 `validate_task` **从不比对 meta.level 与目录名** |
| provenance gate | dir | `provenance_release_planning_gate.py:104` | dir scan / task_id |
| provenance census | dir | `source_provenance_census.py:247` | dir scan |
| freeze_source | dir | `freeze_source.py:118,131` | dir scan / frozen bytes |
| apply oracle verdict (C2) | dir | `apply_oracle_equivalence_verdict.py:79` | dir scan |
| phase_a_prime (S2) | dir | `phase_a_prime.py:691` | dir scan(其余 `.level` 命中是 AST ImportFrom 深度,无关) |
| phase_b golden capture | neither | `scripts/phase_b_golden_capture.py` | 经 golden_store dir 解析 |
| oracle_pyref_precompute | dir | `oracle_pyref_precompute.py:273,768` | `audit.level = task_dir.parent.name`(目录名,恒等于 dir) |
| daemon oracle-equiv runner | dir | `scripts/loop/oracle_equiv_one.py:79` | daemon 侧 dir scan |
| daemon gpu-rank runner | dir | `scripts/loop/gpu_rank_one.py:104` | daemon 侧 dir scan;排名独立于 field |
| daemon_a 任务校验 | neither | `daemon_a.py:56-70` `_TASK_ID_RE=^L[1-3]_[0-9]{1,3}_...` | 校验**未变的 task_id**,非 field;38 题全匹配 |
| orchestrator meta 同步 | dir | `orchestrator.py:111,407` | glob `tasks/level*/<task>/meta.yaml`;编辑后的 meta 被正确同步 |
| cert-stage shell | dir | `scripts/gpu_pipeline/oracle_equivalence_verify.sh:110-151` | `for level in level1 level2 level3` dir scan |
| per_task_diagnostic | neither | `per_task_diagnostic.py:71-73` | 从 task_id 前缀建 dir,非 field |
| generate_task_scaffolds | field | `generate_task_scaffolds.py:482` | 仅影响**新** scaffold 生成,不碰现存 101 题 |
| `cli.py:435-441` run-suite --level | field | — | A 下语义**正确**:field 即权威难度桶,`--level L2` 现含升级题 |

### 3g. CI / run-suite --level 的 functional 影响(adversarial 抓出,初版遗漏)

`cli.py:435-441` run-suite `--level N` 按 **level 字段**过滤题池,这是唯一**会真正改变行为**的 field 消费点(**非 cosmetic**):重标后 `--level 1` 返回的题集从 **40 → 36**(13 个 former-L1 升 L2 离开、9 个 former-L2 降 L1 进入);`--level 2/3` 同理换集。

**CI 消费点(必须处理)**:
- `.github/workflows/ci.yml:182`:`tab run-suite --level 1 --solutions-root /workspace/solutions` —— CI 用 `--level 1` 选题跑候选 solution 评测,重标后评测**新的 36 题 level-1 集**。
- `.github/workflows/verified_runner.yml:75,88`:`tab run-suite`(需核对是否带 `--level`;带则同步核对)。

**处置**:
1. **确认 CI 语义意图**:`--level 1` 是想"按权威难度桶跑 L1 题"(则字段口径**正确**,自动跟随新集,CI 无需改)还是"跑固定 40 题集"(则需改成显式题列表 / `--level all`)。**倾向前者**(run-suite --level 本就是难度桶语义),但属 CI 行为变更,需 codex gate 显式确认 + 用户知情。
2. **solution 覆盖**:9 个新进 level-1 的 former-L2 题(`L2_16/18/27/28/32/35/36/38/40`)若 `/workspace/solutions` 无对应 solution.py,run-suite 报 missing-solution(`cli.py:442-453`)。这是 run-suite 对**缺候选解**的正常行为(corpus 不含 solution,由 CI/提交方提供),非 level bug —— 但题集换批,CI 的 solutions fixture 需相应覆盖。
3. **验证(并入 Gate 3)**:`tab run-suite --level 1` 改前/改后的**选中 task_id 集** diff,确认 = 预期 {40} → {36},无漏选/多选。

> 这是本计划**唯一的 functional 行为变化**,且是 A 策略"field=权威难度桶"的**预期正确语义**;但因触及 CI,必须显式验证 + codex/用户确认,不可当 cosmetic 略过。

---

## 4. 验证矩阵

### Gate 1 — 字段编辑后(本地,零 GPU)
```bash
# 1) 执行批量编辑
python3 scripts/oneoff/relabel_levels_field_only.py        # 期望: OK 38/38

# 2) YAML 合法性(全 101)
python3 -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('tasks/level*/*/meta.yaml')]; print('yaml ok')"

# 3) level 字段分布精确断言
python3 - <<'PY'
import yaml,glob,collections
c=collections.Counter(yaml.safe_load(open(f))['level'] for f in glob.glob('tasks/level*/*/meta.yaml'))
print(dict(c)); assert c['L1']==36 and c['L2']==35 and c['L3']==30 and sum(c.values())==101, c
print('dist ok 36/35/30')
PY

# 4) field-only 证明: 仅 38 个 meta.yaml, 每个恰 1 增 1 删
git diff --numstat tasks/ | tee /tmp/level_diff.txt
test "$(wc -l < /tmp/level_diff.txt)" -eq 38 && awk '$1!=1||$2!=1{print "BAD:",$0; bad=1} END{exit bad}' /tmp/level_diff.txt && echo "diff ok 38x(1,1)"

# 5) corpus 校验: 错误集与 baseline 相同 (dev 有 18 个 pre-existing golden-missing, 与 level 无关)
python3 scripts/validate_tasks.py 2>&1 | grep ERROR | sort > /tmp/vt_after.txt
echo "validate ERROR count: $(wc -l < /tmp/vt_after.txt) (期望 18, 与 baseline 同)"
diff /tmp/vt_before.txt /tmp/vt_after.txt && echo "validate error-set IDENTICAL to baseline" || echo "REGRESSION: level 改动引入了新错误, 停下走 §6"
python3 scripts/validate_corpus_consistency.py             # 期望: 与 baseline 同 (level-agnostic)
```
> **为何不是 exit 0**(adversarial 实测):dev 上 `validate_tasks.py` 恒 exit 1 / 18 个 `golden/ present but missing safetensors` 错误 —— golden safetensors 封存在 daemon/gpfs、dev 本地 `golden/` 目录无真 bytes,这是 **dev 环境固有限制,与 level 完全无关**。所以验收用"**错误集 vs baseline 完全相同**"(改 level 不增减任何错误),exit 0 仅在有 golden 的 daemon 侧才可达(归入 Gate 3)。`/tmp/vt_before.txt` 由阶段0 baseline 快照抓取(§2)。

| 检查 | 期望 |
|------|------|
| YAML parse | 全 101 通过 |
| level 分布 | `L1=36 / L2=35 / L3=30`,合 101 |
| `git diff --numstat tasks/` | 恰 38 行,每行 `1 1`(只 level 行变) |
| `validate_tasks.py` 错误集 | 与 baseline **完全相同**(18 个 pre-existing golden-missing,不增不减;枚举门全过,无 dir/task_id/idx 交叉门) |

### Gate 2 — dev 单测全量(本地)
```bash
free -g            # 启动前内存检查; <16G 可用则串行 (-p no:xdist)
cd /data1/code/agentic-rl/kernel/triton-ascend-bench && pytest tests/unit_tests/ -q -rs
```
期望:**0 failed / 0 skipped**(`-rs` 显式列出 skip,若有 skip 视为未通过,按 CLAUDE.md "Zero skipped tests" 处理)。阶段0 的 23 个 pending fix 已并入,这里是合并后的全绿确认。

### Gate 3 — 全量 GPU 管线回归(经 orchestrator + mailbox,daemon 侧)
> 拓扑(记忆 [TAB loop topology]):只能 ssh 到跳板机 B;eval daemon 在够不到的 GPU A;**只能投 mailbox job 探活**,绝不在 B 起 daemon。

**前置 sync(记忆 [TAB sync staleness] + [auto sync after push])**:meta 改动后必须 force-touch meta 时间戳 + `rsync --delete` 到 gpfs,否则 `--update` 跳过 daemon-touched meta、stale meta 让 daemon 跑旧档。
```bash
python3 scripts/loop/orchestrator.py sync-code --touch-meta --delete   # 强制把 38 个新 meta 推到 gpfs
```
逐 stage 投全 101 题,与阶段0 baseline 快照 diff:

| Stage | 入口 | 期望 vs baseline |
|-------|------|------------------|
| C1/C2 oracle-equiv | `oracle_equiv_one.py:79`(dir scan)→ resolve verdict | `resolve_all.json` 每题 verdict **bit-identical**(verdict 不带 level key) |
| phase_a_prime / S2 | `phase_a_prime.py:691`(dir scan) | S2 结论 **identical** |
| cost_model_xcheck | asymptotic_hbm_xcheck | 逐题 PASS/FLOOR/BLOCKED **identical** |
| golden | `golden_store.py:86`(dir scan) | golden manifest 逐题 hash **identical** |
| release-gate | `provenance_release_planning_gate.py:104` + `eligibility.py`(无 field) | **GREEN 101 / release-ready 集合 identical** |
| gpu-rank | `gpu_rank_one.py:104`(dir scan)+ do_bench timing | **排名集合与顺序不变**(原始 ns 有 run-to-run 抖动,断言 ranking eligibility/order 稳定,非 ns bit-identical) |

### Gate 3.5 — run-suite --level 题池 + CI(本地,dev 可跑,§3g)
```bash
# level 字段过滤的选中题池 = run-suite --level 的语义; 改后期望 36/35/30
python3 - <<'PY'
from triton_ascend_bench.dataset import iter_all_tasks
import collections
c=collections.Counter(t.level for t in iter_all_tasks() if t.level in (1,2,3))
print("run-suite --level 题池:", dict(c))      # 期望 {1:36, 2:35, 3:30}
assert c[1]==36 and c[2]==35 and c[3]==30, c
print("run-suite pool ok 36/35/30")
PY
```
期望:`--level 1/2/3` 选中集 = 36/35/30,换集符合 §1.3 的 38 项映射。确认 CI(`ci.yml:182 --level 1`)在新 36 题集下行为符合 §3g 决策(语义跟随 / solution 覆盖)。**这是唯一 functional 行为变化的验证点,不可略过。**

**总判据**:除 38 个 `meta.yaml` 的 level 字节外,所有 stage 产物功能性一致;run-suite 题池按 §1.3 预期换集(唯一 functional 变化,预期正确)。任何 stage 出现 verdict/golden/gate 差异(非预期题池换集)= 审计盲点,**停下走 §6**,不继续推进。

---

## 5. 风险与回滚

### 5.1 已知有意发散(审计已定性,非 bug)
- **task_id 前缀 ↔ level 字段 ↔ 目录三方发散**(如 `L2_16` / field `L1` / dir `tasks/level2/`)。后果被审计界定为:① `cli.py:257` init-solution 错路径提示(cosmetic,§3d 决策);② 人读文档需知"field 权威"(§3b §7 已加注)。**无功能后果**。
- **`cli.py:435` run-suite --level** 行为变化:`--level 1` 现返回 36 题(40−13+9)。这是 A 下的**正确**语义(field=权威难度桶),非回归;`test_cli.py:85` 仅断言 `>0`,通过。

### 5.2 可能仍意外的点(列出以便 codex 重点审)
- daemon 侧 gpfs **stale meta**:若 sync 未 `--touch-meta --delete`,daemon 可能读旧 level 字节 —— 但 level 不影响 GPU 语义,即便 stale 也不改 verdict;真风险是"diff 不为零造成误判",已用 §4 Gate3 前置 sync 消解。
- 文档落盘权限:`docs/` 在开发机只读;若由非 root 账户执行 §3b/§3c 编辑会 Permission denied —— 需在有权处落盘。
- §3d 若误改 `cli.py` 触发未预期单测红 —— 已用 grep guard 前置拦截。

### 5.3 回滚
field-only 的最大优点是**回滚成本=git revert**,零 GPU 重跑:
```bash
git revert <level-field-commit>      # 仅还原 38 行 level 字节
# docs / cli.py 若同 commit, revert 一并还原; 建议三者拆 commit 便于精准回滚:
#   commit A: 23 pending unit fix
#   commit B: 38 meta level 字段
#   commit C: docs 对齐 (+ 可选 cli.py:257)
```
拆 commit 使回滚粒度=单一关注点;阶段0/1/2 的归因边界与 commit 边界对齐。

---

## 6. codex 监督 gate(强制,push 前)

按 CLAUDE.md "Mandatory codex review" + "Approval required for risky operations(A1:git push)":

1. **方案签字**:本计划文档先过 codex —— `humanize:ask-codex`,让其读 `/tmp/level_reclass_review.txt`(含:策略 A 定义、38 题映射、§3f SAFE 证据链、§4 验证矩阵、§3d cli.py:257 决策点)。重点请 codex 裁决:① 审计"level 仅 `dataset.py:129` 一处生产读"是否穷尽;② `validate_tasks.py` 确无 level↔dir/idx 隐藏门;③ §3d 修 vs 接受的取舍;④ GPU 全量回归"期望零 delta"的断言是否站得住。
2. **按 CLAUDE.md "Review external feedback critically"** 逐条评审 codex 反馈(confirmed 核证据、likely/speculative 独立判断),修到 **0 FAIL**,再 re-review 闭环。
3. **实现后复审**:阶段1+2 代码/文档落地后,diff(`/tmp/level_diff.txt` + docs diff + 可选 cli.py diff)再过一次 codex,确认"仅 level 字节 + 文档计数"无越界。
4. **push gate**:全部绿(Gate1/2/3 + codex 0-FAIL)后,**展示结果,等用户明确说"提交/推送"才 commit→push**。push 后立即 `orchestrator.py sync-code` 把代码同步到 gpfs(记忆 [auto sync after push]),不让用户去 GPU 端 git pull。

---

### 附:终态自检 checklist
- [ ] 阶段0:`pytest tests/unit_tests/` 0-fail/0-skip;GPU baseline 快照 + dev `validate_tasks` 错误集(`/tmp/vt_before.txt`,18 golden-missing)已留存
- [ ] 阶段1:`relabel_levels_field_only.py` OK 38/38;`git diff --numstat tasks/` = 38×`(1,1)`;dist=36/35/30;`validate_tasks` 错误集 = baseline(18 个,不增减)
- [ ] 阶段1:kernel-select.md §1/§3/§7 + README:88-90/129/140 已手改(未跑 phase_c_docs 正常模式);benchmark_select errata(含 L3_07 appendix 口径)
- [ ] 阶段1:cli.py:257 决策已落(修或接受,有 grep guard 记录)
- [ ] 阶段1:run-suite --level 题池 = 36/35/30(Gate 3.5);CI `ci.yml:182` 行为按 §3g 确认
- [ ] 阶段2:dev 单测全绿;全 101 GPU 管线逐 stage 与 baseline diff = "仅 level 字节"
- [ ] codex 0-FAIL(方案 + 实现两轮);用户确认后方 push

**计划文件建议落盘路径**(有写权限处):`docs/reviews/2026-06-07-level-reclassification-plan.md`(开发机 `docs/` 只读,需由有权账户写入或改放可写目录)。