---
title: "feat: Baseline Accuracy Provenance — Automation Pipeline Implementation"
type: feat
status: active
date: 2026-05-26
deepened: 2026-05-26
origin: docs/brainstorms/2026-05-16-baseline-accuracy-provenance-requirements.md
supersedes:
  - docs/triage_backlog/2026-05-26-automation-pipeline-v2-plan.md
  - docs/plans/2026-05-26-001-feat-baseline-accuracy-provenance-impl-plan.v1-draft.md
review-rounds:
  - 4 specialist agents pass A (architecture / data-integrity / security / repo-patterns)
  - codex pass A (8 additional findings) → 17 MUST-FIX applied to V2
  - 5 specialist agents pass B (coherence / feasibility / adversarial / security-lens / scope-guardian) + codex pass B (5 additional findings) → 14 additional MUST-FIX applied to V3
  - codex rounds 4-19 iterative cleanup → V8.12 ALL PASS
  - V9 architectural shift (user-decided 2026-05-26):corpus oracle_pytorch.py + R10 (3-段 dev/GPU/sync);RID rename;`oracle_pytorch` truth-source path (vs V8 author-test path)
---

# Baseline Accuracy Provenance — Automation Pipeline Implementation

> **V9 architectural shift (user-decided 2026-05-26)**:
> - **Source ① 改用 corpus 现有 `tasks/<id>/oracle_pytorch.py` + GPU equivalence check**(deprecate 原 V8 plan 的 M1.1 上游 author-test 抽取);若 oracle ↔ kernel 等价校验通过,oracle_pytorch.py 升 truth source。
> - **R10 GPU equivalence 拆 3 段:dev 自动化预算 + GPU operator 手动跑 + dev 自动同步**(v1 不实施 SSH 编排完全自动化;复用既有 Tier 1 半自动模式)。
> - **RID-prefixed 命名全 deprecated**(R10/R12/R13c 等只在 plan section header / context 中保留作 traceability;**不进** scripts / yaml fields / code identifiers)。


## Overview

把 TritonAscendBench 的 golden 真值来源从 AI 生成的 `oracle_pytorch.py` 切换到 **R1 双路径**(上游作者 unit test ref + 容差 / 上游 GPU kernel 离线跑),做成 **0-touch automation pipeline**:新 op 进来,pipeline 自动 vendor + 抽取 + 验证 + 输出 verdict.

这是一次 **benchmark truth-source contract 重构**,不是"加 provenance 字段"。改造的核心是抽出 3 个 first-class authority:
- **`GoldenStore`**:统一 public/sealed golden 路径解析、完整性校验、可读性 predicate(替代当前 4 处 hard-coded `tasks/<id>/golden/`)
- **`ToleranceAuthority`**:统一 upstream `explicit` / `default` 二态(**v1 scope;`migration_drift_budget` v2 follow-up**),跟 `tolerance.compare` 兼容
- **`ProvenanceEligibility`**:**新** predicate(不复用 `is_score_eligible`,避免循环依赖)

100 task corpus 是 **regression test suite**,success metric = `release_ready_with_provenance count ≥ 60 ∧ ≥ 8 category(含稀有类 coverage)`(用 release-ready 不是 score-eligible denominator)。

## Problem Frame

> *Carries from origin doc §Problem Frame.*

TritonAscendBench 当前正确性真值(`oracle_pytorch.py::run`)是 **AI/脚本按算子文档生成**,无对上游 kernel 的机器等价验证,无人工专家 review;`formula_exact_manual` 占 73%(74/101)。**真值本身是未经证明的 AI 产物 = 整个基准最根本的 soundness 缺口**。

更可信来源:`infer/vllm @ 18f6bf5a` + `infer/sglang @ 409d350f`(`/data1/code/agentic-rl/kernel/infer/`,bench repo 兄弟目录,正确 commit,root-owned)。**当前流水线无作者参考抽取逻辑** — R4 是净新增抽取阶段。

`meta.yaml.upstream.commit_sha` **95/100 已记录且与 HEAD 匹配**,残余 3 类(`snapshot-no-git` 5 题 / L3_08 不可达 / kernel-SHA ≠ test-SHA 未实际比对)由 R12 处理。

## Requirements Trace

> *All 20 carried from origin doc (see origin: `docs/brainstorms/2026-05-16-baseline-accuracy-provenance-requirements.md`).*

**Golden 来源与优先级**:R1(双路径)/ R2(① > ② priority,census 后可推翻)/ R3(AI oracle 降级为诊断旗标)/ R10(source ① 等价校验,GPU 上)/ R10b(成本不可接受时按算子族抽样,未覆盖标 R13b)

**容差来源与门禁**:R7(`golden_tolerance_source` enum)/ R7c(`upstream_test_default` 是 **soft gate** — task 进 R13b provisional 单独披露,不直接 hard-block score-eligible — V3 修正,避免 ~50% upstream default tolerance 触发 R7c hard-block 让 projected_eligible 腰斩)/ R11(`migration_drift_budget` 是 v1 三态之一中**仅作记录 + R13b provisional 标签**;non-AI derivation **方法学 v2 follow-up,v1 不要求实施完整方法学**)

**安全与完整性**(threat model T1-T4):
- R14(user-decided 完全 withhold output;**hidden inputs(knobs)在 submitter shapes.yaml documented Limitation**)
- R15(评测期 hash 完整性;**R18 cover 范围扩到 4 artifacts 共一签**:`tasks/<id>/golden/manifest.json` + `meta.yaml.provenance.golden` canonical-json + `triton_ascend_bench/sanity_bounds.yaml` + `tasks/<id>/provenance_commit.json`,见 V3 修正)
- **R16 minimum-viable 修正**(V3):sanity bound 来源**不**从 `oracle_pytorch.run` 推(self-attesting collapse),改 **per-op-family CODEOWNERS-reviewed bound table**(`triton_ascend_bench/sanity_bounds.yaml`,changes 需 separate reviewer)+ M3.0 显式声明 R16 v1 **scope = gross-corruption defense**(NaN / Inf / shape / all-zero),**不是 T2 supply-chain full defense**(诚实 downgrade scope claim)
- R17(public/sealed + 不可重构论证 + 高 risk task **exclude from leaderboard scoring 完全**,不仅 R13b provisional 标签;R17 algorithm M0a 显式定义:`risk_score = closed_form_math_weight × public_input_knob_visibility × tolerance_breadth_factor`,thresholds: low<3 / medium 3-6 / high>6)
- R18(OOB trust root,M3.2 ship — cover 范围 V3+V8 扩到 **`tasks/<id>/golden/manifest.json` + meta.yaml.provenance.golden canonical-json + sanity_bounds.yaml + tasks/<id>/provenance_commit.json (R12 commit-chain proof, includes meta_upstream_commit_sha)**;single ed25519 v1 但 key 存储为 JSON file `triton_ascend_bench/trust_roots.json` 支持多 active key + valid_from/until 字段,**留 rotation 路径**)
- **R19 minimum-viable 修正**(V3):M2.0 v1 **只 ship Layer A safetensors/torch.load deny + sealed_golden file mode 0700 + fail-closed if non-root**(non-root daemon 无 CAP_SETUID → refuse to start,不 silent 退化);**uid switching + MADV_DONTFORK 完整 implementation defer security follow-up plan**(显式 acknowledged as scope limit)
- R20(确定性冻结,v1 仅 deterministic;band-mode v2 follow-up)
- **R12(commit-SHA 对应,V3 加 explicit unit M2.0.5)**:实施 `meta.upstream.commit_sha == provenance.golden.test_capture_sha` 比对(both 40-hex git commit SHA;不用 kernel_py_sha256 file content hash) + 不符 → `oracle_kernel_equivalence = NEEDS_MANUAL`,**不是 deferred to research**

**流水线改造**:R4(净新增抽取阶段,**two-phase extractor**:module-local + registry + tests)/ R5(provider-侧冻结)/ R6(eval 改 golden-load)/ R8(provenance schema,**嵌套 `provenance.golden:` 避免与既有 `{added_in, authors, last_modified}` collision**)/ R9(score-eligible 复合条件)/ **R13 V3 修正**:R13 PROCEED + R17 高 risk task 解耦(R17 high → exclude leaderboard,不进 R13b provisional 也不进 PROCEED 分母);**rare-class set 跟 R17 high-risk set 不重叠** — rare class 重定义为 category 内 task 数 ≤ 5 的 family(`convolution / spec_decode / normalization`),非 plan v2 的 R17-high-risk 集 `elementwise / norm / gemm / sampling`(后者已经 R17 exclude,不入 rare 候选)/ R13b(单独披露层)/ R13c(go/no-go 决策函数,**`projected_eligible` 排除 R17 high-risk + 排除 R11 v1 不强制 task**)

## Scope Boundaries

- 评测期 / submitter 零 GPU(GPU 仅 provider 离线)
- 不重写 `tolerance.compare` / `scoring` 计分公式;只换"真值来源" + 容差注入接缝
- 不涉及性能 baseline
- 不人工逐题重写 74 题 oracle 数学
- 本 plan **不改** `kernel.py` / `infer/` / `scripts/resolve_upstream_entrypoints.py`(评估器自身)
- 100 task 不剔分母(SETTLED)
- V2 plan(`docs/triage_backlog/2026-05-26-automation-pipeline-v2-plan.md`)完全 deprecated
- **v1 source ②:仅接受 deterministic kernel**;band-mode (R20 非确定) 单独 follow-up plan
- **hidden shape inputs(knob values)仍在 submitter-visible `shapes.yaml`** — 只 hidden *outputs* sealed;input knobs 不在 R14 scope 内(documented Limitation)

## Context & Research

### Relevant Code and Patterns

| 文件 | 行号 / 函数 | 角色 |
|---|---|---|
| `scripts/extract_kernel.py` | `KernelExtractor` (36) — 单 `ast.NodeVisitor` 类,**M1.1 加 sibling `AuthorRefExtractor`** | R4 起点 |
| `scripts/phase_b_golden_capture.py` | `_run_capture_child` (727) signature `(task, shape_ids, golden_dir, timeout_s, *, entrypoint_source, upstream_callable)` — **不接收 inputs/outputs/sha** | source ② plumbing 已就绪 |
| `scripts/phase_b_golden_capture.py` | `_select_tasks` (620) / line 671 / 686 用 `is_score_eligible` 选 capture 范围 | **循环依赖证据** — V2 必须新建 predicate 而非改写 `is_score_eligible` |
| `scripts/phase_b_golden_capture.py` | `capture_task(dry_run=True)` (838) | **仅 shape enumeration,不真 invoke** — M0a.1 census 不能复用此当 source ② 真证据 |
| `scripts/resolve_upstream_entrypoints.py` | `oracle_consistency_check` (1182), aggregation (1488-1503) — **primary-PASS-wins 已实现**;不动 |
| `triton_ascend_bench/eligibility.py` | `is_score_eligible` (41), `release_status_pass` (255), `golden_complete` (192), `golden_artifact_usable` (142) — **不改语义**;V2 加新 predicate |
| `triton_ascend_bench/eval.py` | `_resolve_golden` (759), `_UPSTREAM_ORACLE_TIER` (756) — **已有 verified-mode 走 offline golden**;M4.1 只补 SHA verify chain |
| `triton_ascend_bench/tolerance.py` | `ToleranceClass` (24), `compare` (128) — **新加 `ToleranceAuthority`** sibling abstraction;不重写 `compare` |
| `scripts/validate_tasks.py` | `REQUIRED_META_TOP_LEVEL` (70-92), `run_release_gate` (1285) | R8 schema 扩展入口 |
| `scripts/automation/lib/file_blocks.py` | `validate_meta_content` (493), `_validate_defer_meta_fields` (571), `validate_patch` (676) | 5-layer validator(不是 "Layer 5" 单一函数) |
| `scripts/automation/lib/atomic.py` | `atomic_write_yaml/json/text` + `_atomic_replace` (45,private) | 多 typed writer;**M3.2 binary safetensors 需 promote `_atomic_replace` 公开** |
| `triton_ascend_bench/dataset.py` | `TaskDef` (40) — 加 `provenance_golden: dict | None` field(从 yaml `provenance.golden` 嵌套加载) |
| `triton_ascend_bench/static_checker.py` | `_STRING_DENY_PATTERNS` (300-307) **作用于 submitter source text,不是 tasks/ paths** — **M3.2 不需要 carve-out**(misframed) |
| `triton_ascend_bench/process_isolation.py` | 51 KB / `sanitised_shapes_view` (682-699) — **R19 边界** + safetensors 在 candidate Layer A 当前未 deny |
| `triton_ascend_bench/cli.py` | `add_subparsers` (34), `_cmd_<name>` pattern (line 36-94) | **新 unit M-CLI** 加 `tab census` / `tab freeze-golden` / `tab verify-source-1` |
| `tasks/<id>/meta.yaml` | **既有 `provenance: {added_in, authors, last_modified}` 块 101/101** — V2 嵌套 `provenance.golden:`,不破既有字段 |
| `tasks/<id>/shapes.yaml` | `public_shapes` / `hidden_shapes` (line 81+) — hidden_shapes **knobs 在 submitter tree**;R14 完全 withhold 只对 *outputs* |
| `/data1/code/agentic-rl/kernel/infer/{vllm,sglang}/.git`(**bench repo 兄弟,不在 repo 内**)| HEAD `18f6bf5a` / `409d350f`,**dev-owned(stat `dev:dev 755`,not root** — V3 修 V2 误判)| 所有 unit 强制用 `INFER_ROOT = REPO_ROOT.parent / "infer"`(`phase_a_freeze.py:43` 既有约定)或绝对路径,不用 repo-relative `infer/`(V3 修路径混淆)|
| `tests/unit_tests/` | 现有目录;**`tests/integration_tests/` 不存在** — GPU-mark test 改用 `tests/unit_tests/` + `@pytest.mark.gpu` |
| `scripts/automation/state_machine.py` | `CANONICAL_STATES` (41), `BACKLOG_PENDING_GOLDEN` | 既有 17-state machine,V2 fan-out 必须接 |

### Institutional Learnings

- **SETTLED**:≥80% PyTorch oracle / ≤20% provider GPU 预跑(`docs/reviews/2026-05-19-design-v2-golden-policy-review.md`)
- **既有 17-state LLM repair orchestrator**(`docs/architecture/automation-llm-repair-architecture.md`)+ 5-layer validator + byte-exact rsync rollback — V2 复用
- **三机 deployment**(`docs/plans/2026-05-23-001` 142KB + `2026-05-23-002` 245KB orchestrator)已落 M0-M4,V2 在此 framework 内扩展
- **3-task fix-verification 实测**(`docs/triage_backlog/2026-05-26-3-task-fix-verification.md`):
  - L1_18 libdevice 在 `TRITON_INTERPRET=1` 返 None → **source ② 路径强制走 GPU**(R10 + R5)
  - L2_15 severance subprocess `sglang.srt.utils` import leak — **R16 隔离要严格 clean subprocess**
  - L1_07 bf16/fp8 `vector_reduce` 容差太紧 → 真容差来自 source ① `upstream_test_explicit`(R7)

### External References

- Triton 3.x interpreter limitations(libdevice ops 返 None,需 real GPU)— 影响 R10 必须真在 GPU 跑
- safetensors(IPC 格式)— `_pa_subproc_runner.py` 已用,R5 golden 工件复用

## Key Technical Decisions

| 决策 | 理由 |
|---|---|
| **GoldenStore 单一新 module + tolerance.for_task / eligibility.is_provenance_eligible 函数(V3 简化)**| pass B SG-1/SG-6 confirm:GoldenStore 是真新 module(5 hard-coded path callsites + sealed/public + signing),但 ToleranceAuthority + ProvenanceEligibility 是单函数 + 单 predicate,**应 fold 进现有 tolerance.py / eligibility.py 而非新 module**;减少 3 module → 1 module,~150 LOC + 2 test file 节省 |
| **`compute_score_eligible` V3 加 provenance gate**(用户 ADV-3 + codex R2#2 confirm)| V2 改 `_cmd_analyze` 漏 `compute_score_eligible:cli.py:851`,denominator 不 gate provenance → 加 helper `compute_release_ready_eligible(tasks)` = `compute_score_eligible AND is_provenance_eligible`;`compute_score_eligible` 自身**仍不动**(`phase_b_capture._select_tasks` 仍用) |
| **`is_score_eligible()` 语义不变**,V2 新建 `is_provenance_eligible()` / `release_ready_with_provenance()` | codex C2 实证:`phase_b_golden_capture._select_tasks` 用 `is_score_eligible` 选 capture 范围;改写它会让 source ② task 进不了 capture(循环依赖) |
| **R13 升级 50/6 → 60/8 + 稀有类 coverage 双门**(原 user-decided 50/6 太弱)+ **V3:R17 high-risk task 跟 rare-class 解耦**(rare class = task 数 ≤ 5 的 `convolution / spec_decode / normalization`,非 R17 high-risk 的 `elementwise / norm / gemm / sampling`;后者直接 leaderboard-exclude,不入 rare 候选)| codex C6 + adversarial ADV-1:rare-class 全是 R17-high 时 PROCEED 数学不可达;V3 修通过解耦 |
| **R16 sanity bound 不从 oracle_pytorch.run 推**(V3 修):per-op-family CODEOWNERS-reviewed `triton_ascend_bench/sanity_bounds.yaml`,changes 需 separate reviewer;M3.0 R16 scope **诚实 downgrade 为 gross-corruption defense**(NaN/Inf/shape/all-zero),不再 claim "T2 supply-chain 唯一防线" | security SEC-1 + adversarial ADV-5:从同 commit oracle 推 bound = T2 攻击 self-attesting bypass |
| **R17 algorithm explicit + leaderboard-exclude**(V3 修):M0a.1 emit `risk_score = closed_form_math_weight × public_input_knob_visibility × tolerance_breadth_factor`;high (>6) → **leaderboard exclude 完全**,不进 PROCEED / 不进 R13b provisional;medium / low 正常 | security SEC-4 + adversarial ADV-1:hand-wavy + R13b 仅 disclosure 非 control |
| **R18 cover 范围扩 + key 管理 minimum-viable**(V3 修):R18 sign 包含 `tasks/<id>/golden/manifest.json` + meta.yaml.provenance.golden canonical-json + sanity_bounds.yaml + **`tasks/<id>/provenance_commit.json`(R12 commit-chain proof,含 meta_upstream_commit_sha)**;key 存 `$TAB_GPFS_ROOT/keys/ed25519_v1.priv` 0400 owner=provider;**trust root 用 `triton_ascend_bench/trust_roots.json` 多 active key + valid_from/until 支持 rotation** | security SEC-2/SEC-5 + codex R2#4:V2 只 cover golden manifest,R12 + meta 可篡;public key 常量 in `__init__.py` foreclose rotation |
| **R19 minimum-viable + fail-closed 非 root**(V3 修):M2.0 v1 仅 ship Layer A safetensors/torch.load deny + sealed_golden file mode 0700 + **non-root daemon refuse to start** if no CAP_SETUID(fail-closed, **不 silent 退化**);uid switching + MADV_DONTFORK 完整实施 **defer security follow-up plan** | feasibility F1 + security SEC-3 + adversarial ADV-3:`subprocess.Popen(user=)` 需 CAP_SETUID,dev 非 root 必 silently fail |
| **R10 + R7c soft mode**(V3 修):`upstream_test_default` → R13b provisional 单独披露,**不 hard-block score-eligible**;R7c hard-block 仅当 `migration_drift_budget` 缺 evidence | adversarial ADV-6 + codex R2#1 互动:~50% upstream test default → R7c hard-block → projected_eligible 腰斩 |
| **R12 加 unit M2.0.5**(V3 修):meta.upstream.commit_sha == provenance.golden.test_capture_sha equality assertion + 不符 → `oracle_kernel_equivalence = NEEDS_MANUAL`,不再 deferred | security SEC-2 + codex R2#3:V2 emit field 但无 compare-and-gate unit |
| **R11 V3 一致化**:plan 一致表述 v1 仅 `explicit / default` 二态;**`migration_drift_budget` 推导方法学 v2 follow-up,v1 不实施**;test scenario 修去掉 "R11-landed" 假设 | coherence C1/C6 + codex R2#1:V2 line 46 vs line 922 自相矛盾 |
| **`op_family` → `category` 全局统一**(V3,deprecated alias `op_family` is removed across the document):`meta.yaml.category` is canonical source of truth;census output schema field name MUST be `category` (legacy `op_family` was rejected as plan-invented term per codex C5) | coherence C4 (initial V3 sweep): plan 决定用 category;V7 sed pass 已 0 stale `op_family` remaining |
| **family taxonomy 用 `meta.yaml.category` 字段**(repo 已有,12 enum)| codex C5:plan 之前的"category"不存在;统一用 category;先做一次性 `norm` vs `normalization` 作两 disjoint category 保持(不 merge) |
| **`provenance.golden:` 嵌套字段,不顶层 collision** | repo T1 + 数据 H4:101 task 已有 `provenance: {added_in, authors, last_modified}`,V2 nest `provenance.golden:` 保留既有 |
| **R18 提前 ship 跟 R15 同 phase**(原 V1 把 R18 deferred,导致 R15 self-attesting placebo)| 数据 C1 + 安全 C3 + codex 间接:R15 hash check without R18 = 自洽检查不是完整性门 |
| **R16 sanity gate 入 M3.2**(原 V1 把 R16 在 M5 placeholder) | 安全 C5:T2 防 supply-chain-poisoning 的唯一防线 |
| **R19 提前 M2 前** — 加 safetensors / torch.load 入 candidate Layer A deny + sealed_golden 0700 owner uid + verified runner 独立 uid | 安全 C4:safetensors 当前在 candidate 可读 → 候选可 `safetensors.torch.load_file(sealed_path)` |
| **R17 reconstructability 新加 M0a census 第 6 维**(per-task R17 risk 评分) | 安全 C2:73% formula_exact_manual 可被重构;高 risk task **leaderboard-exclude 完全**,不进 R13b provisional 也不进 PROCEED 分母 |
| **M0 拆 M0a 静态 + M0b GPU spike** | codex C4:`capture_task(dry_run=True)` 不真 invoke;source ② 真证据要分层 GPU spike(8-12 task) |
| **R11 method 前置成 M0b 子单元** | codex C3:`projected_eligible` 不能在 R11 未落地时算 source ② 入分母 |
| **M1.1 two-phase extractor**(module-local + registry → tests fallback)| codex C7:ref 常在 kernel 自身模块/registry 间接(`impls["native"]`),只搜 tests 严重低估 |
| **R10 GPU equivalence 用 upstream tolerance**(从 M1.1 抽的 `atol/rtol`)而非 `task.tolerance_class` | codex C8:R10 测的是"author-ref 是否在上游背书容差内 vs kernel",必须用 upstream tolerance |
| **v1 仅接受 deterministic source ②**,band-mode 单独 plan | codex C9:`tolerance.compare` 现仅 pairwise tensor compare;band envelope schema + comparator + manifest 整链路 v1 不做 |
| **`safe.directory` master-side 一次配 + frozen HEAD reference**(census/extract 跨 batch 用同一 HEAD)| 架构 C1 + codex 暗合:race-free fingerprint;subagent 不 git pull |
| **`upstream_closure_fingerprint` schema 显式定义**:`sha256({infer_repo: HEAD-at-batch-start, kernel_py_sha256, test_file_path, test_capture_sha})` | 架构 C3:cache invalidation 设计 |
| **`tab census` 等 CLI 加新 unit M-CLI** | codex C10:success metrics 承诺,implementation units 必须覆盖 |
| **Subagent fan-out 接既有 17-state machine** | 架构 cross-cut:用 `BACKLOG_PENDING_GOLDEN` 作 source ① fail → ② fallback 显式 state |
| **不动 `static_checker.py \bgolden\b` denylist** | repo T2 + 数据 H6 + 安全 H1:denylist 扫 submitter source text,不扫 `tasks/` paths;V1 carve-out misframed |
| **采纳 user-decided** R14 完全 withhold,但 R17 reconstructability 论证作为 M0a 必交付 | 安全 C2:withhold ≠ unforgeable |

## Open Questions

### Resolved During Planning

- R13 阈值 + 构成:**60 task / ≥ 8 category + 稀有类 coverage 双门**(原 user-decided 50/6 升级,codex C6)
- R14/R17 机制:user-decided 完全 withhold + R17 论证 M0a 必交付
- Census 粒度:**M0a 100 task 静态 + M0b 分层 8-12 task GPU spike**(原 user-decided 100 全量升级,codex C4)
- V2 plan(triage 下)处理:完全 deprecate(user-decided)
- R10 实施:复用 `phase_b_golden_capture.py::_run_capture_child` 双 entrypoint 路径(`promoted-kernel` + `upstream-callable` 已存在,加 third value `extracted-ref`)
- R5 工件格式:safetensors + per (task, shape) hash(GOLDEN_SEED 固定常量,不入文件名)
- R8 schema 位置:**嵌套 `provenance.golden:`**(避免 collision)+ validator grace period
- `is_score_eligible` 演化:**不动**;新建 `is_provenance_eligible()` predicate
- family taxonomy:用 `meta.yaml.category`(repo 既有 12 enum;**`norm` 和 `normalization` 是 repo 真实两个独立 category,不 merge**(R17/rare-class 规则下分别处理))
- R10 tolerance:**upstream `atol/rtol`**(M1.1 抽的),非 `task.tolerance_class`
- v1 source ②:仅 deterministic;band-mode v2

### Deferred to Implementation

- R4 ref extractor 启发式:不同 upstream(vllm vs sglang vs trtllm)的 module-local ref / registry pattern 各异.[Technical]
- R7 容差注入 seam:在 eval 调用点合成 `ToleranceClass` vs 扩展 `load_policy`;NaN/dtype 语义差异处理.[Technical]
- R12 SHA 真值源统一:`phase_a_freeze.py`(硬编码常量)vs `extract_kernel.py`(`git rev-parse`)— 选 `git rev-parse` 路径作权威.[Technical,定向]
- `R10b 阈值`:何时降级抽样(per op family)— 等 M0b spike 数据决.[Needs research,M0b 输出]

### Deferred to follow-up plan

- **R20 band-mode**(非确定 source ② kernel):整链路设计(band artifact schema + statistical envelope comparator + manifest fields + eligibility gate + eval logging + test oracle)
- **R18 / R16 / R19 完整安全栈进阶**(本 plan 是 minimum-viable + ship-blocker 强制;完整 verified-runner spec + GPG/minisign 选型 + nsjail/bwrap sandbox 选型 + key 轮换流程)单独 security-focused plan

## High-Level Technical Design

> *This illustrates the intended approach. Directional guidance for review, not implementation specification.*

### Authority Triad(refactor 基础)

```
┌────────────────────────────────────────────────────────────────────┐
│  Authority 1: GoldenStore                                          │
│  - locate(task_id, shape_id) -> Path | None (public)                │
│  - locate_sealed(task_id, shape_id) -> Path | None (provider-only) │
│  - integrity_check(task_id, shape_id) -> bool (SHA + R18 signature)│
│  - is_complete(task_id) -> bool (all expected shapes accounted)    │
│  All consumers (eligibility / eval / validate_tasks / analyze /   │
│  phase_b_capture) MUST go through GoldenStore;no direct fs read.  │
├────────────────────────────────────────────────────────────────────┤
│  Authority 2: ToleranceAuthority                                   │
│  - for_task(task) -> ToleranceSpec                                  │
│  - ToleranceSpec { source, atol, rtol, class }                     │
│    source ∈ {upstream_test_explicit, upstream_test_default}        │
│              (v1 scope; migration_drift_budget v2 follow-up)       │
│  - compose(into ToleranceClass) for tolerance.compare              │
│  R10 / M4.1 / eval ALL consume same authority.                     │
├────────────────────────────────────────────────────────────────────┤
│  Authority 3: ProvenanceEligibility (NEW predicate, not is_score_) │
│  - is_provenance_eligible(meta) -> bool                            │
│    True iff (oracle_kernel_equivalence == VERIFIED OR                   │
│              gpu_kernel_capture_status == DONE)                              │
│         AND tolerance_source != migration_drift_budget             │
│             (v1: that source 为 v2 follow-up)                       │
│  - release_ready_with_provenance(meta) -> bool                     │
│    release_status_pass(meta) AND is_provenance_eligible(meta)      │
│    (release-readiness predicate; V5 canonical)                     │
│  - compute_release_ready_eligible(tasks) -> set[str]               │
│    compute_score_eligible(tasks) ∩ provenance_eligible_set         │
│    (denominator-membership helper; different from above)           │
│  - projected_eligible(corpus, exclude_r11_pending=True,            │
│                       exclude_r17_high=True)                       │
│  is_score_eligible() UNCHANGED — backward compat with               │
│  phase_b_golden_capture._select_tasks.                              │
└────────────────────────────────────────────────────────────────────┘
```

### 双源 golden pipeline 全景(集成 GoldenStore + ToleranceAuthority)

```
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 0 INGEST    vendor upstream file → tasks/<id>/kernel.py (existing)│
├──────────────────────────────────────────────────────────────────────┤
│ STAGE 1 SOURCE-① EXTRACT (NEW R4 — two-phase)                       │
│   Phase 1a: scan kernel module-local refs (rms_norm_ref / impls)    │
│   Phase 1b: scan registry (impls["native"], etc.)                   │
│   Phase 1c: fallback tests/*.py + assert_close / approx callsites   │
│   Emit: ref_callable + inputs_factory + atol/rtol +                  │
│         tolerance_source(explicit/default) + test_capture_sha       │
│   Vendor extracted ref → tasks/<id>/upstream_ref.py (pinned snapshot)│
│   Fail-fast on isolation issues → fallback ② via state-machine      │
│   (BACKLOG_PENDING_GOLDEN)                                           │
├──────────────────────────────────────────────────────────────────────┤
│ STAGE 2 R10 GPU EQUIVALENCE (NEW, GPU host)                          │
│   Inputs: vendored ref (Stage 1) + kernel.py                        │
│   Process: subprocess + safetensors IPC (reuse _run_capture_child   │
│            with new entrypoint_source='extracted-ref'),             │
│            isolated CPython (R19 — see Stage 5)                     │
│   Compare: ref_out vs kernel_out via ToleranceAuthority.for_task    │
│            (using UPSTREAM tolerance, NOT task.tolerance_class)     │
│   PASS → provenance.golden.oracle_kernel_equivalence = VERIFIED          │
│   FAIL → state-machine BACKLOG_PENDING_GOLDEN; fallback ②           │
├──────────────────────────────────────────────────────────────────────┤
│ STAGE 2b SOURCE-② GPU CAPTURE (existing phase_b_golden_capture,     │
│           v1 ONLY deterministic + R16 sanity gate before sign)      │
│   Verify run-to-run bit-identity (R20 deterministic);non-det → ABORT│
│   R16 sanity gate (range / NaN / shape / non-all-zero) — failed     │
│           output rejected (not signed)                               │
├──────────────────────────────────────────────────────────────────────┤
│ STAGE 3 FREEZE + R18 SIGN (R5/R8/R15/R18)                            │
│   GoldenStore.put(task, shape, tensor, mode={public,sealed})        │
│   Compute SHA256 of safetensors + sign manifest (provider key)      │
│   Write meta.yaml.provenance.golden = { ... }                       │
│   Sealed-private (GPFS) for hidden shapes;file mode 0700 owner=provider│
├──────────────────────────────────────────────────────────────────────┤
│ STAGE 4 ELIGIBILITY GATE (R9 / R7c / R13c)                           │
│   Use NEW release_ready_with_provenance() — not is_score_eligible   │
│   release_status_pass AND is_provenance_eligible                    │
│   Compute projected_eligible (EXCLUDE source ② task if R11 not landed AND EXCLUDE R17 high-risk task (math closed-form) — `projected_eligible(exclude_r11_pending=True, exclude_r17_high=True)`)│
│   PROCEED if ≥ 60 task ∧ ≥ 8 category ∧ rare-class coverage        │
│   else PROVISIONAL (R13b) or PROVISIONAL_MISSING_RARE or STOP (4-state per M0a.2 gate def)                                   │
├──────────────────────────────────────────────────────────────────────┤
│ STAGE 5 EVAL REPLACE (R6 / R15 / R18)                                │
│   Replace eval.py oracle.run path (81 task — only oracle_simplified  │
│     ∈ {simple_pytorch_ok, formula_exact_manual};19 upstream_oracle_  │
│     required already on golden path);                                 │
│   load via GoldenStore + R18 signature verify + SHA256 verify       │
│   AI oracle (oracle_pytorch.run) demoted: public shapes only,       │
│     diagnostic output to release/ai_oracle_consistency_report.md     │
├──────────────────────────────────────────────────────────────────────┤
│ STAGE 5 R19 BOUNDARY (proactive, enforced before Stage 2/4)         │
│   Layer A deny: safetensors, torch.load, torch.serialization         │
│   process_isolation: candidate uid ≠ verified-runner uid             │
│   sealed_golden path 0700 owner=provider uid                        │
└──────────────────────────────────────────────────────────────────────┘
```

### Subagent fan-out(集成 state-machine)

```
Master:
  pre-flight:
    - git config --add safe.directory infer/{vllm,sglang}  (once)
    - freeze upstream HEAD reference (infer/vllm @ HEAD, sglang @ HEAD)
    - emit upstream_closure_fingerprint base

Phase A — M0a static census (CPU dev host, no GPU):
  5 subagent batch parallel:
    each subagent: 20 task / batch
    actions per task:
      - source ① feasibility static classification
        (test file presence, tolerance source, gpu_skip detect, etc.)
      - source ② static classification
        (commit_sha verifiable, kernel.py import-only smoke, no invoke)
      - R17 reconstructability risk score (category + math closed-form heuristic)
      - emit /tmp/m0a_<N>.json (no shared write)
  Master aggregates → docs/triage_backlog/2026-05-27-census-report.md

Phase B — M0b GPU spike (GPU host, 8-12 stratified sample):
  GPU subagent: run real source ② invoke + run-to-run determinism
    (3 runs bit-identical check)
  Output: per-sample source_2 actually_invocable + deterministic_yes
  Master aggregates → census-report.md addendum

Phase C — M1 R4 extraction (CPU dev host) — only after R13c PROCEED:
  5 subagent batch parallel (same partition):
    each task in batch: run two-phase extractor
      - vendor extracted ref → tasks/<id>/upstream_ref.py
      - emit meta.yaml.provenance.golden = { ... }
    crash mid-batch: master re-dispatch *remaining* tasks
      (resume sentinel: meta.provenance.golden.source != null)

Phase D — M2 R10 equivalence (GPU host) + M3 R5 capture (GPU host):
  GPU subagent fan-out (sequential or 2-parallel max for GPU contention):
    each task: subprocess + R19 isolation + R16 sanity gate

Subagent contract (race-free):
  - Atomic write own task meta (atomic_write_yaml)
  - DO NOT git pull / branch switch (master frozen HEAD)
  - DO NOT write results/resolve_all.json (write /tmp/<batch>.json)
  - DO NOT modify scripts/ (read-only;tooling is master-owned)
  - On crash mid-batch: master re-dispatches remaining tasks idempotently
```

## Implementation Units

### Phase 0 — Authority Triad + Census(M0,blocking)

- [ ] **Unit M0.0: GoldenStore + ToleranceAuthority + ProvenanceEligibility abstractions**

**Goal:** 抽 3 个 first-class authority,作为后续所有 unit 的 single source of truth.

**Requirements:** Underpins R1, R5, R7, R9, R14, R15.

**Dependencies:** None.

**Files:**
**Files (V3 simplified — fold tolerance + provenance helpers into existing modules):**
- Create: `triton_ascend_bench/golden_store.py` — `GoldenStore` class(唯一新 module;5 hard-coded callsites + sealed/public + signing 真值得新 module)
- Modify: `triton_ascend_bench/tolerance.py` — **add `for_task(task) -> ToleranceSpec`** function + `ToleranceSpec` dataclass(不新 module,fold 进现 tolerance.py — 真小)
- Modify: `triton_ascend_bench/eligibility.py` — **add 3 functions next to existing predicates** + **migrate `golden_complete` to GoldenStore**:
  - `is_provenance_eligible(meta) -> bool` = `(provenance.golden.oracle_kernel_equivalence == VERIFIED OR provenance.golden.gpu_kernel_capture_status == DONE) AND provenance.golden.tolerance_source != migration_drift_budget`(v1 scope:`migration_drift_budget` task 不 release-ready;v2 follow-up 实施)
  - `release_ready_with_provenance(meta) -> bool = release_status_pass(meta) AND is_provenance_eligible(meta)`(release-readiness predicate)
  - **`compute_release_ready_eligible(tasks) -> set[str]`** — **NEW HELPER** = `compute_score_eligible(tasks) ∩ {t : is_provenance_eligible(t.meta)}`(denominator-membership;**`compute_score_eligible` 自身 UNCHANGED**)
  - `projected_eligible(corpus, exclude_r11_pending=True, exclude_r17_high=True) -> set[str]`(V3:加 `exclude_r17_high` 默认 True)
  - `is_score_eligible` **UNCHANGED**
  - **`golden_complete` MIGRATE**(V4 critical):从 hardcoded `tasks/<id>/golden/<shape_id>.safetensors` 改用 `GoldenStore.is_complete(task_id)`;同时 `golden_artifact_usable` 经 `GoldenStore.locate + integrity_check`(避免 cli analyze 读 hardcoded path 跟 eval 走 GoldenStore 漂移)
  - **`compute_verified_blocked` MIGRATE**(V4 critical):`cli.py:_cmd_analyze:882` 调此函数,内部读 `golden_complete` → 已经经过 V4 migrate 自动 wires through GoldenStore
- Modify: `triton_ascend_bench/eval.py:_resolve_golden` — call `GoldenStore.locate` 替代 hardcoded path
- Modify: `triton_ascend_bench/cli.py:_cmd_analyze:851` — call **`compute_release_ready_eligible`** for denominator(不是 `_cmd_analyze` 末尾 wrap;真改 denominator chokepoint)
- Modify: `scripts/validate_tasks.py:_validate_golden` — call `GoldenStore.is_complete`
- Modify: `scripts/phase_b_golden_capture.py` — read GoldenStore for shape list(`_select_tasks` 仍用 `is_score_eligible` 不动)
- Create: `tests/unit_tests/test_golden_store.py`
- Modify: `tests/unit_tests/test_tolerance.py` — add `test_for_task` cases
- Modify: `tests/unit_tests/test_eligibility.py` — add `test_is_provenance_eligible / test_compute_release_ready_eligible / test_projected_eligible` cases

**Approach:**
- `GoldenStore` 隐藏 4 个 hard-coded path:`tasks/<id>/golden/` 公开 + `$TAB_GPFS_ROOT/sealed_golden/<id>/` 私有 + manifest 路径 + R18 signature 路径
- `ToleranceAuthority.for_task(task)` 读 `provenance.golden.tolerance_source` + 对应字段,emit unified `ToleranceSpec`
- `is_provenance_eligible(meta)` 检查 `provenance.golden.oracle_kernel_equivalence == VERIFIED OR provenance.golden.gpu_kernel_capture_status == DONE`
- `release_ready_with_provenance(meta)` = `release_status_pass(meta) AND is_provenance_eligible(meta)`(其中 `is_provenance_eligible` 已含 `tolerance_source != migration_drift_budget` v1 guard;see canonical def above)
- `projected_eligible(corpus, exclude_r11_pending=True, exclude_r17_high=True)` 计 R13c 输入,**默认排除** R11 未落地的 source ② 任务

**Execution note:** Test-first. Author unit tests for each authority before wiring consumers.

**Patterns to follow:**
- `eligibility.release_status_pass` 既有 conjunctive predicate pattern
- `tolerance.ToleranceClass` 数据类
- `dataset.TaskDef` dataclass 模式

**Test scenarios:**
- Happy path: `GoldenStore.locate('L1_12', 'P1')` returns existing path
- Edge case: sealed shape locate from non-provider context returns `None`(uid check)
- Edge case: `is_provenance_eligible(meta_with_no_provenance)` returns `False`(grace period)
- Error path: missing manifest → `GoldenStore.integrity_check` returns `False`
- Error path: R18 signature invalid → `GoldenStore.integrity_check` returns `False`
- Integration: `release_ready_with_provenance(meta)` AND `is_score_eligible(meta)` 不冲突;`is_score_eligible` 仍是 `phase_b_capture._select_tasks` 真值源

**Verification:**
- All three authorities have ≥ 90% unit-test coverage
- Existing tests `tests/unit_tests/test_eligibility.py` / `test_eval_w4_3.py` 全过(backward compat)
- 100 task 跑 `analyze` + `validate_tasks --release-gate` 不破

---

- [ ] **Unit M0a.1: Source provenance static census tool**

**Goal:** 100 task 全量静态 census,emit (a)-(d) + 第 6 维 R17 reconstructability score;产 R13c 输入。**纯 CPU,无 GPU 调用**.

**Requirements:** R2, R12, R13c, R17

**Dependencies:** `INFER_ROOT/{vllm,sglang}/.git` 可达(`INFER_ROOT = REPO_ROOT.parent / "infer"`,dev-owned,**`safe.directory` 实际不需要因为非 root-owned** — V3 修 V2 误判;但 prefer master-side `git config --add safe.directory $INFER_ROOT/vllm` 作 defensive guard,fail-fast if HEAD 不可读)。

**Files:**
- Create: `scripts/source_provenance_census.py`
- Modify: `triton_ascend_bench/cli.py` — 加 `tab census` subcommand(unit M-CLI)
- Create: `tests/unit_tests/test_source_provenance_census.py`
- Create(output): `docs/triage_backlog/2026-05-27-census-report.md`(公开 summary)+ `$TAB_GPFS_ROOT/census-detail.json`(provider-private 详细)

**Approach:**
- 每 task 5 维度静态分类:
  - (a) source ① static feasibility:`infer/{repo}/tests/**/*.py` AND `infer/{repo}/**/*.py`(module-local + registry)内是否有 pytest invoking 该 kernel 的可隔离 tensor ref(static AST 扫,无 import 真跑)
  - (b) tolerance source:explicit `atol/rtol` vs framework default(`torch.allclose` 无参)
  - (c) GPU/platform skip detect:`@pytest.mark.gpu`,`skipif("not torch.cuda.is_available()")`,`from torch.distributed import`
  - (d) commit_sha verifiable rate:`git -C infer/{repo} cat-file -e <sha>`(配 master-side `safe.directory`)
  - (e) **R17 reconstructability risk**(NEW):category-based score(`elementwise / norm / gemm / sampling` math closed-form → high risk;`paged-attention / SSM` seed-dependent layout → lower);per-task 评分 0-10
  - **(NO source ② dynamic invoke — moved to M0b GPU spike)**
- Emit per-task JSON:`{task_id, category, source_1: {static_extractable, blocked_by_static}, tolerance_source, commit_sha_verifiable, r17_risk}`
- Aggregate report:family taxonomy 验证(`norm` vs `normalization` 是 disjoint category,不 merge)+ source ① 可行率 + 待 M0b 验 source ② 候选 list

**Execution note:** Test-first. Wire `tab census` CLI early; M0a.1 is the entry point for everything else.

**Patterns to follow:**
- `scripts/automation/select_tasks.py` per-task loop 模式(`per_task_diagnostic.py` 已被 revert,不可用)
- `extract_kernel.py:36 KernelExtractor` AST visitor pattern(M0a.1 重用)
- `scripts/automation/lib/atomic.py:atomic_write_json` 写 per-task JSON

**Test scenarios:**
- Happy path: L1_12_rms_norm 有上游 ref + explicit tolerance → emit `source_1.static_extractable=true, tolerance_source=upstream_test_explicit, r17_risk=high`
- Edge case: L3_08 SHA 不可达 → `commit_sha_verifiable=false`,排除 ①
- Edge case: ref 在 kernel module-local(非 tests/)→ phase 1a 命中
- Error path: `safe.directory` 未配 → fail-fast,**不 silent skip**
- Error path: master 未 freeze HEAD → 拒绝运行
- Integration: 100 task 跑完,report 含 family taxonomy 一致性 verify(`norm` vs `normalization` 是 disjoint categories,数量分别报告)

**Verification:**
- `python3 scripts/source_provenance_census.py --all --output docs/triage_backlog/2026-05-27-census-report.md` exit 0
- Report 含 5 维度 + R13c projected_eligible 数(static 部分)
- Public summary 不含 per-task tolerance values(detailed report 仅 provider)

---

- [ ] **Unit M0b.1: GPU stratified spike for source ② feasibility**

**Goal:** 真 GPU 上跑 8-12 task spike,测 source ② actually-invocable + run-to-run determinism;补 M0a 静态 census 给不到的 dynamic 证据.

**Requirements:** R2, R13c, R20

**Dependencies:** M0a.1 census output;GPU host 可达;`safe.directory` 已配.

**Files:**
- Create: `scripts/source_provenance_gpu_spike.py`
- Modify: `scripts/gpu_pipeline/05_provenance_spike.sh`
- Create: `tests/unit_tests/test_source_provenance_gpu_spike.py`(`@pytest.mark.gpu`)

**Approach:**
- 8-12 task 分层抽样:每 category 1-2 task(基于 M0a static classification),覆盖 source ② 候选 + 跨 family 代表
- 复用 `_run_capture_child(task, shape_ids, golden_dir, timeout_s=60, entrypoint_source='promoted-kernel')` 真跑 1 个 public shape
- determinism check:`N=3 runs same seed/env`,sha256(output_tensor) 三次相同?
- Emit `meta.provenance.golden.{gpu_kernel_capture_status: PENDING|DONE|FAILED|BLOCKED_NON_DETERMINISTIC, gpu_kernel_deterministic: bool}`(canonical schema fields per M1.2;后续 M3 用)
- Aggregate → 补 census report 第 (e) 真维度

**Execution note:** GPU host only. Master coordinates GPU access (single GPU spike, no parallel competition).

**Patterns to follow:**
- `phase_b_golden_capture._run_capture_child:727` 真 signature `(task, shape_ids, golden_dir, timeout_s, *, entrypoint_source, upstream_callable)`
- `scripts/gpu_pipeline/00_setup.sh` shell wrapper convention

**Test scenarios:**
- Happy path: L1_12_rms_norm GPU 跑 PASS + 3 runs bit-identical → `gpu_kernel_deterministic=true`
- Edge case: kernel uses atomics → N=3 出现 diff → `gpu_kernel_deterministic=false`,task 标"v1 不接受"
- Error path: GPU OOM / 30s timeout → `gpu_kernel_capture_status=FAILED` + `reason=timeout`
- Edge case: launch harness 缺失 → `gpu_kernel_capture_status=FAILED` + `reason=harness_missing`

**Verification:**
- GPU host `python3 scripts/source_provenance_gpu_spike.py --sample-stratified` exit 0
- Census report addendum 含 spike 真 dynamic data
- R13c gate(M0a.2)可用 spike data 计 `projected_eligible(gpu_kernel_subset, exclude_r11_pending=True, exclude_r17_high=True)`

---

- [ ] **Unit M0a.2: R13c go/no-go gate(60/8 + rare-class double gate)**

**Goal:** 实施 R13c 决策函数;census + spike 数据直接驱动 PROCEED / PROVISIONAL / PROVISIONAL_MISSING_RARE / STOP.

**Requirements:** R13, R13b, R13c

**Dependencies:** M0.0 `release_ready_with_provenance` + `projected_eligible`;M0a.1 + M0b.1 census 完成.

**Files (V4 corrected — split R13c into separate command,don't overload binary release gate):**
- Create: `scripts/provenance_release_planning_gate.py` — **NEW separate planning gate**(exit 0/1/2/3 = PROCEED/PROVISIONAL/PROVISIONAL_MISSING_RARE/STOP);**don't** reuse `validate_tasks.py:run_release_gate`,因为 `gpu_pipeline/40_release_gate.sh:4` 和 `tab_quickstart.sh:653` 已固化它作 binary hard release gate
- Modify: `triton_ascend_bench/cli.py` — 加 `tab r13c-gate` subcommand(M-CLI 也加)
- Create: `tests/unit_tests/test_r13c_gate.py`
- **`validate_tasks --release-gate` 保留既有 binary hard semantics 不动**(契约连续性)

**Approach (V3 corrected — R17 high-risk fully excluded;rare-class set disjoint from R17):**

**Term unification (V3+V7)**:全 unit 用 `category`(canonical,来自 `meta.yaml.category`);**deprecated alias `op_family` 已 V7 全 sed-removed**,所有原 `op_family` 引用现在都是 `category`。

**Two disjoint sets**:
- **R17 high-risk set**(M0a.1 emit;leaderboard-exclude 完全):`elementwise / norm / gemm / sampling` — math closed-form 类
- **rare-class set**(R13c gate input;**V4 最终修正:`norm` 在 R17 已 exclude 不入 rare**):corpus task 数 ≤ 5 的 category **AND 不在 R17 high-risk set**:`convolution(3) / spec_decode(1) / normalization(5)`(3 个 candidates;`norm(1)` 已在 R17 high-risk 不双重计;`normalization` 跟 `norm` 是 repo 真实**两个独立 category** — `L1_12_rms_norm.meta.category: norm`(R17 excluded) vs `L2_17_layer_norm_gated_fwd.meta.category: normalization`(rare candidate))

**R13c 决策 4 路**:
- PROCEED:`projected_eligible(exclude_r11_pending=True, exclude_r17_high=True) ≥ 60 AND ≥ 8 category AND rare_class_coverage_OK`
- PROVISIONAL:`projected_eligible ≥ 30 AND ≥ 6 category AND rare_class_coverage_OK` → R13b 单独披露层
- PROVISIONAL_MISSING_RARE:size OK but `convolution / spec_decode / normalization` 中 < 50% coverage(rare class set)
- STOP:size < 30 OR category < 6

`projected_eligible(corpus, exclude_r11_pending=True, exclude_r17_high=True)`:**V3 新加 `exclude_r17_high` 默认 True**;`exclude_r11_pending=True` 默认 True(R11 v1 不实施 migration_drift_budget,所有标 `tolerance_source=migration_drift_budget` 的 source ② 不入 PROCEED 分母)

**Patterns to follow:**
- `eligibility.release_status_pass` conjunctive pattern
- `validate_tasks.run_release_gate` 既有 set-intersection check

**Test scenarios (V3 corrected — 无 R11-landed 假设):**
- Happy path: census 报 50 (source ①) + 25 (source ② deterministic, R17 low/medium) = 75 eligible,8 category,rare_class_coverage_OK → PROCEED
- Edge case: 75 size but only 5 category → STOP(`category < 6` triggers STOP per gate def)
- Edge case: 75 size 8 category but `convolution+spec_decode+normalization` 全 0 coverage → PROVISIONAL_MISSING_RARE
- Edge case: 28 task → STOP
- Edge case: R17 high-risk task `elementwise/norm/gemm/sampling` 一律 leaderboard-exclude(不进 projected_eligible 任何分子)
- Edge case: source ② `tolerance_source=migration_drift_budget`(R11 v1 不实施)→ exclude_r11_pending 自动剔
- Integration: `tab r13c-gate` (NEW)exit code maps deterministically to gate state(0/1/2/3 = PROCEED/PROVISIONAL/PROVISIONAL_MISSING_RARE/STOP);`validate_tasks --release-gate` 既有 binary 语义不变(V4 split)

**Verification:**
- pytest 全绿;gate test 覆盖 4 路 × 边界
- `tab r13c-gate` cli exit code = 0/1/2/3(对应 PROCEED/PROVISIONAL/PROVISIONAL_MISSING_RARE/STOP)
- Census + spike data 100% deterministic → R13c output 100% deterministic

### Phase 1 — Source ① Extractor(M1)

- [ ] ~~**Unit M1.1: Two-phase AuthorRefExtractor**~~ **(V9 DEPRECATED — replaced by corpus `oracle_pytorch.py` + R10 GPU equivalence path via M2.1a/b/c;若后续需要 author-test 作 fallback source,留为 v2 follow-up plan 选项)**

下面段落保留作 historical reference / 备选 future v2 path,**V9 v1 不实施**:

**Goal:** 净新增 R4 抽取阶段,**two-phase**(module-local + registry → tests fallback);emit upstream tolerance + test_capture_sha;vendor extracted ref to `tasks/<id>/upstream_ref.py`.

**Requirements:** R4, R7, R12

**Dependencies:** M0a.1 census(决定哪些 task 走 ①);M0.0 ToleranceAuthority.

**Files:**
- Modify: `scripts/extract_kernel.py` — sibling `AuthorRefExtractor(ast.NodeVisitor)` + `--extract-ref` CLI flag
- Modify: `triton_ascend_bench/cli.py` — `tab extract-ref` subcommand
- Create: `tasks/<id>/upstream_ref.py` 约定 — vendored copy of extracted ref(SHA-pinned)
- Create: `tests/unit_tests/test_author_ref_extractor.py`

**Approach:**
- Phase 1a:scan `infer/{repo}/**/*.py` 找 kernel module-local refs(`def rms_norm_ref`,`impls["native"]`,registry pattern)
- Phase 1b:scan registry / `__init__.py` exported native impls
- Phase 1c:fallback `infer/{repo}/tests/**/*.py` callsites `torch.testing.assert_close` / `pytest.approx`
- 抽取的 ref **vendor** 到 `tasks/<id>/upstream_ref.py`(SHA-pinned snapshot,**不 import 真跑时的 infer/**);after vendor:
  - emit `provenance.golden.{ref_path: 'tasks/<id>/upstream_ref.py', test_capture_sha, tolerance_source, atol, rtol, extraction_phase ∈ {1a, 1b, 1c}}`
- Fail-fast enum:`{distributed_dep, current_platform_coupled, gpu_skip, no_ref_found, ref_not_tensor_only, multi_callsite_ambiguity}`

**Execution note:** Tests use embedded source-string fixtures(模仿 `test_static_checker.py` 形式;无 NodeVisitor-subclass test 先例).

**Patterns to follow:**
- `KernelExtractor:36` AST visitor pattern;`self.lines[start_line-1 : node.end_lineno]` snippet-capture trick
- `test_static_checker.py` embedded source + checker call

**Test scenarios:**
- Happy path 1a: ref in kernel module-local → `extraction_phase=1a, ref_path='tasks/X/upstream_ref.py'`
- Happy path 1c: ref only in tests → `extraction_phase=1c`
- Edge case: explicit tolerance `assert_close(rtol=1e-5, atol=1e-5)` → `tolerance_source=upstream_test_explicit`
- Edge case: `torch.allclose(out, ref)` 无参 → `tolerance_source=upstream_test_default` → 触发 R7c gate
- Error path: ref 含 `from torch.distributed` → `extraction_blocked_reason=distributed_dep`
- Error path: ref 含 `pytest.skip("requires GPU")` → `gpu_skip`
- Error path: 多个 callsite 数学冲突 → `multi_callsite_ambiguity` → fallback ②
- Integration: vendor + provenance 字段写入,validator 接受

**Verification:**
- `python3 scripts/extract_kernel.py --extract-ref --task L1_12_rms_norm` emit vendored `upstream_ref.py` + `provenance.golden.*`
- pytest `test_author_ref_extractor.py` 全绿
- 100 task 跑 source ① extractable rate ≥ 50%(census 校验,失败接 ②)

---

- [ ] **Unit M1.2: Provenance schema bump(nested `provenance.golden:`)**

**Goal:** 加 `provenance.golden:` 嵌套字段,**保留既有 `provenance: {added_in, authors, last_modified}`**;validator + dataclass + Layer 5 同步.

**Requirements:** R8, R12

**Dependencies:** M1.1 extractor 产嵌套字段.

**Files:**
- Modify: `scripts/validate_tasks.py:REQUIRED_META_TOP_LEVEL` — `provenance` 已 implicit 在;新加 grace period for `provenance.golden:` sub-block absent
- Modify: `triton_ascend_bench/dataset.py:TaskDef` — 加 `provenance_golden: dict | None`(嵌套读取)
- Modify: `scripts/automation/lib/file_blocks.py:validate_meta_content` (493) / `_validate_defer_meta_fields` (571) — 加 `provenance.golden.*` schema-shape check
- Modify: `specs/task_api_contract.md` — 加 §5d `provenance.golden:` 嵌套字段定义
- Create: `tests/unit_tests/test_provenance_golden_schema.py`

**Approach:**
- Closed-set fields under `provenance.golden:`:
  ```yaml
  provenance:
    added_in: v0.1                  # existing, untouched
    authors: [...]                  # existing
    last_modified: '2026-05-26'     # existing
    golden:                         # NEW nested block
      source: upstream_author_test | gpu_kernel_rerun
      tolerance_source: upstream_test_explicit | upstream_test_default | migration_drift_budget
      atol: <float>                 # NEW — pinned from upstream
      rtol: <float>                 # NEW — pinned from upstream
      ref_path: tasks/<id>/upstream_ref.py   # source ① only
      test_file_path: <relative to infer/{repo}>   # source ① only
      test_capture_sha: <40 hex>    # source ① only
      golden_sha256: <64 hex>       # required when artifact present
      # golden_signature is NOT a yaml field (would be self-referential); R18 detached signature stored as tasks/<id>/provenance.sig file
      oracle_kernel_equivalence: VERIFIED | FAILED | SKIPPED | NEEDS_MANUAL   # V4: NEEDS_MANUAL added for R12 SHA-chain drift (M2.0.5)
      gpu_kernel_capture_status: PENDING | DONE | FAILED | BLOCKED_NON_DETERMINISTIC   # V4: BLOCKED_NON_DETERMINISTIC for R20 v1 non-det reject (M3.1)
      gpu_kernel_deterministic: bool   # R20 N=3 check
      upstream_closure_fingerprint: <64 hex>  # cache invalidation key
      reviewers: [<name1>, <name2>]   # R8 真实署名
      reconstructability_risk: low | medium | high   # M0a-1 6th dim
  ```
- Validator grace period:既有 100 task meta 缺 `provenance.golden:` 不强制(只 warning);写 / R10/R5 跑出来的 task 必须含
- schema_version bump 1.0 → 1.1

**Patterns to follow:**
- `cost_model:` 字段添加历史(2026-05-21 cost-model-fill plan)
- `phase_a_prime / phase_b_capture` 字段添加历史

**Test scenarios:**
- Happy path: 写 nested `provenance.golden.source: upstream_author_test, ...` → validator pass + dataclass parse OK
- Edge case: 缺 `provenance.golden:` 但 `provenance:` 顶层有 → grace period not fail
- Edge case: 既有 `provenance.added_in/authors/last_modified` 仍可读
- Error path: `provenance.golden.source: invalid` → validator reject(closed enum)
- Error path: `golden_sha256` 不是 64-hex → Layer 5 reject
- Integration: 100 既有 meta 跑 validator 全过

**Verification:**
- pytest `test_provenance_golden_schema.py` 全绿
- `validate_tasks --all` 100 task 无 regression
- `dataclasses.asdict(TaskDef(...).provenance_golden)` 跟 yaml round-trip

### Phase 2 — Source ① GPU Equivalence(M2)+ R19 Boundary

- [ ] **Unit M2.0: R19 verified-runner boundary**(**提前 M2 前 ship**,不再 M5 placeholder)

**Goal:** 把 R19 verified-runner ↔ submitter ↔ golden 三者隔离落到 `process_isolation.py` + Layer A deny;**M2 前必须 ship**(防 sealed-golden 在 fs scope 时被候选 `safetensors.load_file` 读走).

**Requirements:** R19

**Dependencies:** M0.0 GoldenStore(`locate_sealed` uid check).

**Files (V4 corrected — trimmed to v1 minimum-viable scope):**
- Modify: `triton_ascend_bench/static_checker.py:_DENYLIST_IMPORT_EXACT` — 加 `safetensors`, `torch.load`, `torch.serialization` deny for **candidate code**(Layer A)
- Modify: `scripts/automation/bench_evaluator_daemon.py` — daemon 启动 self-check:`os.getuid() == provider_uid OR has CAP_SETUID` 才允许 start;否则 **refuse to start**(fail-closed,不 silent 退化)
- Create: `tests/unit_tests/test_r19_boundary.py`
- **V4 不在 v1 改**(显式 defer security follow-up plan):
  - `triton_ascend_bench/process_isolation.py` uid switching(需 CAP_SETUID + submitter uid lifecycle)
  - `bench_evaluator_daemon.py` `MADV_DONTFORK` mmap 隔离(需 daemon-as-root + ptrace-restrict)
- **sealed_golden 0700 owner=provider**:M3.2 写时设(file mode setup 在 M3.2 unit,不是 M2.0 改 process_isolation)

**Approach (V3 重写,minimum-viable + fail-closed if non-root):**
- **v1 仅 ship**:
  1. Static Layer A:候选 source 含 `safetensors` / `torch.load` / `torch.serialization` 导入 → reject(`_DENYLIST_IMPORT_EXACT` 加)
  2. Sealed_golden file mode 0700,owner = `$TAB_PROVIDER_USER`(M3.2 写时设;`os.access(path, os.R_OK)` from submitter uid returns False)
  3. **Fail-closed startup**:daemon 启动 self-check `os.getuid() == provider_uid OR has CAP_SETUID`,如果都不满足 → **refuse to start**(emit clear error,**不 silent 退化** 到 no-isolation)
- **v1 不 ship**(显式 defer 到 security follow-up plan):
  - `subprocess.Popen(..., user=submitter_uid)` 候选 uid 切换 — 需 CAP_SETUID + 完整 submitter uid 生命周期管理
  - `madvise(MADV_DONTFORK)` daemon mmap 隔离 — 需 daemon-as-root + ptrace-restrict + 完整 ASLR config
  - 完整 verified-runner ↔ candidate ↔ sealed-golden 三方 process isolation 模型
- **V3 honesty**:v1 防的是 candidate 通过 Python source 直接读 sealed file(Layer A) + 候选 uid 无 read access(file mode);**不防** in-process memory sharing / shared CUDA pages / kernel namespace 共享。完整 R19 等 security follow-up plan

**Patterns to follow:**
- `process_isolation.sanitised_shapes_view:682-699` candidate view sanitisation
- `static_checker._BANNED_REFLECT_NAMES` deny list pattern

**Test scenarios:**
- Happy path: candidate source `import torch` OK
- Edge case: candidate `import safetensors` → Layer A reject
- Edge case: candidate `torch.load("/tmp/x")` → audit hook reject
- Error path: submitter uid `cat $TAB_GPFS_ROOT/sealed_golden/.../H1.safetensors` → EACCES
- Integration: existing candidate tests 不破

**Verification:**
- R19 boundary test 全绿
- 候选源含 `safetensors` 之类被 rejected
- sealed_golden 文件 mode 0700 + owner check

---

- [ ] **Unit M2.0.5: R12 SHA-chain integrity check**(V3 新加)

**Goal:** kernel.py 源 commit 跟 author test 抽取 commit equality assertion + 不符 → `provenance.golden.oracle_kernel_equivalence = NEEDS_MANUAL`,**不再 deferred to research**(原 V2 V1)。

**Requirements:** R12

**Dependencies:** M1.1 emit test_capture_sha;M1.2 schema 含 kernel.py commit SHA(`meta.upstream.commit_sha`).

**Files:**
- Create: `scripts/verify_upstream_commit_chain.py`
- Modify: `triton_ascend_bench/eligibility.py` — add R12 verify hook in `is_provenance_eligible`
- Create: `tests/unit_tests/test_r12_sha_chain.py`

**Approach (V3 corrected — 统一 identifier):**

**重要 distinction**(V3 修正 V2 confusion):
- `meta.upstream.commit_sha`:**kernel.py 抽取时的 git commit SHA**(40-hex,既有字段)
- `provenance.golden.test_capture_sha`:**author test 抽取时的 git commit SHA**(40-hex,M1.1 新加)
- 这两个**都是 commit SHA**(40-hex 格式),可直接比较
- `meta.upstream.file_sha256`:kernel.py 文件 SHA-256(64-hex,既有 file content hash) — **R12 不用这个,只比 commit**

**R12 assertion**:`meta.upstream.commit_sha == provenance.golden.test_capture_sha`(kernel + test 来自同一 commit)
- 相等 → VERIFIED:kernel.py 跟 author test 同 commit 抽取,R12 fully passes
- 不符 + 两个 commit 都 `git cat-file -e <sha>` reachable in `INFER_ROOT/{repo}` → `NEEDS_MANUAL`(可能 acceptable drift,human triage)
- snapshot-no-git tasks(5 个)→ 强制 source ② path,R12 不适用
- L3_08 等 commit_sha 不可达 → 强制 source ② or BACKLOG_PENDING_GOLDEN

**Test scenarios:**
- Happy path: kernel + test 同 commit `18f6bf5a` → R12 VERIFIED
- Edge case: kernel `18f6bf5a` test `409d350f` 都 reachable → NEEDS_MANUAL(人工 triage)
- Edge case: snapshot-no-git → reject source ①
- Error path: commit unreachable → BACKLOG_PENDING_GOLDEN + 强制 source ②

**Verification:**
- 100 task R12 check 全跑过,emit ~95 VERIFIED + 5 NEEDS_MANUAL/EXCLUDED(snapshot-no-git + L3_08)

---

- [ ] **Unit M2.1a: oracle_pyref_precompute.py(dev,CPU 阶段)**

**Goal:** dev 机自动化 R10 第一段 — 静态 audit + CPU 跑 oracle_pytorch.py + save reference artifacts;**V9 v1 deliverable,operator dev 机立即可用**.

**Requirements:** R10 (dev portion)

**Dependencies:** M0.0 ToleranceAuthority;M1.2 provenance schema;**V9 不再依赖 M1.1 author-test extractor (deprecated)**.

**Files:**
- Create: `scripts/oracle_pyref_precompute.py`(dev 机自动化主程序)
- Modify: `triton_ascend_bench/cli.py` — 加 `tab oracle-pyref-precompute` subcommand
- Create: `tests/unit_tests/test_oracle_pyref_precompute.py`(CPU-only)

**Approach (V9):**

对每 task,dev 机自动跑(无需 GPU,纯 CPU + AST):

1. **Static signature audit**(AST parse):
   - oracle_pytorch.py::run 真读 inputs[?] key set
   - kernel.py 上游 entrypoint 真签名
   - shapes.yaml.inputs 真定义 keys
   - 三者交叉对齐 → `ALIGNED / SIGNATURE_MISMATCH / NEEDS_BRIDGE`
2. **若 ALIGNED**:`build_inputs(shapes.yaml.public_shapes[P1, P2], seed=GOLDEN_SEED)` on CPU
3. **跑 `oracle_pytorch.run(inputs)` on CPU**:save tensors to `tasks/<id>/r10_pyref/{inputs_<sid>.safetensors, ref_<sid>.safetensors}`
4. **emit metadata**:`provenance.golden.oracle_pyref_status ∈ {ALIGNED_READY, SIGNATURE_MISMATCH, NEEDS_BRIDGE, NEEDS_GPU_FOR_ORACLE, RUNTIME_ERROR}`
5. 若 `SIGNATURE_MISMATCH` → emit **suggested fix detail**(具体 missing key / mismatched key)给 user 看;**不自动 fix oracle**(避免捏造,需 human review)

**Execution note:** Pure CPU. Subagent fan-out 5 batch × 20 task 并发(每 task ~10-30 sec)。**V9 关键**:这一步 dev 机能跑全 100 task,不卡 GPU 可用性。

**Patterns to follow:**
- `extract_kernel.py:36 KernelExtractor` AST visitor 模式
- `triton_ascend_bench/inputs.py:build_inputs` 既有 CPU input builder
- `scripts/automation/lib/atomic.py:_atomic_replace`(safetensors binary write)

**Test scenarios:**
- Happy path L1_12_rms_norm: oracle aligned, CPU run OK → `oracle_pyref_status=ALIGNED_READY`,artifacts saved
- Edge case L2_15_fused_dual_residual_rmsnorm: oracle `(residual_a, residual_b, weight)` vs kernel `(residual, weight1, weight2)` → SIGNATURE_MISMATCH + suggested fix detail
- Edge case L1_07_dequantize_k_cache_fast: oracle `(nope_q, nope_s, rope, group_size)` vs kernel `(quant_k_cache, group_size)` → NEEDS_BRIDGE(layout 不同,需 input bridge)
- Edge case oracle 用 libdevice / Triton-only op → NEEDS_GPU_FOR_ORACLE(标记;不在 dev 跑;M2.1b GPU 那段处理)
- Error path: CPU OOM / shape 太大 → RUNTIME_ERROR + skip
- Integration: 跑 100 task,~85-95 ALIGNED_READY

**Verification:**
- `python3 scripts/oracle_pyref_precompute.py --all --workers 8` exit 0
- ALIGNED_READY 的 task → `tasks/<id>/r10_pyref/` 含 inputs + ref safetensors
- emit `docs/triage_backlog/<date>-oracle-pyref-audit.md` 报告 SIGNATURE_MISMATCH / NEEDS_BRIDGE 名单 + 建议 fix

---

- [ ] **Unit M2.1b: gpu_pipeline/oracle_equivalence_verify.sh(GPU,operator 手动跑,v1 半自动)**

**Goal:** GPU 机 operator 手动跑 oracle ref vs kernel.py 数值等价比对;**V9 v1 不自动化 SSH 编排**(复用既有 Tier 1 半自动模式,从 `gpu_pipeline/` 既有命名约定).

**Requirements:** R10 (GPU portion)

**Dependencies:** M2.1a artifacts(`tasks/<id>/r10_pyref/`)git committed OR rsync 到 GPU 机;M2.0 R19 boundary;GPU 机可达;`_run_capture_child` plumbing.

**Files:**
- Create: `scripts/gpu_pipeline/oracle_equivalence_verify.sh`(operator-facing wrapper)
- Create: `scripts/oracle_equivalence_gpu_runner.py`(被 .sh 调的 Python runner)
- Create: `docs/runbooks/operator-oracle-equivalence-gpu.md`(operator manual,2-3 页 step-by-step)

**Approach (operator-facing manual workflow,V9 v1):**

```bash
# 1. operator ssh GPU machine
ssh <gpu-host>
cd /path/to/triton-ascend-bench
git pull  # 获取 dev 机 commit 的 M2.1a artifacts

# 2. operator runs one command
bash scripts/gpu_pipeline/oracle_equivalence_verify.sh --task-list all
# 或 single task:
bash scripts/gpu_pipeline/oracle_equivalence_verify.sh --task L1_12_rms_norm

# 3. 内部对每 ALIGNED_READY task:
#    load tasks/<id>/r10_pyref/{inputs_<sid>, ref_<sid>}.safetensors → GPU
#    run kernel.py::<entrypoint>(inputs) on GPU (via _run_capture_child)
#    tolerance.compare(ref, kernel_out, ToleranceAuthority.for_task(task))
#    emit /tmp/r10_verdict/<task>_<sid>.json
# 
# 4. operator 完了 rsync /tmp/r10_verdict/ 回 dev 机 OR commit 到 git
```

**Execution note:** GPU host only,operator 手动操作。**V9 v1 显式不自动化 SSH 编排**(reuse 既有 Tier 1 半自动);v2 follow-up plan 实施完整 daemon-based 自动化.

**Patterns to follow:**
- `scripts/phase_b_golden_capture.py:_run_capture_child:727` 真 signature
- `scripts/gpu_pipeline/30_golden_capture.sh` 既有 operator wrapper 模式
- `scripts/gpu_pipeline/40_release_gate.sh` 既有 numbered shell convention

**Test scenarios:**
- Happy path: ALIGNED_READY task on GPU → kernel.py run + compare → VERIFIED verdict JSON
- Edge case: kernel uses libdevice → GPU 上跑正常(dev 跑不动的就在这里跑)
- Error path: kernel launch harness 缺 → emit `equivalence_blocked_reason=launch_harness_missing`
- Edge case: tolerance fail → `oracle_kernel_equivalence=FAILED` + per-shape diff report

**Verification:**
- 5 task PoC:operator 在 GPU 机跑 5 task,verdict JSON 100% 生成
- GPU 时间 ≤ 1 min for 5 task(verify 高效)
- ~85 task 全跑 ≤ 10 min GPU 时间

---

- [ ] **Unit M2.1c: apply_oracle_equivalence_verdict.py(dev,自动同步 GPU verdict)**

**Goal:** GPU operator 跑完后,dev 机自动同步 verdict JSON 回 meta.yaml + emit final provenance status.

**Requirements:** R10 (sync portion)

**Dependencies:** M2.1b 产 verdict JSONs;dev 机上 verdict 可达(via git or rsync)

**Files:**
- Create: `scripts/apply_oracle_equivalence_verdict.py`
- Modify: `triton_ascend_bench/cli.py` — 加 `tab apply-equivalence-verdict` subcommand
- Create: `tests/unit_tests/test_apply_oracle_equivalence_verdict.py`

**Approach (V9):**
- 读 `/tmp/r10_verdict/*.json`(或 `$TAB_GPFS_ROOT/r10_verdict/*.json`)
- 对每 task:
  - load verdict
  - write `provenance.golden.oracle_kernel_equivalence ∈ {VERIFIED, FAILED, MIXED, BLOCKED}`
  - if VERIFIED: write `provenance.golden.golden_sha256` (= ref safetensors hash);**oracle_pytorch.py 升 truth source**(V9 关键 — user-decided 修正 origin R3)
  - if FAILED: write `provenance.golden.equivalence_diff_report`(per-shape max_abs / max_rel) + 标 fallback to source ②(GPU 跑 kernel 直接 emit golden via M3.2)
- atomic_write_yaml(meta.yaml) for each task

**Test scenarios:**
- Happy path VERIFIED → meta 写 oracle_kernel_equivalence = VERIFIED + golden_sha256
- Edge case FAILED → meta 写 FAILED + diff_report,标 source ② fallback
- Edge case MIXED(部分 shape VERIFIED 部分 FAILED)→ MIXED status + per-shape detail
- Error path: verdict file 缺 → skip + warn,不破已写 task
- Integration: 100 task verdict sync 全 atomic

**Verification:**
- pytest `test_apply_oracle_equivalence_verdict.py` 全绿
- `tab apply-equivalence-verdict --verdict-dir /tmp/r10_verdict/` exit 0
- All updated meta.yaml within 30 sec

### Phase 3 — Source ② Capture + R16 Sanity Gate + R20 Determinism(M3)

- [ ] **Unit M3.0: R16 sanity gate**(**入 M3.2 前,不再 M5 placeholder**)

**Goal:** R16 output sanity gate — 不可信上游 code 产 golden 后,**先过 sanity 才能签**.

**Requirements:** R16

**Dependencies:** None.

**Files (V4 corrected — central table, NOT per-task):**
- Create: `triton_ascend_bench/sanity_bounds.yaml` — **single central table by category**(NOT `tasks/<id>/sanity.yaml`,V2 per-task design deprecated)
- Modify: `scripts/phase_b_golden_capture.py` — 加 sanity gate step before manifest sign(读 central table)
- Modify: `specs/task_api_contract.md` — 加 §5e `sanity_bounds.yaml` schema(central,非 per-task)
- Create: `tests/unit_tests/test_sanity_gate.py`
- **CODEOWNERS update**:`triton_ascend_bench/sanity_bounds.yaml` reviewer 不能跟改 `kernel.py` / `oracle_pytorch.py` 同人(强制 review boundary 分离)

**Approach (V3 重写,reflects decisions table):**
- **Single central** `triton_ascend_bench/sanity_bounds.yaml`,by-category bounds(不 per-task `tasks/<id>/sanity.yaml` — V2 design 已 deprecate);changes to this file require separate CODEOWNERS reviewer(reviewer 不能跟改 `kernel.py` 或 `oracle_pytorch.py` 的同 reviewer 是同人,实现 review boundary 分离)
- Schema(closed):
  ```yaml
  # triton_ascend_bench/sanity_bounds.yaml
  by_category:
    elementwise: {max_abs: 1e6, forbid_all_zero: true}
    normalization: {max_abs: 1e6, forbid_all_zero: true}
    paged-attention: {max_abs: 1e3, must_match_input_rank: true}
    quantization: {max_abs: 256, forbid_all_zero: true}
    # ... rest of 12 enum
  ```
- 默认 bounds **不**从 `oracle_pytorch.run` 推(self-attesting collapse,V2 design 已 deprecate);采 per-op-family hand-curated bounds,**M3.0 ship 时 reviewer 用领域知识填**(elementwise SiLU output 数学 max_abs ~= input max_abs;quantized int8 trivially ≤ 256;etc.)
- **R16 scope downgrade**(V3 诚实):本 unit ship gross-corruption defense(NaN / Inf / shape / all-zero),**不**声明 T2 supply-chain full defense;T2 真防线在 R10 GPU equivalence(source ①)+ R18 signed trust root(R12 commit provenance)
- Sanity FAIL → `provenance.golden.sanity_gate_status = FAILED, sanity_gate_reason: <which gate>`,**不签 manifest**;按 `sanity_bounds.yaml` 内 category 自动 dispatch

**Patterns to follow:**
- `cost_model:` schema 加新文件先例

**Test scenarios (V4 corrected — central table):**
- Happy path: kernel 输出 in bounds(via central `sanity_bounds.yaml[by_category][task.category]`)→ R16 PASS,进入 R5 freeze
- Error path: kernel 输出 all-zero(input non-zero) → R16 FAILED,reject
- Error path: kernel 输出 NaN/Inf > tolerance → R16 FAILED
- Error path: kernel 输出 shape rank 跟 input 不符 → R16 FAILED
- Edge case: `sanity_bounds.yaml` 缺 category 条目 → fail-closed,reject(不 silent skip;迫 reviewer 填)

**Verification:**
- R16 sanity test 全绿
- `sanity_bounds.yaml` 含全 12 个 corpus category(每条 reviewer-attested)
- CODEOWNERS 配置 verify:`sanity_bounds.yaml` 跟 `kernel.py` / `oracle_pytorch.py` reviewer 不重叠

---

- [ ] **Unit M3.1: R20 determinism verification(v1 deterministic-only,band-mode v2)**

**Goal:** Source ② kernel 强制 N=3 runs bit-identical;不稳定 task 标"v1 不接受",**band-mode 延 v2**.

**Requirements:** R20

**Dependencies:** `phase_b_golden_capture.py` 现有 capture 路径.

**Files:**
- Modify: `scripts/phase_b_golden_capture.py` — 加 `--verify-determinism N=3` flag
- Create: `tests/unit_tests/test_r20_determinism_gpu.py`(`@pytest.mark.gpu`)

**Approach:**
- 对 source ② task:fixed seed/env,跑 N=3,emit `bit_identical ∈ {true, false}`
- `true` → exact-hash golden(R15 hash gate)
- `false` → **v1:`provenance.golden.gpu_kernel_capture_status = BLOCKED_NON_DETERMINISTIC, gpu_kernel_deterministic = false`**;task 不进 **release_ready_with_provenance**(via `is_provenance_eligible == false`)直到 v2 band-mode;`is_score_eligible` 仍可 true(denominator membership 由其它谓词决)

**Patterns to follow:**
- `_run_capture_child` GOLDEN_SEED 常量 + deterministic env

**Test scenarios:**
- Happy path: deterministic kernel 3 runs bit-identical → exact-hash freeze
- Edge case: atomics-using kernel → byte diff → BLOCKED_NON_DETERMINISTIC
- Error path: kernel launch failed → emit `kernel_capture_blocked_reason=launch_failed`

**Verification:**
- pytest `test_r20_determinism_gpu.py` 全绿
- 100 task source ② 全标 `gpu_kernel_deterministic ∈ {true, false}`

---

- [ ] **Unit M3.2: Golden artifact freeze + signed manifest**

**Goal:** Provider 物化 golden + signed manifest;public vs sealed 分离;走 `GoldenStore.put`.

**Requirements:** R5, R14, R15, **R18(共 ship)**

**Dependencies:** M2.0 R19 boundary;M3.0 sanity gate;M3.1 R20 verified;M0.0 GoldenStore.

**Files:**
- Modify: `scripts/phase_b_golden_capture.py` — emit `golden_pub.safetensors` (public) + `golden_hid.safetensors` (sealed)
- Create: `scripts/golden_manifest.py` — manifest writer + R18 sign
- Modify: `scripts/automation/lib/atomic.py` — promote `_atomic_replace` to public `atomic_write_bytes`
- Create: `$TAB_GPFS_ROOT/sealed_golden/<id>/` directory convention(provider-private)
- Create: `tests/unit_tests/test_golden_manifest.py`

**Approach (V3 重写,reflects expanded R18 scope):**
- Public golden → `tasks/<id>/golden/public_<shape_id>.safetensors`(submitter tree)
- Sealed golden → `$TAB_GPFS_ROOT/sealed_golden/<id>/sealed_<shape_id>.safetensors`(mode 0700,owner=$TAB_PROVIDER_USER)
- **R18 OOB signature scope V3 扩展**(不仅 golden manifest):
  - `tasks/<id>/golden/manifest.json`:`{shape_id, sha256, tolerance_class}`
  - `tasks/<id>/provenance_commit.json`:**新 file**,canonical JSON of `{task_id, meta_upstream_commit_sha, kernel_py_sha256, test_capture_sha, upstream_closure_fingerprint}`(R12 commit-chain proof;**includes meta.upstream.commit_sha so R18 signature covers R12 gate input**)
  - `triton_ascend_bench/sanity_bounds.yaml`:全局表(M3.0 ship)
  - `meta.yaml.provenance.golden` canonical-JSON 视图(per-task provenance fields)
  - 4 artifacts 用 **single detached signature** `tasks/<id>/provenance.sig`,签内容含 4 个 sha256 anchor
- **Trust root V3**:`triton_ascend_bench/trust_roots.json`(**不是** pinned in `__init__.py` 常量):
  ```json
  {
    "active_keys": [
      {"key_id": "v1", "pubkey_ed25519_hex": "...", "valid_from": "2026-05-26", "valid_until": "2027-12-31"}
    ],
    "revoked_keys": []
  }
  ```
  支持 rotation:provider 添 new key (valid_from 早于 expiry)→ overlap period → revoke old key
- Private key 存 `$TAB_GPFS_ROOT/keys/ed25519_v1.priv` mode 0400 owner=$TAB_PROVIDER_USER;runbook 文档 rotation 流程
- `static_checker` carve-out **不做**(misframed)— denylist 扫 submitter source,不扫 tasks/ paths

**Patterns to follow:**
- atomic_write_yaml/json/bytes
- `shapes.yaml.{public_shapes, hidden_shapes}` 既有分离

**Test scenarios:**
- Happy path: 5 task 跑 → public golden 在 task 目录,sealed 在 GPFS,external detached signature `tasks/<id>/provenance.sig` (not embedded in manifest)
- Edge case: task 无 hidden shape → 不 emit sealed,manifest 只 public
- Error path: GPFS write fail → `GoldenWriteError`,abort capture
- Adversarial: 篡改 manifest.json + safetensors 同 atomic write → signature verify fail(R18)
- Adversarial(verify R14): submitter uid `cat $TAB_GPFS_ROOT/sealed_golden/.../*.safetensors` → EACCES

**Verification:**
- 5 task 跑后,public 在 task,sealed 在 GPFS
- `golden_manifest.py --verify <task>` exit 0(SHA + signature 双重 verify)
- Submitter uid 不能 grep / cat sealed golden(file-system + signature)
- pytest 全绿

### Phase 4 — Eval Replacement(M4)+ AI Oracle Demotion

- [ ] **Unit M4.1: Eval golden-load path via GoldenStore**

**Goal:** 替换 `eval.py` 81 task(`oracle_simplified ∈ {simple_pytorch_ok, formula_exact_manual}`)的 `oracle_pytorch.run` 调用;**19 个 `upstream_oracle_required` 已走 offline,只补 R15+R18 verify chain**.

**Requirements:** R3, R6, R15

**Dependencies:** M3.2 signed golden + GoldenStore.

**Files:**
- Modify: `triton_ascend_bench/eval.py:_resolve_golden:759` — call `GoldenStore.locate` instead of hardcoded path
- Modify: `triton_ascend_bench/eligibility.py:golden_artifact_usable:142` — 扩 SHA + signature verify chain via GoldenStore
- Create: `tests/unit_tests/test_eval_golden_path.py`

**Approach:**
- `_resolve_golden` 三态扩四态:`oracle / static_golden / formula_diagnostic / none`
- For 81 task:`load via GoldenStore + R15 SHA verify + R18 signature verify(都过)→ compare_candidate_vs_golden(via tolerance.compare with ToleranceSpec)`
- 任一 verify FAIL → raise `GoldenIntegrityError`(submitter 看 "evaluation failed: golden integrity")
- AI oracle 降级:**仅 public shapes 跑** `oracle_pytorch.run` 作 diagnostic;hidden shapes skip;不一致 emit `ai_oracle_inconsistency` to `release/ai_oracle_consistency_report.md`

**Patterns to follow:**
- `tolerance.compare` 既有 ToleranceClass 接口
- `eligibility.golden_artifact_usable` 既有 path.is_file + load chain

**Test scenarios:**
- Happy path: candidate 在 ToleranceSpec 内 → eval PASS;ai_oracle_inconsistency 不写
- Edge case: candidate PASS 但 AI oracle disagree → eval PASS + emit `ai_oracle_inconsistency` 诊断
- Error path: golden 缺失 → `GoldenMissingError`
- Error path: golden SHA 不符 → `GoldenIntegrityError`(R15)
- Error path: golden signature invalid → `GoldenIntegrityError`(R18)
- Adversarial: 篡改 manifest + safetensors → signature mismatch → reject
- Integration: 5 task 跑 eval,verdict 跟 golden 比对一致;AI oracle 仅诊断
- Integration: 19 个 `upstream_oracle_required` task 走既有 offline path + 新 R15+R18 verify

**Verification:**
- pytest `test_eval_golden_path.py` 全绿
- 既有 eval 测试不破
- 100 task 跑 eval,无 false-PASS(R18 signature check 拦)

### Phase 5 — CLI + Security Hardening Follow-up(M-CLI + M5)

- [ ] **Unit M-CLI: tab CLI subcommands**

**Goal:** 加 `tab census` / `tab extract-ref` / `tab verify-source-1` / `tab freeze-golden` 子命令,达 plan success metric "新 op onboarding ≤ 30 min wall-clock".

**Requirements:** Operator UX,success metrics

**Dependencies:** M0.0, M0a.1, M0b.1, M1.1, M2.1, M3.2.

**Files:**
- Modify: `triton_ascend_bench/cli.py` — add `add_subparsers` entries + `_cmd_<name>` functions:
  - `_cmd_census(args)`: dispatch to `scripts/source_provenance_census.py`
  - `_cmd_extract_ref(args)`: dispatch to `scripts/extract_kernel.py --extract-ref`
  - `_cmd_verify_source_1(args)`: dispatch to `scripts/oracle_kernel_equivalence_check.py`
  - `_cmd_freeze_golden(args)`: dispatch to `scripts/phase_b_golden_capture.py`
- Modify: `scripts/automation/runners/tab_quickstart.sh` — add `quickstart census` / `quickstart freeze-golden` steps

**Approach:**
- 复用既有 `add_subparsers + _cmd_<name>` pattern(`cli.py:34-94`,9 个先例)
- 每子命令 thin dispatch(parse args → call script main)

**Patterns to follow:**
- `_cmd_analyze:766` 既有 dispatch pattern

**Test scenarios:**
- Happy path: `tab census --task L1_12` → census output
- Happy path: `tab freeze-golden --task L1_12` → golden freeze + manifest
- Edge case: 缺前置(census 未跑) `tab freeze-golden` → fail-fast 提示先 census

**Verification:**
- 5 CLI 跑通
- `tab --help` 列出新子命令

---

- [ ] **Unit M5.1: Security follow-up plan placeholder**

**Goal:** 记录后续 security plan 入口点(R16/R18/R19 已 minimum-viable ship;完整 spec 留 follow-up).

**Requirements:** R16, R18, R19(扩 spec)

**Dependencies:** M0-M4 全部 ship.

**Files:**
- Modify: `specs/task_api_contract.md §7` — add security boundary placeholder
- Modify: this plan §"Deferred to follow-up plan" — explicit list

**Approach:** Out of scope this plan;placeholder.

**Verification:** 后续 security plan 接管.

## System-Wide Impact

- **Interaction graph**:
  - `meta.yaml.provenance.golden:` 嵌套字段下游消费:`is_provenance_eligible` (NEW) / `release_ready_with_provenance` (NEW) / `eval._resolve_golden` (via GoldenStore) / `validate_tasks --release-gate` / `phase_b_golden_capture.{capture_task, _select_tasks}`(`_select_tasks` 仍读 `is_score_eligible` 不动)
  - `tasks/<id>/golden/` 目录:`GoldenStore.locate`(public)/ `validate_tasks` 校验 / `static_checker` **不需要 carve-out**
  - `$TAB_GPFS_ROOT/sealed_golden/<id>/`:`GoldenStore.locate_sealed`(provider-only,uid check)/ provider 跑 manifest sign
  - subagent fan-out 在 task 维度执行,接 17-state machine(BACKLOG_PENDING_GOLDEN 状态)
- **Error propagation**:
  - `GoldenIntegrityError`(R15 SHA + R18 signature)→ eval 直接 fail
  - `R10 FAILED` → state-machine transit BACKLOG_PENDING_GOLDEN → fallback ②
  - `R16 FAILED` → reject golden,不签 manifest
  - extraction_blocked_reason 显式 enum,fallback 路径机器可读
- **State lifecycle risks**:
  - golden 冻结后,upstream commit 更新触发 `upstream_closure_fingerprint` 失配 → state-machine transit `golden_stale`
  - sealed golden GPFS write race:atomic_write_bytes + per-task lock(`scripts/automation/lib/lock.py`)
  - R10 FAILED 是 sticky 直到 `upstream_closure_fingerprint` 变化(`test_capture_sha` / `kernel.py sha256` 变 → 重跑 STAGE 1)
- **API surface parity**:
  - `tab` CLI 加 4 subcommand(`census`/`extract-ref`/`verify-source-1`/`freeze-golden`)
  - `tab_quickstart.sh` 加 `census` / `freeze-golden` 步骤
- **Integration coverage**:
  - 100 task M0a static + M0b GPU spike + M1 extraction + M2 R10 是 cross-layer integration,subagent 5 batch × 20 task 并发
  - Provider GPU host ↔ CPU dev host SSH 复用既有 `bench_evaluator_daemon.py` 通道
- **Unchanged invariants**:
  - `kernel.py` 永不改(SETTLED)
  - `scripts/resolve_upstream_entrypoints.py`(评估器自身)不改
  - `tolerance_policy.yaml` 全局表不改(R7 注入在调用点)
  - `oracle_pytorch.py::run` 签名不变(降级,public shapes only)
  - **`eligibility.is_score_eligible()` 语义不变**(`phase_b_golden_capture._select_tasks` 仍用,避免循环依赖)
  - 既有 `provenance: {added_in, authors, last_modified}` 字段保留;V2 只新加 `provenance.golden:` 嵌套块
  - `static_checker.\bgolden\b` denylist 不动(扫 submitter source,不扫 tasks/ paths)
- **New invariants**:
  - `provenance.golden.oracle_kernel_equivalence == VERIFIED ⇒ provenance.golden.upstream_closure_fingerprint != null`
  - `provenance.golden.gpu_kernel_deterministic == false ⇒ v1 不进 **release_ready_with_provenance**(`gpu_kernel_capture_status = BLOCKED_NON_DETERMINISTIC` 阻止 `is_provenance_eligible` 返 true)`(band-mode v2)
  - `release_ready_with_provenance(meta) = release_status_pass(meta) AND is_provenance_eligible(meta)`(V4 canonical:与 §M0.0 定义一致;`is_score_eligible` 是 denominator 判据,不是 release-readiness 判据)
  - `compute_release_ready_eligible(tasks) = compute_score_eligible(tasks) ∩ {t : is_provenance_eligible(t.meta)}`(denominator-membership 用 helper,**与 release_ready_with_provenance 不同语义**)
  - `projected_eligible(corpus, exclude_r11_pending=True, exclude_r17_high=True)` 默认排 source ② if R11 未落地

## Risks & Dependencies

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `infer/{vllm,sglang}` `safe.directory` 未配 → census/extract fail-fast | High | High | M0a.1 第一步 master-side `git config --add safe.directory` 配置 + verify;不重试 |
| Census M0a 显示 source ① 静态可隔离率 < 30% → 反转主路径 | Medium | High | R13c 决策框架 deterministic;PROVISIONAL 仍可继续(R13b 单独披露) |
| M0b GPU spike 显示 source ② deterministic 率 < 50% → v1 score-eligible 不够 60 | Medium | High | R13c downgrade to PROVISIONAL/PROVISIONAL_MISSING_RARE/STOP per gate def(具体看 size+category+rare coverage 三者);band-mode v2 plan |
| R10 GPU 等价校验:vendored ref 跟 kernel 数学不等价 | Medium | High | R10 FAILED → fallback ②;census 阶段 R17 risk score 预警 |
| `provenance.golden:` 嵌套 schema 与既有 `cost_model` collision(语义重叠) | Low | Low | nested 隔离;不同 sub-key |
| **R15 + R18 signing key 泄露** | Low | Critical | provider-side key escrow + 轮换流程(M5 follow-up plan);v1 单 key |
| **R16 sanity gate bypass**(攻击者绕过 bound) | Low | Medium | sanity bounds per category default + per-task override;sanity FAIL = reject |
| GPU 机不可达 → M2 / M3 / M0b 阻塞 | Medium | High | M0a 可在 dev 机完成;M0b/M2/M3 需 GPU,有 buffer + 协调 |
| Subagent fan-out 资源紧 | Medium | Medium | A5 rule:`free -g < 16 GB` 降串行 |
| 100 task census M0a 单线程慢(50 min serial) | Low | Low | 5 subagent 并发 = ~10 min |
| `_select_tasks` 用 `is_score_eligible` 循环依赖(V2 fix) | Low(已 fix) | High(若未 fix) | M0.0 ship 后,V2 plan 已 settle 不复用此 predicate;新建 `release_ready_with_provenance` |
| R11 `migration_drift_budget` 推导方法学未定 → 影响 R7c gate | High | Medium | R11 前移 M0b 子单元(v1 仅 explicit / default 二态;migration_drift_budget v2 follow-up) |
| R17 reconstructability 论证不充分 → submitter AI-fit 重构 hidden golden | High | Medium | M0a 第 6 维 R17 risk score;高 risk task **leaderboard-exclude 完全**,不进 R13b provisional 也不进 PROCEED 分母 |
| Hidden_shapes knob values 在 submitter tree → submitter 知道 hidden 输入分布 | Medium | Medium | documented Limitation(scope boundaries);knob 不在 R14 scope;只 outputs sealed |
| `oracle_pytorch.run` 在 trusted runner 跑 hidden inputs → T1 channel | Low | High | M4.1 only public shapes;hidden skip AI oracle diagnostic |
| `tests/integration_tests/` 不存在 → V1 path 错 | Resolved | — | V2 用 `tests/unit_tests/` + `@pytest.mark.gpu` |
| M2.1 `_run_capture_child` 签名 V1 描述错(plan claim 不实) | Resolved | — | V2 真实签名 documented + new entrypoint_source value 设计 |

## Phased Delivery Summary

```
Phase 0  M0.0 Authority Triad + M0a.1 Static Census  ~3 d  CPU dev host  ✓ blocking
   └── R17 risk + family taxonomy cleanup + provenance.golden schema
Phase 0b M0b.1 GPU Spike (8-12 stratified)            ~1 d  GPU host (single)
   └── source ② deterministic dynamic evidence
Phase 0c M0a.2 R13c Decision Gate                     ~1 d  CPU dev host
   └── PROCEED | PROVISIONAL | PROVISIONAL_MISSING_RARE | STOP
Phase 1  M1.1 Two-phase Extractor + M1.2 Schema       ~3 d  CPU dev host  ── subagent
   └── vendored upstream_ref.py per task
Phase 2  M2.0 R19 Boundary + M2.1 R10 Equivalence     ~3 d  CPU + GPU     ── subagent (GPU)
   └── oracle_kernel_equivalence VERIFIED/FAILED
Phase 3  M3.0 Sanity + M3.1 Determinism + M3.2 Sign   ~3 d  GPU host + provider
   └── public golden + sealed golden artifacts(GPFS provider-private) + external detached signature `tasks/<id>/provenance.sig` (R18)
Phase 4  M4.1 Eval Replace                            ~2 d  CPU dev host
   └── GoldenStore + R15/R18 verify
Phase 5  M-CLI subcommands + M5.1 follow-up           ~1 d  CPU dev host
```

Total **~17 工作日**(V2 加 R18/R16/R19 提前 ship + M0.0 authority triad + M0a/M0b 拆分 + M-CLI;V1 ~12 d 估算偏低 codex C8/C10 已论证).

## Success Metrics

> *Carries + tightens from origin doc §Success Criteria.*

**过程 / 出处(必要)**:
- Zero **release-ready (release_ready_with_provenance=true)** 任务 golden 仅依赖 AI oracle(R3 / R9 enforce)
- 每 score-eligible 任务可追溯 `provenance.golden.{source, tolerance_source, test_capture_sha, golden_sha256, reviewers, upstream_closure_fingerprint, reconstructability_risk}` + external detached signature `tasks/<id>/provenance.sig` (R8 + R18)
- 评测期零 GPU(`eval.py` 改 golden-load 经 GoldenStore;R15 + R18 verify chain)
- Sealed-hidden golden **输出** 零 submitter 可见性(R14 完全 withhold:tasks 目录 + GPFS `0700 provider`;hidden *inputs* knobs 在 submitter shapes.yaml — documented Limitation)
- 评测期 100% 经 R15 + R18 完整性门(self-attestation loop 修复)
- 对外措辞与 golden 可信度一致:`upstream_test_default` / R10b 未覆盖 / R13b 任务不标"已验证正确性"

**结果(充分)**:
- R13c gate 输出 PROCEED 当 `projected_eligible(exclude_r11_pending=True, exclude_r17_high=True) ≥ 60 task ∧ ≥ 8 category ∧ rare_class_coverage_OK`
- AI oracle vs golden 一致性按 category 作诊断统计披露(`docs/release/ai_oracle_consistency_report.md`)— public shapes only(R14 隔离)
- 100 task corpus 在新 pipeline 跑出的 verdict 跟 baseline 可比对(regression detector)

**新 op onboarding**(终极交付):
- `tab census --task <id>` 1 命令 emit feasibility 分类
- `tab freeze-golden --task <id>` 1 命令完成 R4 + R10 + R5 路径(provider-side)
- 新 op 入仓 ≤ 30 min wall-clock(census + 1 路径 freeze)

## Documentation / Operational Notes

- `docs/triage_backlog/2026-05-27-census-report.md` — M0a.1 + M0b.1 输出(public summary),provider detail 在 `$TAB_GPFS_ROOT/census-detail.json`
- `docs/release/ai_oracle_consistency_report.md` — M4.1 持续披露(public shapes only)
- `specs/task_api_contract.md §5d provenance.golden` — schema 文档,M1.2 加
- `specs/task_api_contract.md §5e sanity_bounds.yaml` — schema 文档,M3.0 加
- `specs/task_api_contract.md §7 security boundaries` — R19 + R18 trust root,M2.0 加
- `docs/architecture/baseline-accuracy-provenance.md` — 新加,描述双源 pipeline + Authority Triad + threat model
- `docs/runbooks/freeze-golden.md` — provider operator runbook,M3.2 落

## Sources & References

- **Origin document**: [docs/brainstorms/2026-05-16-baseline-accuracy-provenance-requirements.md](../brainstorms/2026-05-16-baseline-accuracy-provenance-requirements.md) — R1-R20 完整 carry-forward
- **Supersedes**:
  - [docs/triage_backlog/2026-05-26-automation-pipeline-v2-plan.md](../triage_backlog/2026-05-26-automation-pipeline-v2-plan.md)(完全 deprecated)
  - [docs/plans/2026-05-26-001-feat-baseline-accuracy-provenance-impl-plan.v1-draft.md](2026-05-26-001-feat-baseline-accuracy-provenance-impl-plan.v1-draft.md)(V1 17 MUST-FIX 收集前的 draft)
- **Review rounds applied to V2**:
  - 4 specialist agents:architecture-strategist / data-integrity-guardian / security-sentinel / repo-research-analyst
  - codex deep review round 1(8 additional MUST-FIX)
  - 17 confirmed MUST-FIX 全部 applied
- **Prior art**(framework 复用):
  - [docs/plans/2026-05-23-001-feat-tab-three-machine-automation-plan.md](2026-05-23-001-feat-tab-three-machine-automation-plan.md)
  - [docs/plans/2026-05-23-002-feat-tab-automation-orchestrator-impl-plan.md](2026-05-23-002-feat-tab-automation-orchestrator-impl-plan.md)
  - [docs/architecture/automation-llm-repair-architecture.md](../architecture/automation-llm-repair-architecture.md)
- **Independent ground truth**:
  - [docs/triage_backlog/2026-05-26-truth-inventory.md](../triage_backlog/2026-05-26-truth-inventory.md)
  - [docs/triage_backlog/2026-05-26-100-task-real-attribution.md](../triage_backlog/2026-05-26-100-task-real-attribution.md)
  - [docs/triage_backlog/2026-05-26-3-task-fix-verification.md](../triage_backlog/2026-05-26-3-task-fix-verification.md)
- **SETTLED reviews**:
  - [docs/reviews/2026-05-19-design-v2-golden-policy-review.md](../reviews/2026-05-19-design-v2-golden-policy-review.md)
- **CLAUDE.md constraints**:`/data1/code/agentic-rl/CLAUDE.md`
- **Contract spec**:`specs/task_api_contract.md §6a`

---

**V2 plan 结束。本文档自含;无需补充信息即可起手 M0.0 Authority Triad。**
