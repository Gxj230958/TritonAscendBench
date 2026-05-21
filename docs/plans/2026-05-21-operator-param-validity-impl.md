# 实施方案:算子参数合法性 — 约束声明与静态校验

**Date:** 2026-05-21
**对象:** `triton-ascend-bench`
**依据:** `docs/reviews/2026-05-20-operator-param-validity-gap.md`(差距诊断,codex 7 轮核验收敛)

## 1. 目标

让每个算子的**完整输入契约**(张量维度 / dtype / 取值分布 / 标量配置 / 跨张量不变量)从「散落在 Python 注释与散文里、零静态校验」变为「结构化声明 + validator 静态校验 + eval 路径强制门」,杜绝机械占位参数导致的两类危害:(a) 功能跑不通;(b) 在无意义参数点评测(oracle 与 candidate 同用错参数假性"通过")。

## 2. 关键约束:两类求值域 + 两类 evaluator entrypoint

参数约束按求值对象分两类,**机制不同,不可混用**:
- **knob 约束**(`dims` / `scalars` 值域与派生):对象是 size-knob 整数与标量,纯表达式。
- **物化产物约束**(`tensor_invariants`:`cu_seqlens` 单调、`len(seqused_k)==E`):对象是 `build_inputs` 物化**之后**的张量,knob 表达式无法表达 ⇒ 采用**枚举化 invariant schema**(预定义类型+参数),物化后校验。

knob 表达式本身又分两种,**WA.2 的共享 AST evaluator 提供两个独立 entrypoint**:
- **numeric 表达式**(`eval_numeric`):返回数值 —— `cost_model` flops/bytes 公式、`dims` 维度公式、`derive` 派生标量、`randint` 上下界、`offsets:to=`。**保留 W1.2 `cost_model` 现有 numeric 契约不变**。
- **predicate 表达式**(`eval_predicate`):返回 bool —— `dims` 约束、`scalars` 值域约束;支持比较(`== != < <= > >=`)、布尔(`and or not`)、`in`、list/tuple literal。
两 entrypoint 共享同一 AST 白名单安全核(拒任意代码)。

## 3. 范围与边界

| 类别 | 内容 | 执行位置 |
|---|---|---|
| 代码骨架 | WA.1 schema、WA.2 共享 evaluator(双 entrypoint)、WA.3 标量派生+字面量收口、WA.4 validator 校验+eval 门+fallback 收紧、WA.5 生成器更新 | **dev 机** |
| 长尾 | WB.1 100 题逐题 reconcile | **dev 机**(框架+样板先行,逐题铺开) |

## 4. gap → W 项覆盖映射

| 诊断 gap | W 项 |
|---|---|
| G1 约束零声明 | WA.1(`param_constraints` schema,五子段)|
| G2 零校验 + 静默 fallback | WA.4(validator 校验五子段 + dist 校验 + eval 门 + `build_inputs` fail-loud)|
| G3 knob 网格 99/100 雷同 | WB.1(逐题定算子专属网格)|
| G4 kernel `constexpr` 隐式约束未结构化 | WA.1(`dims` 含 constexpr-derived 约束)+ WB.1 |
| G5 contract 无 machine-checkable 约束 | WA.1(`param_constraints` 机器可校验)+ WA.5 |

## 5. 详细 Work Breakdown

### WA.1 — `param_constraints` schema | gap G1/G4/G5 | 文件 `specs/task_api_contract.md`、`triton_ascend_bench/dataset.py`、`scripts/validate_tasks.py`
`meta.yaml` 新增 `param_constraints` 段,**五子段**:
- `dims`:knob 约束 predicate 表达式列表(`"D in [16,32,64,128]"`、`"H % G == 0"`、`"is_pow2(D)"`、`"L >= 16"`)—— 含 kernel `tl.constexpr`/`BLOCK_SIZE` 推导的维度约束。
- `dtypes`:每个 input 的 dtype 约束(`x: float32`、W8A8 的 `a: int8`)。取值为**封闭 enum**,与 `inputs.py:30-37` 的 builder dtype alias 表绑定为**单一权威**;`validate_tasks` 与 `build_inputs` 共用该表,unknown dtype(typo)被静态拒(配合 WA.4 ③ 的 `build_inputs` fail-loud)。
- `dists`:每个 input 的 **dist 约束** —— 声明该 input 应使用的 dist kind 及其参数(如 `probs: {kind: softmax_rows}`、`seqused_k: {kind: randint, hi: "L"}`、`q: {kind: randn}`);wrong-but-supported 的 dist(如该用 `softmax_rows` 却用 `randn`)由此被静态拒。**表示桥接**:`shapes.yaml.inputs.*.dist` 维持现有 string DSL 形态(`randint:0:E`、`offsets:to=B*L`、`full:expr` —— `inputs.py:57-118`),`param_constraints.dists` 用结构化 `{kind, args}`;二者经**唯一共享 `parse_dist_spec` normalizer**(string DSL ⇄ 规范化 `{kind, args}`)比对 —— `inputs.py` 物化、`validate_tasks` dist 校验、生成器三方均经它,杜绝重复解析/漂移(string DSL 为 `shapes.yaml` 权威书写形态、不迁移)。
- `scalars`:**非张量字面量(`scalar` 与 `scalar_seq`)的约束** —— `scalar` 的值域 predicate(**Python 兼容链式比较**,如 `"0 < eps <= 1e-2"`,不用非 Python-parseable 的数学区间记法 `(0,1e-2]`)或派生规则(见 WA.3);`scalar_seq` 的约束 predicate 支持 `len()`/`sum()`/索引(如 `L1_31` 的 `"sum(mrope_section) == D // 2"`、`L2_10` 的 `"len(block_size) == 2"`)—— list-literal 因此也纳入 machine-checkable contract。
- `tensor_invariants`:**枚举化** invariant 列表,每项 `{type, args}`,type ∈ 预定义集 —— `monotonic_nondecreasing`、`first_eq`/`last_eq`、`length_eq`(`len(x) == <knob-expr>`)、`bounded_above_by`、`index_in_range`。物化后校验。
- `scaffold`:显式布尔字段(`true` = `phase_c_docs` 生成的未 reconcile 草稿)—— 作为 generator/validator/eval **共用的 sentinel**(显式 schema 字段,取代隐式标记,避免跨组件 drift);WA.4/WA.5 一律引用此字段。**不参与 score-eligibility 判定**(score-eligible 集只看 stub / `PHASE-C-MANUAL`,见 WA.4 —— 否则会从分母剔题、subset-escape)。
**requiredness(与 `cost_model` 同分层)**:`param_constraints` 加入 `validate_tasks.REQUIRED_META_TOP_LEVEL`(缺失 ⇒ validator 报错);`dataset.load_task` 对 `param_constraints` **全程 lenient** —— `TaskDef` 新增字段 `param_constraints: dict | None = None`(**带默认值** —— canary adapter / tests 等直接 `TaskDef(...)` 构造点向后兼容、无需逐个改;字段存 raw dict、loader 不解析成结构化对象);缺失 ⇒ 该字段 `None`;**malformed-present(schema 非法 / 约束表达式语法错)⇒ 原样保留 raw dict、parse 期不 validate / 不 raise**(须显式偏离 `cost_model` 在 `dataset.py:111-120` eager-parse-then-raise 的现状)。所有 schema/semantic 校验 defer 给 WA.4 共享 validator/eval helper —— 使 malformed task 能走到 `task_status=blocked` 门,而非在 `load_task` 阶段就崩。spec 明写「loader-lenient / validator-strict」。
**canary 豁免**:`is_canary: true` 的 task(`tasks/_canary/`)**不要求** `param_constraints` —— canary 是 provider 精心设计的陷阱题、复用 source task 的 solution、不进 score-eligible;`validate_tasks` 与 WA.4 eval 门对 `is_canary` task 跳过 `param_constraints` 校验/门(否则会把 W2.5 canary 检测链封死)。
schema 固化进 `specs/task_api_contract.md`。
**DoD**:schema 文档化(五子段 + 枚举 invariant 类型表 + dist-kind 表);`dataset` 解析;`validate_tasks` 把 `param_constraints` 列为 required;1–2 样板任务手写 `param_constraints` 通过校验。

### WA.2 — 共享 evaluator(numeric + predicate 双 entrypoint)| gap G1 | 文件 `triton_ascend_bench/cost_model.py`(或提取为 `expr.py`)、`triton_ascend_bench/inputs.py`
现状:`inputs.py` 有**多处裸 `eval`**(`dims`/`randint`/`offsets`/`full`/`scalar_knob`,`inputs.py:47,79-80,95,116,162-163`),与 W1.2 `cost_model.py` 的 AST 白名单 evaluator 是两套语义。
修法:把 W1.2 的 AST 白名单安全核**提为共享模块**,提供**两个独立 entrypoint**(见 §2):`eval_numeric`(覆盖 `cost_model` 公式 + `inputs.py` 全部 numeric 表达式入口,取代裸 `eval`;**显式支持 derive-time helper `sqrt`/`min`/`max`/`ceil`/`floor`** —— 现 `cost_model` evaluator 仅 `min`/`max`/`ceil`/`floor`,补 `sqrt` 以支撑 WA.3 的 `1/sqrt(D)`;新增 helper 不影响不用它们的现有 `cost_model` 公式,numeric 契约/调用方不变)、`eval_predicate`(约束 bool 表达式 —— 比较/布尔/`in`/list literal/helper `is_pow2`/`sqrt`/`gcd`/`len`/`sum`,后两者支撑 `scalar_seq` 约束)。一套 AST 安全核、两类语义边界明确。
**DoD**:`inputs.py` 无裸 `eval`;`eval_numeric`/`eval_predicate` 各有单测(含非法/不安全表达式静态报错);现有 `cost_model` 调用方回归不变;现有 shapes 公式回归不变。

### WA.3 — 标量派生 + 非张量字面量 schema 收口 | gap(scale=0.125 实证)| 文件 `triton_ascend_bench/inputs.py`、`shapes.yaml` schema、`specs/task_api_contract.md`
① `inputs` 标量项支持派生形 `scale: {derive: "1/sqrt(D)"}`(取代硬编码 `{scalar: 0.125}`),`build_inputs` 物化时用 `eval_numeric` 按 knobs 算。
② **非张量字面量 schema 写死**:`scalar`/`derive` 的值只能是 JSON scalar(`bool`/`int`/`float`/`str`/`None`),与 W1.0 sidecar 的 IPC 类型边界(`inputs.py:190-215`、`process_isolation.py:663-679`)一致;现语料里 `scalar` 是 list 的场景(已知两处:`L2_10` 的 `block_size: [8,8]`、`L1_31` 的 `mrope_section`)迁到独立字段 `scalar_seq`,sidecar/validator 同步支持。
③ `inputs.py` 的 `dist` 词表**按需扩展**:WB.1 reconcile 时若发现某 task 的输入无法用现有 dist 物化(已知:`L1_24` 的 `cu_num_draft_tokens` 是非-leading-zero cumulative,现 `offsets` dist 只支持 leading-zero indptr),补对应 dist 变体(统一走 WA.2 `eval_numeric`);无法通用化的留 `PHASE-C-MANUAL`。
**DoD**:派生标量按 knobs 算出正确值(`D=64⇒0.125`、`D=128⇒1/√128`);`scalar`/`derive` 非 JSON-scalar 值被 validator 拒;`L2_10.block_size`、`L1_31.mrope_section` 迁 `scalar_seq` 且 IPC 跑通;1 个 attention 样板任务 `scale` 改派生形。

### WA.4 — validator 校验 + eval 门 + fallback 收紧 | gap G2 | 文件 `scripts/validate_tasks.py`、`triton_ascend_bench/inputs.py`、`triton_ascend_bench/eval.py`、`triton_ascend_bench/cli.py`、`triton_ascend_bench/eligibility.py`
① **validator 校验**:`validate_tasks` 据 `param_constraints` 五子段校验 —— 每组 public/hidden knobs 满足 `dims`(`eval_predicate`);`inputs` 各项 dtype 满足 `dtypes`;`inputs` 各项 dist kind+args 满足 `dists`;标量满足 `scalars` 值域;`tensor_invariants` 按枚举类型 dispatch、对物化后样本张量校验。违反 ⇒ validator 报错。
② **共享校验 helper + repo-authoritative status resolver**:`eval.py` 直接 `load_task` 评测、不经 `validate_tasks`(只 block `PHASE-C-MANUAL`)—— 仅靠 validator 拒不够,且**不能只看 missing/`scaffold`**:malformed-but-present 的 `param_constraints`(schema 非法 / 约束语法错 / 五子段 semantic 违反)仍会 runnable。修法两层:

  (a) **共享校验 helper**:`validate_tasks`/`eval`/status resolver **共享同一个 `param_constraints` 校验 helper**(pass/fail predicate)。helper **先查 `scaffold == true`** —— 是则**立即**返回 `scaffold` failure(单一干净信号)、**不**再跑 knob/input/tensor 的物化语义校验(否则 scaffold task 的 placeholder `shapes.yaml`(无 `size_knobs`/`inputs`)会触发杂红、污染「仅 scaffold 信号」、矛盾 WA.5 non-regression);否则才做 schema 合法性 + 五子段 semantic 校验(含物化后张量校验)。`validate_tasks` 对 fail 报错(schema 模式,发布流程跑完整 `validate_tasks` 即拦)。

  (b) **repo-authoritative status resolver**:权威 `task_status` 由 `cli analyze` 的 `_authoritative_task_status` 从 repo state 重算(`cli.py:843-903`),**不信** eval result JSON 的自报 status —— 故光改 `eval.py` 不会落进 `score_report.json`。`param_constraints` blocking 判定作为**新增一档接入 `_authoritative_task_status`**(该函数即全部 `task_status` 的唯一权威裁决点 —— **不另起并行 resolver**,避免与现有分散在 `eval._resolve_golden`/`eligibility.compute_verified_blocked` 的判定形成双权威/漂移);`eval` 侧经同一共享 helper 取得一致 status。**控制流要求**:`eval.evaluate()` 须在**任何 early return 之前**调用该 resolver —— 含 static-check `static_report.blocked` return(`eval.py:892-896`)、`gradeable` / oracle-import / input-build short-circuit;现码 `_resolve_golden()` 仅在 `if gradeable:` 块内 reach(`eval.py:919-962`),上述早退路径因此 fall through 到默认 `task_status="evaluated"`(`eval.py:163-165`);修正后**所有** early-return 分支(含 static-blocked、non-gradeable stub/manual/scaffold)须返回 resolver 的 `unsupported`/`blocked`/`evaluated` verdict、不得遗留默认 `evaluated`。裁决按**固定优先级**(高→低,先命中者定):
  1. `is_canary: true` ⇒ 豁免 `param_constraints` 门(canary 复用 source solution、不进 score-eligible);
  2. self-reported mode 下 `upstream_oracle_required` 的 task ⇒ `unsupported`(W4.3 既定:self-reported 模式无 golden、此类 task 故意不在分母 —— `cli.py:868-880`/`scoring.py:37-53`;`param_constraints` 门**不得**把它重分类成 `blocked` 而拉回分母);
  3. verified mode 下 upstream `golden/` 不完整 ⇒ `blocked`(既有 verified-blocked 权威判定,保持不变、优先于 `param_constraints` 档);
  4. 其余 task:`param_constraints` 共享 helper 判 fail(缺失 / `scaffold:true` / schema 非法 / 约束违反)⇒ `blocked` —— 沿用 Roofline 的 blocked 机制,**留在 score-eligible 固定分母、计 ε**(**不**从 score-eligible 集剔除 —— 从分母剔除 = subset-escape,违反 DESIGN P6/P7「BLOCKED 不可通过从分母剔除规避」);
  5. 否则 ⇒ `evaluated`(正常评分)。

  **不改** DESIGN §4.4 二元 `release_status` 谓词(与 Roofline 对 golden 完整性的处理同模式)、**不改** score-eligible 集判定逻辑(仍只看 stub / `PHASE-C-MANUAL`,`blocked` 留分母)。
③ `build_inputs` 的 unknown dtype 静默 fallback `float32`、unknown dist 静默 fallback `randn` 改 **fail-loud**。
**DoD**:违反任一子段的 task 被 `validate_tasks` 报错;`param_constraints` fail 的 non-canary / non-`unsupported` task 在 `score_report.json` 里显示 `task_status=blocked`(**留固定分母计 ε、不剔除**);`is_canary` task 豁免、self-reported `upstream_oracle_required` 保持 `unsupported`、verified-mode golden-incomplete 保持既有 `blocked` 判定(三者均不被 `param_constraints` 门覆盖/干扰);`eval` 与 `cli analyze` 经同一 `_authoritative_task_status` 得出一致 status;`build_inputs` 对 unknown dtype/dist 报错;Phase A 期间全语料 `validate_tasks` 预期 red(WB.1 前),pytest 套件保持绿。

### WA.5 — 生成器更新(`phase_c_docs.py` + `generate_task_scaffolds.py`)| gap G3/G5 | 文件 `scripts/phase_c_docs.py`、`scripts/generate_task_scaffolds.py`(+ `meta.yaml` 写入)
现状:统一 knob 网格来自 `KNOB_DEFAULT`/`PUBLIC_ROWS`/`HIDDEN_ROWS`(`:38-52`),占位标量来自 `_SCALAR_DEFAULTS`(`:62-73`),自动 contract 吐 `TODO(phase-c-review)`(`:588`);且 generator 当前**只写 `shapes.yaml`/`contract.md`,不写 `meta.yaml`**(`:640-653`)。
修法:`phase_c_docs.py` 降级为 **scaffold-only 生成器** ——
- generator 写出的 `param_constraints` 一律 **`scaffold: true`** —— WA.1 定义的这个显式布尔字段是**唯一**逻辑哨兵;`shapes.yaml`/`contract.md` 里若保留 `SCAFFOLD` 文案,只能是展示性注释,**不得**参与任何 gate/skip/eligibility 判定。
- `param_constraints` 在 `meta.yaml`,故 generator **新增 `meta.yaml` 写入能力**:**merge** —— 只新增/更新 `param_constraints` 段,保留 `meta.yaml` 其它字段;**meta-skip** —— **仅当** task 的 `param_constraints` 字段**已存在且 `scaffold == false`**(= 已人工 reconcile)时,generator 跳过 **`meta.param_constraints` 的写入**、绝不覆盖人工内容;`param_constraints` **缺失**(存量旧 task 普遍如此 —— 正是待播种对象)或 `scaffold == true` 的 task **一律 merge-写入 `scaffold: true` 骨架**,不可误判为 skip。此 meta-skip **只保护 `meta.param_constraints`**,**不**连带跳过 `contract.md` 约束镜像的刷新(见下条)。该 seeding / contract-mirror 路径**不套用** `phase_c_docs.py` 既有的 `SKIP_POC`(`:31-34,600-603`)与 `skip-stub` 早退(`:606-607`)—— 这两个早退原仅为不重生成 PoC / stub-oracle task 的 `shapes.yaml`/`contract.md` 主体;`param_constraints` 骨架 seeding 是新增能力、且 `param_constraints` 属 `REQUIRED_META_TOP_LEVEL`(缺失即 validator 报错),故须对**全部 100 题**(含 5 个 PoC 与所有 stub-oracle 旧 task)生效(现皆无 `param_constraints`)。
- generator 当前 via `phase_c_docs.py:190-191` 把 list 值 emit 到 `scalar` 下 —— 改为 emit `scalar_seq`(与 WA.3 字面量 schema 一致),不再产出 schema/sidecar 不接受的 list-under-`scalar`。
- **`contract.md` 约束镜像**:`_emit_contract_md` 当前对约束段只吐 `TODO(phase-c-review)` 占位(`:588`)—— 改为从 `meta.param_constraints` 渲染**结构化约束镜像段**(五子段的人读呈现),使 G5「contract 有 machine-checkable 约束」真正闭合;`meta.yaml` 是约束的**唯一权威源**,`contract.md` 段为其纯派生渲染镜像(无人工内容)。镜像段用显式 marker 包裹,generator **始终**对该 section 做**定点 rewrite**(从当前 `meta.param_constraints` 重渲染),**不**因 `meta-skip`(`scaffold == false`)而跳过 —— 否则 WB.1 把 task reconcile 后 `contract.md` 镜像会永久停在旧值/`TODO`;定点 rewrite 只动 marker 内 section,保留 `contract.md` 其它内容。**marker 首次落位 + 旧段迁移**:若 `contract.md` 尚无 marker(存量语料皆如此),首次落位时 —— 若该 `contract.md` 已存在旧**手写**约束 section(如 `L3_04_fwd_kernel/contract.md:49` 起的 `## Constraints`),**原位替换/删除旧约束段**、由 marker-镜像段取代;若无旧约束段则以 marker 包裹 append 到文末。两路径都保证 `contract.md` 全程只有**一个**由 `meta.param_constraints` 渲染的权威约束 section、无新旧双权威。此后均定点 rewrite marker 内 section(`generate_task_scaffolds` 新建的 `contract.md` 直接含 marker)。
- WA.4 validator + eval 门据 `param_constraints.scaffold == true` 判「未 reconcile」拒收。
- **另一生成器 `scripts/generate_task_scaffolds.py`**(生成新 task scaffold、当前 emit `meta.yaml` 不带 `param_constraints`、**也不带 W1.2 已设为 required 的 `cost_model`**)同样须更新:新 scaffold 的 real task `meta.yaml` emit schema-conforming 的 `param_constraints` 块(`scaffold: true` + 五子段空/默认占位),**并顺带补 `cost_model` 占位块**(补 W1.2 把 `cost_model` 设 required 时遗漏的生成器更新)—— 占位块须是**现有 `validate_tasks` 可通过的常量型合法块**(常量公式、不引用尚不存在的 `size_knobs`,配合法 `flops_class`/`bytes_resource` 枚举值),使 `cost_model` 占位**不新增** validator 红 —— 新 scaffold task 仍会因既有 gate(`validate_tasks._validate_golden` 的 upstream-golden 缺失、`param_constraints.scaffold == true`)而红,这些是预期 unreconciled 信号、非 WA.5 引入的回归;**且同样把 `param_constraints` 渲染成 `contract.md` 镜像约束段**(对称于 `phase_c_docs.py` 的处理 —— 否则新 scaffold task 的 `contract.md` 停在 TODO-only,且 `phase_c_docs.py` skip stub task、不会二次补镜像,G5 对新 scaffold task 漏覆盖)。**承诺收窄**:这两个子块本身 schema-conforming(validator 不因「必填顶层段缺失 / 字段类型错」报错);但 `param_constraints.scaffold == true` 本身即 `validate_tasks` 报错项、`eval` 判 `task_status=blocked` —— 这是预期(reconcile 完成度 red,与 `cost_model` 的 Phase 1-2 red 同理,WB.1 后转 green);**不**宣称新 scaffold 整体 validator-clean / runnable。该生成器产出的 `shapes.yaml` 仍是非 executable 占位形式(无 `size_knobs`/`inputs`),WA.5 **不**负责升级它 —— 这类 scaffold task(`oracle_simplified: upstream_oracle_required` + stub oracle)按 WA.4 ②(b) 优先级 self-reported 先命中档② `unsupported`、verified 先命中档③ upstream-golden `blocked`(**先于** `param_constraints` 档④),本就不可评测、不会走到 WA.4「物化后语义校验」;`param_constraints.scaffold` 仅保证 validator 报错及非上述优先级路径下的阻断(后续测试不可按「scaffold 主导 blocked」写断言)。`shapes.yaml` 转 Phase-C executable schema 属 WB.1 逐题 reconcile 范畴。
**DoD**:phase_c 产出的 `param_constraints` 带 `scaffold: true`;phase_c 能 merge-写 `meta.yaml` 的 `param_constraints` 段而不破坏其它字段;phase_c **对 `param_constraints` 已存在且 `scaffold == false` 的 task 跳过 `meta.param_constraints` 写入**(保护人工内容),对**缺失**或 `scaffold == true` 的 task(**含 5 个 PoC**)merge-播种骨架;`contract.md` 约束镜像段对**所有 task**(含已 reconcile 的)始终定点 rewrite、不再是 `TODO(phase-c-review)` 占位;`generate_task_scaffolds` 新建的 real task 带 schema-conforming 的 `scaffold: true` `param_constraints` 块、`cost_model` 常量型合法占位块(`validate_tasks` 公式校验可通过)、及 `contract.md` 渲染的约束镜像段;WA.5 对生成器的改动**不新增** validator 红(`param_constraints`/`cost_model` 子块 schema-conforming),新 scaffold task 既有的 upstream-golden gate 红(`_validate_golden`)与 `param_constraints.scaffold` 红保持不变 —— 均为预期 unreconciled 信号(`scaffold` 红经 WB.1 逐题 reconcile 转 green;upstream-golden 红由 W4.3 离线 golden 关闭,不在本方案范围);validator/eval 据 `scaffold: true` 拒收。

### WB.1 — 逐题 reconcile(长尾)| gap G3/G4 | 文件 `tasks/level*/`(100 题 `meta.yaml`/`shapes.yaml`)
100 题对照算子真实输入契约:① 填 `param_constraints`(五子段);② 修占位标量(`scale` 改派生形、`eps`/`block_size` 对齐上游);③ 修错 dtype/dist;④ 定**算子专属 knob 网格**(打破 99/100 雷同 —— attention `head_dim` 取 `{16,32,64,128}`、matmul `K` 对齐);⑤ 移除 `SCAFFOLD` 标记。
策略:可机械化的(elementwise/reduce 维度约束、scale 派生)批量起草;融合算子(attention/recurrent)逐题人工对照 `kernel.py` 的 `tl.constexpr` 与 oracle 签名;先 1–2 样板任务端到端跑通再铺开。
**DoD**:每个 task `meta.yaml` 含通过 `validate_tasks`、无 `SCAFFOLD` 标记的 `param_constraints`;knob 网格不再 99/100 雷同(已按 13 个算子族打散);占位标量修正。**`L1_24`(已 resolve)**:其 `cu_num_draft_tokens` 是非-leading-zero cumulative,已按 WA.3 ③ 补 `inputs.py` 的 `cumcount` dist 变体使其可物化、忠实 reconcile 转 green。**`L1_21_gumbel_sample_kernel`(acknowledged 特例)**:其 `oracle_pytorch.py` 是未实现 stub(`raise NotImplementedError`)⇒ `eligibility.is_score_eligible` 判其 non-score-eligible、**不进评分分母**;Task #20 范围内无真值 oracle 可对照、无法忠实 reconcile,故保留 `param_constraints.scaffold: true` + `PHASE-C-MANUAL` marker,待该 task 补 oracle(另行工作项)后再 reconcile —— status resolver 对它判 `unsupported`(self-reported)/`blocked`(verified),均为既有 stub-oracle 路径既定行为,`param_constraints` 不污染评分。总 DoD:WB.1 完成后,**`param_constraints` 相关的 `validate_tasks` 校验与 `param_constraints` eval 门**对**除 `L1_21` acknowledged 特例外的全部 task** 转 green(不再有缺失 / malformed / 约束违反的 `param_constraints` 红;`scaffold:true` 仅余 `L1_21` 一题、且其 non-score-eligible);**不含** W4.3 既有的 upstream-golden gating —— self-reported 模式下 `upstream_oracle_required` task 仍按设计保持 `unsupported`、verified 模式下缺离线 `golden/` 仍 `blocked`(W4.3 既定,不在 WB.1 范围)。

## 6. 阶段顺序与依赖

```
WA.1 schema ──┬─ WA.2 共享 evaluator(双 entrypoint;WA.3/WA.4 依赖)
              ├─ WA.3 标量派生 + 字面量收口(依赖 WA.2)
              ├─ WA.4 validator 校验 + eval 门 + fallback(依赖 WA.1/WA.2)
              └─ WA.5 生成器降级(依赖 WA.1 schema)
WB.1 逐题 reconcile(依赖 WA.1–WA.5 全部,长尾)
```

## 7. 测试策略

- 每个 WA 项配单元测试;**0 skipped**。
- Phase A(WA.1–WA.5)期间全语料 `validate_tasks`/eval 门的 `param_constraints` 校验预期 red(100 题未填/带 `SCAFFOLD`),用 `--task` 验证逻辑;WB.1 后转 green。
- 不破坏现有 904 passed 测试套件。

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| `tensor_invariants` 难纯静态校验 | 枚举化 invariant schema + 物化后样本校验,不引入通用张量表达式 evaluator |
| 统一 evaluator 取代 `inputs.py` 裸 `eval` 影响面广 | WA.2 先做、先冻结双 entrypoint API;`cost_model` numeric 契约不变 + 现有 shapes 公式回归测试兜底 |
| WB.1 100 题逐题 reconcile 抗机械化长尾 | 批量起草可机械化部分 + 融合算子逐题人工;先样板端到端;`SCAFFOLD` 标记 + eval 门确保未 reconcile 的不被误当权威/不可评测 |
| 算子真实输入契约需对照冻结上游 | 逐题对照 `kernel.py` `tl.constexpr` 与 oracle 签名;不确定标 TODO 待 GPU 机核 |

## 9. 交付物清单

- 代码:`dataset.py`/`cost_model.py`(或 `expr.py`)/`inputs.py`/`validate_tasks.py`/`eval.py`/`cli.py`/`eligibility.py`(或新 status-resolver helper)/`scripts/phase_c_docs.py`/`scripts/generate_task_scaffolds.py` 改动(WA.4 改 `eligibility.py` 仅为加 `param_constraints` blocking status resolver —— **不**改 score-eligible 集判定,`blocked` 留分母不剔除)
- 新增:WA 各项测试
- 数据:100 题 `meta.param_constraints` + 修正的 `shapes.yaml`(派生标量、算子专属 knob 网格、`scalar_seq` 迁移)
- 文档:`param_constraints` schema(五子段 + 枚举 invariant 类型表 + dist-kind 表)固化进 `specs/task_api_contract.md`
