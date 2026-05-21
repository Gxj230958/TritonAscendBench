# 算子参数合法性 — 差距诊断

**Date:** 2026-05-20
**对象:** `triton-ascend-bench`(GPU Triton → 昇腾 910B NPU Triton 迁移评测基准)
**核心问题:** benchmark 怎么为每个算子选择合法的输入参数?不合法参数会让功能跑不通、或在无意义参数点评测。

## 背景

每个算子(task)的**完整输入契约**包含五类参数:① 张量维度(`shapes.yaml` 的 size-knob:B/L/D…);② dtype;③ 标量配置(`scale`/`eps`/`causal`/`softcap`/`block_size`…);④ 张量取值分布(`dist`);⑤ 跨张量不变量(`cu_seqlens` 单调首0末total、`seqused_k ≤ kv_len`、`alibi_slopes` 长度=head 数、`H % G == 0`…)。

每个算子对这些参数有真实约束。不合法参数的危害有两层:**(a)** 功能直接跑不通(维度不对齐、constexpr 约束违反);**(b)** 更隐蔽 —— 功能跑通但参数无意义,oracle 与 candidate 同用错参数会"自洽地通过"correctness,benchmark 测的不是算子真实行为。

## 盘点方法

读 `scripts/phase_c_docs.py`(shapes 生成)、`triton_ascend_bench/inputs.py`(dist 物化)、`scripts/validate_tasks.py`(校验)、抽样 8–10 个不同类型 task 的 `shapes.yaml`/`meta.yaml`/`contract.md`/`kernel.py`;对 100 个 `shapes.yaml` 的 knob 网格做哈希统计;全语料统计 `dist` 用量。

## 一句话结论

算子参数合法性当前**零声明、零静态校验**:`meta.yaml` 无任何参数约束字段;约束散落在 `phase_c_docs.py` 的 Python 注释和少数手写 `contract.md` 的散文里,不可机器消费;`validate_tasks` 不检查任何参数合法性。**100 个 task 中 99 个共用同一套机械 knob 网格**,无法表达算子专属维度约束;attention `scale` 全部硬编码 `0.125` —— 在 code/contract 明确 `scale=1/√head_dim` 的子集 task 上与实际 head_dim 系统性不一致,其余只 consume caller `scale` 的 task 属代表性风险。

## 现状:参数合法性靠什么保证

| 参数类 | 现状机制 | 静态校验 |
|---|---|---|
| 张量维度 | `phase_c_docs.SHAPE_OVERRIDES`(~60 个手写 per-task spec)+ `inputs.py` size-knob 公式求值;某 input 缺 shape annotation 时 per-input fallback `[B,D]` | ❌ 无 |
| dtype | 键名子串启发式(`_INT_HINT`:含 `_id`/`cu_`/`offsets`… ⇒ int64,否则 float32) | ❌ 无 |
| 标量值域 | `_SCALAR_DEFAULTS` 按键名机械默认(`scale→0.125`/`eps→1e-5`/`block_size→4`)+ override 手写 | ❌ 无 |
| 取值分布 | 选对 `dist` 隐式满足(`offsets`/`randint`/`arange`/`tril`/`softmax_rows`);其余 `randn`/`rand` | ❌ 无(选错 dist 静默通过) |
| 跨张量不变量 | 只有窄子集被 encode:`offsets:to=`(cu_seqlens 单调+总量对齐)、`full:`、`scalar_knob`(标量绑 knob,`inputs.py:161-163`,如 `L1_35`/`L1_37`/`L2_26`);`seqused≤kv_len` 靠 `randint:1:L`。无通用 invariant schema | ❌ 无声明、无校验 |

> 量化:100 个 `shapes.yaml` 的 `size_knobs+public_shapes+hidden_shapes` 哈希 —— **仅 2 种 distinct 网格,99/100 字节完全相同**(由 `phase_c_docs.py:38-52` 三个写死常量产生)。全语料 **333 个输入张量中 222 个用无约束的 `randn`/`rand`**,其余 111 个用带约束意识的 dist(`offsets`/`randint`/`arange`/`tril`/`softmax_rows`/`full`/`ones`)。

## 差距全景

| # | gap | 证据 | 置信度 |
|---|---|---|---|
| G1 | **零声明** —— `meta.yaml` 无 `*_constraint`/`invariant` 字段;约束散落 `phase_c_docs.py` Python 注释(如 `L3_03` 注释 "ngroups==nheads keeps H%G constraint satisfied")、手写 `contract.md` 散文,不可机器消费 | `validate_tasks.py:64-78` `REQUIRED_META_TOP_LEVEL` 18 键无参数约束;`interface.stride_layout` 是自由文本(`"TODO(scaffold)"` 等) | [confirmed] |
| G2 | **零校验 + 静默 fallback** —— `validate_tasks` 不检查任何参数合法性(维度对齐/标量值域/张量间约束),`shapes.yaml` 写错 dims 或值域 validator 全绿;更糟:`build_inputs` 对 unknown dtype 静默 fallback `float32`(`inputs.py:56`)、unknown/不支持的 dist 静默 fallback `randn`(`inputs.py:118`)—— spec typo 退化成「plausible-but-wrong」的输入而非大声失败 | `validate_tasks.py:734-869` 检查项无一涉及 `inputs` 合法性;`_validate_cost_model` 只校验 cost_model 公式可求值 | [confirmed] |
| G3 | **knob 网格 99/100 雷同** —— elementwise/attention/matmul/quant/recurrent 共用同一套 `B=2 L=8 D=16 H=3 …` 网格,无法表达算子专属约束(head_dim=64、H%G==0、D 为 2 的幂) | `phase_c_docs.py:38-52` `KNOB_DEFAULT`/`PUBLIC_ROWS`/`HIDDEN_ROWS` 写死常量;100 题哈希 99 相同 | [confirmed] |
| G4 | **kernel 隐式约束未结构化** —— `@triton.jit` 的 `tl.constexpr`/`BLOCK_SIZE`(head_dim padding、page size、滑窗、int32 溢出门限)隐式约束 shape,无任何地方结构化记录,靠默认值碰运气 | `L3_04/kernel.py:41-46`、`L2_14/kernel.py:70-78`、`L3_05/kernel.py:67-110` 大量 constexpr;`meta.yaml`/`shapes.yaml` 均无记录 | [confirmed] |
| G5 | **contract.md 无 machine-checkable 约束** —— 手写型有散文 `## Constraints`(不可机器消费);phase_c 自动生成型有 prose semantics/helper math 但**无结构化可校验约束段**,`## Metamorphic checks` 纯 `TODO(phase-c-review)` 占位 | `phase_c_docs.py:547-597` `_emit_contract_md`;`L3_04` 手写 vs `L2_14` 自动生成对比 | [confirmed] |

## 不合法参数实证清单

| 参数 | 现状 | 问题 | 置信度 |
|---|---|---|---|
| attention `scale`/`sm_scale` | 全硬编码 `0.125`(全语料 11 处) | `0.125=1/√64`。在 code/contract 明确 `scale=1/√head_dim`(或 `1/√K`/`1/√last_dim`)默认规则的 **8 个 task**(`L3_04` oracle/contract、`L3_02` kernel default、`L3_14`/`L3_15` kernel docs,以及 kernel.py 同写该默认的 `L3_05`/`L3_16`/`L3_17`/`L3_19`)上 —— head_dim 跟通用 `D` knob 走(`D∈{8,16,64,128,256}`)、几乎从不=64 ⇒ scale 与 head_dim 系统性不一致 **[confirmed]**;其余只 consume caller-supplied `scale`、不在任何 artifact encode 该规则的 task(`L2_33`/`L3_01`/`L3_06`)⇒ 代表性风险 **[likely]**。两类都因 oracle 与 candidate 同用同一 `scale`,`qk_scale_inverse` metamorphic 抓不到(它检相对等价、与 scale 绝对值无关) | [confirmed] 8 task / [likely] 其余 |
| `eps` | `phase_c._SCALAR_DEFAULTS` 机械默认 `1e-5`(语料中另有 PoC/手写的 `1e-6`,如 `L1_12`) | 不同 norm 算子上游 eps 不同,机械默认很可能与冻结上游不符 | [likely] |
| `block_size`/`group_size`/`page_size` | 多数机械 `4`/`8` 标量占位(例外:`L2_10` 是列表 `[8,8]`) | 与算子真实 tiling/分组语义无关;若与 kernel constexpr 默认不整除则功能失败 | [likely] |
| fallback dims `[B,D]` | 某 input 缺 shape 标注时 per-input fallback(`phase_c_docs.py:163-166`) | 走此 fallback 的张量维度结构很可能与真实输入契约不符;fallback 机制 [confirmed],但本诊断未逐题核出具体错例,故风险标 [likely] | [likely] |
| `alibi_slopes` dist=`rand` | `L3_05` | `rand` 满足"长度=H"且数值上可跑(L3_05 oracle/kernel 按给定 slope 用、未强制几何序列);但 `rand` 不是 model-typical 的几何斜率序列 `2^{-8i/H}` —— 属**参数真实性/代表性**问题(在无意义参数点评测),非严格"不合法" | [likely] |
| int 张量误用 `randn`/`rand`(**latent**) | `inputs.py:121-124` 对 int dtype + `randn`/`rand` 取 `abs()*8` 转 int | 索引值域失控、可越界 —— **当前 corpus 0 实例**,是 code-path 风险非实证问题;fallback 机制 [confirmed],风险 [likely] | [likely] |
| `L1_24` 输入集 | phase_c 仍生成 `shapes.yaml`/`contract.md`、但追加 `PHASE-C-MANUAL` 警告(`phase_c_docs.py:623-653`) | 非-indptr 卷积约定 `build_inputs` 无法自动生成自洽输入,被显式标记「需手工 author、当前不可安全启用」 | [confirmed] |

## 修复方向(诊断阶段仅列出)

1. **结构化约束声明**:`meta.yaml` 加 `param_constraints` 段 —— 维度约束(`D in [16,32,64,128]`、`H % G == 0`、`D` 为 2 的幂)、dtype 约束、标量值域/派生(`scale = 1/sqrt(head_dim)`)、跨张量不变量。可复用 W1.2 的 AST formula evaluator。
2. **validator 校验**:`validate_tasks` 据 `param_constraints` 校验每组 public/hidden knobs + `inputs` spec,不合法**静态拒绝**(事前,而非运行时跑不通才发现)。
3. **算子专属 knob 网格**:打破 99/100 雷同,逐算子按真实约束定合理 knob 网格。
4. **标量随维度派生**:`scale` 等不硬编码,声明为随维度派生(`1/sqrt(head_dim)`)。
5. **逐题 reconcile**:100 题对照算子真实输入契约填约束、修占位标量/dtype/dist —— per-task 长尾。

## 局限性

- 盘点重点在 shapes 生成/物化/校验链;`kernel.py` 的 constexpr 约束仅抽样确认存在,未逐题枚举。
- "不合法参数清单"中 `eps`/`block_size`/`alibi_slopes` 标 [likely] —— 确证需对照每个算子的冻结上游 kernel,本诊断未逐题核对。
- 结构化约束声明 schema 的具体设计、与 W1.2 cost_model evaluator 的复用边界,留待方案阶段。
