# triton-ascend-bench 验证体系合理性评估

date: 2026-05-31
purpose: 供外部专家评审当前 bench 自动化流程对"抽取算子与上游 vllm/sglang 对齐"的保证强度
method: 只读源码审计(Claude 通读验证链路 + Codex gpt-5.5 对抗式独立审计),无代码改动
scope: 100 个从 vllm/sglang 抽取的 GPU Triton kernel task 的 PASS 判定可信度

---

## 0. 待评审的核心命题

bench 声称:某个 task 标为 PASS,意味着"抽取出来的 standalone 算子,在精度和正确性上完全等同于它在原始 vllm/sglang 中的行为"。

本文把这个复合命题**拆成两个独立子命题**,分别评估当前机制对每个子命题的保证强度:

| 子命题 | 含义 | 谁来保证 |
|--------|------|---------|
| **C1 抽取忠实性 (faithfulness)** | 抽取/改写后的 standalone kernel,数值行为与原始上游 kernel **逐位一致** | S2 验证(`phase_a_prime.py`) |
| **C2 数学正确性 (correctness)** | kernel 计算的数学,与算子的参考实现(oracle)一致 | oracle-consistency 锚点(`resolve_upstream_entrypoints.py`) |

**关键结论先行**:当前机制对 **C1 在"当前枚举 shape + seed=0 + manifest 捕获的张量输出/alias/mutated-input 表面"上保证极强**(GPU 逐位比对),对 **C2 是经验性锚点**(bench 源码原文标注为 "a strong empirical anchor, NOT a formal impossibility proof",见 `resolve_upstream_entrypoints.py:11-13`;`_pa_oracle_check_runner.py:35-36` 有同义句,但首字大写为 "STRONG"),存在已知的假阳入口。两者是**不同的验证阶段、不同的判据**,不能混为一谈。

> 注:本文用"经验性锚点"是对 C2 的**定性**;源码原文用词是 "strong empirical / strong dual anchor",并明确补一句 "NOT a formal impossibility proof"。下文凡标注为源码 quote 处均与原文一致。

---

## 1. 验证体系总览(两层 + 完整性 gate)

```
                抽取算子可信度 = C1(忠实) ∩ C2(正确) ∩ 完整性gate

 ┌─────────────────────────────────────────────────────────────┐
 │ Layer 1  S2 抽取忠实性  (phase_a_prime.py :: s2_verify_task)  │
 │   severed(抽取改写版) vs original(原始上游版) GPU 逐位比对    │
 │   判据: torch.equal 无ULP + dtype/shape/stride + alias        │
 │         + mutated-input + S3 5次determinism                   │
 │   产出: phase_a_prime.status = PASS / BLOCKED / UNRUN          │
 └─────────────────────────────────────────────────────────────┘
 ┌─────────────────────────────────────────────────────────────┐
 │ Layer 2  oracle 正确性锚点 (resolve_upstream_entrypoints.py)  │
 │   原始上游 wrapper 输出 vs PyTorch oracle 输出                 │
 │   判据: tolerance_class (atol/rtol/bit_exact), >=2 个 shape    │
 │   产出: RESOLVED / BLOCKED(no-upstream-entrypoint /           │
 │         oracle-mismatch / shapes-incomplete /                  │
 │         upstream-not-invocable-here / smoke-fail /             │
 │         resolver-error) / UNRUN(needs-gpu)                     │
 └─────────────────────────────────────────────────────────────┘
 ┌─────────────────────────────────────────────────────────────┐
 │ Gate     完整性绑定 (validate_tasks.py)                       │
 │   standalone_sha256 必须 == 磁盘 kernel.py 的 hash            │
 │   防止 "PASS metadata 陈旧 / 与 artifact 不一致"              │
 └─────────────────────────────────────────────────────────────┘
```

---

## 2. Layer 1 — S2 抽取忠实性(对 C1 的保证)

### 2.1 机制(源码确认)

S2 的核心是 **两侧 GPU 真跑 + 逐位比对**(`phase_a_prime.py::s2_verify_task`):

- **original 侧**:加载 task 目录里**未修改的** `kernel.py`,把 `infer/vllm` 与 `infer/sglang/python`(上游源码树)放入 `sys.path`,用 `meta.oracle.gpu_triton` 指定的上游 positional callable 调用。即真实接触上游运行时(`_pa_subproc_runner.py:642`)。
- **severed 侧**:从同一份 `kernel.py` 读源码,经 `rewrite_imports_to_standalone()` 改写 import 为 standalone,追加合成的 `run(inputs)`,写入临时 staged 文件(`phase_a_prime.py:5517`)。**severed 不编辑保留的 kernel 数值逻辑**;它做的是 import rewrite、shim/vendor redirect、必要时把整模块 vendored `kernel.py` 切片到 graded entrypoint 的依赖闭包并丢弃 dead siblings/imports(closure slicing,`phase_a_prime.py:3774,3938`)(Codex [confirmed])。
- **比对**:`s2_smoke_compare` / `compare_s2_manifests`(`phase_a_prime.py:1488,1565`)断言两侧输出的:
  - 输出 key 集合相同
  - 每个输出 `dtype` / `shape` / `stride` / `storage_offset` 相同
  - `torch.equal` **逐位相等**(注释明确 "NO fp32-cast escape" / "no ULP tolerance in S2")
  - **alias 关系**一致(哪些输出共享底层 storage buffer,`_alias_groups`,以 `untyped_storage().data_ptr()` 为键)
  - **mutated-input** 一致(输入被原地修改后的值+layout)

### 2.2 强度加固

- **唯一 PASS 来源是真 GPU**:S2-smoke(CPU / `TRITON_INTERPRET=1`)只做预筛,**永不产生 PASS**;无 CUDA 时诚实返回 `UNRUN(needs-gpu)`(`phase_a_prime.py:2080`)。subprocess 在 `device='cuda'` 请求下拒绝静默退回 CPU tensor(`_pa_subproc_runner.py:804`,关键判断 `if device.startswith("cuda") and not torch.cuda.is_available()`)。
- **S3 确定性**:PASS 前 severed 侧重跑 5 次,每次必须与首次逐位一致,否则 BLOCKED,无 tolerance 后门(`phase_a_prime.py:2112`)。
- **severed 侧禁止 vllm/sglang runtime leak**:若 severed 仍 import 到上游运行时,标 `severed-leak` BLOCKED。

### 2.3 这层保证什么、不保证什么

- **保证(C1,极强,但有明确边界)** [confirmed]:在**当前枚举的 shape、seed=0、manifest 所捕获的张量输出 / alias / mutated-input 表面**上,抽取/改写没有引入可观察偏差。LLM 若在改写中引入哪怕 1 个 bit 的偏差,GPU 比对立即 mismatch → BLOCKED。**这是防"LLM 幻想"最硬的一道闸**。注意它**不是全输入空间的等价证明**,也**不覆盖非张量 side effect / 全局状态 / registry / RNG 状态**(除非这些反映到输出张量)。
- **不保证** [confirmed]:它是**同源行为等价**,不是独立正确性证明。original 与 severed 的 `@triton.jit` 核函数体很可能来自**同一份** `kernel.py`(只是 import/wrapper 不同),所以 bit-exact 在很大程度上是结构性成立的——它证明"severing 没改行为",**不证明"这份 kernel 就是正确的上游算子"**。

### 2.4 original 侧的信任根边界(最关键:S2 基准本身的可信度)

S2 把 severed 跟 original 比 bit-exact,**original 侧就是整个 C1 的参照基准——它若跑偏,S2 名存实亡**。original 侧的可信度分三段(强度递减):

**① 强保证 [confirmed]:跑的是上游原文代码**
- `_pa_subproc_runner.py:642` 把 `infer/vllm` + `infer/sglang/python`(vendored 上游源码树)insert 到 `sys.path` 最前
- 加载 task 的 byte-frozen `kernel.py`,其内部 `from vllm.X import Y` 解析到 `infer/` 真上游源码;relative import 经 `Source path:` header 桥接到 upstream sibling(`_pa_subproc_runner.py:508`)
- entrypoint 来自 `meta.oracle.gpu_triton` 解析的 `spec["entrypoint"]`(不是默认 `run`)
- → "执行的代码 = 上游原文" 这件事可信

**② 弱保证 [confirmed,两处声称被 Codex 实证降级]:shim 数值中立**
original 侧装 4 个 shim 让其在无完整运行时环境跑起来。逐个数值影响:
- `install_vllm_c_stub`:**fail-loud** ✓(`vllm_runtime_hygiene.py:123`,访问缺失属性 raise `AttributeError`,不伪造 compute op)
- `install_vllm_capability_probe_stubs`(`vllm_runtime_hygiene.py`,快照 `:444`):缺失的 cutlass bool probe → `False`。**注释自称 "dispatch-only" 过强** [Codex confirmed]:这些 probe 在上游确实影响 fp8/block-fp8/fp4/group-gemm 支持判断与量化 config 选择(实证 `infer/.../w8a8_utils.py:41`、`fp8.py:274`)。**当前 corpus likely 不踩**:被测 task 多走 `_pa_launcher.py::run` 直接 launch Triton JIT、绕过 host dispatch(如 `L2_11` 虽 import 量化 utils 但 entrypoint 走 launcher),但这是 corpus/entrypoint 条件化结论,**非全局保证**
- `vllm.triton_utils` forward shim:**非严格 verbatim** [Codex confirmed]:强制 `HAS_TRITON=True`、replace `triton.autotune`(`vllm_triton_utils_shim.py:74,108`);GPU 有 driver 时委托真 autotune,无 driver/interpret 时选 first config(bench 定义的 normalized fallback)
- `install_vllm_source_version_shim`:metadata-only(`vllm_runtime_hygiene.py:82`),likely 不入数值路径(除非上游按 vLLM version 分支)

**③ 信任根缺口 [confirmed]:执行环境 ≠ 真实 stock vLLM 安装**
- **无 stock / full-install differential 验证**:original 侧是 "shim-normalized contract",不是装了完整 vLLM + 编译 `_C` 的生产环境
- **共模偏差 S2 结构上抓不到**:若 original 因 `_C` 缺失 / probe=`False` 选了某条路径,而 severed/launcher 复现同一条路径,**S2 bit-exact 照样 PASS**——它发现不了"真实 stock vLLM 在编译 `_C` / 真 CUTLASS probe 下会选另一条数值不同的路径"
- **唯一独立交叉验证 = oracle 锚点(C2)**,但它经验性(≥2 shape,源码自评非形式证明),且 oracle runner 也调 `install_vllm_capability_probe_stubs` 等同一组 hygiene shim(`_pa_oracle_check_runner.py::main`,快照 `:758-770`)

**诚实定性**:original 侧"跑上游原文代码"强保证可信;"执行环境完全等同真实 vLLM 生产"是**弱保证 + 两处声称被实证降级**。**最大残余风险在量化 / CUTLASS / fused-MoE 类 task**——若其 entrypoint 走 host dispatch 而非直 launch triton,shim-normalized 路径可能偏离 stock 环境,且 S2 抓不到。

---

## 3. Layer 2 — oracle 正确性锚点(对 C2 的保证)

### 3.1 机制(源码确认)

correctness 不在 S2 PASS 判定里 [confirmed:Codex+Claude]。`s2_verify_task` 全程不调用 oracle;oracle-consistency 在**独立的 resolver/anchor 阶段**(`resolve_upstream_entrypoints.py`):

- 调用解析出的上游 wrapper,与 `oracle_pytorch.py` 的 PyTorch 参考输出比对(`_pa_oracle_check_runner.py`)。
- 判据用 `tolerance_class`(`tolerance.py::compare`):float 转 fp32 后 `allclose(atol,rtol)`,integer 默认 bit-exact,部分 quant class 用 `int_atol`。**不是 S2 那种逐位**。
- **只抽取少量 shape**:smallest + median 等 `>=2` 个 shape(`_pa_oracle_check_runner.py:81`)。

### 3.2 bench 源码自己的诚实声明

源码原文(逐字 quote,已对照核实):
- `resolve_upstream_entrypoints.py:11-13`:**"this is a strong empirical anchor, NOT a formal impossibility proof"**
- `resolve_upstream_entrypoints.py:33-35`(完整表述):**"this is a strong dual anchor (oracle-consistency on >=2 shapes + structure/dtype match + a real S2-smoke), NOT a formal impossibility proof"**
- `_pa_oracle_check_runner.py:35-36`:同义句,首字大写 **"This is a STRONG empirical anchor, NOT a formal impossibility proof"**

即 bench 设计者既声称它是 strong empirical / dual anchor,**也在源码里明确补一句它不是充分证明**。本文对这层的定性"经验性锚点"与此一致。

### 3.3 这层的已知弱点(假阳入口)

[likely] 以下任一成立,一个**错误的**候选都可能通过 oracle 锚点,进而被 S2 自一致地确认为 PASS:
1. **oracle 本身写错**:oracle 与 kernel 一起错 → "对齐"但都错。
2. **oracle 被窄化(narrowing)**:oracle 砍掉 gate/scale/分支让它更易通过(历史教训:L2_35/L2_36 曾因此被识别为假阳——此条据内部 review memory,**外部不可独立复核**,作为弱点类型示例而非可验证证据引用)。
3. **输入判别力不足**:shapes 分布太弱,错误实现 trivially pass(历史教训:L3_21/L3_23 用 plain `tril` 会让丢了 forward-substitution 递推的实现也通过 → 已修为 `tril_chunk`,见两 task 当前 `shapes.yaml` 的 `dist: tril_chunk` 注释,可复核)。
4. **entrypoint 选错但碰巧在 sampled shapes/tolerance 下过 oracle**。
5. **tolerance 过松**。
6. **shared launcher seam**(Codex [confirmed],本文新增):同一个 `_pa_launcher.py` 被 **exec 进 S2 original 侧 + oracle runner** 的 kernel module(`_pa_subproc_runner.py:594`、`_pa_oracle_check_runner.py:330`);**severed 侧则是把同一 `_pa_launcher.py` 源码 append 到 staged candidate**(`phase_a_prime.py::build_staged_candidate`(快照 `:6036`,append 在 `:6062`、写 staged 在 `:6076`))——即三处共用同一份 launcher 源码,只是注入方式不同(exec vs append)。若这份 launcher 本身用错,S2 只能证明"两边同错",不是 launch semantics 的独立见证。

### 3.4 输入判别力现状(源码扫描)

[confirmed] 当前 **100 个 task**:**绝大多数(~94)为 5 public + 4 hidden**,其余为 5+5 / 5+1 等变体(精确分布随 shapes.yaml 并行修改会微调,此为 2026-05-31 快照;关键论点不依赖精确分母)。**核心事实**:**23 个 raw PASS 全部是 5+4**,没有只测 1-2 shape 的 PASS,也没有全常量 tensor 输入的 PASS。L3_21/L3_23 已用 `tril_chunk`。
[likely] 但仍非形式化覆盖:**单一 seed=0**,branch/gate/scalar 组合不做 property-based sweep,`randint/offsets/full` 可能未打到边界值。**输入分布是当前正确性的主要剩余风险**。

---

## 4. 完整性 gate — PASS metadata 与 artifact 的绑定

### 4.1 机制

`meta.phase_a_prime.standalone_sha256` 记录 promote 时 staged candidate 的 hash;`validate_tasks.py` 的 phase-aware hash gate 要求它 == 磁盘 `kernel.py` 的 hash,否则判 metadata 陈旧。

### 4.2 当前实测(重要,影响"PASS 计数"口径)

[confirmed] 在当前 dev 仓库快照:23 个 `phase_a_prime.status: PASS` 中,**0/23** 的 `standalone_sha256` 匹配磁盘 `kernel.py`,**0/23** 的 `kernel.py` 暴露合成的 `run()`。

[confirmed] 可确认的事实:dev 机无 GPU(S2-gpu 是唯一 PASS 来源),且当前本地快照 0/23 sha 匹配、0/23 暴露 `run()`。
[speculative] 成因推断(非源码/快照可证):最可能是 PASS metadata 由 GPU 端运行后写回/同步,而 promote 对 `kernel.py` 的替换未同步回本地(本地 `kernel.py` 仍是原始 vendored 版);也可能是 promote 后被 git checkout 回原始。具体成因需查 GPU 端 + 同步链路才能定论。

**推论(对专家讨论至关重要)**:
- "数 raw meta PASS 字段"**不是**可信口径——它可能与磁盘 artifact 不一致。
- **artifact 一致性的必要口径** = 过 `validate_tasks.py` 的 phase-aware sha256 hash gate(`validate_tasks.py:1287`)。但要注意:hash gate **只证明 metadata 与磁盘 artifact 绑定一致**,它**不**重新验证 oracle resolver 的新鲜度,也**不**含 release gate 的 cost_model 等条件。因此它是"可信 PASS"的**必要非充分**条件——完整可信 = hash gate(artifact 一致) ∩ C1(S2 faithfulness) ∩ C2(oracle correctness)。
- 因此"已有 N 个完全 pass"这类陈述,**必须注明是哪个口径、在哪台机的快照、是否过 hash gate、C2 锚点强度如何**,否则不可比、且可能含陈旧假阳。

---

## 5. 系统性假阳风险汇总(Codex 对抗审计,confirmed/likely 标注)

| # | 风险 | 等级 | 所在层 |
|---|------|------|--------|
| 1 | raw meta PASS 陈旧:metadata 与磁盘 kernel.py 不一致(当前 23/23 mismatch) | confirmed | 完整性 gate(需强制 validator) |
| 2 | S2 是同源行为等价,不是独立正确性证明:错的 frozen kernel.py 或错 entrypoint 可自一致 PASS | confirmed | Layer 1 边界 |
| 3 | original 侧被 4 个 shim 正规化(normalized contract ≠ stock vLLM 安装)。Codex 实证降级两处声称:capability probe **非 dispatch-only**(影响量化 config 选择,`w8a8_utils.py:41`/`fp8.py:274`)、triton_utils **非 verbatim**(改 autotune,无 driver 选 first config)。最大风险:量化/CUTLASS/MoE 类若走 host dispatch | **confirmed**(声称已实证降级);当前 corpus likely 不踩(多走直 launch triton) | Layer 1 环境 / 信任根 |
| 4 | oracle + shapes 是经验锚点:oracle 错 / shape 弱 / output projection 窄,会放错误候选进 S2 | likely | Layer 2 |
| 5 | S2 只比对输出 tensor + input mutation/alias;非张量 side effect、全局状态、registry/RNG 状态不在判定面内(除非反映到输出) | speculative | Layer 1 判定面 |
| 6 | shared launcher seam:同一 `_pa_launcher.py` exec 进 S2 original 侧 + oracle runner,severed 侧 append 同源到 staged;launcher 本身用错时 S2 只证"两边同错",非 launch semantics 独立见证(`_pa_subproc_runner.py:594`,`_pa_oracle_check_runner.py:330`,`phase_a_prime.py::build_staged_candidate`(快照 `:6062`)) | confirmed | Layer 1 judging seam |

---

## 6. 合理性评估(结论)

### 6.1 体系设计的优点(可向专家强调)
1. **PASS 只认真 GPU**,CPU/interpret 永不算 PASS,无 GPU 诚实 UNRUN——杜绝"无硬件假完成"。
2. **C1 的 bit-exact + alias + mutated-input + S3 determinism** 是工业级强判据,对"抽取改写引入偏差"几乎零假阳容忍。
3. **C1/C2 职责分离清晰**:忠实性与正确性分两层,各有判据。
4. **源码自带诚实 caveat**:oracle 锚点原文标注 "strong empirical anchor, NOT a formal impossibility proof",不夸大。
5. **有完整性 hash gate** 防陈旧 PASS(只要强制走 validator)。

### 6.2 待专家讨论的真实弱点
1. **C2 正确性是经验锚点而非证明**:依赖 oracle 写对 + 输入有判别力。这是整个体系**最弱的一环**,也是历史假阳(L2_35/36)的来源。
2. **输入仅单 seed、无 property-based sweep**:边界/分支覆盖不形式化。
3. **PASS 计数口径未统一**:raw meta PASS ≠ 过 hash gate 的 PASS;跨机/跨时间快照可能不一致(当前本地 23 raw / 0 过 hash gate 即是实证)。
4. **original 侧信任根缺口(见 §2.4)**:S2 的参照基准本身经 4 个 shim 正规化;capability probe **非 dispatch-only**、triton_utils **非 verbatim**(均 Codex 实证)。与"完整 stock vllm 安装"存在环境差,且 **shim 共模偏差 S2 结构上抓不到**。当前 corpus likely 不踩(多走直 launch triton),但量化/CUTLASS/MoE 类若走 host dispatch 有真实偏离风险。

### 6.3 建议的强化方向(供讨论,非本次实施)
- 将 `validate_tasks.py` hash gate 设为**唯一**对外 PASS 计数入口,任何绕过它的统计视为不可信。
- C2 引入多 seed + property-based 输入 sweep,提升输入判别力。
- 对每个 PASS task 留存 oracle 锚点的**诚实强度标签**(用了几个 shape、tolerance class、是否 bit-exact),让 PASS 可分级而非二元。
- 明确并文档化 bench 声称的等价边界(normalized contract vs stock install)。
- **关 §2.4 信任根缺口**:在装了完整 vLLM + 编译 `_C` 的环境跑一遍 original 做 **stock / full-install differential**;或对"读 capability probe / 走 host dispatch"的 task 加显式 provenance gate(标注其正确性依赖 stock 环境)。

---

## 附:可被专家独立 verify 的源码锚点

> **行号 caveat**:本表行号为 2026-05-31 审计时的快照值。该 repo 当前有多个 session 并行修改(`inputs.py` / `phase_a_prime.py` 等正在被改动),行号会漂移。**请以函数名/符号名为准定位**(如 `phase_a_prime.py::s2_verify_task`、`s2_smoke_compare`、`inputs.py::build_inputs`),行号仅作快速跳转参考。本文每条论断的成立性不依赖具体行号。

| 论断 | 源码位置 |
|------|---------|
| S2 逐位比对(无 ULP) | `phase_a_prime.py:1488` `s2_smoke_compare` |
| alias / mutated-input 比对 | `phase_a_prime.py:1518,1565` |
| 唯一 PASS = S2-gpu + S3 | `phase_a_prime.py:2090,2112,2150` |
| 无 CUDA → UNRUN | `phase_a_prime.py:2080` |
| original 侧加载 + infer/ on path + shim | `_pa_subproc_runner.py:642`;relative import 桥接 `:508` |
| device='cuda' 无 GPU 拒绝 CPU fallback | `_pa_subproc_runner.py:804` |
| _C stub fail-loud | `vllm_runtime_hygiene.py:123` |
| capability probe → False(非 dispatch-only,上游影响量化 config) | `vllm_runtime_hygiene.py::install_vllm_capability_probe_stubs`(快照 `:444`);实证 `infer/.../w8a8_utils.py:41`,`fp8.py:274` |
| triton_utils shim 非 verbatim(改 autotune) | `vllm_triton_utils_shim.py:74,108` |
| oracle runner 也装同组 hygiene shim | `_pa_oracle_check_runner.py::main`(hygiene 安装,快照 `:758-770`) |
| severed 改写(非数值重写)+ closure slicing | `phase_a_prime.py:5517,3774,3938` |
| oracle 锚点 tolerance + >=2 shape | `_pa_oracle_check_runner.py:81`; `tolerance.py:287,308` |
| oracle 锚点 "strong empirical anchor, NOT a formal impossibility proof" | `resolve_upstream_entrypoints.py:11-13,33-35`; `_pa_oracle_check_runner.py:35-36` |
| 单 seed=0 输入物化 | `phase_a_prime.py:2018` (`seed = 0`); `inputs.py::build_inputs` 默认 `seed: int = 0`(当前 `:707`)+ `gen.manual_seed(seed)`(当前 `:731`)（注:`inputs.py` 在并行修改中,以函数名 `build_inputs` 为准,行号可能漂移） |
| shared launcher seam(两侧 + oracle runner;severed 侧 append 同一 launcher) | `_pa_subproc_runner.py:594`; `_pa_oracle_check_runner.py:330`; `phase_a_prime.py::build_staged_candidate`(快照 `:6062`) |
| resolver 状态机(完整) | `phase_a_prime.py:2188` |
| PASS metadata↔artifact hash gate | `validate_tasks.py:1287`(phase-aware sha256 gate) |
