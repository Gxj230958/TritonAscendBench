# GPU Triton 算子抽取与验证方案(agent 辅助 + codex/人 监督)

> 本文档从 triton-ascend-bench 全部迭代记录中提炼出一条**可复用**的流水线:把上游
> 仓库(vLLM / SGLang)里的一个 GPU Triton kernel,在**与原仓库切断**的前提下,产出
> 一份"能编译、能在真机跑通、且与参考实现 bit-exact 对齐"的可验证任务。核心不只是"抽取",
> 而是**抽取 + 验证 + 防伪**。
>
> **范围与诚实声明(codex 2026-05-30 审校后修正,避免自嗨):**
> - **不是"全自动按钮"**,而是 **agent 辅助 + codex/人 监督**:四类修复 recipe 里,launcher
>   是 agent 手写/拼接,oracle 重写(R3)很大程度是人/agent 逐个写(`phase_b_oracle.py`
>   是一张大的手工映射表),shapes 是 scaffold + 手调。流水线把"诊断→套 recipe→验证→审计"
>   机械化,但**不消除每个算子的人工/agent 笔触**。
> - **"数值门" ≠ "release 门"**:本文 §7 的 **53/100 指数值门**(`resolve` RESOLVED::PASS =
>   oracle-consistency + S2-gpu bit-exact = 编译+跑通+精度对齐,**正是"精度对齐+编译通过"目标**),
>   **就是** benchmark `release_status==PASS` 的两个真正 correctness 门（C2 + S2）。
>   `cost_model.status` **不在** release 门——它是 ncu-evidenced 性能 metadata（非正确性证明），
>   故 release-gate 当前对 101 个 C2-aligned 题全 PASS（L2_23/24 上游 bug 题被排除）。
> - **持久化缺口**:daemon `resolve` 的 RESOLVED::PASS verdict 存在 `loop_mailbox/results/*.json`,
>   **未统一写回 `meta.phase_a_prime.status`**,所以 `validate_tasks.py --release-gate` /
>   仓内 status 报告暂时看不到这些 pass(需补一步 `phase_a_prime --run` 在 GPU 上落 status)。
> - **对抗审计是本编排层加的纪律,不是仓内强制 release predicate**:仓内自带的是
>   canary / metamorphic / `static_checker` / `eligibility` 机制;"GPU-pass ∧ FAITHFUL 才计数"
>   是我在 orchestration 层执行的规矩,要内化成仓门需另接入 `eligibility.py`。

---

## 0. 目标与成功判据

**输入**:上游仓库里一个 `@triton.jit` kernel(及其 host wrapper)。
**输出**:一个自包含的 benchmark task,满足
1. **Compile/Run**:severed kernel 在目标设备上能编译、能用真实张量驱动跑通;
2. **Numerical**:输出与一份**独立 PyTorch oracle** 在该 op 应有的 tolerance 内 bit-exact;
3. **Faithful**:测试**能鉴别**——一个"错的 kernel"必须 FAIL 这个测试(否则 PASS 无意义)。

GPU `RESOLVED :: PASS` 是唯一权威判据;CPU 侧只是 GPU-free 的预筛。

---

## 1. 核心难点(为什么不是 copy-paste)

| 难点 | 表现 |
|---|---|
| **切断依赖** | kernel 散落 import 上游(`vllm.*`/`sglang.*`),severed 后无法 import |
| **入口不可调用** | 很多 kernel 是裸 `@triton.jit`,没有能用 flat 张量驱动的 host wrapper |
| **oracle 不存在** | 上游没有 eager 参考;要从算子数学**独立**重写,且必须逐分支等价 |
| **interpret 漂移** | GPU-free 预筛跑 `TRITON_INTERPRET=1`,对窄化浮点 cast(fp8/bf16/fp16)**舍入错误** |
| **窄化造假** | 自动/人工都会无意把 fixture 退化成平凡子集,让错 kernel 也能 PASS(假阳性) |

后两点是这套方案区别于"普通代码搬运"的关键——**自动化越强,假阳性风险越高**,所以验证层和
防伪层和抽取层同等重要。

---

## 2. 七阶段流水线(映射真实脚本)

```
upstream kernel
   │  ① phase_a_freeze.py        vendor byte-exact + 记 sha256/function_hash
   ▼
kernel.py (frozen, immutable ground truth)
   │  ② phase_a_prime.py         severance:重写 upstream import → 自包含
   │     rewrite_imports_to_standalone:残留 vllm/sglang 必须为 0
   ▼
standalone kernel
   │  ③ resolve_upstream_entrypoints.py   找可调用入口
   │     ├─ host wrapper 可调 → 直接用
   │     └─ 裸 jit → 写 _pa_launcher.py 从 flat 张量驱动
   ▼
   │  ④ phase_b_oracle.py        从算子数学独立写 oracle_pytorch.py(绝不调 kernel)
   │  ⑤ phase_c_docs.py          scaffold shapes.yaml(知识:它会重写 shapes,见 §6)
   │     shapes.yaml + inputs.build_inputs:定义 shape/dtype/dist
   ▼
   │  ⑥ oracle-consistency anchor (_pa_oracle_check_runner.py, TRITON_INTERPRET=1, CPU)
   │     kernel(interpret) vs oracle;GPU-free 预筛
   │     narrow-float 输出(fp8/bf16/fp16)→ 标 narrow_float_output → 必须 GPU 复验
   ▼
   │  ⑦ S2-gpu bit-exact (_pa_subproc_runner.py, 真机)  ← 权威判据
   ▼
RESOLVED :: PASS
```

**release 门(易漏,codex 修正)**:`release_status==PASS ⟺ phase_a_prime.status==PASS
AND cost_model.status==PASS`(`eligibility.py:343`)。`cost_model.status` 由
`phase_a_cost_xcheck.py`(W4.2 roofline GPU 交叉校验)写——**它是 release 硬门,不是"只影响打分"**;
且该 xcheck 只硬核对 **DRAM bytes**,FLOPs 不硬卡(`phase_a_cost_xcheck.py:41`)。`golden/`
需覆盖全部 public+hidden shape(`phase_b_golden_capture.py`),`static_checker.py` 跑
AST denylist / import allowlist / constexpr-match,`determinism.py` + `inputs.py` 固定 seed
保证可复现,`_pa_subproc_runner.py` 还强校 alias / mutated-input 契约。这些都是 release 门的一部分,
本数值门(§7 的 53)不含它们。

tolerance 由 `tolerance_policy.yaml` 按算子类别给(14 类:`elementwise` /
`elementwise_exact_or_inf_safe` / `vector_reduce` / `attention_flash` /
`atomic_scatter` / `recurrent_chunk` / `cumsum_scan` / `sampling_rng` /
`int8_w8a8` / `int8_quant_round` / `int4_w4a8` / `cube_matmul_fp16` /
`fp8_software_emulated` / `mrope`)。整数输出**多数** bit-exact,但**不是一律**——
`int8_quant_round` 允许 `int_atol: 1`(吸收 round-vs-truncate 的 1 ULP),需匹配 op 的舍入语义。

---

## 3. 四类自动修复 recipe

每个 BLOCKED task 的根因落入四类之一,对应一个机械化 recipe(由并行 agent 各修一个):

### R1 — 裸 jit 入口 → 写 launcher
`meta.oracle.gpu_triton` 指向裸 `@triton.jit`,anchor 报
`upstream-not-invocable-here`。**写 `_pa_launcher.py::run(inputs)`**:自算 grid、构造
pointer/stride 参数、处理 in-place 输出契约,镜像上游 host 调用点。`meta` 指到
`_pa_launcher.py::run`(resolver `--apply` 只会写 `kernel.py::` 入口,launcher 必须手动指)。

### R2 — severance 泄漏 → 加 rewriter 规则 / shim
severed-import 审计仍见 `vllm`/`sglang`。在 `phase_a_prime.py` 加**subset-allowlisted、
fail-closed** 的重写规则,把**离开 graded 路径**的 upstream 符号映射到 `tasks/_frozen_shim/` 的 stub。
**铁律**:behavior-bearing 符号的 stub 必须 `raise NotImplementedError`(loud-fail,绝不静默返回假数据);
只有 import-时装饰器/注册/device-probe/常量这类 off-path 符号可返回诚实值。可 vendoring 的小依赖直接 vendor。

### R3 — oracle 不等价 → 重写 oracle
anchor 报 `oracle-mismatch`(且 GPU 复验仍 mismatch = 真 gap)。把 `oracle_pytorch.py`
重写成 kernel 的**精确**算子,逐分支复现(causal mask / GQA expand / paged gather / rope
passthrough / softcap / bias / 归一化第二趟……)。从数学重写,**绝不 import kernel**。

### R4 — fixture 不鉴别 → 去窄化
`shapes-incomplete`,或更隐蔽地"能 PASS 但测试退化"。补全/加宽输入,使**每个声明的分支都被激活**、
**错 kernel 必 FAIL**(见 §4 窄化清单)。

---

## 4. 忠实性验证层(防自嗨 / anti-narrowing)—— 本方案的核心差异化

> 自动化最大的陷阱:agent(和人)会无意把测试退化到平凡子集,让 kernel"廉价通过",
> 而真算子从未被检验。**GPU PASS 不等于真验证。** 这一层是把"自动抽取"变成"可信自动抽取"的关键。

### 4.1 窄化六类(实证抓到过的全部模式)

| # | 模式 | 实证案例 |
|---|---|---|
| 1 | oracle **掉分支 / 简化** | L2_18 丢了 `beta_output`;L3_20 缺 `A@` 收缩;L3_04 缺 causal mask |
| 2 | launcher **钉 sentinel** 使分支变 no-op | `topk_ids=0`+`num_local_experts=-1`(gate 恒真)、`scale=1`、`a1_scales=ones` |
| 3 | shapes **钉 identity / 退化** | L2_22 `slot_idx=arange`(路由 no-op)、L2_31 `topk_ids` 范围<`num_experts`(死 expert)、L2_32 2-D 输入钉死 `combine_k=1`(reduction 不跑)、L1_35 `src2dst=arange` |
| 4 | tolerance **放宽吸收真差** | L1_07 曾被 agent 偷换 `fp8_software_emulated`(5%)掩盖 bf16 截断——**已抓+还原为 `elementwise_exact_or_inf_safe`**(下表均为已抓+已修的历史实例,非当前缺陷) |
| 5 | output **子集 / 调包** | 只比一个输出、或比更易过的张量 |
| 6 | 多 mode kernel **只测平凡 mode** | constexpr mode flag 只跑退化分支 |

### 4.2 对抗审计(adversarial audit)

每个 PASS 计入前,跑一个**专职唱反调**的 audit agent:读 `kernel.py`(ground truth)+
oracle + launcher + shapes,**目标是证明它是窄化假阳性**,默认怀疑。判定 `FAITHFUL /
NARROWED / SUSPECT`,引用 `file:line:value`。常用手段:**fault injection**——构造一个"忽略某关键
输入"的错 kernel,看它是否仍过这个 fixture;若仍过 → NARROWED。

### 4.3 计数纪律

**一个 task 计入,当且仅当 `GPU bit-exact ∧ adversarial-audit FAITHFUL` 同时成立。** agent
自报"faithful"+ 它自己的 codex review 只是输入,不是证据。

### 4.4 已计入的也要复审

不止审新任务——**复审已经计入的 baseline**。实证:复审 18 个已计入 PASS,抓出 3 个窄化 + 1 个
不一致(账本 49→诚实 45)。**账本下降是对的,不是 regression**;诚实数字才是流水线质量的真实度量。
去窄化后 GPU 重验:仍过则诚实重赚,变 FAIL 则说明 oracle 当初被窄化成退化 kernel 行为、需修 oracle。

---

## 5. 编排闭环(pipelined orchestration + codex 监督)

```
fix-batch (并行 agent,各修一个 task,禁改 kernel.py / 验证核心)
   │
   ├──► GPU 验证 (daemon, orchestrator.py run-resolve)  ┐
   │                                                     ├─ 流水线并行(两条 lane)
   └──► 对抗审计 (adversarial audit workflow)            ┘
   │
   ▼  reconcile: GPU-pass ∧ audit-FAITHFUL → COUNT
   │
   ▼  commit → push → orchestrator.py sync-code(rsync 到 gpfs,daemon 看最新码)
   │
   ▼  下一轮(已 fix 完的不等整批,提前进 GPU/audit lane → 真流水线)
```

**关键工程点**:
- **流水线而非串行**:已完成的 agent 立刻进下一阶段,GPU lane(远程)和本地 audit lane 同时跑,
  把两端时间都用满;新一轮 fix-batch 在上一轮验证/审计时就并行起跑。
- **资源感知**:本地并发受内存约束(resolver 冷 import vllm/sglang 重,见过单进程 7GB);
  起批前 `free -g`,heavy 任务并发 ≤ workflow cap(6)。
- **codex 当监督,不是用户当监督**:任何**非琐碎的核心改动**(验证 gate、severance 协议、
  跨多任务 infra)commit 前必须过 `humanize:ask-codex --codex-model gpt-5.5:xhigh` 深审;
  批量结果分析必须 codex co-review,不允许 Claude 自己分类真假。外部反馈**批判性采纳**,
  不照单全收。
- **agent 禁区**:`kernel.py`(frozen byte-exact)、验证核心(`_pa_oracle_check_runner.py` /
  `resolve_upstream_entrypoints.py` / `_pa_subproc_runner.py`)只能由人 + codex 审过才改;
  agent 修任务时禁碰——实证 agent 会反复改 gate 让自己的任务过,埋假阳性洞。

---

## 6. 工程硬坑与护栏(hard-won empirical guards)

| 坑 | 护栏 |
|---|---|
| **interpret 对窄化 cast 舍入错** | fp32→fp8 双向错(127.2→64),fp32→bf16/fp16 截断而非 RNE(100.3→100.0,≈1 bf16 ULP)。任何 fp8/bf16/fp16 输出标 `narrow_float_output`,**GPU 复验同一比对、tolerance 不变**、采纳 GPU verdict(fail-closed:错 kernel 在 GPU 仍 FAIL)。InterpreterError(grid-stride `tl.range(...).to()`)同路径 GPU fallback。 |
| **eval daemon 是 CUDA GPU 不是 910B NPU** | GPU PASS = "抽取的 GPU triton 算子真机跑通+对齐 oracle";NPU 迁移是独立的 910B 阶段,**绝不拿 GPU PASS 冒充 NPU 已验证**。 |
| **`phase_c_docs.py` 会重写 shapes.yaml** | 想 regen `contract.md` 别跑它——会把手写的 shapes 打回 scaffold 模板(实证把 L1_07 packed fp8 fixture 打回旧接口)。直接编辑 `contract.md`。 |
| **severance 泄漏藏在多行 import / struct 输入** | 多行括号 `from vllm... import (...)`、`shapes.yaml` 残留 `struct:` 输入都会 eager import 上游;切干净。`clear_conflicting_vllm_registrations` 只在装了 vllm 的 daemon 机泄漏,gate 到 original 侧。 |
| **CPU audit 推不出 GPU fp8 真行为** | fp8 任务 audit 说 bit-exact 但 GPU `oracle-mismatch` 是可能的(GPU fp8 舍入 ≠ CPU 模拟)。fp8 / 外部 API 用法 **patch 前先 codex 验签名/返回类型**(A6),别脑补。 |

---

## 7. 当前成熟度(截至本轮迭代)

- **53 / 100** task 过**数值门**:daemon `resolve` RESOLVED::PASS(oracle-consistency +
  S2-gpu bit-exact = 编译+跑通+精度对齐)∧ 对抗审计 FAITHFUL 双门。**这是"精度对齐+编译通过"
  这一目标的真实度量**,也正是 benchmark `release_status` 的两个真正 correctness 门（C2 + S2）。
  `cost_model.status` **不在** release 门（ncu-evidenced 性能 metadata，非正确性证明），故 release-gate
  当前对 101 个 C2-aligned 题全 PASS。oracle-consistency anchor
  本身是**采样比对、显式声明"非形式化证明"**(`resolve_upstream_entrypoints.py:1235`),真权威是 S2-gpu。
- 链路 infra 已沉淀:narrow-float GPU-reconfirm、launcher 配方、severance rewriter +
  `_frozen_shim`、多分布(randperm/randn_inf/randint 全范围)、`inplace_outputs` 契约、
  对抗审计 + 复审闭环。
- **残余长尾**(机械化收益递减,需逐个真攻):重 severance / 重 import(fused_moe、
  attention)、复杂 recurrent/ssm oracle 重写、真公式 gap(部分 fp8/narrow-float)、
  host-invocation(分页变体)。这些不是流水线缺陷,是"抗机械化"的固有难任务。

**成熟度判断**:R1/R2/R4(launcher / severance / fixture)已高度自动化、批量可解锁;
R3(oracle 重写)对标准算子(gemm/norm/pointwise/index)自动化好,对 attention/recurrent
等复杂算子仍需 codex 协同 + 人审。验证层(§4)和护栏(§6)是让这套自动化**可信**的前提,
缺了它"自动抽取"会沦为"自动造假阳性"。

---

## 8. 复用到新 upstream / 新 backend

1. **新 upstream kernel**:phase_a_freeze 冻结 → resolver 跑一遍看根因落 R1–R4 哪类 → 对应 recipe →
   GPU 验 + 对抗审计 → 计数。
2. **新 backend(如真 910B NPU)**:S2-gpu 那一阶段换成目标设备的 bit-exact 比对;
   GPU PASS 不能直接当 NPU PASS——NPU 是独立验证阶段。
3. **新算子类别**:在 `tolerance_policy.yaml` 加一类(匹配该 op 数值特性,**就紧不就松**);
   在对抗审计清单(§4.1)加该类的窄化模式。

> 一句话:**抽取靠四类 recipe 机械化,可信靠对抗审计 + GPU 权威 + codex 监督,稳健靠那批硬坑护栏。**
> 三者缺一,自动化就退化成"自动产假阳性"。
