# 正确性模型

> English: [../../reference/correctness_model.md](../../reference/correctness_model.md)

TritonAscendBench 如何判定一个算子的输出是**正确**的——即移植后的内核被据以打分的精度对齐标准，以及为每一档容差提供依据的误差模型。

本文是 [`tolerance_policy.md`](tolerance_policy.md)（按容差类整理的速查表）和 [`ARCHITECTURE.md`](../ARCHITECTURE.md)（端到端数据流）的机制配套说明。权威实现位于 `ascendbench/correctness/tolerance.py`；按容差类列出的具体数值则存放在 `ascendbench/policies/tolerance_policy.yaml`。

关于*证明*与*推理*的区别需要说明一句：策略文件给出了每个容差类的 `atol`/`rtol` 边界以及一行注释——它**并不**携带推导过程。下文的误差模型依据（例如"为什么 bf16 GEMM 取 5e-3"）是标准的数值分析推理，用来解释*为什么取这一量级的边界是合适的*；它们并不是代码计算或断言出来的边界。凡是某个数字属于外部/标准事实、而非代码实际检查的内容，文中都会明确指出。

---

## 1. 两个正确性层（最容易混淆之处）

TAB 在**两个不同的维度**上、用**两套不同的标准**来判定正确性：

| 层 | 问题 | 标准 | 位置 |
|-------|----------|----------|-------|
| **C2 — oracle 等价性** | 内核计算的数学是否与 PyTorch oracle 一致？ | 一个**容差类**（数值容差） | 每次提交者打分；提供方 GPU 交叉核对 |
| **S2 — 切断验证** | 被*冻结*的独立 `kernel.py` 是否忠实于原始的上游调用？ | **逐位精确（bit-exact）**（数值**以及**别名关系**以及**被原地修改的输入行为） | 仅在提供方侧构建语料库时 |

最重要的一条规则：**容差只适用于 C2。** S2 永远是逐位精确——它将切断后的内核与真实上游调用相比对，必须完全一致地匹配，包括哪些输入被原地修改、哪些输出张量与哪些输入互为别名。提交者永远不会被 S2 打分；S2 是用来保证该任务 `kernel.py` 是生产代码真实副本的门。

本文余下部分讲的都是 **C2**，即用于打分的精度标准。

---

## 2. 比较的是什么，以及 shape 判定如何聚合

oracle 与候选实现都返回 `dict[str, Tensor]`。打分是按**输出 key**、按 **shape 行**进行的：

- **先看结构。** **oracle** 返回的每一个 key 都必须出现在候选实现的输出中，且张量 shape 匹配。提交者打分（`eval.py`）遍历 oracle 的 key 并忽略候选实现中任何*多余*的 key；而更严格的提供方 oracle 等价性交叉核对（`run_oracle_equivalence_gpu.py`）则要求内核和参考实现的 key 集合完全一致。shape 不匹配会立即判为失败——张量绝不会被转换或广播以强行进行比较。
- **逐 shape 进行。** 每一个被打分的 shape 行（在 `self-reported` 模式下为公开 shape；在 `verified` 模式下为公开**以及**封存的 `hidden_shapes`）都用 `tolerance.compare` 独立比较。
- **任务判定。** 一个任务的 `correctness_status` 只有在**每一个**被打分的 shape 都通过时（`all(r.correctness_passed for r in ran)`）才为 `"ok"`；任何一个 shape 失败都使其变为 `"fail"`。`coverage` 则单独报告通过的比例。
- **设备内核必须真正运行。** 若候选实现没有产生**任何**设备/triton 内核事件，说明它已悄悄退回到通用的 torch 路径；即使数值完全相同，也会被强制判为正确性**失败**——迁移必须在目标设备上运行一个真实的内核，而不是 eager PyTorch。

---

## 3. 数值 dtype 基础

容差标准建立在 IEEE-754（以及有限的 fp8）位布局之上。比较代码**并不**自行推导机器 epsilon——浮点 dtype 由 `torch.allclose(atol, rtol)` 打分，预算取自 `tolerance_policy.yaml`。只有两条 dtype 感知的 ULP 路径（`_int_atol_compare`、`_fp8_ulp_compare`）会操作原始位。

### 位布局

| dtype | 符号位 | 指数位 | 尾数位（存储） | 总位数 | 编码 |
|-------|------|----------|-------------------|-------|------|
| fp32 | 1 | 8 | 23 | 32 | IEEE-754 binary32 |
| bf16 | 1 | 8 | 7 | 16 | 截断的 binary32（相同的 8 位指数） |
| fp16 | 1 | 5 | 10 | 16 | IEEE-754 binary16 |
| fp8 e4m3fn | 1 | 4 | 3 | 8 | 仅有限值（"fn"），无 Inf 编码 |
| int8 | — | — | — | 8 | 二进制补码，范围 −128..127 |

bf16 与 fp32 共享 8 位指数（相同的动态范围），但只保留 7 个尾数位；fp16 则以指数范围（5 位）换取 10 个尾数位。

### 两个 epsilon——务必分清

下文的误差量级用到两个相关的量；二者相差一个 2× 因子，否则极易引起混淆：

- **机器 epsilon `ε` = `2^-(存储的尾数位数)`**——1.0 与下一个可表示值之间的间隔（在 1.0 binade 处的一个 ULP）。
- **单位舍入误差 `u` = `ε/2` = `2^-(存储位数+1)`**——*单次正确舍入操作的最坏情况相对误差*。误差预算就是以 `u` 为单位来推理的。

| dtype | 存储的尾数位 | `ε = 2^-m` | `u = 2^-(m+1)` |
|-------|-----------------|------------|----------------|
| fp32 | 23 | 1.19e-7 | 5.96e-8 |
| bf16 | 7 | 7.81e-3 | 3.91e-3 |
| fp16 | 10 | 9.77e-4 | 4.88e-4 |
| fp8 e4m3 | 3 | 0.125 | 0.0625 |

（这些是标准的 IEEE-754 / OCP-fp8 数值，独立计算得出；代码并不对它们做断言。）bf16/fp16 的网格之所以粗，正是浮点类带有宽松 `atol`/`rtol` 的原因；而 fp8——其在 1.0 处的 `ε` 高达整整 0.125——则按可表示的*编码*而非数值来打分。

### 整数 ULP

对于整数输出，"1 ULP"就是一个绝对的整数差。`_int_atol_compare` **先**把两个张量都转换为 **int64**，再计算 `(a − e).abs()`，这样减法就不会在窄的输入 dtype 上溢出。（具体来说，int8 的 `(−128) − (127)` 在原 dtype 中会回绕为 `+1`，但在加宽之后是 `−255`——正是这次加宽才使 `|a−e| ≤ int_atol` 成立。）

### fp8 e4m3 有序符号-幅值编码

`_fp8_ulp_compare` 将每个 fp8 字节重解释为 `uint8 → int16`，并构造一个有符号的有序编码，使得相邻的可表示值之间的距离恰好为 1：

- `mag = b & 0x7F` 保留低 7 位（指数+尾数）。因为指数占据较高位、尾数占据较低位，这个 7 位值在 |x| 上是**单调的**（这是 IEEE 位序的一个性质）。
- `code = −mag if sign-bit else mag`；`mag == 0 → 0` 将 **+0（0x00）和 −0（0x80）**规范化为同一个编码，使有符号零比较相等。
- 唯一的位模式 `(b & 0x7F) == 0x7F` 是 **NaN**——e4m3fn 只有有限值，没有 Inf 编码。NaN 的位置必须完全匹配。

**一个 1-ULP 步进的实例：** `1.0` 编码为字节 `0b00111000`（= 56）；下一个可表示值 `1.125` 编码为 `0b00111001`（= 57）。有序编码 56 和 57 相差 1，而 `1.125 − 1.0 = 0.125` 恰好是在 1.0 指数处的一个 fp8 ULP（`2^0 · 2^-3`）。因此一个接近中点、可能舍入到任一邻值的量化结果，会表现为有序编码距离 1——这正是 fp8 上 `int_atol = 1` 的对应物。

---

## 4. `compare()` —— dtype 路由

`tolerance.compare(actual, expected, cls)` 在任何数值检查**之前**先按 dtype 分派，因为"接近"的正确定义因 dtype 而异。按顺序为：

1. **shape 不匹配** → 失败（绝不转换）。
2. **整数 dtype**（`bool`/`uint8`/`int8`/`int16`/`int32`/`int64`）：
   - `cls.bit_exact`，或该容差类未声明 `int_atol` → **`torch.equal`**。
   - 否则 → **整数 ULP**：`|actual − expected| ≤ int_atol`。
   - 显式的 `bit_exact` 始终优先于 `int_atol`。
   - *为何绝不做浮点转换：* 把整数输出转换为 fp32 再跑 `allclose`，会悄悄把例如 `int64(2²⁴+1)` 舍入为 `2²⁴`，将一个 off-by-one 错误变成假通过。
3. **fp8**（该容差类设置了 `fp8_ulp`）：
   - 两侧都是 fp8 → **有序编码 ULP**（见下文 §3 / §4）。
   - 恰好一侧是 fp8 → 失败。被打分的 fp8 输出必须*以* fp8 进行比较；把一侧 fp32 转换以达到通过是被禁止的。
4. **`cls.bit_exact`**（非整数）→ `torch.equal`。
5. **其余情况（浮点）** → 将**两侧**都转换为 fp32，然后：
   - 一次**采样快速失败**（仅在 `numel > 1024` 且 `sample_fraction < 1` 时）：若连随机采样都失败就提前拒绝，但
   - **权威**判定始终是对每一个元素进行的**完整** `allclose`——采样通过永远不能成为最终答案（NaN/Inf 或落在采样窗口之外的漂移仍必须被捕获）。

---

## 5. 四个比较器

| 比较器 | dtype | 通过判据 |
|------------|-------|----------------|
| **浮点 allclose**（`_check_full`） | fp16/bf16/fp32 | inf 和 NaN 的**位置**必须匹配（`inf_must_match`/`nan_must_match`，两者默认为 true）；然后对有限元素跑 `torch.allclose(atol, rtol)`。 |
| **整数 ULP**（`_int_atol_compare`） | int* | dtype 必须**完全一致**；加宽到 `int64` 后的 `\|a−e\|` `≤ int_atol`。 |
| **fp8 有序编码 ULP**（`_fp8_ulp_compare`） | `float8_e4m3fn` | dtype 一致；有序 e4m3 编码（§3）；当且仅当 `max_ordered_dist ≤ ulp` **且** off-by-one 的**比例 ≤ 1%**（`max_frac`）时通过。 |
| **逐位精确（bit-exact）**（`_bit_exact_compare`） | 任意 | dtype 一致；`torch.equal`。 |

两点设计说明：

- **fp8 的两段式检验。** 单个 off-by-one 的有序编码步进是真正的边界舍入歧义（一个落在 fp8 桶中点的 `silu`/`exp` epsilon 范围内的值可能向任一侧舍入）。但*大量*的 off-by-one 元素则是系统性的舍入模式偏差（一个真实 bug），因此即便每个单独步进都只有 1 ULP，`off_frac ≤ 1%` 这一上限也会拒绝它。
- **dtype 相等性由整数 / fp8 / bit-exact 比较器强制执行**，而非浮点路径。浮点路径把**两侧**都转换为 fp32，所以一个 `fp32`-对-`bf16` 的*浮点*输出差异不会被 `compare` 单独捕获——一次严格的 `VERIFIED` 浮点比较本身并不断言输出 dtype。它在别处被捕获：切断验证 anchor 的 dtype 检查（§10），以及——仅当一个严格失败的 shape 随后被考虑进行上限接受时——**上限 dtype 门**（§8）。

---

## 6. 各容差类的误差模型

`tolerance_policy.yaml` 定义了 **14 个类**。打分路径按名称解析其一（`eval.py` → `load_policy().get(tolerance_class)`；提供方 GPU 交叉核对 → `_tolerance_for(...)`，在名称缺失时带一个宽松的 `vector_reduce` 式回退）。下文的依据解释了*为什么*每条边界落在它的取值上；它们是误差模型推理，而非策略所证明的边界。

**逐点 —— `elementwise`（1e-3 / 1e-3）。** silu/gelu/bias 之类。一次输出舍入加上超越函数自身的库近似（libdevice 对 torch）。单次 fp16 舍入约为 `≈ u_fp16 ≈ 4.9e-4`；1e-3 以约 2× 的余量覆盖它。无累积 → 最紧的非精确档。示例：`L2_03_act_and_mul`、`L2_05_gelu_and_mul`、`L1_13_fused_softcap`。

**Logit 相加 —— `elementwise_exact_or_inf_safe`（1e-6 / 0，inf+nan 必须匹配）。** 对 logits 的单次加法运算在 fp32 中本质上是精确的，因此 `rtol=0` 加一个接近机器精度的 `atol` 是合适的；起承重作用的约束是 `−inf` 掩码位置必须**完全**匹配（一个被错误掩码的 logit 会改变 softmax 的支撑集）。这是唯一*显式命名* inf/nan 要求的类，尽管 `_check_full` 对每个浮点类都强制执行它。示例：`L1_01_bias`、`C1_identity_trap` canary。

**归约 / softmax —— `vector_reduce`（1e-3 / 1e-3）。** RMSNorm/LayerNorm/softmax 在 fp32 累加器中沿 hidden 维归约（`u_fp32 ≈ 6e-8`，累积抖动可忽略）；主导误差是最终输出的向下转换加上 `exp` 库的差异。与逐点同档。示例：`L1_12_rms_norm`、`L1_28_fused_rmsnorm`。（`cumsum_scan` 和 `mrope` 均为 1e-3/1e-3，依同样的单次舍入逻辑归入此档；关于 `cumsum_scan` 零任务的状态见 §7。）

**矩阵乘 / GEMM —— `cube_matmul_fp16`（5e-3 / 1e-3）。** K 维乘积在 fp32 中累加（接近精确），但**输出向下转换为 bf16** 带来 `≈ u_bf16 ≈ 3.9e-3`；5e-3 恰好高于该值，同时也吸收 cube 单元 / TF32 式累积顺序的差异。这是更宽松的 GEMM 档（相对 1e-3 的逐点），正是因为 bf16 输出的粒度。（类名虽写 `fp16`，但其注释指明使用 fp32 累加器，且大多数被分配的任务使用 bf16 I/O；若某任务的输出确实是 fp16，则 `u_fp16 ≈ 4.9e-4`，5e-3 就更加宽松。）示例：`L2_19_awq_gemm`、`L2_16_sgemm_lora_a`。

**Flash attention —— `attention_flash`（5e-3 / 1e-3）。** 与 GEMM 数值相同，因为它*就是*两次 fp32 累加器矩阵乘（QKᵀ、PV）加一个 bf16 输出，外加来自 online-softmax 重缩放的额外浮点运算（见 §7）。`rtol` 保持 1e-3，因为 softmax 会自归一化。示例：`L3_13_kernel_unified_attention_3d`、`L3_16_fwd`。

**整数量化 —— `int8_w8a8`（5e-2 / 1e-2）、`int4_w4a8`（5e-2 / 1e-2）、`int8_quant_round`（1e-5 / 1e-5，`int_atol 1`）。** W8A8/W4A8 反量化档之所以宽松（`rtol 1e-2 = 1%`），是因为 int32 的 K 累加是精确的，但**逐 token/逐 group 在宽动态范围上的 fp 尺度乘法**放大了绝对误差。`int8_quant_round` 是本文档记录得最干净的一处推导：其 int8 `q` 输出按 `int_atol = 1` 打分，因为 fp32 的银行家舍入（`torch.round`）与 Triton 的就近-远离零舍入**在相等情况（ties）下至多相差一步**；`1e-5` 的 atol/rtol 只对辅助的 fp32 `scale` 输出（`≈ max(|x|)/127`，一次接近精确的归约）打分。

**fp8 —— `fp8_software_emulated`（1e-1 / 5e-2，`fp8_ulp 1`）。** 910B 没有 fp8 cube（§9），因此在 Roofline 中将其建模为 vector + HBM。fp8 e4m3 张量按**有序编码距离 ≤ 1** 打分（fp8 上 `int_atol = 1` 的对应物，§3），并带 1% 的系统性偏差上限；宽松的数值 `atol/rtol` 只对辅助的 fp32 scale 输出打分，反映 e4m3 粗糙的 `ε ≈ 12.5%`。示例：`L2_13_silu_mul_fp8_quant_deep_gemm`、`L1_09_quantize_k_cache_fast`。

**循环 / 分块 —— `recurrent_chunk`（5e-3 / 5e-3）。** Mamba/GDN/KDA 分块流水线在块与块之间携带隐藏状态；误差跨块累积（§7）。这是**唯一一个将 `rtol` 放宽到 1e-3 以上的非量化浮点类**，因为被携带、未归一化的状态会同时增长绝对误差和相对误差。示例：`L1_27_chunk_cumsum_fwd`、`L3_01_chunk_scan_fwd`。

**采样 —— `sampling_rng`（0 / 0，`bit_exact`）。** 当种子匹配时，Philox/Triton 的抽取流必须**完全一致**，因此走 `torch.equal`；任何非零容差都会掩盖一个错误的 RNG 流。示例：`L2_08_gumbel_sample`、`L2_11_rejection_sample`。

**原子散射 —— `atomic_scatter`（1e-4 / 1e-4，`rerun_count 3`）。** 原子加散射的求和顺序不确定 → 浮点非结合性带来逐次运行的抖动；紧的 1e-4 反映了各个原子贡献都很小。这里大多数任务发出**整数索引**输出，无论 1e-4 如何都走 bit-exact——浮点容差只对浮点散射输出起作用。`non_deterministic`/`rerun_count` 是元数据，`compare()` 本身并不消费它们。示例：`L1_24_compute_seg_indptr_triton`、`L2_28_moe_align_block_size_stage2`。

---

## 7. 误差累积与数值稳定性

单次逐点运算按最紧的浮点带（1e-3）打分。流水化的运算获得**更宽**的带，因为它们串联了许多注入舍入的步骤。这里的误差复合论证是标准的数值推理——策略放宽了带，但没有任何代码计算一个 N 步边界。

| 容差类 | atol | rtol | 说明 |
|-------|------|------|-------|
| `elementwise`（基线） | 1e-3 | 1e-3 | 一次舍入步进 |
| `cumsum_scan` | 1e-3 | 1e-3 | 线性前缀和——不放宽 |
| `attention_flash` | **5e-3** | 1e-3 | online softmax + 2 次 fp32 GEMM |
| `recurrent_chunk` | **5e-3** | **5e-3** | 状态跨块携带 |

**Online softmax（`attention_flash`）。** Flash attention 从不物化完整的分数行：它在保持一个运行最大值 `m` 和分母 `l` 的同时流式处理 K/V 块，并在每个块边界用 `exp(m_old − m_new)` 重缩放 V 累加器。oracle 是朴素的单遍 fp32 softmax；内核的流式重排对相同的数学产生了不同的舍入序列。某个任务的 contract 给出了一个启发式漂移模型——`≈ ε_fp16 × sequence_length`（`L3_16_fwd/contract.md`），这是作者陈述的估计，而非代码证明的边界。`atol` 放宽到 GEMM 带（两次 fp32 累加的矩阵乘），但 `rtol` 保持 1e-3，因为 softmax 会自归一化（输出是 V 行的凸组合，不会有相对放大）。*范围说明：* 若干标记为 `attention_flash` 的 L1 任务（`L1_20`/`L1_21` kv 索引构建器、`L1_22` 页表变换）是**整数分页/索引内核**，它们继承该类是因为服务领域，而非因为累积了 softmax 误差。

**分块循环（`recurrent_chunk`）。** Mamba/GDN/KDA 流水线跨块携带一个隐藏状态，因此每个块的舍入误差喂给下一个块并被重新缩放——误差按块复合。oracle 直接对此建模（例如 `L1_26_state_passing_fwd`：`S_c = exp(dA_cumsum[…,−1]) · S_{c−1} + states[c]` 沿块跨度循环），并注入**与内核所用相同的逐块降精度舍入**（`L3_01` 将操作数舍入到 bf16；`L3_22` 通过 `_to_tf32` 丢弃低 13 个尾数位，§10）。由于该循环是乘法性的、且携带的状态未归一化，绝对误差和相对误差都会逐块增长——故而它是唯一一个将 `rtol` 放宽到 5e-3 的浮点类。

**前缀扫描（`cumsum_scan`）。** 纯前缀扫描是一次*线性*的 fp32 累加，因此运行误差只按加法增长（`~N · u_fp32`），并远低于 bf16/fp16 的输出舍入下限——故而它保持 1e-3 的逐点带。实际上**没有任务使用 `cumsum_scan`**：本基准的分块 cumsum 内核（`L1_27`、`L2_35`）按 `recurrent_chunk` 打分，因为它们的逐块 `cumsum` 不是终端输出，而是喂给一个 `exp()`-衰减循环，因而继承了更宽的分块带。

---

## 8. 上限接受（ceiling acceptance）—— 受限的大 shape 放宽

某个已正确对齐的内核的一些大 shape 仅仅因为 dtype 累积上限（例如一个 bf16 `tl.dot` MMA）而落在严格容差之外一丝，而非因为真实的偏离。这些被打分为 `CEILING_ACCEPTED`——一个**与 `VERIFIED` 不同、绝不并入严格通过计数、也绝不允许提升 `golden_sha256`** 的状态。它在严格的 `tolerance.compare` 失败**之后**，由 `run_oracle_equivalence_gpu.py` 中的 `_ceiling_verdict` 依照 `ascendbench/policies/ceiling_acceptance.yaml` 中的允许列表来施加。`tolerance.py` 本身从不放宽；`release_gate/eligibility.py` 只*读取*所记录的状态。

允许列表是**按任务限定的**，并使用一个就地的判别量：

```
max_tol_units = max( |actual − ref| / (atol + rtol·|ref|) )
```

这与严格 `compare` 用的是**同一条带**，所以一个接近零（接近抵消）的参考值仍被 `atol` 约束住，无法被掩盖。两种模式：

- **模式 A —— units 上限。** `tol_units_tasks` 中的任务被接受，当且仅当它有一个非空的失败 key 集合、无 compare 错误、dtype 正确、所有统计量有限、且 `max_tol_units ≤ tol_units_cap`（当前为 **8.0**）。一个干净的 dtype 上限落在 1 上方几个单位处；而 `L2_23/24` off-by-one 的真实上游 bug 落在 1000–6200——超过上限 100 倍以上——故而它在构造上保持严格。
- **模式 B —— 物理尺度边界。** 针对极少数其最差元素是合法的接近抵消点（接近零的 ref、由 `atol` 主导）的任务，列在 `physical_tasks` 中（当前为 `L3_22`）。被接受当且仅当**每一个**失败的输出 key 都满足 `max_abs ≤ physical_factor · max|ref[key]| · ulp_rel`。

不在允许列表上的一切都保持严格。`validate_corpus_consistency` 强制只有被列入允许列表的任务才能携带上限状态，且需具备完整的 shape 覆盖、`golden_source = dtype_ceiling_accepted`、且无 `golden_sha256`。

---

## 9. GPU → NPU 迁移的精度细节

这是一个迁移基准：它在 CUDA 设备上为 GPU-Triton 内核打分，但*目标*是 Ascend 910B。两个设备事实驱动了精度设计。

**910B 的数值特性。** 910B 将计算拆分到一个 **Cube（矩阵）单元**和一个 **Vector（SIMD）单元**。Roofline 峰值表恰好反映了这一点：Cube 的 `fp16`/`bf16` 与 `int8` 类、单个 `vector` 类，以及**有意不设 `fp8_cube` 类**。`NpuBackend.dtype_supported` 对 `float8_e4m3fn`/`e5m2` 返回 `False`，并附注释 *"910B does not have FP8 cube arithmetic; software emulation only"*——所以在 NPU 上一个 fp8 内核运行在 Vector / 软件转换路径上，而非张量核 fp8 MAC。（这是一个有硬件能力依据的论断，立足于项目自身的代码注释 + 峰值表的形态，而非外部数据手册。）Cube 矩阵乘在 fp32 中累加（因此 `cube_matmul_fp16` 取 5e-3）；fp8 没有这样的硬件累加器，这就是为什么 `fp8_software_emulated` 很宽、而 fp8 张量按有序编码 ULP 打分（§6）。`dtype_supported = False` 是一个硬件标志，**并非**分母逃逸——fp8 任务仍在该类下被打分（资格仅由 `eligibility.py` 管辖）。

**提供方的 GPU 交叉核对是一个代理（PROXY），不是保证。** 参考 grade 运行在一台 **CUDA GPU，而非 910B** 上（CUDA 峰值表被标注为 "self-check, not for 910B scoring"）。所以 GPU oracle 等价性证明了抽取出的 GPU-Triton 内核**在 CUDA 上**能运行并匹配其 PyTorch oracle——即忠实抽取加上正确的 fp32 模型逻辑。它**并不**证明在 Ascend 芯片上的复现。一个单独的 910B 阶段负责那件事。在真实 NPU 上仍然不同的有：fp8 算术（软件模拟对 CUDA 硬件 fp8）、Cube-对-Vector 的舍入与累加器行为、`triton-ascend` JIT codegen，以及相对（学术规格、暂定的）vector 峰值的绝对吞吐。NPU 运行时的接缝是存在的，但在非设备上被打了桩——`NpuBackend.event_timer` 在没有 `torch_npu` 时返回 `None`，而 NPU profiler 抛出 *"only available on the 910B machine"*——所以内核事件 / 非默认流信号只有在真实硬件上才可观测。

---

## 10. 提供方侧的 oracle 对齐保障

CPU 侧的 oracle 必须复现*被冻结的 GPU 内核实际计算的内容*，而非数学上理想的结果。两个机制解决符号相反的问题：TF32 截断是对硬件张量核 `tl.dot` 的**正确**模拟；窄浮点截断则是 CPU 解释器**错误**的行为，anchor 必须绕开它。

### TF32 模拟 —— 位运算

一个 fp32 值有 23 个尾数位；TF32 张量核格式保留完整的 8 位指数，但只保留最高的 **10** 个尾数位。oracle 辅助函数恰好丢弃低 13 位（`23 − 13 = 10`）：

```python
def _to_tf32(x):  # in the task's own oracle_pytorch.py
    return (x.contiguous().view(torch.int32) & ~((1 << 13) - 1)).view(torch.float32)
```

`~((1<<13)-1)` 清除尾数位 0–12，留下位 13–22（= 10 位）。它从不触及符号位或指数位，所以这是**向零截断**，而非 RNE。截断的选择起承重作用：Triton 默认的 `tl.dot` 截断低尾数位，而 cuBLAS 做 RNE 舍入，只有截断才能复现实测的硬件（某个任务的 oracle 记录了截断下 GPU-anchor 匹配于 `max_rel ≈ 21.4`，而 RNE 本会小约 3 倍）。容差**保持不变**——TF32 模拟使 oracle 与内核的真实算术对齐，于是残余的差距只是 fp32 累积顺序的噪声。`_to_tf32` 被包裹在任何运行*未固定（unpinned）* fp32 `tl.dot` 内核（无 `allow_tf32=False`、无 `input_precision="ieee"`）的矩阵乘的**两个**操作数外面；它按任务存在于 oracle 中（已确认的示例：`L2_14`、`L2_19`、`L2_21`、`L3_21`、`L3_22`、`L3_27`）。

### 窄浮点 anchor —— 解释器误舍入

切断验证 anchor 运行在 `TRITON_INTERPRET=1`（CPU，无 GPU）下。解释器会误舍入每一次**窄化**浮点存储：`fp32 → bf16/fp16` **向零截断**而非 RNE（`100.3 → 100.0` 而非 `100.5`），而 `fp32 → fp8` 则双向误舍入。所以对于任何 oracle 输出为 fp8/fp16/bf16 的任务，无论它通过还是失败，interpret 判定都不可信——一个正确的内核可能假失败，而一个错误的内核可能因截断进入 oracle 值而假通过。runner 将此标记为 `narrow_float_output: True`（它检测 oracle 输出 dtype 字符串是否包含 `float8` 或 `float16`，后者也匹配 `bfloat16`）。

当该标志被置位**且有 GPU 可用**时，父进程用任务**未改动**的容差在真实设备上把**同一个** anchor **重跑一次**，并采纳 GPU 判定（解释器结果被丢弃；`interpret=False` 无法重新进入该分支，所以这次救援至多触发一次）。救援只纠正**舍入模型**，绝不改动比较——一个真正错误的算子在相同的 atol/rtol 下仍然失败，而一个真实的数值/dtype/结构不匹配会返回失败，永远不会到达救援路径。

*主机依赖，如实陈述：* 救援以存在 GPU 为门控。在部署中，oracle 一致性 anchor 运行在 CUDA daemon 上，所以救援会启动。在无 GPU 的机器上，该标志无法通过重跑来解决——代码返回原始的 interpret 判定，而不是伪造一个 CPU 结果；而真正 fail-closed 的路径（`consistent: false`、"no CUDA device"）属于在 CUDA 缺失时显式的 `interpret=False` 请求。因此一个窄浮点任务的正确性按部署约定取决于在真实硬件上打分——而非取决于解释器。

### 无静默 torch 回退

一个不产生任何真实设备内核事件的候选实现，即使数值正确也会失败（§2）——迁移必须在设备上运行一个内核。

---

## 11. 输出哨兵（sentinels）—— 已声明，尚未接线

`tolerance_policy.yaml` 携带一个按 dtype 的 `sentinels:` 表（其自带注释写道 "output buffer poisoning detection"）：float → `NaN`，int8 → `−128`，int16 → `−32768`，int32 → `−2147483648`，bool → `0xAA`。其设计意图是标准的"先毒化目标区"模式——用一个内核绝不会写入的值预填充输出区域，运行它，然后标记任何仍持有该哨兵值的区域（堵上"把陈旧 / 侥幸为零的缓冲区读作通过"的漏洞）。

**目前没有任何代码消费它。** `load_policy()` 只读取 `classes` 块并忽略 `sentinels`；没有模块导入任何哨兵常量。也不存在可供毒化的缓冲区：内核 contract 是**基于返回的，而非基于填充的**（`npu_triton_impl(inputs) -> dict[str, Tensor]`）——候选实现分配并返回它自己的张量，runner 从不向它交付一个预分配的目标。所以以当前的 harness 形态，一个"未写入区域哨兵"方案没有任何东西可以挂靠。

runner 用以对抗一个跳过某区域的内核的，反而是：shape 不匹配拒绝、对返回数据的严格 ULP / bit-exact 数值比较，以及 `_check_full` 中的 NaN/Inf **位置**门（`inf_must_match`/`nan_must_match` 默认为 true，在有限 `allclose` 之前运行）。哨兵表最好被理解为面向未来某个缓冲区写入式 harness（例如一次写入调用方提供的缓冲区的 Ascend 侧运行）的设计意图，而非当前打分器执行的某项检查。

---

## 12. 打分在数据流中的位置

对于一个 shape，在受信任的父进程中：

```
build_inputs(shapes, shape_id, seed)
   → run candidate in the isolated subprocess (correctness pass)
   → _resolve_golden(task tier × mode) picks the reference
   → tolerance.compare(actual, golden, load_policy().get(tolerance_class))
   → per-shape correctness_passed
```

逐 shape 的结果按 §2 聚合。隔离保证（无 golden 泄漏、仅 safetensors 的 IPC、审计钩子、正确性/计时通道分离）以及 `_resolve_golden` 路由（oracle tier × mode）在 [`ARCHITECTURE.md` §3](../ARCHITECTURE.md) 中描述。本文档只涵盖在 `compare` 步骤所应用的精度标准。
