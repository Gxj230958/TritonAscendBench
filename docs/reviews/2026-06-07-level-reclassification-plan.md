I have everything needed: the 101-task data, the documented vs branch rubric, the branch's full-renumber scheme (every task_id mutates; 70 other_op.py files), the loader/golden/test coupling, and the L2_23/24 deprecation that reconciles the 98-shared / 3-current-unique / 2-branch-unique arithmetic. Here is the plan.

---

# Level 重分级方案 — TritonAscendBench 101 题库

REVIEW-ONLY：本文档只提出方案与逐题建议，不修改任何 `meta.yaml` / 目录 / 代码。所有落地动作待用户在 §8 的决策点拍板后再做。

---

## 1. 背景与目标

### 1.1 为什么要重分级

当前题库的分级（`docs/kernel-select.md` §3）用的是一套**较松、口语化**的标准：

> L1 = 基础单算子、纯 VV / 单 stage；L2 = 复合 fused / CV 混合；L3 = 完整子系统 / 高复杂度。

这套标准有两个结构性问题：

1. **"复合 fused / CV 混合" 把两件事混为一谈**。一个 kernel 可以是"多算子融合"但完全不碰 Cube（如 `layer_norm` 两次 reduce + affine），也可以是"单算子但用 `tl.dot`"（如 `bmm`）。旧 L2 定义把这两类塞进同一格，导致 VV 复杂算子和纯 Cube GEMM 难度被当成同级——但在 910B 迁移语境里它们的难点完全不同（一个是 UB 内多 reduce 编排，一个是 `tl.dot` 16 对齐 + padding）。
2. **缺少可判定的主轴**。"完整子系统"是结果描述不是判据，reviewer 无法机械复核某题该不该是 L3。

`docs/kernel-select.md` 本身已经漂移：§1 写的是 "**100** 个 kernel，L1 40 · L2 **38** · L3 22"，而实际 scored 语料是 **101**（L1 40 / L2 39 / L3 22）。重分级必须顺带把这个对账修正到 101。

### 1.2 目标：对齐 910B 迁移难度

`clean-kernel-contract-ops` 分支提出了一套**更锋利、可判定**的 rubric（见 §2），并据此把 101 语料中 **42 / 98 个共享算子**搬了级（L1=35 / L2=35 / L3=30）。它的核心是：**这是一套 Ascend-910B 迁移难度分类，不是算子计数分类**。"good" 的定义就是——**同一级里的题，迁到 910B `triton-ascend` 的真实工程难度应当同档**：

- L1：纯 Vector-Core，搬一遍就能跑通；
- L2：要么 UB 内多 reduce/elemwise 编排，要么 `tl.dot` 对齐/padding，要么困难非连续访存——单一难点；
- L3：CV 融合流水线（attention/SSM = dot + reduce + 状态更新）/ 复杂 Cube（MoE 路由+matmul）/ 卷积——多难点叠加。

本方案做的事：**独立从真实 kernel 源码重新判级**，逐题核对分支的 42 个搬动，给出推荐分布，并把分歧、norm 张力、3 个 current-unique 题、fp8-cube 排除候选全部仲裁清楚。

---

## 2. 采用的 Level Rubric（目标标准）

直接采用 `clean-kernel-contract-ops` §3 的锋利 rubric（**不另造**）：

**主轴 — VV vs CV：这个 kernel 用没用 Cube / `tl.dot`？**
**次轴 — 复杂度：单 stage，还是 多 reduce / scan / 状态递推 / 多阶段流水？**

| 级别 | 定义（判据） | 典型形态 |
|---|---|---|
| **L1** | 纯 Vector-Core：elementwise / 单 reduce / 数据搬运 / 索引。语义直接，**单 stage**。 | bias、KV-cache 索引、reorder/permute/combine、concat-cast、纯 rmsnorm（单 reduce）、单 stage 量化、单 stage scan/state-passing |
| **L2** | 满足以下**任一**：① 复杂 Vector（**多 reduce + elemwise 融合**）；② 纯 Cube（GEMM/BMM matmul、`tl.dot`）；③ 困难非连续访存（gather/scatter/router/pack）。仍是**单一难点**。 | layer_norm（mean+var 双 reduce）、activation+mul、sampling topk/min_p、W8A8 GEMM、LoRA-A shrink、BMM、MoE router-align、bitmatrix pack |
| **L3** | 满足以下任一：① **CV 融合流水线**（attention/SSM = dot + vector/reduce + 状态更新）；② **复杂 Cube**（MoE expert 路由 + matmul 融合）；③ **卷积**（conv1d）。**多难点叠加 / 跨 chunk 状态**。 | flash/paged/unified attention、fused/expert MoE、mamba/GDN/KDA chunk pipeline、recurrent gated delta、causal_conv1d、LoRA-B fused gemm chain |

**判级算法（reviewer 可机械复核）：**

1. 源码里有 `tl.dot` 吗？
   - **有** → 至少 L2（纯 Cube/GEMM = L2；若同时叠加 online-softmax/state/多阶段融合 = L3）。
   - **没有** → 看次轴：
     - 单 reduce / 单 stage / 纯搬运 → **L1**；
     - 多 reduce + elemwise 融合，或困难 gather/scatter/router/pack → **L2**；
     - 跨时间步状态递推（scan/SSM）或卷积 → **L3**（次轴可单独把 VV 题顶到 L3，**不经 Cube**）。

**Tie-breaker / 例外（rubric 显式列举，优先于上面的机械判据）：**

- `sampling topk/min_p` —— 即便只有单 reduce，**显式枚举为 L2**（`min_p` 单 `tl.max` 也算 L2）。
- `conv1d` —— 即便无 `tl.dot`，**显式枚举为 L3**（卷积轴）。
- `activation+mul`（SwiGLU/GeGLU）—— 即便无 reduce/dot，**显式枚举为 L2**。
- `combine` / `reorder/permute` / `concat-cast` / `pure rmsnorm` —— **显式枚举为 L1**。
- 允许**人工难度校正**（rubric 末句："允许人工难度校正"）——但校正必须有源码依据，不能反向覆盖主轴。

**norm 张力的官方裁决（rubric 内蕴）：** `rms_norm`（单 reduce sum(x²)）= L1；`layer_norm`（mean **和** variance = 双 reduce）= L2。这条线贯穿全题库，是本方案最高频的判据。

---

## 3. 三方对比总览

| 维度 | Current（线上 101） | Branch `clean-kernel-contract-ops`（100） | **Recommended（本方案 101）** |
|---|---|---|---|
| L1 | 40 | 35 | **36** |
| L2 | 39 | 35 | **35** |
| L3 | 22 | 30 | **30** |
| 合计 | 101 | 100 | 101 |

**语料口径对账（关键）：**

- Current 与 Branch **共享 98 个算子**。
- Current 多出 **3 个 current-unique**：`L2_39_scaled_mm` / `L2_40_gemma_qkv_rmsnorm` / `L2_41_selective_state_update`（分支没有这三题，故其 `branch_level=NEW`）。→ 98 + 3 = **101**。
- Branch 多出 **2 个 branch-unique**：`fwd_kv_parallel` / `fwd_kv_reduce`（lightning-attn 拆分）。→ 98 + 2 = **100**。
- 这 2 个 branch-unique 在**线上语料里就是 `L2_23` / `L2_24`，已因 vLLM `lightning_attn.py:444` 的 cross-block decay **off-by-one** 上游 bug 被移入 `tasks/_deprecated/`**（修正版在 `tasks/_corrected/`，证明见 `docs/verification/L2_24_lightning_attn_offbyone_proof.py`）。所以它们**不在 101 scored 表里**——只在 §5 作为信息项仲裁，不计入分布。

**三方一致的题数：** 共享 98 题中分支搬动 42 题、未动 56 题；本方案对未动题中的 **55 题认可**，唯一例外是 `L1_20_log_softmax`（cur=branch=L1，但它与 `layer_norm` 同为 double-reduce，本方案为消除 rubric 内部矛盾升 L2，见 §5.7）。因此 **55 题 `cur == branch == recommended` 三方完全一致**。

**Recommended vs Branch 的分歧（仅 7 个共享题）：**

| 题 | branch | recommended | 方向 |
|---|---|---|---|
| L1_18_fused_softcap | L2 | **L1** | 反对过度提升 |
| L1_25_merge_attn_states | L2 | **L1** | 反对过度提升 |
| L1_26_merge_state | L2 | **L1** | 反对过度提升 |
| L1_30_act_quant | L2 | **L1** | 反对过度提升 |
| L2_15_fused_dual_residual_rmsnorm | L1 | **L2** | 反对过度降级 |
| L2_31_moe_align_block_size_stage2 | L1 | **L2** | 反对过度降级 |
| L2_34_silu_mul_static_tensorwise_quant…cutlass_moe | L1 | **L2** | 反对过度降级 |

净效应（相对 current 40/39/22）：L1 −4、L2 −4、L3 +8 → **L1 36 / L2 35 / L3 30**。其中 L1_20 的一致性升级（§5.7）把分支式 37/34/30 调整为 36/35/30。3 个 current-unique 由本方案独立定级（L2_39→L2、L2_40→L1、L2_41→L3）。

**Recommended vs Current 的分歧：** 38 题（认可分支 35 个搬动 + L1_20 一致性升级 + 2 个 current-unique 重定级 L2_40/L2_41）；63 题维持线上级别不变。

---

## 4. 逐题建议表（全 101 题，按 Recommended 级别分组）

列含义：`cur`=线上级别，`br`=分支级别（NEW=current-unique，分支无此题），`rec`=**本方案推荐**，`T`=VV/CV，`flag`=与分支分歧 / 低置信 / 类型注意。**置信度**：未标 flag 的为 `[confirmed]` 高一致；`±` 为 borderline `[likely]`；`✗br` 为与分支分歧。

### 4.1 Recommended = L1（36 题）

| task | cur | br | rec | T | paradigm | flag |
|---|---|---|---|---|---|---|
| L1_01_bias | L1 | L1 | L1 | VV | 按 index 加 logit bias，单 stage elementwise | |
| L1_02_compute_slot_mappings | L1 | L1 | L1 | VV | 算 KV-cache slot 映射索引（元数据搬运） | |
| L1_03_convert_req_index_to_global_index | L1 | L1 | L1 | VV | req→global 索引重映射 | |
| L1_04_copy_page_indices | L1 | L1 | L1 | VV | 拷贝 page index 表（搬运） | |
| L1_05_gather_block_tables | L1 | L1 | L1 | VV | gather block-table 行（索引拷贝） | |
| L1_06_reshape_and_cache_kernel_flash | L1 | L1 | L1 | VV | reshape+scatter K/V 进 paged cache | |
| L1_07_dequantize_k_cache_fast | L1 | L1 | L1 | VV | K-cache int→fp dequant（elemwise scale） | |
| L1_08_dequantize_k_cache_paged | L1 | L1 | L1 | VV | paged K-cache dequant（elemwise+gather） | |
| L1_09_quantize_k_cache_fast | L1 | L1 | L1 | VV | K-cache fp→int quant（per-group scale） | |
| L1_10_per_token_group_quant_int8 | L1 | L1 | L1 | VV | per-token-group int8 quant（单 amax+scale） | quant 同族锚 |
| L1_11_awq_dequantize | L1 | L1 | L1 | VV | AWQ 4bit unpack+dequant | |
| L1_12_rms_norm | L1 | L1 | L1 | VV | **单 reduce** RMSNorm（rubric L1 锚） | norm 锚 |
| **L1_18_fused_softcap** | L1 | **L2** | **L1** | VV | 单 stage elementwise：x=tanh(x/cap)·cap，in-place | ✗br 过度提升 |
| **L1_25_merge_attn_states** | L1 | **L2** | **L1** | VV | LSE 加权 combine 两路 attn 输出；max 是两标量非 axis-reduce | ✗br 过度提升 |
| **L1_26_merge_state** | L1 | **L2** | **L1** | VV | 同上更简版 combine，无 FP8/双路 | ✗br 过度提升 |
| L1_27_apply_grammar_bitmask | L1 | L1 | L1 | VV | 按 grammar bitmask 屏蔽 logits（bit unpack+mask） | |
| L1_29_prepare_eagle_inputs | L1 | L1 | L1 | VV | 构建投机解码输入张量（index/gather/scatter） | |
| **L1_30_act_quant** | L1 | **L2** | **L1** | VV | block-wise fp8 quant：**单** tl.max(axis=1) amax+scale | ✗br 过度提升；与 L1_07~11 同族 |
| L1_31_triton_mrope_forward | L1 | L1 | L1 | VV | 多模态 RoPE 旋转（elemwise sin/cos） | |
| L1_32_create_flashinfer_kv_indices_triton | L1 | L1 | L1 | VV | 构建 flashinfer KV 索引数组 | |
| L1_33_create_flashmla_kv_indices_triton | L1 | L1 | L1 | VV | 构建 flashMLA KV 索引 | |
| L1_34_transform_index_page_table_decode | L1 | L1 | L1 | VV | decode page-table 索引变换 | |
| L1_35_compute_src2dst_triton | L1 | L1 | L1 | VV | 算 MoE src→dst scatter 映射 | |
| L1_36_compute_seg_indptr_triton | L1 | L1 | L1 | VV | 算 segment indptr offsets | |
| L1_37_compute_problem_sizes_w4a8 | L1 | L1 | L1 | VV | 算 per-group GEMM problem sizes（元数据） | |
| L1_39_state_passing_fwd | L1 | L1 | L1 | VV | SSM 跨 chunk state 传递（单 stage 衰减累加） | scan L1 锚 |
| L1_40_chunk_cumsum_fwd | L1 | L1 | L1 | VV | chunk-local cumsum（单 stage prefix） | scan L1 锚 |
| L2_16_fused_rmsnorm | L2 | L1 | **L1** | VV | **单** reduce RMSNorm（rubric L1 'pure rmsnorm'） | norm 降级 ✓ |
| L2_18_fused_gdn_gating | L2 | L1 | **L1** | VV | 纯 elementwise GDN gating（softplus/exp/sigmoid） | ✓ 降级 |
| L2_27_deepep_permute_triton | L2 | L1 | **L1** | VV | 行 permute/scatter：out[src2dst]=x[i] | ✓ 降级 |
| L2_28_deepep_post_reorder_triton | L2 | L1 | **L1** | VV | gather+topk 加权 combine（单 reduce） | ± 单 reduce combine |
| L2_32_experts_combine | L2 | L1 | **L1** | VV | combine：sum_k moe[k]+mlp，单小 reduce | ✓ 降级 |
| L2_35_pre_reorder_triton_kernel_for_cutlass_moe | L2 | L1 | **L1** | VV | scaled scatter/permute（专家 gate skip） | ✓ 降级 |
| L2_36_post_reorder_triton_kernel_for_cutlass_moe | L2 | L1 | **L1** | VV | gather+topk 加权+routed scale（单 reduce） | ± gather combine |
| L2_38_concat_and_cast_mha_k | L2 | L1 | **L1** | VV | concat+cast：k_nope/k_rope 拼相邻切片 | ✓ rubric L1 锚 |
| **L2_40_gemma_qkv_rmsnorm** | **L2** | NEW | **L1** | VV | Q/K/V per-head **单**统计 RMSNorm（rsqrt(mean(x²))） | ✗cur current-unique；norm L1 |

### 4.2 Recommended = L2（35 题）

| task | cur | br | rec | T | paradigm | flag |
|---|---|---|---|---|---|---|
| L1_13_layer_norm_fwd | L1 | L2 | **L2** | VV | layer_norm：mean+var **双 reduce**+affine+gate | norm 提升 ✓（rubric 锚） |
| **L1_20_log_softmax** | L1 | **L1** | **L2** | VV | vocab log-softmax：max+sumexp **双 reduce**+subtract | ✗cur ✗br 一致性升级（§5.7，与 layer_norm 同 double-reduce） |
| L1_14_layer_norm_fwd_1pass | L1 | L2 | **L2** | VV | single-pass layer_norm，双 reduce 融一次 launch | norm 提升 ✓ |
| L1_15_act_and_mul | L1 | L2 | **L2** | VV | gated activation+mul（SwiGLU/GeGLU）+专家路由 | rubric 显式 L2 |
| L1_16_silu_and_mul | L1 | L2 | **L2** | VV | SwiGLU：silu(x1)·x3 | rubric 显式 L2 |
| L1_17_gelu_and_mul | L1 | L2 | **L2** | VV | GeGLU：gelu(x1)·x3（erf） | rubric 显式 L2 |
| L1_19_topk_log_softmax | L1 | L2 | **L2** | VV | log-softmax（max+sumexp 双 reduce）+topk gather | 提升 ✓ |
| L1_21_gumbel_sample | L1 | L2 | **L2** | VV | gumbel-max：argmax-reduce+RNG+多 block combine | 提升 ✓ |
| L1_22_min_p | L1 | L2 | **L2** | VV | min-p：单 tl.max+阈值 mask+gather | ± 单 reduce 但 rubric 显式枚举 |
| L1_23_penalties_and_temperature | L1 | L2 | **L2** | VV | 罚项 bin-count 累加+packed bitmask unpack | 提升 ✓ |
| L1_24_rejection_sample | L1 | L2 | **L2** | VV | 投机 rejection：有状态 scan+按 token_id gather/scatter | ± 标量序列但 gather+state |
| L1_28_pack_bitmatrix | L1 | L2 | **L2** | VV | bitmatrix pack：reduce_or+scatter（+stage2 sort/scan） | rubric pack=L2 |
| L1_38_silu_mul_fp8_quant_deep_gemm | L1 | L2 | **L2** | VV | silu+mul+per-group amax+fp8 quant（MoE pre-gemm，**无 dot**） | 提升 ✓；非 fp8-cube |
| L2_01_bmm_chunk_fwd | L2 | L2 | L2 | CV | 纯 BMM chunk（tl.dot） | BMM L2 锚 |
| L2_05_chunked_lora_shrink | L2 | L2 | L2 | CV | LoRA-A shrink grouped GEMM | |
| L2_08_sgemm_lora_a | L2 | L2 | L2 | CV | 单 slice LoRA-A SGMV GEMM | |
| L2_10_w8a8_block_int8_matmul | L2 | L2 | L2 | CV | W8A8 block-scaled INT8 GEMM | |
| L2_11_w8a8_triton_block_scaled_mm | L2 | L2 | L2 | CV | W8A8 block-scaled matmul | |
| L2_12_awq_gemm | L2 | L2 | L2 | CV | AWQ 4bit dequant+GEMM | |
| L2_13_bmm | L2 | L2 | L2 | CV | batched matmul | |
| L2_14_matmul_kernel_persistent | L2 | L2 | L2 | CV | persistent-tile GEMM | |
| **L2_15_fused_dual_residual_rmsnorm** | L2 | **L1** | **L2** | VV | **双** tl.sum（两 normalize pass）+residual+双 weight+中间 store | ✗br norm 张力（见 §5） |
| L2_17_layer_norm_gated_fwd | L2 | L2 | L2 | VV | gated layer_norm（mean+var reduce+gate） | |
| L2_22_linear_attn_decode | L2 | L2 | L2 | VV | 线性注意力 decode（逐步 state，无 dot） | |
| L2_25_fwd_kernel_ep_gather | L2 | L2 | L2 | VV | 专家并行 gather（困难非连续访存） | |
| L2_26_fwd_kernel_ep_scatter_1 | L2 | L2 | L2 | VV | 专家并行 scatter（困难非连续访存） | |
| L2_29_fused_moe_router_cudacore | L2 | L2 | L2 | VV | MoE router softmax+topk（**无 dot**，多 reduce） | 与 L2_30 成对（无 dot 故 L2） |
| **L2_31_moe_align_block_size_stage2** | L2 | **L1** | **L2** | VV | 4-stage：count→cumsum scan→padded cumsum→**sorted scatter** | ✗br router/scatter=L2 |
| **L2_34_silu_mul_static_tensorwise_quant…cutlass_moe** | L2 | **L1** | **L2** | VV | silu+mul+per-group **absmax reduce**+fp8 quant（vector，无 dot） | ✗br；与 L2_37 同族 |
| L2_37_silu_and_mul_post_quant | L2 | L2 | L2 | VV | silu+mul+post quant | L2_34 的姐妹锚 |
| **L2_39_scaled_mm** | **L2** | NEW | **L2** | CV | W8A8：int8 A@B→int32 acc（tl.dot）+per-token/channel scale | current-unique；fp8 排除已核否 |
| L3_07_correct_attn_cp_out | L3 | L2 | **L2** | VV | 跨 rank logsumexp（max+sum reduce）+输出 rescale（**无 dot**） | 降级 ✓ |
| L3_12_moe_mmk | L3 | L2 | **L2** | CV | batched per-expert grouped GEMM（K-loop tl.dot，token 预分组） | 降级 ✓；fp8 路径被 launcher 禁用 |
| L3_21_solve_tril_16x16 | L3 | L2 | **L2** | VV | 16×16 下三角前代（单 tl.sum，**graded 入口无 dot**） | 降级 ✓ |
| L3_22_chunk_local_cumsum_vector | L3 | L2 | **L2** | CV | cumsum-via-GEMM：单 tl.dot(三角 ones, chunk) | 降级 ✓（纯 Cube） |

### 4.3 Recommended = L3（30 题）

| task | cur | br | rec | T | paradigm | flag |
|---|---|---|---|---|---|---|
| L2_02_chunk_scan_fwd | L2 | L3 | **L3** | CV | Mamba2 SSD chunk-scan：2× tl.dot+decay+state | 提升 ✓ |
| L2_03_chunk_state_fwd | L2 | L3 | **L3** | CV | Mamba2 chunk-state：decay 加权 tl.dot(x,b) | ± 单 decay-weighted matmul |
| L2_04_chunked_lora_expand | L2 | L3 | **L3** | CV | chunked LoRA-B expand：multi-slice+permutation gather+dot | LoRA-B 链 ✓ |
| L2_06_qkv_lora_b | L2 | L3 | **L3** | CV | LoRA-B 打包 q/k/v 3 sgemms（3D grid+gather） | LoRA-B 链 ✓ |
| L2_07_gate_up_lora_b | L2 | L3 | **L3** | CV | LoRA-B 打包 gate/up 2 sgemms | LoRA-B 链 ✓ |
| L2_09_sgemm_lora_b | L2 | L3 | **L3** | CV | 单 slice LoRA-B SGMV（gather+scale+residual add） | ± LoRA-B 组**最弱** L3 |
| L2_19_fused_sigmoid_gating_delta_rule_update | L2 | L3 | **L3** | VV | recurrent gated-delta-rule state update（T-loop+rank-1，**无 dot**） | 经 scan/state 到 L3，非 Cube |
| L2_20_causal_conv1d_update | L2 | L3 | **L3** | VV | depthwise causal conv1d update（滑窗 FMA+conv-state，**无 dot**） | 经卷积轴到 L3 |
| L2_21_causal_conv1d_fwd | L2 | L3 | **L3** | VV | causal conv1d prefill（滑窗 FMA+cache 管理，**无 dot**） | 经卷积轴到 L3 |
| L2_30_fused_moe_router_tensorcore | L2 | L3 | **L3** | CV | MoE router GEMM（tl.dot）+softcap+top1/2 选择 | 复杂 Cube ✓ |
| L2_33_fwd_grouped_kernel_stage1_rope | L2 | L3 | **L3** | CV | MLA decode attn stage1：3× tl.dot+online softmax+RoPE | attention CV ✓ |
| **L2_41_selective_state_update** | **L2** | NEW | **L3** | VV | Mamba2 selective state update：state 递推+sum(state·C)+gating（**无 dot**） | ✗cur current-unique；SSM L3（见 §5，borderline） |
| L3_01_kernel_unified_attention_3d | L3 | L3 | L3 | CV | unified flash attention（dot+online softmax） | |
| L3_02_kernel_paged_attention_2d | L3 | L3 | L3 | CV | paged attention（dot+online softmax） | |
| L3_03_chunk_state_varlen | L3 | L3 | L3 | CV | varlen Mamba chunk-state（dot+decay+state） | |
| L3_04_fwd | L3 | L3 | L3 | CV | flash attention forward（dot+softmax） | |
| L3_05_fwd_kernel_alibi | L3 | L3 | L3 | CV | flash prefix-prefill+ALiBi（2-stage dot+online softmax+fp8 KV） | |
| L3_06_fwd_grouped_kernel_stage1 | L3 | L3 | L3 | CV | grouped flash-decode stage1（多 dot+online softmax+paged KV） | |
| L3_09_fused_moe | L3 | L3 | L3 | CV | fused MoE expert GEMM 流水 | |
| L3_10_fused_moe_kernel_gptq_awq | L3 | L3 | L3 | CV | fused MoE+GPTQ/AWQ dequant GEMM | |
| L3_11_expert_triton | L3 | L3 | L3 | CV | per-expert grouped GEMM driver | |
| L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | L3 | L3 | L3 | CV | GDN chunk pipeline（dot+state） | |
| L3_14_fused_recurrent_gated_delta_rule_fwd | L3 | L3 | L3 | VV | recurrent GDN forward（**无 dot**，scan+state+gating） | VV+state→L3 |
| L3_15_fused_recurrent_gated_delta_rule_update_fwd | L3 | L3 | L3 | VV | recurrent GDN decode update（**无 dot**，state cache+tree mask） | VV+state→L3 |
| L3_16_chunk_fwd_kernel_o | L3 | L3 | L3 | CV | chunk 输出 matmul（dot） | |
| L3_17_chunk_gla_fwd_kernel_o | L3 | L3 | L3 | CV | GLA chunk 输出（dot+decay） | |
| L3_18_chunk_scaled_dot_kkt_fwd | L3 | L3 | L3 | CV | chunk K@Kᵀ scaled dot | |
| L3_19_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter | L3 | L3 | L3 | CV | KDA intra/inter chunk dot pipeline | |
| L3_20_recompute_w_u_fwd | L3 | L3 | L3 | CV | recompute w/u 因子（dot 链） | |
| L3_23_merge_16x16_to_32x32_inverse | L3 | L3 | L3 | CV | block 三角逆合并（tl.dot） | |

**完整性核对：** L1 = 27(L1源，L1_20 升 L2 后) + 9(L2源含L2_40) = **36**；L2 = 13(L1源含L1_20) + 18(L2源) + 4(L3源) = **35**；L3 = 12(L2源含L2_41) + 18(L3源) = **30**。合计 **101**。✓（L2_23/L2_24 已 deprecated，不在表内。）

---

## 5. 分歧与仲裁

### 5.1 与分支分歧的 7 个共享题（最终采纳本方案）

**A. 反对分支「L1→L2 过度提升」（4 题，rec 退回 L1）**

1. **L1_18_fused_softcap → L1**。`[confirmed]` graded kernel 只做 `x/=cap; x=libdevice.tanh(x); x*=cap`，in-place 单算子 elementwise。无 reduce、无第二张量乘、无 gate、无非连续访存、无 `tl.dot`——L2 三条判据**全不满足**，比 `bias`（L1 锚）还简单，`difficulty=low`。分支把它当 activation 提升，但它不是 gated activation+mul（没有第二张量），不适用 rubric 的 `activation+mul=L2` 例外。**留 L1。**

2. **L1_25_merge_attn_states / 3. L1_26_merge_state → L1**。`[confirmed]` 两题都是 LSE 加权 combine：`tl.maximum(p_lse, s_lse)` 是**两个标量**的 max，不是 axis-reduce；head-dim 连续 load，无 gather/scatter/pack；无 `tl.dot`。rubric 显式把 `combine` 列为 L1。exp/log LSE rescale 让它成为"富 L1"，但**不增加 reduce 或 stage**。L2 三条全不满足。**留 L1。**

4. **L1_30_act_quant → L1**。`[confirmed]` 单 `tl.max(axis=1)` per-group amax + scale/clamp + scale store。**单 reduce 是 L1 特征**，无 dot、无非连续访存。与 `L1_07/08/09/10/11`（尤其 `L1_10_per_token_group_quant_int8`，同 per-group amax+scale）结构一致，**那一族全留 L1**——单独提升 act_quant 破坏同族一致性。注意：`min_p`（L1_22）单 reduce 仍是 L2 因为 rubric **显式枚举 sampling**，而 quant **没有**这个例外。**留 L1。**

**B. 反对分支「L2→L1 过度降级」（3 题，rec 退回 L2）**

5. **L2_15_fused_dual_residual_rmsnorm → L2**。`[confirmed]` **两次** `tl.sum`（两 normalize pass）+ residual add + 两次 weight apply + 中间 store。结构上**等同甚至重于 layer_norm**——而分支恰恰把 layer_norm（L1_13/14）从 L1 提到 L2。分支给的理由"rmsnorm=1 reduce"对 dual-residual 变体**不成立**（它是 2-reduce）。**退回 L2，消除内部矛盾**（见 5.3 norm 张力）。

6. **L2_31_moe_align_block_size_stage2 → L2**。`[confirmed]` graded `run()` 驱动 4-kernel 流水：count → cumsum scan(stage2) → padded-block cumsum(stage3) → 数据依赖 **sorted scatter**(stage4)。这是 MoE routing/alignment 的经典 router 算子，rubric 显式列 `router/scatter=L2`。分支当"纯搬运 L1"**误判了一个多阶段 sort/scan/scatter 算法**。**退回 L2。**

7. **L2_34_silu_mul_static_tensorwise_quant…cutlass_moe → L2**。`[confirmed]` 不是搬运：SiLU + elemwise mul + per-group **absmax reduce** + fp8 quant（fp32 compute、fp8 store，**vector quant 非 fp8-cube，无 dot**）。= activation+mul + reduce + quant = L2。降到 L1 还**自相矛盾**：分支把无 reduce/quant 的 `act_and_mul/silu_and_mul`（L1_15/16/17）提到 L2，又把近乎同款的姐妹 `L2_37_silu_and_mul_post_quant` 留 L2。**退回 L2。**

### 5.2 三个 current-unique 题（分支无此题，本方案独立定级）

- **L2_39_scaled_mm → L2 [confirmed]（维持 cur）**。纯 Cube W8A8 GEMM：`tl.dot(a,b,acc)` int8→int32，per-token/per-channel scale + bias epilogue，单 stage 无融合。**fp8-cube 排除已主动核否**：核对 `shapes.yaml`/`contract.md`，A/B 为 **int8**、acc int32、输出 bf16（`int8_w8a8` tolerance）——是 **W8A8 整数 GEMM**，rubric 显式列为 L2 例，§6 的 fp8-cube 排除**不适用**。维持 L2。

- **L2_40_gemma_qkv_rmsnorm → L1 [confirmed]（cur L2 误分类）**。`_gemma_qkv_rmsnorm_kernel` 无 dot，per-head 只算**单**统计 `rsqrt(mean(x²))` 并按 per-tensor weight 缩放。QKV/多头是**同一个单 reduce 的 launch 融合**，不是第二个统计量，不满足 L2 的"多 reduce"。与分支对每种 rmsnorm 一律 L1（L1_12 / L2_15 / L2_16）一致，L2 保留给 layer_norm 的真双 reduce。**降到 L1。** 想保 L2 的 reviewer 只能用"融合广度"论，而非 rubric 的 VV/复杂度轴——本方案不采纳该论点。

- **L2_41_selective_state_update → L3 [likely]（cur L2 偏低，borderline 标注）**。`_selective_scan_update_kernel` 无 dot，但有真 SSM 状态递推 `state=state*exp(A*dt)+(B*dt)*x` + `sum(state·C)` reduce + softplus/exp/sigmoid gating，per-token loop 内 in-place 改 state。按 rubric 次轴 scan/state + SSM/mamba 归宿，且与分支把 recurrent SSM update（L2_19、L3_14/15）放 L3 一致。**净判 L3。** **反向证据（故标 borderline）**：rubric 的 canonical L3 SSM 是"dot + reduce + state"而本题**无 dot**；graded fixture 跑 `seq_len=1`（单步，scan 不展开）。reviewer 若坚持 L2 属可辩——交由用户在 §8 拍板。

### 5.3 norm L1/L2 张力（统一裁决）

rubric 把 norm 分两类，本方案严格执行：

- **单 reduce（sum(x²)→rsqrt）= L1**：`L1_12_rms_norm`、`L2_16_fused_rmsnorm`、`L2_40_gemma_qkv_rmsnorm`。
- **双 reduce（mean **且** variance）= L2**：`L1_13/14_layer_norm_fwd*`、`L2_17_layer_norm_gated_fwd`。
- **争议点 `L2_15_fused_dual_residual_rmsnorm`**：它跑**两遍** rmsnorm（两次 `tl.sum`）+ residual + 中间 store——**reduce 计数上等同 layer_norm**。分支把它降 L1 与"提升 layer_norm 到 L2"**直接冲突**。本方案裁定 **L2_15 = L2**（多 reduce），`L2_16 = L1`（单 reduce）。这样"reduce 计数"成为贯穿 norm 全族的唯一判据，无内部矛盾。

### 5.4 VV-vs-CV 类型注记（L3 但非经 Cube，3 题不改级别只澄清标签）

`L2_19`（gated-delta-rule update）、`L2_20`/`L2_21`（causal_conv1d）三题 `type=VV` 且**无 `tl.dot`**，但分支把它们归在"CV-fusion L2→L3"组。**级别 L3 正确**（L2_19 经 state 递推、L2_20/21 经 rubric 显式 conv1d 条款），但**它们的 L3 来自次轴（scan/state、卷积），不是主轴 Cube**。建议：保 L3，但报告/文档里**不要把它们的 `dtype_class` 标成 CV**——保持 VV，难度归因写明"经次轴"。（任务方曾怀疑 L2_19 是 spurious L3，源码确认它是真 recurrent state pipeline，留 L3。）

### 5.5 fp8-cube 排除候选（§6 排除准则的逐项核查）

101 题中**没有任何一题是 graded fp8-cube**，但三处需登记核查结论：

- **L2_39_scaled_mm**：疑似 fp8 → **核否**，实为 int8 W8A8（见 5.2）。**不排除，合法 L2。**
- **L2_34 / L1_38**：输出 fp8-e4m3，但经 **vector quant（无 dot）**，是 910B 可测的 vector-core 量化，非 fp8-cube 算术。**不排除。**
- **L3_12_moe_mmk**：frozen 源码**含** fp8-cube 路径（`use_fp8_w8a8` 分支 `tl.dot` on fp8 w8a8，按 §5 在 910B 不可测），但 **launcher 硬禁用量化**（`use_fp8_w8a8=False`），graded 单元只跑 fp32 `acc+=tl.dot(a,b)`——可测 fp32 grouped GEMM。**不排除，降级 L2 成立。**

### 5.6 两个 branch-unique / deprecated 题（信息项，不计入 101）

`fwd_kv_parallel` / `fwd_kv_reduce`（线上 `L2_23` / `L2_24`，已移入 `tasks/_deprecated/`，因 vLLM `lightning_attn.py:444` cross-block decay **off-by-one** 上游 bug；修正版在 `tasks/_corrected/`）。若日后从 `_corrected/` 重新纳入：
- **fwd_kv_parallel → L2**：用 `tl.dot` 的 decay 加权 K^T@V（纯 BMM，无 softmax/state）= 纯 Cube L2。（分支也放 L2。）
- **fwd_kv_reduce → L1**：跨 block decayed exclusive prefix scan，**无 dot**、单 stage running accumulation，与 `L1_39_state_passing_fwd`/`L1_40_chunk_cumsum_fwd` 同类 = L1。**与分支分歧**（分支 `L2_46_fwd_kv_reduce` 放 L2，本方案判 L1）。lightning-attn 被拆成 Cube"parallel"半（L2）+ VV scan"reduce"半（L1）——若纳入应分两级，不应都 L2。

> 数据口径注：`/tmp` 特征表把 `fwd_kv_parallel/fwd_kv_reduce` 标 `branch_level=NEW`（按线上 task_id 匹配，分支重排后前缀对不上），与本节"分支有 `L2_45/L2_46`"的散文表述看似冲突——实为同一对算子的两套编号；二者已 deprecated，不计入 101，不影响 scored 分布。

### 5.7 L1_20_log_softmax：rubric 一致性升级（L1→L2）

`L1_20_log_softmax` graded kernel 做 **两次** axis-reduce：`tl.max`（kernel.py:36）求行最大 + `tl.sum`（kernel.py:42）求 sumexp，再 subtract 得 log-prob——与 `layer_norm`（L1_13/14，mean+var 双 reduce）**结构同级**。初版方案沿用线上/分支把它放 L1，但这与"本方案按 double-reduce 把 layer_norm 升 L2"**直接冲突**：同为 VV 双 reduce，不能一个升一个不升。

裁定：**L1_20 → L2** `[confirmed]`（codex + adversarial 双确认源码 double-reduce，均"找不到支持它留 L1 的源码区分"）。这样"double-reduce ⇒ L2"成为贯穿全题库的一致判据。它与 `L1_19_topk_log_softmax`（同 max+sumexp，已 L2）现在同级，差别只在 L1_19 额外有 topk gather——佐证 double-reduce 本身已足够 L2，topk 非必要条件。

净效应：分布从初版的 37/34/30 调整为 **36/35/30**。

---

## 6. clean-kernel / `other_op.py` 单算子拆分是否采纳

**这是与级别号正交的独立决策，建议与重分级解耦。**

### 6.1 分支做了什么

`clean-kernel-contract-ops` 在每个含多算子的 task 目录里新增 `other_op.py`，把 `kernel.py` 里**非 graded 的兄弟算子**外迁出去，使 `kernel.py` 只剩被打分的那一个 `@triton.jit`。分支共有 **70 个 `other_op.py`**。

### 6.2 与现状的关系（关键）

线上语料**已经做了单算子抽取**（per MEMORY：`extraction_verified 100/103`、`92/103 single-operator`）——graded 候选就是 `kernel.py` 里被 oracle 选中的那一个算子，多算子文件靠 candidate enumeration / operator-slicer 在打分期定位。也就是说：**"只打一个算子"这件事现状已经达成**，`other_op.py` 解决的是**源文件物理整洁度**，不是打分正确性。

### 6.3 成本 / 收益

**收益（弱-中）：**
- `kernel.py` 物理上只剩 graded 算子，静态审计（`ast_denylist`/`import_allowlist`/`constexpr_match`）面更小、人读更清。
- severance 审计更干净（减少兄弟算子误 import 上游的泄漏面，呼应 MEMORY 里 severed-vllm-hygiene / struct-input 泄漏教训）。
- 与分支对齐，未来 merge 冲突小。

**成本（中-高）：**
- **70 个目录的 `kernel.py` 字节改变** → 触发 `phase_a_prime.standalone_sha256` / `kernel_sha256` / `provenance.golden.golden_sha256` **全部失效**，需重跑 S2 / 重新封 golden（GPU 真机）。这与"frozen disk bytes"原则冲突（MEMORY：device_assert strip / dtype-infidelity 都强调 disk bytes frozen）。
- `runner_checks.expected_kernel_count_range`（如 `[1,4]`）、candidate enumeration 逻辑可能依赖"kernel.py 内有多少 jit"——外迁后 `n_jit` 变化（数据里 `n_jit` 1~8），需同步核对 `静态检查 + resolver`。
- 必须重跑全量 C1/C2/S2/release-gate GPU 回归（per MEMORY "Full regression = whole GPU pipeline"）。

### 6.4 建议

**[likely] 不与本次重分级捆绑。** 重分级（改 `level` 字段）**不动 `kernel.py` 字节**，可零 GPU 成本落地（见 §7）；而 `other_op.py` 拆分**强制全量 GPU 重封 golden**。把两者绑一起会让一个低风险元数据改动背上高风险 re-seal。

**推荐：** 重分级先行（Phase 1-2）；`other_op.py` 拆分作为**独立 RFC**，仅在下一次"反正要全量 re-seal"的窗口（如批量 dtype 修正、golden 重采）顺带做，且必须：①逐题 severance 复审；②`n_jit` / `expected_kernel_count_range` 同步；③全量 GPU 回归。**不为了"和分支长得一样"而单独付 re-seal 代价。**

---

## 7. 影响分析

### 7.1 改级别的三种粒度（核心权衡）

| 粒度 | 动作 | `task_id` | 目录 | `idx` | blast radius |
|---|---|---|---|---|---|
| **(A) 字段-only** | 只改 `meta.yaml` 的 `level:`（3 个 current-unique 的 `dtype_class` 已存在，不回填） | **不变** | **不变** | 不变 | **最小** |
| **(B) 字段+目录移动** | (A) + `git mv tasks/levelX/<id> tasks/levelY/<id>` | 不变 | 变 | 不变 | 中 |
| **(C) 全量 renumber** | 重命名 `task_id`、移目录、重排 `idx`、改 `source_md_id`（= 分支做法） | **全变** | 全变 | 全变 | **最大** |

**实测耦合（已在本仓核对）：**

- **loader 用字段，不用前缀**：`dataset.py:129 raw_level = meta["level"]`，`level_int` 由字段推导。→ **(A) 字段-only 即被打分/聚合正确认领**，哪怕目录还叫 `tasks/level1/` 而 `level: L2`。
- **golden_store 按 task_id 解析**：`golden_store.py:86-88` 扫 `level1/2/3` 找 `<task_id>`；gpfs `sealed_golden/<task_id>/sealed_*.safetensors` 按 **task_id** 键。→ (A)/(B) 都**不破** golden 键（task_id 稳定）；**(C) renumber 会让每个 golden 键失效，需全量重封**。
- **`validate_corpus_consistency.py:124`** 用 `glob("tasks/level*/*/meta.yaml")`——level-dir 无关，(A)/(B)/(C) 都能发现题。
- **⚠ (A) 字段/目录分歧的唯一可见副作用**：`tab init-solution`（`cli.py:257`）从 `task.level` **字段**拼 contract 路径 `tasks/level{level}/{id}/contract.md`。(A) 下若 `L2_16` 字段=L1 但仍住 `tasks/level2/`，这条 CLI 给用户的路径提示会指向不存在的 `tasks/level1/L2_16_.../`。**非 daemon-breaking**（只是 scaffold 命令的 user-facing 提示串，不在打分/封存路径上），但**不是纯 cosmetic**——落地 (A) 前应 grep 所有从 level 字段拼路径的构造（另见 `generate_task_scaffolds.py:482`），决定修 CLI 注释还是接受该提示偏差。`per_task_diagnostic.py:73` 按 task_id **前缀**拼路径（与目录一致），安全。

### 7.2 对"刚完成的单元工作"的影响

- **18 个 `phase_b_capture.upstream_callable` 回填**：存在 `meta.yaml` 内，随文件走。(A)/(B) 零影响；(C) 内容不变但需连同新 task_id 重新校验。
- **wrapper / resolver 测试的硬编码路径**：`test_wrapper_contract.py:259` 钉 `tasks/level3/L3_13_…/oracle_pytorch.py`；`test_kernel_resolver.py` 钉 `L3_23`、`L3_07_correct_attn_cp_out`、`L1_20`、`L1_34`、`L1_17_gelu_and_mul`；`test_r10_multi_output_order.py:150` 钉 `tasks/level1/L1_36_…`。→ **(A) 全安全**（路径不变）。**(B) 会破被移动的题**：`L3_07→L2`（破 `:217`）、`L1_17→L2`（破 `:205`）必须同步改测试。(C) 破全部。
- **`test_corpus_count`（`test_phase_c_generators_wa5.py:432` `== 101`）**：计数不变量，(A)/(B)/(C) 都保持 101，**安全**。`test_summary_metrics.py:415` 也用 `n=101`，安全。
- **`test_corpus_contract_mirror_matches_meta`（同文件）：与本方案无关，无需 regen contract.md 镜像。** codex 复核：该 mirror 只渲染 `meta.param_constraints`（dims/dtype/dist/scalars/invariants，`phase_c_docs.py:640`；test 比对 `_render_constraint_mirror(meta["param_constraints"])`，`:452`），**`level`/`dtype_class` 都不在 mirror 内**——纯改 level 后此测试**保持绿**。（初版误把它列为必同步项，已删除该步骤。）
- **static_checker poc list**：相关代码文件（`static_checker.py`/`eval.py`/`cli.py`/`manifest.py` 等）按 task_id/检查项组织，非 level 耦合。(A)/(B) 预期零影响——**建议落地后 grep 复核一遍是否有按 level 前缀过滤的 allowlist**。

### 7.3 daemon 流水（C1/C2/S2/release-gate/gpu-rank）+ golden/provenance

- **全部按 `task_id` 键**：`results/*.json`、`resolve_all.json`、`sealed_golden/<task_id>`、`provenance.golden`（在 meta 内）。
- **(A) 字段-only**：C1/C2 oracle-equiv、S2/phase_a_prime、release-gate、gpu-rank **全部零破**（task_id 稳定，golden/provenance 键不动）。**这是本方案首选粒度的最大理由。**
- **(B) 目录移动**：golden_store 仍按 task_id 扫三 level 目录找得到，gpfs 键不变 → daemon 仍零破；**只破硬编码测试路径**（7.2）。
- **(C) renumber**：sealed_golden 键、results、resolve_all、provenance source_md_id 全失效 → **必须全量 re-seal + 重跑全 GPU 流水**。
- **ranking 是否按 level 分桶？** `scoring.py` 的 `s_t`/Roofline 是 per-task（按 task_id），未发现按 level 分桶的聚合；level 仅用于**报告/分布**。→ 改 level 字段不改任何排名数值，只改报告里的分组。

### 7.4 `docs/kernel-select.md` 必须重生成

- **§1 计数**：现写"100 / L1 40·L2 38·L3 22"（已 stale）→ 改为 **101 / L1 36·L2 35·L3 30**（采纳本方案）。
- **§1 VV/CV 计数**：现写"VV 65 / CV 35"亦 stale——实际现有 metadata tally 已是 **VV 66 / CV 35**（codex 复核）。3 个 current-unique 的 `dtype_class` **已存在**（L2_39=CV、L2_40=VV、L2_41=VV），**无需回填**；重分级**不改 VV/CV 计数**，只把 docs 的 §1 数字对齐到 66/35。
- **§3 rubric**：用 §2 的锋利版替换旧口语版。
- **§4 类别计数**：不变（类别字段不动）。
- **§7 全表**：每题 `level` 列更新；若走 (C) 则 task_id 列全改。
- **⚠ regen 工具警告**：`scripts/phase_c_docs.py` 的 **normal 模式会重写 `shapes.yaml`**（`:971`），**不要用它**做这个改动；`--refresh-mirror-only`（`:1021`）只刷 param_constraints 镜像、**不碰 shapes/meta**——但本方案根本不需要刷 mirror（§7.2）。`docs/kernel-select.md` 直接手工编辑，或用只读 meta、不回写 task 目录的专用脚本。

---

## 8. 执行策略与分阶段

### 8.1 粒度决策：推荐 (A) 字段-only，(C) 不采纳

**推荐 (A) 字段-only**：只改 `meta.yaml:level`（3 个 current-unique 的 `dtype_class`/`npu_difficulty` 已存在，**不回填**）+ regen `docs/kernel-select.md`。

理由：
- **零 GPU 成本**：不动 `kernel.py` 字节、不动 golden/provenance 键、不破 daemon 流水（§7.3）。
- **loader 已按字段认领级别**（§7.1），打分/报告立即正确。
- **可逆**：纯文本字段，rollback = `git revert`。

**不推荐 (C) 全量 renumber（= 分支做法）**：分支让**每个 task_id 都变**（如 `L2_16→L1_41`、`L3_07→L2_77`、全局重排 idx），代价是 sealed_golden / results / resolve_all / provenance / 全部硬编码测试 + 全量 GPU re-seal。**收益只是"前缀和级别一致"的美观**，不值这个 blast radius。

**(A) 的已知代价（需用户知情接受）**：`task_id` 前缀与 `level` 字段会**故意不一致**（如 `L2_16_fused_rmsnorm` 的 `level: L1`，仍住 `tasks/level2/`）。`idx` 也会跨级重复。这基本是**命名层**遗留，loader / daemon / 打分 / golden 封存全部无感——**唯一例外**是 `tab init-solution` 的 contract 路径提示（§7.1 caveat，非 daemon-breaking），落地前决定修 CLI 注释或接受。可在 §7.4 文档里加一句"task_id 前缀是历史编号，权威级别看 `level` 字段"。若用户无法接受这个不一致，再升级到 (B)（移目录、同步 2 处测试路径，仍不 re-seal），**仅在确有外部消费者按目录前缀判级时**才考虑。

### 8.2 与 pending 23 个单元修复的顺序

**先做完 pending 23 个 unit 修复，再做重分级。** 理由：重分级会 regen `docs/kernel-select.md` 并触碰 contract.md 镜像 / `test_corpus_contract_mirror`，需要一个**绿色基线**来区分"是不是我这次改坏的"。在脏基线上叠字段改动，回归失败归因会被污染。顺序：**23 unit fix → 全量回归绿 → 重分级（字段+文档）→ 再回归。**

### 8.3 rollback-safe 落地序列（采纳 (A) 时）

1. **新分支** `chore/level-reclassify-fields`（不在 main 上动）。
2. **批量改 `meta.yaml:level`**：**38 个** differ-from-current 题（§4 中 rec≠cur 的行，含 L1_20→L2 一致性升级）；脚本化、一次性全改（CLAUDE.md：batch all fixes）。**只动 `level:` 一行**——3 个 current-unique 的 `dtype_class`/`npu_difficulty` 已存在，**不回填**（codex 复核：L2_39=CV/medium、L2_40=VV/medium、L2_41=VV/medium 均已在 meta）。
3. **regen `docs/kernel-select.md`** §1（101 / 36·35·30，VV/CV 对齐 66/35）/§3（锋利 rubric）/§7（每题 level 列）——手工编辑或只读 meta 脚本，**不碰 shapes/contract 镜像**（§7.2/§7.4）。
4. **本地单元回归**：`test_corpus_count`(==101)、resolver/wrapper 路径测试、`test_validate_tasks_*`（level enum）——(A) 下应全绿且 0 skip；`test_corpus_contract_mirror` 不受影响（§7.2）。
5. **codex review**（CLAUDE.md 强制：非琐碎、跨多题协议改动）→ 0 FAIL。
6. **全量 GPU 回归**（C1/C2 oracle-equiv + phase_a_prime/S2 + cost_xcheck + golden + release-gate + gpu-rank，全 101 题，per MEMORY "Full regression = whole GPU pipeline"）。预期：因 task_id/golden 键不变，**应与改前 bit-identical**——这正是 (A) 的验证点（级别字段变了，数值/封存零变化）。
7. 用户确认后再 push（CLAUDE.md：push 前必须暂停问用户）。

### 8.4 落地前用户必须拍板的决策点

1. **粒度**：(A) 字段-only（推荐）/ (B) 字段+目录 / (C) 全量 renumber（分支式）。**默认建议 (A)。**
2. **采纳本方案 vs 照搬分支**：是否接受本方案对分支的 **7 个分歧**（L1_18/25/26/30 退回 L1；L2_15/31/34 退回 L2）+ **L1_20→L2 一致性升级**（消除 double-reduce 矛盾，§5.7）。每个都有 `[confirmed]` 源码依据。
3. **3 个 current-unique 重定级**：`L2_40→L1`（norm 单 reduce，`[confirmed]`）、`L2_41→L3`（SSM state，`[likely]` borderline——**这一个最需用户表态**：净判 L3，但 reviewer 可坚持 L2，理由是无 dot + fixture seq_len=1）。`L2_39` 维持 L2。
4. **`other_op.py` 拆分**：是否采纳（§6 建议**解耦、不本次做**）。
5. **顺序**：是否同意"先清 23 unit fix 拿绿基线，再重分级"。
6. **类型标签**：L2_19/20/21 保 VV（经次轴到 L3），是否同意不把它们标 CV。

**最终推荐摘要**：走 **(A) 字段-only**，采纳本方案 7 个分歧 + L1_20 一致性升级 + 3 个 current-unique 定级（L2_41 标 borderline 待用户确认），`other_op.py` 解耦，顺序上 unit-first。落地后分布 **L1 36 / L2 35 / L3 30**，三方一致 55 题，零 GPU re-seal、零 golden 键失效。

> 本方案已经 1 轮内部对抗审查 + 1 轮 codex 独立复核（gpt-5.5:xhigh）。codex 确认 core sound（field-only daemon/golden-safe + 7 分歧中 6 个 spot-check 支持），并指出 5 处必修缺陷，**已全部修正落入本文档**：①L1_20→L2（§5.7，counts 36/35/30）；②删除 contract.md mirror 同步步骤（§7.2，mirror 只含 param_constraints）；③删除 current-unique dtype 回填（§7.4/§8.3，三者已有字段，tally 已 66/35）；④补 `tab init-solution` 字段/目录分歧 caveat（§7.1）；⑤L2_41 保持显式用户决策（§5.2/§8.4-决策点3）。

---

相关文件（绝对路径）：
- 目标 rubric / 待 regen 文档：`/data1/code/agentic-rl/kernel/triton-ascend-bench/docs/kernel-select.md`（§1 已 stale 写 100/40·38·22）
- meta schema（`level` / `dtype_class` / `provenance.golden` / `phase_a_prime`）：`/data1/code/agentic-rl/kernel/triton-ascend-bench/tasks/level1/L1_13_layer_norm_fwd/meta.yaml`
- loader 按字段判级：`/data1/code/agentic-rl/kernel/triton-ascend-bench/triton_ascend_bench/dataset.py:129`
- golden 按 task_id 键：`/data1/code/agentic-rl/kernel/triton-ascend-bench/triton_ascend_bench/golden_store.py:86`
- 硬编码 level 路径（(B)/(C) 会破）：`/data1/code/agentic-rl/kernel/triton-ascend-bench/tests/unit_tests/test_kernel_resolver.py:205,217`、`test_wrapper_contract.py:259`、`test_r10_multi_output_order.py:150`
- corpus 计数不变量：`/data1/code/agentic-rl/kernel/triton-ascend-bench/tests/unit_tests/test_phase_c_generators_wa5.py:432`
- deprecated lightning-attn（L2_23/24 口径来源）：`/data1/code/agentic-rl/kernel/triton-ascend-bench/SCORE.md:19-52`、`tasks/_deprecated/`、`tasks/_corrected/`
- 分支单算子拆分参照：`origin/clean-kernel-contract-ops`（70 个 `other_op.py`，level1/2/3 = 35/35/30，task_id 全重排）
## 附录 A：全量逐题对比表（当前 / 分支 / 本方案）

> 标记：`—`=三方一致；`认可分支`=本方案=分支≠当前；`✗分支`=本方案与分支分歧；`current-unique`=分支无此题。

| # | task_id | 当前 | 分支 | 本方案 | 类型 | 标记 |
|---|---|---|---|---|---|---|
| 1 | L1_01_bias | L1 | L1 | L1 | VV | — |
| 2 | L1_02_compute_slot_mappings | L1 | L1 | L1 | VV | — |
| 3 | L1_03_convert_req_index_to_global_index | L1 | L1 | L1 | VV | — |
| 4 | L1_04_copy_page_indices | L1 | L1 | L1 | VV | — |
| 5 | L1_05_gather_block_tables | L1 | L1 | L1 | VV | — |
| 6 | L1_06_reshape_and_cache_kernel_flash | L1 | L1 | L1 | VV | — |
| 7 | L1_07_dequantize_k_cache_fast | L1 | L1 | L1 | VV | — |
| 8 | L1_08_dequantize_k_cache_paged | L1 | L1 | L1 | VV | — |
| 9 | L1_09_quantize_k_cache_fast | L1 | L1 | L1 | VV | — |
| 10 | L1_10_per_token_group_quant_int8 | L1 | L1 | L1 | VV | — |
| 11 | L1_11_awq_dequantize | L1 | L1 | L1 | VV | — |
| 12 | L1_12_rms_norm | L1 | L1 | L1 | VV | — |
| 13 | L1_13_layer_norm_fwd | L1 | L2 | L2 | VV | 认可分支（L1→L2） |
| 14 | L1_14_layer_norm_fwd_1pass | L1 | L2 | L2 | VV | 认可分支（L1→L2） |
| 15 | L1_15_act_and_mul | L1 | L2 | L2 | VV | 认可分支（L1→L2） |
| 16 | L1_16_silu_and_mul | L1 | L2 | L2 | VV | 认可分支（L1→L2） |
| 17 | L1_17_gelu_and_mul | L1 | L2 | L2 | VV | 认可分支（L1→L2） |
| 18 | L1_18_fused_softcap | L1 | L2 | L1 | VV | ✗分支（分支判 L2，本方案 L1） |
| 19 | L1_19_topk_log_softmax | L1 | L2 | L2 | VV | 认可分支（L1→L2） |
| 20 | L1_20_log_softmax | L1 | L1 | L2 | VV | ✗分支（分支判 L1，本方案 L2） |
| 21 | L1_21_gumbel_sample | L1 | L2 | L2 | VV | 认可分支（L1→L2） |
| 22 | L1_22_min_p | L1 | L2 | L2 | VV | 认可分支（L1→L2） |
| 23 | L1_23_penalties_and_temperature | L1 | L2 | L2 | VV | 认可分支（L1→L2） |
| 24 | L1_24_rejection_sample | L1 | L2 | L2 | VV | 认可分支（L1→L2） |
| 25 | L1_25_merge_attn_states | L1 | L2 | L1 | VV | ✗分支（分支判 L2，本方案 L1） |
| 26 | L1_26_merge_state | L1 | L2 | L1 | VV | ✗分支（分支判 L2，本方案 L1） |
| 27 | L1_27_apply_grammar_bitmask | L1 | L1 | L1 | VV | — |
| 28 | L1_28_pack_bitmatrix | L1 | L2 | L2 | VV | 认可分支（L1→L2） |
| 29 | L1_29_prepare_eagle_inputs | L1 | L1 | L1 | VV | — |
| 30 | L1_30_act_quant | L1 | L2 | L1 | VV | ✗分支（分支判 L2，本方案 L1） |
| 31 | L1_31_triton_mrope_forward | L1 | L1 | L1 | VV | — |
| 32 | L1_32_create_flashinfer_kv_indices_triton | L1 | L1 | L1 | VV | — |
| 33 | L1_33_create_flashmla_kv_indices_triton | L1 | L1 | L1 | VV | — |
| 34 | L1_34_transform_index_page_table_decode | L1 | L1 | L1 | VV | — |
| 35 | L1_35_compute_src2dst_triton | L1 | L1 | L1 | VV | — |
| 36 | L1_36_compute_seg_indptr_triton | L1 | L1 | L1 | VV | — |
| 37 | L1_37_compute_problem_sizes_w4a8 | L1 | L1 | L1 | VV | — |
| 38 | L1_38_silu_mul_fp8_quant_deep_gemm | L1 | L2 | L2 | VV | 认可分支（L1→L2） |
| 39 | L1_39_state_passing_fwd | L1 | L1 | L1 | VV | — |
| 40 | L1_40_chunk_cumsum_fwd | L1 | L1 | L1 | VV | — |
| 41 | L2_01_bmm_chunk_fwd | L2 | L2 | L2 | CV | — |
| 42 | L2_02_chunk_scan_fwd | L2 | L3 | L3 | CV | 认可分支（L2→L3） |
| 43 | L2_03_chunk_state_fwd | L2 | L3 | L3 | CV | 认可分支（L2→L3） |
| 44 | L2_04_chunked_lora_expand | L2 | L3 | L3 | CV | 认可分支（L2→L3） |
| 45 | L2_05_chunked_lora_shrink | L2 | L2 | L2 | CV | — |
| 46 | L2_06_qkv_lora_b | L2 | L3 | L3 | CV | 认可分支（L2→L3） |
| 47 | L2_07_gate_up_lora_b | L2 | L3 | L3 | CV | 认可分支（L2→L3） |
| 48 | L2_08_sgemm_lora_a | L2 | L2 | L2 | CV | — |
| 49 | L2_09_sgemm_lora_b | L2 | L3 | L3 | CV | 认可分支（L2→L3） |
| 50 | L2_10_w8a8_block_int8_matmul | L2 | L2 | L2 | CV | — |
| 51 | L2_11_w8a8_triton_block_scaled_mm | L2 | L2 | L2 | CV | — |
| 52 | L2_12_awq_gemm | L2 | L2 | L2 | CV | — |
| 53 | L2_13_bmm | L2 | L2 | L2 | CV | — |
| 54 | L2_14_matmul_kernel_persistent | L2 | L2 | L2 | CV | — |
| 55 | L2_15_fused_dual_residual_rmsnorm | L2 | L1 | L2 | VV | ✗分支（分支判 L1，本方案 L2） |
| 56 | L2_16_fused_rmsnorm | L2 | L1 | L1 | VV | 认可分支（L2→L1） |
| 57 | L2_17_layer_norm_gated_fwd | L2 | L2 | L2 | VV | — |
| 58 | L2_18_fused_gdn_gating | L2 | L1 | L1 | VV | 认可分支（L2→L1） |
| 59 | L2_19_fused_sigmoid_gating_delta_rule_update | L2 | L3 | L3 | VV | 认可分支（L2→L3） |
| 60 | L2_20_causal_conv1d_update | L2 | L3 | L3 | VV | 认可分支（L2→L3） |
| 61 | L2_21_causal_conv1d_fwd | L2 | L3 | L3 | VV | 认可分支（L2→L3） |
| 62 | L2_22_linear_attn_decode | L2 | L2 | L2 | VV | — |
| 63 | L2_25_fwd_kernel_ep_gather | L2 | L2 | L2 | VV | — |
| 64 | L2_26_fwd_kernel_ep_scatter_1 | L2 | L2 | L2 | VV | — |
| 65 | L2_27_deepep_permute_triton | L2 | L1 | L1 | VV | 认可分支（L2→L1） |
| 66 | L2_28_deepep_post_reorder_triton | L2 | L1 | L1 | VV | 认可分支（L2→L1） |
| 67 | L2_29_fused_moe_router_cudacore | L2 | L2 | L2 | VV | — |
| 68 | L2_30_fused_moe_router_tensorcore | L2 | L3 | L3 | CV | 认可分支（L2→L3） |
| 69 | L2_31_moe_align_block_size_stage2 | L2 | L1 | L2 | VV | ✗分支（分支判 L1，本方案 L2） |
| 70 | L2_32_experts_combine | L2 | L1 | L1 | VV | 认可分支（L2→L1） |
| 71 | L2_33_fwd_grouped_kernel_stage1_rope | L2 | L3 | L3 | CV | 认可分支（L2→L3） |
| 72 | L2_34_silu_mul_static_tensorwise_quant…cutlass_moe | L2 | L1 | L2 | VV | ✗分支（分支判 L1，本方案 L2） |
| 73 | L2_35_pre_reorder_triton_kernel_for_cutlass_moe | L2 | L1 | L1 | VV | 认可分支（L2→L1） |
| 74 | L2_36_post_reorder_triton_kernel_for_cutlass_moe | L2 | L1 | L1 | VV | 认可分支（L2→L1） |
| 75 | L2_37_silu_and_mul_post_quant | L2 | L2 | L2 | VV | — |
| 76 | L2_38_concat_and_cast_mha_k | L2 | L1 | L1 | VV | 认可分支（L2→L1） |
| 77 | L2_39_scaled_mm | L2 | NEW | L2 | CV | current-unique |
| 78 | L2_40_gemma_qkv_rmsnorm | L2 | NEW | L1 | VV | current-unique（cur L2→L1） |
| 79 | L2_41_selective_state_update | L2 | NEW | L3 | VV | current-unique（cur L2→L3） |
| 80 | L3_01_kernel_unified_attention_3d | L3 | L3 | L3 | CV | — |
| 81 | L3_02_kernel_paged_attention_2d | L3 | L3 | L3 | CV | — |
| 82 | L3_03_chunk_state_varlen | L3 | L3 | L3 | CV | — |
| 83 | L3_04_fwd | L3 | L3 | L3 | CV | — |
| 84 | L3_05_fwd_kernel_alibi | L3 | L3 | L3 | CV | — |
| 85 | L3_06_fwd_grouped_kernel_stage1 | L3 | L3 | L3 | CV | — |
| 86 | L3_07_correct_attn_cp_out | L3 | L2 | L2 | VV | 认可分支（L3→L2） |
| 87 | L3_09_fused_moe | L3 | L3 | L3 | CV | — |
| 88 | L3_10_fused_moe_kernel_gptq_awq | L3 | L3 | L3 | CV | — |
| 89 | L3_11_expert_triton | L3 | L3 | L3 | CV | — |
| 90 | L3_12_moe_mmk | L3 | L2 | L2 | CV | 认可分支（L3→L2） |
| 91 | L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | L3 | L3 | L3 | CV | — |
| 92 | L3_14_fused_recurrent_gated_delta_rule_fwd | L3 | L3 | L3 | VV | — |
| 93 | L3_15_fused_recurrent_gated_delta_rule_update_fwd | L3 | L3 | L3 | VV | — |
| 94 | L3_16_chunk_fwd_kernel_o | L3 | L3 | L3 | CV | — |
| 95 | L3_17_chunk_gla_fwd_kernel_o | L3 | L3 | L3 | CV | — |
| 96 | L3_18_chunk_scaled_dot_kkt_fwd | L3 | L3 | L3 | CV | — |
| 97 | L3_19_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter | L3 | L3 | L3 | CV | — |
| 98 | L3_20_recompute_w_u_fwd | L3 | L3 | L3 | CV | — |
| 99 | L3_21_solve_tril_16x16 | L3 | L2 | L2 | VV | 认可分支（L3→L2） |
| 100 | L3_22_chunk_local_cumsum_vector | L3 | L2 | L2 | CV | 认可分支（L3→L2） |
| 101 | L3_23_merge_16x16_to_32x32_inverse | L3 | L3 | L3 | CV | — |
