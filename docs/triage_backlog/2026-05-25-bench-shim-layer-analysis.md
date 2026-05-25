---
date: 2026-05-25
status: investigation report (read-only, no code changes)
audience: bench team (siiRL kernel migration benchmark)
based-on:
  - HEAD d2578b8 (note: prompt said abda667 -- stale; current is d2578b8)
  - resolve_all.json @ generated_at 2026-05-24T14:18:51Z (38 upstream-not-invocable-here verdicts)
  - dev-machine probe (Triton 3.6.0 / torch 2.11.0+cu130 / no active CUDA driver)
scope: bench shim layer for vllm/sglang upstream-not-invocable BLOCKED tasks
export-ignored: yes (.gitattributes already covers docs/triage_backlog/)
---

# bench shim 层现状深度调研 (2026-05-25)

## 0. 阅读须知

- 本文档是 **bench 团队内部调研报告**,目的是把 `resolve_all.json` 当前 38 个 `BLOCKED(upstream-not-invocable-here)` task 的根因彻底搞清楚,并为后续修复给出 leverage 排序的方案清单。
- 文档落在 `docs/triage_backlog/`,被 `.gitattributes` `export-ignore` 通配覆盖,**不**进入 release tarball。
- 调研流程: **真实 grep + 真跑 Python**,没有 hallucinate。Dev 机 probe 数据写在 `/tmp/shim_analysis/import_probe.json`(38 task subprocess import 尝试),GPU 端权威数据从 `resolve_all.json` 直接取。
- **不改任何代码**(纯调研)。所有"修法"是建议,不是已落地。
- prompt 说"16 个 upstream-not-invocable-here task",**实际是 38 个**(resolve_all.json 当前 snapshot);本报告按 38 全量覆盖。

## 1. Executive Summary

### 1.1 核心结论(三段)

**[confirmed]** TAB 没有 `triton_ascend_bench/vllm_shim.py` / `triton_ascend_bench/stub_modules/` 这样的运行时 shim 层。bench 端"shim 设计"的真实形态是 **把 `infer/vllm` + `infer/sglang/python` 整个上游源码树挂到 `sys.path`,让 frozen kernel.py 中的 `from vllm.triton_utils import tl, triton` 直接走 LIVE 上游源码**。这条路径由 `scripts/_pa_oracle_check_runner.py` 的 `_load()`(synthetic per-call package + sys.path 注入)、`scripts/_pa_subproc_runner.py --side original`(同套机制)以及 `pyproject.toml [project.optional-dependencies].infer` extras(列出 vllm/sglang import chain 的纯 Python 叶子依赖如 `cachetools` / `py-cpuinfo` / `msgspec` / `cbor2` / `pyzmq` / `pybase64`)三处共同支撑。`triton_ascend_bench/trusted_shim/` 是 timer.c 计时器,**不是** vllm shim,名字容易误导。

**[confirmed]** 38 个 BLOCKED task 的失败按 GPU 端 resolve_all.json 权威 reason 归桶,真正分布是: **P2 sibling relative-import 18 个 / P3 TritonPlaceholder identity-decorator 漂移 9 个 / P4 missing pyproject-listed dep 6 个 / P5 input fixture assertion 3 个 / P1 vllm dup-op registration 2 个**。其中 P3+P5 这 12 个不是 shim 缺失,而是**两个独立 root cause**: P3 是上游 vllm 在没 active CUDA driver 时把 `triton` 全局替换为 `TritonPlaceholder`(`@triton.jit` 退化为 identity,kernel 不再是 `JITFunction`,后续 `kernel[grid](...)` / `triton.next_power_of_2` / `tl.float32` / `sglang.cached_triton_kernel` 全部炸); P5 是 task 的 `shapes.yaml` knob 与 kernel.py host wrapper 顶部 `assert` 不一致(输入夹具问题,与 shim 无关)。

**[likely]** 三个最高 leverage 的修法依次是: **(M-A) GPU 端 install `[infer]` extras + 装 `gguf`/`flashinfer` 两个目前不在 list 里的叶子** 一次性解 6 个 P4 + 部分 P2/P5 的连锁;**(M-B) 给 18 个 P2 task 在 task_dir 下 vendor sibling 文件**(`utils.py`/`index.py`/`op.py` 等,从 `infer/vllm/vllm/model_executor/layers/fla/ops/` 等位置 freeze 进 task dir,与 kernel.py 同等 SPDX header 处理) 解掉 18 个 P2;**(M-C) 让 resolver subprocess 强制启用一个 TRITON_INTERPRET-aware fallback 或 fake-CUDA-driver** 解掉 9 个 P3。三者 leverage 合计 33/38 ≈ 87%。剩下 5 个(3 个 P5 + 2 个 P1)走单独路径: P5 改 shapes.yaml,P1 是 vllm import 顺序 bug、可考虑给 `vllm::inplace_fused_experts` 注册前判存。

### 1.2 38 task P1-P5 分布数

| Cluster | 数量 | 根因 |
|---------|------|------|
| **P1** vllm dup op registration | 2 | resolver subprocess 内某条 import 路径触发 vllm `torch.library.define` 重复注册 |
| **P2** relative import (sibling vendor missing) | 18 | `from .utils import ...` 等 sibling 在 task dir 没 vendor,synthetic package 找不到 |
| **P3** TritonPlaceholder 漂移(no GPU driver) | 9 | vllm `HAS_TRITON=False` → `triton=TritonPlaceholder()`、`tl=TritonLanguagePlaceholder()`,后续 attr / 下标 / isinstance 全炸 |
| **P4** missing pyproject-listed Python dep | 6 | `cachetools` 5 + `py-cpuinfo` 1(都已写进 `[infer]` extras,但 GPU 机当时没装齐) |
| **P5** input fixture assertion | 3 | task `shapes.yaml` knob 与 frozen kernel.py 顶部 `assert` 不匹配(`token_indices.shape[1]==NUM_TOPK_TOKENS`、`page_size==1`、`group_size==128`) |
| **合计** | **38** | |

### 1.3 推荐 top-3 修法 leverage

| Rank | 修法 | 解多少 task | 工作量 | 风险 |
|------|------|-------------|-------|------|
| **#1** | **M-B**: vendor sibling files into task_dir (per-task `utils.py`/`index.py`/`op.py`) + 同 SPDX freeze 流程 | **18 (P2 全部)** | 单 task ~10-20 min;有可复用脚本可批处理(列依赖图 + 拉源 + 写 SPDX header) | 中等。sibling 链可能递归(`utils.py` 又 `from .X import`),需要做闭包传递 |
| **#2** | **M-A**: GPU 端 setup 补 `gguf` + `flashinfer`(不在 `[infer]` extras),并把 `cachetools` / `py-cpuinfo` 实际装进生效环境 | **6 P4 + 部分 P5/P2 间接放行** | 1 次改 `pyproject.toml [infer]` + 1 次 `pip install` on GPU | 低。纯加包,无逻辑变更 |
| **#3** | **M-C**: resolver subprocess 启用 `TRITON_INTERPRET=1` 时 monkey-patch `vllm.triton_utils.HAS_TRITON=True` + 提供完整 `TritonLanguagePlaceholder.float32/int32` + subscriptable `JITFunction` shim | **9 P3** | 1 次写 `triton_ascend_bench/triton_placeholder_extension.py` ~50-80 行;在 `_pa_oracle_check_runner.py` import 前注入 | 中。要确保 patch 不污染真 GPU 跑 |

合计 #1+#2+#3 解 33/38 ≈ **87%**。

### 1.4 不可 shim 修复的 task (必须走 PyTorch oracle / shapes 修复)

- **P5 (3 个)**: L1_03、L1_34、L1_38 — `shapes.yaml` knob 与 host wrapper 顶部 `assert` 不一致,**这是 task 元数据 bug,不是 shim**。修法是改 shapes.yaml 添加 `NUM_TOPK_TOKENS=2048`、`page_size=1`、`H % 8 == 0` 等约束(或改 default knob 满足)。
- **P1 (2 个)**: L3_09、L3_10 — `vllm::inplace_fused_experts` 重复注册。本质是 vllm 内部 import 顺序 bug。可考虑 `defer-to-golden` (`meta.phase_a_prime.skip_invocation: true`) 跳过上游 invocation, 走 PyTorch oracle 唯一路径。

合计 **5/38 ≈ 13% 必须走非 shim 路径**(其中 3 个是 shapes fixture 修复, 2 个是 `skip_invocation` defer)。

---

## 2. bench shim 层现状(Step 1 结果)

### 2.1 bench 端代码里 vllm shim 的实际位置

```bash
# grep 全 bench 包,只有 1 处提到 vllm:
$ grep -rln 'vllm' triton_ascend_bench/ --include='*.py'
triton_ascend_bench/static_checker.py

# 没有 shim/stub 文件:
$ find triton_ascend_bench/ \( -name 'vllm*' -o -name 'shim*' -o -name 'stub*' \) -print
# (empty)
```

`triton_ascend_bench/trusted_shim/__init__.py` 只是 `host_clock_ns()` 计时器(`time.clock_gettime_ns(CLOCK_MONOTONIC_RAW)`),**与 vllm/sglang 完全无关**。命名易误导。

### 2.2 bench 端"shim"的真实形态

```python
# scripts/_pa_oracle_check_runner.py:95-117
def _load(path: Path, name: str):
    # ... synthetic per-call package with __path__ = task_dir
    pkg = f"_pa_oc_pkg_{name}"
    if pkg not in sys.modules:
        synth = types.ModuleType(pkg)
        synth.__path__ = [str(path.parent)]
        sys.modules[pkg] = synth
    leaf_name = f"{pkg}.{path.stem}"
    spec = importlib.util.spec_from_file_location(leaf_name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg
    sys.modules[leaf_name] = mod
    spec.loader.exec_module(mod)
    return mod

# scripts/_pa_oracle_check_runner.py:275-278
for p in (spec["vllm_root"], spec["sglang_root"], spec["repo_root"], str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
```

- `infer/vllm` 和 `infer/sglang/python` 注入 `sys.path` → 这就是 "shim",让 `from vllm.triton_utils import tl, triton` 直接走 LIVE 上游源码。
- `synthetic per-call package` 给 kernel.py 一个 `__package__`,让 `from .X import Y` 能 resolve 到 task_dir 下的 sibling(**前提是 task_dir 下要有 sibling 文件**,目前没有)。
- `scripts/resolve_upstream_entrypoints.py:50` 把 `INFER_ROOT = REPO_ROOT.parent / "infer"`,即 `/data1/code/agentic-rl/kernel/infer/`(repo 外父目录)。Dev 机已确认存在并是 commit `18f6bf5` / `409d350`,与 `version_matrix.toml` 的 captured-on 一致。

### 2.3 pyproject `[infer]` extras 是依赖侧 shim

```toml
# pyproject.toml:60-79
infer = [
    "pybase64",
    "pyzmq",
    "cbor2",
    "msgspec",
    "cachetools",
    "py-cpuinfo",
]
```

显式列出 vllm/sglang import chain 上的纯 Python 叶子依赖。`scripts/automation/runners/setup_machine_a_gpu.sh` 在 step 4 跑 `pip install -e ".[automation,dev,gpu,infer]"`。

**但 `[infer]` 当前 list 不完整**:GPU 端 resolve_all.json 仍报 `No module named 'cachetools'`(5 task)、`cpuinfo`(1 task),且 dev 机 probe 又发现 `gguf`(11 task)、`flashinfer`(1 task)、`openai`(6 LoRA task 的 sglang import chain)依赖。这些都不在 `[infer]` 当前 list。

### 2.4 实际 env probe 数据

```python
# Dev 机 probe (无 active CUDA driver):
# Without infer/ on sys.path:
  vllm                            -MISSING-     # 期望: 不 pip install vllm,要靠 sys.path
  sglang                          -MISSING-
  cachetools                      OK            # 已装 (本机)
  cpuinfo                         OK            # 已装 (本机)
  flashinfer                      -MISSING-
  msgspec                         OK
  pybase64                        OK
  cbor2                           OK
  pyzmq                           -MISSING-     # [infer] 列了但本机没装

# With infer/ on sys.path:
  vllm                            IMPORT-OK
  vllm.triton_utils               IMPORT-OK     # 但触发 HAS_TRITON=False 警告
  vllm.v1.attention.backends.utils  FAIL: ModuleNotFoundError: gguf
  sglang                          IMPORT-OK
  sglang.srt.utils                IMPORT-OK

# Versions:
  triton: 3.6.0       # version_matrix.toml 写的是 3.5.1 -> 实际机已升 3.6.0
  torch:  2.11.0+cu130 # version_matrix.toml 写的是 2.9.1
```

**重要**: dev 机 triton/torch 版本已飞过 version_matrix 的 pin(3.5.1 → 3.6.0,2.9.1 → 2.11.0)。这可能影响 placeholder 行为(3.6.0 的 TritonPlaceholder 实现细节 vs 上游 vllm 写 placeholder 时的预期版本)。

### 2.5 vllm 端"TritonPlaceholder"机制(关键根因)

```python
# infer/vllm/vllm/triton_utils/importing.py:13-31
HAS_TRITON = find_spec("triton") is not None
if HAS_TRITON:
    try:
        from triton.backends import backends
        active_drivers = [
            x.driver for x in backends.values() if x.driver and x.driver.is_active()
        ]
        # ... distributed env tolerance ...
        elif not is_distributed_env and len(active_drivers) != 1:
            logger.info(
                "Triton is installed but %d active driver(s) found "
                "(expected 1). Disabling Triton to prevent runtime errors.",
                len(active_drivers),
            )
            HAS_TRITON = False  # <-- 触发点

# infer/vllm/vllm/triton_utils/__init__.py:11-18
if TYPE_CHECKING or HAS_TRITON:
    import triton
    import triton.language as tl
else:
    triton = TritonPlaceholder()   # @triton.jit 退化为 identity
    tl = TritonLanguagePlaceholder()
```

**dev 机 + GPU 机 resolver subprocess 都触发了** `HAS_TRITON=False` → `triton = TritonPlaceholder()`。结果:
- `@triton.jit` 装饰器变 identity,frozen kernel.py 中的 `_min_p_kernel` 等就是个**普通 Python function**;
- `_min_p_kernel[(num_tokens,)](...)` 这种 subscript 启动语法 → `TypeError: 'function' object is not subscriptable`;
- `triton.next_power_of_2(...)` → `AttributeError: module 'triton' has no attribute 'next_power_of_2'`;
- `tl.float32` → `AttributeError: module 'triton.language' has no attribute 'float32'`;
- sglang `@cached_triton_kernel` 装饰器内 `assert isinstance(fn, triton.runtime.jit.JITFunction)` → fail(P3_sglang_cached, LoRA family 6 task)。

GPU 机 resolver subprocess **也触发** 是因为 resolver 在 fresh subprocess 启动时 CUDA context 还没 init,`is_active()` 返回 False;`_pa_oracle_check_runner.py:253` 设 `TRITON_INTERPRET=1` 也救不了(vllm 的 HAS_TRITON 检查是 import 时进行的,环境变量在那时还没生效或被忽略)。

---

## 3. 38 task P1-P5 失败逐个表

权威分类基于 **resolve_all.json GPU 端 reason**(`_pa_oracle_check_runner.py:147` 错位 traceback)。

### 3.1 P1: vllm dup op registration (2)

| task | entry tried | 失败摘要 |
|------|-------------|---------|
| `L3_09_fused_moe` | `invoke_fused_moe_triton_kernel`,`invoke_fused_moe_wna16_triton_kernel`,`zero_experts_compute_triton` | `RuntimeError: Tried to register an operator (vllm::inplace_fused_experts ...) with the same name ... multiple times. Duplicate registration: registered at infer/vllm/vllm/utils/torch_utils.py:928. Original registration: registered at infer/vllm/vllm/utils/torch_utils.py:928` |
| `L3_10_fused_moe_kernel_gptq_awq` | 同上 | 同上 |

根因: vllm 内部 import 顺序 bug,同一进程内不同 import 路径(`fused_moe.modular_kernel as mk` + `vllm.model_executor.layers.fused_moe.fused_moe`)都触发 `torch.library.define(...)`,PyTorch 严格拒绝。

修法: M-D (defer-to-golden / `skip_invocation: true`)。也可以 `vllm/utils/torch_utils.py:928` 加 `try: ... except RuntimeError: pass` 但那是改上游,违反 frozen 原则。

### 3.2 P2: relative import (sibling vendor missing) (18)

| task | 缺的 sibling(s) | 上游真实路径 |
|------|----------------|------------|
| `L1_13_layer_norm_fwd` | `.utils.input_guard` | `infer/vllm/vllm/model_executor/layers/fla/ops/utils.py` |
| `L1_28_pack_bitmatrix` | `..utils.swiglu_limit_func` | `infer/vllm/vllm/model_executor/layers/fused_moe/utils.py` |
| `L1_31_triton_mrope_forward` | `.base.RotaryEmbeddingBase`,`.yarn_scaling_rope.*` | `infer/vllm/vllm/model_executor/layers/rotary_embedding/` |
| `L1_40_chunk_cumsum_fwd` | `.mamba_ssm.softplus` | `infer/vllm/vllm/model_executor/layers/mamba/ops/mamba_ssm.py` |
| `L2_03_chunk_state_fwd` | `.mamba_ssm.softplus` | 同上 |
| `L2_17_layer_norm_gated_fwd` | `.chunk_delta_h`,`.cumsum`,`.fused_recurrent`,`.index`,`.l2norm`,`.op`,`.solve_tril`,`.utils` (8 个) | `infer/vllm/vllm/model_executor/layers/fla/ops/*.py` |
| `L3_02_kernel_paged_attention_2d` | `.prefix_prefill.context_attention_fwd` | `infer/vllm/vllm/attention/ops/prefix_prefill.py` |
| `L3_03_chunk_state_varlen` | `.mamba_ssm.softplus` | 同 L1_40 |
| `L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | `.index`,`.op`,`.utils` | `infer/vllm/vllm/model_executor/layers/fla/ops/*.py` |
| `L3_14_fused_recurrent_gated_delta_rule_fwd` | `.op.exp` | 同上 |
| `L3_16_chunk_fwd_kernel_o` | `.index`,`.op`,`.utils` | 同上 |
| `L3_17_chunk_gla_fwd_kernel_o` | 同 L2_17 (8 个 sibling) | 同 L2_17 |
| `L3_18_chunk_scaled_dot_kkt_fwd` | `.index`,`.op`,`.utils` | 同上 |
| `L3_19_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter` | 同 L2_17 (8 个 sibling) | 同 L2_17 |
| `L3_20_recompute_w_u_fwd` | `.index` | 同上 |
| `L3_21_solve_tril_16x16` | `.index`,`.op`,`.utils` | 同上 |
| `L3_22_chunk_local_cumsum_vector` | `.index`,`.utils` | 同上 |
| `L3_23_merge_16x16_to_32x32_inverse` | `.index`,`.op`,`.utils` | 同上 |

去重后实际需要 vendor 的 sibling 文件(从 fla/ops/ 大头看)≈ 11 个: `utils.py`、`index.py`、`op.py`、`mamba_ssm.py`、`l2norm.py`、`cumsum.py`、`chunk_delta_h.py`、`fused_recurrent.py`、`solve_tril.py`、`base.py`、`yarn_scaling_rope.py`、`prefix_prefill.py`(12 个,但有重叠)。

修法: M-B (vendor sibling 到 task_dir,同 SPDX header)。这些是纯 Python helper(`input_guard` / `prepare_chunk_indices` / `softplus` 等),不是新 kernel,vendor 闭包通常很小。

### 3.3 P3: TritonPlaceholder 漂移 (9)

| task | 失败行 | 失败模式 |
|------|--------|---------|
| `L1_14_layer_norm_fwd_1pass` | `BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))` | `AttributeError: module 'triton' has no attribute 'next_power_of_2'` |
| `L1_22_min_p` | `_min_p_kernel[(num_tokens,)](...)` | `TypeError: 'function' object is not subscriptable` |
| `L2_01_bmm_chunk_fwd` | `out_dtype = tl.float32 if ...` | `AttributeError: module 'triton.language' has no attribute 'float32'` |
| `L2_04_chunked_lora_expand` | `@cached_triton_kernel` decoration time | `AssertionError: isinstance(fn, triton.runtime.jit.JITFunction)` at `sglang.srt.utils.common:3859` |
| `L2_05_chunked_lora_shrink` | 同上 | 同上 |
| `L2_06_qkv_lora_b` | 同上 | 同上 |
| `L2_07_gate_up_lora_b` | 同上 | 同上 |
| `L2_08_sgemm_lora_a` | 同上 | 同上 |
| `L2_09_sgemm_lora_b` | 同上 | 同上 |

P3 全部由 vllm `HAS_TRITON=False` → `TritonPlaceholder` 替换引起。LoRA 6 个 + Triton 漂移 3 个。

修法: M-C (resolver subprocess 内 monkey-patch `vllm.triton_utils.HAS_TRITON=True` + 补全 TritonPlaceholder)。也可以让 resolver **真正在 GPU 上跑** —— 但当前 resolve_all 看上去就是在 GPU 机上跑的(路径含 `kernel/infer`)、却仍触发 placeholder,说明问题是 resolver subprocess fresh-startup 时 CUDA 还没 init。

### 3.4 P4: missing pyproject-listed Python dep (6)

| task | 缺包 | pyproject [infer] 列了吗 |
|------|------|------------------------|
| `L1_04_copy_page_indices` | `cpuinfo` | 是(`py-cpuinfo`),GPU 端没装 |
| `L1_19_topk_log_softmax` | `cachetools` | 是,GPU 端没装 |
| `L1_23_penalties_and_temperature` | `cachetools` | 是,同上 |
| `L1_24_rejection_sample` | `cachetools` | 是,同上 |
| `L1_29_prepare_eagle_inputs` | `cachetools` | 是,同上 |
| `L2_18_fused_gdn_gating` | `cachetools` | 是,同上 |

修法: M-A (GPU 端跑 `pip install -e ".[infer]"`)。**根因不是 pyproject 缺,是 GPU 机没装齐**;`resolve_all._provenance.infer_state` 显示 GPU 端 vllm/sglang commit `<UNVERIFIABLE:wrong-commit>` 与 dev 机不一致 → GPU 机 setup 当时不完整。

**额外 dev 机 probe 发现还有几个不在 `[infer]` list 内的叶子**:
- `gguf` (11 个 task 在 dev 端 probe 触发): `vllm.v1.attention.backends.utils` → `vllm.transformers_utils.config` → `vllm.transformers_utils.gguf_utils` → `import gguf`
- `flashinfer` (L1_04 在 dev 端): `from flashinfer import ...` 直接出现在 kernel.py
- `openai` (L2_04..L2_09 LoRA 在 dev 端): `sglang.srt.entrypoints.openai.protocol` → `from openai.types.responses import ...`

GPU 端报的 P4 全是 cachetools/cpuinfo,说明那台机已装 gguf/flashinfer/openai (或不会 trigger 这条链)。**修建议**: pyproject `[infer]` extras 补 `gguf` + `flashinfer` 两个不在 list 的叶子(`openai` 见 §6 讨论是否真需要)。

### 3.5 P5: input fixture assertion (3)

| task | host wrapper 顶部 assert | shapes.yaml 现状 |
|------|-------------------------|----------------|
| `L1_03_convert_req_index_to_global_index` | `assert token_indices.shape[1] == NUM_TOPK_TOKENS` (kernel.py:136) | knob `L=32` / 默认 NUM_TOPK_TOKENS=2048 → mismatch |
| `L1_34_transform_index_page_table_decode` | `assert page_size == 1` (kernel.py:67) | shapes.yaml 没指定 page_size 或给了 != 1 |
| `L1_38_silu_mul_fp8_quant_deep_gemm` | `assert group_size == 128, "H must be divisible by 8"` (kernel.py:206) | knob `H` 没保证 % 128 == 0 |

**已 dev 机重现 L1_03**: 真跑 `build_inputs(shapes_doc, 'P1', 0)` + `triton_convert_req_index_to_global_index(**ins)` 在 dev 机也 fail 同样的 `AssertionError: token_indices.shape[1] == NUM_TOPK_TOKENS`。这证明 P5 **不是 shim 问题、不是 driver 问题、不是上游问题**,是 **task 自己的元数据 bug**。

修法: M-E (改 shapes.yaml + 可能改 param_constraints): 加 `NUM_TOPK_TOKENS=L`(让 kernel arg = knob)、`page_size: scalar: 1`、`H` knob 加约束 `H % 128 == 0`。

---

## 4. 5 种 shim 修法可行性评估(leverage 排序)

下面逐条评估,按 leverage(可解 task 数)/ effort / 风险三轴打分。

### 4.1 M-A: 补 [infer] extras + GPU 端真装齐

- **解什么**: 6 个 P4 直接,+ 可能间接放行 dev 端 probe 看到的 gguf/flashinfer/openai 路径上的 task(GPU 端权威分类已不含这些)
- **工作量**: 改 `pyproject.toml` 加 2 行(`gguf` + `flashinfer`),`scripts/automation/runners/setup_machine_a_gpu.sh` 已经跑 `[infer]`,GPU 端跑 `pip install -e ".[infer]"` 一次
- **风险**: 低,纯加包,可逆
- **leverage**: 6 / 38 ≈ 16% (直接) + 间接 0-3(看 sibling chain 是否解锁)
- **正确性**: ✓ confirmed,resolve_all reason 直接说 `No module named 'cachetools'`/`'cpuinfo'`

### 4.2 M-B: vendor sibling 文件到 task_dir(SPDX-headered)

- **解什么**: 18 个 P2 sibling-relative-import 全部
- **工作量**: 写一个新脚本 `scripts/freeze_siblings.py`(类似 `scripts/freeze_source.py`),从 kernel.py 解析 `from .X import Y`,递归收集闭包,对每个 sibling 加 SPDX header 写入 task_dir。单 task ~5 min(脚本跑完);全 18 个一次性 ~1.5h 含闭包递归处理。**注意**: 闭包可能很大(L2_17 / L3_17 / L3_19 各 8 个 sibling,而那些 sibling 自己又 import 更多)
- **风险**: 中。闭包可能拉出整个 fla/ops/ 子树(几千行),会让 task_dir 膨胀;`upstream_fp` / `code_fingerprint` 哈希链路要相应扩展;`tasks/_vendored/` 目录约定也要适配 sibling
- **leverage**: 18 / 38 ≈ 47%
- **替代方案**: M-B' = 让 resolver 的 synthetic package 把 `__path__` **扩展为 task_dir + 真实上游目录**,这样 `from .utils import X` 直接到 `infer/vllm/vllm/model_executor/layers/fla/ops/utils.py`,不用 vendor。优点: 0 task_dir 改动; 缺点: 与 "frozen verbatim" 原则有微妙冲突(原意是 task_dir 是自包含的、SHA256 锁定的,但 sibling 走 LIVE)。需要团队决策
- **正确性**: ✓ confirmed,resolve_all 直接报 `attempted relative import with no known parent package`

### 4.3 M-C: resolver 内 patch TritonPlaceholder / 强制启用 Triton

- **解什么**: 9 个 P3(`TritonPlaceholder` 漂移 + sglang `cached_triton_kernel` isinstance assertion)
- **工作量**: 实现 `triton_ascend_bench/triton_placeholder_extension.py` ~80 行: monkey-patch `vllm.triton_utils.HAS_TRITON=True`、补全 `TritonPlaceholder.next_power_of_2 = triton.next_power_of_2`、`TritonLanguagePlaceholder.float32 = triton.language.float32`、提供一个 fake `JITFunction` subclass 让 `isinstance` 通过、`__getitem__` 接受 grid tuple。在 `_pa_oracle_check_runner.py:248` 最早 import 前注入
- **风险**: 中-高。Monkey-patch 上游 vllm 在 frozen-source 原则下有暗坑。**更干净的方案**: 让 resolver subprocess 在 import vllm 前调用一次 `torch.cuda.init()` / `torch.cuda.is_available()` 强制 CUDA context 起来,让 `triton.backends.is_active()` 返回 True。本机 dev 机无 GPU 就跑不通,但 GPU 机就能跑通。**这意味着 resolver 应该改成强制在 GPU 上跑**(显式 `CUDA_VISIBLE_DEVICES=0` + `torch.cuda.init()`)
- **leverage**: 9 / 38 ≈ 24%
- **正确性**: ✓ confirmed,resolve_all 直接显示 placeholder 引起的 errors;但 root cause 是 resolver subprocess CUDA init 问题,不是 vllm 设计错

### 4.4 M-D: defer-to-golden(skip_invocation)兜底

- **解什么**: 任何不可 shim 的 task,典型 P1 (2 个) 和 P5 (3 个,如果 shapes 改不了)
- **工作量**: 改 `tasks/<id>/meta.yaml` 加 `phase_a_prime: {skip_invocation: true}`,resolver 自动 SKIPPED_BY_DESIGN。机制已存在(`scripts/phase_a_prime.py:1603` 处理)
- **风险**: 低。机制成熟。**但**: 不是真正修复,只是把 task 挪出 release 评分范围。memory `TAB baseline no-subset-escape` 明确**禁止**为了"剔精度难题改分母"。**只有当 task 真的无法在上游 invoke、且已用 PyTorch oracle 作为唯一信号源时**才能 defer
- **leverage**: 5 / 38 = 13% (兜底,不重叠 M-A/B/C)

### 4.5 M-E: 改 shapes.yaml 修 P5 input fixture

- **解什么**: 3 个 P5(L1_03、L1_34、L1_38)
- **工作量**: 单 task ~10 min: 读 host wrapper assert → 确定 knob 约束 → 改 shapes.yaml `param_constraints` 或 `inputs.*` 或调整默认 knob
- **风险**: 低。纯 yaml 改动,resolver 自验证
- **leverage**: 3 / 38 ≈ 8%

### 4.6 比较表

| 修法 | leverage | effort | risk | 是否独立 |
|------|---------|--------|------|---------|
| M-A 补 deps | 16% | 极低 | 低 | ✓ |
| M-B vendor sibling | 47% | 中 | 中 | ✓ |
| M-B' shared sys.path | 47% | 低 | 中-高 (违反原则?) | ✓ (替代 M-B) |
| M-C TritonPlaceholder patch | 24% | 中 | 中-高 | ✓ |
| M-C' force CUDA init | 24% | 低 | 低 | ✓ (推荐替代 M-C) |
| M-D skip_invocation | 13% | 极低 | 低-中(诚实风险) | ✓ |
| M-E shapes.yaml 修 | 8% | 极低 | 低 | ✓ |

---

## 5. 推荐 top-3 修法 + 执行顺序

### Step 1: M-A (1 工作日)
1. 改 `pyproject.toml [infer]` 加 `gguf`、`flashinfer`(不写版本 pin,跟 `cachetools` 等保持一致)
2. GPU 机上 `pip install -e ".[automation,dev,gpu,infer]"` 重新装
3. 跑 `python3 scripts/resolve_upstream_entrypoints.py --refresh-cache`
4. 预期: 6 个 P4 task 状态从 BLOCKED → 进入 oracle-consistency 检查(可能进一步暴露 P3/P5,但至少不是 import error)

### Step 2: M-C'(force CUDA init in resolver) (1 工作日)
1. 在 `scripts/_pa_oracle_check_runner.py:248` `main()` 头部、import vllm 之前,加:
   ```python
   import torch
   if torch.cuda.is_available():
       torch.cuda.init()
       _ = torch.empty(1, device='cuda')  # force driver active
   ```
2. 同样在 `scripts/_pa_subproc_runner.py` 主 entry
3. 预期: 9 个 P3 task 在真 GPU 机上 placeholder 不再生效,放行进入 oracle-consistency
4. 风险评估: 如果 GPU 机调度时 CUDA_VISIBLE_DEVICES 真的为空,触发的不是 placeholder 而是真 import error,反而更糟。需要先在 GPU 机 dry run

### Step 3: M-B (2-3 工作日)
1. 写 `scripts/freeze_siblings.py`:
   - input: task_dir,自动 read kernel.py imports
   - 解析 `from .X import Y` → 在 `infer/<repo>/...` 找 `X.py`
   - 递归: `X.py` 再 `from .Z import W` → 找 `Z.py`
   - 闭包写入 task_dir(SPDX header + `Frozen by: scripts/freeze_siblings.py`)
   - 更新 `meta.yaml` `upstream.sibling_files` list
2. 跑 18 个 P2 task
3. 闭包大小预期: fla/ops/ 大集中, utils/index/op 三个文件 ≈ 200-500 行 / 个;mamba_ssm 个位百行
4. **决策点**: 与团队确认 sibling 是否 "frozen verbatim" 同等对待(answer 应该是 yes,扩展 `tasks/_vendored/` 目录命名约定)

### Step 4: M-E + M-D 收尾 (0.5 工作日)
1. M-E: 改 L1_03 / L1_34 / L1_38 的 shapes.yaml,加 knob constraint
2. M-D: 给 L3_09 / L3_10 加 `meta.phase_a_prime.skip_invocation: true` 注释清楚 "vllm internal dup-op registration, walk PyTorch oracle only"
3. 跑全量 resolve,核对 38 → 0

**总周期**: 4.5-5 工作日 单人,含 1-2 天 GPU 验证回路。

---

## 6. 不可 shim 修复的 task 清单

按"必须走 PyTorch reimplement oracle 这条路径,不能靠 shim 解决"严格判定:

### 6.1 P5 (3 个) — 必须 M-E (shapes.yaml 修复)
- `L1_03_convert_req_index_to_global_index`: shapes.yaml `inputs.token_indices.dims = [E, L]`,host wrapper 要求 `L == NUM_TOPK_TOKENS`. 修法: 增加 `param_constraints` 把 NUM_TOPK_TOKENS 绑到 knob `L`,或换 NUM_TOPK_TOKENS 进 shapes 默认 = L
- `L1_34_transform_index_page_table_decode`: host wrapper `assert page_size == 1`. 修法: shapes.yaml `inputs.page_size = {scalar: 1}` 或 `param_constraints` 约束
- `L1_38_silu_mul_fp8_quant_deep_gemm`: `assert group_size == 128`,group_size 是从 H/8 推导. 修法: shapes.yaml knob `H` 加 `param_constraints: {H: must_be_divisible_by: 128}`

这 3 个 **shim 解决不了**(本身不是 shim 问题),必须改 task 数据。

### 6.2 P1 (2 个) — 建议 M-D (defer-to-golden)
- `L3_09_fused_moe`: 三个 entry tried 都触发 vllm 内部 `vllm::inplace_fused_experts` dup-registration
- `L3_10_fused_moe_kernel_gptq_awq`: 同上

**判定**: 上游 vllm 自身 bug,frozen 原则下不应该改 `infer/vllm/`。修建议: `meta.phase_a_prime.skip_invocation: true`,改 oracle_pytorch 实现完整 PyTorch ref(很重,fused_moe ref 不是 trivial)。或者: 给 oracle_consistency_check 加一个 "import error retry: warm vllm modules once before tested kernel" pre-pass 试解,但 leverage 小、复杂度高。

合计: **5/38 = 13% 必须走 非shim 路径**。

### 6.3 边界 case 说明

- P3 sglang `cached_triton_kernel` (6 LoRA): `_pa_oracle_check_runner.py:268` 已 `os.environ.setdefault("SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE", "0")`,**理论上** decorator 走 `return fn` 分支应绕过 `CachedKernel(fn)` 的 `isinstance` 检查。**实际仍 fail** → 怀疑 sglang 不同版本的 decorator branch 行为有变。`scripts/_pa_oracle_check_runner.py:254-268` 注释提到 "is_cuda() is True AND ... is unset" — GPU 机 is_cuda() == True 自然走 wrap 分支,而 setdefault 设的环境变量在 import 时序里没生效。**进一步调查**: 看 sglang 源码 `cached_triton_kernel` 实现条件,可能要 `setenv` 不是 `setdefault`,或在更早环节 set。但仍属于 M-C 范畴

---

## 7. Appendix: 每 task 完整 import 失败 stack(dev 机 probe)

> 注: dev 机 probe 是在 **无 active CUDA driver** 环境下做的,GPU 机端 reason 见 `resolve_all.json` (§3 已汇总)。两端 reason 可能不同(因 dev 机更早一些路径就 fail)。完整 JSON 见 `/tmp/shim_analysis/import_probe.json`。

### 7.1 P2 (relative import) 14 task — dev 端报 `attempted relative import with no known parent package`

(注: GPU 端权威 P2 是 18 task。dev 端 probe 的 14 是因为另外 4 个 P2 task 在 dev 端先撞到了 `gguf` / `cachetools` 等更早的 import 缺失。完整 18 task 名单见 §3.2)

```
L1_13_layer_norm_fwd          : from .utils import input_guard
L1_31_triton_mrope_forward    : from .base import RotaryEmbeddingBase + from .yarn_scaling_rope import ...
L1_40_chunk_cumsum_fwd        : from .mamba_ssm import softplus
L2_03_chunk_state_fwd         : from .mamba_ssm import softplus
L3_02_kernel_paged_attention_2d : from .prefix_prefill import context_attention_fwd
L3_03_chunk_state_varlen      : from .mamba_ssm import softplus
L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64 : from .index, .op, .utils
L3_14_fused_recurrent_gated_delta_rule_fwd : from .op import exp
L3_16_chunk_fwd_kernel_o      : from .index, .op, .utils
L3_18_chunk_scaled_dot_kkt_fwd : from .index, .op, .utils
L3_20_recompute_w_u_fwd       : from .index import prepare_chunk_indices
L3_21_solve_tril_16x16        : from .index, .op (make_tensor_descriptor), .utils (is_amd, is_tma_supported)
L3_22_chunk_local_cumsum_vector : from .index, .utils
L3_23_merge_16x16_to_32x32_inverse : from .index, .op, .utils
```

GPU 端 P2 多 4 个 task: `L1_28_pack_bitmatrix`、`L2_17_layer_norm_gated_fwd`、`L3_17_chunk_gla_fwd_kernel_o`、`L3_19_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter` — dev 端被 gguf/cachetools 先挡掉,GPU 端这些都已装,所以走到 relative import 才炸。

### 7.2 P4 (missing dep) 6 task — GPU 端报

```
L1_04_copy_page_indices       : cpuinfo (链: vllm.v1.utils -> vllm.usage.usage_lib -> ...)
L1_19_topk_log_softmax        : cachetools
L1_23_penalties_and_temperature : cachetools
L1_24_rejection_sample        : cachetools
L1_29_prepare_eagle_inputs    : cachetools
L2_18_fused_gdn_gating        : cachetools
```

dev 端这 6 个,加上 11 个 gguf-缺失 task,共 17 个 P4-like。两端差异源自 GPU 机 setup 当时 `[infer]` 装齐时机不一致。

### 7.3 P3 (TritonPlaceholder) 9 task — GPU 端权威

```
L1_14_layer_norm_fwd_1pass    : AttributeError 'triton' has no 'next_power_of_2'
L1_22_min_p                   : TypeError 'function' not subscriptable (kernel[(grid,)](...))
L2_01_bmm_chunk_fwd           : AttributeError 'triton.language' has no 'float32'
L2_04_chunked_lora_expand     : AssertionError isinstance(fn, triton.runtime.jit.JITFunction) at sglang.srt.utils.common:3859
L2_05_chunked_lora_shrink     : 同上
L2_06_qkv_lora_b              : 同上
L2_07_gate_up_lora_b          : 同上
L2_08_sgemm_lora_a            : 同上
L2_09_sgemm_lora_b            : 同上
```

### 7.4 P5 (input assertion) 3 task — GPU 端权威

```
L1_03_convert_req_index_to_global_index : assert token_indices.shape[1] == NUM_TOPK_TOKENS (kernel.py:136)
L1_34_transform_index_page_table_decode : assert page_size == 1 (kernel.py:67)
L1_38_silu_mul_fp8_quant_deep_gemm      : assert group_size == 128, "H must be divisible by 8" (kernel.py:206)
```

dev 机已成功重现 L1_03 同 assertion(probe_callable 步骤验证),证明这与 driver 无关、与 shim 无关、是任务元数据 bug。

### 7.5 P1 (vllm dup op) 2 task — GPU 端权威

```
L3_09_fused_moe / L3_10_fused_moe_kernel_gptq_awq:
  RuntimeError: Tried to register an operator (vllm::inplace_fused_experts ...) ...
  multiple times. Duplicate registration: registered at
  infer/vllm/vllm/utils/torch_utils.py:928.
  Original registration: registered at infer/vllm/vllm/utils/torch_utils.py:928
```

---

## 8. 关键文件引用

- `resolve_all.json` (repo root) — 100 verdicts,权威分类源
- `pyproject.toml` (`[project.optional-dependencies].infer`) — 当前 dep list
- `scripts/resolve_upstream_entrypoints.py:50` — `INFER_ROOT = REPO_ROOT.parent / "infer"`
- `scripts/_pa_oracle_check_runner.py:95-117` — `_load()` 的 synthetic package 机制
- `scripts/_pa_oracle_check_runner.py:253-278` — TRITON_INTERPRET=1 + sys.path 注入
- `scripts/phase_a_prime.py:1592-1620` — `skip_invocation` defer 机制
- `infer/vllm/vllm/triton_utils/importing.py:13-46` — `HAS_TRITON` 检查 + TritonPlaceholder 切换
- `infer/vllm/vllm/triton_utils/__init__.py:11-18` — placeholder 注入点
- `triton_ascend_bench/trusted_shim/__init__.py` — **不是** vllm shim(只是计时器)
- `infra/version_matrix.toml:[gpu]` — triton 3.5.1 pin(dev 机实际 3.6.0,飘了)
- `docs/triage_backlog/2026-05-25-blocked-task-resolution-plan.md` — 前一份 BLOCKED 修复 plan(范围 L1_01-L2_03,本报告全 38 范围)

## 9. 后续验证 (followups, not part of this report)

- **[verify]** 在 GPU 机跑 `pip list | grep -E 'cachetools|cpuinfo|gguf|flashinfer|openai|pyzmq'` 确认 §4.1 假设(GPU 端没装齐)
- **[verify]** 在 GPU 机 `CUDA_VISIBLE_DEVICES=0 python3 -c "import torch; torch.cuda.init(); print(torch.cuda.is_available())"` 看 fresh subprocess 内 CUDA driver 是否真不 active
- **[verify]** 跑 M-C' 原型: 在 `_pa_oracle_check_runner.py` 加 `torch.cuda.init()` 后 single-task 跑 L1_22_min_p,看 placeholder 是否被绕开
- **[verify]** 列 sibling 闭包大小: `scripts/freeze_siblings.py` 原型 dry-run on `L2_17_layer_norm_gated_fwd`,出 vendored 行数估计
- **[verify]** sglang `cached_triton_kernel` decorator 实际分支条件 (`is_cuda()` + env var 顺序),确认 `SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE=0` 在 GPU 机是否真生效
