# 使用指南

本指南带你从一次全新 checkout 走到一个被评分的任务。它是 [README](../../README.zh.md)
Quick Start 的动手版；想了解 *bench 的工作原理*,接着读 [ARCHITECTURE.md](ARCHITECTURE.md)。

## TL;DR —— 整个流程

```bash
pip install -e .                                        # 从 clone 或 release tarball
tab init-solution L2_04_silu_and_mul                    # scaffold solution.py
#   把 npu_triton_impl 实现成真 Triton 内核             # 完整例子见 §3
tab run L2_04_silu_and_mul \
    --solution solutions/L2_04_silu_and_mul/solution.py  # 评分 -> round_result.json
#   在 solution 目录里编写 manifest.yaml                # validate-submission 需要它
tab package solutions/L2_04_silu_and_mul --out submission.tar.gz
tab validate-submission submission.tar.gz               # 之后:tab submit ... --endpoint <url>
```

每一步下面详述。

## 1. 获取 bench 并安装

`tasks/`(任务语料)随 **git clone 或 release tarball** 一起提供 —— 它**不在** PyPI wheel 里。
在一个 checkout 中:

```bash
pip install -e .            # 核心
pip install -e '.[infer]'   # 可选:仅 provenance/severance 工具需要
```

需要 Python 3.10+。验证 CLI:

```bash
tab --help                  # 或:python3 -m ascendbench.engine.cli --help
```

## 2. Scaffold 一个 solution

```bash
tab init-solution L2_04_silu_and_mul
# -> scaffolded solutions/L2_04_silu_and_mul/solution.py
```

`init-solution` 只写**一个 `solution.py`**(不生成 manifest —— `manifest.yaml` 由你在 package
阶段编写)。用 `--out <dir>` 指定位置。

stub 定义了你必须实现的入口:

```python
def npu_triton_impl(inputs: dict) -> dict:
    """`inputs` 由任务的 shapes 构造;返回一个输出张量的 dict,
    键名按 contract 指定。"""
    ...
```

## 3. 实现你的内核

读 `tasks/<level>/<task_id>/contract.md` 了解 I/O dict 键名、tolerance class 和 shape 覆盖。
grader 强制两条规则(见 [reward_hack_defense.md](reference/reward_hack_defense.md)):

- 入口**必须是 `npu_triton_impl(inputs) -> dict`**。
- 你的方案**必须是真正的 Triton 内核** —— `@triton.jit` + `tl.load`/`tl.store`。纯 PyTorch
  实现会被**静态检查器拒绝**(它的存在就是为了挡住"直接用 torch 算"的提交)。

一个完整、可直接复制的 `L2_04_silu_and_mul` 的 `solution.py`(输出 = `silu(x1) * x3`,
其中 `hidden_states` 从中间劈成 `x1, x3`):

```python
import torch
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_kernel(x_ptr, out_ptr, hidden_dim, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < hidden_dim
    x1 = tl.load(x_ptr + row * n_cols + col, mask=mask, other=0.0)
    x3 = tl.load(x_ptr + row * n_cols + hidden_dim + col, mask=mask, other=0.0)
    x1f = x1.to(tl.float32)
    silu = x1f * tl.sigmoid(x1f)
    out = x3 * silu.to(x3.dtype)
    tl.store(out_ptr + row * hidden_dim + col, out, mask=mask)


def npu_triton_impl(inputs: dict) -> dict:
    hs = inputs["hidden_states"]                 # 键名来自 contract.md
    bs, n_cols = hs.shape
    hidden_dim = n_cols // 2
    out = torch.empty((bs, hidden_dim), dtype=hs.dtype, device=hs.device)
    BLOCK = 128
    grid = (bs, triton.cdiv(hidden_dim, BLOCK))
    _silu_and_mul_kernel[grid](hs, out, hidden_dim, n_cols, BLOCK=BLOCK)
    return {"output": out}                       # 输出键名来自 contract.md
```

输入/输出键名(这里是 `hidden_states` / `output`)和 tolerance class 都从该任务的
`contract.md` 拿;不同任务键名不同。

## 4. 评分单个任务

`--solution` 指向 **`solution.py` 文件**(不是目录):

```bash
tab run L2_04_silu_and_mul --solution solutions/L2_04_silu_and_mul/solution.py
# 打印一行摘要(score=… coverage=… static_ok=… → round_result.json)
```

`round_result.json` 记录逐维度判定:

| 字段 | 含义 |
|---|---|
| `static_checks_passed` | 是否通过反作弊静态门(真 Triton 内核、无禁用调用) |
| `correctness_status` | `ok` / `fail` / `error` —— 内核输出 vs PyTorch oracle 是否在 tolerance 内 |
| `runtime_status` | `ok` / `error` / … —— 内核是否执行 |
| `performance_status` | `ok` / … —— 计时/roofline 结果 |
| `task_status` | `evaluated` / `blocked` / … —— 总体 |
| `coverage` | 通过的 shape 占比 |

(各 status 的枚举比这里列的常见值更宽;上面是你最先遇到的那些。)

> `VERIFIED` / `CEILING_ACCEPTED` 是**语料 provenance** 状态(参考内核如何被认证的),
> 不是你 `tab run` 的判定。你的 run 报告的是上面的 `*_status` 字段。

### 读精度 + 性能细节

顶层 `*_status` 是汇总。**逐 shape** 的细节在 `round_result.json` 的 `shape_results[]`
里 —— 每个 shape 一条,以 `shape_id` 为键:

**精度**(逐 shape):
- `correctness_passed` —— 该 shape 输出是否在容差内匹配 oracle。
- `correctness_reason` —— 过/挂的原因(容差细节,或报错)。
- `metamorphic_results` —— metamorphic 检查结果。

**性能**(逐 shape):
- `efficiency` —— Roofline 效率(实测吞吐 vs 峰值)。
- `t_ref` / `t_npu` —— 参考时间 vs 实测设备时间。
- `baseline_ns` / `measured_ns` / `median` / `p99` / `cv` / `trials` —— 延迟统计。
- `w_per_class` / `m_per_resource` —— 按资源类(vector / cube)的工作量与实测代价。
- `roofline_error` —— 任何性能测量问题。

几行 Python 读出来:

```python
import json
d = json.load(open("round_result.json"))
for s in d["shape_results"]:
    print(s["shape_id"], "correct:", s["correctness_passed"],
          "| reason:", s["correctness_reason"], "| eff:", s["efficiency"])
```

`L2_04_silu_and_mul` 在纯 CPU(无设备)smoke 出的一条真实逐 shape 记录的**摘录**
(上面列出的其余 emitted 键为简洁在此省略):

```json
{"shape_id": "P1", "correctness_passed": false,
 "correctness_reason": "RuntimeError: 0 active drivers ([]). There should only be one.",
 "t_ref": 1.28e-09, "t_npu": null, "efficiency": null, "measured_ns": null}
```

> **真数字需要目标后端。** 解释器能干净跑内核时,CPU 上的 correctness *可以*是真的(张量留 host);
> 但**没有可用目标后端时,性能字段是 `null`/`error` 或 host/解释器诊断,不是权威 910B 数**。
> Roofline 性能需要真实 Ascend 910B(README §Status)。上面这条是无设备的情形,连 correctness
> 也是 `error`(`0 active drivers`)。

**agent 该断言什么:** run 通过 = `correctness_status == "ok"` 且 `coverage == 1.0`;逐 shape
看 `correctness_passed`(精度)和 `efficiency`(性能)。摘要行的单一 `score` 是 emitted
`aggregate_score`(有 Roofline 时是 `s_t`,否则是 legacy 诊断值);`coverage` 单独报告。
最终打分用「正确性-coverage gate + Roofline 性能指标」(见
[scoring_spec.md](reference/scoring_spec.md))。

**真实 passing 结果长啥样。** `L2_04_silu_and_mul` 参考内核的 committed metadata 记录
**每个 shape `VERIFIED`、`max_abs = 0.0`**(精确匹配 oracle)。性能在参考 CUDA-GPU 打分上测
(非 910B 目标):这种小 vector 算子的逐 shape Roofline `efficiency` 很低 —— 量级 `0.02`、
设备延迟几微秒(`do_bench` 测)。这些性能数字是**示意的 reference-daemon 值,非仓库可核、
也非 910B 榜单分**。Roofline 效率按算子判,不是对着固定门槛。

### 验证你的 kernel:精度对齐 + 性能确认

`npu_triton_impl` 一跑,`tab run` 回答两件事 —— 输出跟参考**数值对齐**了吗、**性能**如何。

**对齐精度。** 每个 task 声明一个 **tolerance class**(`meta.yaml` / `contract.md` §Tolerance
class —— L2_04 是 `elementwise`)。你的输出按该 class 跟 PyTorch oracle 比:float `allclose`
(该 class 的 atol/rtol)、int8 量化用 `int_atol`、fp8 用 ULP。逐 shape `correctness_passed`
在容差内为 `true`;数值不过时 `correctness_reason` 带比对细节 —— float 报 `max_abs`/`max_rel`,
int-atol 报 `max_abs > int_atol`,fp8 报最大 ordered-code 距离 vs ULP cap;其它失败报具体
mismatch/报错。常见修法是**匹配 oracle 的
数值语义** —— task 的 `contract.md` 直接点出承重处(L2_04:*"cast-back to input dtype before
the multiply is load-bearing"* —— fp32 算 SiLU、cast 回、再乘)。
→ [tolerance_policy.md](reference/tolerance_policy.md)、[correctness_model.md](reference/correctness_model.md)。

**确认性能。** 逐 shape `efficiency = min(cap, t_ref / t_npu)` —— cost-model 参考时间 / 你的
实测设备时间(`t_npu` = 中位实测延迟;GPU rank 助手用 `triton.testing.do_bench` 做参考代理计时)。
`efficiency` 越高越接近 Roofline;`measured_ns`
/ `median` / `p99` 是原始延迟,`s_t` 汇总吞吐分。
→ [timing_arbitration.md](reference/timing_arbitration.md)、[scoring_spec.md](reference/scoring_spec.md)。

> 精度在**任何能跑的后端**都能对 —— 真 NPU/GPU,或解释器可跑的内核在 CPU 解释器上。权威**性能**
> (Roofline 效率 vs 910B 峰值)需真 Ascend 910B(README §Status)。非 910B 的计时(若有)是参考
> 代理、不是榜单分;纯 CPU 跑通常让性能字段是 `null`/`error`。

### 在目标 NPU(Ascend 910B)上运行

`tab run` 和 `run-suite` **默认 `--backend npu`** —— NPU 路径把你的 triton-ascend 内核跑在真
Atlas 910B 上。(`--backend gpu` 是 CUDA 自检路径,**不是**权威 910B 运行。)

**环境。** 真 NPU 运行需 `torch_npu` + `triton-ascend` + CANN 及驱动环境
(`LD_LIBRARY_PATH`、`ASCEND_TOOLKIT_HOME`、`ASCEND_OPP_PATH`)。参考版本 pin 在
[`docker/Dockerfile.npu`](../../docker/Dockerfile.npu)(CANN `8.5.0`、`torch_npu 2.8.0.post1`、
`triton_ascend 3.2.0`);裸机 910B 搭建在本仓之外。

**没 NPU 是降级,不是回退 GPU。** 没 `torch_npu` 时 NPU 后端无法把张量放上设备;隔离 runner 把它们
留在 CPU、无设备计时,结果标 `npu_antihack_unverified` —— 不是有效 NPU 证明。真结果需真 910B;它
**不会**悄悄回退到 GPU。

**不能用 torch 作弊。** NPU 上真 triton-ascend 内核必须发一个 device kernel;runner 在 parent 侧
解析 NPU trace 找 kernel event。无 kernel 发射的 torch-only 路径被标记(`fallback_detected` /
`torch_fallback`),强制 `correctness_status = "fail"`,终评把该 task 塌到 ~0。

**结果读法一样**(`shape_results[]`,见上)。910B 上 Roofline `efficiency` 对着打包的 910B peak 表
(`ascendbench/policies/peak_table_910b.toml`)归一化。部分 peak 项是估计(`bf16_cube` 架构估、
`vector` 学术、`interconnect` placeholder),所以绝对效率受 `absolute_verified` /
`roofline_data_complete` 门控 —— 把 `efficiency` 当归一化排名信号,不是硬件精确百分比。

**实用提示。** 真 NPU Triton 运行会在打分调用前做一次编译/JIT 预热;`run-suite` 也接
`--backend npu`;`upstream_oracle_required` 任务的 verified-mode 打分可能需其离线 golden 制品。

> **来源说明。** 本节据代码 + 已 ship 的 NPU runner 写成。精确的裸机搭建与在设备上的行为**未在本次
> 文档轮次复验**(此处可达的 910B 缺 `torch_npu`)。

## 5. 评分整个 suite

```bash
tab run-suite --solutions-root solutions/    # root 下每个任务一个 solution.py
```

## 6. 打包与提交

```bash
tab package solutions/L2_04_silu_and_mul --out submission.tar.gz
tab validate-submission submission.tar.gz     # 本地校验 bundle
tab submit submission.tar.gz --endpoint <url> # 校验 package;见下
```

`tab package` 在 `manifest.yaml` 缺失时**告警**(按 `ascendbench.manifest.SolutionManifest`
编写);`validate-submission` 随后强制 schema。**`tab submit` 做校验并报告** —— 托管
leaderboard 的 POST 是 roadmap 的 v0.1 服务(README §Leaderboard),准备一个提交并不需要
在线 endpoint。

## Worked example —— CPU smoke

在**仅 CPU 的机器**上,你可以跑通工作流、静态门,以及 correctness(无加速器时 runner 把张量
留在 CPU 上);权威**性能**(Roofline)需要目标 910B。下面是 dev 机上诚实的端到端:

```bash
tab init-solution L2_04_silu_and_mul --out /tmp/demo/sol
```

**(a) 纯 torch 的尝试被拦下** —— 证明反作弊门有效:

```text
static_checks_passed: False
findings: "no @triton.jit kernel found", "no tl.load found", "no tl.store found",
          "banned attribute path 'torch.nn.functional.silu'",
          "host-side Tensor BinOp (eager evaluation; use a triton kernel)"
```

**(b) 真 Triton 内核通过静态门**;用 Triton 解释器做 CPU smoke:

```bash
TRITON_INTERPRET=1 tab run L2_04_silu_and_mul --solution /tmp/demo/sol/solution.py
# static_ok=True   correctness_status=fail   runtime_status=error   coverage=0.00
```

这里静态门通过(`static_checks_passed: true` —— 是合法 Triton 内核),但这个最小内核在 CPU
解释器下出错了(`runtime_status=error`,于是 `correctness_status=fail`、`coverage=0`)。
correctness 评分本身能在 CPU 上跑:一个在 `TRITON_INTERPRET=1` 下干净执行的内核会得到真实的
`correctness_status=ok`。smoke 的作用是验证工作流 + 静态门;权威**性能**(Roofline)仍需真实
Ascend 910B 硬件(README §Status)。

## 排错

| 症状 | 原因 / 修法 |
|---|---|
| `No such task` / 缺 `tasks/` | 你装的是 wheel;`tasks/` 需要 **clone 或 release tarball**。 |
| `--solution` 报 `IsADirectoryError` | 传 **`solution.py` 文件路径**,不是目录。 |
| `static_checks_passed: false`、"no @triton.jit kernel" | 方案必须是**真 Triton 内核**,不是 PyTorch。 |
| `TRITON_INTERPRET=1` 下 `runtime_status: error` | 内核没在 CPU 解释器下执行成功;简化它或在目标 NPU/GPU 上跑。correctness 本身对解释器可跑的内核在 CPU 上就能评。 |
| `NotImplementedError: implement npu_triton_impl` | scaffold stub —— 去实现入口。 |

## 下一步

- **bench 如何工作** → [ARCHITECTURE.md](ARCHITECTURE.md)
- **设计理念** → [DESIGN.md](DESIGN.md)
- **深度参考**(correctness、scoring、shapes、integrity、provenance)→
  [docs/zh/reference/](reference/)
