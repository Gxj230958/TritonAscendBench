# 进程隔离架构

默认评测入口采用[流式生命周期](../../reference/streaming_evaluation.md)：
每次仅有一个未消费 trial，通过封存 memfd/safetensors 传输，输出立即验证，
golden 按当前 shape 加载和释放。每个 pass 使用私有缓存与 scratch；清理只作用于
本次运行拥有的目录。下图包含部署层尚待实现的隔离能力，不能视为 Python 进程
隔离已提供的完整保证。

> English: [../../reference/process_isolation.md](../../reference/process_isolation.md)

> 两层隔离。**已交付（shipped）**、可在开发机运行的那一层
> （`ascendbench/integrity/process_isolation.py`）在任意机器上都强制执行与宿主无关
> 的子集：子进程隔离、仅 safetensors 的 ABI，以及 oracle/golden/shapes 不可见。
> 下图所示的**加固版 verified-runner**
> 隔离环境（Docker / seccomp / cgroup / NPU 设备
> allowlist）是 910B 部署目标——属于未来工作，参见
> `docs/DESIGN.md §6`。打分模型见：`docs/DESIGN.md §4`。

## 三进程模型

```
┌─── Trusted Runner Process (host) ────────────┐
│  • Loads oracle_pytorch.py                   │
│  • Computes golden_outputs                   │
│  • Holds tolerance_policy, scoring spec      │
│  • Reads sampled outputs from candidate      │
│  • Performs allclose check                   │
└──────────────────────────────────────────────┘
        │ fork() with restricted mount
        ↓
┌─── Candidate Process #1 (correctness sandbox) ─┐
│  • No network, read-only FS, seccomp           │
│  • mount: solution.py + materialised inputs    │
│  • NOT mount: oracle_pytorch.py, shapes.yaml    │
│  • NOT mount: golden_outputs        │
│  • NOT mount: cost-model GPU-measured baseline  │
│  • cgroup: 32 GB mem, 16 core                   │
│  • Trusted shim wraps solution.npu_triton_impl  │
│  • Writes output.safetensors (NOT .pt)          │
└─────────────────────────────────────────────────┘
        │ exit + SIGKILL process group
        ↓ (delete owned pass directories; device context exits)
┌─── Candidate Process #2 (timing sandbox) ──────┐
│  • Same mounts as Process #1                   │
│  • Trusted shim runs warm-up + measured loops  │
│  • Uses CLOCK_MONOTONIC_RAW (host time)        │
│  • Cross-checks with torch.npu.Event           │
│  • Writes timing + sampled outputs to pipe     │
└─────────────────────────────────────────────────┘
        │ exit + SIGKILL
        ↓
Trusted Runner reads pipe, computes verdict
```

## 为什么要用两个候选实现进程

正确性检验阶段（correctness pass）和计时检验阶段（timing pass）**绝不能**共享状态。

否则：候选实现可能在正确性检验阶段预先计算出答案，缓存在 `/tmp` 或设备内存中，然后在计时检验阶段直接查表读取（从而在被测量的试验中支付约为 0 的计算成本）。

强制执行的缓解措施：
- 每个阶段使用不同的进程
- 每个阶段结束后执行 SIGKILL
- 每个阶段独立使用 HOME、scratch、编译缓存和 profiler 目录
- 删除已完成阶段拥有的目录；设备分配随所属进程退出而释放

## 为什么输出用 Safetensors（而非 Pickle）

`.pt` / `.pkl` 基于 pickle → 加载时会执行任意代码。

如果 trusted runner 执行 `torch.load(candidate_out.pt)`，候选实现就能构造恶意 pickle。

因此：
- 候选实现写出 `output.safetensors`（仅含数据的格式）
- runner 使用 `safetensors.torch.load_file()` 读取（不执行代码）
- 在加固版 verified-runner（未来工作）中：输出元数据（dtype/stride/storage_offset/data_ptr/alias）由 trusted shim 签名，并通过 runner 控制的管道发送——在序列化**之前**完成校验

## Trusted Shim（加固版 verified-runner——未来工作）

由 runner 控制的二进制文件，挂载在 `/runner/trusted_shim.so`，候选实现无法修改。

Shim 的职责：
1. 接收 `inputs`（从 /runner/inputs/ 读取）
2. 调用 `solution.npu_triton_impl(inputs)`
3. 在序列化**之前**：检查 `output` 的 dtype/shape/stride/storage_offset/data_ptr
4. 校验 `output.data_ptr() != any input.data_ptr()`（不与输入产生 alias）
5. 通过 runner 控制的已签名管道发送元数据
6. 将 `output` 序列化为 `safetensors`
7. 写入 `/runner/output.safetensors`
8. 管理 CLOCK_MONOTONIC_RAW 计时窗口
9. 禁止候选实现 import `torch.npu.Event` / `time.perf_counter` / `LD_PRELOAD`

## 沙箱限制（加固版 verified-runner——未来工作）

```bash
docker run --rm \
    --network=none \
    --read-only \
    --tmpfs /tmp:rw,size=256m \           # writable scratch only
    --device=/dev/davinci0 \              # NPU device allowlist
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --cap-drop=ALL \
    --security-opt seccomp=/runner/seccomp.json \
    --cgroup-rules='c 200:0 r' \
    --memory=32g --cpus=16 \              # cgroup limits
    --user=runner:runner \                # non-privileged
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /runner/inputs:/inputs:ro \
    -v /runner/solution:/solution:ro \
    -v /runner/kernel:/kernel:ro \
    -v /runner/output:/output:rw \
    -v /runner/shim:/runner_shim:ro \
    tab-npu:verified candidate_entry.py
```

## 候选实现沙箱内禁止的操作

- 网络（不允许覆盖 `--network=none`）
- 在 `/output/` 之外写文件
- 在没有 runner broker 签名的情况下对 `.so` 执行 `dlopen`
- `torch.npu.Event` / `time.perf_counter` / `LD_PRELOAD`
- 读取 `/runner_shim/`（shim 源码）
- 读取 `oracle_pytorch.py`、`golden_outputs`，或 cost-model 的 GPU 实测基线
- 向**任何**端点发起网络出站流量（**无**上传例外——上传由 runner 在沙箱外部完成）
