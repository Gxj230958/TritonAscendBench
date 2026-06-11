# 抽取失败分类

> English: [../../reference/extraction_failure_taxonomy.md](../../reference/extraction_failure_taxonomy.md)

当 `scripts/internal/extract_upstream_kernel.py` 尝试从 `infer/vllm` 或 `infer/sglang` 上游拉取某个 kernel 时，各类失败情况的分类。

| Failure | Cause | Resolution |
|---------|-------|-----------|
| `constexpr_runtime_dep` | `tl.constexpr` 参数由运行时值计算得到 | 手动 vendor 并在 `meta.yaml` 中标注 `extraction: manual` |
| `autotune_multi_config` | 带有多个变体的 `@triton.autotune` | 选取 1 个代表性 config，并在 `contract.md` 中记录该选择 |
| `runtime_meta_param` | `num_warps`/`num_stages` 在调用点设置 | 固定为默认值；在 `meta.yaml` 中标注 |
| `deep_helper_chain` | 辅助函数跨越 >3 个模块 | 手动 vendor 整条调用链 |
| `cpp_stub_dep` | 调用自定义 C++ 扩展（例如 `torch.ops._C.*`） | 上游没有 `@triton.jit` → 无法构成有效任务；**在抽取阶段被排除**（区别于 `BLOCKED`，后者仍保留在 `\|S\|` 中）。PyTorch oracle **不是**独立的真值来源，无法替代 |
| `dynamic_shape_branch` | 带有依赖 shape(形状) 分支的 `tl.where` | 创建多个 shape 配置，记录分支覆盖情况 |
| `vendor_specific_op` | 调用 `torch_npu.*` / `sgl_kernel_npu.*` | 不是 Triton kernel — 从基准中排除 |
| `private_no_source` | 没有 OSS 源码的 Bytedance / Huawei 内部 kernel | 移至 `_appendix/`，标注 `is_hidden_only: true` |

## 解决工作流

```
1. python scripts/internal/extract_upstream_kernel.py --task L1_xx --from infer/vllm
2. If extraction fails:
   - Categorize per taxonomy above
   - Write resolution note to tasks/_extraction_log/L1_xx.md
3. If resolution = manual vendor:
   - cp upstream kernel to tasks/L?_xx/kernel.py
   - Add commit_sha + function_hash to meta.yaml.upstream
4. If resolution = skip:
   - Add to skip_registry.yaml with reason
   - Generate placeholder task only if appears in scored set
```
