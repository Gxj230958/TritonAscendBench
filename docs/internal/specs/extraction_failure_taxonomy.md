# Extraction Failure Taxonomy

> v0 — per v4.13 Phase 1.4 + R7-6

Categories of failures when `scripts/internal/extract_upstream_kernel.py` tries to pull a kernel from `infer/vllm` or `infer/sglang` upstream.

| Failure | Cause | Resolution |
|---------|-------|-----------|
| `constexpr_runtime_dep` | `tl.constexpr` argument computed from runtime value | Manually vendor + annotate `extraction: manual` in `meta.yaml` |
| `autotune_multi_config` | `@triton.autotune` with multiple variants | Select 1 representative config, document choice in `contract.md` |
| `runtime_meta_param` | `num_warps`/`num_stages` set at call site | Fix to default values; mark in `meta.yaml` |
| `deep_helper_chain` | Helper functions span >3 modules | Manually vendor entire chain |
| `cpp_stub_dep` | Calls into custom C++ extension (e.g. `torch.ops._C.*`) | No `@triton.jit` upstream → cannot form a valid task; **excluded at extraction** (distinct from `BLOCKED`, which stays in `\|S\|`). The PyTorch oracle is **not** an independent truth source and cannot substitute (`docs/DESIGN.md §1.3`) |
| `dynamic_shape_branch` | `tl.where` with shape-dependent branches | Create multiple shape configs, document branch coverage |
| `vendor_specific_op` | Calls `torch_npu.*` / `sgl_kernel_npu.*` | Not a Triton kernel — exclude from benchmark |
| `private_no_source` | Bytedance / Huawei internal kernel without OSS source | Move to `_appendix/`, mark `is_hidden_only: true` |

## Resolution Workflow

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
