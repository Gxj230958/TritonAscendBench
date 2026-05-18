# Task API Contract

> **v3.2 ERRATA — AscendC removed.** This is a GPU-triton → NPU-triton *migration* benchmark; there is no AscendC native baseline, no `ascendc_baseline.yaml`, no `in_s_native`/`oracle.ascendc`. Correctness is the scored axis; performance is raw NPU latency (deferred to 910B). AscendC-specific text below is **superseded** and kept only as historical context.


> v0 — derived from `docs/benchmark_impl_plan.md` v4.13 §Phase 2.5 + R11-P1-1 + R14-P1-2
> Source of truth: per-task `meta.yaml`. Human-readable docs (this file + `contract.md`) explain but do not override `meta.yaml`.

## 1. Task Directory Layout

Every task is a directory under `tasks/level{1,2,3}/<task_id>/`:

```
tasks/level1/L1_12_rms_norm_kernel/
├── kernel.py              # GPU triton reference (pure operator, no oracle/meta pollution)
├── oracle_pytorch.py      # PyTorch eager reference (runner-only, not mounted to candidate)
├── meta.yaml              # Structured metadata + runner_checks (source of truth)
├── shapes.yaml            # SHAPE_CONFIGS + ADVERSARIAL cases
├── contract.md            # Human-readable semantic contract
└── ascendc_baseline.yaml  # AscendC native baseline (`available: false` if not in S_native)
```

## 2. `kernel.py` Contract

- Pure operator implementation extracted from vLLM / SGLang upstream
- Must expose `run_kernel(inputs: dict) -> dict[str, Tensor]`
- Must contain at least one `@triton.jit`-decorated kernel
- May be imported by submitter as reference; no oracle/meta-yaml pollution
- Upstream commit SHA + file hash locked in `meta.yaml.upstream`

## 3. `oracle_pytorch.py` Contract

- Independent PyTorch eager reference (no triton)
- Must expose `run_oracle(inputs: dict) -> dict[str, Tensor]`
- Device-agnostic (tensors stay on caller's device)
- **Never mounted to candidate sandbox** — per v4.13 §5b.1a
- Only loaded by `Trusted Runner` process

## 4. Submitter `solution.py` Contract

```python
def npu_triton_impl(inputs: dict) -> dict:
    """
    Required interface for all submissions.
    
    Args:
        inputs: dict with keys per `meta.yaml.interface.alias_map`
                (all tensors on NPU device)
    
    Returns:
        dict with keys matching `meta.yaml.oracle.gpu_triton` output schema
        (all tensors on NPU device; candidate-allocated per R14-P1-2)
    """
```

**Forbidden in `solution.py`** (per AST denylist, `specs/ast_denylist.md`):
- `import oracle_pytorch` / `import triton_ascend_bench` / `import tasks.*`
- High-level torch ops: `torch.nn.functional.*`, `Tensor.matmul`, `@` operator
- Native AscendC ops: `sgl_kernel_npu.*`, `vllm_ascend.npu_*`, `torch_npu.npu_*`
- Code execution: `exec`, `eval`, `compile`, `__import__`, `ctypes`, `cffi`, `torch.utils.cpp_extension`
- IPC / network: `subprocess`, `socket`, `requests`, `urllib`
- Timing manipulation: `torch.npu.Event`, `torch.cuda.Event`, `time.perf_counter`, `LD_PRELOAD`
- File / IO: `os`, `os.path`, `pathlib`, `mmap`, `pickle`, `open` (beyond stdlib)
- Threading: `threading`, `multiprocessing`, `asyncio`

## 5. Mandatory `meta.yaml` Fields

See `specs/meta_yaml_schema.yaml` for full machine-readable schema. Required:

```yaml
schema_version: "1.0"
task_id: "L1_12_rms_norm_kernel"
level: "L1"  # enum: L1, L2, L3, appendix, canary
idx: 12
name: "_rms_norm_kernel"
repo: "vLLM"  # enum: vLLM, SGLang, FBGEMM, Bytedance_internal
dtype_class: "VV"  # enum: VV, CV
category: "norm"  # enum: elementwise, norm, reduce, sampling, kv_cache, lora, attention, moe, state_space
npu_difficulty: "medium"  # enum: low, medium, high

upstream:
  repo: "vllm-project/vllm"
  path: "vllm/model_executor/layers/batch_invariant.py"
  commit_sha: "<frozen at Phase 1.2>"
  function_hash: "<sha256 of kernel source>"
  license: "Apache-2.0"

interface:
  alias_map:
    input: ro
    weight: ro
    output: wo_candidate_allocated  # per R14-P1-2
  mutability: "no_inplace"
  stride_layout: "input_2d.contiguous() required"
  workspace_policy: "none"
  stream_sync_policy: "runner-managed"
  rng_state_policy: "deterministic"

oracle:
  pytorch: "oracle_pytorch.py::run_oracle"
  gpu_triton: "kernel.py::run_kernel"
  ascendc:
    available: true
    config_file: "ascendc_baseline.yaml"

tolerance_class: "vector_reduce"  # references specs/tolerance_policy.yaml

oracle_simplified: "formula_exact_manual"  # per R18-S2 (simple_pytorch_ok / formula_exact_manual / upstream_oracle_required)

runner_checks:
  expected_kernel_count_range: [1, 5]
  min_dram_bytes: 65536
  min_flops: 524288
  metamorphic_checks:
    - "differential_perturbation_general"
  output_format: "safetensors"
  output_allocation: "candidate_allocated"

is_canary: false
is_hidden_only: false
in_s_native: true  # if true → oracle.ascendc.available must be true
```

## 6. Authority + Cross-File Consistency

- `meta.yaml` is **source of truth** for all machine-readable fields.
- `contract.md` is **human-readable only** — must not introduce facts not in `meta.yaml`.
- `shapes.yaml` configs may be augmented at runtime with `shapes_hidden.yaml` (sealed, never in public bundle).
- If `meta.yaml.in_s_native = true` but `oracle.ascendc.available = false` → task validator rejects.
- If `meta.yaml.oracle_simplified = "upstream_oracle_required"` and `oracle_pytorch.py` body contains `F.silu`, `F.gelu`, etc. high-level ops → static check rejects.

## 7. Validator

`tab validate-task <task_dir>` checks:
- All 6 files present (except `ascendc_baseline.yaml` may be `available: false`)
- `meta.yaml` matches schema
- `kernel.py` has at least one `@triton.jit` and exports `run_kernel`
- `oracle_pytorch.py` exports `run_oracle`
- `contract.md` references match `meta.yaml`
- Hash of `kernel.py` matches `meta.yaml.upstream.function_hash`
