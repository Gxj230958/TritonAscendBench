# Task API Contract

> Authoritative scoring model: `docs/DESIGN.md §9` (two-objective — correctness gate + 910B Roofline-normalized performance). GPU-Triton → 910B NPU-Triton migration; no AscendC baseline.


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
└── contract.md            # Human-readable semantic contract
```

(5-file contract — see `docs/DESIGN.md §5.1`. Auxiliary dirs: `_frozen_deps_pool/`, `_frozen_shim/`, `_canary/`, `_vendored/`.)

## 2. `kernel.py` Contract

- Pure operator implementation extracted from vLLM / SGLang upstream
- Must export the real upstream entrypoint verified by the dual-anchor resolver, recorded in `meta.yaml.oracle.gpu_triton` (`kernel.py::placeholder` until the resolver fills it)
- Must contain at least one `@triton.jit`-decorated kernel
- May be imported by submitter as reference; no oracle/meta-yaml pollution
- Upstream commit SHA + file hash locked in `meta.yaml.upstream`

## 3. `oracle_pytorch.py` Contract

- PyTorch eager reference aligned to frozen upstream semantics via the dual-anchor (no triton); **not an independent truth source** (`docs/DESIGN.md §1.3`)
- Must expose `run(inputs: dict) -> dict[str, Tensor]`
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

The machine-readable schema is enforced by `scripts/validate_tasks.py` (`REQUIRED_META_TOP_LEVEL` + nested-structure checks). Required:

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
  pytorch: "oracle_pytorch.py::run"
  gpu_triton: "kernel.py::placeholder"   # resolver fills the verified upstream entrypoint

tolerance_class: "vector_reduce"  # one of the classes in triton_ascend_bench/tolerance_policy.yaml

cost_model:                          # performance baseline — see docs/DESIGN.md §5.6
  flops: "2*B*L*D"                   # size-knob FLOPs formula (scalar form)
  flops_class: fp16_cube             # scalar form: name the compute class
  bytes: "(B*L*D + B*D) * dtype_size"  # HBM bytes (scalar form)
  bytes_resource: hbm                # scalar form: name the resource
  gpu_xcheck_tol: 0.05               # analytic vs instrumented GPU-Triton mismatch tolerance
  # status (PASS / BLOCKED) is produced by the GPU cross-check, not authored here

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
```

### 5.1 `cost_model` schema (performance baseline)

`meta.yaml.cost_model` supplies the per-task `W_c` (FLOPs) and `M_r` (bytes)
the Roofline baseline `t_ref = max(max_c W_c/π_c, max_r M_r/β_r)` needs
(`docs/DESIGN.md §5.6 / §9`). Three keys:

| key | meaning |
|-----|---------|
| `flops` | per-compute-class FLOPs `W_c`, as a size-knob formula |
| `bytes` | per-resource HBM/interconnect bytes `M_r`, as a size-knob formula |
| `flops_class` | **scalar form only, required** — the compute class the scalar `flops` formula belongs to (`fp16_cube` / `bf16_cube` / `int8` / `vector`) |
| `bytes_resource` | **scalar form only, required** — the resource the scalar `bytes` formula belongs to (`hbm` / `interconnect`) |
| `gpu_xcheck_tol` | analytic-vs-instrumented GPU-Triton mismatch tolerance, in `(0, 1)` (default `0.05`) |

`flops` and `bytes` each accept **two forms**, independently:

- **scalar** — a single formula string, the single-compute-class /
  single-resource degenerate form. A scalar `flops` **must** carry a
  companion `flops_class`, and a scalar `bytes` a companion `bytes_resource`,
  naming the peak-table class/resource the formula belongs to:

  ```yaml
  cost_model:
    flops: "2*B*L*D"
    flops_class: fp16_cube
    bytes: "(B*L*D + B*D) * dtype_size"
    bytes_resource: hbm
    gpu_xcheck_tol: 0.05
  ```

- **map** — `{compute_class: formula}` / `{resource: formula}` when the
  operator mixes compute classes or data resources:

  ```yaml
  cost_model:
    flops:
      fp16_cube: "2*M*N*K"   # matmul on the cube unit
      vector:    "M*N"       # epilogue bias on the vector unit
    bytes:
      hbm:          "(M*K + K*N + M*N) * dtype_size"
      interconnect: "M*N * dtype_size"   # cross-device collective traffic
    gpu_xcheck_tol: 0.05
  ```

  Compute-class keys are the `infra/peak_table_910b.toml` classes
  (`fp16_cube` / `bf16_cube` / `int8` / `vector`); resource keys are its
  resources (`hbm` / `interconnect`).

The validator and the dataset loader accept either form and **normalise to a
map internally** — the scalar form becomes a one-entry map keyed by its
companion `flops_class` / `bytes_resource`, a real peak-table class/resource
(never a synthetic key). Every downstream consumer sees one uniform structure,
and the normalised map is always directly scorable against the peak table. A
`flops_class` / `bytes_resource` on a map-form field is rejected — the map
already keys the class/resource itself.

**Formula language.** A formula is a size-knob arithmetic expression over the
same parameterisation as `shapes.yaml` (`B` / `L` / `D` / …) plus a fixed set
of *derived symbols* — currently `dtype_size` (element size in bytes of the
task's primary dtype). At scoring time `eval.py` derives the **representative
`dtype_size`** as the byte width of the **first declared tensor input** in
`shapes.yaml.inputs` (declaration order, skipping `scalar` / `scalar_knob`
entries) — e.g. a `float32` primary operand gives `dtype_size = 4`, `int64`
gives `8`. A scalar-only `inputs` block has no tensor element size, so the
fp16 default (`2.0`) is used. Per-input precise handling for genuinely
mixed-dtype tasks is a W3.1 long-tail item; until then this single
representative value feeds every `flops` / `bytes` formula. The validator
evaluates formulas with the fp16 default `dtype_size` — it only confirms a
formula *parses and evaluates*, not its numeric value. The shared evaluator
(`triton_ascend_bench/cost_model.py`) defines exactly which derived symbols
exist and is reused verbatim by `dataset.py`, `validate_tasks.py` and the W4.2
GPU cross-check, so the symbol set never drifts. Allowed: `+ - * / // ** %`,
unary `+` / `-`, numeric literals, knob/derived names, and `min` / `max` /
`ceil` / `floor` calls. Anything else (attribute access, subscripts,
comprehensions, unknown names) is rejected — formulas are evaluated through an
AST whitelist, never `eval()`.

**`cost_model` presence — layered (loader lenient, validator strict).** The
two consumers treat a missing `cost_model` block differently *by design*:

- **Dataset loader (`dataset.py`)** — *backward compatible*. A task whose
  `meta.yaml` has no `cost_model` block loads fine with
  `TaskDef.cost_model = None`; a *malformed* block still raises. This keeps
  pre-W3.1 tasks (and any synthetic test fixture) loadable, and lets the
  Roofline path simply skip a task with no cost model rather than fail.
- **Task validator (`scripts/validate_tasks.py`)** — *strict*. As of R1-4
  `cost_model` is a **required** top-level `meta.yaml` key: a missing block is
  a hard validation error. The red→green transition for the 100-task corpus
  is carried by the validator (W3.1 backfills every task), never by the
  loader.

In short: a missing `cost_model` is *not an error to load* but *is an error to
release*. When the block is present it is validated strictly by both.

## 6. Authority + Cross-File Consistency

- `meta.yaml` is **source of truth** for all machine-readable fields.
- `contract.md` is **human-readable only** — must not introduce facts not in `meta.yaml`.
- `shapes.yaml` carries both the public and the hidden eval shapes in one file: a `public_shapes` list and a sealed `hidden_shapes` list (`docs/DESIGN.md §3.3 / §5.2`). There is **no** separate `shapes_hidden.yaml` overlay — `shapes.yaml.hidden_shapes` is the single source of truth for hidden shapes. In `verified` mode the trusted runner materialises the `hidden_shapes` rows' inputs and grades them; in `self-reported` mode the hidden shapes are not materialised. The candidate sandbox never receives the `shapes.yaml` document at all (it gets only runner-materialised inputs — `specs/process_isolation.md`), so the `hidden_shapes` section is structurally absent from the candidate view.
- If `meta.yaml.cost_model` is absent, or its analytic FLOPs/bytes mismatch the instrumented GPU-Triton reference beyond `gpu_xcheck_tol` → `cost_model.status = BLOCKED` (release-gate hard block, `docs/DESIGN.md §4.4/§5.6`).
- If `meta.yaml.oracle_simplified = "upstream_oracle_required"` and `oracle_pytorch.py` body contains `F.silu`, `F.gelu`, etc. high-level ops → static check rejects.

## 7. Validator

`python scripts/validate_tasks.py` (per-task schema / hash validation; `--release-gate` runs the S6 gate) checks:
- All 5 files present (`kernel.py` / `oracle_pytorch.py` / `meta.yaml` / `shapes.yaml` / `contract.md`)
- `meta.yaml` matches schema; the top-level `cost_model` key is **required** (R1-4) — a missing block is a hard error. `cost_model.{flops,bytes}` (scalar or map form) parse under the shared evaluator and every formula evaluates against the task's `shapes.yaml` knob universe (§5.1). (The dataset *loader* stays lenient — a missing block yields `TaskDef.cost_model = None`; only the validator enforces presence. See §5.1.)
- `kernel.py` has at least one `@triton.jit` and exports the resolved upstream entrypoint (`meta.yaml.oracle.gpu_triton`)
- `oracle_pytorch.py` exports `run`
- `contract.md` references match `meta.yaml`
- Hash of `kernel.py` matches `meta.yaml.upstream.function_hash`
