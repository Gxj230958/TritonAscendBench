# Task API Contract

> Authoritative scoring model: `docs/DESIGN.md §9` (two-objective — correctness gate + 910B Roofline-normalized performance). GPU-Triton → 910B NPU-Triton migration; no AscendC baseline.


> v0 — derived from `docs/benchmark_impl_plan.md` v4.13 §Phase 2.5 + R11-P1-1 + R14-P1-2
> Source of truth: per-task `meta.yaml`. Human-readable docs (this file + `contract.md`) explain but do not override `meta.yaml`.

## 1. Task Directory Layout

Every task is a directory under `tasks/level{1,2,3}/<task_id>/`:

```
tasks/level1/L1_12_rms_norm/
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
task_id: "L1_12_rms_norm"
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
`shapes.yaml.inputs` (declaration order, skipping the non-tensor forms
`scalar` / `derive` / `scalar_seq` / `scalar_knob` which have no element
size) — e.g. a `float32` primary operand gives `dtype_size = 4`, `int64`
gives `8`. A scalar-only `inputs` block has no tensor element size, so the
fp16 default (`2.0`) is used. Per-input precise handling for genuinely
mixed-dtype tasks is a W3.1 long-tail item; until then this single
representative value feeds every `flops` / `bytes` formula. The validator
evaluates formulas with the fp16 default `dtype_size` — it only confirms a
formula *parses and evaluates*, not its numeric value. `cost_model.py` defines
exactly which derived symbols exist and is reused verbatim by `dataset.py`,
`validate_tasks.py` and the W4.2 GPU cross-check, so the symbol set never
drifts. The arithmetic safety core itself lives in
`triton_ascend_bench/expr.py` (WA.2): `cost_model` formula evaluation is the
`eval_numeric` entrypoint of that shared AST-whitelist evaluator. Allowed in
numeric mode: `+ - * / // ** %`, unary `+` / `-`, numeric literals,
knob/derived names, and `min` / `max` / `ceil` / `floor` / `sqrt` calls.
Anything else (attribute access, subscripts, comprehensions, unknown names) is
rejected — formulas are evaluated through an AST whitelist, never `eval()`.

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

### 5.2 `param_constraints` schema (operator input contract)

`meta.yaml.param_constraints` declares the operator's **full input contract**
in machine-checkable form — tensor dimensions, dtypes, value distributions,
non-tensor literals, and cross-tensor invariants. It exists so a benchmark
task can no longer ship with mechanical placeholder parameters that either
fail to run or, worse, evaluate the oracle and the candidate at a meaningless
parameter point where both spuriously "pass" (`docs/plans/2026-05-21-operator-param-validity-impl.md`).

`param_constraints` has **six subsections**:

| subsection | object | form |
|-----------|--------|------|
| `dims` | size-knob integers | list of **predicate** expression strings |
| `dtypes` | each tensor `inputs.*` | `{input_name: dtype_enum}` |
| `dists` | each tensor `inputs.*` | `{input_name: {kind, args}}` |
| `scalars` | non-tensor literals (`scalar` / `scalar_seq`) | `{name: predicate-or-derive}` |
| `tensor_invariants` | materialised tensors | list of `{type, args}` |
| `scaffold` | the block itself | bool — `true` = un-reconciled draft |

#### `dims` — knob constraint predicates

A list of predicate expression strings over the `shapes.yaml` size knobs.
Each string is a boolean expression evaluated against every public / hidden
knob row. Examples:

```yaml
dims:
  - "D in [16, 32, 64, 128]"   # closed knob set
  - "H % G == 0"               # divisibility (constexpr-derived)
  - "is_pow2(D)"               # power-of-two head dim
  - "L >= 16"                  # lower bound (BLOCK_SIZE tile floor)
```

These cover the constraints a kernel's `tl.constexpr` / `BLOCK_SIZE` impose
on the knob universe. The evaluator (the WA.2 shared `eval_predicate`
entrypoint) supports comparisons (`== != < <= > >=`), boolean operators
(`and or not`), `in`, list / tuple literals, and the predicate helpers
`is_pow2` / `gcd` / `sqrt` / `len` / `sum`. WA.1 only validates the *shape*
of this subsection (that it is a list of strings); semantic evaluation is
WA.4.

#### `dtypes` — per-input dtype constraint

A `{input_name: dtype}` map; one entry per tensor input declared in
`shapes.yaml.inputs`. The value is a **closed enum** — the keys of the
`build_inputs` dtype alias table (`triton_ascend_bench/inputs.py`), which is
the **single authority** for the legal dtype set:

```
float32  float16  bfloat16  bf16  fp16  fp32
int8  int32  int64  long  bool  float8_e4m3fn
```

`validate_tasks` and `build_inputs` share this one table, so an unknown
dtype string (a typo) is statically rejected rather than silently falling
back to `float32`. Example:

```yaml
dtypes:
  x: float32
  a: int8        # W8A8 quantised activation
```

#### `dists` — per-input distribution constraint

A `{input_name: {kind, args}}` map declaring the distribution each tensor
input must be materialised with. `kind` is a value from the **dist-kind
table** below; `args` is a (possibly empty) map of structured arguments.
This statically rejects a *wrong-but-supported* distribution — e.g. an
input that must be a row-stochastic `softmax_rows` matrix declared as plain
`randn`.

```yaml
dists:
  q:          {kind: randn}
  probs:      {kind: softmax_rows}
  seqused_k:  {kind: randint, args: {lo: 0, hi: "L"}}
```

**Representation bridge.** `shapes.yaml.inputs.*.dist` keeps its existing
string DSL form (`randint:0:E`, `offsets:to=B*L`, `full:expr`); the
`param_constraints.dists` block uses the structured `{kind, args}` form.
The two are reconciled through a single shared `parse_dist_spec` normaliser
(`triton_ascend_bench/inputs.py`, WA.3 — string DSL ⇄ `{kind, args}`) so
there is no duplicated parsing or drift: `build_inputs` materialisation, the
`validate_tasks` dist check (WA.4) and the generators all route through it.
The string DSL stays the authoritative `shapes.yaml` authoring form and is
not migrated.

**dist-kind table** (the closed `kind` enum, bound to the `build_inputs`
distribution vocabulary):

| kind | meaning | `args` |
|------|---------|--------|
| `randn` | standard normal | — |
| `rand` | uniform `[0, 1)` | — |
| `ones` | constant 1 | — |
| `zeros` | constant 0 | — |
| `eye` | identity matrix | — |
| `tril` | well-conditioned lower-triangular | — |
| `randint` | uniform integers | `lo`, `hi` (knob exprs) |
| `softmax_rows` | row-stochastic (softmax over last dim) | — |
| `l2norm_rows` | randn rows L2-normalised over the last dim (unit-norm keys; the stable delta-rule key recipe) | — |
| `logsigmoid` | `logsigmoid(rand)` — strictly negative gate so `exp(g)` is a genuine decay in `(0, 1)` | — |
| `offsets` | monotonic **leading-zero** indptr int64 vector (`[0] == 0`, total in `[-1]`) | `step` *or* `to` (knob expr) |
| `cumcount` | monotonic **non-leading-zero** cumulative count int64 vector (element `r` = running count through segment `r`, total in `[-1]`) | `to` (knob expr) |
| `arange` | `0 .. n-1` int64 | — |
| `cyclic_perm` | single-cycle permutation `perm[i] = (i+1) % n` — a guaranteed non-identity bijection of `[0, n)` (every position remapped for `n >= 2`); honors the declared `int32`/`int64` dtype | — |
| `full` | constant-filled int64 | `value` (knob expr) |

#### `scalars` — non-tensor literal constraint

Constraints on the **non-tensor literal** inputs — `shapes.yaml.inputs`
entries declared with a `scalar` key (a single JSON scalar) or a
`scalar_seq` key (a list literal). Each entry is either:

- a **value-domain predicate** — a Python-compatible boolean expression
  (chained comparisons allowed) over the literal's name **and the
  `shapes.yaml` size knobs**, e.g. `"0 < eps <= 1e-2"` or the
  knob-referencing `"sum(mrope_section) == D // 2"`. (Python-parseable form
  only — *not* the mathematical interval notation `(0, 1e-2]`.) The
  predicate is evaluated against a symbol table that merges the shape row's
  knobs with the literal's own name, mirroring the `dims` predicate
  evaluation; the literal name wins on a knob/name clash.
- a **derive rule** — see WA.3 (`{derive: "1/sqrt(D)"}`). A derive rule is
  reconciled against `shapes.yaml.inputs[name]`: that input must itself be a
  `derive` form and carry a structurally equivalent formula. A task may not
  declare `scale: {derive: "1/sqrt(D)"}` while `shapes.yaml` pins `scale` to
  a fixed `{scalar: 0.125}` — the validator rejects the mismatch.

`scalar_seq` predicates may use `len()`, `sum()` and **constant-integer
indexing** (`block_size[0] == 8`, `mrope_section[-1] == 2`), so a list
literal is itself brought into the machine-checkable contract. Indexing
accepts only a constant integer literal (negative indices allowed); a slice
or a non-constant index is rejected, and an out-of-range index is a
validation error.

```yaml
scalars:
  eps:           "0 < eps <= 1e-2"
  mrope_section: "sum(mrope_section) == D // 2"   # scalar_seq, knob-referencing
  block_size:    "len(block_size) == 2 and block_size[0] == 8"  # scalar_seq, indexed
```

**`shapes.yaml.inputs` non-tensor forms (WA.3).** The materialisation layer
(`triton_ascend_bench/inputs.py`) declares a non-tensor input with exactly
one of these keys; the value type is **schema-fixed**:

| key | meaning | value |
|-----|---------|-------|
| `scalar` | a literal constant | a single JSON scalar (`bool`/`int`/`float`/`str`/`None`) |
| `derive` | a per-shape-row derived scalar | a numeric knob expression string, e.g. `"1/sqrt(D)"`; `build_inputs` evaluates it through `eval_numeric` against the shape row's knobs |
| `scalar_seq` | a literal list | a list of JSON scalars (e.g. `L2_10` `block_size`, `L1_31` `mrope_section`) |
| `scalar_knob` | a derived integer | an integer-valued knob expression string (legacy form) |

A `scalar` / `derive` value **must** be a JSON scalar — it crosses the W1.0
sidecar IPC type boundary, which carries no pickle. A list literal therefore
*may not* be authored under `scalar`; it uses the dedicated `scalar_seq`
key, which the IPC sidecar carries as a JSON array. `build_inputs` rejects a
non-JSON-scalar `scalar` / `derive` value with a `TypeError`.

#### `tensor_invariants` — materialised-tensor invariants

A list of `{type, args}` entries, each an **enumerated** invariant checked
against the materialised (post-`build_inputs`) tensors. `type` is a value
from the closed set below; `args` carries the per-type parameters. These
express contracts a knob expression cannot — e.g. `cu_seqlens` must be
monotonic, `len(seqused_k)` must equal a knob.

**Enumerated invariant type table:**

| `type` | meaning | `args` |
|--------|---------|--------|
| `monotonic_nondecreasing` | values are non-decreasing along the last axis | `input` |
| `first_eq` | first element equals a value | `input`, `value` (knob expr) |
| `last_eq` | last element equals a value | `input`, `value` (knob expr) |
| `length_eq` | `len(input)` equals a knob expression | `input`, `value` (knob expr) |
| `bounded_above_by` | every element `<=` a bound | `input`, `value` (knob expr) |
| `index_in_range` | every element is a valid index into `[0, value)` | `input`, `value` (knob expr) |

```yaml
tensor_invariants:
  - {type: monotonic_nondecreasing, args: {input: cu_seqlens}}
  - {type: first_eq,  args: {input: cu_seqlens, value: 0}}
  - {type: length_eq, args: {input: seqused_k, value: "E"}}
```

#### `scaffold` — un-reconciled-draft sentinel

An explicit boolean field. `scaffold: true` marks a `param_constraints`
block that was emitted by the `phase_c_docs` scaffold generator and **not
yet manually reconciled** against the operator's real contract. It is the
**single shared sentinel** consulted by the generator, the validator, and
the eval gate (WA.4 / WA.5) — an explicit schema field replacing the older
implicit text markers, so no component drifts. `scaffold` does **not**
participate in score-eligibility (the score-eligible denominator looks only
at stub / `PHASE-C-MANUAL`) — a `scaffold` task is `blocked` and stays in
the fixed denominator, never dropped from it.

#### `param_constraints` as a `task_status` gate (WA.4)

Beyond the validator, `param_constraints` is an **eval gate**: a task whose
block is missing / `scaffold: true` / schema-illegal / semantically violated
is `task_status = "blocked"`. The verdict is decided by the single
repo-authoritative arbiter `cli analyze`'s `_authoritative_task_status`
(mirrored by `eval.evaluate`, both via the shared
`param_constraints.resolve_task_status`) under a fixed priority:

1. `is_canary: true` ⇒ `evaluated` (exempt from the `param_constraints` gate);
2. self-reported-mode `upstream_oracle_required` ⇒ `unsupported` (W4.3 — the
   `param_constraints` gate does **not** pull it back into the denominator);
3. verified-mode incomplete offline `golden/` ⇒ `blocked` (W4.3 — priority
   over the `param_constraints` gate);
4. `param_constraints` fails the shared helper ⇒ `blocked` — the task stays
   in the **fixed** score-eligible denominator and is scored as ε (it is
   **not** dropped from the set; dropping it would be a subset-escape);
5. otherwise ⇒ `evaluated`.

A `blocked` `param_constraints` task therefore behaves exactly like a
Roofline-data-incomplete one: it stays in `|S|` and counts as a failure. It
does **not** change the score-eligible set itself (still `stub` /
`PHASE-C-MANUAL` only).

#### `param_constraints` presence — layered (loader lenient, validator strict)

Like `cost_model` (§5.1), the two consumers treat `param_constraints`
differently *by design* — but the loader is **more** lenient here:

- **Dataset loader (`dataset.py`)** — *fully lenient*. `TaskDef.param_constraints`
  is a `dict | None` field defaulting to `None`. A missing block yields
  `None`. A **malformed** block (illegal schema, bad constraint expression)
  is kept **verbatim as the raw dict** — the loader does **not** validate
  or raise. This deliberately differs from `cost_model`, which eager-parses
  and raises on a malformed block: a malformed `param_constraints` must be
  allowed to load so it can reach the `task_status = blocked` gate (WA.4)
  rather than crash inside `load_task`. The loader stores the raw dict; it
  never builds a structured object.
- **Task validator (`scripts/validate_tasks.py`)** — *strict*.
  `param_constraints` is a **required** top-level `meta.yaml` key; a missing
  block is a hard validation error. The WA.1 validator layer checks only
  the **schema shape** (the six subsections exist and have the right
  container types); the five-subsection *semantic* checks are WA.4. The
  red→green transition for the 100-task corpus is carried by the validator
  (WB.1 backfills every task), never by the loader.

**`is_canary` exemption.** A task with `is_canary: true` is **not** required
to carry `param_constraints` — a canary reuses its source task's solution,
is a deliberately-crafted trap, and is never score-eligible. `validate_tasks`
(and the WA.4 eval gate) skip the `param_constraints` requirement for any
`is_canary` task so the canary detection chain is not sealed shut.

In short: a missing or malformed `param_constraints` is *not an error to
load* but *is an error to release*.

### 5d. `provenance.golden` schema (baseline trust chain — V9 plan M1.2)

The `provenance:` top-level block carries the *content* metadata
historically (`added_in`, `authors`, `last_modified`). V9 adds a
nested `golden:` sub-block that records the *baseline trust chain* —
HOW a task's reference values were produced, verified, and signed.
Top-level keys are untouched; the nested block is purely additive.

```yaml
provenance:
  added_in: v0.1                   # existing top-level metadata, untouched
  authors: [...]                   # existing
  last_modified: '2026-05-26'      # existing
  golden:                          # nested baseline trust chain (V9 M1.2)
    source: upstream_author_test | gpu_kernel_rerun
    tolerance_source: upstream_test_explicit | upstream_test_default | migration_drift_budget
    atol: 1.0e-5                   # pinned from upstream test
    rtol: 1.0e-5
    ref_path: tasks/<id>/upstream_ref.py     # source ① only — vendored author test ref
    test_file_path: <relative to infer/{repo}>  # source ① only
    test_capture_sha: <40 hex>     # source ① only — author test git SHA
    golden_sha256: <64 hex>        # SHA of the captured ref artifact (R5)
    oracle_kernel_equivalence: VERIFIED | FAILED | MIXED | SKIPPED | NEEDS_MANUAL | LAUNCH_BLOCKED | WRAPPER_CONTRACT_MISMATCH | ORACLE_KERNEL_LAYOUT_MISMATCH | INPUT_DRIFT
    gpu_kernel_capture_status: PENDING | DONE | FAILED | BLOCKED_NON_DETERMINISTIC
    gpu_kernel_deterministic: true   # R20 N=3 bit-identity check
    upstream_closure_fingerprint: <64 hex>  # cache invalidation key
    reviewers: [<name1>, <name2>]    # R8 reviewer attestation
    reconstructability_risk: low | medium | high  # M0a.1 6th-dim risk score
```

**Closed enums** (validator-strict; see `triton_ascend_bench/provenance.py`):

| Field | Allowed values |
|-------|----------------|
| `source` | `upstream_author_test`, `gpu_kernel_rerun` |
| `tolerance_source` | `upstream_test_explicit`, `upstream_test_default`, `migration_drift_budget` |
| `oracle_kernel_equivalence` | `VERIFIED`, `FAILED`, `MIXED`, `SKIPPED`, `NEEDS_MANUAL`, `LAUNCH_BLOCKED`, `WRAPPER_CONTRACT_MISMATCH`, `ORACLE_KERNEL_LAYOUT_MISMATCH`, `INPUT_DRIFT` |
| `gpu_kernel_capture_status` | `PENDING`, `DONE`, `FAILED`, `BLOCKED_NON_DETERMINISTIC` |
| `reconstructability_risk` | `low`, `medium`, `high` |

**Hash field constraints** (length + hex):

| Field | Required hex length |
|-------|--------------------|
| `golden_sha256` | 64 |
| `upstream_closure_fingerprint` | 64 |
| `test_capture_sha` | 40 |

**Open / operational extras** (preserved verbatim by the loader, NOT
closed-enum checked — these are pipeline artifacts written by
`scripts/apply_oracle_equivalence_verdict.py` and similar):
`verified_at`, `tolerance_class_used`, `equivalence_per_shape`,
`golden_source`.

**Grace period**: the validator accepts an absent `golden:` block on
any task (legacy meta + tasks that have not yet run R10/R5 capture).
This is documented as a *warning concern* in the release-gate report
but not a release-blocking error. Once the block is present, every
closed enum and hash-length value MUST conform — a malformed value
IS release-blocking, because the release gate, eval boundary, and
GoldenStore all read these fields directly.

**Loader contract**: `triton_ascend_bench.dataset.TaskDef` carries a
typed view of the nested block via `provenance_golden:
ProvenanceGoldenSpec | None` (loader-lenient, same pattern as
`cost_model` / `param_constraints`); the raw YAML structure stays
reachable through `TaskDef.meta['provenance']` for forward
compatibility.

## 6. Authority + Cross-File Consistency

- `meta.yaml` is **source of truth** for all machine-readable fields.
- `contract.md` is **human-readable only** — must not introduce facts not in `meta.yaml`.
- `shapes.yaml` carries both the public and the hidden eval shapes in one file: a `public_shapes` list and a sealed `hidden_shapes` list (`docs/DESIGN.md §3.3 / §5.2`). There is **no** separate `shapes_hidden.yaml` overlay — `shapes.yaml.hidden_shapes` is the single source of truth for hidden shapes. In `verified` mode the trusted runner materialises the `hidden_shapes` rows' inputs and grades them; in `self-reported` mode the hidden shapes are not materialised. The candidate sandbox never receives the `shapes.yaml` document at all (it gets only runner-materialised inputs — `specs/process_isolation.md`), so the `hidden_shapes` section is structurally absent from the candidate view.
- If `meta.yaml.cost_model` is absent, or its analytic FLOPs/bytes mismatch the instrumented GPU-Triton reference beyond `gpu_xcheck_tol` → `cost_model.status = BLOCKED`. This is **ncu-evidenced performance metadata, NOT a release-gate blocker**: cost_model is an analytic access model, and conditional access / L2 reuse / tile geometry / HBM burst floor make a portable formula structurally unable to hit per-shape tol everywhere. The release gate keys on the two real correctness proofs — C2 oracle-equivalence + S2 severance (`docs/DESIGN.md §4.4`); `asymptotic_hbm_xcheck` gates cost_model PASS on large/asymptotic shapes and records small shapes as `FLOOR_OBSERVED`.
- If `meta.yaml.oracle_simplified = "upstream_oracle_required"` and `oracle_pytorch.py` body contains `F.silu`, `F.gelu`, etc. high-level ops → static check rejects.

## 6a. Task State Fields (bench-team-internal writes)

The following `meta.yaml` fields are **optional, bench-team-internal state**
written by the automation orchestrator (`scripts/automation/`) and the phase
runners (`scripts/phase_a_prime.py`, `scripts/phase_b_golden_capture.py`).
They drive the three-verdict pipeline (`shapes-incomplete` / `oracle-mismatch`
/ unfixable) and the release-scope eligibility gate. All four field groups
below are **optional with explicit defaults**: a task `meta.yaml` that
predates this section (the 100-task corpus authored before M2) loads and
validates exactly as before — the defaults are *implicit absent*, never an
error. The fields are written by automation, never hand-authored.

The fields are **not** added to `REQUIRED_META_TOP_LEVEL` (precedent: the
existing `phase_a_prime` block has always been optional). Schema-shape
checks for these fields, when they are present, live in the patch validator
(`scripts/automation/lib/file_blocks.py` Layer 5) — that layer rejects a
malformed write before it lands on disk. `scripts/validate_tasks.py`
ignores absent / well-formed blocks.

### 6a.1 `phase_a_prime.skip_invocation`

| key | value |
|-----|-------|
| **type** | `bool` |
| **required** | optional |
| **default** | `false` (treated as absent) |
| **written by** | orchestrator (`scripts/automation/fix_orchestrator.py`) on a `defer-to-golden` decision under the `oracle-mismatch` verdict |
| **read by** | `scripts/phase_a_prime.py` `s2_verify_task` early-return; eligibility gate (`triton_ascend_bench/eligibility.py`) |

**Semantics.** When `true`, the phase A' GPU S2 verify step **does not**
invoke the operator in an isolated subprocess. Instead it records
`phase_a_prime.status = SKIPPED_BY_DESIGN` and treats the offline golden
capture (phase B, see §6a.3) as the sole truth source. This handles the
`upstream-not-invocable-here` case — an operator whose upstream entrypoint
cannot be called standalone on the GPU verify host (e.g. requires a
distributed launcher, a custom CUDA build, or a private dependency).

`skip_invocation: true` **must** be accompanied by
`phase_a_prime.status = SKIPPED_BY_DESIGN` and a non-empty
`phase_a_prime.reason`. A task with `skip_invocation: true` AND
`status: PASS` is a schema violation (cross-field invariant, §6a.5).

### 6a.2 `phase_a_prime.status` — `SKIPPED_BY_DESIGN` enum extension

| existing values | `PASS` / `BLOCKED` / `UNRUN` |
|-----------------|------------------------------|
| **new value** | `SKIPPED_BY_DESIGN` |
| **written by** | `phase_a_prime.update_meta_phase_a_prime` (existing writer) |
| **read by** | eligibility gate; `scripts/validate_tasks.py` golden-presence check |

**Semantics.** `SKIPPED_BY_DESIGN` is the truth-source delegation marker:
phase A' deliberately did not run isolated invocation; the offline phase B
golden (§6a.3) is authoritative. The eligibility gate treats this status
the same as `PASS` *only if* the corresponding `phase_b_capture.status`
records a successful capture (`DONE`).

When `phase_a_prime.status = SKIPPED_BY_DESIGN`, `phase_a_prime.reason`
(free-form string) is **required** and must explain why the operator
cannot be invoked standalone.

### 6a.3 `phase_b_capture` block

The audit-trail record of the offline phase B golden capture for a single
task.

| key | type | required | default | semantics |
|-----|------|----------|---------|-----------|
| `phase_b_capture.status` | string enum `PENDING` / `DONE` / `FAILED` | optional | absent (= no capture attempted) | capture state machine |
| `phase_b_capture.upstream_callable.module` | string, canonical dotted Python path | optional | absent | which upstream module exports the entrypoint to capture |
| `phase_b_capture.upstream_callable.function` | string, valid Python identifier | optional | absent | which function under that module to call |
| `phase_b_capture.reason` | string | optional; **required when** `status` is `FAILED` | absent | free-form failure / skip explanation |
| `phase_b_capture.captured_at` | ISO 8601 UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`) | optional; **required when** `status` is `DONE` | absent | when the capture succeeded |

**Writers.**

- Orchestrator (`scripts/automation/fix_orchestrator.py`) writes the
  initial `{status: PENDING, upstream_callable: {module, function},
  reason: …}` on a `defer-to-golden` patch. `upstream_callable.{module,
  function}` are sourced from the dual-anchor resolver evidence.
- `scripts/phase_b_golden_capture.py` updates `status` to `DONE` (sets
  `captured_at`) on a successful capture child or `FAILED` (sets `reason`)
  on a capture failure. The `upstream_callable` subblock is not rewritten
  by the capture script — it is the orchestrator's authored intent.

**Readers.** Eligibility gate; `scripts/validate_tasks.py` golden-
directory presence check (a `DONE` capture implies a populated `golden/`
directory).

**`upstream_callable.module` format.** A canonical dotted Python module
path: a non-empty sequence of valid Python identifiers separated by
single dots, with no leading / trailing dot and no embedded whitespace
or slashes. Examples — legal: `vllm.model_executor.layers.layernorm`,
`sgl_kernel.activation`; illegal: `vllm/layernorm`, `vllm..layernorm`,
`3vllm.layernorm`, `vllm. layernorm`. Canonical-path enforcement is
performed by the Layer 5 patch validator
(`scripts/automation/lib/file_blocks.py`).

**`upstream_callable.function` format.** A single valid Python
identifier (no dots, no whitespace, must satisfy `str.isidentifier()`).
A multi-segment dotted name belongs in `module`, not `function`.

### 6a.4 `phase_b_capture_required`

| key | value |
|-----|-------|
| **type** | `bool` |
| **required** | optional |
| **default** | `false` (treated as absent) |
| **written by** | orchestrator on a `defer-to-golden` patch under either the `oracle-mismatch` or `upstream-not-invocable-here` decision |
| **read by** | eligibility gate (`triton_ascend_bench/eligibility.py::release_status_pass`) |

**Semantics.** When `true`, the task's release eligibility requires
`phase_b_capture.status == DONE` (i.e. the golden has been captured and
the operator's authoritative oracle is on disk). When `false` (default),
the eligibility predicate does not consult `phase_b_capture` at all.

This boolean is the explicit promotion of the offline golden into the
release-gate conjunctive predicate — without it, an
`oracle-mismatch` task could spuriously be `release_status = PASS` based
solely on the phase A' result while its golden is still missing.

### 6a.5 `release_scope` block

The explicit exclusion record for a task that the orchestrator has marked
unfixable by the LLM patch flow and removed from the release scope (the
`--mark-unfixable` flow, see `scripts/automation/fix_orchestrator.py::apply_release_scope_exclusion`).

| key | type | required | default | semantics |
|-----|------|----------|---------|-----------|
| `release_scope.excluded` | `bool` | optional | absent (= `false`) | sentinel — task is excluded from release scoring |
| `release_scope.reason` | string | optional; **required when** `excluded` is `true` | absent | free-form human-readable exclusion reason |
| `release_scope.excluded_at` | ISO 8601 UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`) | optional; **required when** `excluded` is `true` | absent | when the exclusion was recorded |
| `release_scope.excluded_by` | string identifying the actor (e.g. `fix_orchestrator/automation`, operator handle) | optional; **required when** `excluded` is `true` | absent | who recorded the exclusion |

The shipped writer constructs the four-key block atomically; partial
writes (e.g. `excluded: true` with no `reason`) are not produced by the
helper and are rejected by the Layer 5 patch validator. A task whose
`release_scope.excluded` is `true` is dropped from the eligibility
denominator by `is_score_eligible` (early-out) so it does not weigh on
the release pass-rate.

### 6a.6 Cross-field invariants

These invariants tie the four field groups together. They are enforced
at *write* time by the patch validator (Layer 5) and at *gate* time by
the eligibility predicate. The validator (`scripts/validate_tasks.py`)
does not re-derive them — a malformed cross-field combination cannot
land in `meta.yaml` because the patch is rejected at Layer 5 before
disk write.

1. `phase_a_prime.skip_invocation == true` **⇒**
   `phase_a_prime.status == "SKIPPED_BY_DESIGN"` **and**
   `phase_a_prime.reason` is a non-empty string.
2. `phase_a_prime.status == "SKIPPED_BY_DESIGN"` **⇒**
   `phase_a_prime.skip_invocation == true` **and**
   `phase_a_prime.reason` is a non-empty string.
3. `phase_b_capture.status == "DONE"` **⇒**
   `phase_b_capture.upstream_callable.module` is a canonical dotted
   Python path **and** `phase_b_capture.upstream_callable.function` is
   a valid Python identifier **and** `phase_b_capture.captured_at` is an
   ISO 8601 UTC timestamp.
4. `phase_b_capture.status == "FAILED"` **⇒**
   `phase_b_capture.reason` is a non-empty string.
5. `phase_b_capture_required == true` **and**
   `phase_b_capture.status != "DONE"` **⇒** the task is
   `release_status = BLOCKED` (not `PASS`); the eligibility gate enforces
   this.
6. `release_scope.excluded == true` **⇒**
   `release_scope.reason` is a non-empty string **and**
   `release_scope.excluded_at` is an ISO 8601 UTC timestamp **and**
   `release_scope.excluded_by` is a non-empty string. The task is
   excluded from the eligibility denominator (early-out in
   `is_score_eligible`).

**ISO 8601 timestamp format.** All `*_at` timestamps in this section are
written by `scripts/automation/state_machine.py::iso_utc_now()` in
exactly the form `YYYY-MM-DDTHH:MM:SSZ` (UTC, second resolution,
trailing `Z`). The `±HHMM` numeric-offset form is **not** produced by
the bench — readers should accept either form per the ISO 8601 standard,
but the bench writes only the `Z` form.

## 7. Validator

`python scripts/validate_tasks.py` (per-task schema / hash validation; `--release-gate` runs the S6 gate) checks:
- All 5 files present (`kernel.py` / `oracle_pytorch.py` / `meta.yaml` / `shapes.yaml` / `contract.md`)
- `meta.yaml` matches schema; the top-level `cost_model` key is **required** (R1-4) — a missing block is a hard error. `cost_model.{flops,bytes}` (scalar or map form) parse under the shared evaluator and every formula evaluates against the task's `shapes.yaml` knob universe (§5.1). (The dataset *loader* stays lenient — a missing block yields `TaskDef.cost_model = None`; only the validator enforces presence. See §5.1.)
- the top-level `param_constraints` key is **required** (WA.1) — a missing block is a hard error, except for `is_canary` tasks which are exempt. Two validator layers run: the **schema-shape** layer (WA.1 — the six subsections `dims` / `dtypes` / `dists` / `scalars` / `tensor_invariants` / `scaffold` exist with the right container types) and the **five-subsection semantic** layer (WA.4 — `dims` predicates evaluate true against every knob row, `dtypes` / `dists` agree with `shapes.yaml.inputs`, `scalars` value-domain predicates hold, `tensor_invariants` hold on the materialised sample tensors). A `scaffold: true` block short-circuits to a single `scaffold` failure (no semantic check runs). Both layers route through the shared `triton_ascend_bench.param_constraints` helper so the validator, `eval`, and the `cli analyze` status resolver cannot drift. (The dataset *loader* stays fully lenient — a missing block yields `TaskDef.param_constraints = None`, a malformed block is kept as a raw dict without raising. See §5.2.)
- `kernel.py` has at least one `@triton.jit` and exports the resolved upstream entrypoint (`meta.yaml.oracle.gpu_triton`)
- `oracle_pytorch.py` exports `run`
- `contract.md` references match `meta.yaml`
- Hash of `kernel.py` matches `meta.yaml.upstream.function_hash`

The bench-team-internal state fields defined in §6a
(`phase_a_prime.skip_invocation`, `phase_a_prime.status` (incl.
`SKIPPED_BY_DESIGN`), `phase_b_capture`, `phase_b_capture_required`,
`release_scope`) are **not** added to `REQUIRED_META_TOP_LEVEL`. They are
optional with explicit defaults. Schema-shape and cross-field invariant
checks for these fields live in the patch validator
(`scripts/automation/lib/file_blocks.py` Layer 5), which gates writes
before they reach disk; the corpus validator here ignores absent or
already-well-formed blocks.
