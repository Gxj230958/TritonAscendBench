# TritonAscendBench — Architecture

How TAB is built and how an evaluation flows: the "how it works" companion to
the [README](../README.md).

TAB is a **migration benchmark**. Each task takes a real production GPU Triton
kernel (from vLLM / SGLang / flash-linear-attention) and asks: can it be ported
to a Huawei Ascend 910B NPU via `triton-ascend` without losing correctness or
performance? The corpus is **101 tasks** — 36 at L1 (single-op / elementwise /
reduce / quant), 35 at L2 (fused / matmul / MoE / state-space step), 30 at L3
(full Flash Attention / fused MoE / chunked pipelines).

Two facts shape the whole design:

1. **Correctness is the live, scored axis.** A submitted kernel is graded
   per-shape against a trusted PyTorch oracle with a tolerance policy. This path
   runs with **no GPU and no NPU** — it is the part anyone can reproduce locally.
2. **Two machines, two roles.** The submitter and the offline correctness grader
   run on commodity hardware. *Verified* performance is measured on the 910B NPU;
   the GPU is used by the provider only for an offline oracle-equivalence
   cross-check, never as the migration target.

---

## 1. Repository layout

```
triton-ascend-bench/
├── tasks/                  # the corpus — level1/ level2/ level3/, one dir per task
├── ascendbench/            # the Python package (harness, grader, scoring, loaders)
├── scripts/                # provider-side corpus-construction pipeline
├── docker/                 # Dockerfile.gpu (oracle cross-check) / Dockerfile.npu (910B)
└── docs/                   # this file, reference specs, kernel-select rationale
```

### 1.1 A task directory: the five-file anatomy

Every `tasks/level{1,2,3}/<task_id>/` is exactly five graded files. The schema
validator (`scripts/corpus/validate_tasks.py`, `REQUIRED_FILES`) checks the set
is present; the loader (`ascendbench/corpus/dataset.py::load_task`) parses
`meta.yaml` / `shapes.yaml` / `contract.md` / `kernel.py` into a `TaskDef`
(`oracle_pytorch.py` is loaded separately, runner-side, never handed to the
candidate).

| File | Role |
|------|------|
| `kernel.py` | The **frozen upstream GPU Triton reference** — the migration *source*. A vendored `@triton.jit` kernel with original Apache-2.0/BSD attribution preserved. SHA-pinned (`TaskDef.kernel_py_sha256`). It is the porting target a submitter reads; it is **not** executed during candidate grading. |
| `meta.yaml` | All task metadata: `task_id` / `level`, `upstream` provenance (repo, `commit_sha`, `file_sha256`, license), the `interface` alias map and mutability contract, `tolerance_class`, the `oracle_simplified` tier, `cost_model`, `param_constraints` (the operator input contract), `runner_checks` (static checks, metamorphic checks, shape coverage), and the baseline trust-chain records (`provenance.golden`, `phase_a_prime`). |
| `shapes.yaml` | `size_knobs` defaults, `public_shapes`, `hidden_shapes`, and an `inputs:` block. Each input is a knob-parameterised tensor spec (`dims` / `dtype` / `dist`) or a non-tensor scalar. One inputs block covers every shape row. Consumed by `ascendbench.input_spec.inputs.build_inputs`. |
| `contract.md` | The human-readable contract: the `solution.py` signature to implement, prohibited shortcuts, an inputs table, required semantics, the tolerance class, and the metamorphic invariants. |
| `oracle_pytorch.py` | The **PyTorch ground truth**: `run(inputs: dict) -> dict[str, Tensor]`. Device-independent (runs on CPU); depends on neither triton, CUDA, nor NPU. |

For example, `tasks/level1/L1_12_rms_norm/` ports vLLM's `_rms_norm_kernel`: its
`oracle_pytorch.py::run` computes `x / sqrt(mean(x²)+eps) * weight` in float32,
`shapes.yaml` declares 5 public + 4 hidden shape rows over knob `D`
(hidden_size), and `meta.yaml` pins `tolerance_class: vector_reduce` (atol/rtol
1e-3) plus two metamorphic checks (weight-identity, row-independence).

Canary trap tasks live under `tasks/_canary/`. The loader's `iter_all_tasks()`
skips every `_`-prefixed directory by design, so canaries can never enter the
score-eligible denominator; they are surfaced only by the dedicated
`iter_canary_tasks()` enumerator.

### 1.2 The `ascendbench/` package

| Subpackage | What it holds |
|------------|---------------|
| `corpus/` | `dataset.py` — `TaskDef`, `load_task`, `iter_all_tasks`, canary discovery; `provenance.py` — typed view of the baseline trust chain. |
| `input_spec/` | `inputs.py` — `build_inputs` (materialises a shape's inputs from `shapes.yaml`), `split_inputs`/`merge_inputs` for the IPC; `expr.py` — `eval_numeric`, an AST-whitelisted knob-formula evaluator (never bare `eval`); `param_constraints.py` — the operator input contract + `resolve_task_status`. |
| `integrity/` | `process_isolation.py` — the isolated candidate runner; `static_checker.py` — the AST allow/deny pass + the runtime audit hook; `adversarial.py` — canary and metamorphic checks; `wrapper_contract.py`. |
| `correctness/` | `tolerance.py` — `load_policy` + `compare`; `golden_store.py`; `determinism.py`. |
| `engine/` | `eval.py` — `evaluate()`, the per-task orchestrator; `cli.py` — the `tab` command-line interface. |
| `perf/` | `scoring.py` — `capped_geomean_native`, `compute_t_ref`, the Roofline efficiency math; `cost_model.py`; `peak_table.py`; `timing.py`. |
| `backends/` | `get_backend(...)` → `GpuBackend` / `NpuBackend`: device init, event timer, kernel-event profiler, compile-warmup policy. |
| `release_gate/` | `eligibility.py` — `compute_release_ready_eligible`, `is_provenance_eligible`, `release_status_pass`; `sanity_gate.py`. |
| `frozen/` | Vendored upstream `snapshots/`, plus `deps/` and `shim/` used to make a frozen `kernel.py` standalone. |
| `policies/` | `tolerance_policy.yaml` (tolerance classes) and `version_matrix.toml` (pinned numerics stacks). |

Plus `manifest.py` (`SolutionManifest`), `triton_compat.py`, and helper
modules in `trusted_shim/`, `upstream_shims/`, `_util/`.

### 1.3 `docker/`

- `Dockerfile.gpu` — the offline oracle-equivalence cross-check host (H100/A100).
  Base image is a required `--build-arg` so a build cannot silently land on the
  wrong numerics stack.
- `Dockerfile.npu` — the verified 910B runner (Ascend CANN base; pinned
  `torch-npu` + `triton-ascend`).

---

## 2. Corpus construction (provider-side, offline)

Building a task is a staged, idempotent pipeline. It runs only when the
benchmark is *authored* — never during a submitter's evaluation.

```
freeze upstream kernel ─▶ standalone-severance verify ─▶ author PyTorch oracle
        ─▶ generate shapes + contract ─▶ schema-validate ─▶ oracle-equivalence
        cross-check ─▶ capture golden
```

**1. Freeze the upstream kernel.** `scripts/verify/freeze_upstream_sources.py`
vendors the production `@triton.jit` kernel into `kernel.py`, recording the
upstream repo, commit, file SHA, and license. The raw snapshot is also kept
under `ascendbench/frozen/snapshots/`.

**2. Make it standalone, then verify bit-exactly.**
`scripts/verify/standalone_freeze_verifier.py` performs a symbol-level
reachable-closure analysis: every external symbol the frozen kernel touches is
classified (vendor it, rewrite it to a public Triton import, replace it with a
provably-inert shim, or — if it cannot be made deterministic and standalone —
**BLOCKED**, which is a hard gate, never a silent drop). The severed kernel is
then verified for **bit-exact equivalence** to the original upstream call
(values plus aliasing and mutated-input behaviour). That verification's
authoritative pass requires a real GPU; on a CPU-only box it honestly reports
"needs GPU" rather than claiming a pass. The result is recorded in `meta.yaml`
under `phase_a_prime` (status, the GPU run id, and the standalone kernel SHA).

**3. Author the PyTorch oracle.**
`scripts/corpus/generate_oracle_from_kernel.py` AST-parses the frozen kernel,
extracts the real (non-pointer, non-constexpr) tensor argument list, and
scaffolds an `oracle_pytorch.py` with a typed `run(inputs)`. For well-defined
math (norm / activation / gemm families) it writes the reference directly and
marks the task `oracle_simplified: simple_pytorch_ok`. The three oracle tiers:

| `oracle_simplified` | Meaning | Golden source |
|---------------------|---------|---------------|
| `simple_pytorch_ok` | A straightforward PyTorch reference matches within tolerance. | in-process oracle |
| `formula_exact_manual` | A hand-authored closed-form reference. | in-process oracle |
| `upstream_oracle_required` | No closed-form oracle exists; ground truth can only come from running the frozen upstream kernel. | offline `golden/` artifacts (verified mode only) |

**4. Generate shapes and contract.**
`scripts/corpus/generate_shapes_and_contract.py` emits the knob grid, the
public/hidden shape rows, the `inputs:` distribution DSL, and the human-readable
`contract.md`. The `inputs.build_inputs` bridge makes `shapes.yaml` executable —
the oracle can then be run on any shape row.

**5. Validate.** `scripts/corpus/validate_tasks.py --strict` checks every task
against the schema in [`docs/reference/task_api_contract.md`](reference/task_api_contract.md):
all five files present, required `meta.yaml` fields, a known `tolerance_class`,
a parseable `@triton.jit` kernel, an `oracle_pytorch.py` that exports `run`,
cost-model formula evaluability, and `param_constraints` shape. The dataset
loader stays lenient (so a malformed task still loads and reaches the gate);
**presence and schema are enforced here**, not in the loader.

**6. Oracle-equivalence cross-check + golden capture.** On the GPU host,
`scripts/oracle/` runs the frozen kernel and compares it to the PyTorch oracle
under the task's tolerance class, recording the per-shape verdict
(`VERIFIED` / `CEILING_ACCEPTED`) into `meta.yaml`'s `provenance.golden`.
`scripts/provenance/` (`capture_golden_baselines.py`, `write_golden_manifest.py`,
`verify_upstream_commit_chain.py`, `verify_version_matrix.py`) captures the
offline golden artifacts for `upstream_oracle_required` tasks and writes the
SHA + signature manifest the evaluator later re-verifies.

---

## 3. Evaluation flow

`ascendbench/engine/eval.py::evaluate(task_id, solution_path, backend, mode)` is
the per-task orchestrator. One call grades one `(task, solution, backend)` triple.

```
load_task ─▶ static checks ─▶ build inputs ─▶ run candidate in isolation
          ─▶ grade vs oracle / tolerance ─▶ score
```

**Resolve the task.** `load_task` parses the five files into a `TaskDef`. `mode`
selects the eval-shape set: `self-reported` grades only public shapes;
`verified` grades public **and** hidden shapes. The trusted parent always holds
the full `shapes.yaml`; the candidate never does (see isolation below), so
grading hidden shapes never leaks the hidden knobs.

**Static checks.** `static_checker.check_solution` runs the task's declared
checks (AST denylist, import allowlist, constexpr match) over the submitted
source. A blocked solution fails correctness without ever running.

**Build inputs.** The trusted parent materialises every graded shape's inputs
via `inputs.build_inputs(shapes_doc, shape_id, seed)`. The correctness pass gets
one dict per shape (seed 0); the timing pass gets a per-trial sequence — warm-up
trials followed by measured trials, each a **fresh** input at a distinct seed,
so a candidate cannot cache one answer across trials.

**Run the candidate in isolation.** `integrity/process_isolation.py` never
executes candidate code in the trusted process. The submitted
`npu_triton_impl(inputs) -> dict` runs in a separate subprocess with these
host-independent guarantees:

- **Two independent passes.** A *correctness* pass, then — after a SIGKILL of the
  whole process group plus a scrub of the Triton compile cache, `/tmp` scratch,
  and device memory — a separate *timing* pass. The scrub closes the
  correctness→timing pre-computation channel.
- **Safetensors-only IPC.** Inputs flow down and outputs flow up as safetensors
  payloads plus JSON sidecars. **No pickle**, so loading a candidate's output
  cannot execute code.
- **No golden leak.** The child receives only `solution.py` plus the
  runner-materialised inputs — never the task dir, the oracle, `contract.md`,
  the `golden/` artifacts, or the shapes document. Because it is handed
  materialised inputs and not the shapes file, the hidden shapes are
  structurally absent from the candidate's view.
- **Runtime audit hook.** Inside the child, after the candidate module loads, a
  `sys.addaudithook` net (`static_checker.install_runtime_audit_hook`) blocks
  dynamic import / `exec` / subprocess / `ctypes` escapes the static AST pass
  cannot see. Trusted helper modules are imported and cached *before* the hook
  goes live; a brief "compile" phase permits the one-time Triton JIT warmup.
- **Host-authoritative timing with sampled re-verification.** The timing pass
  times each measured trial with `CLOCK_MONOTONIC_RAW`. A runner-secret RNG
  (seeded fresh per evaluation from `secrets`, distinct from the input seed)
  picks a subset of trials whose outputs are shipped up; the parent re-runs the
  oracle on those trials' inputs and compares — closing the
  "honest in correctness, fast-but-wrong in timing" cheat path.

**Grade against the oracle.** For each shape the parent obtains the golden
output and compares the candidate's correctness-pass output with
`tolerance.compare(actual, expected, tol_cls).passed`, where `tol_cls` comes
from `tolerance.load_policy()[task.tolerance_class]` (the full dtype dispatch
and the C2/S2/ceiling standards are in
[`docs/reference/correctness_model.md`](reference/correctness_model.md)). The golden source is routed
by `_resolve_golden` (oracle tier × mode):

| Task tier | Mode | Golden | `task_status` |
|-----------|------|--------|---------------|
| `simple_pytorch_ok` / `formula_exact_manual` | either | in-process PyTorch oracle | `evaluated` |
| `upstream_oracle_required` | self-reported | none | `unsupported` |
| `upstream_oracle_required` | verified, golden complete | offline `golden/<shape>.safetensors` (SHA/signature chain re-verified) | `evaluated` |
| `upstream_oracle_required` | verified, golden missing | none (fail-closed) | `blocked` |

A candidate that silently falls back to the generic torch path (producing no NPU
kernel event) is normalised to a correctness **fail** even if numerically
correct — a migration must actually run a kernel on the device.

**Score.** Correct shapes feed `perf/scoring.py::capped_geomean_native`. The
performance model (`compute_t_ref` / Roofline efficiency) needs verified 910B
peaks and is **evidence-only** — see §3.1 and
[`docs/reference/scoring_spec.md`](reference/scoring_spec.md). Timing arbitration
(host vs device-event drift, sub-microsecond quarantine) follows
[`docs/reference/timing_arbitration.md`](reference/timing_arbitration.md).

### 3.1 The release gate is correctness — not cost

What makes a task **release-ready** is correctness, computed by
`release_gate/eligibility.py::compute_release_ready_eligible`:

```
release-ready = score_eligible ∩ { tasks where is_provenance_eligible(meta) }
```

Two correctness conditions:

- **Oracle-equivalence** (`is_provenance_eligible`): the baseline trust chain is
  established — `provenance.golden.oracle_kernel_equivalence` is `VERIFIED` or an
  allowlisted large-shape dtype-ULP `CEILING_ACCEPTED`, **or** the GPU capture is
  `DONE` — and the tolerance source is not a deferred fallback.
- **Bit-exact severance** (`release_status_pass`): the frozen kernel passed
  standalone-severance verification (`phase_a_prime.status == PASS`, or
  `SKIPPED_BY_DESIGN` only when the offline golden capture is `DONE`).

`cost_model.status` is **deliberately not** a release blocker. The cost model is
an ncu-evidenced performance/access model, not a correctness proof — conditional
memory access, L2 reuse, tile geometry, and the HBM burst floor make a portable
analytic bytes formula structurally unable to hit a tight per-shape tolerance on
every shape. Its per-shape evidence and PASS/BLOCKED status stay recorded as
metadata; gating release on it would either pin release red on a perf-model
technicality or push toward dishonest shape-specific fitting. All three
release-status authorities (`validate_tasks.py`, the `tab analyze` reporter, and
the severance verifier) call the **same** eligibility functions, so the blocking
set is identical by construction.

---

## 4. What runs where

| Actor | Hardware | Runs |
|-------|----------|------|
| **Submitter** | any (no GPU/NPU needed) | `tab run` / `tab run-suite`: static check → isolated subprocess → `tolerance.compare` vs the in-process PyTorch oracle. This is the standalone, locally-reproducible correctness path. |
| **Provider (offline)** | GPU (H100/A100) | The oracle-equivalence cross-check — frozen `kernel.py` vs `oracle_pytorch.py` under the tolerance class — plus golden capture and severance verification. The GPU is the *cross-check* host, never the migration target. |
| **Provider (verified)** | Ascend 910B NPU | The authoritative verified run: the ported kernel re-executed on the 910B over public + hidden shapes, with measured latency. This is where migration is actually proven. |

The correctness grader is engineered so the host-independent, security-critical
properties (process isolation, safetensors-only IPC, no golden leak, audit hook)
hold even on a CPU-only dev box. The heavyweight device sandbox is layered on top
on the GPU/NPU machines.

---

## 5. Roadmap (not yet implemented)

These are designed but **not** shipped; nothing above depends on them.

- **Hardened verified runner.** `process_isolation.py` today enforces the
  host-independent subset (subprocess isolation, group-kill, scrub, audit hook,
  safetensors IPC). The full device sandbox — Docker + seccomp + cgroup limits,
  an NPU device allowlist, and the trusted-shim alias / `data_ptr` checks — is
  specified for the 910B machine layer but is not yet wired into the runner.
- **Public leaderboard + submission broker.** The `tab submit` endpoint and the
  two-tier (self-reported / verified) leaderboard, plus a signed-artifact broker
  for trusted golden distribution, are planned. CLI surface exists; the hosted
  service does not.
- **Roofline-scored performance ranking.** The per-task efficiency math
  (`compute_t_ref`, `capped_geomean_native`) is implemented, but ranking stays
  gated on verified 910B peak tables; until those land, performance is reported
  as raw latency only and stays out of the release gate.

See the [README](../README.md) for current status.
