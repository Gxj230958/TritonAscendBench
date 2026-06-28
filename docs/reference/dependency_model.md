# Dependency Model — Runtime Decoupling from vLLM / SGLang

TritonAscendBench's tasks are Triton operators **extracted from vLLM and SGLang
production inference code**. A natural worry follows: *does running the benchmark
drag in the whole vLLM / SGLang stack?* It does not. This document explains how
the runtime is decoupled, what you actually install, and how that decoupling is
verified — without weakening the upstream-fidelity guarantee (for the precision
side of that guarantee, see [`correctness_model.md`](correctness_model.md)).

## 1. Summary

- **The runtime imports zero vLLM / SGLang.** Importing the `ascendbench`
  package and running `tab run` never imports `vllm` or `sglang`, and neither is
  a runtime dependency.
- **Default install is 8 standard packages** — `torch`, `triton`, `numpy`,
  `pyyaml`, `safetensors`, `click`, `rich`, `einops` (Python ≥ 3.10). None of
  vLLM / SGLang's heavy transitive stack (transformers, flash-attn, uvloop,
  sentencepiece, …) is pulled in.
- **vLLM / SGLang are needed only by maintainers** re-deriving provenance or
  re-running the upstream-equivalence checks — gated behind the opt-in `[infer]`
  extra and the wheel-excluded `scripts/` tree, never by a submitter grading a
  kernel.

## 2. Two roles, two dependency footprints

| Role | What they do | What they install |
|------|--------------|-------------------|
| **Submitter / evaluator** | Port a kernel, grade it (`tab run`) | the package + `torch`/`triton`/numpy stack; **no vLLM/SGLang** |
| **Benchmark maintainer** | Re-derive provenance, re-run upstream-equivalence (S2/C2), add tasks | additionally the `[infer]` extra + an external `infer/` checkout of the vLLM/SGLang sources |

The submitter footprint is the public-release surface. The maintainer footprint
exists only in a source checkout.

## 3. How the runtime is decoupled

The original upstream kernels imported vLLM / SGLang utilities, e.g.
`from vllm.triton_utils import triton` or SGLang serving symbols
(`ForwardBatch`, tensor-parallel collectives). Three mechanisms remove that at
the runtime layer:

**(a) Kernel severance.** Each task's `kernel.py` is rewritten to standalone
form: upstream imports become `import triton` (plain Triton) or
`from ascendbench.frozen.shim import …` (the bench's own stubs), and an authored
`run()` launcher is appended. The promoted `kernel.py` is therefore **not** the
verbatim upstream text — fidelity is to upstream *behaviour*, verified bit-exact
by S2 over the declared shapes (§5), with `kernel_sha256` locking the promoted
standalone file. The Triton `@triton.jit` compute remains the upstream algorithm;
what the rewrite changes is the import surface and the launch wrapper.

**(b) The `frozen/shim` stubs.** For the **non-numeric** upstream helpers a
kernel references (loggers, platform/dtype predicates, integer grid math like
`cdiv`/`next_power_of_2`, config constants), `ascendbench/frozen/shim` provides
inert stand-ins. Invariant: these are read **before** the compute path
(configuration / dtype constants), never mixed into the floating-point math.
A symbol is therefore either *unused* by the extracted single operator (a
module-level import the surrounding serving class needed — provably 0 effect) or
*used* as a config value — and **S2 verifies the latter empirically** (below).

**(c) Re-authored oracle, not a re-run kernel.** `tab run` grades a submission
against the hand-written PyTorch oracle `oracle_pytorch.py` (or a pre-stored
golden tensor), **not** by re-executing the frozen upstream kernel. So the
runtime grade path needs neither the upstream sources nor the
`vllm.triton_utils` compatibility shim.

## 4. Why the dependency is *kept* (provenance) — and isolated

The benchmark deliberately preserves its link to vLLM / SGLang as **provenance
evidence**, isolated so it never reaches the runtime:

- **`ascendbench/frozen/snapshots/`** — verbatim upstream source archives (with
  their original `import vllm`). Kept for attribution + freeze reproducibility,
  but made **non-importable**: a guard `__init__` raises on any
  `import ascendbench.frozen.snapshots.*`, so they are data, not modules.
- **`ascendbench/upstream_shims/`** — a `vllm.triton_utils` forward-shim used
  *only* on the S2 **original** side, to run the real upstream kernel under
  `TRITON_INTERPRET=1` for the equivalence comparison. Its only `import vllm`
  statements are lazy (inside functions) and never reached on `tab run`.
- **`[infer]` extra + `scripts/`** — the resolve / oracle / severance machinery
  that puts an external `infer/` vLLM/SGLang checkout on `sys.path`. Wheel-
  excluded; degrades with an actionable message if invoked without the extra.

## 5. The fidelity guarantee survives the decoupling

Decoupling does **not** trade away upstream fidelity, because the severance is
**verified against the real upstream**, not asserted:

- **S2 (severance)** runs the kernel twice — once with the **real** vLLM/SGLang
  on `sys.path` (*original* side), once with the `frozen.shim` stubs (*severed*
  side) — and asserts the outputs are **bit-exact** (values + aliasing +
  mutated-input). Any stub whose value affected output would fail the byte-for-
  byte comparison.
- **C2 (oracle-equivalence)** independently checks the kernel against the
  re-authored PyTorch oracle within a dtype-aware tolerance.

S2 is bit-exact and C2 is tolerance-graded; both are per-shape, bounded by the
corpus's shape coverage. The benchmark does **not** claim perfect alignment for
all 101 tasks: the current release status is **83 VERIFIED + 18
CEILING_ACCEPTED** (a documented large-shape precision relaxation). The task
schema also defines a **BLOCKED** status for the case where the upstream kernel
itself diverges from a correct reference; no task in the current corpus carries
it, but the mechanism exists so such a case would be reported, not papered over.
Precision details: [`correctness_model.md`](correctness_model.md).

## 6. Verified, not just intended

Runtime purity is enforced mechanically by the import-deny gate
(`tests/unit_tests/test_runtime_no_upstream_imports.py`), run in CI on a clean
`[dev,automation]` venv (no `[infer]`):

1. import the full discovered `ascendbench` runtime surface with `vllm`/`sglang`
   blocked at `sys.meta_path` → assert neither enters `sys.modules`;
2. a static scan asserts no module-top `import vllm` / `import sglang` in the
   shipped package or task runtime files — the load-bearing check, since it
   covers code paths a single run would not reach.

A `tab run` smoke (a skeleton solution, which exits at static-check before full
grading) confirms the entry path imports nothing upstream; the import-surface
walk and static scan above are what carry the full-runtime guarantee.

A future task cannot silently re-introduce the dependency: the struct-input
importer is fail-closed to an ascendbench/stdlib/torch allowlist, and the gate
fails the build on any regression.

## 7. Wheel vs. git checkout

Neither the pip **wheel** nor the Python **sdist** ships the 101 `tasks/` — both
package only `ascendbench*` (the loader resolves `REPO_ROOT/tasks`). To run the
corpus you use a **git checkout** (or the `git archive` release tarball), which
includes `tasks/` and is pruned by `.gitattributes export-ignore` to exclude
internal tooling. Either way the
importable runtime is vLLM/SGLang-free; the wheel is the cleanest demonstration
of that (its import surface pulls zero upstream).
