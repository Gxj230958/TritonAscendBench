# TritonAscendBench (TAB)

A **migration benchmark** that measures whether production GPU Triton kernels —
extracted from vLLM and SGLang inference paths — can be ported to the Huawei
Ascend 910B (Atlas A2) via `triton-ascend` **without losing correctness or
performance**. Where generation benchmarks ask *"can a model write a kernel?"*,
TAB asks *"can a real kernel survive a hardware migration?"*

[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](pyproject.toml)
[![Tasks](https://img.shields.io/badge/Tasks-101-blue)](tasks/)
[![Status](https://img.shields.io/badge/Status-research_preview-orange)](#status)

English | [中文](README.zh.md)

## Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Quick Start](#quick-start)
- [Dependencies](#dependencies)
- [Task Suite](#task-suite)
- [Scoring](#scoring)
- [Leaderboard](#leaderboard)
- [Integrity](#integrity)
- [Status](#status)
- [Documentation](#documentation)
- [Citation](#citation)
- [License & Attribution](#license--attribution)
- [Contributing](#contributing)

## Overview

TritonAscendBench evaluates a candidate `triton-ascend` (NPU) implementation
against the **GPU Triton reference** the kernel was extracted from, on two axes:
**correctness** (graded now, on CPU) and **performance** (raw NPU latency,
requires 910B hardware). The 101 tasks are real operators from vLLM / SGLang —
not synthetic — so a passing submission demonstrates behaviorally faithful
handling of a production-kernel migration task, within the documented
tolerance / ceiling policy.

| | Starting point | Target | Question |
|---|---|---|---|
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | PyTorch module | CUDA | Can a model **generate** a kernel? |
| [TritonBench](https://github.com/thunlp/TritonBench) | PyTorch op | Triton | Can a model **generate** a Triton kernel? |
| **TritonAscendBench** | **Production GPU Triton** (vLLM/SGLang) | **NPU triton-ascend (910B)** | Can a real kernel **migrate** without regression? |

## Motivation

vLLM and SGLang already serve LLMs on H100 / A100 with mature, hand-tuned Triton
kernels. Bringing that ecosystem to the Ascend 910B is a **porting** problem, not
a creative one: the port must reproduce upstream numerical semantics exactly,
and ideally match its performance envelope. TAB measures precisely that gap, with
upstream itself as the ground truth.

## Quick Start

> `pip install` provides the gradable package; **clone the repository** (or use
> the release tarball) to obtain the 101 `tasks/` — they are not in the pip wheel
> or sdist (see [Dependencies](#dependencies)).

Port a kernel and grade it locally (no GPU/NPU required for correctness):

```bash
pip install triton-ascend-bench        # or: pip install -e . from a clone

# 1. Scaffold a solution directory for a task (task_id is positional).
tab init-solution L1_12_rms_norm --out my_solutions/L1_12_rms_norm

# 2. Implement your NPU triton kernel in solution.py + a manifest.yaml
#    (schema: ascendbench.manifest.SolutionManifest).

# 3. Grade one task against the PyTorch oracle.
tab run L1_12_rms_norm \
    --solution my_solutions/L1_12_rms_norm/solution.py \
    --backend gpu --out round_result.json

# 4. Grade the whole suite (one solution.py per task under the root).
tab run-suite --solutions-root my_solutions --backend gpu --out suite_results.json

# 5. Package + validate a submission. (`tab submit` POSTs to a leaderboard
#    endpoint; the hosted submission service is roadmap — see Leaderboard.)
tab package my_solutions/L1_12_rms_norm --out submission.tar.gz
tab validate-submission submission.tar.gz
```

The correctness grader needs no accelerator: a submission flows static-check →
isolated subprocess (`ascendbench.integrity.process_isolation`, safetensors-only
IPC so the candidate never sees oracle output) → `tolerance.compare` per shape
against the in-process PyTorch oracle. Performance timing requires a 910B
machine (see [Status](#status)).

<details>
<summary><b>Maintainer: rebuild the task bundle</b></summary>

The corpus is produced by a staged, idempotent pipeline (provider-side only):

```bash
python scripts/corpus/validate_tasks.py --strict        # validate against the spec schema
python scripts/provenance/verify_version_matrix.py
python scripts/corpus/generate_task_scaffolds.py        # scaffold the task tree
python scripts/verify/freeze_upstream_sources.py        # vendor upstream kernel.py (attribution preserved)
python scripts/corpus/generate_oracle_from_kernel.py    # author the PyTorch ground-truth oracle
python scripts/corpus/generate_shapes_and_contract.py   # emit shapes.yaml + contract.md
```

Re-running upstream-equivalence (S2/C2) additionally needs the `[infer]` extra and
an external `infer/` vLLM/SGLang checkout — see [Dependencies](#dependencies).
</details>

## Dependencies

Although the tasks are extracted from vLLM / SGLang, **the runtime imports
neither.** A default install is 8 standard packages — `torch`, `triton`,
`numpy`, `pyyaml`, `safetensors`, `click`, `rich`, `einops` (Python ≥ 3.10) —
with none of vLLM / SGLang's heavy transitive stack. Importing `ascendbench` and
running `tab run` pull zero `vllm` / `sglang`, enforced in CI by an import-deny
gate.

| Role | Installs | vLLM / SGLang? |
|---|---|---|
| **Submitter / evaluator** | the package + `clone` for `tasks/` | none |
| **Maintainer** (re-derive provenance, re-run S2/C2) | `+ [infer]` extra + external `infer/` checkout | yes, opt-in only |

How the decoupling works and how upstream fidelity is preserved through it:
[`docs/reference/dependency_model.md`](docs/reference/dependency_model.md).

## Task Suite

| Level | Tasks | Kernel class | 910B units |
|-------|:-----:|--------------|------------|
| **L1** | 36 | Single-op / elementwise / reduce / quant / sample | Vector |
| **L2** | 35 | Fused / Cube matmul / LoRA / MoE EP / Mamba step | Cube + Vector |
| **L3** | 30 | Flash Attention / Fused MoE / Mamba chunk pipeline | Full pipeline |

Selection rationale: [`docs/kernel-select.md`](docs/kernel-select.md).

## Scoring

**Correctness is the live scored axis** — per-shape `tolerance.compare` against
the PyTorch oracle. There is **no AscendC native baseline**: the question is
"does the port preserve the GPU-Triton original's behaviour," not "is
triton-ascend as fast as hand-written native code."

Performance is a Roofline-efficiency metric, deferred until 910B hardware is
wired end-to-end. Each correct task gets an efficiency score `s_t` (achieved vs.
Roofline-attainable throughput); the headline metrics aggregate it:

$$\text{FastP}_{\text{native}}@\tau = \frac{\bigl|\{\,t \in \text{catalogue} : s_t \ge \tau\,\}\bigr|}{|\text{catalogue}|}, \quad \tau \in \{1.0, 1.5\}$$

- `s_t` — the task's Roofline efficiency (reported once 910B latency is measured;
  an incorrect task scores 0, not excluded).
- `fastp_native_at_1_0/1_5`, `capped_geomean_native` — `_native` denotes the
  NPU-triton track (not AscendC); names retained for output-schema stability.
  Capped at 5×, floor ε = 0.01.

Full specification: [`docs/reference/scoring_spec.md`](docs/reference/scoring_spec.md).

## Leaderboard

Two tiers are designed; **v0.1 is local-only today** (package + validate + hash
check), and the hosted submission service plus the v1.0 verified runner are
roadmap (see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)).

| Tier | Verification | Status |
|------|--------------|--------|
| **v0.1 self-reported** | submitter's own measurements + hash check | local tooling shipped |
| **v1.0 verified** | official 910B runner: 100% re-execution + hidden shapes + sandbox + audit | planned |

## Integrity

Submissions run under a layered reward-hacking defense — 29 by-design principles
across 7 layers (Process / Code / Correctness / Sanity / Submission / Test-set /
Cross-cutting) covering 31 attack categories. See
[`docs/reference/reward_hack_defense.md`](docs/reference/reward_hack_defense.md).

## Status

> **Research preview.** Correctness is fully wired and exercised across the whole
> corpus on a plain CPU box. Performance (Roofline efficiency) requires real
> Ascend 910B hardware and is **not yet wired end-to-end** — until then NPU
> latency is reported as raw numbers only and is out of the release gate.
> Upstream-fidelity grading is honest about its limits: the current release
> status is **83 verified + 18 ceiling-accepted** (a documented large-shape
> precision relaxation), with a **blocked** status reserved for any task where
> upstream itself diverges from a correct reference (none today). Nothing is
> reported as perfect that is not.

## Documentation

Start by intent:

- **Use it** → [Usage Guide](docs/USAGE.md) — scaffold → implement → grade → package → submit, with a worked example.
- **Understand how it works** → [Architecture](docs/ARCHITECTURE.md) (companion overview) → [Design](docs/DESIGN.md) (rationale) · [Selection rationale](docs/kernel-select.md) (why these 101 kernels).
- **Deep reference** ([docs/reference/](docs/reference/)):
  - *Contract* — [Task API contract](docs/reference/task_api_contract.md)
  - *Correctness* — [Correctness model](docs/reference/correctness_model.md) · [Tolerance policy](docs/reference/tolerance_policy.md)
  - *Scoring* — [Scoring spec](docs/reference/scoring_spec.md) · [Timing arbitration](docs/reference/timing_arbitration.md)
  - *Integrity* — [Reward-hacking defense](docs/reference/reward_hack_defense.md) · [Process isolation](docs/reference/process_isolation.md) · [AST denylist](docs/reference/ast_denylist.md)
  - *Provenance* — [Dependency model](docs/reference/dependency_model.md) (why the runtime needs no vLLM/SGLang) · [Artifact provenance](docs/reference/artifact_provenance.md)

## Citation

```bibtex
@misc{tritonascendbench,
  title  = {TritonAscendBench: A Migration Benchmark for Porting Production
            GPU Triton Kernels to Ascend NPU},
  year   = {2026},
  note   = {https://github.com/sii-ai-infra/TritonAscendBench}
}
```

## License & Attribution

Apache 2.0 — see [LICENSE](LICENSE). The benchmark vendors and adapts third-party
sources, primarily Apache-2.0, with preserved upstream notices where applicable;
full attribution is in [NOTICE](NOTICE):

- **vLLM** / **SGLang** — task source kernels (extracted, severed, and re-graded).
- [**KernelBench**](https://github.com/ScalingIntelligence/KernelBench) — eval / timing patterns.
- [**TritonBench**](https://github.com/thunlp/TritonBench) — stdout-comparison pattern (reimplemented).

## Contributing

This is a **provider-only** repository: we maintain the task bundles, harness,
baselines, and leaderboard, and deliberately do **not** author NPU triton kernel
implementations (that is the submitter's role). Bug reports and harness
improvements are welcome via GitHub issues.
