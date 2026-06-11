# TritonAscendBench (TAB)

> A **Migration Benchmark** for porting GPU Triton kernels to Huawei Ascend 910B via `triton-ascend`.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE)
[![Tasks](https://img.shields.io/badge/Tasks-101-blue)](tasks/)

## What

TritonAscendBench evaluates whether a NPU triton-ascend implementation can match the **GPU Triton reference** from vLLM/SGLang inference paths on Huawei 910B (Atlas A2 series), in both **correctness and performance**.

Unlike [KernelBench](https://github.com/ScalingIntelligence/KernelBench) (PyTorch → CUDA generation) and [TritonBench](https://github.com/thunlp/TritonBench) (PyTorch → Triton generation), this is a **Migration Benchmark**: starting point is *existing production GPU Triton code* from vLLM / SGLang, target is NPU triton-ascend on 910B.

## Why

- **Generation benchmarks** (KernelBench, TritonBench) test "can an LLM write a kernel from scratch?"
- **Migration benchmark** (this) tests "can a real production kernel be ported to NPU without regression?"
- Real industrial scenario: vLLM / SGLang already runs on H100/A100 with mature Triton kernels. Porting them to 910B requires fidelity to upstream semantics, not creativity.

## Quick Start

### As a Submitter (porting a kernel)

```bash
pip install triton-ascend-bench

# 1. Scaffold a solution dir for a specific task. The task_id is positional.
tab init-solution L1_12_rms_norm_kernel --out my_solutions/L1_12_rms_norm_kernel

# 2. Write your NPU triton implementation in
#    my_solutions/L1_12_rms_norm_kernel/solution.py
#    Also drop a manifest.yaml at the root (see ascendbench.manifest.SolutionManifest).

# 3. Run one task locally (positional task_id).
tab run L1_12_rms_norm_kernel \
    --solution my_solutions/L1_12_rms_norm_kernel/solution.py \
    --backend gpu \
    --out round_result.json

# 4. Run the whole suite (one solution.py per task expected at <root>/<task_id>/solution.py).
tab run-suite \
    --solutions-root my_solutions \
    --backend gpu \
    --out suite_results.json

# 5. Package and submit. `package` warns if manifest.yaml is missing.
tab package my_solutions/L1_12_rms_norm_kernel --out submission.tar.gz
tab validate-submission submission.tar.gz                     # schema-check the manifest
tab submit submission.tar.gz --endpoint https://leaderboard.example.com
```

### As a Benchmark Maintainer

The task bundle is built by a staged, idempotent pipeline:

```bash
# Phase 0: validate every task against the spec schema.
python scripts/corpus/validate_tasks.py --strict
python scripts/provenance/verify_version_matrix.py

# Phase 0.5: scaffold the 101-task tree from docs/kernel-select.md.
python scripts/corpus/generate_task_scaffolds.py        # 96 scaffolds (+5 PoC)

# Phase A: vendor the upstream GPU triton kernel.py (Apache-2.0/BSD-3
# attribution preserved; raw snapshot kept in ascendbench/frozen/snapshots/).
python scripts/verify/freeze_upstream_sources.py --limit 9999    # 93 frozen

# Phase B: author the PyTorch ground-truth oracle_pytorch.py.
python scripts/corpus/generate_oracle_from_kernel.py                 # 96 working refs

# Phase C: emit size-knob shapes.yaml + contract.md; the oracle input
# bridge is ascendbench.input_spec.inputs.build_inputs.
python scripts/corpus/generate_shapes_and_contract.py                   # 92 (+5 PoC migrated)
```

The candidate **correctness grader runs without GPU/NPU**:
`ascendbench.integrity.process_isolation` executes the submitted
`npu_triton_impl(inputs)` in an isolated subprocess (safetensors-only
IPC; candidate never sees oracle output) and `tolerance.compare`
grades it per shape against the in-process oracle. Timing/performance
needs the 910B machine — see **Maturity** below.

## Task Levels

| Level | Tasks | Type | Hardware |
|-------|-------|------|----------|
| **L1** | 36 | Single-op / elementwise / reduce / quant / sample | NPU Vector |
| **L2** | 35 | Fused / Cube matmul / LoRA / MoE EP / Mamba step | NPU Cube + Vector |
| **L3** | 30 | Full Flash Attention / Fused MoE / Mamba chunk pipeline | NPU full pipeline |

Selection rationale: see [`docs/kernel-select.md`](docs/kernel-select.md).

## Scoring

This is a GPU-triton → NPU-triton **migration** benchmark. **Correctness
is the live scored axis** (per-shape `tolerance.compare` vs the PyTorch
oracle). There is **no AscendC native baseline** — the migration
question is "does the port preserve correctness/perf vs the GPU-triton
original," not "is triton-ascend as fast as hand-written native."

Performance metric (deferred — requires Ascend 910B):

$$\text{FastP}(p) = \frac{|\{t \in \text{catalogue} : \text{correct} \land t_{\text{npu}} \le p \cdot t_{\text{baseline}}\}|}{|\text{catalogue}|}$$

- `t_baseline` is the task's declared perf reference (currently
  unmeasured; raw NPU latency is reported when hardware is available)
- `FastP_native@1.0/1.5`, `capped_geomean_native` — function/key names
  retained for output-schema stability; `_native` = the NPU-triton
  track (not AscendC). Capped at 5×, floor ε=0.01.

Full scoring spec: [`docs/reference/scoring_spec.md`](docs/reference/scoring_spec.md).

## Two-Tier Leaderboard

| Tier | Verification | Use case |
|------|-------------|----------|
| **v0.1 self-reported** | Submitter's own measurements + hash check | Iteration, sandbox |
| **v1.0 verified** | Official 910B runner 100% re-execution + hidden shapes + sandbox + audit | Public ranking |

## Reward Hacking Defense

29 by-design defense principles across 7 layers (Process / Code / Correctness / Sanity / Submission / Test-set / Cross-cutting), covering 31 attack categories.

## Documentation

- [Selection rationale](docs/kernel-select.md)
- [Task API contract](docs/reference/task_api_contract.md)
- [Tolerance policy](docs/reference/tolerance_policy.md)
- [Correctness model](docs/reference/correctness_model.md) — how outputs are graded correct (precision alignment)

## Maturity

The **correctness** grader runs on a plain CPU dev box (no GPU/NPU required) and
is exercised across the whole corpus: a submission flows static-check → isolated
subprocess → `tolerance.compare` vs the PyTorch oracle. The **performance** axis
(Roofline efficiency) requires real Ascend 910B hardware and is not yet wired
end-to-end; until then performance is reported as raw latency only and is out of
the release gate. There is no AscendC baseline — this is a migration benchmark
scored against the upstream GPU-Triton reference.

## License

Apache 2.0 — see [LICENSE](LICENSE).

Vendored / inspired sources:
- [KernelBench](https://github.com/ScalingIntelligence/KernelBench) (Apache 2.0) — eval / timing patterns
- vLLM (Apache 2.0) — task source kernels
- SGLang (Apache 2.0) — task source kernels

Full attribution: [NOTICE](NOTICE).

## Contributing

This is a **provider-only** repo. We maintain task bundles, harness, baselines, and leaderboard. We do **not** write NPU triton kernel implementations (that's the submitter's role).

Bug reports and improvements welcome via GitHub issues.
