# TritonAscendBench (TAB)

> A **Migration Benchmark** for porting GPU Triton kernels to Huawei Ascend 910B via `triton-ascend`.

[![Status](https://img.shields.io/badge/Plan-v4.13_FROZEN-brightgreen)](docs/benchmark_impl_plan.md)
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

# Phase 0.5: scaffold the 100-task tree from docs/benchmark_select.md.
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
needs the 910B machine — see **Pipeline Status** below.

## Task Levels

| Level | Tasks | Type | Hardware |
|-------|-------|------|----------|
| **L1** | 40 | Single-op / elementwise / reduce / quant / sample | NPU Vector |
| **L2** | 38 | Fused / Cube matmul / LoRA / MoE EP / Mamba step | NPU Cube + Vector |
| **L3** | 23 | Full Flash Attention / Fused MoE / Mamba chunk pipeline | NPU full pipeline |

Selection rationale: see [`docs/benchmark_select.md`](docs/benchmark_select.md).

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

Full scoring spec: [`specs/scoring_spec.md`](specs/scoring_spec.md).

## Two-Tier Leaderboard

| Tier | Verification | Use case |
|------|-------------|----------|
| **v0.1 self-reported** | Submitter's own measurements + hash check | Iteration, sandbox |
| **v1.0 verified** | Official 910B runner 100% re-execution + hidden shapes + sandbox + audit | Public ranking |

## Reward Hacking Defense

29 by-design defense principles across 7 layers (Process / Code / Correctness / Sanity / Submission / Test-set / Cross-cutting), covering 31 attack categories.

Full defense spec: [`specs/reward_hack_defense.md`](specs/reward_hack_defense.md).

## Documentation

- [Selection rationale (100 tasks)](docs/benchmark_select.md) — v3.1
- [Implementation plan v4.13 FROZEN](docs/benchmark_impl_plan.md) — 19 codex review rounds
- [Task API contract](specs/task_api_contract.md)
- [Tolerance policy](specs/tolerance_policy.md)

## Project Status

| Component | Status |
|-----------|--------|
| Plan | ✅ FROZEN v4.13 |
| Repo skeleton + 13 provider modules | ✅ Done |
| 101 task directories (L1=40, L2=38, L3=23) | ✅ Done |
| Phase A — frozen upstream `kernel.py` | ✅ 101/101 real upstream kernels (3 v3.2 errata slots re-tasked from CSV) |
| Phase B — `oracle_pytorch.py` ground truth | ✅ 99 working references; 1 manual (L1_24); 1 RNG stub (L1_21) |
| Phase C — `shapes.yaml` + `contract.md` | ✅ 99 executable via `inputs.build_inputs` |
| Phase 2.B — isolated correctness grader | ✅ Works on dev (no GPU/NPU) |
| Perf baseline + timing grader | ⬜ Deferred (no AscendC; needs 910B) |
| Leaderboard | ⬜ Phase 4 |

### Pipeline Status

End-to-end on a non-GPU dev box, a legitimate NPU-triton solution
flows: **static-check → isolated subprocess → `tolerance.compare`
vs oracle**, for **99 / 101** tasks.

| Bucket | Count | Note |
|--------|------:|------|
| Executable & gradeable (correctness) | 99 | `build_inputs → oracle.run` verified across ≥6 shape rows |
| Phase-C manual | 1 | `L1_24` — non-indptr cumulative input convention; documented `PHASE-C-MANUAL` |
| RNG stub | 1 | `L1_21` (philox RNG → upstream oracle). The 3 former errata slots (`L1_30`/`L3_03`/`L3_07`) were re-tasked to real CSV operators in v3.2 |

**Honest scope:** correctness grading is fully exercised here.
Timing/`FastP` requires the Ascend 910B (process-isolation §2);
there is no AscendC baseline (removed — migration benchmark). 17 oracles are canonical references with
documented `ORACLE-SCOPE` gaps (uncovered optional kernel paths) and
carry `oracle_simplified: upstream_oracle_required` so the runner does
not over-claim exactness.

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
