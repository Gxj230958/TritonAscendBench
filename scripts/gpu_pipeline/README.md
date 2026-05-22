# GPU-host pipeline (`scripts/gpu_pipeline/`)

Automation for the **release-readiness work that must run on an NVIDIA
CUDA GPU host**. The dev machine side (scoring chain, `param_constraints`,
`cost_model` formulas) is already done; this directory drives what is
left and can only be done with a real GPU.

It does **not** cover the Ascend 910B NPU side (peak-table vector-peak
backfill) — that needs a 910B host, out of scope here.

## Quick start (on the GPU host)

```bash
git clone <triton-ascend-bench repo>          # or: git pull
cd triton-ascend-bench

# 1. one-time environment setup (frozen torch/triton stack + the bench)
bash scripts/gpu_pipeline/00_setup_env.sh

# 2. run the whole pipeline
bash scripts/gpu_pipeline/run_all.sh
```

`run_all.sh` is idempotent — re-running skips finished steps (markers
in `results/gpu_pipeline/`). `--fresh` wipes markers; `FORCE=1` ignores
them; `--from 10` starts at a given step.

Each step script is **also runnable on its own**, e.g.
`bash scripts/gpu_pipeline/10_phase_a_prime.sh --task L1_02_xxx`.

## Prerequisite NOT handled automatically: the frozen upstream sources

The Phase A' entrypoint resolver and severing flow read the **frozen
upstream `vllm` / `sglang` checkouts** from a path *outside* this repo:

```
<repo>/../infer/vllm
<repo>/../infer/sglang
```

(hard-coded in `scripts/resolve_upstream_entrypoints.py`). They are not
vendored. `00_setup_env.sh` checks for them and prints the exact clone
commands if missing. Each must be a **clean, pinned git checkout** — the
resolver is fail-closed and raises `StaleResolverCacheError` on a dirty
or unpinned tree.

## Prerequisite NOT installed by `pip`: Nsight Compute (`ncu`)

Step 20 (the cost-model GPU cross-check) measures HBM traffic with the
**Nsight Compute** CLI (`ncu`) — it reads the hardware DRAM counters
`dram__bytes_read.sum` / `dram__bytes_write.sum`. `ncu` ships with the
CUDA toolkit (it is on the `nvcr.io/nvidia/pytorch` image) but is not a
`pip` package. `00_setup_env.sh` warns if it is missing. Without `ncu`
the cross-check is fail-closed: every task is `BLOCKED` (never a faked
PASS). `ncu` may also need profiling permissions (`--cap-add` /
`NVreg_RestrictProfilingToAdminUsers=0`) on some hosts.

## Pipeline steps

| # | Script | Does | Needs GPU | Status |
|---|--------|------|-----------|--------|
| 00 | `00_setup_env.sh` | python/CUDA precheck; install the complete frozen pin set (`torch/triton/numpy/pyyaml/safetensors/click/rich/einops`) + `pip install -e .[dev]`; CUDA smoke check | driver only | ready |
| 10 | `10_phase_a_prime.sh` | resolve upstream entrypoints (`--apply`) → S2-gpu-authoritative verify + N=5 determinism → atomic-promote `kernel.py` + write `meta.phase_a_prime.status` → status report | yes | ready |
| 20 | `20_cost_model_xcheck.sh` | cost-model GPU cross-check → write `meta.cost_model.status` | yes (+ `ncu`) | ready |
| 30 | `30_golden_capture.sh` | capture offline golden for `upstream_oracle_required` tasks → `tasks/.../golden/<shape>.safetensors` | yes | ready |
| 40 | `40_release_gate.sh` | `validate_tasks.py` schema/hash + `--release-gate`; `verify_version_matrix.py` | no | ready |

Every step is implemented. The release gate (step 40) is expected to be
**RED** until every earlier step has actually run on this GPU host and
stamped its status into `meta.yaml` — that is the gate doing its job.
`run_all.sh` reports a red gate honestly and exits non-zero; it never
reports a green pipeline while the gate is red.

## The GPU-side components

### cost-model GPU cross-check (`scripts/phase_a_cost_xcheck.py`)

Per task it evaluates the analytic `W_c`/`M_r` formulas at each public
eval shape (reusing `triton_ascend_bench.cost_model` verbatim), runs the
promoted `kernel.py` on a live CUDA device under **Nsight Compute**
(`ncu`) to measure HBM traffic via the hardware DRAM counters
`dram__bytes_read.sum` + `dram__bytes_write.sum`, reconciles
`|analytic-measured|/measured <= gpu_xcheck_tol` for the **bytes** model,
and atomically writes `meta.cost_model.status = PASS|BLOCKED` plus a
per-shape evidence block (mirroring `update_meta_phase_a_prime`).

It is **measurement-authoritative and fail-closed**: HBM traffic is only
ever taken from the `ncu` DRAM counters. If `ncu` is not on `PATH`, the
profiled run fails, or the DRAM metrics are absent, the shape/task is
recorded `BLOCKED` with an explicit reason — it is **never** approximated
from CUDA-allocator memory bookkeeping or static tensor sizes (both of
which would cross-check the analytic model against another analytic
figure). The analytic FLOPs (`W_c`) are recorded as evidence but are
**not** GPU-measured: a `@triton.jit` kernel is opaque to portable FLOP
instrumentation, so rather than fabricate a FLOPs measurement the
cross-check verifies only the measured bytes (the verdict additionally
requires `ncu` to have profiled a real kernel launch).

CLI: `--task <id>` / `--all`, `--apply` (write meta.yaml), `--dry-run`
(analytic-only, no-GPU/no-`ncu` dev machine, writes nothing). A
non-score-eligible task (stub oracle / `PHASE-C-MANUAL` shapes, e.g.
`L1_21_gumbel_sample`) is recorded `SKIPPED_NON_ELIGIBLE` and never
cross-checked.

### offline golden capture (`scripts/phase_b_golden_capture.py`)

`eval.py` / `eligibility.py` read `tasks/.../golden/<shape_id>.safetensors`
for the `upstream_oracle_required` tasks (whose PyTorch oracle is known
to deviate from the frozen upstream kernel). `scripts/phase_b_golden_capture.py`
produces them: it must run **after** step 10 promote, and per
`upstream_oracle_required` task it enumerates all eval shapes (public +
sealed hidden), materialises inputs with the verified-runner seed via
`inputs.build_inputs`, runs the promoted frozen upstream `kernel.py` on
the GPU in an isolated subprocess, and writes each output dict to
`golden/<shape_id>.safetensors` (safetensors-only). It gates each task
on `meta.phase_a_prime.status == PASS` and honestly fails any task that
is not promoted yet — it never fabricates a golden artifact, and a
no-CUDA host is a hard error.

## Outputs

* `results/gpu_pipeline/*.log` — per-step logs.
* `results/gpu_pipeline/*.done` — idempotency markers.
* `tasks/*/*/meta.yaml` — updated `oracle.gpu_triton`, `phase_a_prime.*`,
  `cost_model.status` (+ per-shape `xcheck_evidence`).
* `tasks/*/*/golden/<shape_id>.safetensors` — captured ground-truth
  tensors for the `upstream_oracle_required` tasks.
* `results/gpu_pipeline/release_gate.log` — the authoritative blocking
  task list from the release gate.
