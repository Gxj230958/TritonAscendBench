# Contract — L3_03_kernel_unified_attention_2d (SCAFFOLD)

> **TODO(scaffold)**: replace this contract with the frozen upstream
> kernel's behavioural contract. Until then, this file documents only
> the metadata that has been parsed from `docs/npu/benchmark_select.md`.

## Identity

- Benchmark id: `L3_03_kernel_unified_attention_2d` (markdown id `L3-03`)
- Upstream: `vLLM` — `kernel_unified_attention_2d`
- Source path: ``vllm/v1/attention/ops/chunked_prefill_paged_decode.py``
- CV/VV: `CV` · NPU difficulty: `high` · category: `attention`

## Selection rationale

> vLLM 2D chunked prefill 形态，与 L3-01 3D 互补

(Verbatim from `docs/npu/benchmark_select.md`.)

## Required semantics

TODO(scaffold): describe the upstream kernel's per-tile semantics,
mutability of arguments, stride layout, workspace policy, and any
stage ordering invariants — mirror `tasks/level1/L1_01_bias_kernel/contract.md`.

## Tolerance class

`attention_flash` (see `triton_ascend_bench/tolerance_policy.yaml`).

## Metamorphic checks

TODO(scaffold): list runner-enforced invariants per
`specs/canary_task_design.md`.

## Shape coverage

TODO(scaffold): enumerate at least 5 public + 4 hidden shapes when the
task is promoted out of scaffold state.
