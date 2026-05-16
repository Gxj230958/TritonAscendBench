# Contract — L1_30_prepare_mrope_positions_kernel (SCAFFOLD)

> **TODO(scaffold)**: replace this contract with the frozen upstream
> kernel's behavioural contract. Until then, this file documents only
> the metadata that has been parsed from `docs/npu/benchmark_select.md`.

## Identity

- Benchmark id: `L1_30_prepare_mrope_positions_kernel` (markdown id `L1-30`)
- Upstream: `vLLM` — `_prepare_mrope_positions_kernel`
- Source path: `vLLM mrope`
- CV/VV: `VV` · NPU difficulty: `low` · category: `attention`

## Selection rationale

> Qwen2-VL / Qwen3-VL mRoPE 位置准备

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
