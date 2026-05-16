# Contract — L3_07_sdpa (attention+mask) (SCAFFOLD)

> **TODO(scaffold)**: replace this contract with the frozen upstream
> kernel's behavioural contract. Until then, this file documents only
> the metadata that has been parsed from `docs/npu/benchmark_select.md`.

## Identity

- Benchmark id: `L3_07_sdpa (attention+mask)` (markdown id `L3-07`)
- Upstream: `Bytedance_internal` — `sdpa (attention+mask)`
- Source path: `无公开源码`
- CV/VV: `未标` · NPU difficulty: `high` · category: `attention`

## Selection rationale

> 保留作为完整 SDPA + mask + softmax + matmul 融合形态参考

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
