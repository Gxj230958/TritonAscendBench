# Upstream Contacts

Per v4.13 Phase 0.5 — establish liaison with upstream projects before Phase 1 task extraction.

## vLLM (Apache 2.0)

- **Repo**: https://github.com/vllm-project/vllm
- **Triton kernels location**: `vllm/model_executor/layers/`, `vllm/v1/attention/ops/`
- **AGENTS.md compliance**: Per `infer/vllm/AGENTS.md`, our contributions (if any) must use `uv`, follow pre-commit lint, and include `Co-authored-by:` trailer.
- **Issue tracker**: https://github.com/vllm-project/vllm/issues
- **Slack**: vllm.slack.com
- **License**: Apache 2.0
- **Contact**: TBD (point of contact during Phase 0.5)

## SGLang (Apache 2.0)

- **Repo**: https://github.com/sgl-project/sglang
- **Triton kernels location**: `python/sglang/srt/layers/`, `python/sglang/srt/mem_cache/`
- **Issue tracker**: https://github.com/sgl-project/sglang/issues
- **License**: Apache 2.0
- **Contact**: TBD

## sgl-kernel-npu (Apache 2.0)

- **Repo**: https://github.com/sgl-project/sgl-kernel-npu
- **AscendC native ops** (used as `S_native` baseline for ~30 tasks)
- **Contact**: TBD

## vLLM-Ascend

- **Repo**: https://github.com/vllm-project/vllm-ascend
- **Triton-ascend implementations** (used as `S_native` baseline for some tasks)
- **Contact**: TBD

## triton-ascend (compiler)

- **Repo**: https://github.com/triton-lang/triton-ascend (mirror: https://github.com/Ascend/triton-ascend)
- **Documentation**: https://ascend.github.io/triton-ascend/
- **Examples / known patterns**: https://github.com/Ascend/triton-ascend-ops
- **Compiler version frozen**: 3.2.0 (x86) / 3.5.0 (ARM)
- **Contact**: TBD (Huawei team)

## KernelBench (Apache 2.0)

- **Repo**: https://github.com/ScalingIntelligence/KernelBench
- **Used as**: reference for eval/timing/scoring patterns (adapted, see NOTICE)
- **Contact**: TBD (Stanford Scaling Intelligence)

## Communication Cadence (v4.13 §Phase 0.5)

- **Best-effort**: monthly sync with each upstream (Should Have, not blocking)
- **Hard requirement**: file issue/PR + record internal workaround for any blocking discovery
- **Escalation**: documented in `infra/escalation_log.md`
