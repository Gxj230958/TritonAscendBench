# Findings

Real correctness issues in *upstream* GPU Triton kernels, surfaced while building
the TritonAscendBench corpus (extracting and verifying production kernels from
vLLM / SGLang / flash-linear-attention against PyTorch reference oracles).

These are a useful side effect of the benchmark's faithfulness requirement: a
ported kernel must match the upstream GPU reference bit-for-bit (S2) and a
PyTorch oracle within tolerance (C2). When a kernel cannot satisfy both because
the *upstream* kernel itself is wrong, that is a finding.

## What qualifies

An entry belongs here only if it meets all of:

1. **Upstream, not ours** — the defect is in a third-party kernel under test, not
   in this benchmark's harness, fixtures, or oracle.
2. **Reproducible standalone** — accompanied by a self-contained script that
   demonstrates the divergence with no GPU and no benchmark dependencies beyond
   `torch` (see `L2_24_lightning_attn_offbyone_proof.py` for the format).
3. **Root-caused** — the report identifies the exact defect (file, line,
   mechanism) and the minimal fix, not just "outputs differ".
4. **Numerically shown** — the report quantifies the error and the input regime
   in which it appears.

Anything that turns out to be a benchmark-side fixture/oracle bug is fixed in the
corpus instead and does not belong here.

## Current entries

| Finding | Upstream | Summary |
|---|---|---|
| [vLLM lightning-attn cross-block decay off-by-one](vllm_lightning_attn_crossblock_decay_offbyone.md) | vLLM | Decay exponent off by one (`arange + 1`) silently corrupts output for sequence length > 256. Proof: [`L2_24_lightning_attn_offbyone_proof.py`](L2_24_lightning_attn_offbyone_proof.py). |
