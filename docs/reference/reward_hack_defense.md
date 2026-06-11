# Reward Hacking by-Design Defense

> Principle IDs (`§5b.N`) are this catalogue's own stable numbering.

## Core Principle

Reward hacking defense must be **built into benchmark design**, not bolted on as runtime detectors. Runtime checks are a fallback layer, not the primary line.

## 7-Layer Defense

| Layer | Purpose | Principles |
|-------|---------|-----------|
| L1 Process / Sandbox | Isolate candidate from oracle/golden/timing | §5b.1 Oracle isolation, §5b.1a Safe output ABI, §5b.2 Runner-owned timing |
| L2 Code Enforcement | AST + import + artifact provenance | §5b.3 AST denylist, §5b.4 Artifact provenance |
| L3 Correctness Hardening | 5 sub-checks beyond bare allclose | §5b.5 (input hash / sentinel / strict output / differential / NaN consistency) |
| L4 Sanity / Anomaly | Absolute bounds catch obvious cheats | §5b.6 Sanity bound, §5b.7 Profiler verdict, §5b.8 HW telemetry |
| L5 Submission Policy | Throttle / canary / similarity / reproducibility | §5b.9–§5b.12 |
| L6 Test-Set Design | Hidden sealed + rotation + 3-random generator | §5b.13 |
| L7 Cross-Cutting | Cache scrub / delayed disclosure / audit | §5b.14–§5b.16 |
| **Addendum — Simplification Safety** | Safe CI fast-fail + L1 oracle tiers + cross-oracle drift + stdout out of verdict | §5b.25–§5b.29 |

## 31-Category Attack Matrix

Categories:

- **A1–A8** Correctness hacking (import oracle, identity, hardcode, buffer reuse, NaN abuse, tolerance fitting, input tampering)
- **B1–B9** Performance hacking (async no-sync, stream switching, JIT cache abuse, warm-up precompute, empty kernel, timing API tampering, shape routing, partial computation, frequency manipulation)
- **C1–C5** Submission / Schema (hash tampering, symlink, path traversal, manifest inconsistency, sandbox escape)
- **D1–D4** Test-set overfit (multi-submit probe, hidden timing inference, adversarial overfit, hidden dimension birthday attack)
- **E1–E5** LLM-specific (copy kernel.py, easy-task focus, comment injection, cross-task contamination, memorization)
- **F1–F10** Runner-boundary (oracle co-process pollution, native artifact whitewashing, stateful call-order hack, CPU fallback, cross-task leakage, sybil, side channels, supply chain, prompt injection, output contract abuse)

## Active Mitigation Map (per attack)

Key examples:

| Attack | Mitigation |
|--------|-----------|
| A1 Import oracle | Oracle isolation: oracle_pytorch.py NEVER mounted to candidate sandbox |
| A3 Identity hack | Differential 3 randomized inputs + canary identity trap task |
| A4 Hardcode lookup | Runner-secret RNG inputs + canary shape-route trap |
| A5 Buffer reuse | Byte-pattern sentinel detection (soft flag) |
| A8 Input tampering | Input hash before/after |
| B1 Async no-sync | Trusted shim with `torch.npu.synchronize()` device-wide |
| B2 Non-default stream | host_ms authoritative + event cross-check, drift > 10% → invalid_timing |
| B5 Empty kernel | Profiler NPU kernel event presence check |
| B7 Shape routing | Hidden adversarial set + hidden generator 3-random |
| D1 Multi-submit probe | Submission throttling (3/week/task) + sybil detection |
| F1 Oracle co-process pollution | Process isolation: candidate / oracle / comparison in separate processes |
| F2 Native artifact whitewashing | Broker-only artifact provenance |

## Simplification Safety Constraints

When applying KernelBench/TritonBench-inspired simplifications:

1. **CI fast-fail must use sampled numeric allclose, NOT stdout text** (§5b.25)
2. **L1 oracle simplification per `oracle_simplified` enum**, not blanket (§5b.26)
3. **Cross-oracle drift never auto-resolves PyTorch as truth** (§5b.27)
4. **Hidden generator randomizes shape AND value AND edge** (§5b.28)
5. **Candidate stdout is debug log ONLY, never enters verdict** (§5b.29)

Violation of any of the above re-introduces reward hacking attack surface.
