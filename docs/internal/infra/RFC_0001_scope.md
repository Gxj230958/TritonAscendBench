# RFC_0001 — Project Scope Decision

> **Status**: ⏳ AWAITING USER SIGN-OFF (Phase 0.4)
> Per v4.13 §Phase 0 Go/No-Go, this RFC must be signed before Phase 1 task generation begins.

## Decision Points

### Q1. Open Source vs Internal

| Option | Implications |
|--------|--------------|
| **A. Open source (Apache 2.0)** | Public GitHub repo + HF dataset; needs license review of vLLM/SGLang vendored snapshots; broader submitter pool |
| **B. Internal only** | Restricted to internal teams; no leaderboard public site; Phase 5 = internal release only |
| **C. Hybrid** | Open task definitions + harness; keep `S_native` AscendC wrappers + hidden adversarial set internal |

**Recommended**: A (open source), with C-style hidden adversarial sealing per §5b.13.

**Decision**: ___________________________________

### Q2. Leaderboard Tier

| Option | Resource | Schedule |
|--------|----------|----------|
| **A. v0.1 self-reported only** | 1 × 910B + 1 × GPU + 2.5-3.5 FTE | 9-15 weeks |
| **B. v1.0 verified leaderboard** | 2 × 910B (dev + verified-runner) + 1 × GPU + 3.5-5 FTE | 14-20 weeks |
| **C. v0.1 first, then v1.0** | Phase 4 v0.1 → eval ROI → decide v1.0 | 9-15 + (4-6 weeks if v1.0) |

**Recommended**: C (incremental).

**Decision**: ___________________________________

### Q3. Resource Commitment

| Resource | v0.1 self-reported | v1.0 verified |
|----------|-------------------|---------------|
| 910B NPU | 1 (dev/CI mock) | 2 (dev + verified-runner) |
| H100/A100 GPU | 1 | 1-2 |
| Lead engineer FTE | 1 | 1 |
| Kernel engineer FTE | 1-2 | 2-3 |
| NPU infra/CI FTE | 0.5 | 0.5-1 |
| AscendC wrapper specialist FTE | 0.5 | 0.5 |
| PM/DevRel FTE | 0.5 (Phase 4) | 0.5 |

**Decision**: ___________________________________

### Q4. KernelBench Apache 2.0 Fork — Legal Review

- KernelBench `src/kernelbench/` Apache 2.0 — directly forkable per License §4
- Inspired by, not directly copied — Modifications and contributions explicitly licensed under Apache 2.0
- NOTICE file attribution required (already drafted in repo NOTICE)
- Legal review: ⏳ pending

**Decision**: ⏳ awaiting legal team sign-off ___________________________________

## Sign-Off

- [ ] Q1 — Open Source / Internal scope: __________
- [ ] Q2 — Leaderboard tier: __________
- [ ] Q3 — Resource commitment (hardware + FTE): __________
- [ ] Q4 — License legal review: __________ (date: _________)

Signed by: ___________________ (Project Owner)
Date: ______________
