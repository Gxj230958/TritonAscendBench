# Artifact Provenance (Broker Model)

> Verified-runner deployment design — future work (`docs/DESIGN.md §6`). The
> shipped layer ignores submitter-provided binary artifacts and enforces
> source-level static checks (`ascendbench/integrity/static_checker.py`).
>
> Active spec: **broker provenance only**. The earlier "two-stage allowlist" model is **deprecated** (kept only as historical reference).

## Model

Runner-controlled compiler broker. ALL `.so` loaded into candidate process must have **broker signature** in ELF section.

```
Candidate process starts
        │
        ↓
LD_AUDIT hook intercepts dlopen(*.so)
        │
        ├─ .so has broker_signature in ELF? ─┬─ yes → verify signature with runner key
        │                                     │           └─ valid → allow load
        │                                     │           └─ invalid → reject + log
        │                                     │
        │                                     └─ no → reject + log
```

## Broker Workflow

```
1. Runner starts candidate with LD_PRELOAD=/runner/broker.so
2. broker.so hooks dlopen, intercepts triton-ascend JIT compilation
3. When candidate triggers JIT:
   a. broker passes IR to runner-controlled compiler
   b. compiler produces .so
   c. broker computes (triton_ir_hash, compiler_version, sm_arch) → signs with runner private key
   d. signature embedded in .so ELF section ".broker_provenance"
   e. cache to /runner_cache/<submission_id>/triton-ascend/
4. dlopen verifies .broker_provenance signature with runner public key
5. Verified → allow; unverified → reject
```

## Hidden Shape JIT Resolved

Old "two-stage allowlist" failed at hidden shape JIT (build-time enumeration impossible without exposing hidden set).

New broker model: hidden shape JIT triggered at runtime → broker compiles + signs → loads. No conflict.

## Rejection Cases

| Case | Reject reason |
|------|--------------|
| `.so` without `.broker_provenance` ELF section | Not produced by runner-controlled compiler |
| `.so` with invalid signature | Signature corrupted or forged |
| `.so` produced via `ctypes.CDLL` outside broker | Bypass attempt |
| `.so` from `torch.utils.cpp_extension.load_inline` | Bypass attempt |
| Submitter-provided `.so` in `runtime_env/lib/*.so` | Treated as untrusted; verified runner ignores submitter `solution_artifacts.json` |

## What Submitter Sees

Submitter writes `solution.py` with `@triton.jit` kernels. They cannot control:
- Where JIT artifacts are cached (runner namespace)
- Signing key (runner private)
- Which compiler version (frozen per `infra/version_matrix.toml`)

This is invisible to legitimate use — JIT works normally. Only attempts to load arbitrary `.so` fail.

## Self-Reported vs Verified Mode

| Mode | Provenance check |
|------|-----------------|
| `self-reported` (v0.1) | submitter `solution_artifacts.json` trusted as metadata; runner does NOT enforce broker (CI speed > strict provenance) |
| `verified` (v1.0) | broker strictly enforced; submitter `solution_artifacts.json` ignored |

## Implementation Notes

- Broker is C++ binary, ~few hundred lines (intercept dlopen + sign/verify)
- Signing key rotation: runner private key rotates per benchmark version (v0.1 / v1.0 / future versions)
- Public verification key embedded in `infra/broker_pubkey.pem` (committed)
