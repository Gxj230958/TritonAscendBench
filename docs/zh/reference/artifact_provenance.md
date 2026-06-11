# Artifact Provenance（Broker 模型）

> English: [../../reference/artifact_provenance.md](../../reference/artifact_provenance.md)

> Verified-runner 部署设计 —— 未来工作（`docs/DESIGN.md §6`）。当前已交付的层会忽略提交者提供的二进制构件，并强制执行源码级静态检查（`ascendbench/integrity/static_checker.py`）。
>
> 当前生效的规范：**仅 broker provenance**。早先的 "two-stage allowlist" 模型已**废弃**（仅作为历史参考保留）。

## Model

由 runner 控制的编译器 broker。加载进候选实现进程的所有 `.so` 都必须在 ELF section 中带有 **broker signature**。

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

旧的 "two-stage allowlist" 在隐藏 shape 的 JIT 上失败了（在不暴露隐藏集合的前提下，构建期枚举是不可能的）。

新的 broker 模型：隐藏 shape 的 JIT 在运行时触发 → broker 编译 + 签名 → 加载。不存在冲突。

## Rejection Cases

| Case | Reject reason |
|------|--------------|
| `.so` without `.broker_provenance` ELF section | 不是由 runner 控制的编译器产出的 |
| `.so` with invalid signature | 签名被破坏或伪造 |
| `.so` produced via `ctypes.CDLL` outside broker | 绕过尝试 |
| `.so` from `torch.utils.cpp_extension.load_inline` | 绕过尝试 |
| Submitter-provided `.so` in `runtime_env/lib/*.so` | 视为不可信；verified runner 会忽略提交者的 `solution_artifacts.json` |

## What Submitter Sees

提交者编写带有 `@triton.jit` kernel 的 `solution.py`。他们无法控制：
- JIT 构件缓存到哪里（runner 命名空间）
- 签名密钥（runner 私有）
- 使用哪个编译器版本（按 `infra/version_matrix.toml` 冻结）

对于合法使用而言，这一切都是不可见的 —— JIT 正常工作。只有试图加载任意 `.so` 的行为才会失败。

## Self-Reported vs Verified Mode

| Mode | Provenance check |
|------|-----------------|
| `self-reported` (v0.1) | 提交者的 `solution_artifacts.json` 作为元数据被信任；runner 不强制执行 broker（CI 速度 > 严格 provenance） |
| `verified` (v1.0) | 严格强制执行 broker；提交者的 `solution_artifacts.json` 被忽略 |

## Implementation Notes

- Broker 是 C++ 二进制，约几百行（拦截 dlopen + 签名/验签）
- 签名密钥轮换：runner 私钥按基准版本轮换（v0.1 / v1.0 / 未来版本）
- 公开验签密钥嵌入在 `infra/broker_pubkey.pem` 中（已提交入库）
