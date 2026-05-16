# Contract — L2_10 W8A8 block INT8 matmul

## What you must implement

```python
def w8a8_block_int8_matmul(
    A, B, As, Bs, block_size, output_dtype=torch.float16,
) -> torch.Tensor: ...
```

INT8 GEMM with **block-wise** quantization:
- A activations: per-token × k-group scale `As`
- B weights: per-(n-block × k-group) scale `Bs`

## Required semantics

Mathematically:

```
A_fp[m, k] = A_int8[m, k] * As[m, k // block_k]
B_fp[n, k] = B_int8[n, k] * Bs[n // block_n, k // block_k]
C[m, n] = (A_fp @ B_fp.T).to(output_dtype)
```

The kernel **must** accumulate in fp32 (`tl.zeros((M, N), dtype=tl.float32)`)
and cast only at the final store.

## Tolerance class

`int8_w8a8` — atol 5e-2, rtol 1e-2 in fp16 output. INT8 GEMM is
deterministic over fixed accumulation order, so rounding is bounded by
the final cast.

## Required ops

The kernel **must use `tl.dot`** for the inner GEMM block. A static-check
gate rejects solutions that compute the matmul element-by-element (those
fail to use 910B cube units and miss the point of the task).

## Out of scope

- The optional `get_w8a8_block_int8_configs` JSON-lookup autotune path is
  out of scope. Use the default config (BLOCK_SIZE_M=64, BLOCK_SIZE_N=block_n,
  BLOCK_SIZE_K=block_k, GROUP_SIZE_M=32, num_warps=4, num_stages=3) or any
  static choice you can defend on 910B; per-shape autotune is not graded.

## Metamorphic checks

1. **Scale homogeneity** — `As → 2*As` ⇒ `C → 2*C` exactly (within final cast).
2. **Zero activation** — `A = 0` ⇒ `C = 0` regardless of B/As/Bs.

## Shape coverage

Public: 6 (square / Llama / Qwen MLP sizes + a small-block variant).
Hidden: 4 (sealed).

## Hardware notes

- 910B has INT8 cube arithmetic — `tl.dot(int8, int8) → int32` is the path.
- UB budget: 160 KB max per tile; default 128×128 fits comfortably.
- Watch for stride misuse with B: `stride_bk = B.stride(1)`, `stride_bn = B.stride(0)`
  per upstream wrapper — this is **B transposed in memory**, not a bug.
