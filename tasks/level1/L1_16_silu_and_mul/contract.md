# Contract — L1_16 SiLU + multiply gating

## What you must implement

```python
def silu_and_mul_triton(hidden_states: torch.Tensor) -> torch.Tensor: ...
```

Input shape `(bs, 2*hidden_dim)`; output shape `(bs, hidden_dim)`.

## Required semantics

```
x1 = hidden_states[:, :hidden_dim]
x3 = hidden_states[:, hidden_dim:]
silu_x1 = silu(x1.to(float32)).to(input.dtype)
output = x3 * silu_x1
```

The cast-back to input dtype **before** the multiply is load-bearing — it
matches the upstream kernel's training-time behavior. Skipping it
introduces fp32-cumulation rounding that exceeds the tolerance class.

## Out of scope

The quant_max / quantization branch in upstream raises NotImplementedError.
**Do not** implement it. Static checks reject any solution that emits
quantized output.

## Tolerance class

`elementwise` — atol 1e-3, rtol 1e-3.

## Metamorphic checks

1. `x1 = 0` → output is exactly 0 (since silu(0)=0).
2. Scaling `x3` by `k` scales output by `k` exactly (linear in x3).

## Shape coverage

Public: 5 (covers Llama / Qwen MLP intermediate widths + a small non-power-of-2).
Hidden: 3.
