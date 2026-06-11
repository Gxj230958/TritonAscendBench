# [Bug] lightning_attn: cross-block decay is off by one (weight too large by exp(slope))

**Repo:** vllm-project/vllm
**File:** `vllm/model_executor/layers/lightning_attn.py`
**Severity:** numerical-correctness (silent wrong output on sequences longer than one
`BLOCK`, i.e. seqlen > 256). Affects MiniMax-01 / lightning-attention models on long context.

## Summary

The non-diagonal (cross-block) decay path applies one fewer decay step than the
operation's definition `o[t] = Σ_{s≤t} exp(-slope·(t−s)) (q_t·k_s) v_s`. Every
cross-block (key in an earlier BLOCK than the query) contribution is therefore
scaled by `exp(+slope)` — too much weight on older tokens.

## Root cause (one character)

```python
# lightning_attn.py (current)
array  = torch.arange(0, BLOCK, device=q.device) + 1          # line 430  <-- the +1
k_decay = torch.exp(-s * (BLOCK - array.reshape(1, -1)))        # line 431
```

Combined with the other two decay factors actually used by the kernel:

```python
block_decay = tl.exp(-s * block_size)                          # line 299, exp(-s·BLOCK)
q_decay     = tl.exp(-s * (off_c*CBLOCK + c_array[:, None]))   # line 380, exp(-s·local_q)
```

the total cross-block decay for a query at global pos `t` (block i, local `lq`)
attending a key at global pos `s` (block j<i, local `lr`) reduces to:

```
q_decay · block_decay^(i-1-j) · k_decay[lr]
  = exp(-s·lq) · exp(-s·BLOCK·(i-1-j)) · exp(-s·(BLOCK - lr - 1))
  = exp(-s · (lq + BLOCK·(i-j) - lr - 1))
  = exp(-s · ((t - s) - 1))          # ONE decay step short
```

The intended factor is `exp(-s·(t-s))`. The kernel's OWN diagonal (within-block)
stage is correct (`lightning_attn.py:104-108`, `decay = exp(-s·(q_index - kv_index))`),
so the two stages are mutually inconsistent at the block boundary — a self-evident bug.

The canonical OpenNLPLab `lightning_attn2._fwd_kernel` uses `arange(0, BLOCK)` with **no
`+1`**, giving `exp(-s·(t-s))`.

## Fix (1 line)

```diff
- array = torch.arange(0, BLOCK, device=q.device) + 1
+ array = torch.arange(0, BLOCK, device=q.device)
```

## Numerical proof (reproducible, no GPU)

A faithful pure-PyTorch replica of the chunked algorithm vs the naive O(L²) reference:

```
kernel(+1)      vs naive: max_abs_diff = 0.2778   (O(1) structured error)
corrected(no+1) vs naive: max_abs_diff = 0.00e+00 (exact)
```

Per cross-block pair the error is a constant `exp(+slope)` ratio — the signature of a
decay off-by-one, not fp noise. Full script:
`triton-ascend-bench/docs/findings/L2_24_lightning_attn_offbyone_proof.py`.

## Why CI missed it

`tests/kernels/attention/test_lightning_attn.py` uses `SEQ_LENGTHS = [16]` while
`BLOCK = 256`. seqlen 16 < BLOCK → only ONE block → the cross-block path is **never
exercised**. Tolerances are also loose (`rtol=atol=1e-1 .. 1.5e-1`). Add a test case
with `seqlen >= 2*BLOCK` (e.g. 512/1024) at a tight tolerance to cover the regression.

## Discovered by

The triton-ascend-bench GPU→NPU extraction benchmark: the extracted kernel diverges
from an independent PyTorch oracle by a structured (not fp-noise) amount that vanishes
exactly when the `+1` is removed.
