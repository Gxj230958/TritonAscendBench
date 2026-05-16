# Contract — L3_04 prefill attention (FlashAttention style)

## What you must implement

```python
def context_attention_fwd(
    q, k, v, o,
    b_start_loc, b_seq_len, max_input_len,
    is_causal=True, sm_scale=None,
) -> None: ...
```

In-place write to `o`. Page size = 1. Variable-length sequences are packed
into a single batch dimension; `b_start_loc[i]` and `b_seq_len[i]` describe
the boundaries.

## Required semantics

For each sequence `i`:

```
S_i = b_seq_len[i]
range_i = [b_start_loc[i], b_start_loc[i] + S_i)
q_i, k_i, v_i = q/k/v[range_i]
if num_q_heads != num_kv_heads:
    k_i = k_i.repeat_interleave(num_q_heads / num_kv_heads, dim=1)   # GQA
    v_i = v_i.repeat_interleave(num_q_heads / num_kv_heads, dim=1)
attn_i = softmax(q_i @ k_i^T * sm_scale + causal_mask) @ v_i
o[range_i] = attn_i
```

`sm_scale` defaults to `1/sqrt(head_dim)`. `causal_mask` is `-inf` above
diagonal when `is_causal=True`, zero otherwise.

## Tolerance class

`attention_flash` — atol 5e-3, rtol 1e-3. The online-softmax recurrence
allows drift vs naive softmax up to ≈ fp16-eps × sequence_length, which
is within these bounds for the public shape set's seq lengths.

## Required ops

- **Two `tl.dot` calls** per K-block iteration: one for `qk`, one for `pv`.
- **At least one `tl.exp`** inside the K-block loop (online-softmax).

Static checks `cube_arithmetic_present` and `online_softmax_present`
enforce both.

## Constraints

- `head_dim ≤ 128` at v0.1.
- `BLOCK_M`, `BLOCK_N`, `BLOCK_DMODEL` are `tl.constexpr`. Default 64
  for 910B; submitter may override but must stay within UB 192 KB.
- `IS_CAUSAL` is a `tl.constexpr` boolean.

## Metamorphic checks

1. **Softmax translation invariance** — adding a constant to all q values
   on a sequence changes outputs only by fp rounding.
2. **Zero V** — `v = 0` ⇒ `o = 0`.
3. **Single token per seq** — every seq has length 1 ⇒ `o` equals `v`.

## Cross-oracle status

`single_oracle` at v0.1 (naive PyTorch SDPA only). A future oracle from
flash-attn reference is desirable but not yet integrated; runners must
not auto-resolve drift between sources if/when a second oracle is added.

## Shape coverage

Public: 6 (tiny single-seq → ragged batch-8 GQA + a decode-style short-seq).
Hidden: 5 (sealed, including some > 1024 sequences for streaming pressure).
