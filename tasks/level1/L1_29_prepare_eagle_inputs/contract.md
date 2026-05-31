# Contract — L1_29_prepare_eagle_inputs

> The oracle docstring is the authoritative interim spec until a
> hand-written contract supersedes it.

## Identity

- Benchmark id: `L1_29_prepare_eagle_inputs`
- Upstream: `vllm-project/vllm` — `_prepare_eagle_inputs_kernel`
- Source path: `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py`
- Category: `spec_decode` · oracle tier: `formula_exact_manual` · tolerance: `sampling_rng`

## Required semantics

EAGLE speculative-decoding draft input prep. The frozen `@triton.jit`
`_prepare_eagle_inputs_kernel` runs one program per request (grid
`(num_reqs,)`) over a token buffer packed contiguously by request. For
request `r` with token span `[start, end)` where `start =
query_start_loc[r]`, `end = query_start_loc[r + 1]`:

- truncated query length `Lt = (end - start) - num_rejected[r]`;
- `next_token = last_sampled[idx_mapping[r]]` if `num_sampled[r] > 0`
  else `next_prefill_tokens[idx_mapping[r]]` (chunked-prefill);
- `eagle_input_ids` over the truncated prefix is `target_input_ids`
  shifted left by one — `eagle_input_ids[start + k - 1] =
  target_input_ids[start + k]` for `k` in `[1, Lt)` — with the last
  truncated slot overwritten by `next_token`;
- `last_token_indices[r] = start + Lt - 1`;
- `eagle_positions[start : start + Lt] = target_positions[start : start + Lt]`;
- `eagle_query_start_loc[r] = start`; `eagle_seq_lens[r] = seq_lens[r]`;
- the final program zeros `eagle_current_draft_step` and pads the
  cudagraph tail: `eagle_query_start_loc[num_reqs : max_num_reqs + 1] =
  end_last` (the raw, un-truncated end of the last request, i.e. the total
  token count), `eagle_seq_lens[num_reqs : max_num_reqs] = 0`,
  `last_token_indices[num_reqs : max_num_reqs] = 0`.

A request's rejected-tail slots `[start + Lt, end)` of `eagle_input_ids`
/ `eagle_positions` keep the pre-launch (zero-init) buffer contents.

The fixtures pack requests with a constant raw query length `Q =
query_len` (so `query_start_loc = arange(0, num_reqs + 1) * Q`, total
tokens `= num_reqs * Q`); the oracle derives the same span layout from the
`query_len` / `num_reqs` knobs without reading any kernel pointer.

## Wiring

The graded unit is the in-module `@triton.jit
_prepare_eagle_inputs_kernel`; `kernel.py` is the frozen upstream
`EagleSpeculator` source. `_pa_launcher.py::run` rebuilds the flat
tensors the upstream host wrapper `prepare_eagle_inputs` forwards (grid
`(num_reqs,)`, `BLOCK_SIZE=1024`, zero-init output buffers) and is the
`oracle.gpu_triton` entrypoint. The oracle is independent slice math and
never invokes the kernel.

## I/O contract

`oracle_pytorch.run(inputs: dict) -> dict`. Inputs (keys consumed):
- `target_input_ids` `[num_reqs * query_len]` int32
- `target_positions` `[num_reqs * query_len]` int64
- `idx_mapping` `[num_reqs]` int64 (permutation of `[0, num_reqs)`)
- `num_sampled` `[num_reqs]` int64
- `num_rejected` `[num_reqs]` int64 (in `[0, query_len)`)
- `last_sampled` `[max_num_reqs]` int32
- `next_prefill_tokens` `[max_num_reqs]` int32
- `seq_lens` `[num_reqs]` int32
- `num_reqs`, `query_len`, `max_num_reqs` (scalar knobs)

Outputs (dict keys): `eagle_input_ids` (int32), `eagle_positions`
(int64), `eagle_query_start_loc` (int32), `eagle_seq_lens` (int32),
`last_token_indices` (int64), `eagle_current_draft_step` (int64 scalar).

Concrete per-shape tensors are built by
`triton_ascend_bench.inputs.build_inputs(shapes.yaml, shape_id)`.

## Tolerance class

`sampling_rng` — bit-exact (atol=rtol=0). All outputs are integer index /
metadata buffers.

## Shape coverage

Public: 5 (P1..P5). Hidden: 4 (H1..H4). Knob grid in `shapes.yaml`.

## Constraints

> Mirror of `meta.yaml.param_constraints` — `meta.yaml` is the single
> authoritative source.

**Status:** `scaffold: false` — hand-reconciled.

### Knob predicates (`dims`)

- `B >= 2`
- `T >= 2`
- `G >= 1`

### Input dtypes (`dtypes`)

- `target_input_ids`: `int32`
- `target_positions`: `int64`
- `idx_mapping`: `int64`
- `num_sampled`: `int64`
- `num_rejected`: `int64`
- `last_sampled`: `int32`
- `next_prefill_tokens`: `int32`
- `seq_lens`: `int32`

### Input distributions (`dists`)

- `target_input_ids`: `randint:1:1024`
- `target_positions`: `randint:0:N`
- `idx_mapping`: `cyclic_perm` (non-identity permutation of `[0, B)`)
- `num_sampled`: `window_pos:2` (alternating `[0, 1, ...]`, both branches)
- `num_rejected`: `randint:0:T` (in `[0, T)`)
- `last_sampled`: `randint:1:1024`
- `next_prefill_tokens`: `randint:1:1024`
- `seq_lens`: `randint:1:N`

### Scalar constraints (`scalars`)

- `num_reqs >= 2`
- `query_len >= 2`
- `max_num_reqs > B`

### Tensor invariants (`tensor_invariants`)

- `num_rejected` in `[0, T)` (truncated query length stays `>= 1`)
- `idx_mapping` in `[0, B)` (a permutation of `[0, num_reqs)`)
