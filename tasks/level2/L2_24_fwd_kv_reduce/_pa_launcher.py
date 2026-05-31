import torch

# Authored launcher: drives the frozen in-module lightning-attention prefill
# pipeline (vLLM vllm/model_executor/layers/lightning_attn.py, commit
# 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19) whose cross-block prefix-sum reduce
# stage is the graded kernel `_fwd_kv_reduce`.
#
# Why a launcher rather than a host wrapper symbol: kernel.py's host entry for
# this op is `lightning_attention(q, k, v, ed, block_size, kv_history)`, which
# returns a 2-tuple `(output, kv_history)`. The oracle contract is a single
# `{"output": [B,H,L,D]}`, so a direct `kernel.py::lightning_attention` bind
# would emit an extra kv_history tensor and trip the resolver's STRUCTURE-arity
# anchor (extra unexplained output => wrong operator). We call the SAME frozen
# wrapper and forward only its first return (the prefill attention output),
# matching the oracle arity exactly. No math is re-implemented here.
#
# Pipeline the wrapper drives (all four frozen @triton.jit kernels):
#   1. _fwd_diag_kernel       -- within-block causal attention (diagonal blocks)
#   2. _fwd_kv_parallel       -- per-block decay-weighted KV outer products
#   3. _fwd_kv_reduce  <-- GRADED -- cross-block prefix recurrence over the
#        per-block KV: KV[i] <- (prefix-before-i) ; kv_pre = block_decay*kv_pre
#        + kv_cur. This rewrites each block's KV slot to the inclusive prefix
#        from earlier blocks, which stage 4 reads back. With >=2 blocks the
#        recurrence carry is load-bearing: a no-op/identity reduce leaves the
#        spurious within-block sum in KV[i] and corrupts every cross-block
#        (non-diagonal) output entry -> oracle compare FAILS.
#   4. _fwd_none_diag_kernel  -- cross-block contribution q @ KV * q_decay,
#        summed onto the diagonal output. This is the consumer of the reduce.
#
# Inputs (oracle contract, fp32): q,k,v [B,H,L,D], slope [H]. The wrapper takes
# the per-head decay `ed=slope` and lazily allocates kv_history=zeros on the
# first (and here only) call, so the prefix recurrence starts from a zero carry
# exactly as the O(L^2) oracle's `s <= t` causal accumulation does.
def run(inputs: dict) -> dict:
    q = inputs["q"].contiguous()
    k = inputs["k"].contiguous()
    v = inputs["v"].contiguous()
    slope = inputs["slope"].contiguous()

    # `lightning_attention` is a kernel.py module-level symbol; the loaders exec
    # this launcher into the kernel module, so it resolves at call time. It
    # returns (output [B,H,L,D], kv_history); the graded operator is the prefill
    # output, so forward only that to match the oracle's single-output arity.
    output, _kv_history = lightning_attention(
        q, k, v, slope, block_size=256, kv_history=None,
    )
    return {"output": output.to(q.dtype)}
