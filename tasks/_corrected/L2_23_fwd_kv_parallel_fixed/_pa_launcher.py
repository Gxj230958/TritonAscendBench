import torch
import triton


# BLOCKED (upstream kernel bug) -- this launcher is a faithful repro driver, not a
# passing entrypoint; meta.oracle.gpu_triton stays at the placeholder.
#   The frozen vLLM prefill pipeline realizes a cross-block decay of
#   exp(-slope*(t-s-1)) instead of the documented exp(-slope*(t-s)): host
#   k_decay=exp(-slope*(BLOCK-(arange+1))) composed with the reduce block_decay
#   and none_diag q_decay=exp(-slope*i) is off by one position. It is a constant
#   exp(+slope) factor on EVERY cross-block term (verified by a single-key probe:
#   realized/oracle = exp(+slope) for all i,j). Same-block (diagonal) terms are
#   exact. So single-block shapes (NUM_BLOCK==1) match the oracle to ~1e-6 but
#   multi-block shapes are off by rel ~0.18 -- above attention_flash tolerance.
#   The task oracle is CORRECT: it matches vLLM's own reference_lightning_attention
#   recurrence (tests/kernels/attention/test_lightning_attn.py) to ~1e-7 on every
#   shape. The upstream bug is invisible to upstream CI, which exercises only
#   SEQ_LENGTHS=[16] (single-block) at atol=rtol=0.15. No faithful launcher driving
#   the UNCHANGED frozen kernels can pass: the only bridges (drop the +1 in k_decay,
#   patch q_decay, or relax/rewrite the oracle) either edit the frozen kernel math,
#   fabricate the K_decay the kernel reads, or corrupt the correct reference.
#
# Authored launcher: drives the in-module 4-stage @triton.jit lightning-
# attention prefill pipeline exactly as the upstream host wrapper
# `_attention.forward` does (vLLM
# vllm/model_executor/layers/lightning_attn.py, commit 18f6bf5). The graded
# operator is the FULL prefill output o[t] = sum_{s<=t} decay^(t-s)(q_t.k_s)v_s
# (meta.oracle.pytorch == oracle_pytorch._lightning_kv, shape [B,H,L,D]).
# `_fwd_kv_parallel` -- the declared meta.name stage -- is the KV-parallel pass
# WITHIN that pipeline: it writes the per-block KV outer products
# kv[b,h,blk,d,e] that `_fwd_kv_reduce` (prefix-sum across blocks) and
# `_fwd_none_diag_kernel` (cross-block term) read to build the non-diagonal
# contribution to o. The stage cannot be oracle-compared standalone (its
# output is [B,H,NUM_BLOCK,d,e], not [B,H,L,D]); the full pipeline is the only
# context in which it is exercised against the oracle, and a kernel that
# ignores its load-bearing K/V/K_decay inputs drops every cross-block term
# (verified to fail the seeded multi-block anchor by >0.1 rel-err).
#
# Upstream launch sequence (kernel.py `_attention.forward`, verbatim order):
#     BLOCK = 256; NUM_BLOCK = cdiv(n, BLOCK)
#     k_decay = exp(-s * (BLOCK - (arange(BLOCK)+1)))            # [H, BLOCK]
#     _fwd_diag_kernel      [b*h*NUM_BLOCK, NUM_CBLOCK]   CBLOCK=32  -> o (diag)
#     _fwd_kv_parallel      [b*h, NUM_BLOCK]              CBLOCK=64  -> kv
#     _fwd_kv_reduce        [b*h, NUM_FBLOCK]                        -> kv prefix
#     _fwd_none_diag_kernel [b*h, NUM_BLOCK*NUM_CBLOCK]             -> o (+non-diag)
#
# kv_history is allocated as zeros (B,H,D,E): the oracle is a FRESH sequence
# with no external carry-in (it sums only s <= t within the input), so a
# zero prior history reproduces the oracle's prefix exactly. This matches
# the upstream wrapper's own default (`kv_history = torch.zeros(...)`).
#
# A single full-D forward is used (NOT the upstream `lightning_attention()`
# D-chunk loop): q.k is linear in D, so the full-D result equals the summed
# chunked result, and a single pass keeps the launched-kernel count within
# meta.runner_checks.expected_kernel_count_range = [1, 4] and runs
# `_fwd_kv_parallel` over the declared full d/e feature axes.
#
# Output key matches the oracle contract (oracle_pytorch.py::run) exactly:
#   output : [B, H, L, D] float32 -- the full lightning-attention prefill.
def run(inputs: dict) -> dict:
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    s = inputs["slope"]

    # The kernels index q/k/v/o as flat [b*h*n*d] pointers, so the tensors
    # must be contiguous (mirrors `_attention.forward`'s .contiguous() calls).
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    s = s.contiguous()

    b, h, n, d = q.shape
    e = v.shape[-1]
    device = q.device

    o = torch.empty((b, h, n, e), dtype=q.dtype, device=device)

    BLOCK = 256
    NUM_BLOCK = triton.cdiv(n, BLOCK)

    CBLOCK = 32
    NUM_CBLOCK = BLOCK // CBLOCK
    assert BLOCK % CBLOCK == 0, "BLOCK must be a multiple of CBLOCK"

    # Per-head, per-position key decay: exp(-s * (BLOCK - (c+1))). The kernel
    # indexes K_decay as a flat [H * BLOCK] buffer (`K_decay + off_h * BLOCK
    # + c`), so k_decay must be [H, BLOCK]: slope reshaped to [H, 1] broadcast
    # against array.reshape(1, -1) -> [1, BLOCK]. (The kernels themselves read
    # the slope via `S + off_h`, so they receive the raw [H] tensor `s`.)
    array = torch.arange(0, BLOCK, device=device) + 1
    k_decay = torch.exp(-s.reshape(h, 1) * (BLOCK - array.reshape(1, -1)))

    # Step 1: diagonal (intra-block causal) blocks -> o.
    grid = (b * h * NUM_BLOCK, NUM_CBLOCK)
    _fwd_diag_kernel[grid](
        q,
        k,
        v,
        o,
        s,
        b,
        h,
        n,
        d,
        e,
        BLOCK=BLOCK,
        NUM_BLOCK=NUM_BLOCK,
        CBLOCK=CBLOCK,
    )

    NUM_FBLOCK = 1
    D_FBLOCK = d // NUM_FBLOCK
    assert d % NUM_FBLOCK == 0
    E_FBLOCK = e // NUM_FBLOCK
    assert e % NUM_FBLOCK == 0

    CBLOCK = 64
    NUM_CBLOCK = BLOCK // CBLOCK
    assert BLOCK % CBLOCK == 0, "BLOCK must be a multiple of CBLOCK"

    # Step 2: per-block KV outer products (the declared `_fwd_kv_parallel`
    # graded stage) -> kv[b, h, NUM_BLOCK, d, e].
    kv = torch.empty(
        (b, h, NUM_BLOCK, d, e), dtype=torch.float32, device=device
    )
    grid = (b * h, NUM_BLOCK)
    _fwd_kv_parallel[grid](
        k,
        v,
        k_decay,
        kv,
        b,
        h,
        n,
        d,
        e,
        BLOCK=BLOCK,
        NUM_BLOCK=NUM_BLOCK,
        D_FBLOCK=D_FBLOCK,
        E_FBLOCK=E_FBLOCK,
        NUM_FBLOCK=NUM_FBLOCK,
        CBLOCK=CBLOCK,
        NUM_CBLOCK=NUM_CBLOCK,
    )

    # Step 3: prefix-sum reduce across blocks (zero history carry) -> kv holds,
    # per block, the decayed sum of all PRIOR blocks' KV state.
    kv_history = torch.zeros(
        (b, h, d, e), dtype=torch.float32, device=device
    )
    grid = (b * h, NUM_FBLOCK)
    _fwd_kv_reduce[grid](
        s,
        kv,
        kv_history,
        b,
        h,
        n,
        d,
        e,
        BLOCK=BLOCK,
        NUM_BLOCK=NUM_BLOCK,
        D_FBLOCK=D_FBLOCK,
        E_FBLOCK=E_FBLOCK,
    )

    # Step 4: non-diagonal (cross-block) term: o += (q * q_decay) @ kv_prefix.
    grid = (b * h, NUM_BLOCK * NUM_CBLOCK)
    _fwd_none_diag_kernel[grid](
        q,
        o,
        s,
        kv,
        b,
        h,
        n,
        d,
        e,
        BLOCK=BLOCK,
        NUM_BLOCK=NUM_BLOCK,
        E_FBLOCK=E_FBLOCK,
        CBLOCK=CBLOCK,
        NUM_CBLOCK=NUM_CBLOCK,
    )

    return {"output": o}
