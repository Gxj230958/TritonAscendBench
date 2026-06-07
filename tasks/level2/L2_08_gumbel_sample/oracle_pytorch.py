"""Oracle reference for L2_08_gumbel_sample_kernel.

Tier: upstream_oracle_required
Tolerance class: sampling_rng (bit-exact when the seed matches)
Category: sampling

The kernel under test is the GPU triton implementation vendored in
kernel.py; this file authors an eager PyTorch reference from the
operation's documented mathematical definition (no upstream code copy).

ORACLE-SCOPE: reproduces the full gumbel-max sampling contract of vLLM
`_gumbel_sample_kernel` -- per token, per vocab block:
  * optional temperature scaling (APPLY_TEMPERATURE and temp != 0),
  * gumbel noise from Triton's Philox PRNG (temp != 0) added in fp64,
  * masked argmax over the block (ties -> first index),
then the host reduction argmax-over-blocks + gather. Bit-exact because the
Philox 4x32-10 PRNG is reproduced exactly (validated against
tl.randint / tl.randint4x under TRITON_INTERPRET). NOT covered: the
optional processed-logits side output (not part of the sampling contract).
"""
from __future__ import annotations

import numpy as np
import torch

# --- Triton Philox 4x32-10 PRNG, reproduced bit-exact in integer math. ------
# Mirrors triton.language.random.philox_impl / randint / randint4x for the
# uint32 path (the only path the kernel uses). Validated bit-exact against
# tl.randint / tl.randint4x under TRITON_INTERPRET=1.
_M32 = 0xFFFFFFFF
_M64 = 0xFFFFFFFFFFFFFFFF
_PHILOX_KEY_A = 0x9E3779B9
_PHILOX_KEY_B = 0xBB67AE85
_PHILOX_ROUND_A = 0xD2511F53
_PHILOX_ROUND_B = 0xCD9E8D57
_N_ROUNDS = 10
# 1 / 2**64, the fp64 scale tl_rand64 applies to the 64 random bits.
_RAND64_SCALE = 5.421010862427522170037e-20
# float64 tiny -- tl_rand64(includes_zero=False) clamps u to this so log(u)
# is finite (the kernel's `tl.maximum(u, 2.2250738585072014e-308)`).
_TINY_F64 = 2.2250738585072014e-308


def _umulhi32(a: torch.Tensor, b: int) -> torch.Tensor:
    """High 32 bits of (a * b) for uint32 values held in int64 tensors."""
    prod = (a.to(torch.int64) & _M32) * (int(b) & _M32)
    return (prod >> 32) & _M32


def _philox_impl_u32(c0, c1, c2, c3, k0: int, k1: int):
    """``_N_ROUNDS`` Philox rounds on state (c0..c3) and key (k0, k1).

    c0..c3 are int64 tensors holding uint32 values; k0, k1 are python ints.
    """
    A, B = _PHILOX_ROUND_A, _PHILOX_ROUND_B
    for _ in range(_N_ROUNDS):
        _c0, _c2 = c0, c2
        c0 = (_umulhi32(_c2, B) ^ c1 ^ k0) & _M32
        c2 = (_umulhi32(_c0, A) ^ c3 ^ k1) & _M32
        c1 = ((B * (_c2 & _M32)) & _M32)
        c3 = ((A * (_c0 & _M32)) & _M32)
        k0 = (k0 + _PHILOX_KEY_A) & _M32
        k1 = (k1 + _PHILOX_KEY_B) & _M32
    return c0, c1, c2, c3


def _randint4x(seed: int, offset: torch.Tensor):
    """Four uint32 random blocks for ``seed`` (scalar) and ``offset`` block."""
    off = offset.to(torch.int64)
    offset_lo = off & _M32
    offset_hi = (off >> 32) & _M32
    z = torch.zeros_like(offset_lo)
    seed = int(seed) & _M64
    return _philox_impl_u32(
        offset_lo, offset_hi, z, z, seed & _M32, (seed >> 32) & _M32
    )


def _randint(seed: int, offset: torch.Tensor) -> torch.Tensor:
    return _randint4x(seed, offset)[0]


def _tl_rand64(gseed: int, offset: torch.Tensor) -> torch.Tensor:
    """fp64 uniform in (0, 1) from 64 Philox bits (tl_rand64, includes_zero=False)."""
    lo, hi, _, _ = _randint4x(gseed, offset)
    r = (((hi & _M32) << 32) | (lo & _M32)).to(torch.int64)
    # uint64 -> fp64 (numpy keeps the unsigned interpretation; r holds the
    # uint64 bit pattern in two's-complement int64, matching tl.to(uint64)).
    r_u64 = r.numpy().astype(np.uint64)
    u = torch.from_numpy(r_u64.astype(np.float64) * _RAND64_SCALE)
    return torch.maximum(u, torch.tensor(_TINY_F64, dtype=torch.float64))


def _argmax_first(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """(max, first-index-of-max) -- matches tl.max(return_indices=True)."""
    m = torch.max(x)
    idx = int((x == m).nonzero(as_tuple=False)[0].item())
    return m, idx


def run(inputs: dict) -> dict:
    logits = inputs["logits"]
    expanded_idx_mapping = inputs["expanded_idx_mapping"]
    temperature = inputs["temperature"]
    seed = inputs["seed"]
    pos = inputs["pos"]
    apply_temperature = bool(inputs.get("apply_temperature", True))

    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    num_blocks = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    local_argmax = torch.empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = torch.empty(num_tokens, num_blocks, dtype=torch.float64)

    neg_inf32 = torch.tensor(float("-inf"), dtype=torch.float32)
    neg_inf64 = torch.tensor(float("-inf"), dtype=torch.float64)
    for t in range(num_tokens):
        req = int(expanded_idx_mapping[t].item())
        temp = float(temperature[req].to(torch.float32).item())
        s = int(seed[req].item())
        p = int(pos[t].item())
        for b in range(num_blocks):
            block = b * BLOCK_SIZE + torch.arange(BLOCK_SIZE, dtype=torch.int64)
            mask = block < vocab_size
            # Load with other=-inf (kernel: tl.load(..., other=-inf)), fp32.
            gathered = logits[t, block.clamp(max=vocab_size - 1)].to(torch.float32)
            lg = torch.where(mask, gathered, neg_inf32)
            if temp != 0.0 and apply_temperature:
                lg = lg / temp
            lg64 = lg.to(torch.float64)
            if temp != 0.0:
                gseed = int(_randint(s, torch.tensor([p], dtype=torch.int64))[0].item())
                u = _tl_rand64(gseed, block)
                gumbel_noise = -torch.log(-torch.log(u))
                lg64 = torch.where(mask, lg64 + gumbel_noise, neg_inf64)
            value, idx = _argmax_first(lg64)
            local_argmax[t, b] = b * BLOCK_SIZE + idx
            local_max[t, b] = value

    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return {"sampled": sampled}
