"""Reproducible proof: the frozen L2_24/L2_23 lightning-attention kernel has an
upstream off-by-one in its cross-block decay.

The kernel computes the decay across blocks as
    q_decay[lq] * block_decay^(i-1-j) * k_decay[r]
with (from kernel.py):
    k_decay[r]  = exp(-s * (BLOCK - r - 1))   # array = arange(0,BLOCK) + 1
    block_decay = exp(-s * BLOCK)
    q_decay[lq] = exp(-s * lq)
which algebraically reduces to exp(-s * ((t - s) - 1)) -- ONE decay step short
(weight too large by exp(+s)). The textbook / canonical (OpenNLPLab) form uses
array = arange(0, BLOCK) with no +1, giving exp(-s * (t - s)).

Run: python3 docs/findings/L2_24_lightning_attn_offbyone_proof.py
Expected: kernel(+1) diverges O(1) from the naive reference; the no-+1 variant
matches it to 0.00.
"""
import math
import torch


def naive(q, k, v, slope):
    L = q.shape[0]
    o = torch.zeros(L, v.shape[1])
    for t in range(L):
        for s in range(t + 1):
            o[t] += math.exp(-slope * (t - s)) * (q[t] @ k[s]) * v[s]
    return o


def chunked(q, k, v, slope, BLOCK, plus_one):
    L = q.shape[0]
    o = torch.zeros(L, v.shape[1])
    block_decay = math.exp(-slope * BLOCK)
    for t in range(L):
        i, lq = t // BLOCK, t % BLOCK
        for s in range(t + 1):
            j, lr = s // BLOCK, s % BLOCK
            if j == i:                                   # within-block: exact
                dec = math.exp(-slope * (t - s))
            else:                                        # cross-block: chunked
                kdec = math.exp(-slope * (BLOCK - (lr + 1) if plus_one
                                          else BLOCK - lr))
                dec = math.exp(-slope * lq) * block_decay ** (i - 1 - j) * kdec
            o[t] += dec * (q[t] @ k[s]) * v[s]
    return o


def main():
    torch.manual_seed(0)
    slope, BLOCK, L, d, e = 0.3, 4, 8, 2, 2
    q, k, v = torch.randn(L, d), torch.randn(L, d), torch.randn(L, e)
    ref = naive(q, k, v, slope)
    ker = chunked(q, k, v, slope, BLOCK, plus_one=True)    # frozen kernel
    fix = chunked(q, k, v, slope, BLOCK, plus_one=False)   # canonical / no +1
    d_ker = (ker - ref).abs().max().item()
    d_fix = (fix - ref).abs().max().item()
    print(f"kernel(+1)      vs naive: max_abs_diff = {d_ker:.4f}  (O(1) -> bug)")
    print(f"corrected(no+1) vs naive: max_abs_diff = {d_fix:.2e}  (exact match)")
    assert d_fix < 1e-6 < d_ker, "expected: no-+1 exact, +1 diverges O(1)"
    print("PROOF HOLDS: the +1 is a real off-by-one (oracle correctly diverges).")


if __name__ == "__main__":
    main()
