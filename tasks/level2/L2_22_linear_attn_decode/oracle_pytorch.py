"""Oracle reference for L2_22_linear_attn_decode_kernel.

Tier: formula_exact_manual
Tolerance class: attention_flash
Category: attention

The kernel under test is the GPU triton implementation vendored in
kernel.py; this file authors an eager PyTorch reference from the
operation's documented mathematical definition (no upstream code copy).

Contract matches kernel.py::linear_decode_forward_triton (the frozen
host wrapper):
  q, k, v:    [B, H, 1, D]  (single decode step)
  kv_caches:  [B, H, D, D]  (mutated in place, slot-routed)
  slope_rate: [H]
  slot_idx:   [B] int       (slot_idx[b] selects the cache row)
  output:     [B, H*D]      (rearrange "b h n d -> b n (h d)" then squeeze)
"""
from __future__ import annotations

import torch


def _linear_attn_decode(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    kv_caches: torch.Tensor, slope_rate: torch.Tensor,
    slot_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lightning/linear-attention single-step decode, slot-routed.

    For each batch b with cache slot s = slot_idx[b]:
      S' = exp(-slope)·kv_caches[s] + k^T v ; out = q · S'
    and the cache row s is overwritten with S'. q/k/v carry a singleton
    decode-step axis [B, H, 1, D]; the math runs on the squeezed
    [B, H, D] view. Returns (out [B, H*D], updated kv_caches).
    """
    B, H, _, D = q.shape
    q3 = q[:, :, 0, :].to(torch.float32)
    k3 = k[:, :, 0, :].to(torch.float32)
    v3 = v[:, :, 0, :].to(torch.float32)
    slot = slot_idx.to(torch.long)

    decay = torch.exp(-slope_rate.to(torch.float32)).reshape(1, H, 1, 1)
    new_cache = kv_caches.to(torch.float32).clone()
    # k^T v outer product per (batch, head): [B, H, D, D]
    kv_update = torch.einsum("bhd,bhe->bhde", k3, v3)
    updated = decay * kv_caches[slot].to(torch.float32) + kv_update
    new_cache[slot] = updated
    # out[b] = q3[b] · updated_state[b] -> [B, H, D]
    out = torch.einsum("bhd,bhde->bhe", q3, updated)
    out_flat = out.reshape(B, H * D).contiguous().to(q.dtype)
    return out_flat, new_cache.to(kv_caches.dtype)


def run(inputs: dict) -> dict:
    out, new_cache = _linear_attn_decode(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["kv_caches"], inputs["slope_rate"], inputs["slot_idx"],
    )
    # candidate_inplace: the kernel mutates kv_caches in place, so mirror
    # that here and let mutated-input tracking compare the post-call cache.
    inputs["kv_caches"].copy_(new_cache)
    return {"output": out}
