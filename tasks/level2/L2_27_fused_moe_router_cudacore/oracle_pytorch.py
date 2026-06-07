"""Oracle reference for L2_27_fused_moe_router_cudacore_kernel.

Tier: formula_exact_manual
Tolerance class: atomic_scatter
Category: moe

Independent eager-PyTorch reference authored from the operation's
mathematical definition (no upstream code copy). It reproduces the
contract of the frozen ``fused_moe_router_cudacore_kernel``:

    logits        = x @ router_weight^T                (fp32 accumulate)
    softcap       = moe_softcapping * tanh(logits / moe_softcapping)
                    (applied only when moe_softcapping != 0)
    logits        = logits + correction_bias            (after softcap)
    probs         = softmax(logits, dim=-1)
    topk_weights  = top-k softmax probabilities         (NOT renormalised)
    topk_ids      = top-k expert indices

The upstream kernel hardwires ``moe_renormalize=False`` and asserts
against renormalisation, so ``topk_weights`` are the *raw* softmax
probabilities of the selected experts: ``topk_weights[:, 0] = invsumexp``
(= softmax prob of the argmax) and ``topk_weights[:, j] =
exp(logit_j - logit_max) * invsumexp`` for the remaining picks. Their
row sum is therefore < 1 in general -- they are deliberately not
normalised to sum to one.
"""
from __future__ import annotations

import torch


def _fused_moe_router(
    x: torch.Tensor, router_weight: torch.Tensor, topk: int,
    moe_softcapping: float = 0.0, correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MoE router: logits = x @ W^T (+softcap +bias), softmax, top-k.

    Returns (topk_weights, topk_ids). ``topk_weights`` are the raw
    softmax probabilities of the selected experts (no renormalisation),
    matching the cudacore kernel which runs with moe_renormalize=False.
    """
    logits = x.to(torch.float32) @ router_weight.to(torch.float32).T
    if moe_softcapping and moe_softcapping > 0:
        c = float(moe_softcapping)
        logits = c * torch.tanh(logits / c)
    if correction_bias is not None:
        logits = logits + correction_bias.to(torch.float32)
    probs = torch.softmax(logits, dim=-1)
    w, idx = torch.topk(probs, topk, dim=-1)
    # No renormalisation: the kernel emits raw softmax probabilities.
    return w.to(torch.float32), idx.to(torch.int32)


def run(inputs: dict) -> dict:
    w, idx = _fused_moe_router(
        inputs["x"], inputs["router_weight"], int(inputs["topk"]),
        float(inputs.get("moe_softcapping", 0.0)),
        inputs.get("correction_bias"))
    return {"topk_weights": w, "topk_ids": idx}
