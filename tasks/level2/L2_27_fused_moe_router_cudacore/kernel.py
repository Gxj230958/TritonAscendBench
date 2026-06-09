"""Task L2_27_fused_moe_router_cudacore_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `fused_moe_router_cudacore_kernel`
Source path: `sglang/srt/layers/moe/router.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): a2e48dbbc5d4db0f81202c8060fa45ab6ccf6f5d4bc930da9591258740171aeb
"""
from typing import Optional, Tuple
import torch
import triton
import triton.language as tl
from ascendbench.frozen.shim import fused_topk  # Phase A': sibling-only CUDA topk dispatcher (B-inert-shim)
from ascendbench.frozen.shim import is_hip  # Phase A': platform/math predicate (B-inert-shim)
_is_hip = is_hip()

@triton.jit
def fused_moe_router_cudacore_kernel(input_ptr, moe_router_weight_ptr, topk_weights_ptr, topk_ids_ptr, correction_bias_ptr, is_correction_bias: tl.constexpr, num_experts: tl.constexpr, topk: tl.constexpr, moe_softcapping: tl.constexpr, moe_renormalize: tl.constexpr, hidden_dim: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    expert_offsets = tl.arange(0, num_experts)[:, None]
    router_mask = mask[None, :]
    w_router = tl.load(moe_router_weight_ptr + expert_offsets * hidden_dim + offsets[None, :], mask=router_mask, other=0.0)
    x = tl.load(input_ptr + pid * hidden_dim + offsets, mask=mask, other=0.0)
    logits = tl.sum(w_router.to(tl.float32) * x[None, :].to(tl.float32), axis=-1)
    if moe_softcapping == 0:
        logits_softcapped = logits
    else:
        logits_scaled = logits / moe_softcapping
        exped = tl.exp(2 * logits_scaled)
        top = exped - 1
        bottom = exped + 1
        logits_softcapped = top / bottom * moe_softcapping
    if is_correction_bias:
        bias = tl.load(correction_bias_ptr + tl.arange(0, num_experts))
        logits_softcapped = logits_softcapped + bias
    top1 = tl.argmax(logits_softcapped, axis=0)
    tl.store(topk_ids_ptr + pid * topk + 0, top1)
    top1_v = tl.max(logits_softcapped, axis=0)
    invsumexp = 1.0 / tl.sum(tl.exp(logits_softcapped - top1_v), axis=0)
    tl.store(topk_weights_ptr + pid * topk + 0, invsumexp)
    if topk >= 2:
        top2 = tl.argmax(tl.where(tl.arange(0, num_experts) != top1, logits_softcapped, float('-inf')), axis=0)
        tl.store(topk_ids_ptr + pid * topk + 1, top2)
        top2_v = tl.sum(logits_softcapped * (tl.arange(0, num_experts) == top2), axis=0)
        tl.store(topk_weights_ptr + pid * topk + 1, tl.exp(top2_v - top1_v) * invsumexp)
    if topk > 2:
        topk_mask = tl.full(logits_softcapped.shape, 1.0, dtype=logits_softcapped.dtype)
        topk_mask = tl.where(tl.arange(0, num_experts) != top1, topk_mask, float('-inf'))
        topk_mask = tl.where(tl.arange(0, num_experts) != top2, topk_mask, float('-inf'))
        for i in range(2, topk):
            topi = tl.argmax(logits_softcapped + topk_mask, axis=0)
            topk_mask = tl.where(tl.arange(0, num_experts) != topi, topk_mask, float('-inf'))
            tl.store(topk_ids_ptr + pid * topk + i, topi)
            topi_v = tl.sum(logits_softcapped * (tl.arange(0, num_experts) == topi), axis=0)
            tl.store(topk_weights_ptr + pid * topk + i, tl.exp(topi_v - top1_v) * invsumexp)

def fused_moe_router_cudacore(x: torch.Tensor, router_weight: torch.Tensor, topk: int, moe_softcapping: float, correction_bias: Optional[torch.Tensor]=None):
    assert len(x.shape) == 2 and x.shape[1] == router_weight.shape[1]
    bs, hidden_dim = x.shape
    num_experts = router_weight.shape[0]
    topk_weights = torch.empty((bs, topk), dtype=torch.float32, device=x.device)
    topk_ids = torch.empty((bs, topk), dtype=torch.int32, device=x.device)
    is_correction_bias = correction_bias is not None
    max_warps = 16 if _is_hip else 32
    config = {'BLOCK_SIZE': triton.next_power_of_2(hidden_dim), 'num_warps': max(min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4)}
    fused_moe_router_cudacore_kernel[bs,](x, router_weight, topk_weights, topk_ids, correction_bias, is_correction_bias=is_correction_bias, num_experts=num_experts, topk=topk, moe_softcapping=moe_softcapping, moe_renormalize=False, hidden_dim=hidden_dim, **config)
    return (topk_weights, topk_ids)


# ── Phase A' synthesised standalone entrypoint (HOLE B) ─────────────
# The staged candidate MUST expose run(inputs)->dict so S2 can compare
# it against the UNMODIFIED upstream (which exposes the positional
# entrypoint 'fused_moe_router_cudacore', NOT run). This wrapper is a PURE dict->positional
# argument shuffle of the EXISTING upstream entrypoint -- it never
# reimplements numeric logic (changing values = an A' failure). The
# original side runs the same upstream callable via the identical
# signature adapter in _pa_subproc_runner._signature_adapter, so the
# two manifests are structurally comparable. Promotion of this file is
# gated on a genuine S2-gpu-authoritative PASS (never fabricated).
def run(inputs: dict) -> dict:  # noqa: D401
    """Standalone S2 entrypoint -- maps `inputs` onto the upstream
    `fused_moe_router_cudacore` signature STRICTLY BY NAME (never a silent
    skip+positional-shift; a required no-default param with no value is
    a hard BLOCKED(shapes-incomplete), NOT a mis-call), returns a dict
    (mutated args surfaced when the upstream entrypoint mutates in
    place and returns None). This mirrors
    _pa_subproc_runner._signature_adapter EXACTLY so the original and
    severed sides bind identically."""
    import inspect

    import torch as _torch

    _MARK = "BLOCKED(shapes-incomplete"
    _fn = fused_moe_router_cudacore
    _PNM = {}  # meta.interface.param_name_map (wrapper param -> task key)
    try:
        _sig = inspect.signature(_fn)
    except (TypeError, ValueError):
        _sig = None
    if _sig is None:
        _ret = _fn(inputs)
        _bound = []
    else:
        _kw = {}
        _bound = []
        for _n, _p in _sig.parameters.items():
            if _p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if _n in inputs:
                _kw[_n] = inputs[_n]
                _bound.append(_n)
            elif _n in _PNM and _PNM[_n] in inputs:
                # meta.interface.param_name_map (baked): same task tensor
                # under a different key. _bound records the TASK KEY so the
                # in-place output reconstruction indexes inputs correctly.
                _src = _PNM[_n]
                _kw[_n] = inputs[_src]
                _bound.append(_src)
            elif _p.default is not inspect.Parameter.empty:
                continue
            else:
                raise RuntimeError(
                    _MARK + ": missing param " + repr(_n)
                    + " for entrypoint " + repr('fused_moe_router_cudacore')
                    + " -- shapes.yaml / build_inputs provides no value "
                    "and the parameter has no default. Refusing to "
                    "silently skip+shift positional args. Fix: add the "
                    "param to the task's shapes.yaml inputs (or "
                    "meta.interface.param_name_map)."
                )
        _ret = _fn(**_kw)
    if _ret is None:
        _out = {
            p: inputs[p]
            for p in _bound
            if _torch.is_tensor(inputs.get(p))
        }
        return _out or {"_inplace": None}
    if isinstance(_ret, dict):
        return _ret
    if isinstance(_ret, (list, tuple)):
        return {f"_out{i}": v for i, v in enumerate(_ret)}
    return {"_out0": _ret}
