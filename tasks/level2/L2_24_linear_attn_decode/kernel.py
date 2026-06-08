"""Task L2_24_linear_attn_decode_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_linear_attn_decode_kernel`
Source path: `vllm/model_executor/layers/lightning_attn.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): 48d2a84a9a3427ad8848a3be780d1bbe94505eb2f3a4f98c096b44b784747561
Frozen by: scripts/phase_a_freeze.py
"""
import torch
from einops import rearrange
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
from ascendbench.frozen.shim import PAD_SLOT_ID  # Phase A': attention sentinel constants (B-inert-shim)

@triton.jit
def _linear_attn_decode_kernel(q_ptr, k_ptr, v_ptr, kv_cache_ptr, slope_rate, slot_idx, output_ptr, D: tl.constexpr, qkv_b_stride, qkv_h_stride, cache_b_stride, cache_h_stride, cache_d0_stride, cache_d1_stride, pad_slot_id: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Kernel for linear attention decoding with KV cache.

    This kernel computes attention for a single token using the KV cache.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)
    slot_id = tl.load(slot_idx + pid_b).to(tl.int64)
    if slot_id == pad_slot_id:
        return
    batch_id = pid_b
    head_id = pid_h
    ratio = tl.load(slope_rate + pid_h)
    qk_d_offsets = tl.arange(0, D)
    v_d_offsets = tl.arange(0, BLOCK_SIZE) + pid_d * BLOCK_SIZE
    cache_d_offsets = qk_d_offsets[:, None] * cache_d0_stride + v_d_offsets[None, :] * cache_d1_stride
    q_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    k_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    v_offset = batch_id * qkv_b_stride + head_id * qkv_h_stride
    cache_offset = slot_id * cache_b_stride + head_id * cache_h_stride
    qk_mask = qk_d_offsets < D
    v_mask = v_d_offsets < D
    q = tl.load(q_ptr + q_offset + qk_d_offsets, mask=qk_mask, other=0.0)
    k = tl.load(k_ptr + k_offset + qk_d_offsets, mask=qk_mask, other=0.0)
    v = tl.load(v_ptr + v_offset + v_d_offsets, mask=v_mask, other=0.0)
    kv_outer = k[:, None] * v[None, :]
    kv_mask = qk_mask[:, None] & v_mask[None, :]
    ratio = tl.exp(-ratio)
    kv_ptr = kv_cache_ptr + cache_offset + cache_d_offsets
    kv_cache_old = tl.load(kv_ptr, mask=kv_mask, other=0.0)
    kv_outer = kv_outer + ratio * kv_cache_old
    output = q[:, None].to(tl.float32) * kv_outer
    output = tl.sum(output, axis=0)
    tl.store(kv_ptr, kv_outer, mask=kv_mask)
    tl.store(output_ptr + q_offset + v_d_offsets, output, mask=v_mask)

def linear_decode_forward_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv_caches: torch.Tensor, slope_rate: torch.Tensor, slot_idx: torch.Tensor, BLOCK_SIZE: int=32) -> torch.Tensor:
    """
    Perform linear attention decoding using Triton kernels.

    Args:
        q: Query tensor of shape [B, H, 1, D]
        k: Key tensor of shape [B, H, 1, D]
        v: Value tensor of shape [B, H, 1, D]
        kv_caches: Key-value cache tensor
        slope_rate: Decay rate tensor
        slot_idx: Slot indices for batches
        BLOCK_SIZE: Size of blocks for processing

    Returns:
        output: Attention output tensor
    """
    B, H, _, D = q.shape
    assert k.shape == (B, H, 1, D)
    assert v.shape == (B, H, 1, D)
    output = torch.empty_like(q)
    grid = (B, H, D // BLOCK_SIZE)
    qkv_b_stride = q.stride(0)
    qkv_h_stride = q.stride(1)
    cache_b_stride = kv_caches.stride(0)
    cache_h_stride = kv_caches.stride(1)
    cache_d0_stride = kv_caches.stride(2)
    cache_d1_stride = kv_caches.stride(3)
    _linear_attn_decode_kernel[grid](q, k, v, kv_caches, slope_rate, slot_idx, output, D, qkv_b_stride, qkv_h_stride, cache_b_stride, cache_h_stride, cache_d0_stride, cache_d1_stride, pad_slot_id=PAD_SLOT_ID, BLOCK_SIZE=BLOCK_SIZE)
    output = rearrange(output, 'b h n d -> b n (h d)')
    return output.squeeze(1).contiguous()


# ── Phase A' synthesised standalone entrypoint (HOLE B) ─────────────
# The staged candidate MUST expose run(inputs)->dict so S2 can compare
# it against the UNMODIFIED upstream (which exposes the positional
# entrypoint 'linear_decode_forward_triton', NOT run). This wrapper is a PURE dict->positional
# argument shuffle of the EXISTING upstream entrypoint -- it never
# reimplements numeric logic (changing values = an A' failure). The
# original side runs the same upstream callable via the identical
# signature adapter in _pa_subproc_runner._signature_adapter, so the
# two manifests are structurally comparable. Promotion of this file is
# gated on a genuine S2-gpu-authoritative PASS (never fabricated).
def run(inputs: dict) -> dict:  # noqa: D401
    """Standalone S2 entrypoint -- maps `inputs` onto the upstream
    `linear_decode_forward_triton` signature STRICTLY BY NAME (never a silent
    skip+positional-shift; a required no-default param with no value is
    a hard BLOCKED(shapes-incomplete), NOT a mis-call), returns a dict
    (mutated args surfaced when the upstream entrypoint mutates in
    place and returns None). This mirrors
    _pa_subproc_runner._signature_adapter EXACTLY so the original and
    severed sides bind identically."""
    import inspect

    import torch as _torch

    _MARK = "BLOCKED(shapes-incomplete"
    _fn = linear_decode_forward_triton
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
                    + " for entrypoint " + repr('linear_decode_forward_triton')
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
