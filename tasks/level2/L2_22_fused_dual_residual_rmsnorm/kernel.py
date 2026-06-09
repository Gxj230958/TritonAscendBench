"""Task L2_22_fused_dual_residual_rmsnorm_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `fused_dual_residual_rmsnorm_kernel`
Source path: `sglang/srt/layers/elementwise.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): 62126ca8893757ae05a0d75da2087dec4cbf289f4af3d248fa84afe4fac739a9
"""
from typing import Optional, Tuple
import torch
import triton
import triton.language as tl
from ascendbench.frozen.shim import is_hip  # Phase A': platform/math predicate (B-inert-shim)
from ascendbench.frozen.shim import register_custom_op  # Phase A': inert custom-op decorator (B-inert-shim)
_is_hip = is_hip()
rmsnorm_autotune = triton.autotune(configs=[triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1), triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8, num_stages=1), triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=16, num_stages=1), triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=4), triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8), triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=16), triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=4, num_stages=4), triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8, num_stages=4), triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=16, num_stages=4), triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8, num_stages=8), triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=16, num_stages=8), triton.Config(kwargs={'BLOCK_SIZE': 2048}, num_warps=8), triton.Config(kwargs={'BLOCK_SIZE': 2048}, num_warps=16), triton.Config(kwargs={'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4), triton.Config(kwargs={'BLOCK_SIZE': 2048}, num_warps=16, num_stages=4), triton.Config(kwargs={'BLOCK_SIZE': 4096}, num_warps=8), triton.Config(kwargs={'BLOCK_SIZE': 4096}, num_warps=16), triton.Config(kwargs={'BLOCK_SIZE': 8192}, num_warps=8), triton.Config(kwargs={'BLOCK_SIZE': 8192}, num_warps=16), triton.Config(kwargs={'BLOCK_SIZE': 8192}, num_warps=32), triton.Config(kwargs={'BLOCK_SIZE': 8192}, num_warps=8, num_stages=1), triton.Config(kwargs={'BLOCK_SIZE': 8192}, num_warps=16, num_stages=1), triton.Config(kwargs={'BLOCK_SIZE': 8192}, num_warps=32, num_stages=1), triton.Config(kwargs={'BLOCK_SIZE': 8192}, num_warps=8, num_stages=4), triton.Config(kwargs={'BLOCK_SIZE': 8192}, num_warps=16, num_stages=4), triton.Config(kwargs={'BLOCK_SIZE': 8192}, num_warps=32, num_stages=4), triton.Config(kwargs={'BLOCK_SIZE': 16384}, num_warps=8), triton.Config(kwargs={'BLOCK_SIZE': 16384}, num_warps=16), triton.Config(kwargs={'BLOCK_SIZE': 16384}, num_warps=32), triton.Config(kwargs={'BLOCK_SIZE': 16384}, num_warps=8, num_stages=1), triton.Config(kwargs={'BLOCK_SIZE': 16384}, num_warps=16, num_stages=1), triton.Config(kwargs={'BLOCK_SIZE': 16384}, num_warps=32, num_stages=1), triton.Config(kwargs={'BLOCK_SIZE': 16384}, num_warps=8, num_stages=4), triton.Config(kwargs={'BLOCK_SIZE': 16384}, num_warps=16, num_stages=4), triton.Config(kwargs={'BLOCK_SIZE': 16384}, num_warps=32, num_stages=4)], key=['hidden_dim'])

@triton.jit
def fused_dual_residual_rmsnorm_kernel(output_ptr, mid_ptr, activ_ptr, residual_ptr, weight1_ptr, weight2_ptr, eps: tl.constexpr, hidden_dim: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    input_start = pid * hidden_dim
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    a_ = tl.load(activ_ptr + input_start + offsets, mask=mask, other=0.0)
    a = a_.to(tl.float32)
    rms = tl.sqrt(tl.sum(a * a, axis=0) / hidden_dim + eps)
    r = tl.load(residual_ptr + input_start + offsets, mask=mask, other=0.0)
    w1_ = tl.load(weight1_ptr + offsets, mask=mask, other=0.0)
    w1 = w1_.to(tl.float32)
    a2r = r + (a / rms * w1).to(r.dtype)
    tl.store(mid_ptr + input_start + offsets, a2r, mask=mask)
    a2r = a2r.to(tl.float32)
    rms2 = tl.sqrt(tl.sum(a2r * a2r, axis=0) / hidden_dim + eps)
    w2_ = tl.load(weight2_ptr + offsets, mask=mask, other=0.0)
    w2 = w2_.to(tl.float32)
    tl.store(output_ptr + input_start + offsets, a2r / rms2 * w2, mask=mask)
fused_dual_residual_rmsnorm_kernel_autotune = rmsnorm_autotune(fused_dual_residual_rmsnorm_kernel)

def fused_dual_residual_rmsnorm(x, residual, weight1, weight2, eps, autotune=False):
    assert len(x.shape) == 2
    assert x.shape == residual.shape and x.dtype == residual.dtype, f'x.shape={x.shape!r} residual.shape={residual.shape!r} x.dtype={x.dtype!r} residual.dtype={residual.dtype!r}'
    output, mid = (torch.empty_like(x), torch.empty_like(x))
    bs, hidden_dim = x.shape
    if autotune:
        fused_dual_residual_rmsnorm_kernel_autotune[bs,](output, mid, x, residual, weight1, weight2, eps=eps, hidden_dim=hidden_dim)
    else:
        max_warps = 16 if _is_hip else 32
        config = {'BLOCK_SIZE': triton.next_power_of_2(hidden_dim), 'num_warps': max(min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4)}
        fused_dual_residual_rmsnorm_kernel[bs,](output, mid, x, residual, weight1, weight2, eps=eps, hidden_dim=hidden_dim, **config)
    return (output, mid)


# ── Phase A' synthesised standalone entrypoint (HOLE B) ─────────────
# The staged candidate MUST expose run(inputs)->dict so S2 can compare
# it against the UNMODIFIED upstream (which exposes the positional
# entrypoint 'fused_dual_residual_rmsnorm', NOT run). This wrapper is a PURE dict->positional
# argument shuffle of the EXISTING upstream entrypoint -- it never
# reimplements numeric logic (changing values = an A' failure). The
# original side runs the same upstream callable via the identical
# signature adapter in _pa_subproc_runner._signature_adapter, so the
# two manifests are structurally comparable. Promotion of this file is
# gated on a genuine S2-gpu-authoritative PASS (never fabricated).
def run(inputs: dict) -> dict:  # noqa: D401
    """Standalone S2 entrypoint -- maps `inputs` onto the upstream
    `fused_dual_residual_rmsnorm` signature STRICTLY BY NAME (never a silent
    skip+positional-shift; a required no-default param with no value is
    a hard BLOCKED(shapes-incomplete), NOT a mis-call), returns a dict
    (mutated args surfaced when the upstream entrypoint mutates in
    place and returns None). This mirrors
    _pa_subproc_runner._signature_adapter EXACTLY so the original and
    severed sides bind identically."""
    import inspect

    import torch as _torch

    _MARK = "BLOCKED(shapes-incomplete"
    _fn = fused_dual_residual_rmsnorm
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
                    + " for entrypoint " + repr('fused_dual_residual_rmsnorm')
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
