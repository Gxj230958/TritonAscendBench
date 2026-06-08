"""Task L2_05_gelu_and_mul_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `gelu_and_mul_kernel`
Source path: `sglang/srt/layers/elementwise.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): 62126ca8893757ae05a0d75da2087dec4cbf289f4af3d248fa84afe4fac739a9
Frozen by: scripts/phase_a_freeze.py
"""
from typing import Optional, Tuple
import torch
import triton
import triton.language as tl
from triton_ascend_bench.frozen.shim import is_hip  # Phase A': platform/math predicate (B-inert-shim)
from triton_ascend_bench.frozen.shim import register_custom_op  # Phase A': inert custom-op decorator (B-inert-shim)
_is_hip = is_hip()

@triton.jit
def gelu_and_mul_kernel(out_hidden_states_ptr, out_scales_ptr, hidden_states_ptr, quant_max: tl.constexpr, static_scale: tl.constexpr, hidden_dim: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    input_start = pid * hidden_dim * 2
    output_start = pid * hidden_dim
    input1_offs = tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < hidden_dim
    input3_offs = hidden_dim + tl.arange(0, BLOCK_SIZE)
    output_offs = tl.arange(0, BLOCK_SIZE)
    x1 = tl.load(hidden_states_ptr + input_start + input1_offs, mask=mask, other=0.0).to(tl.float32)
    x3 = tl.load(hidden_states_ptr + input_start + input3_offs, mask=mask, other=0.0).to(tl.float32)
    gelu_x1 = 0.5 * (1.0 + tl.erf(x1 * 0.7071067811865475)) * x1
    out = x3 * gelu_x1.to(hidden_states_ptr.dtype.element_ty)
    if quant_max is not None:
        raise NotImplementedError()
    tl.store(out_hidden_states_ptr + output_start + output_offs, out, mask=mask)

def gelu_and_mul_triton(hidden_states, scales=None, quantize=None, out=None):
    bs, in_hidden_dim = hidden_states.shape
    hidden_dim = in_hidden_dim // 2
    if out is None:
        out_hidden_states = torch.empty((bs, hidden_dim), dtype=quantize or hidden_states.dtype, device=hidden_states.device)
    else:
        assert out.shape == (bs, hidden_dim)
        assert out.dtype == (quantize or hidden_states.dtype)
        out_hidden_states = out
    out_scales = None
    static_scale = False
    if quantize is not None:
        if scales is None:
            out_scales = torch.empty((bs,), dtype=torch.float32, device=hidden_states.device)
        else:
            out_scales = scales
            static_scale = True
    max_warps = 16 if _is_hip else 32
    config = {'num_warps': max(min(triton.next_power_of_2(triton.cdiv(hidden_dim, 8 * 32)), max_warps), 4)}
    gelu_and_mul_kernel[bs,](out_hidden_states, out_scales, hidden_states, quant_max=torch.finfo(quantize).max if quantize is not None else None, static_scale=static_scale, hidden_dim=hidden_dim, BLOCK_SIZE=triton.next_power_of_2(hidden_dim), **config)
    if quantize is not None:
        return (out_hidden_states, out_scales)
    else:
        return (out_hidden_states, None)


# ── Phase A' synthesised standalone entrypoint (HOLE B) ─────────────
# The staged candidate MUST expose run(inputs)->dict so S2 can compare
# it against the UNMODIFIED upstream (which exposes the positional
# entrypoint 'gelu_and_mul_triton', NOT run). This wrapper is a PURE dict->positional
# argument shuffle of the EXISTING upstream entrypoint -- it never
# reimplements numeric logic (changing values = an A' failure). The
# original side runs the same upstream callable via the identical
# signature adapter in _pa_subproc_runner._signature_adapter, so the
# two manifests are structurally comparable. Promotion of this file is
# gated on a genuine S2-gpu-authoritative PASS (never fabricated).
def run(inputs: dict) -> dict:  # noqa: D401
    """Standalone S2 entrypoint -- maps `inputs` onto the upstream
    `gelu_and_mul_triton` signature STRICTLY BY NAME (never a silent
    skip+positional-shift; a required no-default param with no value is
    a hard BLOCKED(shapes-incomplete), NOT a mis-call), returns a dict
    (mutated args surfaced when the upstream entrypoint mutates in
    place and returns None). This mirrors
    _pa_subproc_runner._signature_adapter EXACTLY so the original and
    severed sides bind identically."""
    import inspect

    import torch as _torch

    _MARK = "BLOCKED(shapes-incomplete"
    _fn = gelu_and_mul_triton
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
                    + " for entrypoint " + repr('gelu_and_mul_triton')
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
