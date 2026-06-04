"""Task L2_17_layer_norm_gated_fwd_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `layer_norm_gated_fwd_kernel`
Source path: `vllm/model_executor/layers/fla/ops/kda.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): 16ccf19821e32783c049d89faa3da6e6dc0ff4bda991e9b5d04b11cfb144f837
Frozen by: scripts/phase_a_freeze.py
"""
import torch
import torch.nn as nn
from tasks._frozen_shim import CustomOp  # Phase A': inert CustomOp base + no-op register (B-inert-shim)
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
from tasks._frozen_shim import cdiv, next_power_of_2  # Phase A': pure-math shim (B-inert-shim)
from tasks._frozen_deps_pool.h_af43637ae818.fla_ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h  # Phase A': U2 FLA-ops package vendoring
from tasks._frozen_deps_pool.h_af43637ae818.fla_ops.cumsum import chunk_local_cumsum  # Phase A': U2 FLA-ops package vendoring
from tasks._frozen_deps_pool.h_af43637ae818.fla_ops.fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel  # Phase A': U2 FLA-ops package vendoring
from tasks._frozen_deps_pool.h_af43637ae818.fla_ops.index import prepare_chunk_indices  # Phase A': U2 FLA-ops package vendoring
from tasks._frozen_deps_pool.h_af43637ae818.fla_ops.l2norm import l2norm_fwd  # Phase A': U2 FLA-ops package vendoring
from tasks._frozen_deps_pool.h_af43637ae818.fla_ops.op import exp, log  # Phase A': U2 FLA-ops package vendoring
from tasks._frozen_deps_pool.h_af43637ae818.fla_ops.solve_tril import solve_tril  # Phase A': U2 FLA-ops package vendoring
from tasks._frozen_deps_pool.h_af43637ae818.fla_ops.utils import FLA_CHUNK_SIZE, is_amd  # Phase A': U2 FLA-ops package vendoring

@triton.heuristics({'STORE_RESIDUAL_OUT': lambda args: args['residual_out'] is not None, 'HAS_RESIDUAL': lambda args: args['residual'] is not None, 'HAS_WEIGHT': lambda args: args['w'] is not None, 'HAS_BIAS': lambda args: args['b'] is not None})
@triton.jit
def layer_norm_gated_fwd_kernel(x, g, y, w, b, residual, residual_out, mean, rstd, eps, T, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr, ACTIVATION: tl.constexpr, IS_RMS_NORM: tl.constexpr, STORE_RESIDUAL_OUT: tl.constexpr, HAS_RESIDUAL: tl.constexpr, HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr):
    i_t = tl.program_id(0)
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    if HAS_RESIDUAL:
        p_res = tl.make_block_ptr(residual, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
        b_x += tl.load(p_res, boundary_check=(0, 1)).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        p_res_out = tl.make_block_ptr(residual_out, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
        tl.store(p_res_out, b_x.to(p_res_out.dtype.element_ty), boundary_check=(0, 1))
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=1) / D
        p_mean = tl.make_block_ptr(mean, (T,), (1,), (i_t * BT,), (BT,), (0,))
        tl.store(p_mean, b_mean.to(p_mean.dtype.element_ty), boundary_check=(0,))
        b_xbar = tl.where(m_d[None, :], b_x - b_mean[:, None], 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    else:
        b_xbar = tl.where(m_d[None, :], b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))
    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (b_x - b_mean[:, None]) * b_rstd[:, None] if not IS_RMS_NORM else b_x * b_rstd[:, None]
    b_y = b_x_hat * b_w[None, :] if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b[None, :]
    p_g = tl.make_block_ptr(g, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == 'sigmoid':
        b_y = b_y * tl.sigmoid(b_g)
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))

@triton.heuristics({'STORE_RESIDUAL_OUT': lambda args: args['residual_out'] is not None, 'HAS_RESIDUAL': lambda args: args['residual'] is not None, 'HAS_WEIGHT': lambda args: args['w'] is not None, 'HAS_BIAS': lambda args: args['b'] is not None})
@triton.jit
def layer_norm_gated_fwd_kernel1(x, g, y, w, b, residual, residual_out, mean, rstd, eps, D: tl.constexpr, BD: tl.constexpr, ACTIVATION: tl.constexpr, IS_RMS_NORM: tl.constexpr, STORE_RESIDUAL_OUT: tl.constexpr, HAS_RESIDUAL: tl.constexpr, HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    g += i_t * D
    if HAS_RESIDUAL:
        residual += i_t * D
    if STORE_RESIDUAL_OUT:
        residual_out += i_t * D
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    b_x = tl.load(x + o_d, mask=m_d, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        b_x += tl.load(residual + o_d, mask=m_d, other=0.0).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        tl.store(residual_out + o_d, b_x, mask=m_d)
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=0) / D
        tl.store(mean + i_t, b_mean)
        b_xbar = tl.where(m_d, b_x - b_mean, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    else:
        b_xbar = tl.where(m_d, b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    tl.store(rstd + i_t, b_rstd)
    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
    b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b
    b_g = tl.load(g + o_d, mask=m_d, other=0.0).to(tl.float32)
    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == 'sigmoid':
        b_y = b_y * tl.sigmoid(b_g)
    tl.store(y + o_d, b_y, mask=m_d)

def layer_norm_gated_fwd(x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, activation: str='swish', eps: float=1e-05, residual: torch.Tensor=None, out_dtype: torch.dtype=None, residual_dtype: torch.dtype=None, is_rms_norm: bool=False):
    if residual is not None:
        residual_dtype = residual.dtype
    T, D = x.shape
    if residual is not None:
        assert residual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (D,)
    if bias is not None:
        assert bias.shape == (D,)
    y = x if out_dtype is None else torch.empty_like(x, dtype=out_dtype)
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = torch.empty((T,), dtype=torch.float, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    if D <= 512:
        BT = 32
        layer_norm_gated_fwd_kernel[cdiv(T, BT),](x=x, g=g, y=y, w=weight, b=bias, residual=residual, residual_out=residual_out, mean=mean, rstd=rstd, eps=eps, T=T, D=D, BD=BD, BT=BT, ACTIVATION=activation, IS_RMS_NORM=is_rms_norm, num_warps=4)
    else:
        layer_norm_gated_fwd_kernel1[T,](x=x, g=g, y=y, w=weight, b=bias, residual=residual, residual_out=residual_out, mean=mean, rstd=rstd, eps=eps, D=D, BD=BD, ACTIVATION=activation, IS_RMS_NORM=is_rms_norm, num_warps=4)
    return (y, mean, rstd, residual_out if residual_out is not None else x)


# ── Phase A' synthesised standalone entrypoint (HOLE B) ─────────────
# The staged candidate MUST expose run(inputs)->dict so S2 can compare
# it against the UNMODIFIED upstream (which exposes the positional
# entrypoint 'layer_norm_gated_fwd', NOT run). This wrapper is a PURE dict->positional
# argument shuffle of the EXISTING upstream entrypoint -- it never
# reimplements numeric logic (changing values = an A' failure). The
# original side runs the same upstream callable via the identical
# signature adapter in _pa_subproc_runner._signature_adapter, so the
# two manifests are structurally comparable. Promotion of this file is
# gated on a genuine S2-gpu-authoritative PASS (never fabricated).
def run(inputs: dict) -> dict:  # noqa: D401
    """Standalone S2 entrypoint -- maps `inputs` onto the upstream
    `layer_norm_gated_fwd` signature STRICTLY BY NAME (never a silent
    skip+positional-shift; a required no-default param with no value is
    a hard BLOCKED(shapes-incomplete), NOT a mis-call), returns a dict
    (mutated args surfaced when the upstream entrypoint mutates in
    place and returns None). This mirrors
    _pa_subproc_runner._signature_adapter EXACTLY so the original and
    severed sides bind identically."""
    import inspect

    import torch as _torch

    _MARK = "BLOCKED(shapes-incomplete"
    _fn = layer_norm_gated_fwd
    _PNM = {'weight': 'w', 'bias': 'b'}  # meta.interface.param_name_map (wrapper param -> task key)
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
                    + " for entrypoint " + repr('layer_norm_gated_fwd')
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
