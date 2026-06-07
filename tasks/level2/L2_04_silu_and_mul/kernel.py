"""Task L2_04_silu_and_mul — GPU triton reference

Source: sglang/srt/layers/elementwise.py
Upstream repo: sgl-project/sglang
Commit SHA: <frozen-snapshot>
File SHA-256: see meta.yaml.upstream.file_sha256

SiLU + multiply gating used in SwiGLU MLP:
  x = hidden_states  shape (bs, 2*hidden_dim) = [x1 | x3]
  out = silu(x1) * x3   shape (bs, hidden_dim)

This task locks the unquantized variant (quant_max=None). The quantization
branch in upstream raises NotImplementedError; the task replicates only the
pure silu*mul path.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def silu_and_mul_kernel(
    out_hidden_states_ptr,
    out_scales_ptr,
    hidden_states_ptr,
    quant_max: tl.constexpr,
    static_scale: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    input_start = pid * hidden_dim * 2
    output_start = pid * hidden_dim

    input1_offs = tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < hidden_dim
    input3_offs = hidden_dim + tl.arange(0, BLOCK_SIZE)
    output_offs = tl.arange(0, BLOCK_SIZE)

    x1 = tl.load(
        hidden_states_ptr + input_start + input1_offs, mask=mask, other=0.0
    ).to(tl.float32)
    x3 = tl.load(
        hidden_states_ptr + input_start + input3_offs, mask=mask, other=0.0
    ).to(tl.float32)

    silu_x1 = x1 * tl.sigmoid(x1)
    out = x3 * silu_x1.to(hidden_states_ptr.dtype.element_ty)

    if quant_max is not None:
        # Task locks the non-quantized variant.
        raise NotImplementedError()

    tl.store(out_hidden_states_ptr + output_start + output_offs, out, mask=mask)


def silu_and_mul_triton(hidden_states: torch.Tensor) -> torch.Tensor:
    """Host wrapper. Quantization path intentionally omitted."""
    bs, in_hidden_dim = hidden_states.shape
    hidden_dim = in_hidden_dim // 2
    assert hidden_dim * 2 == in_hidden_dim, "input hidden dim must be even"

    out_hidden_states = torch.empty(
        (bs, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    silu_and_mul_kernel[(bs,)](
        out_hidden_states,
        None,                              # out_scales_ptr
        hidden_states,
        quant_max=None,
        static_scale=False,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
    )
    return out_hidden_states


# ── Phase A' synthesised standalone entrypoint (HOLE B) ─────────────
# The staged candidate MUST expose run(inputs)->dict so S2 can compare
# it against the UNMODIFIED upstream (which exposes the positional
# entrypoint 'silu_and_mul_triton', NOT run). This wrapper is a PURE dict->positional
# argument shuffle of the EXISTING upstream entrypoint -- it never
# reimplements numeric logic (changing values = an A' failure). The
# original side runs the same upstream callable via the identical
# signature adapter in _pa_subproc_runner._signature_adapter, so the
# two manifests are structurally comparable. Promotion of this file is
# gated on a genuine S2-gpu-authoritative PASS (never fabricated).
def run(inputs: dict) -> dict:  # noqa: D401
    """Standalone S2 entrypoint -- maps `inputs` onto the upstream
    `silu_and_mul_triton` signature STRICTLY BY NAME (never a silent
    skip+positional-shift; a required no-default param with no value is
    a hard BLOCKED(shapes-incomplete), NOT a mis-call), returns a dict
    (mutated args surfaced when the upstream entrypoint mutates in
    place and returns None). This mirrors
    _pa_subproc_runner._signature_adapter EXACTLY so the original and
    severed sides bind identically."""
    import inspect

    import torch as _torch

    _MARK = "BLOCKED(shapes-incomplete"
    _fn = silu_and_mul_triton
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
                    + " for entrypoint " + repr('silu_and_mul_triton')
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
