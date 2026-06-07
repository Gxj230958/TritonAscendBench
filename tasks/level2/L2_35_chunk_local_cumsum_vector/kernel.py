"""Task L2_35_chunk_local_cumsum_vector_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `chunk_local_cumsum_vector_kernel`
Source path: `vllm/model_executor/layers/fla/ops/cumsum.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): d818215669b451868823202ac0bcea77e7a9d75957254c7c1060a73a55dc39b6
Frozen by: scripts/phase_a_freeze.py
"""
import torch
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
from triton_ascend_bench.upstream_shims.triton_autotune_interp import make_triton_clone as _pa_make_triton_clone  # Phase A': interpreter-aware autotune fallback
triton = _pa_make_triton_clone()
from tasks._frozen_deps_pool.h_af43637ae818.fla_ops.index import prepare_chunk_indices  # Phase A': U2 FLA-ops package vendoring
from tasks._frozen_deps_pool.h_af43637ae818.fla_ops.utils import check_shared_mem, input_guard  # Phase A': U2 FLA-ops package vendoring
BS_LIST = [32, 64] if check_shared_mem() else [16, 32]

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BS': BS}, num_warps=num_warps) for BS in BS_LIST for num_warps in [2, 4, 8]], key=['B', 'H', 'S', 'BT', 'IS_VARLEN', 'REVERSE'])
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_vector_kernel(s, o, cu_seqlens, chunk_indices, T, B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr, REVERSE: tl.constexpr, IS_VARLEN: tl.constexpr, HEAD_FIRST: tl.constexpr):
    i_s, i_t, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_b, i_h = (i_bh // H, i_bh % H)
    if IS_VARLEN:
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
        T = eos - bos
    else:
        bos, eos = (i_b * T, i_b * T + T)
    o_i = tl.arange(0, BT)
    if REVERSE:
        m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
    else:
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + (bos * H + i_h * T) * S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + (bos * H + i_h * T) * S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    else:
        p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H * S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + (bos * H + i_h) * S, (T, S), (H * S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

def chunk_local_cumsum_vector(g: torch.Tensor, chunk_size: int, reverse: bool=False, cu_seqlens: torch.Tensor | None=None, chunk_indices: torch.Tensor | None=None, head_first: bool=False, output_dtype: torch.dtype | None=torch.float) -> torch.Tensor:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), 'chunk_size must be a power of 2'
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    BT = chunk_size
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g_org, g = (g, torch.empty_like(g, dtype=output_dtype or g.dtype))

    def grid(meta):
        return (triton.cdiv(meta['S'], meta['BS']), NT, B * H)
    chunk_local_cumsum_vector_kernel[grid](g_org, g, cu_seqlens, chunk_indices, T=T, B=B, H=H, S=S, BT=BT, HEAD_FIRST=head_first, REVERSE=reverse)
    return g


# ── Phase A' synthesised standalone entrypoint (HOLE B) ─────────────
# The staged candidate MUST expose run(inputs)->dict so S2 can compare
# it against the UNMODIFIED upstream (which exposes the positional
# entrypoint 'chunk_local_cumsum_vector', NOT run). This wrapper is a PURE dict->positional
# argument shuffle of the EXISTING upstream entrypoint -- it never
# reimplements numeric logic (changing values = an A' failure). The
# original side runs the same upstream callable via the identical
# signature adapter in _pa_subproc_runner._signature_adapter, so the
# two manifests are structurally comparable. Promotion of this file is
# gated on a genuine S2-gpu-authoritative PASS (never fabricated).
def run(inputs: dict) -> dict:  # noqa: D401
    """Standalone S2 entrypoint -- maps `inputs` onto the upstream
    `chunk_local_cumsum_vector` signature STRICTLY BY NAME (never a silent
    skip+positional-shift; a required no-default param with no value is
    a hard BLOCKED(shapes-incomplete), NOT a mis-call), returns a dict
    (mutated args surfaced when the upstream entrypoint mutates in
    place and returns None). This mirrors
    _pa_subproc_runner._signature_adapter EXACTLY so the original and
    severed sides bind identically."""
    import inspect

    import torch as _torch

    _MARK = "BLOCKED(shapes-incomplete"
    _fn = chunk_local_cumsum_vector
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
                    + " for entrypoint " + repr('chunk_local_cumsum_vector')
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
