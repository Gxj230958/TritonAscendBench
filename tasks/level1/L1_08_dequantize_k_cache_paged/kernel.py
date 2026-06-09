"""Task L1_08_dequantize_k_cache_paged_kernel — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_dequantize_k_cache_paged_kernel`
Source path: `sglang/srt/layers/attention/nsa/dequant_k_cache.py`
Upstream repo: SGLang
Commit SHA: 409d350fb6f6a1e7c7546e39028f811092a8e489
File SHA-256 (this vendored copy): 2106420f2a25b69481596b1f2e79ab1043462244db46810d451d35d176cd02bc
"""
import torch
import triton
import triton.language as tl

def dequantize_k_cache_paged(quant_k_cache: torch.Tensor, page_table_1_flattened: torch.Tensor, group_size: int=128) -> torch.Tensor:
    """
    De-quantize the k-cache with paged layout
    Args:
        quant_k_cache: [total_num_tokens, 1, dim_quant] or [num_blocks, block_size, 1, dim_quant], the quantized k-cache in paged layout
        page_table_1_flattened: [num_tokens], the flattened page_table_1 with the page indices in each requests concatenated together
    Returns:
        output: [num_tokens, 1, dim_nope + dim_rope], the de-quantized k-cache
    """
    dim_quant = quant_k_cache.shape[-1]
    assert dim_quant == 656, f'dim_quant: {dim_quant} != 656 detected in dequantize_k_cache_paged'
    quant_k_cache = quant_k_cache.view((-1, dim_quant))
    num_tokens = page_table_1_flattened.shape[0]
    assert quant_k_cache.dtype == torch.float8_e4m3fn
    dim_nope = 512
    dim_rope = 64
    num_tiles = dim_nope // group_size
    output = torch.empty((num_tokens, 1, dim_nope + dim_rope), dtype=torch.bfloat16, device=quant_k_cache.device)
    num_blocks_per_token = triton.cdiv(dim_nope + dim_rope, group_size)
    assert num_blocks_per_token == 5
    assert dim_nope % group_size == 0
    input_nope_q = quant_k_cache[:, :dim_nope]
    input_nope_s = quant_k_cache[:, dim_nope:dim_nope + num_tiles * 4].view(torch.float32)
    input_rope = quant_k_cache[:, dim_nope + num_tiles * 4:].view(torch.bfloat16)
    _dequantize_k_cache_paged_kernel[num_tokens, num_blocks_per_token](output, input_nope_q, input_nope_s, input_rope, page_table_1_flattened, output.stride(0), input_nope_q.stride(0), input_nope_s.stride(0), input_rope.stride(0), NUM_NOPE_BLOCKS=num_tiles, GROUP_SIZE=group_size, DIM_NOPE=dim_nope, DIM_ROPE=dim_rope)
    return output

@triton.jit
def _dequantize_k_cache_paged_kernel(output_ptr, input_nope_q_ptr, input_nope_s_ptr, input_rope_ptr, page_table_1_ptr, output_stride_0: int, input_nope_q_stride_0: int, input_nope_s_stride_0: int, input_rope_stride_0: int, NUM_NOPE_BLOCKS: tl.constexpr, GROUP_SIZE: tl.constexpr, DIM_NOPE: tl.constexpr, DIM_ROPE: tl.constexpr):
    token_id = tl.program_id(0)
    token_id_paged = tl.load(page_table_1_ptr + token_id).to(tl.int32)
    raw_block_id = tl.program_id(1)
    if raw_block_id < NUM_NOPE_BLOCKS:
        effective_block_id = raw_block_id
        offs_q = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs_q < DIM_NOPE
        ptr_q = input_nope_q_ptr + token_id_paged * input_nope_q_stride_0 + offs_q
        ptr_s = input_nope_s_ptr + token_id_paged * input_nope_s_stride_0 + effective_block_id
        y_q = tl.load(ptr_q, mask=mask, other=0.0).to(tl.float32)
        y_s = tl.load(ptr_s)
        y = (y_q * y_s).to(output_ptr.dtype.element_ty)
        dst_ptr = output_ptr + token_id * output_stride_0 + offs_q
        tl.store(dst_ptr, y, mask=mask)
    else:
        effective_block_id = raw_block_id - NUM_NOPE_BLOCKS
        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE
        src_ptr = input_rope_ptr + token_id_paged * input_rope_stride_0 + offs
        dst_ptr = output_ptr + token_id * output_stride_0 + DIM_NOPE + offs
        data = tl.load(src_ptr, mask=mask).to(tl.bfloat16)
        tl.store(dst_ptr, data, mask=mask)
if __name__ == '__main__':
    raise Exception('UT is in quant_k_cache.py')


# ── Phase A' synthesised standalone entrypoint (HOLE B) ─────────────
# The staged candidate MUST expose run(inputs)->dict so S2 can compare
# it against the UNMODIFIED upstream (which exposes the positional
# entrypoint 'dequantize_k_cache_paged', NOT run). This wrapper is a PURE dict->positional
# argument shuffle of the EXISTING upstream entrypoint -- it never
# reimplements numeric logic (changing values = an A' failure). The
# original side runs the same upstream callable via the identical
# signature adapter in _pa_subproc_runner._signature_adapter, so the
# two manifests are structurally comparable. Promotion of this file is
# gated on a genuine S2-gpu-authoritative PASS (never fabricated).
def run(inputs: dict) -> dict:  # noqa: D401
    """Standalone S2 entrypoint -- maps `inputs` onto the upstream
    `dequantize_k_cache_paged` signature STRICTLY BY NAME (never a silent
    skip+positional-shift; a required no-default param with no value is
    a hard BLOCKED(shapes-incomplete), NOT a mis-call), returns a dict
    (mutated args surfaced when the upstream entrypoint mutates in
    place and returns None). This mirrors
    _pa_subproc_runner._signature_adapter EXACTLY so the original and
    severed sides bind identically."""
    import inspect

    import torch as _torch

    _MARK = "BLOCKED(shapes-incomplete"
    _fn = dequantize_k_cache_paged
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
                    + " for entrypoint " + repr('dequantize_k_cache_paged')
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
