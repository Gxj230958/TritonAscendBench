"""Task L1_16_silu_and_mul — GPU triton reference

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
