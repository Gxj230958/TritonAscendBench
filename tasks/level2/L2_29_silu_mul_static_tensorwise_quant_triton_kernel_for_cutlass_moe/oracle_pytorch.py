"""Oracle reference for
L2_29_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe.

Tier: upstream_oracle_required
Tolerance class: vector_reduce
Category: moe

Independent eager-PyTorch reference for the GRADED frozen upstream GPU triton
host wrapper ``silu_and_mul_masked_post_quant_fwd`` (kernel.py line 380), which
drives the ``@triton.jit`` kernel ``_silu_and_mul_post_quant_kernel`` (SGLang
``sglang/srt/layers/moe/ep_moe/kernels.py``, commit
409d350fb6f6a1e7c7546e39028f811092a8e489; copied from lightllm
``moe_silu_and_mul_mix_quant_ep.py``). This file authors the math from the
wrapper's documented contract; it never imports or calls the kernel.

Wrapper contract (kernel.py docstring, lines 388-394):

* ``input``        ``[expert_num, token_num_padded, hidden_dim]`` (gate || up
  along the last dim; the first half is gate, the second half is up).
* ``output``       ``[expert_num, token_num_padded, hidden_dim // 2]``,
  ``torch.float8_e4m3fn``.
* ``output_scale`` ``[expert_num, token_num_padded, (hidden_dim // 2) //
  quant_group_size]``, fp32.
* ``masked_m``     ``[expert_num]`` per-expert valid token count; tokens
  ``t >= masked_m[e]`` are never written by the kernel.

Kernel math (per expert e, per group block, per valid token t < masked_m[e]):
    gate    = silu(input[e, t, :size_n])      cast back to input dtype
    gate_up = up * gate    (up = input[e, t, size_n:])
    absmax  = max(max(|gate_up|), 1e-10)       # 1e-10 floors the AMAX
    out_s   = absmax / fp8_max
    out_q   = clamp(gate_up / out_s, -fp8_max, fp8_max) -> float8_e4m3fn
Tokens ``t >= masked_m[e]`` and their scale rows stay 0 (launcher zero-inits;
the oracle masks them to 0 so the full-tensor compare matches).
"""
from __future__ import annotations

import torch


def _silu(t: torch.Tensor) -> torch.Tensor:
    """SiLU from its definition x * sigmoid(x) = x / (1 + exp(-x))."""
    return t / (1.0 + torch.exp(-t))


def _silu_and_mul_masked_post_quant(
    x: torch.Tensor,
    masked_m: torch.Tensor,
    quant_group_size: int,
    fp8_max: float = 448.0,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked SiLU+mul followed by per-group symmetric FP8 quant.

    Args:
        x: ``[E, T, 2*D]`` gate||up per expert/token.
        masked_m: ``[E]`` per-expert valid token count.
        quant_group_size: group width along the D (== size_n) axis.
        fp8_max: positive saturation magnitude for float8_e4m3fn (448.0).
        eps: AMAX floor applied BEFORE dividing by fp8_max, exactly as the
            kernel (``_absmax = max(max(|gate_up|), 1e-10)``).

    Returns:
        ``(yq, ys)`` where ``yq`` is ``[E, T, D]`` float8_e4m3fn and ``ys`` is
        ``[E, T, D // quant_group_size]`` fp32. Rows ``t >= masked_m[e]`` are 0.
    """
    E, T, two_d = x.shape
    d = two_d // 2
    gate, up = x[..., :d], x[..., d:]
    y = _silu(gate.to(torch.float32)) * up.to(torch.float32)        # [E, T, D]

    tok = torch.arange(T, device=x.device).view(1, T)
    valid = tok < masked_m.to(torch.long).view(E, 1)                # [E, T]
    y = y * valid.unsqueeze(-1)

    g = quant_group_size
    yv = y.reshape(E, T, d // g, g)
    amax = yv.abs().amax(dim=-1, keepdim=True)
    amax = torch.maximum(
        amax, torch.tensor(eps, dtype=amax.dtype, device=amax.device)
    )
    s = amax / float(fp8_max)
    yq = (yv / s).clamp(-float(fp8_max), float(fp8_max))
    yq = yq.reshape(E, T, d).to(torch.float8_e4m3fn)
    s = s.squeeze(-1).reshape(E, T, d // g)

    # Tokens t >= masked_m[e] are never written by the kernel; the launcher
    # zero-inits both outputs. yq is already 0 there (y was masked above);
    # mask the scale rows to 0 too so invalid rows compare equal.
    s = s * valid.unsqueeze(-1)
    return yq, s


def run(inputs: dict) -> dict:
    yq, ys = _silu_and_mul_masked_post_quant(
        inputs["input"],
        inputs["masked_m"],
        int(inputs["quant_group_size"]),
        float(inputs.get("fp8_max", 448.0)),
    )
    return {"output": yq, "output_scale": ys}
