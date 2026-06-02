"""PyTorch oracle for SGLang MLA decode stage1
(`_fwd_grouped_kernel_stage1_rope`, rocm_mla_decode_rope).

Per (batch b, query head h) the kernel computes grouped MLA attention over the
batch's paged KV range and writes, for the single KV split (num_kv_splits == 1
so one split spans the whole sequence):

    att_out[b, h, 0, :C] = sum_n softmax(score)[n] * v_latent[n]   # context
    att_out[b, h, 0,  C] = max_n score[n] + log(sum_n exp(...))    # log-sum-exp

with, for kv token n in the batch's page,

    score[n] = sm_scale * (q_pe_roped . k_pe[n] + q_nope . kv_latent[n])

Q = [q_nope(C) ; q_pe(R)], K_Buffer row = [kv_latent(C) ; k_pe(R)], and the
kernel's V_buffer is the aliased latent slice K_Buffer[..., :C]. RoPE (neox)
rotates q_pe by positions[b]; the cached k_pe is already roped for every token
EXCEPT the last one in the page, which the kernel ropes on the fly with the same
(cos, sin) -- so the oracle ropes only that last token.

Mixed precision matches the kernel. cos_sin_cache is bf16 (the kernel keeps
`k_pe_last_token` at q.dtype == bf16 across `if LAST_SPLIT:`, so a wider cos/sin
would not even compile -- the rope multiply must stay bf16). Hence q_pe stays
bf16 through the rope, and every tl.dot here is a bf16 x bf16 tensor-core dot with
fp32 accumulation (no TF32): the rope dot, the nope dot, and the context dot
(`p.to(v.dtype)`). The denominator e_sum stays fp32.
"""
from __future__ import annotations

import torch


def _bf16(x: torch.Tensor) -> torch.Tensor:
    """Round an fp32 tensor through bf16 (the kernel's storage dtype)."""
    return x.to(torch.bfloat16).to(torch.float32)


def _neox_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Neox-style RoPE on the last dim, rounded to bf16. ``x`` is [..., R] and
    ``cos``/``sin`` are [R/2], all bf16-valued. rotate_half(x) = [-x[R/2:], x[:R/2]];
    the full cos/sin tile the half twice -- matching the kernel's
    offs_rotary = arange(R) % (R/2) and the sin offset of R/2. The kernel's rope
    multiply runs in bf16 (bf16 q_pe * bf16 cos_sin_cache), so the result is rounded
    to bf16."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    cos_f = torch.cat([cos, cos], dim=-1)
    sin_f = torch.cat([sin, sin], dim=-1)
    # Triton evaluates `x * cos + rot * sin` as bf16 ops: each product and the sum
    # round to bf16 (both operands bf16), so round op-by-op, not once at the end.
    return _bf16(_bf16(x * cos_f) + _bf16(rot * sin_f))


def run(inputs: dict) -> dict:
    q = inputs["q"]
    k_buffer = inputs["k_buffer"]
    kv_indptr = inputs["kv_indptr"].to(torch.int64)
    kv_indices = inputs["kv_indices"].to(torch.int64)
    cos_sin_cache = inputs["cos_sin_cache"].to(torch.float32)  # bf16 values
    positions = inputs["positions"].to(torch.int64)
    C = int(inputs["kv_lora_rank"])
    R = int(inputs["rotary_dim"])
    sm_scale = float(inputs["sm_scale"])
    logit_cap = float(inputs["logit_cap"])
    use_rope = bool(inputs["use_rope"])

    B, H, _ = q.shape
    half = R // 2
    att_out = torch.zeros((B, H, 1, C + 1), dtype=torch.float32)

    # k_buffer is [total_kv, 1, C+R]; production V aliases the latent K slice.
    kbuf_bf16 = k_buffer[:, 0, :]
    kbuf = kbuf_bf16.to(torch.float32)
    vbuf = kbuf_bf16[:, :C]

    for b in range(B):
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        seq = end - start
        if seq <= 0:
            continue
        loc = kv_indices[start:end]  # [seq]

        pos = int(positions[b].item())
        cos = cos_sin_cache[pos, :half]
        sin = cos_sin_cache[pos, half:R]

        q_nope = q[b, :, :C].to(torch.float32)           # [H, C] (bf16 values)
        q_pe = q[b, :, C:C + R].to(torch.float32)         # [H, R]
        if use_rope:
            q_pe = _neox_rope(q_pe, cos, sin)             # bf16-valued

        kv_latent = kbuf[loc, :C]                         # [seq, C] (bf16 values)
        k_pe = kbuf[loc, C:C + R].clone()                 # [seq, R]
        if use_rope:
            # Only the LAST token of the page is roped on the fly; the rest are
            # already roped in the cache and used as-is.
            k_pe[-1] = _neox_rope(k_pe[-1], cos, sin)

        # Every dot is bf16 x bf16 -> fp32 (values are bf16-exact here).
        rope_score = q_pe @ k_pe.t()                      # [H, seq]
        nope_score = q_nope @ kv_latent.t()               # [H, seq]
        score = sm_scale * (rope_score + nope_score)      # [H, seq]
        if logit_cap > 0:
            score = logit_cap * torch.tanh(score / logit_cap)

        # Online softmax with a single split == plain softmax (S <= BLOCK_N).
        m = score.max(dim=-1).values                      # [H]
        p = torch.exp(score - m[:, None])                 # [H, seq] fp32
        e_sum = p.sum(dim=-1)                             # [H] fp32

        v = vbuf[loc, :].to(torch.float32)                # [seq, C] (bf16 values)
        # context dot rounds p to bf16 (p.to(v.dtype)); e_sum stays fp32.
        acc = _bf16(p) @ v                                # [H, C]
        ctx = acc / e_sum[:, None]

        att_out[b, :, 0, :C] = ctx
        att_out[b, :, 0, C] = m + torch.log(e_sum)

    return {"att_out": att_out}
