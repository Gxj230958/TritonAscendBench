"""Authored launcher for SGLang MLA decode stage1
(`_fwd_grouped_kernel_stage1_rope`).

Drives the in-module @triton.jit `_fwd_grouped_kernel_stage1_rope` exactly as the
upstream host wrapper `_decode_grouped_att_m_fwd_rope` does on the CUDA path (the
non-HIP branch: num_stages=2, no extra ROCm kwargs), then returns the stage1
mid-output `att_out` ([B, H, NUM_KV_SPLITS, kv_lora_rank + 1] fp32 -- normalized
context in [:kv_lora_rank], log-sum-exp in the last column). Stage2
(`_decode_softmax_reducev`) is NOT run: this task grades stage1 only.

Launching the jit directly (rather than calling the host wrapper) keeps the severed
standalone self-contained -- severance keeps the graded jit + its closure, so the
launcher must reproduce the wrapper's grid/stride/BLOCK setup itself, the same
convention the sibling launchers use. k_buffer is [total_kv, 1, kv_lora_rank +
qk_rope_head_dim] so kv_group_num = q.shape[1] // k_buffer.shape[1] resolves to H
(all query heads share the single latent KV head), and v_buffer is the aliased
latent slice k_buffer[:, 0, :kv_lora_rank], matching the DeepSeek MLA caller's
val_cache_buf = key_cache_buf[..., :kv_lora_rank].
"""
import triton
import torch

# `_fwd_grouped_kernel_stage1_rope` is a free name here: the runner execs this
# launcher into the frozen kernel.py module namespace (scripts/_pa_oracle_check_
# runner._load / _pa_subproc_runner), so the in-module jit is already in scope --
# the same convention as the sibling launchers (no `from kernel import`).


def run(inputs: dict) -> dict:
    q = inputs["q"]
    k_buffer = inputs["k_buffer"]
    # SGLang's Triton backend uses int32 indptr and int64 token indices.
    kv_indptr = inputs["kv_indptr"].to(torch.int32)
    kv_indices = inputs["kv_indices"].to(torch.int64)
    cos_sin_cache = inputs["cos_sin_cache"]
    positions = inputs["positions"]
    kv_lora_rank = int(inputs["kv_lora_rank"])
    rotary_dim = int(inputs["rotary_dim"])
    num_kv_splits = int(inputs["num_kv_splits"])
    sm_scale = float(inputs["sm_scale"])
    logit_cap = float(inputs["logit_cap"])
    use_rope = bool(inputs["use_rope"])
    is_neox_style = bool(inputs["is_neox_style"])

    qk_rope_head_dim = k_buffer.shape[-1] - kv_lora_rank
    batch, head_num = kv_indptr.shape[0] - 1, q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]
    v_buffer = k_buffer[:, 0, :kv_lora_rank]

    # Stage1 mid-output (attn_logits): fp32, matching the DeepSeek MLA caller.
    att_out = torch.empty(
        (batch, head_num, num_kv_splits, kv_lora_rank + 1),
        dtype=torch.float32,
        device=q.device,
    )
    # Fused-RoPE requires a non-null buffer for the roped last k_pe token
    # (allocated, not graded).
    k_pe_tokens_out = torch.empty(
        (batch, qk_rope_head_dim), dtype=q.dtype, device=q.device
    )

    BLOCK = 32
    BLOCK_C = triton.next_power_of_2(kv_lora_rank)
    BLOCK_R = triton.next_power_of_2(qk_rope_head_dim)
    BLOCK_H = 16
    NUM_KV_SPLITS = num_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_KV_SPLITS,
    )

    _fwd_grouped_kernel_stage1_rope[grid](
        q,
        k_buffer,
        v_buffer,
        cos_sin_cache,
        positions,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        k_pe_tokens_out,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        v_buffer.stride(0),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        k_pe_tokens_out.stride(0) if use_rope else 0,
        cos_sin_cache.stride(0) if use_rope else 0,
        positions.stride(0) if use_rope else 0,
        rotary_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_C=BLOCK_C,
        BLOCK_R=BLOCK_R,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        logit_cap=logit_cap,
        USE_ROPE=use_rope,
        IS_NEOX_STYLE=is_neox_style,
        num_warps=4,
        num_stages=2,
    )
    return {"att_out": att_out}
