import torch
import triton


# Authored launcher: drives the in-module @triton.jit `_fwd_kernel_alibi`
# (vLLM vllm/v1/attention/ops/prefix_prefill.py, commit
# 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19) exactly as the upstream host
# wrapper `context_attention_fwd` does on its ALiBi branch, but for the
# single-sequence, no-prefix-context case the oracle models.
#
# Why a launcher rather than binding the host wrapper directly: the wrapper
# signature is `context_attention_fwd(q, k, v, o, kv_cache_dtype, k_cache,
# v_cache, b_loc, b_start_loc, b_seq_len, max_seq_len, max_input_len,
# k_scale, v_scale, ...)`. It requires a paged KV cache (k_cache/v_cache,
# b_loc block table, kv_cache_dtype) that shapes.yaml does not — and should
# not — provide: the oracle is plain causal ALiBi SDPA over a single
# sequence with NO prefix context. So we drive the @triton.jit directly,
# reproducing the wrapper's ALiBi-branch argument marshalling with
# cur_batch_ctx_len == 0 (empty prefix). With an empty prefix the kernel's
# first loop `for start_n in range(0, cur_batch_ctx_len, BLOCK_N)` does not
# execute, so K_cache/V_cache/B_Loc/k_scale/v_scale/block_size/x are never
# dereferenced; only the second loop (query-against-itself causal scan over
# the flat K/V) runs, computing the full causal lower triangle.
#
# Layout reconciliation (oracle head-major -> kernel token-major):
#   The kernel addresses q/k/v as [num_tokens, num_heads, head_dim]
#   (token-major): off_q = token*stride_qbs + head*stride_qh + d*stride_qd,
#   and K is loaded transposed (offs_d over stride_kd as the [D] row, offs_n
#   over stride_kbs as the [N] contraction column) so q @ k contracts the
#   head_dim. The oracle provides [H, L, D] (head-major), so we transpose to
#   [L, H, D] and pass the contiguous token-major strides. The output is
#   written token-major [L, H, D] and transposed back to [H, L, D].
#
# Numeric faithfulness: the request-loop dots use input_precision="ieee"
# (hardcoded in the kernel), the ALiBi bias (k_pos - q_pos)*slope masked to
# the causal triangle reproduces the oracle's -slope*(i-j) with j>i masked,
# and BLOCK_M == BLOCK_N so the causal upper bound (start_m+1)*BLOCK_M aligns
# with the BLOCK_N key stepping (the diagonal block is the last one visited).
# acc / l_i is exact (no +eps): every causal row attends to at least its
# diagonal, so l_i >= 1.
#
# Output key (output) matches the oracle contract.
def run(inputs: dict) -> dict:
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    alibi_slopes = inputs["alibi_slopes"].contiguous()
    scale = float(inputs["scale"])

    H, L, D = q.shape
    device = q.device
    dtype = q.dtype

    # Head-major [H, L, D] -> token-major [L, H, D] (upstream
    # [num_tokens, num_heads, head_dim]).
    qt = q.transpose(0, 1).contiguous()
    kt = k.transpose(0, 1).contiguous()
    vt = v.transpose(0, 1).contiguous()
    out = torch.empty_like(qt)

    # Single sequence, no prefix: query_len = L, ctx_len = 0.
    b_start_loc = torch.tensor([0, L], dtype=torch.int32, device=device)
    b_seq_len = torch.tensor([L], dtype=torch.int32, device=device)

    # Tiling. Upstream uses BASE_BLOCK//2 (=64) for fp32 q; BLOCK_M == BLOCK_N
    # is required so the causal request loop's upper bound
    # (start_m + 1) * BLOCK_M aligns with the BLOCK_N key stepping.
    BLOCK = 64
    BLOCK_DMODEL = D
    BLOCK_DMODEL_PADDED = triton.next_power_of_2(D)

    # Dummy paged-cache operands: the ctx_len == 0 prefix loop never executes,
    # so these are never dereferenced. Shapes only need to satisfy stride
    # extraction (5D k_cache, 4D v_cache).
    dummy_kc = torch.empty((1, 1, 1, 1, 1), dtype=dtype, device=device)
    dummy_vc = torch.empty((1, 1, 1, 1), dtype=dtype, device=device)
    dummy_bloc = torch.zeros((1, 1), dtype=torch.int32, device=device)
    dummy_scale = torch.ones((1,), dtype=torch.float32, device=device)

    grid = (1, H, triton.cdiv(L, BLOCK))

    _fwd_kernel_alibi[grid](
        qt,
        kt,
        vt,
        dummy_kc,
        dummy_vc,
        dummy_bloc,
        scale,
        dummy_scale,
        dummy_scale,
        b_start_loc,
        b_seq_len,
        alibi_slopes,
        1,  # block_size (paged-cache only; unused at ctx_len == 0)
        1,  # x (paged-cache only; unused at ctx_len == 0)
        out,
        dummy_bloc.stride(0),
        dummy_bloc.stride(1),
        qt.stride(0),
        qt.stride(1),
        qt.stride(2),
        kt.stride(0),
        kt.stride(1),
        kt.stride(2),
        vt.stride(0),
        vt.stride(1),
        vt.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        dummy_kc.stride(0),
        dummy_kc.stride(1),
        dummy_kc.stride(2),
        dummy_kc.stride(3),
        dummy_kc.stride(4),
        dummy_vc.stride(0),
        dummy_vc.stride(1),
        dummy_vc.stride(2),
        dummy_vc.stride(3),
        num_queries_per_kv=1,
        IN_PRECISION=None,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DMODEL_PADDED=BLOCK_DMODEL_PADDED,
        BLOCK_N=BLOCK,
        SKIP_DECODE=False,
        # Match upstream `context_attention_fwd` codegen launch params
        # (kernel.py: num_warps=NUM_WARPS, num_stages=1). NUM_WARPS is the
        # module-level constant (8 on CUDA, 4 on ROCm); it is in scope here
        # because the launcher is exec'd into the kernel module namespace.
        num_warps=NUM_WARPS,
        num_stages=1,
    )

    # Token-major [L, H, D] -> head-major [H, L, D] (oracle layout).
    output = out.transpose(0, 1).contiguous()
    return {"output": output}
