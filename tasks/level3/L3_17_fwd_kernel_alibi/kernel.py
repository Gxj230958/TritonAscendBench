"""Task L3_17_fwd_kernel_alibi — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_fwd_kernel_alibi`
Source path: `vllm/v1/attention/ops/prefix_prefill.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): 9d100dab54dfacba94e9ee33154712703523980485b517c636e4f83ebc507410
Frozen by: scripts/phase_a_freeze.py
"""
from typing import Any
import torch
from ascendbench.frozen.shim import current_platform  # Phase A': platform stub (B-inert-shim)
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
NUM_WARPS = 4 if current_platform.is_rocm() else 8

@triton.jit
def _fwd_kernel_alibi(Q, K, V, K_cache, V_cache, B_Loc, sm_scale, k_scale, v_scale, B_Start_Loc, B_Seqlen, Alibi_slopes, block_size, x, Out, stride_b_loc_b, stride_b_loc_s, stride_qbs, stride_qh, stride_qd, stride_kbs, stride_kh, stride_kd, stride_vbs, stride_vh, stride_vd, stride_obs, stride_oh, stride_od, stride_k_cache_bs, stride_k_cache_h, stride_k_cache_d, stride_k_cache_bl, stride_k_cache_x, stride_v_cache_bs, stride_v_cache_h, stride_v_cache_d, stride_v_cache_bl, num_queries_per_kv: int, IN_PRECISION: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_DMODEL_PADDED: tl.constexpr, BLOCK_N: tl.constexpr, SKIP_DECODE: tl.constexpr):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)
    cur_kv_head = cur_head // num_queries_per_kv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len
    if SKIP_DECODE and cur_batch_query_len == 1:
        return
    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    dim_mask = tl.where(tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1, 0).to(tl.int1)
    q = tl.load(Q + off_q, mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len), other=0.0)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)
    alibi_slope = tl.load(Alibi_slopes + cur_head)
    alibi_start_q = tl.arange(0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
    alibi_start_k = 0
    for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        bn = tl.load(B_Loc + cur_batch * stride_b_loc_b + (start_n + offs_n) // block_size * stride_b_loc_s, mask=start_n + offs_n < cur_batch_ctx_len, other=0).to(tl.int64)
        off_k = bn[None, :] * stride_k_cache_bs + cur_kv_head * stride_k_cache_h + offs_d[:, None] // x * stride_k_cache_d + (start_n + offs_n[None, :]) % block_size * stride_k_cache_bl + offs_d[:, None] % x * stride_k_cache_x
        off_v = bn[:, None] * stride_v_cache_bs + cur_kv_head * stride_v_cache_h + offs_d[None, :] * stride_v_cache_d + (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl
        k_load = tl.load(K_cache + off_k, mask=dim_mask[:, None] & (start_n + offs_n[None, :] < cur_batch_ctx_len), other=0.0)
        if k_load.dtype.is_fp8():
            k = (k_load.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
        else:
            k = k_load
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
        qk = tl.where(start_n + offs_n[None, :] < cur_batch_ctx_len, qk, float('-inf'))
        qk *= sm_scale
        alibi = (tl.arange(0, BLOCK_N)[None, :] + alibi_start_k - alibi_start_q[:, None]) * alibi_slope
        alibi = tl.where((alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len), alibi, float('-inf'))
        qk += alibi
        alibi_start_k += BLOCK_N
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.math.exp(qk - m_i_new[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij
        acc_scale = alpha
        acc = acc * acc_scale[:, None]
        v_load = tl.load(V_cache + off_v, mask=dim_mask[None, :] & (start_n + offs_n[:, None] < cur_batch_ctx_len), other=0.0)
        if v_load.dtype.is_fp8():
            v = (v_load.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
        else:
            v = v_load
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc=acc, input_precision='ieee')
        l_i = l_i_new
        m_i = m_i_new
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    block_mask = tl.where(block_start_loc < cur_batch_seq_len - cur_batch_ctx_len, 1, 0)
    alibi_slope = tl.load(Alibi_slopes + cur_head)
    alibi_start_q = tl.arange(0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
    alibi_start_k = cur_batch_ctx_len
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs, mask=dim_mask[:, None] & (start_n + offs_n[None, :] < cur_batch_seq_len - cur_batch_ctx_len), other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision='ieee')
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= start_n + offs_n[None, :], qk, float('-inf'))
        alibi = (tl.arange(0, BLOCK_N)[None, :] + alibi_start_k - alibi_start_q[:, None]) * alibi_slope
        alibi = tl.where((alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len), alibi, float('-inf'))
        qk += alibi
        alibi_start_k += BLOCK_N
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.math.exp(qk - m_i_new[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij
        acc_scale = alpha
        acc = acc * acc_scale[:, None]
        v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs, mask=dim_mask[None, :] & (start_n + offs_n[:, None] < cur_batch_seq_len - cur_batch_ctx_len), other=0.0)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc=acc, input_precision='ieee')
        l_i = l_i_new
        m_i = m_i_new
    acc = acc / l_i[:, None]
    off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len))
    return

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
