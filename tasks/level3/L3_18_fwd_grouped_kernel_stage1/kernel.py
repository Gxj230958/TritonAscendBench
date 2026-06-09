"""Task L3_18_fwd_grouped_kernel_stage1 — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_fwd_grouped_kernel_stage1`
Source path: `vllm/v1/attention/ops/triton_decode_attention.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): 0d0a54a4b57e0df2112d8f09d3801ad6fbfad2bfbce37a7c6428c7cb0930b040
"""
'\nMemory-efficient attention for decoding.\nIt supports page size >= 1.\n'
import logging
import torch
from packaging import version
from ascendbench.frozen.shim import current_platform  # Phase A': platform stub (B-inert-shim)
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
logger = logging.getLogger(__name__)
if version.parse(triton.__version__) < version.parse('3.2.0'):
    logger.warning("The following error message 'operation scheduled before its operands' can be ignored.")

@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def _fwd_grouped_kernel_stage1(Q, K_Buffer, V_Buffer, sm_scale, Req_to_tokens, B_Seqlen, Att_Out, stride_req_to_tokens_b, stride_qbs, stride_qh, stride_buf_kbs, stride_buf_kh, stride_buf_vbs, stride_buf_vh, stride_mid_ob, stride_mid_oh, stride_mid_os, k_scale, v_scale, kv_group_num: tl.constexpr, q_head_num: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_DPE: tl.constexpr, BLOCK_DV: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr, NUM_KV_SPLITS: tl.constexpr, PAGE_SIZE: tl.constexpr, logit_cap: tl.constexpr, Lk: tl.constexpr, Lv: tl.constexpr, IS_MLA: tl.constexpr=False):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)
    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = cur_batch
    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=mask_h[:, None] & mask_d[None, :], other=0.0, cache_modifier='.ca')
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        qpe = tl.load(Q + off_qpe, mask=mask_h[:, None] & mask_dpe[None, :], other=0.0, cache_modifier='.ca')
    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float('inf')
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)
    if split_kv_end > split_kv_start:
        base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
        base_offs_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]
        if BLOCK_DPE > 0:
            base_offs_kpe = cur_kv_head * stride_buf_kh + offs_dpe[:, None]
        ks = tl.load(k_scale)
        vs = tl.load(v_scale)
        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_page_number = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n // PAGE_SIZE, mask=offs_n < split_kv_end, other=0, cache_modifier='.ca')
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
            offs_buf_k = kv_loc[None, :] * stride_buf_kbs + base_offs_k
            k = tl.load(K_Buffer + offs_buf_k, mask=(offs_n[None, :] < split_kv_end) & mask_d[:, None], other=0.0, cache_modifier='.cg')
            if k.dtype.is_fp8():
                k = (k.to(tl.float32) * ks).to(q.dtype)
            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = kv_loc[None, :] * stride_buf_kbs + base_offs_kpe
                kpe = tl.load(K_Buffer + offs_buf_kpe, mask=(offs_n[None, :] < split_kv_end) & mask_dpe[:, None], other=0.0, cache_modifier='.cg')
                if kpe.dtype.is_fp8():
                    kpe = (kpe.to(tl.float32) * ks).to(qpe.dtype)
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)
            qk = tl.where(mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float('-inf'))
            if not IS_MLA:
                offs_buf_v = kv_loc[:, None] * stride_buf_vbs + base_offs_v
                v = tl.load(V_Buffer + offs_buf_v, mask=(offs_n[:, None] < split_kv_end) & mask_dv[None, :], other=0.0)
                if v.dtype.is_fp8():
                    v = (v.to(tl.float32) * vs).to(q.dtype)
            else:
                v = tl.trans(k)
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max
        offs_mid_o = cur_batch * stride_mid_ob + cur_head[:, None] * stride_mid_oh + split_kv_id * stride_mid_os + offs_dv[None, :]
        tl.store(Att_Out + offs_mid_o, acc / e_sum[:, None], mask=mask_h[:, None] & mask_dv[None, :])
        offs_mid_o_1 = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + split_kv_id * stride_mid_os + Lv
        tl.store(Att_Out + offs_mid_o_1, e_max + tl.log(e_sum), mask=mask_h)

import torch
import triton


# Authored launcher: drives the frozen in-module @triton.jit
# `_fwd_grouped_kernel_stage1` EXACTLY as the upstream host wrapper
# `_decode_grouped_att_m_fwd` does on its non-MLA path (vLLM
# vllm/v1/attention/ops/triton_decode_attention.py, commit
# 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19). The grid, strides and
# constexprs below are copied verbatim from that wrapper (kernel.py:459);
# the launcher never reimplements the attention numerics -- it only
# marshals the dense fixtures into the wrapper's tensor contract and
# launches the graded kernel.
#
# Why a launcher rather than binding `_decode_grouped_att_m_fwd` directly:
# the wrapper takes the control tensors Req_to_tokens / B_Seqlen / att_out
# plus the scalar controls num_kv_splits / page_size / logit_cap /
# k_scale / v_scale. The strict by-name signature adapter rejects it as
# shapes-incomplete (those have no defaults and are not plain shapes.yaml
# tensors). We supply them from the dense fixtures and pin the wrapper's
# non-MLA, page_size=1, logit_cap=0, fp32 (unit k/v scale) path -- the
# exact branch the oracle models -- launching `_fwd_grouped_kernel_stage1`
# with the wrapper's own grid/stride/constexpr logic.
#
# Tensor contract (non-MLA grouped decode stage-1):
#   q            : [B, H, D]            (one decode query token per request)
#   k_buffer     : [B*C, H_KV, D]       (paged latent KV, kv_group_num=H/H_KV)
#   v_buffer     : [B*C, H_KV, Dv]
#   req_to_tokens: [B*C] -> reshaped [B, C] page table (page_size=1, so
#                  kv_loc == req_to_tokens[b, n]); a non-identity permutation
#                  so the gather is load-bearing.
#   b_seqlen     : [B] per-request cached length (sets the split bounds).
#   att_out      : [B, H, NUM_KV_SPLITS, Dv+1]; [...,:Dv] = per-split context
#                  (acc/e_sum), [..., Dv] = per-split lse (e_max+log(e_sum)).
#                  Zero-initialised so empty splits (start>=end, which the
#                  kernel never writes) match the oracle's zero partials.
#
# Output key (att_out) matches the oracle contract exactly.
def run(inputs: dict) -> dict:
    q = inputs["q"].contiguous()
    k_buffer = inputs["k_buffer"].contiguous()
    v_buffer = inputs["v_buffer"].contiguous()
    scale = float(inputs["scale"])
    num_kv_splits = int(inputs["num_kv_splits"])

    B, head_num, D = q.shape
    Dv = v_buffer.shape[-1]
    C = inputs["req_to_tokens"].numel() // B

    Req_to_tokens = (
        inputs["req_to_tokens"].to(torch.int64).reshape(B, C).contiguous()
    )
    B_Seqlen = inputs["b_seqlen"].to(torch.int64).contiguous()

    page_size = 1
    logit_cap = 0.0
    k_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)

    att_out = torch.zeros(
        (B, head_num, num_kv_splits, Dv + 1),
        dtype=torch.float32,
        device=q.device,
    )

    # ---- mirror _decode_grouped_att_m_fwd (non-MLA), kernel.py:459-559 ----
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]
    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)
    BLOCK = 32
    kv_group_num = q.shape[1] // k_buffer.shape[-2]
    BLOCK_H = 16
    NUM_KV_SPLITS = num_kv_splits
    grid = (
        B,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_KV_SPLITS,
    )
    num_stages = 1 if BLOCK_DMODEL >= 1024 else 2

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        scale,
        Req_to_tokens,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(-3),
        k_buffer.stride(-2),
        v_buffer.stride(-3),
        v_buffer.stride(-2),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        k_scale,
        v_scale,
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        IS_MLA=False,
    )
    return {"att_out": att_out}
