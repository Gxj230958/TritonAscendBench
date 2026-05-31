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
