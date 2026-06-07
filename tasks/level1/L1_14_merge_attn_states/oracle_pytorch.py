"""Oracle reference for L1_14_merge_attn_states_kernel.

Tier: upstream_oracle_required
Tolerance class: attention_flash
Category: attention

Independent eager-PyTorch reference for the vLLM split-KV attention-state
merge (FlashAttention-2 / flash-decoding online-softmax recombination,
section 2.2 of https://www.arxiv.org/pdf/2501.01005). Authored from the
upstream kernel's documented contract; never copies or calls the kernel.

UPSTREAM CONTRACT (vllm/v1/attention/ops/triton_merge_attn_states.py,
commit 18f6bf5; cross-checked against the upstream naive reference
tests/kernels/attention/test_merge_attn_states.py::merge_attn_states_torch):

  output        : [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
  prefix_output : [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
  suffix_output : [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
  prefix_lse    : [NUM_HEADS,  NUM_TOKENS]   (natural-log sum-exp, fp32)
  suffix_lse    : [NUM_HEADS,  NUM_TOKENS]
  output_lse    : [NUM_HEADS,  NUM_TOKENS]

The LSE tensors are stored head-major / token-minor -- the TRANSPOSE of
the [token, head] output layout. The kernel indexes them as
`lse + head_idx * num_tokens + token_idx`, i.e. row `head_idx`, col
`token_idx` of an [NUM_HEADS, NUM_TOKENS] row-major buffer.

ORACLE-SCOPE (tier=upstream_oracle_required): covers the fp32 LSE merge
(USE_FP8=False, output_scale=None) over the full-context case
(prefill_tokens_with_context == num_tokens). NOT covered: USE_FP8 output
quantisation.

"""
from __future__ import annotations

import torch


def _merge_attn_states(
    prefix_output: torch.Tensor,   # [N, H, D]
    prefix_lse: torch.Tensor,      # [H, N]
    suffix_output: torch.Tensor,   # [N, H, D]
    suffix_lse: torch.Tensor,      # [H, N]
    prefill_tokens_with_context: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online-softmax LSE recombination of two partial attention states.

    Mirrors the upstream kernel math exactly:
      * inf -> -inf normalisation on both LSE tensors (FA2/FA3 0-len
        seqlen convention),
      * max-subtracted exp weights p_se, s_se with denom out_se,
      * per (token, head) scale p_se/out_se, s_se/out_se applied to the
        [N, H, D] outputs (scales transposed [H, N] -> [N, H, 1]),
      * tokens at index >= prefill_tokens_with_context copy suffix_output
        directly and take output_lse = suffix_lse,
      * output_lse = log(out_se) + max_lse, layout [H, N].
    """
    num_tokens = prefix_output.shape[0]
    if prefill_tokens_with_context is None:
        prefill_tokens_with_context = num_tokens

    p_lse = prefix_lse.to(torch.float32).clone()
    s_lse = suffix_lse.to(torch.float32).clone()

    # FA2 returns +inf for an empty split where FA3 returns -inf; the
    # kernel collapses +inf -> -inf for consistency.
    pos_inf = torch.tensor(float("inf"), device=p_lse.device)
    neg_inf = torch.tensor(float("-inf"), device=p_lse.device)
    p_lse = torch.where(p_lse == pos_inf, neg_inf, p_lse)
    s_lse = torch.where(s_lse == pos_inf, neg_inf, s_lse)

    max_lse = torch.maximum(p_lse, s_lse)          # [H, N]
    p_se = torch.exp(p_lse - max_lse)
    s_se = torch.exp(s_lse - max_lse)
    out_se = p_se + s_se

    out_lse = torch.log(out_se) + max_lse          # [H, N]

    p_scale = (p_se / out_se).transpose(0, 1).unsqueeze(2)  # [N, H, 1]
    s_scale = (s_se / out_se).transpose(0, 1).unsqueeze(2)  # [N, H, 1]

    po = prefix_output.to(torch.float32)
    so = suffix_output.to(torch.float32)

    # mask[token] == 1 for context tokens, 0 for no-context tokens.
    # A no-context token copies suffix_output directly (s_scale*0 + 1).
    mask = torch.ones((num_tokens, 1, 1), device=po.device, dtype=torch.float32)
    if prefill_tokens_with_context < num_tokens:
        mask[prefill_tokens_with_context:].zero_()

    out = po * p_scale * mask + so * (s_scale * mask + (1.0 - mask))

    if prefill_tokens_with_context < num_tokens:
        out_lse[:, prefill_tokens_with_context:] = (
            suffix_lse.to(torch.float32)[:, prefill_tokens_with_context:]
        )

    return out.to(prefix_output.dtype), out_lse.to(torch.float32)


def run(inputs: dict) -> dict:
    out, lse = _merge_attn_states(
        inputs["prefix_output"],
        inputs["prefix_lse"],
        inputs["suffix_output"],
        inputs["suffix_lse"],
        inputs.get("prefill_tokens_with_context"),
    )
    return {"output": out, "output_lse": lse}
