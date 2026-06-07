import torch


# Authored launcher: drives the in-module 4-stage @triton.jit pipeline
# exactly as the upstream host wrapper `moe_align_block_size_triton` does
# (SGLang sglang/jit_kernel/tests/test_moe_align_block_size.py,
# commit 409d350fb6f6a1e7c7546e39028f811092a8e489 -- the same file frozen
# into kernel.py).
#
# Upstream host wrapper (kernel.py, verbatim launch sequence):
#     numel = topk_ids.numel()
#     grid = (num_experts,)
#     tokens_cnts = torch.zeros((num_experts + 1, num_experts), int32, dev)
#     cumsum = torch.zeros((num_experts + 1,), int32, dev)
#     tokens_per_thread = ceil_div(numel, num_experts)
#     moe_align_block_size_stage1[grid](topk_ids, tokens_cnts, num_experts,
#                                       numel, tokens_per_thread)
#     moe_align_block_size_stage2[grid](tokens_cnts, num_experts)
#     moe_align_block_size_stage3[(1,)](num_tokens_post_pad, tokens_cnts,
#                                       cumsum, num_experts, block_size)
#     moe_align_block_size_stage4[grid](topk_ids, sorted_token_ids,
#                                       expert_ids, tokens_cnts, cumsum,
#                                       num_experts, block_size, numel,
#                                       tokens_per_thread)
#
# The output buffers `sorted_token_ids` / `expert_ids` / `num_tokens_post_pad`
# are caller-allocated. The upstream test allocates them OVER-SIZED to the
# worst-case `max_num_tokens_padded` and then compares only the valid prefix
# (`sorted_ids[:num_tokens_post_pad]`, `expert_ids[:num_tokens_post_pad //
# block_size]`). The padded sentinel for `sorted_token_ids` is `numel`
# (upstream: `sorted_ids.fill_(topk_ids.numel())`).
#
# Output keys + the trimmed valid region match the oracle contract
# (oracle_pytorch.py::run) exactly:
#   sorted_token_ids   : [num_tokens_post_pad] int32, padding slots = numel
#   expert_ids         : [num_tokens_post_pad // block_size] int32
#   num_tokens_post_pad: [1] int32
def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def run(inputs: dict) -> dict:
    topk_ids = inputs["topk_ids"]
    num_experts = int(inputs["num_experts"])
    block_size = int(inputs["block_size"])

    # The kernel loads topk_ids as a flat integer pointer; keep it integral
    # and contiguous (upstream passes the [num_tokens, topk] tensor directly,
    # which is contiguous).
    if topk_ids.dtype not in (torch.int32, torch.int64):
        topk_ids = topk_ids.to(torch.int64)
    topk_ids = topk_ids.contiguous()
    device = topk_ids.device

    numel = topk_ids.numel()
    grid = (num_experts,)

    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts), dtype=torch.int32, device=device
    )
    cumsum = torch.zeros(
        (num_experts + 1,), dtype=torch.int32, device=device
    )
    tokens_per_thread = _ceil_div(numel, num_experts)

    # Worst-case caller allocation, mirroring the upstream test driver.
    max_num_tokens_padded = numel + (num_experts + 1) * (block_size - 1)
    if numel < num_experts + 1:
        max_num_tokens_padded = numel * block_size
    max_num_m_blocks = max_num_tokens_padded // block_size

    sorted_token_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=device
    )
    sorted_token_ids.fill_(numel)  # sentinel for unwritten padding slots
    expert_ids = torch.zeros(
        (max_num_m_blocks,), dtype=torch.int32, device=device
    )
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)

    moe_align_block_size_stage1[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
    )
    moe_align_block_size_stage2[grid](
        tokens_cnts,
        num_experts,
    )
    moe_align_block_size_stage3[(1,)](
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
    )
    moe_align_block_size_stage4[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
    )

    # Trim the over-allocated buffers to the logical region the oracle returns.
    npp = int(num_tokens_post_pad.item())
    n_blocks = npp // block_size
    return {
        "sorted_token_ids": sorted_token_ids[:npp],
        "expert_ids": expert_ids[:n_blocks],
        "num_tokens_post_pad": num_tokens_post_pad,
    }
