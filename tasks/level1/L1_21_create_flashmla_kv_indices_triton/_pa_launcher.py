import torch

# Authored jit-only launcher: drives the in-module @triton.jit
# `create_flashmla_kv_indices_triton` exactly as the upstream FlashMLA decode
# call site does (sglang/srt/layers/attention/utils.py). The kernel has no
# host wrapper in the module, so this run(inputs) reproduces the call site:
#
#   grid = (n_reqs,)
#   create_flashmla_kv_indices_triton[grid](
#       req_to_token,            # [max_batch, max_context_len]
#       req_pool_indices,        # [n_reqs]
#       page_kernel_lens,        # [n_reqs]  (seq_lens)
#       None,                    # kv_start_idx  (decode path passes null)
#       kv_indices,              # [n_reqs, kv_indices_stride]  (out, pre-zeroed)
#       req_to_token.stride(0),  # req_to_token_ptr_stride  (constexpr)
#       kv_indices.stride(0),    # kv_indices_ptr_stride     (constexpr)
#       PAGED_SIZE=page_size,    # page granularity          (constexpr)
#   )
#
# kv_start_idx=None -> the kernel's `if kv_start_idx:` is False, so kv_start=0
# and kv_end = page_kernel_lens[pid]; pages beyond num_paged are never stored,
# so the pre-zeroed buffer supplies the padding. Output key: "kv_indices".
def run(inputs: dict) -> dict:
    req_to_token = inputs["req_to_token"]
    req_pool_indices = inputs["req_pool_indices"]
    page_kernel_lens = inputs["page_kernel_lens"]
    page_size = int(inputs["page_size"])

    n_reqs = req_pool_indices.shape[0]
    max_ctx = req_to_token.shape[1]
    # kv_indices column count = max pages addressable in req_to_token width.
    kv_indices_stride = (max_ctx + page_size - 1) // page_size

    kv_indices = torch.zeros(
        (n_reqs, kv_indices_stride),
        dtype=torch.int32,
        device=req_to_token.device,
    )

    grid = (n_reqs,)
    create_flashmla_kv_indices_triton[grid](
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        None,  # kv_start_idx
        kv_indices,
        req_to_token.stride(0),
        kv_indices.stride(0),
        PAGED_SIZE=page_size,
    )

    return {"kv_indices": kv_indices}
