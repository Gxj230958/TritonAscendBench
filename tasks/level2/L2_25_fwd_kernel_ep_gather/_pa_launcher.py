import torch
import triton


# Authored launcher: drives the in-module @triton.jit `_fwd_kernel_ep_gather`
# exactly as the upstream host wrapper `ep_gather` does (vLLM
# vllm/model_executor/layers/fused_moe/deep_gemm_utils.py, commit 18f6bf5;
# ported from lightllm deepep_scatter_gather.py), specialised to the
# expert_map=None / HAS_EXPERT_MAP=False path that shapes.yaml supplies.
#
# Upstream `ep_gather` grid / launch (verbatim semantics, expert_map=None):
#   num_tokens  = output_tensor.shape[0]          # = T
#   hidden_size = input_tensor.shape[1]           # = D
#   BLOCK_D     = min(hidden_size, 1024)
#   assert hidden_size % BLOCK_D == 0
#   grid = (cdiv(hidden_size, BLOCK_D), min(num_tokens, 1024))
#   num_warps = 2
# All strides are passed straight from .stride() of the row-major inputs.
#
# Math the kernel realises (per output token t, feature d):
#   out[t, d] = sum_k 1[recv_topk_ids[t,k] >= 0]
#                     * recv_topk_weight[t, k]
#                     * input_tensor[input_index[t, k], d]
# accumulated in fp32, cast to output_tensor.dtype. The shapes.yaml output
# buffer (and oracle) dtype matches input_tensor (float32).
#
# BLOCK_D tiling: the kernel reads/writes exactly [cur_block*BLOCK_D + off_d]
# for off_d in tl.arange(0, BLOCK_D) with NO intra-block D masking, and the
# fp32 accumulation runs over the topk axis (independent of the D-tiling). So
# ANY BLOCK_D that divides hidden_size (the kernel's own `assert hidden_size %
# BLOCK_D == 0`) drives the kernel bit-identically -- the D-axis is just a
# contiguous copy partitioned across grid dim 0; output equals the oracle
# regardless of how D is tiled.
#
# Triton requires the `tl.arange(0, BLOCK_D)` range to be a power of 2
# (semantic.arange, enforced at trace time on EVERY backend in triton 3.6.0,
# not just TRITON_INTERPRET). Upstream's `BLOCK_D = min(hidden_size, 1024)` is a
# power of 2 only when hidden_size is; for a hidden_size with a non-power-of-2
# factor (e.g. 384 = 2^7*3) that literal would not compile. We therefore tile
# with the LARGEST power-of-2 that divides hidden_size and is <= min(hidden_size,
# 1024) -- the necessary AND bit-exact way to drive this kernel for any
# hidden_size (verified: 0.0 diff vs the oracle on every shape incl. D=384, and
# different pow-2 divisors agree). For every power-of-2 hidden_size this equals
# upstream's BLOCK_D verbatim; only D=384 (H1) differs (384 -> 128).
#
# Output key: "output" (the oracle contract key).
def _interpret_safe_block_d(hidden_size: int) -> int:
    cap = min(hidden_size, 1024)
    block_d = 1
    p = 1
    while p <= cap:
        if hidden_size % p == 0:
            block_d = p
        p *= 2
    return block_d


def run(inputs: dict) -> dict:
    input_tensor = inputs["input_tensor"]          # [B(=M), D] float32
    recv_topk_ids = inputs["recv_topk_ids"]        # [T, K] int32 (signed)
    recv_topk_weight = inputs["recv_topk_weight"]  # [T, K] float32
    input_index = inputs["input_index"]            # [T, K] int32 in [0, B)

    num_tokens, topk_num = recv_topk_ids.shape
    hidden_size = input_tensor.shape[1]

    output_tensor = torch.zeros(
        (num_tokens, hidden_size),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )

    expert_map = None
    num_warps = 2
    BLOCK_D = _interpret_safe_block_d(hidden_size)
    assert hidden_size % BLOCK_D == 0
    grid = (triton.cdiv(hidden_size, BLOCK_D), min(num_tokens, 1024))

    _fwd_kernel_ep_gather[grid](  # noqa: F821 (injected into kernel.py module)
        num_tokens,
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        recv_topk_ids.stride(1),
        recv_topk_weight,
        recv_topk_weight.stride(0),
        recv_topk_weight.stride(1),
        input_index,
        input_index.stride(0),
        input_index.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        topk_num=topk_num,
        expert_map=expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
    )

    return {"output": output_tensor}
