"""Task L2_25_fwd_kernel_ep_gather — frozen upstream GPU triton source.

Vendored verbatim from the upstream repository for benchmark use
under the Apache-2.0 license. The original license header (below)
is preserved; see NOTICE at the repository root for attribution.

Kernel function: `_fwd_kernel_ep_gather`
Source path: `vllm/model_executor/layers/fused_moe/deep_gemm_utils.py`
Upstream repo: vLLM
Commit SHA: 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19
File SHA-256 (this vendored copy): 502a66386d0be6673d2f582daca0c10e56e2e8c4a0b94cb847c41d612c37e101
Frozen by: scripts/phase_a_freeze.py
"""
'\nTaken from https://github.com/ModelTC/LightLLM/blob/8ed97c74c18f11505b048b1ba00ba5c0cef8bff6/lightllm/common/fused_moe/deepep_scatter_gather.py\nand updated to fit vllm needs and terminology.\n'
import torch
import triton_ascend_bench.frozen.shim.modular_kernel as mk  # Phase A': off-path modular_kernel annotation shim (B-inert-shim)
from triton_ascend_bench.frozen.shim import count_expert_num_tokens  # Phase A': off-path host-wrapper helper (B-inert-shim)
import triton  # Phase A': was a vllm/sglang triton re-export (B-rewrite-public)
import triton.language as tl
from triton_ascend_bench.frozen.shim import get_mk_alignment_for_contiguous_layout  # Phase A': off-path deep_gemm accel surface (B-inert-shim)
from triton_ascend_bench.frozen.shim import round_up  # Phase A': pure-math shim (B-inert-shim)

@triton.jit
def apply_expert_map(expert_id, expert_map):
    if expert_id != -1:
        expert_id = tl.load(expert_map + expert_id).to(expert_id.dtype)
    return expert_id

@triton.jit
def _fwd_kernel_ep_gather(total_token_num, input_tensor, input_tensor_stride0, input_tensor_stride1, recv_topk_ids, recv_topk_ids_stride0, recv_topk_ids_stride1, recv_topk_weight, recv_topk_weight_stride0, recv_topk_weight_stride1, input_index, input_index_stride0, input_index_stride1, output_tensor, output_tensor_stride0, output_tensor_stride1, topk_num: tl.constexpr, expert_map, HAS_EXPERT_MAP: tl.constexpr, BLOCK_D: tl.constexpr):
    cur_block = tl.program_id(0)
    start_cur_token = tl.program_id(1)
    grid_num = tl.num_programs(1)
    for cur_token in range(start_cur_token, total_token_num, grid_num):
        off_d = tl.arange(0, BLOCK_D)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for topk_index in range(0, topk_num):
            expert_id = tl.load(recv_topk_ids + cur_token * recv_topk_ids_stride0 + topk_index)
            if HAS_EXPERT_MAP:
                expert_id = apply_expert_map(expert_id, expert_map)
            if expert_id >= 0:
                source_token_index = tl.load(input_index + cur_token * input_index_stride0 + topk_index)
                acc_weight = tl.load(recv_topk_weight + cur_token * recv_topk_weight_stride0 + topk_index)
                tmp = tl.load(input_tensor + source_token_index * input_tensor_stride0 + cur_block * BLOCK_D + off_d)
                accumulator += tmp.to(tl.float32) * acc_weight
        tl.store(output_tensor + cur_token * output_tensor_stride0 + cur_block * BLOCK_D + off_d, accumulator.to(output_tensor.dtype.element_ty))

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
