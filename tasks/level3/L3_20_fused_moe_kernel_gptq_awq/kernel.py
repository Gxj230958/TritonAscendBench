# Phase A': sliced to the graded entrypoint's module-level closure
# (whole-module vendored kernel; dead upstream siblings + their
# upstream imports dropped -- the kept defs are byte-identical).
import torch
import triton
import triton.language as tl


@triton.jit
def write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N, offs_token, token_mask, BLOCK_SIZE_M, BLOCK_SIZE_N, compute_type):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_gptq_awq(a_ptr, b_ptr, c_ptr, b_scale_ptr, b_zp_ptr, topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr, N: tl.constexpr, K: tl.constexpr, EM, num_valid_tokens, stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn, stride_bse, stride_bsk, stride_bsn, stride_bze, stride_bzk, stride_bzn, block_k_diviable: tl.constexpr, group_size: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr, compute_type: tl.constexpr, has_zp: tl.constexpr, use_int4_w4a16: tl.constexpr, use_int8_w8a16: tl.constexpr):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N, offs_token, token_mask, BLOCK_SIZE_M, BLOCK_SIZE_N, compute_type)
        return
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    if use_int4_w4a16:
        b_ptrs = b_ptr + off_experts * stride_be + offs_k[:, None] // 2 * stride_bk + offs_bn[None, :] * stride_bn
        b_shifter = offs_k[:, None] % 2 * 4
    elif use_int8_w8a16:
        b_ptrs = b_ptr + off_experts * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = offs_bn[None, :] % 2 * 4
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if not block_k_diviable:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs)
        if use_int4_w4a16:
            b = b >> b_shifter & 15
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn + (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size * stride_bsk
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
        b_scale = b_scale.to(tl.float32)
        if has_zp and use_int4_w4a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = b_zp_ptr + off_experts * stride_bze + offs_bn[None, :] // 2 * stride_bzn + offs_k_true * stride_bzk
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = b_zp >> b_zp_shifter & 15
            b_zp = b_zp.to(tl.float32)
        elif has_zp and use_int8_w8a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = b_zp_ptr + off_experts * stride_bze + offs_bn[None, :] * stride_bzn + offs_k_true * stride_bzk
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = b_zp.to(tl.float32)
        if has_zp:
            b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
        else:
            b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            b_ptrs += BLOCK_SIZE_K // 2 * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

import torch
import triton
import triton.language as tl


# Authored launcher: drives the frozen graded @triton.jit
# `fused_moe_kernel_gptq_awq` (vLLM
# vllm/model_executor/layers/fused_moe/fused_moe.py, commit
# 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19) directly, reproducing the grid /
# stride wiring the in-module host wrapper `invoke_fused_moe_wna16_triton_kernel`
# builds on its int4-w4a16 (GPTQ/AWQ) path. We drive the jit kernel directly
# rather than the host wrapper because the wrapper calls
# `get_moe_wna16_block_config(...)` and (through `dispatch_fused_moe_kernel`)
# `should_moe_wna16_use_cuda()` -> `current_platform.is_cuda()`, which pulls the
# vllm platform runtime off-GPU; and its block-config heuristics are perf-only.
# The numeric path realised is identical to the wrapper's int4 branch. (Same
# direct-jit pattern as the passed MoE siblings L3_11 / L3_12.)
#
# Quantization layout (matches the re-authored oracle_pytorch and the kernel's
# unpack arithmetic, asymmetric int4 / AWQ -> has_zp=True):
#   a        [M, K]            bf16 activations (K == D == contraction axis)
#   b        [E, N, K//2]      int32, 2 int4 nibbles packed along K
#                              (even k -> low nibble, odd k -> high; shift (k%2)*4)
#   b_scale  [E, N, K//GS]     fp32 per-(expert,n,group) dequant scale
#   b_zp     [E, N//2, K//GS]  int32, 2 int4 zero-points packed along N
#   The kernel dequantizes b = ((q - zp) * scale).to(compute_type) then
#   accumulates tl.dot(a, b) over K -- the oracle's (q - zp) * scale GEMM.
#
# Scatter / grid wiring (mirrors invoke_fused_moe_wna16_triton_kernel for top_k=1):
#   stride_be/bk/bn = B.stride(0/2/1)   (B read as [E, N, K//2], contracted over K)
#   stride_bse/bsk/bsn = B_scale.stride(0/2/1);  stride_bze/bzk/bzn = B_zp.stride(0/2/1)
#   C is rank-3 [M, top_k=1, N]; stride_cm=C.stride(1) (=N), stride_cn=C.stride(2) (=1),
#   so the kernel's `c_ptr + stride_cm*offs_token` scatters block row -> output row
#   == sorted_token_ids VALUE (the token id). a row read is offs_token//top_k.
#   num_tokens_post_padded = EM (= sorted_token_ids length); all blocks live,
#   no expert_ids == -1, so every block runs its dequant GEMM. Padding slots
#   (token id >= num_valid_tokens=M) are masked out of the store by token_mask.
#
# BLOCK_SIZE_M == block_size (=4): the kernel reads `expert_ids[pid_m]` where
# pid_m indexes BLOCK_SIZE_M-row blocks, so BLOCK_SIZE_M must equal the fixture's
# block_size for expert_ids (length E) to align with the E*4-row sorted buffer
# (E blocks of 4). BLOCK_SIZE_N tiles the N (output) axis -- masked by
# `offs_cn < N`, so any tile is numerically equivalent. BLOCK_SIZE_K is an even
# divisor of K (the int4 b load `tl.load(b_ptrs)` is UNMASKED over K, so K %
# BLOCK_SIZE_K must be 0 to avoid an OOB read); tiling only, the fp32 accumulate
# is K-order invariant within atomic_scatter tolerance. GROUP_SIZE_M=1 / SPLIT_K=1
# only reorder pid scheduling; every (pid_m, pid_n) block is still computed.
def _block_size_k(K: int) -> int:
    """Largest power-of-2 divisor of K, clamped to [16, 64]."""
    bsk = 1
    p = 1
    while p <= min(K, 64):
        if K % p == 0:
            bsk = p
        p *= 2
    return max(bsk, 16)


def run(inputs: dict) -> dict:
    a = inputs["a"].contiguous()
    b = inputs["b"].contiguous()              # [E, N, K//2] int32
    b_scale = inputs["b_scale"].contiguous()  # [E, N, K//GS] fp32
    b_zp = inputs["b_zp"].contiguous()        # [E, N//2, K//GS] int32
    topk_weights = inputs["topk_weights"].contiguous()
    sorted_token_ids = inputs["sorted_token_ids"].to(torch.int64).contiguous()
    expert_ids = inputs["expert_ids"].to(torch.int64).contiguous()
    block_size = int(inputs["block_size"])
    group_size = int(inputs["group_size"])
    apply_weight = bool(inputs.get("apply_weight", True))

    M, K = a.shape
    E, N, K_half = b.shape
    assert K_half == K // 2, (b.shape, K)
    assert N % 2 == 0 and b_zp.shape == (E, N // 2, K // group_size), b_zp.shape
    assert b_scale.shape == (E, N, K // group_size), b_scale.shape
    assert expert_ids.shape[0] == sorted_token_ids.shape[0] // block_size

    top_k = 1
    num_valid_tokens = M * top_k
    EM = sorted_token_ids.shape[0]

    BLOCK_SIZE_M = block_size
    BLOCK_SIZE_N = min(N, 64)
    BLOCK_SIZE_K = _block_size_k(K)
    assert K % BLOCK_SIZE_K == 0, (K, BLOCK_SIZE_K)

    # C is rank-3 [M, top_k, N]; the kernel scatters to flat row == token id.
    c = torch.zeros((M, top_k, N), device=a.device, dtype=torch.float32)
    num_tokens_post_padded = torch.tensor(
        [EM], dtype=torch.int32, device=a.device
    )

    grid = (
        triton.cdiv(EM, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
    )

    fused_moe_kernel_gptq_awq[grid](  # noqa: F821 (injected into kernel.py module)
        a,
        b,
        c,
        b_scale,
        b_zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_valid_tokens,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(2),  # stride_bk (K//2 axis)
        b.stride(1),  # stride_bn (N axis)
        c.stride(1),  # stride_cm (top_k axis -> flat row == token id)
        c.stride(2),  # stride_cn (N axis)
        b_scale.stride(0),
        b_scale.stride(2),  # stride_bsk (group axis)
        b_scale.stride(1),  # stride_bsn (N axis)
        b_zp.stride(0),
        b_zp.stride(2),  # stride_bzk (group axis)
        b_zp.stride(1),  # stride_bzn (N//2 axis)
        block_k_diviable=(K % BLOCK_SIZE_K == 0),
        group_size=group_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=1,
        SPLIT_K=1,
        MUL_ROUTED_WEIGHT=apply_weight,
        top_k=top_k,
        compute_type=tl.bfloat16,
        has_zp=True,
        use_int4_w4a16=True,
        use_int8_w8a16=False,
    )

    return {"output": c.view(M, N)}
