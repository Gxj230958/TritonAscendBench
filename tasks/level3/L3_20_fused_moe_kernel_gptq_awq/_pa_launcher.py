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
#   a        [M, K]            fp32 activations (K == D == contraction axis)
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
