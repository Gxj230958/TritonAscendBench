import torch
import triton
import triton.language as tl


# Authored launcher: drives the in-module @triton.jit `scaled_mm_kernel`
# (vLLM 18f6bf5, compressed_tensors/triton_scaled_mm.py) exactly as its host
# wrapper `triton_scaled_mm` does -- same heuristic tile selection, same launch
# order. The wrapper itself is severance-clean (only torch + triton), so this
# launcher simply reproduces its body so the @triton.jit kernel is graded
# directly with a candidate-allocated output.
#
# Math (matches oracle_pytorch): A[M,K] int8 @ B[K,N] int8 accumulated in int32,
# then C = scale_a[M,1] * scale_b[N,1] * acc cast to out_dtype, then += bias[N].
# Output key: "C". out_dtype is bfloat16 (W8A8 epilogue output).
OUT_DTYPE = torch.bfloat16


def run(inputs: dict) -> dict:
    A = inputs["A"]
    B = inputs["B"]
    scale_a = inputs["scale_a"]
    scale_b = inputs["scale_b"]
    bias = inputs["bias"]

    M, K = A.shape
    N = B.shape[1]
    assert B.shape[0] == K

    scale_a = scale_a.reshape(-1, 1) if scale_a.dim() <= 1 else scale_a
    scale_b = scale_b.reshape(-1, 1) if scale_b.dim() <= 1 else scale_b

    result = torch.empty((M, N), dtype=OUT_DTYPE, device=A.device)

    has_scalar = lambda x: x.shape[0] == 1 and x.shape[1] == 1

    # Heuristic tile selection -- verbatim from triton_scaled_mm.
    next_power_of_2_M = max(32, triton.next_power_of_2(M))
    is_small_N = N < 8192
    if next_power_of_2_M <= 32:
        tile_shape = (64, 64, 256) if is_small_N else (64, 128, 256)
    elif next_power_of_2_M <= 64:
        tile_shape = (64, 64, 256)
    elif next_power_of_2_M <= 128:
        tile_shape = (64, 128, 128)
    else:
        tile_shape = (128, 128, 128)
    block_size_m, block_size_n, block_size_k = tile_shape

    block_size_sa = 1 if has_scalar(scale_a) else block_size_m
    block_size_sb = 1 if has_scalar(scale_b) else block_size_n

    # int8 inputs -> int32 accumulation.
    accumulator_dtype = tl.float32 if A.is_floating_point() else tl.int32

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    scaled_mm_kernel[grid](
        A,
        B,
        scale_a,
        scale_b,
        result,
        bias,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        result.stride(0),
        result.stride(1),
        accumulator_dtype,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_SCALE_A=block_size_sa,
        BLOCK_SIZE_SCALE_B=block_size_sb,
    )

    return {"C": result}
