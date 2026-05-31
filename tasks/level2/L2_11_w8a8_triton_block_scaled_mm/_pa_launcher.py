import torch
import triton


# Authored jit-only launcher: drives the in-module @triton.jit
# `_w8a8_triton_block_scaled_mm` (vLLM 18f6bf5,
# vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py)
# exactly as its host wrapper `w8a8_triton_block_scaled_mm` does on the default
# (un-tuned) config path (kernel.py lines 834-928).
#
# Why a launcher rather than the host wrapper: the wrapper unconditionally calls
# `get_w8a8_block_fp8_configs(N, K, block_n, block_k)`, whose first statement is
# `current_platform.get_device_name()` -- a platform query that only builds a
# per-device autotune-config JSON filename. No such JSON ships with the task, so
# the lookup always returns None and the wrapper falls to the DEFAULT config
# regardless of device name. The device-name probe is therefore a perf-config
# detour with no effect on numeric output, but on the severed bench host it
# leaks the real `vllm.platforms` import (interface.py get_device_name raises
# NotImplementedError). This launcher reproduces the wrapper's DEFAULT-config
# launch body verbatim so the @triton.jit kernel is graded directly, on any
# device, without the platform-gated config detour.
#
# Default config copied verbatim from the wrapper's `else` branch:
#   BLOCK_SIZE_M = 64
#   BLOCK_SIZE_N = block_size[0]   # divisible by block_n (block-wise quant)
#   BLOCK_SIZE_K = block_size[1]   # divisible by block_k (>= 32 for int8 tl.dot)
#   GROUP_SIZE_M = 32 ; num_warps = 4 ; num_stages = 2
# Tile sizes are launch metadata only: the kernel accumulates in fp32 over the
# full K with masked tail loads, so the result is independent of the tiling.
#
# Kernel math (matches the re-authored oracle_pytorch): B is [N, K], the GEMM is
# A @ B.T,
#   C[m, n] = sum_k A[m, k] * B[n, k] * As[m, k // block_k] * Bs[n // block_n, k // block_k]
# accumulated in fp32 and cast once to output_dtype (fp16 default) at the store.
# Output key: "C".
def run(inputs: dict) -> dict:
    A = inputs["A"]
    B = inputs["B"]
    As = inputs["As"]
    Bs = inputs["Bs"]
    block_size = inputs["block_size"]
    output_dtype = inputs.get("output_dtype", torch.float16)

    assert len(block_size) == 2
    block_n, block_k = int(block_size[0]), int(block_size[1])

    # Shape invariants the host wrapper asserts before launch.
    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    M = A.numel() // A.shape[-1]

    assert B.ndim == 2 and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    # Default (un-tuned) config -- verbatim from the wrapper's `else` branch.
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_stages": 2,
    }

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    _w8a8_triton_block_scaled_mm[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
    )

    return {"C": C}
