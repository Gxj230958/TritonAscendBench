"""Task L1_30_prepare_mrope_positions_kernel — GPU triton reference (SCAFFOLD).

TODO(scaffold): replace this placeholder with the frozen upstream kernel.
Source: vLLM mrope
Upstream: vLLM — _prepare_mrope_positions_kernel
Rationale: Qwen2-VL / Qwen3-VL mRoPE 位置准备

Until the real kernel is frozen, this file holds a no-op @triton.jit
declaration so dataset.load_task and validate_tasks accept the directory.
"""

import triton
import triton.language as tl


@triton.jit
def prepare_mrope_positions_kernel(
    x_ptr,                                            # placeholder pointer
    BLOCK: tl.constexpr,
):
    """Placeholder. Replace with the frozen upstream GPU kernel body."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    tl.store(x_ptr + offs, v)
