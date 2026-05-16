"""Task L3_03_kernel_unified_attention_2d — GPU triton reference (SCAFFOLD).

TODO(scaffold): replace this placeholder with the frozen upstream kernel.
Source: `vllm/v1/attention/ops/chunked_prefill_paged_decode.py`
Upstream: vLLM — kernel_unified_attention_2d
Rationale: vLLM 2D chunked prefill 形态，与 L3-01 3D 互补

Until the real kernel is frozen, this file holds a no-op @triton.jit
declaration so dataset.load_task and validate_tasks accept the directory.
"""

import triton
import triton.language as tl


@triton.jit
def kernel_unified_attention_2d(
    x_ptr,                                            # placeholder pointer
    BLOCK: tl.constexpr,
):
    """Placeholder. Replace with the frozen upstream GPU kernel body."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    tl.store(x_ptr + offs, v)
