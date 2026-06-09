"""Phase A-prime frozen dependency -- vendored from upstream.

Content-addressed pool entry. Apache-2.0; original license
header preserved below. See repo-root NOTICE for attribution.

Source module: `vllm.model_executor.layers.mamba.ops.triton_helpers`
Upstream repo: vllm
Commit SHA: snapshot-no-git
Content SHA-256: 119bf34e8937dd56165c19f07bb69dc5ff0a7012bc5db979b1edf32e1af7ccdd
"""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import triton  # Phase A': was vllm.triton_utils re-export
import triton.language as tl


@triton.jit
def fast_exp(x):
    """Faster alternative to tl.exp() using the hardware exp2 instruction.

    tl.math.exp2 maps directly to a single ex2.approx.f32 PTX instruction,
    while tl.exp goes through libdevice __nv_expf which adds function call
    overhead and extra range checking.
    """
    # exp(x) = exp2(x * log2(e)), where log2(e) = 1/ln(2) = 1.4426950408889634
    LOG2E = tl.constexpr(1.4426950408889634)
    return tl.math.exp2(LOG2E * x)
