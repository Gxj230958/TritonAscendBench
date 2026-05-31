"""Phase A-prime frozen dependency -- vendored from upstream.

Content-addressed pool entry. Apache-2.0; original license
header preserved below. See repo-root NOTICE for attribution.

Source module: `vllm.model_executor.layers.mamba.ops.mamba_ssm_softplus`
Upstream repo: vllm
Commit SHA: snapshot-no-git
Content SHA-256: ccee54797b1f149aae1105fe1df02755cd2bb6d728729b31c70d63c9a5c0d5ec
Frozen by: scripts/phase_a_prime.py (U2 pool vendoring)
"""

import triton
import triton.language as tl
from packaging import version

# HAS_TRITON pinned True: the severed runtime always has triton, and
# the S2 original side forces vllm.triton_utils.HAS_TRITON=True with
# the same triton.__version__, so both pick the SAME TRITON3 branch.
HAS_TRITON = True

TRITON3 = HAS_TRITON and (version.parse(triton.__version__) >= version.parse("3.0.0"))

if TRITON3:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
        return dt
else:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
        return dt
