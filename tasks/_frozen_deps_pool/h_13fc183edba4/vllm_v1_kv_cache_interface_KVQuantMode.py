"""Phase A-prime frozen dependency -- vendored from upstream.

Content-addressed pool entry. Apache-2.0; original license
header preserved below. See repo-root NOTICE for attribution.

Source module: `vllm.v1.kv_cache_interface_KVQuantMode`
Upstream repo: vllm
Commit SHA: snapshot-no-git
Content SHA-256: 13fc183edba48fee053a923a5ea857d89f7156c5f090fec55d0cc2535a905fdb
Frozen by: scripts/phase_a_prime.py (U2 pool vendoring)
"""

from enum import IntEnum


class KVQuantMode(IntEnum):
    """KV cache quantization mode.

    Used by attention backends and kernels to dispatch quantization logic
    without string matching on ``kv_cache_dtype``.
    """

    NONE = 0
    FP8_PER_TENSOR = 1  # per-tensor scales (current fp8 path)
    INT8_PER_TOKEN_HEAD = 2  # per-token-head dynamic scales for int8
    FP8_PER_TOKEN_HEAD = 3  # per-token-head dynamic scales for fp8
    NVFP4 = 4  # packed fp4 data + fp8 block scales

    @property
    def is_per_token_head(self) -> bool:
        """True for any per-token-head quantization mode."""
        return self in (
            KVQuantMode.INT8_PER_TOKEN_HEAD,
            KVQuantMode.FP8_PER_TOKEN_HEAD,
        )

    @property
    def is_nvfp4(self) -> bool:
        """True for NVFP4 packed quantization mode."""
        return self == KVQuantMode.NVFP4
