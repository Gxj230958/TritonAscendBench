"""Phase A' standalone shim package.

When Phase A' rewrites a task's ``kernel.py`` to standalone form
(plan U5), references to non-numeric vllm / sglang helpers (loggers,
platform queries, dtype constants, pure predicates) are redirected to
the stubs below. The rewriter knows about each by name; severing the
real upstream import is the goal -- the shim provides the SAME
runtime values that the upstream code returned on the bench's target
class (CUDA / Ascend NPU, which are operationally equivalent for the
queries below).

Invariants:
- Every stub MUST return the same value the upstream returned on the
  target class. Anything platform-specific that diverges between
  CUDA / NPU is documented inline.
- Numeric/algorithmic logic in kernel.py NEVER calls these stubs in
  a way that would change kernel output -- they are configuration
  predicates / dtype constants the kernel reads BEFORE the compute
  path, not values mixed into the math.
"""
from __future__ import annotations

import torch


def init_logger(name: str = ""):
    """Inert no-op logger sink. Upstream returns a stdlib-logger-like
    object; bench's frozen kernel.py never emits at PASS-relevant
    severity, so a silent stub is correct for both CUDA and NPU."""

    class _Silent:
        def info(self, *a, **kw): pass
        def warning(self, *a, **kw): pass
        def error(self, *a, **kw): pass
        def debug(self, *a, **kw): pass
        def critical(self, *a, **kw): pass

    return _Silent()


class _CurrentPlatformStub:
    """Stub for ``vllm.platforms.current_platform`` queries the bench's
    target classes (CUDA / NPU) answer deterministically.

    - is_cuda(): the bench's authoritative verify host is NVIDIA CUDA -> True
    - is_cuda_alike(): CUDA or ROCm (vllm interface.py:191); True on the
      CUDA verify host. Used only to pick device="cuda" for scheduling.
    - is_rocm(): bench targets NVIDIA CUDA and Ascend NPU, neither is AMD ROCm
    - is_xpu(): same, neither is Intel XPU
    - is_fp8_fnuz(): only ROCm + fnuz fp8 variant; both target classes are False
    - fp8_dtype(): NVIDIA / Ascend both default to ``torch.float8_e4m3fn``
    - num_compute_units(): NPU has a real implementation at runtime; this
      stub returns a representative value (128) used only when the host
      wrapper sizes a grid -- the value affects throughput, not numeric
      output.
    """

    @staticmethod
    def is_cuda() -> bool:
        return True

    @staticmethod
    def is_cuda_alike() -> bool:
        return True

    @staticmethod
    def is_rocm() -> bool:
        return False

    @staticmethod
    def is_xpu() -> bool:
        return False

    @staticmethod
    def is_fp8_fnuz() -> bool:
        return False

    @staticmethod
    def fp8_dtype() -> torch.dtype:
        return torch.float8_e4m3fn

    @staticmethod
    def num_compute_units(device_id: int = 0) -> int:
        return 128


current_platform = _CurrentPlatformStub()


# Module-level alias of the platform-query helper. vllm.utils.platform_utils
# exports num_compute_units as a free function (it delegates to
# current_platform.num_compute_units); the FLA rewrite table maps
# `from vllm.utils.platform_utils import num_compute_units` to this name.
# Scheduling-only value (grid/shared-mem sizing), never numeric output.
def num_compute_units(device_id: int = 0) -> int:
    return _CurrentPlatformStub.num_compute_units(device_id)


def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    """Verbatim from vllm/utils/torch_utils.py -- pure string predicate,
    no platform branching."""
    return (
        kv_cache_dtype.startswith("fp8")
        or kv_cache_dtype.endswith("per_token_head")
        or kv_cache_dtype == "nvfp4"
    )


# Quant utilities the upstream module re-exports.
FP8_DTYPE: torch.dtype = torch.float8_e4m3fn


def get_fp8_min_max() -> tuple[float, float]:
    """Verbatim from vllm/model_executor/layers/quantization/utils/quant_utils.py
    for the non-ROCm path (is_fp8_fnuz() is False on bench target classes)."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    return finfo.min, finfo.max


# vllm.utils.math_utils pure-math helpers. Verbatim against the upstream at
# infer/vllm/vllm/utils/math_utils.py:10,15,20 -- no platform / dtype /
# scheduling branch, so plan U3 inert.
def cdiv(a: int, b: int) -> int:
    """Verbatim from vllm/utils/math_utils.py:10. Ceiling division that
    handles negative divisors correctly (differs from ``(a + b - 1) // b``
    for negative ``b``)."""
    return -(a // -b)


def round_up(a: int, m: int) -> int:
    """Verbatim from vllm/utils/math_utils.py:20. Align ``a`` up to the
    next multiple of ``m``."""
    return ((a + m - 1) // m) * m


def next_power_of_2(n: int) -> int:
    """Verbatim from vllm/utils/math_utils.py:15. Smallest power of two
    >= ``n``. Caller guarantees ``n >= 1``."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


# vllm.v1.attention.backends.utils sentinel constants. Verbatim against
# infer/vllm/vllm/v1/attention/backends/utils.py:44,45 -- plain int
# constants used as padding / null-block sentinels in mask comparisons
# and fill values. No platform branch, no side effect; plan U3 inert.
# Exact value matters: kernels branch `if slot_id == PAD_SLOT_ID: return`
# and fill metadata with these, so they MUST stay -1 / 0.
PAD_SLOT_ID: int = -1
NULL_BLOCK_ID: int = 0


def is_torch_equal_or_newer(target: str) -> bool:
    """Verbatim from vllm/utils/torch_utils.py:810,814 -- True iff the
    installed torch is >= ``target``. A pure version predicate (no
    platform / dtype / device branch), so plan U3 inert. Frozen kernels
    read it only in module-scope setup (a feature gate), never in the
    verified compute path."""
    from packaging.version import Version, parse

    try:
        return parse(str(torch.__version__)) >= parse(target)
    except Exception:
        import importlib.metadata

        return Version(importlib.metadata.version("torch")) >= Version(target)


def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    """Live device query mirroring vllm/utils/mem_utils.py: the max
    shared memory per thread block for ``gpu``. Upstream reads a CUDA
    device attribute via vllm._custom_ops; we query torch directly so the
    returned value is the REAL device's (honest even if a frozen kernel's
    bench-unused setup path consumes it for tile sizing). Scheduling-only:
    it never enters the numeric compute path of a verified entrypoint.
    Fails loudly off-CUDA (a standalone consuming this on a non-CUDA host
    must be classified, not silently defaulted)."""
    props = torch.cuda.get_device_properties(gpu)
    val = getattr(props, "shared_memory_per_block_optin", None)
    if not val:
        val = props.shared_memory_per_block
    assert val and val > 0, "max_shared_mem cannot be zero"
    return int(val)
