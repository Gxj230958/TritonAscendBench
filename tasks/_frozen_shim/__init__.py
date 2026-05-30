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

import typing

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


class CustomOp(torch.nn.Module):
    """Inert shim for ``vllm.model_executor.custom_op.CustomOp``.

    Upstream CustomOp is an ``nn.Module`` whose ``register(name)``
    classmethod returns a decorator that records the class in a module-
    global ``op_registry`` dict. Frozen kernels subclass it ONLY for an
    unrelated registered class (e.g. ``FusedRMSNormGated``) defined at
    module scope; the verified entrypoint is a free function that never
    instantiates that class nor reads ``op_registry``. So at import time
    only the class definition + the ``@CustomOp.register(...)`` decorator
    run -- both satisfied by an nn.Module base + a no-op register.

    The rewriter applies this shim ONLY when the module has no
    ``op_registry`` / ``op_registry_oot`` reference outside the decorator
    site (fail-closed: a module that actually dispatches through the
    registry must NOT get the no-op shim)."""

    @classmethod
    def register(cls, name: str):
        def _decorator(op_cls):
            op_cls.name = name  # verbatim attr upstream also sets; harmless
            return op_cls       # no-op: op_registry is never read on the
            # verified path, so we do not maintain it.

        return _decorator


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


def is_hip() -> bool:
    """Pure platform predicate mirroring sglang.srt.utils.is_hip -- True
    only on a ROCm/HIP build. The severed standalone targets the CUDA /
    Ascend-NPU migration path, never ROCm, so this is a constant False. A
    frozen kernel reads it only in module-scope setup (a backend gate),
    never the verified compute path; B-inert."""
    return False


def is_musa() -> bool:
    """Pure platform predicate mirroring sglang.srt.utils.is_musa -- True
    only on a Moore-Threads MUSA build. Constant False on the bench
    CUDA/NPU path; setup-only backend gate, B-inert."""
    return False


def is_cuda() -> bool:
    """Pure platform predicate mirroring sglang.srt.utils.is_cuda. Live,
    honest query of the real backend (the bench runs the standalone on the
    actual device); a setup-only backend gate, never the numeric compute
    path. B-inert."""
    return bool(torch.cuda.is_available() and torch.version.cuda is not None)


def ceil_div(a: int, b: int) -> int:
    """Pure integer ceil-division mirroring sglang.srt.utils.ceil_div --
    identical to ``cdiv`` (kept under the upstream name so a frozen
    kernel's ``from sglang.srt.utils import ceil_div`` resolves). B-inert
    pure math; text-identical to upstream sglang.srt.utils.common.ceil_div
    ``(x + y - 1) // y`` for the positive shape/alignment divisors used at
    every callsite."""
    return (int(a) + int(b) - 1) // int(b)


def register_custom_op(*args, **kwargs):
    """Inert decorator mirroring sglang.srt.utils.custom_op.register_custom_op.
    Upstream registers a name into sglang's global custom-op dispatch table
    for an unrelated sibling op defined at module scope; the verified
    free-function entrypoint never dispatches through that table. The shim
    returns the decorated object UNCHANGED, supporting both bare
    ``@register_custom_op`` and parameterised ``@register_custom_op("name", ...)``
    forms. B-inert (no registry side effect)."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def _deco(obj):
        return obj

    return _deco


# vllm sampling/output/worker symbols carried at module scope by some frozen
# kernels for SIBLING setup helpers (batch-state managers, add_request type
# hints, logprob output containers). The verified free-function entrypoint
# never constructs or reads them, so the standalone only needs them to RESOLVE
# (import + class-def) without pulling real vllm. B-inert.
MAX_LOGPROB_TOKEN_IDS = 128  # verbatim from vllm/sampling_params.py:28


class SamplingParams:
    """Inert stand-in for vllm.sampling_params.SamplingParams. Referenced only
    as a type hint / in module-scope setup (e.g. ``add_request(... ,
    sampling_params: SamplingParams)``), never the verified numeric path.
    Accepts and records any construction kwargs; B-inert."""

    def __init__(self, *args, **kwargs):
        for _k, _v in kwargs.items():
            setattr(self, _k, _v)


class LogprobsTensors(typing.NamedTuple):
    """Faithful container mirroring vllm/v1/outputs.py:51 LogprobsTensors --
    a 4-field NamedTuple (pure data, no behaviour), so the shim is byte-faithful
    rather than inert. The 4th field ``cu_num_generated_tokens`` is optional
    upstream and a sibling helper builds one WITH it
    (``LogprobsTensors(..., cu_num_generated_tokens=...)``), so omitting it would
    raise TypeError on that construction. The verified entrypoint returns a
    plain tensor and never builds one."""

    logprob_token_ids: "torch.Tensor"
    logprobs: "torch.Tensor"
    selected_token_ranks: "torch.Tensor"
    cu_num_generated_tokens: "list | None" = None


class _SetupOnlyShim:
    """Base for inert setup-only stand-ins: construction is a no-op, but ANY
    attribute/method access raises loudly -- a verified entrypoint must never
    reach real buffer behaviour, and a frozen kernel that does has a genuine
    runtime dependency (classify it, never silently fake it)."""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        raise AttributeError(
            f"{type(self).__name__} shim: '.{name}' reached. This inert "
            f"stand-in is setup-only and must NOT enter a verified "
            f"entrypoint's path; a frozen kernel reaching real buffer "
            f"behaviour is a true runtime dependency, not a clean standalone."
        )


class UvaBackedTensor(_SetupOnlyShim):
    """Inert stand-in for vllm.v1.worker.gpu.buffer_utils.UvaBackedTensor (a
    host/device UVA staging buffer). Constructed only inside module-scope
    batch-state setup classes, never the verified numeric path; loud-fails on
    any ``.gpu`` / ``.np`` / ``.copy_to_uva`` access. B-inert."""


class StagedWriteTensor(_SetupOnlyShim):
    """Inert stand-in for vllm.v1.worker.gpu.buffer_utils.StagedWriteTensor
    (a staged host->device write buffer). Setup-only, same rationale as
    ``UvaBackedTensor``; B-inert."""
