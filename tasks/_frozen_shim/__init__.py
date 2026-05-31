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

import enum
import os
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


class DeviceCapability(typing.NamedTuple):
    """Mirror of ``vllm.platforms.interface.DeviceCapability`` (major,
    minor) with ``to_int()`` -> major*10 + minor. Compares equal to a
    plain ``(major, minor)`` tuple, which is how the frozen kernel.py
    Turing check (``get_device_capability() == (7, 5)``) consumes it."""

    major: int
    minor: int

    def to_int(self) -> int:
        return self.major * 10 + self.minor


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

    @staticmethod
    def get_device_capability(device_id: int = 0) -> "DeviceCapability":
        """Compute-capability of the verify device. On a CUDA verify host
        the real torch query wins (faithful to upstream, which reads
        ``torch.cuda.get_device_capability`` / NVML); off-GPU (import-time
        on the CPU oracle pre-filter) fall back to (8, 0), the sm80 A100
        class the bench targets -- matching upstream BASE_BLOCK=128 /
        IS_TURING=False. These feed module-level tiling/precision constants
        only, never the kernel's numeric path."""
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(device_id)
            return DeviceCapability(int(major), int(minor))
        return DeviceCapability(8, 0)

    @staticmethod
    def has_device_capability(
        capability: "int | tuple[int, int]", device_id: int = 0
    ) -> bool:
        """Upstream semantics (interface.py): True iff the device capability
        meets ``capability``. Accepts an int (compared against
        ``major*10+minor``) or a (major, minor) tuple."""
        cap = _CurrentPlatformStub.get_device_capability(device_id)
        if isinstance(capability, tuple):
            return cap.to_int() >= DeviceCapability(*capability).to_int()
        return cap.to_int() >= capability


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


def register_cuda_ci(est_time=None, suite=None, nightly=False, disabled=None):
    """Runtime no-op mirroring sglang.test.ci.ci_register.register_cuda_ci.
    Upstream is a pure CI-registration MARKER (parsed via AST by the CI
    harness; its function body is a runtime ``return None`` -- see
    sglang/test/ci/ci_register.py:51). A frozen kernel calls it only at
    module scope to register the file in a CI suite; it never touches the
    numeric compute path. Byte-faithful: returns None exactly as upstream."""
    return None


def moe_align_block_size(*args, **kwargs):
    """Loud-fail stand-in for sglang.jit_kernel.moe_align.moe_align_block_size.
    Upstream is the JIT-compiled CUDA reference kernel. The frozen kernel.py
    imports it ONLY for its sibling pytest cross-check tests
    (test_moe_align_block_size_compare_implementations / ..._v2_large_num_experts);
    the graded entrypoint drives the in-module 4-stage @triton.jit pipeline via
    _pa_launcher.py and never calls it. Importing the real symbol pulls the
    sglang JIT/CUDA-compile dependency chain into the severed standalone, so it
    is redirected here. Raising keeps any task that genuinely needs the CUDA
    reference honest -- the severed kernel must not silently fake it."""
    raise NotImplementedError(
        "moe_align_block_size is the sibling-only sglang CUDA reference kernel "
        "(used by the frozen kernel.py's pytest cross-checks, never the graded "
        "entrypoint); it is not vendored into the severed standalone"
    )


def is_dp_attention_enabled() -> bool:
    """Faithful mirror of sglang.srt.layers.dp_attention.is_dp_attention_enabled.
    Upstream returns a module-global flag (``_ENABLE_DP_ATTENTION_FLAG``) set by
    server bootstrap; absent that bootstrap it is False. The bench runs the
    standalone kernel with no DP-attention server context, so the honest value is
    False. The L2_29 frozen kernel.py reads it ONLY inside the sibling
    ``fused_moe_router_tensorcore`` host wrapper (as a dp_attn NaN-masking gate);
    the graded ``fused_moe_router_cudacore`` path never reaches it. B-inert
    setup-only backend predicate."""
    return False


def fused_topk(*args, **kwargs):
    """Loud-fail stand-in for sglang.srt.layers.moe.topk.fused_topk. Upstream
    dispatches into the CUDA ``topk_softmax`` / aiter fused-topk ops, pulling the
    whole sglang MoE topk dependency chain into the severed standalone. The L2_29
    frozen kernel.py imports it ONLY for the sibling ``FusedMoeRouter.forward_torch``
    method; the graded ``fused_moe_router_cudacore`` entrypoint computes its own
    top-k inside the @triton.jit kernel and never calls it. Redirected here so the
    module imports standalone; raising keeps any task that genuinely reaches it
    honest -- the severed kernel must surface the call, never fake it."""
    raise NotImplementedError(
        "fused_topk is the sibling-only sglang CUDA topk_softmax dispatcher "
        "(used by FusedMoeRouter.forward_torch, never the graded "
        "fused_moe_router_cudacore entrypoint); it is not vendored into the "
        "severed standalone"
    )


def cached_triton_kernel(key_fn=None):
    """Passthrough mirror of sglang.srt.utils.cached_triton_kernel. Upstream
    wraps a @triton.jit kernel in a custom key-based launch cache (CachedKernel)
    to skip triton's specialization dispatch; its non-CUDA fallback returns the
    kernel UNCHANGED. The bench takes that native-triton-cache path -- triton
    re-specializes per constexpr, so it is always value-correct, where the custom
    cache's coarser key is only a launch-overhead optimization. The S2 bit-exact
    check validates this severed kernel against the original CachedKernel path."""

    def _deco(fn):
        return fn

    return _deco


def count_expert_num_tokens(*args, **kwargs):
    """Loud-fail stand-in for
    vllm.model_executor.layers.fused_moe.utils.count_expert_num_tokens.
    The frozen kernel.py imports it ONLY inside the host wrapper
    ``deepgemm_moe_permute`` (deep_gemm_utils.py:414), which the graded
    entrypoint ``_fwd_kernel_ep_scatter_1`` (driven by _pa_launcher.py)
    never reaches -- kernel_1 builds the contiguous-group layout straight
    from the per-expert counts. Importing the real symbol pulls vllm's
    fused_moe.utils closure into the severed standalone, so it is
    redirected here. Raising keeps any task that genuinely reaches it
    honest: the severed kernel must surface the call, never fake it."""
    raise NotImplementedError(
        "count_expert_num_tokens belongs to the off-graded-path host wrapper "
        "deepgemm_moe_permute; the graded _fwd_kernel_ep_scatter_1 never calls "
        "it, so it is not vendored into the severed standalone"
    )


def get_mk_alignment_for_contiguous_layout(*args, **kwargs):
    """Loud-fail stand-in for
    vllm.utils.deep_gemm.get_mk_alignment_for_contiguous_layout.
    The frozen kernel.py imports it ONLY inside the host wrapper
    ``deepgemm_moe_permute`` (deep_gemm_utils.py:375) to pick the DeepGEMM
    block (block_m, block_k); the graded entrypoint
    ``_fwd_kernel_ep_scatter_1`` hard-codes BLOCK_E=128 in _pa_launcher.py
    and never reaches this wrapper. The real symbol lives in vllm's
    optional deep_gemm accel surface, so it is redirected here. Raising
    keeps the severed kernel honest if the call path ever changes."""
    raise NotImplementedError(
        "get_mk_alignment_for_contiguous_layout belongs to the off-graded-path "
        "host wrapper deepgemm_moe_permute; the graded _fwd_kernel_ep_scatter_1 "
        "never calls it, so it is not vendored into the severed standalone"
    )


# Block-size table for the CSGMV LoRA expand kernel, mirroring
# sglang.srt.lora.triton_ops.lora_tuning_config.DEFAULT_EXPAND_CONFIG. Tile
# sizes are pure launch-config (they affect only tiling, not the result), so
# the untuned defaults are numerically equivalent to any tuned entry.
def get_lora_expand_config(K, R, num_slices, chunk_size):
    """Stand-in returning the upstream default expand block config."""
    return {"BLOCK_N": 64, "BLOCK_K": 16}


# Block-size table for the CSGMV LoRA shrink (lora_a) kernel, mirroring
# sglang.srt.lora.triton_ops.lora_tuning_config.DEFAULT_SHRINK_CONFIG. Same
# rationale as the expand sibling: tile sizes are pure launch-config (tiling
# only, never the computed result), so the untuned defaults are numerically
# equivalent to any tuned entry.
def get_lora_shrink_config(K, R, num_slices, chunk_size):
    """Stand-in returning the upstream default shrink block config."""
    return {"BLOCK_N": 16, "BLOCK_K": 256}


def sglang_per_token_group_quant_fp8(*args, **kwargs):
    """Loud-fail stand-in for sglang fp8_kernel.sglang_per_token_group_quant_fp8.
    The bundled cutlass-moe file imports it under a `if is_cuda()` gate, but only
    the silu_and_mul_post_quant sibling CALLS it; the reorder entrypoints never
    do. Raising keeps any task that genuinely needs fp8 quant honest -- the
    severed kernel must not silently fake it."""
    raise NotImplementedError(
        "sglang_per_token_group_quant_fp8 is a sibling-only fp8 quant path that "
        "is not vendored; the graded reorder entrypoint must not invoke it"
    )


# Upstream alias used by the cutlass-moe files (imported as
# `sglang_per_token_group_quant_fp8 as per_token_group_quant_fp8`, or directly).
per_token_group_quant_fp8 = sglang_per_token_group_quant_fp8


def scaled_fp8_quant(*args, **kwargs):
    """Loud-fail stand-in for sglang fp8_kernel.scaled_fp8_quant. The frozen
    L1_15 fused_moe_triton_kernels file imports it ONLY for the dead
    invoke_fused_moe_kernel sibling (kernel.py:764, fp8 activation quant); the
    graded act_and_mul_kernel entrypoint never calls it. Importing the real
    symbol pulls the sglang fp8-quant CUDA closure into the severed standalone,
    so it is redirected here. Raising keeps any task that genuinely needs fp8
    quant honest -- the severed kernel must not silently fake it."""
    raise NotImplementedError(
        "scaled_fp8_quant is a sibling-only fp8 quant path (the dead "
        "invoke_fused_moe_kernel branch); it is not vendored into the severed "
        "standalone and the graded act_and_mul_kernel must not invoke it"
    )


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


class InputBatch(_SetupOnlyShim):
    """Inert stand-in for vllm.v1.worker.gpu.input_batch.InputBatch (a sibling
    batch-state container, only a type hint on the graded path); B-inert."""


class RequestState(_SetupOnlyShim):
    """Inert stand-in for vllm.v1.worker.gpu.states.RequestState (a sibling
    per-request batch-state manager). Carried by the L1_23 frozen kernel.py only
    as the PenaltiesState constructor argument type + its attribute reads
    (max_num_reqs / vocab_size / device / all_token_ids / prompt_len /
    prefill_len). PenaltiesState is the host state manager, never constructed on
    the graded path (the launcher drives _penalties_kernel from flat tensors), so
    the name only needs to resolve; any attribute access loud-fails. B-inert."""


class LoRABatchInfo(_SetupOnlyShim):
    """Inert stand-in for sglang.srt.lora.utils.LoRABatchInfo (a batch-state
    dataclass, only a host-wrapper type hint on the graded path); B-inert."""


def async_copy_to_gpu(*args, **kwargs):
    """Stand-in for vllm.v1.worker.gpu.buffer_utils.async_copy_to_gpu -- a real
    host->device copy used only by sibling state managers. The import must
    resolve so the module loads, but it LOUD-FAILS if a verified entrypoint
    ever calls it (never silently a no-op)."""
    raise NotImplementedError(
        "async_copy_to_gpu shim reached: a real host->device copy is setup-only "
        "and must not enter a verified entrypoint's path."
    )


def async_tensor_h2d(*args, **kwargs):
    """Stand-in for vllm.utils.torch_utils.async_tensor_h2d -- a real pinned
    host->device tensor copy used only by sibling state-manager setup (e.g. the
    L1_23 PenaltiesState.apply_staged_writes idx-mapping upload). The import must
    resolve so the module loads, but it LOUD-FAILS if a verified entrypoint ever
    calls it (never silently a no-op): the graded launcher drives the kernel from
    flat tensors and never stages a host->device copy through it."""
    raise NotImplementedError(
        "async_tensor_h2d shim reached: a real host->device copy is setup-only "
        "(sibling state-manager staging) and must not enter a verified "
        "entrypoint's path."
    )


# ───────────────────────────────────────────────────────────────────────────
# sglang logits_processor sibling surface (task L1_18_fused_softcap).
#
# The frozen kernel.py for L1_18 is upstream's whole
# ``sglang/srt/layers/logits_processor.py``; the GRADED entrypoint is the free
# function ``fused_softcap`` (a pure cap*tanh(x/cap) elementwise map) and its
# @triton.jit ``fused_softcap_kernel``. Neither references ANY of the symbols
# below -- they are imported only for the sibling ``LogitsProcessor`` /
# ``LogitsMetadata`` classes (DP/TP collectives, server-args, logprob helpers,
# forward-batch types), which are never constructed or called on the graded
# path. The standalone only needs these names to RESOLVE so the module body
# (class defs + module-scope ``_is_npu = is_npu()``) executes without pulling
# real sglang. Behaviour-bearing callables LOUD-FAIL if ever invoked (fail-
# closed: a verified path reaching a real collective / server-arg read is a
# genuine dependency, never silently faked); pure backend predicates return
# their honest bench-target value; type/enum stand-ins satisfy the import-time
# annotations and class-default reads. B-inert.
# ───────────────────────────────────────────────────────────────────────────
def is_npu() -> bool:
    """Pure backend predicate mirroring sglang.srt.utils.common.is_npu --
    True only on an Ascend-NPU build. The bench's authoritative verify host is
    NVIDIA CUDA, so this is a constant False. The frozen kernel reads it only
    at module scope (``_is_npu = is_npu()``) to gate SIBLING branches; the
    graded ``fused_softcap`` never reads ``_is_npu``, so the value is inert to
    the verified output. B-inert."""
    return False


def use_intel_amx_backend(*args, **kwargs) -> bool:
    """Pure backend predicate mirroring sglang.srt.utils.common.use_intel_amx_backend
    -- True only when an Intel AMX CPU path is selected. Constant False on the
    bench CUDA/NPU GPU path; read only in a sibling ``_get_logits`` branch,
    never the graded path. B-inert."""
    return False


def _sibling_only_call(_name: str):
    """Build a loud-failing stand-in for a sglang helper that is imported by
    L1_18's frozen kernel but only reached on the SIBLING LogitsProcessor path
    (DP/TP collectives, server-args, logprob builders). The name must resolve
    so the module imports; invoking it means a verified entrypoint reached real
    distributed/server behaviour -- a true dependency to classify, never to
    fake. Fail-closed."""

    def _stub(*args, **kwargs):
        raise NotImplementedError(
            f"{_name} shim reached: this sglang helper is sibling-only "
            f"(LogitsProcessor DP/TP/logprob path) and must not enter the "
            f"graded fused_softcap entrypoint's path."
        )

    _stub.__name__ = _name
    _stub.__qualname__ = _name
    return _stub


# sglang.srt.distributed
get_tensor_model_parallel_world_size = _sibling_only_call(
    "get_tensor_model_parallel_world_size"
)
tensor_model_parallel_all_gather = _sibling_only_call(
    "tensor_model_parallel_all_gather"
)

# sglang.srt.layers.dp_attention (collective/query helpers)
attn_tp_all_gather = _sibling_only_call("attn_tp_all_gather")
attn_tp_all_gather_into_tensor = _sibling_only_call(
    "attn_tp_all_gather_into_tensor"
)
dp_gather_replicate = _sibling_only_call("dp_gather_replicate")
dp_scatter = _sibling_only_call("dp_scatter")
get_attention_dp_rank = _sibling_only_call("get_attention_dp_rank")
get_attention_dp_size = _sibling_only_call("get_attention_dp_size")
get_attention_tp_size = _sibling_only_call("get_attention_tp_size")
get_dp_device = _sibling_only_call("get_dp_device")
get_dp_dtype = _sibling_only_call("get_dp_dtype")
get_dp_hidden_size = _sibling_only_call("get_dp_hidden_size")

# sglang.srt.server_args
get_global_server_args = _sibling_only_call("get_global_server_args")

# sglang.srt.layers.utils.logprob (logprob builders / normalizers)
compute_temp_top_p_normalized_logprobs = _sibling_only_call(
    "compute_temp_top_p_normalized_logprobs"
)
get_token_ids_logprobs_chunk = _sibling_only_call(
    "get_token_ids_logprobs_chunk"
)
get_token_ids_logprobs_prefill = _sibling_only_call(
    "get_token_ids_logprobs_prefill"
)
get_top_logprobs_chunk = _sibling_only_call("get_top_logprobs_chunk")
get_top_logprobs_prefill = _sibling_only_call("get_top_logprobs_prefill")

# sglang.jit_kernel.utils.is_arch_support_pdl -- a GPU-arch capability gate
# (Programmatic Dependent Launch, sm_90+). The frozen FlashMLA kernel.py reads
# it ONLY inside the sibling fp8 helper mla_quantize_and_rope_for_fp8
# (enable_pdl= launch hint); the graded create_flashmla_kv_indices_triton
# integer-index path never calls it. Importing the real symbol pulls the
# sglang JIT chain into the severed standalone, so it is redirected here.
# Fail-closed: a task that genuinely needs the PDL gate stays honestly BLOCKED.
is_arch_support_pdl = _sibling_only_call("is_arch_support_pdl")

# sgl_kernel.concat_mla_absorb_q -- the CUDA fused absorb-concat op. The frozen
# FlashMLA kernel.py imports it (under a module-scope `if is_cuda():` gate) and
# calls it ONLY in the sibling concat_mla_absorb_q_general helper; the graded
# create_flashmla_kv_indices_triton page-index path never touches it. The real
# import pulls the sgl_kernel CUDA extension into the severed standalone, so it
# is redirected here. Fail-closed (raises if a graded path ever invokes it).
concat_mla_absorb_q = _sibling_only_call("concat_mla_absorb_q")


class DpPaddingMode(enum.Enum):
    """Inert stand-in for sglang.srt.layers.dp_attention.DpPaddingMode. Used
    only as a sibling annotation (LogitsMetadata.dp_padding_mode) at import time
    and a ``DpPaddingMode.SUM_LEN`` read inside a sibling method body; never the
    graded path. Members mirror upstream's enum names so those reads resolve.
    B-inert (pure enum, no behaviour)."""

    SUM_LEN = 0
    MAX_LEN = 1


class CaptureHiddenMode(enum.Enum):
    """Inert stand-in for
    sglang.srt.model_executor.forward_batch_info.CaptureHiddenMode. The frozen
    kernel reads ``CaptureHiddenMode.NULL`` as a class-level default at import
    time (LogitsMetadata.capture_hidden_mode), so the ``NULL`` member MUST
    exist. Sibling-only; never the graded path. B-inert (pure enum)."""

    NULL = 0
    FULL = 1
    LAST = 2


class ForwardMode(enum.Enum):
    """Inert stand-in for
    sglang.srt.model_executor.forward_batch_info.ForwardMode. Referenced only as
    a sibling annotation at import time; never the graded path. Members mirror
    upstream so any sibling read resolves. B-inert (pure enum)."""

    PREFILL = 0
    EXTEND = 1
    DECODE = 2
    IDLE = 3


class ForwardBatch(_SetupOnlyShim):
    """Inert stand-in for
    sglang.srt.model_executor.forward_batch_info.ForwardBatch (a request batch-
    state container, only a sibling parameter annotation on the graded path);
    loud-fails on any attribute access. B-inert."""


class VocabParallelEmbedding(_SetupOnlyShim):
    """Inert stand-in for
    sglang.srt.layers.vocab_parallel_embedding.VocabParallelEmbedding (the LM
    head module, only a sibling parameter annotation on the graded path);
    loud-fails on any attribute access. B-inert."""


class InputLogprobsResult(_SetupOnlyShim):
    """Inert stand-in for
    sglang.srt.layers.utils.logprob.InputLogprobsResult (a sibling logprob
    result container, only a return-type annotation on the graded path);
    loud-fails on any attribute access. B-inert."""


class _SglangEnvsShim:
    """Inert stand-in for the ``envs`` namespace of sglang.srt.environ. The
    frozen kernel reads ``envs.SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK.get()`` /
    ``envs.SGLANG_LOGITS_PROCESSER_CHUNK_SIZE.get()`` inside the sibling
    ``LogitsProcessor.__init__`` (never constructed on the graded path), so the
    name only needs to import. Any attribute access LOUD-FAILS: a verified
    entrypoint reading a real env flag is a true config dependency, never
    silently defaulted. B-inert (fail-closed)."""

    def __getattr__(self, name):
        raise AttributeError(
            f"sglang envs shim: 'envs.{name}' reached. Env-flag reads are "
            f"sibling-only (LogitsProcessor setup) and must not enter the "
            f"graded fused_softcap entrypoint's path."
        )


envs = _SglangEnvsShim()


# ───────────────────────────────────────────────────────────────────────────
# vllm fused_moe sibling surface (task L1_28_pack_bitmatrix).
#
# The frozen kernel.py for L1_28 is upstream's whole
# ``vllm/.../fused_moe/experts/gpt_oss_triton_kernels_moe.py``. The GRADED op is
# the tiny @triton.jit ``pack_bitmatrix`` (a pure bit-packing kernel using only
# torch/triton/tl), driven by ``_pa_launcher.py::run``. Everything below the jit
# -- the MoE expert classes (``BaseOAITritonExperts``, ``OAITritonExperts``,
# ``UnfusedOAITritonExperts``, ``OAITritonMxfp4ExpertsMonolithic``) and the host
# wrappers (``triton_kernel_moe_forward`` / ``triton_kernel_fused_experts`` /
# ``make_routing_data``) -- is the rest of the upstream MoE file and is NEVER on
# the graded pack_bitmatrix path.
#
# The frozen file has no ``from __future__ import annotations``, so the symbols
# used as CLASS BASES (``LoRAExpertsMixin``) and as DEF-TIME TYPE ANNOTATIONS
# (``MoEActivation``, the ``FusedMoE*Config`` classes, ``RoutingMethodType``,
# ``QuantKey``) must resolve to real objects for the module body to import
# standalone. The rest (``FUSED_MOE_UNQUANTIZED_CONFIG``, ``TopKWeightAndReduceNoOP``,
# ``_resize_cache``, ``kMxfp4Static``, ``ops`` (vllm._custom_ops),
# ``swiglu_limit_func``) are referenced only inside off-path bodies; the name
# only needs to bind. Type/enum stand-ins satisfy the annotations; behaviour-
# bearing callables LOUD-FAIL if invoked (fail-closed: a graded path reaching
# real MoE compute is a genuine dependency, never silently faked); the
# ``has_triton_kernels`` predicate returns its honest standalone value (False),
# matching that the severed surface has no triton_kernels package. B-inert.
# ───────────────────────────────────────────────────────────────────────────
class LoRAExpertsMixin:
    """Inert stand-in for
    ``vllm...fused_moe.experts.lora_experts_mixin.LoRAExpertsMixin`` -- a plain
    mixin base of the frozen file's off-path ``UnfusedOAITritonExperts``. A bare
    class (inherits only ``object``, no metaclass) so the multiple-inheritance
    ``class UnfusedOAITritonExperts(LoRAExpertsMixin, BaseOAITritonExperts)``
    statement executes at import time with a C3-consistent MRO; the subclass is
    never instantiated on the graded pack_bitmatrix path."""


class MoEActivation(enum.Enum):
    """Inert stand-in for ``vllm...fused_moe.activation.MoEActivation`` -- an
    activation-kind enum used by the frozen file as a def-time parameter
    annotation AND as class-level default values (e.g.
    ``activation: MoEActivation = MoEActivation.SWIGLUOAI``) on the off-path host
    wrappers / expert methods, so the member names + string values must mirror
    upstream verbatim (activation.py:16-29, commit 18f6bf5) for those def-time
    defaults to resolve. Never read on the graded pack_bitmatrix path."""

    SILU = "silu"
    GELU = "gelu"
    GELU_TANH = "gelu_tanh"
    RELU2 = "relu2"
    SWIGLUOAI = "swigluoai"
    SWIGLUSTEP = "swiglustep"
    SILU_NO_MUL = "silu_no_mul"
    GELU_NO_MUL = "gelu_no_mul"
    GELU_TANH_NO_MUL = "gelu_tanh_no_mul"
    RELU2_NO_MUL = "relu2_no_mul"


class RoutingMethodType(enum.IntEnum):
    """Inert stand-in for ``vllm...fused_moe.config.RoutingMethodType`` -- a
    router-kind IntEnum used by the frozen file as a def-time annotation AND as
    class-level default reads (``RoutingMethodType.Renormalize`` /
    ``.RenormalizeNaive``) on off-path methods. Member names + values mirror
    upstream verbatim (config.py:102-124, commit 18f6bf5), including the high
    sentinel values, so any redirected task reading any member resolves. Never
    reached on the graded path."""

    Default = 0
    Renormalize = 1
    DeepSeekV3 = 2
    Llama4 = 3
    RenormalizeNaive = 4
    TopK = 5
    SigmoidRenorm = 6
    MiniMax2 = 7
    Unspecified = 8
    DeepseekV4 = 100
    Custom = 101
    Simulated = 102


class QuantKey:
    """Inert stand-in for
    ``vllm...quantization.utils.quant_utils.QuantKey`` -- a quant-scheme key
    dataclass used by the frozen file ONLY as a def-time annotation on off-path
    ``_supports_quant_scheme`` helpers. A bare class satisfies the annotation;
    never constructed or compared on the graded path."""


class FusedMoEConfig(_SetupOnlyShim):
    """Inert stand-in for ``vllm...fused_moe.config.FusedMoEConfig`` -- an MoE
    config container, used by the frozen file ONLY as a def-time annotation on
    off-path expert methods. Loud-fails on any attribute access; never read on
    the graded pack_bitmatrix path. B-inert."""


class FusedMoEParallelConfig(_SetupOnlyShim):
    """Inert stand-in for ``vllm...fused_moe.config.FusedMoEParallelConfig`` --
    a parallelism config container, used only as a def-time annotation on
    off-path methods. Loud-fails on any access; never the graded path. B-inert."""


class FusedMoEQuantConfig(_SetupOnlyShim):
    """Inert stand-in for ``vllm...fused_moe.config.FusedMoEQuantConfig`` -- a
    quant config container, used only as a def-time annotation on off-path
    methods. Loud-fails on any access; never the graded path. B-inert."""


class _OpsShim:
    """Inert stand-in for ``vllm._custom_ops`` (imported as ``ops``). The frozen
    file calls ``ops.moe_sum`` ONLY inside the off-path ``moe_sum`` helper; the
    graded pack_bitmatrix path never touches it. Any attribute access LOUD-FAILS
    -- a verified path reaching a real custom CUDA op is a true dependency, never
    silently faked. B-inert (fail-closed)."""

    def __getattr__(self, name):
        raise NotImplementedError(
            f"vllm._custom_ops shim: 'ops.{name}' reached. Custom CUDA ops are "
            f"off-path (MoE expert helpers) and must not enter the graded "
            f"pack_bitmatrix entrypoint's path."
        )


ops = _OpsShim()

# Off-path value/instance bindings: referenced only inside off-path bodies, so
# the NAME must bind, but any real read/use LOUD-FAILS (fail-closed).
FUSED_MOE_UNQUANTIZED_CONFIG = _SetupOnlyShim()
kMxfp4Static = _SetupOnlyShim()


def _l1_28_sibling_only_call(_name: str):
    """Build a loud-failing stand-in for a vllm fused_moe helper that the frozen
    L1_28 kernel imports but only reaches on the off-graded-path MoE expert
    surface. The name must resolve so the module imports; invoking it means a
    verified entrypoint reached real MoE compute -- a true dependency to
    classify, never to fake. Fail-closed."""

    def _stub(*args, **kwargs):
        raise NotImplementedError(
            f"{_name} shim reached: this vllm fused_moe helper is off-path (MoE "
            f"expert / host-wrapper surface) and must not enter the graded "
            f"pack_bitmatrix entrypoint's path."
        )

    _stub.__name__ = _name
    _stub.__qualname__ = _name
    return _stub


# vllm...fused_moe.topk_weight_and_reduce.TopKWeightAndReduceNoOP -- constructed
# only inside the off-path ``finalize_weight_and_reduce_impl``. A loud-fail
# callable: the name binds, but constructing/calling it on the graded path
# raises.
TopKWeightAndReduceNoOP = _l1_28_sibling_only_call("TopKWeightAndReduceNoOP")

# vllm...fused_moe.utils._resize_cache -- a real workspace-buffer reshape called
# only inside the off-path ``apply`` / ``triton_kernel_fused_experts`` host
# wrappers; loud-fails if a graded path ever calls it.
_resize_cache = _l1_28_sibling_only_call("_resize_cache")

# ..utils.swiglu_limit_func (sibling fused_moe/utils.py) -- a swiglu activation
# helper called only inside the off-path ``activation`` method; loud-fails if
# the graded path ever calls it.
swiglu_limit_func = _l1_28_sibling_only_call("swiglu_limit_func")


def has_triton_kernels() -> bool:
    """Pure predicate mirroring
    ``vllm.utils.import_utils.has_triton_kernels`` -- True iff the optional
    ``triton_kernels`` package is importable. On the severed standalone surface
    that package is absent (and the audit forbids resolving vllm's own copy), so
    this is honestly False. The frozen kernel reads it at module scope
    (``if has_triton_kernels():``) to gate an optional ``triton_kernels.*``
    import block; returning False skips that block (the graded pack_bitmatrix
    path imports none of it), matching the standalone surface. B-inert."""
    return False


# ───────────────────────────────────────────────────────────────────────────
# L1_38_silu_mul_fp8_quant_deep_gemm off-path stand-ins. The frozen file is the
# whole vLLM batched_deep_gemm_moe.py; the graded unit is the @triton.jit
# _silu_mul_fp8_quant_deep_gemm driven by _pa_launcher.py::run (which
# reimplements the host-wrapper body verbatim). Every symbol below is imported
# at module scope ONLY for dead sibling defs (the host wrapper
# persistent_masked_m_silu_mul_quant, the BatchedDeepGemmExperts methods); the
# launched kernel references none of them (each is proven off the launcher's
# launched-jit closure). Names bind so the module imports standalone; any real
# use LOUD-FAILS (fail-closed -- a graded path reaching real DeepGEMM/MoE
# behaviour is a true dependency to classify, never silently fake).
# ───────────────────────────────────────────────────────────────────────────
class _FailClosedSentinel:
    """An inert sentinel whose EVERY interaction loud-fails -- attribute access,
    call, equality, and membership. Used for the ``kFp8*`` QuantKey constants the
    frozen file references only inside the off-path ``_supports_quant_scheme``
    ``(weight_key, activation_key) in SUPPORTED_W_A`` membership test. Unlike a
    plain ``_SetupOnlyShim`` (whose default ``__eq__`` would silently return a
    bool), this raises on ``==``/``in`` too, so a future on-path read can never
    silently resolve to a wrong scheme decision. The name only needs to bind on
    the graded path (the membership test is never reached)."""

    __slots__ = ("_name",)

    def __init__(self, _name: str = "_FailClosedSentinel") -> None:
        object.__setattr__(self, "_name", _name)

    def _fail(self, op: str):
        raise NotImplementedError(
            f"{object.__getattribute__(self, '_name')} shim: '{op}' "
            f"reached. This off-path vLLM quant-scheme sentinel must not enter a "
            f"verified entrypoint's path; a frozen kernel reaching real quant-key "
            f"comparison is a true dependency, not a clean standalone."
        )

    def __getattr__(self, name):
        self._fail(f".{name}")

    def __call__(self, *args, **kwargs):
        self._fail("call")

    def __eq__(self, other):
        self._fail("==")

    def __hash__(self):
        self._fail("hash")


# vllm...quantization.utils.quant_utils.kFp8Dynamic128Sym / kFp8Static128BlockSym
# -- module-scope QuantKey instances referenced only inside the off-path
# BatchedDeepGemmExperts._supports_quant_scheme membership test. Fail-closed
# sentinels (loud-fail on any use, including ==/in).
kFp8Dynamic128Sym = _FailClosedSentinel("kFp8Dynamic128Sym")
kFp8Static128BlockSym = _FailClosedSentinel("kFp8Static128BlockSym")

# vllm.forward_context.{get_forward_context, is_forward_context_available} --
# read only inside the off-path BatchedDeepGemmExperts.estimate_expected_m. Both
# loud-fail callables (fail-closed): the launcher never reaches forward-context
# state, so a graded path calling either is a real dependency.
get_forward_context = _l1_28_sibling_only_call("get_forward_context")
is_forward_context_available = _l1_28_sibling_only_call(
    "is_forward_context_available"
)

# vllm...fused_moe.topk_weight_and_reduce.TopKWeightAndReduceDelegate --
# constructed only inside the off-path finalize_weight_and_reduce_impl. Loud-fail
# callable (the name binds; constructing it on the graded path raises).
TopKWeightAndReduceDelegate = _l1_28_sibling_only_call(
    "TopKWeightAndReduceDelegate"
)

# vllm.utils.deep_gemm.{fp8_m_grouped_gemm_nt_masked, is_deep_gemm_e8m0_used,
# is_deep_gemm_supported} -- the DeepGEMM accel surface, reached only inside the
# off-path apply / _supports_current_device / supports_packed_ue8m0_act_scales.
# All loud-fail (fail-closed): we cannot prove the off-path capability semantics
# and the launcher never runs them, so an honest False/True would silently alter
# off-path logic. (get_mk_alignment_for_contiguous_layout is already defined
# above as the L2_26 loud-fail stand-in and is reused here.)
fp8_m_grouped_gemm_nt_masked = _l1_28_sibling_only_call(
    "fp8_m_grouped_gemm_nt_masked"
)
is_deep_gemm_e8m0_used = _l1_28_sibling_only_call("is_deep_gemm_e8m0_used")
is_deep_gemm_supported = _l1_28_sibling_only_call("is_deep_gemm_supported")

# vllm.utils.deep_gemm.{get_tma_aligned_size, transform_sf_into_required_layout}
# (task L2_11_w8a8_triton_block_scaled_mm). DeepGEMM scale-layout helpers reached
# only inside the off-path fp8 weight-prep / requant functions (get_tma_aligned_size
# at kernel.py:556, transform_sf_into_required_layout at :1041,:1063); the graded
# launcher drives _w8a8_triton_block_scaled_mm directly and never touches them.
# Fail-closed: an honest layout transform we cannot prove would silently alter
# off-path logic, so both loud-fail if a graded path ever reaches them.
get_tma_aligned_size = _l1_28_sibling_only_call("get_tma_aligned_size")
transform_sf_into_required_layout = _l1_28_sibling_only_call(
    "transform_sf_into_required_layout"
)


class DeepGemmQuantScaleFMT(enum.Enum):
    """Inert stand-in for ``vllm.utils.deep_gemm.DeepGemmQuantScaleFMT`` -- the
    quant-scale layout enum. Members mirror upstream verbatim (deep_gemm.py,
    commit 18f6bf5) so the frozen file's DEF-TIME default value
    ``quant_scale_fmt: DeepGemmQuantScaleFMT = DeepGemmQuantScaleFMT.FLOAT32`` on
    the off-path host wrapper ``persistent_masked_m_silu_mul_quant`` resolves at
    import time. The members are never compared on the graded path (the launcher
    drives the @triton.jit kernel directly with ceil_ue8m0=False); the oracle
    decision classmethods loud-fail (fail-closed -- a graded path selecting a
    scale format at runtime is a real dependency)."""

    FLOAT32 = 0
    FLOAT32_CEIL_UE8M0 = 1
    UE8M0 = 2

    @classmethod
    def init_oracle_cache(cls) -> None:
        raise NotImplementedError(
            "DeepGemmQuantScaleFMT.init_oracle_cache shim reached: the oracle "
            "scale-format decision is off the graded path (the launcher pins the "
            "FLOAT32 format with ceil_ue8m0=False) and must not be selected at "
            "runtime on a verified entrypoint."
        )

    @classmethod
    def from_oracle(cls) -> "DeepGemmQuantScaleFMT":
        raise NotImplementedError(
            "DeepGemmQuantScaleFMT.from_oracle shim reached: the oracle "
            "scale-format decision is off the graded path (the launcher pins the "
            "FLOAT32 format with ceil_ue8m0=False) and must not be selected at "
            "runtime on a verified entrypoint."
        )


# ── L2_11_w8a8_triton_block_scaled_mm off-path weight-management surface ──────
# The frozen kernel.py copies the whole vLLM block-fp8 utils module; the graded
# unit is the @triton.jit _w8a8_triton_block_scaled_mm, driven directly by
# _pa_launcher.py. The symbols below are imported at module scope but referenced
# ONLY inside off-path fp8 weight-loading / quant helpers (create_fp8_* and the
# per-tensor dequant cross-checks), never in the launched-jit closure. Each is
# redirected here, gated by phase_a_prime._symbol_off_launched_closure.


def direct_register_custom_op(
    op_name, op_func, mutates_args=None, fake_impl=None, *args, **kwargs
):
    """Inert no-op mirroring ``vllm.utils.torch_utils.direct_register_custom_op``.
    The frozen kernel.py calls this at MODULE IMPORT time (kernel.py:87) to
    register the off-path ``triton_per_token_group_quant_fp8`` custom op (backed
    by a different sibling quant impl). The graded launcher invokes the
    @triton.jit kernel directly and never dispatches through that op, so skipping
    the registration removes only an off-path import-time side effect and changes
    no graded numeric output. Must NOT raise -- it runs unconditionally at import.
    B-inert (no registry side effect)."""
    return None


# Off-path fp8 weight-loading Parameter subclasses
# (vllm.model_executor.parameter) -- constructed only inside the off-path
# create_fp8_scale_parameter / create_fp8_input_scale helpers (kernel.py:1202,
# 1208, 1222, 1242). The name must bind for import; constructing one on the
# graded path loud-fails (fail-closed).
BlockQuantScaleParameter = _l1_28_sibling_only_call("BlockQuantScaleParameter")
ChannelQuantScaleParameter = _l1_28_sibling_only_call(
    "ChannelQuantScaleParameter"
)
PerTensorScaleParameter = _l1_28_sibling_only_call("PerTensorScaleParameter")

# vllm.model_executor.utils.set_weight_attrs -- attaches loader metadata to a
# weight Parameter, called only inside the off-path create_fp8_input_scale helper
# (kernel.py:1232). Loud-fails if a graded path ever reaches it.
set_weight_attrs = _l1_28_sibling_only_call("set_weight_attrs")

# vllm.model_executor.layers.quantization.utils.w8a8_utils.{all_close_1d,
# per_tensor_dequantize} -- weight-comparison / per-tensor dequant helpers reached
# only inside the off-path process_fp8_input_tensor_strategy_moe path
# (kernel.py:1336, 1353). Loud-fail (fail-closed): the launcher never dequantises
# per-tensor scales.
all_close_1d = _l1_28_sibling_only_call("all_close_1d")
per_tensor_dequantize = _l1_28_sibling_only_call("per_tensor_dequantize")


# ───────────────────────────────────────────────────────────────────────────
# L1_15_act_and_mul off-path stand-ins. The frozen kernel.py is the whole
# SGLang fused_moe_triton_kernels.py; the graded unit is the @triton.jit
# `act_and_mul_kernel`, driven by `_pa_launcher.py::run`. Every symbol below is
# imported at module scope, but each is referenced ONLY by module-scope setup
# (platform/env gates) or by the dead sibling host wrappers
# (`invoke_fused_moe_kernel` / `should_enable_swap_ab`); the launched
# `act_and_mul_kernel` closure references NONE of them. The module-scope
# predicates return their honest bench-target value (CUDA verify host / Ascend
# NPU) so the standalone imports + runs; the dead-sibling-only quant helpers
# loud-fail (fail-closed). B-inert.
# ───────────────────────────────────────────────────────────────────────────
def get_moe_padding_size(is_aiter_moe) -> int:
    """Verbatim from sglang.srt.layers.moe.utils.get_moe_padding_size
    (AITER_PADDING_SIZE / TRITON_PADDING_SIZE both == 128). The bench graded
    path is the non-aiter Triton MoE with the default SGLANG_MOE_PADDING env
    (unset -> "0"), so this returns 0. Read at module scope
    (``padding_size = get_moe_padding_size(_use_aiter)``); the value is then
    consumed ONLY inside the dead ``invoke_fused_moe_kernel`` sibling, never
    the graded ``act_and_mul_kernel`` path. B-inert (faithful to upstream)."""
    _AITER_PADDING_SIZE = 128
    _TRITON_PADDING_SIZE = 128
    if is_aiter_moe:
        return _AITER_PADDING_SIZE
    return (
        _TRITON_PADDING_SIZE
        if bool(int(os.getenv("SGLANG_MOE_PADDING", "0")))
        else 0
    )


def cpu_has_amx_support() -> bool:
    """Pure backend predicate mirroring sglang.srt.utils.common.cpu_has_amx_support
    (AMX tile support AND the intel-amx sgl_kernel backend). The bench verify
    host is NVIDIA CUDA / Ascend NPU, neither an Intel-AMX CPU, so this is a
    constant False. Read only at module scope
    (``_is_cpu_amx_available = cpu_has_amx_support()``); the graded
    act_and_mul_kernel path never reads it. B-inert."""
    return False


def is_cpu() -> bool:
    """Pure backend predicate mirroring sglang.srt.utils.common.is_cpu -- True
    only when SGLANG_USE_CPU_ENGINE=1 on a supported x86/arm64 host. The bench
    runs the standalone on the GPU device, so this is False. Module-scope setup
    gate only (``_is_cpu = is_cpu()``); never the graded path. B-inert."""
    return False


def get_bool_env_var(name: str, default: str = "false") -> bool:
    """Faithful mirror of sglang.srt.utils.common.get_bool_env_var -- a real
    env-var truthy parse (no platform/dtype branch). Must run at import time
    (``_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip``), so a
    loud-fail stub would break the module load; a faithful parser is the
    correct severance. Unrecognised values are treated as false (upstream warns
    once; the shim silently treats-as-false, identical return value)."""
    value = os.getenv(name, default).lower()
    return value in ("true", "1")


def is_sm90_supported() -> bool:
    """Live mirror of sglang.srt.utils.common.is_sm90_supported -- upstream's
    ``_check_cuda_device_version(device_capability_majors=[9], cuda_version=
    (12, 3))``: True iff the CUDA device major capability is EXACTLY 9 (SM90,
    Hopper) AND the CUDA toolkit is >= 12.3. Read ONLY inside the dead sibling
    ``should_enable_swap_ab`` (never the graded act_and_mul_kernel path), so the
    value is B-inert; we still answer honestly from the live device (False
    off-GPU / on the CPU oracle pre-filter, where there is no SM90 device).
    B-inert."""
    if not (torch.cuda.is_available() and torch.version.cuda is not None):
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        cuda_major, cuda_minor = (
            int(x) for x in torch.version.cuda.split(".")[:2]
        )
        return major == 9 and (cuda_major, cuda_minor) >= (12, 3)
    except Exception:
        return False


def is_batch_invariant_mode_enabled() -> bool:
    """Faithful mirror of
    sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled -- upstream
    returns a module-global flag (``_batch_invariant_MODE``) that defaults to
    False and is only flipped True by an explicit ``enable_batch_invariant_mode``
    server bootstrap. The bench runs the standalone with no such bootstrap, so
    the honest value is False. Read ONLY inside the dead sibling
    ``should_enable_swap_ab`` (never the graded path). B-inert."""
    return False


def _l1_15_int8_quant(_name: str):
    """Build a loud-failing stand-in for an sglang int8 per-token/group quant
    kernel that the frozen L1_15 kernel imports but only reaches inside the dead
    sibling ``invoke_fused_moe_kernel`` (int8 quant branch). The name must bind
    so the module imports; invoking it means a verified entrypoint reached a
    real int8 quant kernel -- a true dependency to classify, never to fake.
    Fail-closed."""

    def _stub(*args, **kwargs):
        raise NotImplementedError(
            f"{_name} shim reached: this sglang int8 quant kernel is off-path "
            f"(the dead invoke_fused_moe_kernel sibling) and must not enter the "
            f"graded act_and_mul_kernel entrypoint's path."
        )

    _stub.__name__ = _name
    _stub.__qualname__ = _name
    return _stub


# sglang.srt.layers.quantization.int8_kernel.{per_token_group_quant_int8,
# per_token_quant_int8, sglang_per_token_group_quant_int8} -- real CUDA int8
# quant kernels called ONLY inside the dead invoke_fused_moe_kernel sibling
# (kernel.py:785/791/793); loud-fail if a graded path ever reaches them.
per_token_group_quant_int8 = _l1_15_int8_quant("per_token_group_quant_int8")
per_token_quant_int8 = _l1_15_int8_quant("per_token_quant_int8")
sglang_per_token_group_quant_int8 = _l1_15_int8_quant(
    "sglang_per_token_group_quant_int8"
)


def softplus(*args, **kwargs):
    """Loud-fail stand-in for the upstream Mamba2 ``mamba_ssm.softplus``
    @triton.jit device function (``from .mamba_ssm import softplus``).

    The frozen chunk_state kernels (e.g. L3_03 ``_chunk_state_varlen_kernel``,
    L2_03 ``_chunk_state_fwd_kernel``) carry this import at module scope, but
    ``softplus`` is referenced ONLY inside the sibling ``_chunk_cumsum_fwd_kernel``
    -- which the chunk_state launcher never launches (proven by the launched-jit
    closure: the launcher drives ``chunk_state_varlen`` / ``_chunk_state_fwd``,
    not ``_chunk_cumsum_fwd``). The Phase A' rewriter redirects the off-path
    import here so the severed standalone resolves without pulling the real
    ``mamba_ssm`` module (which imports ``_custom_ops`` and the vllm/sglang
    closure). Importing the name is inert; calling it raises -- so a graded path
    that ever reached it (e.g. a cumsum task whose kernel DOES call softplus,
    which keeps the REAL import via the on-path gate) fails loudly here rather
    than silently faking the activation. Fail-closed."""
    raise NotImplementedError(
        "softplus shim reached: the upstream mamba_ssm softplus is referenced "
        "only by the off-path _chunk_cumsum_fwd_kernel sibling, which the "
        "chunk_state launcher never launches; it is not vendored into the "
        "severed standalone and the graded chunk_state kernel must not invoke it"
    )
