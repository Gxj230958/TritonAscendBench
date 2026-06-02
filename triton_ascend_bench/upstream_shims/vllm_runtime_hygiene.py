"""Runtime hygiene helpers for source-checkout vLLM imports.

Phase A' loads vendored kernels out of ``tasks/.../kernel.py`` while
putting ``infer/vllm`` directly on ``sys.path``. That differs from a
wheel install in two ways that matter for oracle/S2 verify:

1. ``vllm.platforms`` asks ``importlib.metadata.version("vllm")``
   during CUDA plugin detection. A source checkout has no dist-info, so
   the call raises ``PackageNotFoundError`` and vLLM falls back to
   ``UnspecifiedPlatform`` even on a real CUDA host.
2. Some vendored kernels re-register vLLM ``CustomOp`` names that may
   already exist in the process-global registry.

These helpers patch only the subprocess-local import surface used by
the bench runners; vendored vLLM / SGLang source stays untouched.
"""
from __future__ import annotations

import importlib.util
from importlib.machinery import EXTENSION_SUFFIXES, ModuleSpec
import re
import sys
import types
from importlib import metadata as importlib_metadata
from pathlib import Path

_REGISTER_RE = re.compile(
    r"@\s*(?:[\w.]+\.)?(?:CustomOp|PluggableLayer)\.register"
    r"\(\s*[\"']([^\"']+)[\"']\s*\)"
)
# Marker for a module-level ``direct_register_custom_op(...)`` call. Unlike the
# decorator forms above, this registers into a torch.library ``Library`` (the
# process-global ``vllm_lib``), so a second exec of the same vendored kernel.py
# (the oracle anchor loads it once per anchor shape) hits torch's hard
# "registered ... multiple times" error -- which clearing vLLM's op_registry
# cannot fix. Presence of this token gates installing the idempotent wrapper.
_DIRECT_REGISTER_TOKEN = "direct_register_custom_op("
_VLLM_C_EXTENSIONS = ("vllm._C", "vllm._C_stable_libtorch")

# Bool cutlass capability probes that a version-skewed daemon's compiled _C
# leaves unregistered, so a transitive import evaluating them AttributeErrors at
# load time. They only gate cutlass-vs-triton dispatch, never the benched
# compute. Schemas verbatim from infer/vllm csrc torch_bindings.cpp. Compute ops
# stay out so a real cutlass dependency still fails loudly.
_CUTLASS_PROBE_SCHEMAS = {
    "cutlass_scaled_mm_supports_fp8": (
        "cutlass_scaled_mm_supports_fp8(int cuda_device_capability) -> bool"
    ),
    "cutlass_scaled_mm_supports_block_fp8": (
        "cutlass_scaled_mm_supports_block_fp8(int cuda_device_capability) "
        "-> bool"
    ),
    "cutlass_scaled_mm_supports_fp4": (
        "cutlass_scaled_mm_supports_fp4(int cuda_device_capability) -> bool"
    ),
    "cutlass_group_gemm_supported": (
        "cutlass_group_gemm_supported(int cuda_device_capability) -> bool"
    ),
}

# Module-global: torch drops a Library's ops via weakref.finalize once the last
# ref goes, so a function-local one would vanish. The name set keeps re-calls
# idempotent (a duplicate .define raises).
_CUTLASS_PROBE_LIB = None
_CUTLASS_PROBES_REGISTERED: set[str] = set()


def _read_vllm_source_version(vllm_root: Path) -> str:
    version_py = vllm_root / "vllm" / "version.py"
    if not version_py.is_file():
        return "dev"
    spec = importlib.util.spec_from_file_location(
        "_tab_vllm_source_version", version_py
    )
    if spec is None or spec.loader is None:
        return "dev"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return str(getattr(mod, "__version__", "dev"))


def install_vllm_source_version_shim(vllm_root: Path) -> bool:
    """Patch ``importlib.metadata.version`` for source-checkout vLLM.

    Returns ``True`` when the shim was installed in this process.
    """
    root_str = str(vllm_root)
    if root_str not in sys.path:
        return False
    if not (vllm_root / "vllm" / "version.py").is_file():
        return False

    current = importlib_metadata.version
    if getattr(current, "__tab_vllm_source_shim__", False):
        return True

    fallback_version = _read_vllm_source_version(vllm_root)
    package_not_found = importlib_metadata.PackageNotFoundError

    def _version(name: str) -> str:
        if name != "vllm":
            return current(name)
        try:
            return current(name)
        except package_not_found:
            return fallback_version

    _version.__tab_vllm_source_shim__ = True
    _version.__wrapped__ = current
    importlib_metadata.version = _version
    return True


def _has_compiled_vllm_extension(vllm_root: Path, module_name: str) -> bool:
    pkg_dir = vllm_root / "vllm"
    stem = module_name.rsplit(".", 1)[-1]
    for suffix in EXTENSION_SUFFIXES:
        if any(pkg_dir.glob(f"{stem}{suffix}")):
            return True
    return False


def _make_missing_vllm_extension_stub(
    module_name: str, vllm_root: Path,
) -> types.ModuleType:
    stub = types.ModuleType(module_name)
    stub.__file__ = "<tab-vllm-extension-stub>"
    stub.__package__ = "vllm"
    stub.__tab_stub__ = True
    stub.__tab_missing_extension__ = module_name

    def __getattr__(attr: str):
        raise AttributeError(
            f"{module_name}.{attr} is unavailable because the compiled "
            f"extension {module_name!r} is missing from the source "
            f"checkout at {vllm_root}"
        )

    stub.__getattr__ = __getattr__
    return stub


def install_vllm_c_stub(vllm_root: Path) -> bool:
    """Inject safe stubs for source-checkout vLLM C extensions.

    ``vllm.platforms.cuda`` imports ``vllm._C`` /
    ``vllm._C_stable_libtorch`` only for registration side effects. A
    source checkout without compiled ``*.so`` files raises
    ``ModuleNotFoundError`` before platform detection can proceed. This
    helper installs module stubs only when the source tree is on
    ``sys.path`` AND the compiled extension file is absent from that
    tree. Any later attribute access still fails loudly at the call
    site via ``AttributeError``.
    """
    root_str = str(vllm_root)
    if root_str not in sys.path:
        return False
    if not (vllm_root / "vllm" / "platforms" / "cuda.py").is_file():
        return False

    installed = False
    for module_name in _VLLM_C_EXTENSIONS:
        if module_name in sys.modules:
            continue
        if _has_compiled_vllm_extension(vllm_root, module_name):
            continue
        sys.modules[module_name] = _make_missing_vllm_extension_stub(
            module_name, vllm_root
        )
        installed = True
    return installed


# FlashInfer import surface needed by the whole-module vendored
# vllm/v1/attention/backends/flashinfer.py (task L1_04_copy_page_indices). The
# graded @triton.jit `_copy_page_indices_kernel` is a pure index copy that
# touches NONE of these symbols -- they feed only the dead FlashInfer* sibling
# classes in the same upstream file. The stub provides the exact names the
# module imports at top level so it loads; every name is an inert object that
# RAISES on instantiation/call, so a graded path reaching one fails loudly
# (never silently faked). Keyed off the EXACT kernel.py import lines (verbatim,
# commit 18f6bf5).
_FLASHINFER_STUB_NAMES: dict[str, tuple[str, ...]] = {
    "flashinfer": (
        "BatchDecodeWithPagedKVCacheWrapper",
        "BatchPrefillWithPagedKVCacheWrapper",
        "BatchPrefillWithRaggedKVCacheWrapper",
        "MultiLevelCascadeAttentionWrapper",
    ),
    "flashinfer.decode": (
        "fast_decode_plan",
        "trtllm_batch_decode_with_kv_cache",
    ),
    "flashinfer.prefill": ("trtllm_batch_context_with_kv_cache",),
    "flashinfer.utils": ("FP4Tensor",),
}
_FLASHINFER_STUB_ATTR = "__tab_flashinfer_stub__"


def _make_flashinfer_inert(qualname: str):
    """An inert stand-in for an off-graded-path FlashInfer symbol. The name
    resolves so the vendored module imports, but instantiating/calling it
    RAISES -- a graded path reaching real FlashInfer is a true dependency, never
    silently faked."""

    class _Inert:
        __tab_inert__ = True

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                f"{qualname} is an inert FlashInfer stub (off the graded "
                f"_copy_page_indices_kernel path); the bench does not vendor the "
                f"real FlashInfer wrapper into this surface."
            )

    _Inert.__name__ = qualname.rsplit(".", 1)[-1]
    _Inert.__qualname__ = _Inert.__name__
    return _Inert


def _real_flashinfer_present() -> bool:
    """True iff a REAL (non-stub) ``flashinfer`` package is importable. Guards
    the stub install so a GPU daemon with FlashInfer actually installed keeps
    the real package."""
    existing = sys.modules.get("flashinfer")
    if existing is not None:
        return not getattr(existing, _FLASHINFER_STUB_ATTR, False)
    try:
        return importlib.util.find_spec("flashinfer") is not None
    except (ImportError, ValueError):
        return False


def _install_flashinfer_find_spec_lie() -> None:
    """Make availability probes (``importlib.util.find_spec("flashinfer")``)
    see the TAB stub as ABSENT, so vllm/utils/flashinfer.py's
    ``has_flashinfer()`` returns False even on an nvcc-equipped daemon. Without
    this a spec-bearing stub would make ``has_flashinfer()`` lie True and run
    the module-level ``if has_flashinfer():`` registration block -- an
    availability lie. Idempotent; delegates for every non-stub module."""
    current = importlib.util.find_spec
    if getattr(current, "__tab_flashinfer_spec_shim__", False):
        return

    def _find_spec(name, package=None):
        base = name.split(".", 1)[0]
        mod = sys.modules.get(base)
        if mod is not None and getattr(mod, _FLASHINFER_STUB_ATTR, False):
            # The TAB stub is installed for this name -> report it ABSENT so
            # has_flashinfer() / has_flashinfer_cubin() probes answer honestly.
            return None
        return current(name, package)

    _find_spec.__tab_flashinfer_spec_shim__ = True
    _find_spec.__wrapped__ = current
    importlib.util.find_spec = _find_spec


def install_flashinfer_stub() -> bool:
    """Inject an inert ``flashinfer`` import surface for source-checkout vLLM
    kernels that import FlashInfer at module top level only for dead sibling
    classes (task L1_04_copy_page_indices).

    Installs spec-bearing stub modules for ``flashinfer`` and its
    ``decode``/``prefill``/``utils`` submodules, exposing exactly the names the
    vendored kernel imports as inert raise-on-use objects, AND installs a
    ``find_spec`` wrapper so availability probes still see FlashInfer as absent
    (``has_flashinfer()`` stays False). Skips entirely when a REAL FlashInfer is
    present (GPU daemon with the package) -- the real surface then loads and the
    upstream availability path runs unchanged. Returns ``True`` iff the stub was
    installed this call."""
    if _real_flashinfer_present():
        return False
    if sys.modules.get("flashinfer") is not None and getattr(
        sys.modules["flashinfer"], _FLASHINFER_STUB_ATTR, False
    ):
        # Already installed by us in this process -> keep the find_spec lie too.
        _install_flashinfer_find_spec_lie()
        return False

    for module_name, names in _FLASHINFER_STUB_NAMES.items():
        is_pkg = module_name == "flashinfer"
        spec = ModuleSpec(module_name, loader=None, is_package=is_pkg)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = "<tab-flashinfer-stub>"
        setattr(mod, _FLASHINFER_STUB_ATTR, True)
        if is_pkg:
            mod.__path__ = []  # mark as a package for submodule resolution
        for n in names:
            setattr(mod, n, _make_flashinfer_inert(f"{module_name}.{n}"))
        sys.modules[module_name] = mod

    # Probes must see the stub as absent so has_flashinfer() answers honestly.
    _install_flashinfer_find_spec_lie()
    return True


# ``vllm.vllm_flash_attn`` is vLLM's vendored FlashAttention package. Its
# ``__init__.py`` imports the compiled ``_vllm_fa2_C`` / ``_vllm_fa3_C``
# extensions and, when BOTH are absent (a source checkout / dev box / a GPU
# daemon whose vLLM was not built with flash-attn), raises a hard
# ``ImportError("vllm.vllm_flash_attn requires the CUDA flash attention
# extensions ...")`` at module load. Whole-file vendored kernels that mirror
# an upstream model module (e.g. L2_18 Qwen3-Next, whose graded
# ``fused_gdn_gating`` is a pure elementwise GDN gate) transitively reach that
# package through a dead attention-backend import even though the graded
# @triton.jit kernel never touches flash-attn -- so the off-graded-path import
# blocks the whole candidate at load. The public surface ``__init__`` re-exports
# (verbatim from the vendored __init__.__all__, commit 18f6bf5).
_VLLM_FLASH_ATTN_STUB_ATTR = "__tab_vllm_flash_attn_stub__"
_VLLM_FLASH_ATTN_PUBLIC_NAMES = (
    "fa_version_unsupported_reason",
    "flash_attn_varlen_func",
    "get_scheduler_metadata",
    "is_fa_version_supported",
)


def _make_vllm_flash_attn_inert(qualname: str):
    """An inert stand-in for an off-graded-path ``vllm.vllm_flash_attn``
    symbol. The name resolves so the vendored module imports, but CALLING it
    RAISES -- a graded path that actually invokes flash-attn is a true
    dependency, never silently faked."""

    def _inert(*args, **kwargs):
        raise RuntimeError(
            f"{qualname} is an inert vllm.vllm_flash_attn stub (off the "
            f"graded kernel path); the compiled flash-attn extensions "
            f"(_vllm_fa2_C / _vllm_fa3_C) are absent from this checkout, so "
            f"the bench does not provide a real flash-attn implementation here."
        )

    _inert.__name__ = qualname.rsplit(".", 1)[-1]
    _inert.__qualname__ = _inert.__name__
    _inert.__tab_inert__ = True
    return _inert


def _discover_vllm_root() -> Path | None:
    """Locate the source-checkout vLLM root on ``sys.path`` -- the directory
    whose ``vllm/vllm_flash_attn/__init__.py`` exists. The bench runners insert
    ``infer/vllm`` onto ``sys.path`` before loading kernel.py, so the flash-attn
    stub can find the checkout even when called without an explicit root (from
    ``clear_conflicting_vllm_registrations``). Returns None when no such root is
    on the path (a wheel install, or vLLM absent)."""
    for entry in sys.path:
        if not entry:
            continue
        try:
            cand = Path(entry) / "vllm" / "vllm_flash_attn" / "__init__.py"
        except (TypeError, ValueError):
            continue
        if cand.is_file():
            return Path(entry)
    return None


def _real_vllm_flash_attn_present(vllm_root: Path | None) -> bool:
    """True iff a REAL (non-stub) ``vllm.vllm_flash_attn`` is importable --
    i.e. a compiled ``_vllm_fa2_C`` or ``_vllm_fa3_C`` extension exists in the
    checkout, OR the package is already loaded as a non-stub. Guards the stub
    install so a GPU daemon whose vLLM WAS built with flash-attn keeps the real
    package and its upstream availability path runs unchanged."""
    existing = sys.modules.get("vllm.vllm_flash_attn")
    if existing is not None:
        return not getattr(existing, _VLLM_FLASH_ATTN_STUB_ATTR, False)
    if vllm_root is None:
        return False
    fa_pkg = vllm_root / "vllm" / "vllm_flash_attn"
    for stem in ("_vllm_fa2_C", "_vllm_fa3_C"):
        for suffix in EXTENSION_SUFFIXES:
            if any(fa_pkg.glob(f"{stem}{suffix}")):
                return True
    return False


def install_vllm_flash_attn_stub(vllm_root: Path | None = None) -> bool:
    """Inject an inert ``vllm.vllm_flash_attn`` import surface for
    source-checkout vLLM kernels that transitively reach the package at module
    load only through a dead attention-backend import (task L2_18, same family
    as L1_29).

    Registers an inert ``vllm.vllm_flash_attn`` module exposing exactly the
    names the upstream ``__init__`` re-exports (``__all__``) as inert
    raise-on-CALL objects, plus ``FA2_AVAILABLE = FA3_AVAILABLE = False`` (the
    honest answer for a checkout without the compiled extensions). The hard
    ``raise ImportError`` in the real ``__init__`` therefore never fires, but
    any graded path that actually CALLS a flash-attn function fails loudly
    (never silently faked). Also stubs the ``flash_attn_interface`` submodule
    the package re-exports from, for the same reason.

    Skips entirely when a REAL flash-attn is present (a daemon whose vLLM was
    built with ``_vllm_fa2_C`` / ``_vllm_fa3_C``) -- the real surface then loads
    and the upstream path is untouched. ``vllm_root`` defaults to autodiscovery
    from ``sys.path`` (the runners insert ``infer/vllm`` before kernel load), so
    this can be called from ``clear_conflicting_vllm_registrations`` -- the
    single shared loader hook -- without threading a root through every call
    site. Skips when no source-checkout vLLM is on the path (wheel install /
    vLLM absent). Idempotent. Returns ``True`` iff the stub was installed this
    call."""
    if vllm_root is None:
        vllm_root = _discover_vllm_root()
        if vllm_root is None:
            # No source-checkout vLLM on sys.path: nothing transitively reaches
            # this package's source __init__, so there is no leak to close (a
            # wheel install ships the compiled extensions or its own surface).
            return False
    if _real_vllm_flash_attn_present(vllm_root):
        return False
    if sys.modules.get("vllm.vllm_flash_attn") is not None and getattr(
        sys.modules["vllm.vllm_flash_attn"], _VLLM_FLASH_ATTN_STUB_ATTR, False
    ):
        return False  # already installed by us in this process

    pkg = types.ModuleType("vllm.vllm_flash_attn")
    pkg.__file__ = "<tab-vllm-flash-attn-stub>"
    pkg.__package__ = "vllm.vllm_flash_attn"
    pkg.__path__ = []  # mark as a package for submodule resolution
    setattr(pkg, _VLLM_FLASH_ATTN_STUB_ATTR, True)
    pkg.FA2_AVAILABLE = False
    pkg.FA3_AVAILABLE = False
    for n in _VLLM_FLASH_ATTN_PUBLIC_NAMES:
        setattr(pkg, n, _make_vllm_flash_attn_inert(f"vllm.vllm_flash_attn.{n}"))

    iface = types.ModuleType("vllm.vllm_flash_attn.flash_attn_interface")
    iface.__file__ = "<tab-vllm-flash-attn-stub>"
    iface.__package__ = "vllm.vllm_flash_attn"
    setattr(iface, _VLLM_FLASH_ATTN_STUB_ATTR, True)
    iface.FA2_AVAILABLE = False
    iface.FA3_AVAILABLE = False
    for n in _VLLM_FLASH_ATTN_PUBLIC_NAMES:
        setattr(
            iface, n,
            _make_vllm_flash_attn_inert(
                f"vllm.vllm_flash_attn.flash_attn_interface.{n}"
            ),
        )

    sys.modules["vllm.vllm_flash_attn"] = pkg
    sys.modules["vllm.vllm_flash_attn.flash_attn_interface"] = iface
    return True


def install_vllm_capability_probe_stubs() -> list[str]:
    """Register MISSING vLLM ``_C`` bool capability probes as ``-> False``.

    A version-skewed daemon leaves the ``torch.ops._C`` namespace present
    but missing the cutlass capability probes (see ``_CUTLASS_PROBE_SCHEMAS``).
    A transitive import that evaluates them at kernel-load raises
    ``AttributeError`` and the whole candidate fails to invoke. We register
    only the ABSENT probes (``hasattr`` guard -> never clobber a real op the
    compiled ``.so`` already provides) with a conservative ``False``
    implementation, matching the vendored wrapper that already treats a
    missing ``cutlass_group_gemm_supported`` as ``False``.

    ``False`` is the honest answer for a build that cannot prove the cutlass
    path exists, and it is dispatch-neutral for the bench: the resolved
    triton entrypoint is called directly, so the probe's value never reaches
    the benched compute. Returns the op names newly registered this call.
    Fail-safe: any torch/library error is swallowed (returns what it managed)
    so the hygiene step never breaks a runner.
    """
    global _CUTLASS_PROBE_LIB
    newly: list[str] = []
    try:
        import torch

        for name, schema in _CUTLASS_PROBE_SCHEMAS.items():
            if name in _CUTLASS_PROBES_REGISTERED:
                continue  # already registered by us in this process
            if hasattr(torch.ops._C, name):
                continue  # provided by the real compiled extension
            if _CUTLASS_PROBE_LIB is None:
                # FRAGMENT extends the pre-existing ``_C`` namespace without
                # clobbering it; kept module-global for lifetime (above).
                _CUTLASS_PROBE_LIB = torch.library.Library("_C", "FRAGMENT")
            _CUTLASS_PROBE_LIB.define(schema)
            # int -> bool probe has no tensor args; CompositeExplicitAutograd
            # is the correct catch-all key (a backend key like CPU would have
            # nothing to dispatch on).
            _CUTLASS_PROBE_LIB.impl(
                name,
                lambda _cuda_device_capability: False,
                "CompositeExplicitAutograd",
            )
            _CUTLASS_PROBES_REGISTERED.add(name)
            newly.append(name)
    except Exception:
        # Never let a hygiene helper break the runner; a probe still missing
        # will surface as the same loud AttributeError it would have anyway.
        return newly
    return newly


def install_vllm_num_compute_units_stub() -> bool:
    """Fall back vLLM ``num_compute_units`` to 128 when it raises on the daemon.

    Un-severed kernels import it for grid sizing (scheduling-only, never the
    numeric path), so a constant SM count is dispatch-neutral; matches
    ``tasks/_frozen_shim``. Must run before the kernel's ``from ... import``.
    Fail-safe: errors are swallowed.
    """
    try:
        import vllm.utils.platform_utils as _pu

        _orig = getattr(_pu, "num_compute_units", None)
        if _orig is None or getattr(_orig, "_tab_ncu_wrapped", False):
            return _orig is not None  # absent or already wrapped (idempotent)

        def _safe_num_compute_units(device_id: int = 0, _orig=_orig):
            try:
                return _orig(device_id)
            except NotImplementedError:
                return 128

        _safe_num_compute_units._tab_ncu_wrapped = True
        _pu.num_compute_units = _safe_num_compute_units
        return True
    except Exception:
        return False


def install_idempotent_direct_register() -> bool:
    """Make ``vllm.utils.torch_utils.direct_register_custom_op`` idempotent.

    The oracle anchor (``_pa_oracle_check_runner``) loads a vendored
    ``kernel.py`` once per anchor shape in a single subprocess. When that
    kernel mirrors an upstream module with a module-level
    ``direct_register_custom_op(op_name=...)`` call, the second exec re-runs
    ``vllm_lib.define(...)`` on the process-global library and torch raises
    "Tried to register an operator ... multiple times" -- a hard crash the
    op_registry hygiene below cannot reach (that registry is vLLM's, not
    torch's dispatcher). We wrap the function so a re-registration of an op
    already present in the dispatcher is a no-op: the first (real)
    registration still installs the implementation, so the graded kernel is
    unaffected; only the redundant duplicate define/impl is skipped.

    Patches only this subprocess's import surface (same philosophy as
    ``install_vllm_c_stub`` / the triton_utils inject shim); the subprocess is
    per-task, so the wrapper dies with it and never leaks across tasks. Marked
    so it installs exactly once. Returns ``True`` when installed/already
    installed, ``False`` if vLLM is absent.
    """
    try:
        import torch
        from vllm.utils import torch_utils as _tu
    except Exception:
        return False

    current = getattr(_tu, "direct_register_custom_op", None)
    if current is None:
        return False
    if getattr(current, "__tab_idempotent_direct_register__", False):
        return True

    def _idempotent_direct_register(
        op_name, op_func, mutates_args=None, fake_impl=None,
        target_lib=None, dispatch_key=None, tags=(),
    ):
        my_lib = target_lib if target_lib is not None else getattr(
            _tu, "vllm_lib", None
        )
        ns = getattr(my_lib, "ns", "vllm")
        try:
            already = torch._C._dispatch_has_kernel(f"{ns}::{op_name}")
        except Exception:
            already = False
        if already:
            # Identical op already defined in this process -- skip the
            # duplicate define/impl/fake so the re-exec does not crash. The
            # first registration's implementation stays installed.
            return None
        return current(
            op_name, op_func, mutates_args, fake_impl,
            target_lib, dispatch_key, tags,
        )

    _idempotent_direct_register.__tab_idempotent_direct_register__ = True
    _idempotent_direct_register.__wrapped__ = current
    _tu.direct_register_custom_op = _idempotent_direct_register
    return True


def clear_conflicting_vllm_registrations(kernel_src: str) -> list[str]:
    """Drop pre-existing vLLM op registrations that this kernel redefines."""
    # Whole-module vendored kernels that import FlashInfer at MODULE TOP LEVEL
    # (e.g. the L1_04 file vllm/v1/attention/backends/flashinfer.py) crash at
    # import when FlashInfer is absent, even though the graded @triton.jit kernel
    # never touches it. Install the inert FlashInfer import surface BEFORE
    # exec_module (this helper runs just before it). Gated on a COLUMN-0 (import-
    # time) flashinfer import -- a lazy import nested in a function body
    # (L1_32/L1_33/L2_18/L2_38) imports fine without the stub, so it must not
    # trigger -- and on the real package being absent, so a daemon with
    # FlashInfer installed is untouched.
    if re.search(r"^(?:from|import)\s+flashinfer\b", kernel_src, re.M):
        install_flashinfer_stub()
    # Whole-file vendored kernels that mirror a vLLM model/attention module
    # (e.g. L2_18 Qwen3-Next) transitively reach ``vllm.vllm_flash_attn`` at
    # module load through a dead attention-backend import even though the graded
    # @triton.jit kernel never touches flash-attn. On a source checkout without
    # the compiled ``_vllm_fa2_C`` / ``_vllm_fa3_C`` extensions that package's
    # ``__init__`` raises ImportError, blocking the whole candidate at load.
    # Install an inert fail-closed surface BEFORE exec_module (raise-on-CALL, so
    # a real flash-attn dependency still fails loudly). The import is TRANSITIVE,
    # not a column-0 line in kernel.py, so it cannot be gated on a kernel_src
    # regex like flashinfer; the install autodiscovers the source-checkout vLLM
    # on sys.path and skips when a real flash-attn (compiled extension) is
    # present, so a daemon whose vLLM was built with flash-attn is untouched.
    # Gated on the kernel importing vllm at all (a non-vllm kernel never reaches
    # the package). This is the single shared loader hook, so it also closes the
    # GPU oracle-equivalence load path that bypasses the runners' setup blocks.
    if re.search(r"^(?:from|import)\s+vllm\b", kernel_src, re.M):
        install_vllm_flash_attn_stub()
    # Kernels that register a torch.library op at module scope crash on the
    # anchor's second per-shape load; install the idempotent wrapper BEFORE the
    # kernel executes ``from vllm.utils.torch_utils import
    # direct_register_custom_op`` (this helper runs just before exec_module).
    if _DIRECT_REGISTER_TOKEN in kernel_src:
        install_idempotent_direct_register()

    names = sorted(set(_REGISTER_RE.findall(kernel_src)))
    if not names:
        return []
    try:
        from vllm.model_executor.custom_op import op_registry, op_registry_oot
    except Exception:
        return []

    removed: list[str] = []
    for name in names:
        if name in op_registry:
            op_registry.pop(name, None)
            removed.append(name)
        op_registry_oot.pop(name, None)
    return removed
