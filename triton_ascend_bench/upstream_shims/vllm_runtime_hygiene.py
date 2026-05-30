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
from importlib.machinery import EXTENSION_SUFFIXES
import re
import sys
import types
from importlib import metadata as importlib_metadata
from pathlib import Path

_REGISTER_RE = re.compile(
    r"@\s*(?:[\w.]+\.)?(?:CustomOp|PluggableLayer)\.register"
    r"\(\s*[\"']([^\"']+)[\"']\s*\)"
)
_VLLM_C_EXTENSIONS = ("vllm._C", "vllm._C_stable_libtorch")

# Boolean capability PROBES that vLLM registers on ``torch.ops._C`` via
# its compiled extension. On a version-skewed / source-checkout daemon the
# ``_C`` operator namespace EXISTS (some other ``.so`` populated it) but
# these specific ops are absent, so a transitive import that evaluates them
# (e.g. ``w8a8_utils.cutlass_fp8_supported()`` -> the module constants
# ``CUTLASS_FP8_SUPPORTED`` / ``CUTLASS_BLOCK_FP8_SUPPORTED``) raises
# ``AttributeError: '_OpNamespace' '_C' object has no attribute ...`` at
# kernel LOAD time. These are DISPATCH GATES (cutlass-vs-triton), never the
# benched kernel's own compute -- the bench resolves a triton entrypoint and
# calls it directly, bypassing this probe. Schemas copied verbatim from
# infer/vllm csrc/libtorch_stable/torch_bindings.cpp so a FRAGMENT define
# matches upstream exactly. Compute ops (cutlass_scaled_mm, cutlass_moe_mm,
# ...) are deliberately NOT here: a kernel that truly needs cutlass compute
# must still fail loudly, never silently stubbed.
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

# Module-global so the registration outlives the call: PyTorch unregisters a
# Library's ops via ``weakref.finalize`` once the last reference is dropped,
# so a function-local Library would silently vanish. One singleton + a
# per-process registered-name set make repeated calls idempotent (a second
# ``.define`` of the same op raises).
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


def clear_conflicting_vllm_registrations(kernel_src: str) -> list[str]:
    """Drop pre-existing vLLM op registrations that this kernel redefines."""
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
