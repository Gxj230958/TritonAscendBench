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
import re
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

_REGISTER_RE = re.compile(
    r"@\s*(?:[\w.]+\.)?(?:CustomOp|PluggableLayer)\.register"
    r"\(\s*[\"']([^\"']+)[\"']\s*\)"
)


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
