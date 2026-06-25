from __future__ import annotations

import enum
import importlib.util
import sys
import types
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import torch

from ascendbench.input_spec.inputs import build_inputs
from ascendbench.upstream_shims.vllm_runtime_hygiene import (
    clear_conflicting_vllm_registrations,
    install_vllm_c_stub,
    install_vllm_flash_attn_stub,
    install_vllm_source_version_shim,
)

REPO = Path(__file__).resolve().parents[2]
INFER_VLLM_ROOT = REPO.parent / "infer" / "vllm"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _knob_corpus import knob_doc  # noqa: E402


def _load_script(rel_path: str, module_name: str):
    path = REPO / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_vllm_source_version_shim_falls_back_to_source_tree(tmp_path, monkeypatch):
    vroot = tmp_path / "vllm_src"
    pkg = vroot / "vllm"
    pkg.mkdir(parents=True)
    (pkg / "version.py").write_text('__version__ = "0.9.9-src"\n')

    from importlib import metadata as importlib_metadata

    orig_version = importlib_metadata.version

    def fake_version(name: str) -> str:
        if name == "vllm":
            raise importlib_metadata.PackageNotFoundError
        return f"ok:{name}"

    monkeypatch.setattr(importlib_metadata, "version", fake_version)
    monkeypatch.syspath_prepend(str(vroot))

    assert install_vllm_source_version_shim(vroot) is True
    assert importlib_metadata.version("vllm") == "0.9.9-src"
    assert importlib_metadata.version("other") == "ok:other"

    monkeypatch.setattr(importlib_metadata, "version", orig_version)


def test_install_vllm_c_stub_injects_safe_placeholder_modules(
    tmp_path, monkeypatch,
):
    vroot = tmp_path / "vllm_src"
    pkg = vroot / "vllm"
    (pkg / "platforms").mkdir(parents=True)
    (pkg / "platforms" / "cuda.py").write_text("import vllm._C\n")
    monkeypatch.syspath_prepend(str(vroot))

    saved = {
        name: sys.modules.get(name)
        for name in ("vllm._C", "vllm._C_stable_libtorch")
    }
    for name in saved:
        sys.modules.pop(name, None)
    try:
        assert install_vllm_c_stub(vroot) is True
        for name in ("vllm._C", "vllm._C_stable_libtorch"):
            mod = sys.modules[name]
            assert getattr(mod, "__tab_stub__", False) is True
            assert getattr(mod, "__tab_missing_extension__", None) == name
            assert hasattr(mod, "missing_symbol") is False
            try:
                mod.missing_symbol
            except AttributeError as exc:
                assert name in str(exc)
            else:
                raise AssertionError("stub attribute access must fail loudly")
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def test_install_vllm_c_stub_skips_when_compiled_extension_exists(
    tmp_path, monkeypatch,
):
    vroot = tmp_path / "vllm_src"
    pkg = vroot / "vllm"
    (pkg / "platforms").mkdir(parents=True)
    (pkg / "platforms" / "cuda.py").write_text("import vllm._C\n")
    (pkg / f"_C{EXTENSION_SUFFIXES[0]}").write_text("")
    (pkg / f"_C_stable_libtorch{EXTENSION_SUFFIXES[0]}").write_text("")
    monkeypatch.syspath_prepend(str(vroot))

    saved = {
        name: sys.modules.get(name)
        for name in ("vllm._C", "vllm._C_stable_libtorch")
    }
    for name in saved:
        sys.modules.pop(name, None)
    try:
        assert install_vllm_c_stub(vroot) is False
        assert "vllm._C" not in sys.modules
        assert "vllm._C_stable_libtorch" not in sys.modules
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


_FA_STUB_MODS = (
    "vllm.vllm_flash_attn",
    "vllm.vllm_flash_attn.flash_attn_interface",
)


def _save_fa_mods():
    return {n: sys.modules.get(n) for n in _FA_STUB_MODS}


def _restore_fa_mods(saved):
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def test_install_vllm_flash_attn_stub_injects_inert_surface(
    tmp_path, monkeypatch,
):
    # Source checkout with the package source but NO compiled extension.
    vroot = tmp_path / "vllm_src"
    fa = vroot / "vllm" / "vllm_flash_attn"
    fa.mkdir(parents=True)
    (fa / "__init__.py").write_text("raise ImportError('needs _vllm_fa2_C')\n")
    monkeypatch.syspath_prepend(str(vroot))

    saved = _save_fa_mods()
    _restore_fa_mods(dict.fromkeys(_FA_STUB_MODS))
    try:
        # Autodiscovery from sys.path (the form clear_conflicting uses).
        assert install_vllm_flash_attn_stub() is True
        pkg = sys.modules["vllm.vllm_flash_attn"]
        assert getattr(pkg, "__tab_vllm_flash_attn_stub__", False) is True
        # FA flags honest: no compiled extension -> both False.
        assert pkg.FA2_AVAILABLE is False
        assert pkg.FA3_AVAILABLE is False
        # Public names resolve but RAISE on call (never silently faked).
        for n in (
            "flash_attn_varlen_func",
            "get_scheduler_metadata",
            "is_fa_version_supported",
            "fa_version_unsupported_reason",
        ):
            fn = getattr(pkg, n)
            try:
                fn()
            except RuntimeError as exc:
                assert "inert" in str(exc)
            else:
                raise AssertionError(f"{n} must fail loudly on call")
        # Submodule the real __init__ re-exports from is stubbed too.
        iface = sys.modules["vllm.vllm_flash_attn.flash_attn_interface"]
        assert getattr(iface, "__tab_vllm_flash_attn_stub__", False) is True
        # Idempotent.
        assert install_vllm_flash_attn_stub() is False
    finally:
        _restore_fa_mods(saved)


def test_install_vllm_flash_attn_stub_skips_when_compiled_extension_exists(
    tmp_path, monkeypatch,
):
    vroot = tmp_path / "vllm_src"
    fa = vroot / "vllm" / "vllm_flash_attn"
    fa.mkdir(parents=True)
    (fa / "__init__.py").write_text("# real package\n")
    (fa / f"_vllm_fa2_C{EXTENSION_SUFFIXES[0]}").write_text("")
    monkeypatch.syspath_prepend(str(vroot))

    saved = _save_fa_mods()
    _restore_fa_mods(dict.fromkeys(_FA_STUB_MODS))
    try:
        # Compiled extension present -> real flash-attn, leave it untouched.
        assert install_vllm_flash_attn_stub(vroot) is False
        assert "vllm.vllm_flash_attn" not in sys.modules
    finally:
        _restore_fa_mods(saved)


def test_install_vllm_flash_attn_stub_noop_without_source_checkout(monkeypatch):
    # No source-checkout vLLM on sys.path -> nothing to close, no-op.
    monkeypatch.setattr(sys, "path", [p for p in sys.path if "vllm" not in p])
    saved = _save_fa_mods()
    _restore_fa_mods(dict.fromkeys(_FA_STUB_MODS))
    try:
        assert install_vllm_flash_attn_stub() is False
        assert "vllm.vllm_flash_attn" not in sys.modules
    finally:
        _restore_fa_mods(saved)


def test_install_vllm_c_stub_imports_real_upstream_cuda_without_extensions(
    monkeypatch,
):
    cuda_py = INFER_VLLM_ROOT / "vllm" / "platforms" / "cuda.py"
    if not cuda_py.is_file():
        return

    monkeypatch.syspath_prepend(str(INFER_VLLM_ROOT))

    saved = {
        name: sys.modules.get(name)
        for name in (
            "vllm",
            "vllm._C",
            "vllm._C_stable_libtorch",
            "vllm.envs",
            "vllm.logger",
            "vllm.utils",
            "vllm.utils.import_utils",
            "vllm.utils.torch_utils",
            "vllm.v1",
            "vllm.v1.attention",
            "vllm.v1.attention.backends",
            "vllm.v1.attention.backends.registry",
            "vllm.platforms",
            "vllm.platforms.interface",
            "vllm.platforms.cuda",
        )
    }
    for name in saved:
        sys.modules.pop(name, None)

    vllm_pkg = types.ModuleType("vllm")
    vllm_pkg.__path__ = [str(INFER_VLLM_ROOT / "vllm")]
    sys.modules["vllm"] = vllm_pkg

    logger_mod = types.ModuleType("vllm.logger")

    class _Logger:
        def warning(self, *args, **kwargs):
            return None

    logger_mod.init_logger = lambda _name: _Logger()
    sys.modules["vllm.logger"] = logger_mod
    sys.modules["vllm.envs"] = types.ModuleType("vllm.envs")
    sys.modules["vllm.utils"] = types.ModuleType("vllm.utils")

    import_utils_mod = types.ModuleType("vllm.utils.import_utils")

    class _Nvml:
        def nvmlInit(self):
            return None

        def nvmlShutdown(self):
            return None

        def nvmlDeviceGetCount(self):
            return 0

    import_utils_mod.import_pynvml = lambda: _Nvml()
    sys.modules["vllm.utils.import_utils"] = import_utils_mod

    torch_utils_mod = types.ModuleType("vllm.utils.torch_utils")
    torch_utils_mod.is_quantized_kv_cache = lambda _dtype: False
    sys.modules["vllm.utils.torch_utils"] = torch_utils_mod

    sys.modules["vllm.v1"] = types.ModuleType("vllm.v1")
    sys.modules["vllm.v1.attention"] = types.ModuleType("vllm.v1.attention")
    sys.modules["vllm.v1.attention.backends"] = types.ModuleType(
        "vllm.v1.attention.backends"
    )
    registry_mod = types.ModuleType("vllm.v1.attention.backends.registry")

    class _AttentionBackendEnum(enum.Enum):
        FLASHINFER_MLA_SPARSE = 1
        FLASHMLA_SPARSE = 2
        FLASHINFER_MLA = 3
        CUTLASS_MLA = 4
        FLASH_ATTN_MLA = 5
        FLASHMLA = 6
        TRITON_MLA = 7
        FLASH_ATTN = 8
        FLASHINFER = 9
        TRITON_ATTN = 10
        FLEX_ATTENTION = 11
        TURBOQUANT = 12

    registry_mod.AttentionBackendEnum = _AttentionBackendEnum
    sys.modules["vllm.v1.attention.backends.registry"] = registry_mod

    platforms_pkg = types.ModuleType("vllm.platforms")
    platforms_pkg.__path__ = [str(cuda_py.parent)]
    sys.modules["vllm.platforms"] = platforms_pkg

    interface_mod = types.ModuleType("vllm.platforms.interface")

    class DeviceCapability:
        major = 0

    class Platform:
        pass

    class PlatformEnum(enum.Enum):
        CUDA = "cuda"

    interface_mod.DeviceCapability = DeviceCapability
    interface_mod.Platform = Platform
    interface_mod.PlatformEnum = PlatformEnum
    sys.modules["vllm.platforms.interface"] = interface_mod

    try:
        assert install_vllm_c_stub(INFER_VLLM_ROOT) is True
        spec = importlib.util.spec_from_file_location(
            "vllm.platforms.cuda", cuda_py
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "vllm.platforms"
        sys.modules["vllm.platforms.cuda"] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        assert mod.__name__ == "vllm.platforms.cuda"
        assert getattr(sys.modules["vllm._C"], "__tab_stub__", False) is True
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def test_clear_conflicting_vllm_registrations_pops_duplicate_names(monkeypatch):
    fake_mod = types.ModuleType("vllm.model_executor.custom_op")
    fake_mod.op_registry = {"dup_op": object(), "keep": object()}
    fake_mod.op_registry_oot = {"dup_op": object(), "keep": object()}

    saved = {
        k: sys.modules.get(k)
        for k in (
            "vllm",
            "vllm.model_executor",
            "vllm.model_executor.custom_op",
        )
    }
    sys.modules["vllm"] = types.ModuleType("vllm")
    sys.modules["vllm.model_executor"] = types.ModuleType("vllm.model_executor")
    sys.modules["vllm.model_executor.custom_op"] = fake_mod
    try:
        src = '@CustomOp.register("dup_op")\nclass X: pass\n'
        removed = clear_conflicting_vllm_registrations(src)
        assert removed == ["dup_op"]
        assert "dup_op" not in fake_mod.op_registry
        assert "dup_op" not in fake_mod.op_registry_oot
        assert "keep" in fake_mod.op_registry
    finally:
        for key, val in saved.items():
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val


def test_l2_34_shapes_and_oracle_follow_wrapper_contract():
    td = (
        REPO / "tasks" / "level2"
        / "L2_29_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe"
    )
    shapes = knob_doc(str(td))   # corpus is v2; build the inputs from the KNOB form (reconstructed from the committed v2)
    inputs = build_inputs(shapes, "P1", seed=0)

    assert inputs["input"].ndim == 3
    assert inputs["output"].dtype == torch.float8_e4m3fn
    assert inputs["masked_m"].ndim == 1

    oracle = _load_script(
        str(td.relative_to(REPO) / "oracle_pytorch.py"),
        "l2_34_oracle_test",
    )
    out = oracle.run(inputs)["output"]
    assert out.shape == inputs["output"].shape
    assert out.dtype == inputs["output"].dtype
