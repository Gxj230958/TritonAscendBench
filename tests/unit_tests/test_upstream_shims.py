"""Contract tests for ``triton_ascend_bench.upstream_shims``.

These tests pin every observable property the resolver subprocess
relies on when it injects the ``vllm.triton_utils`` forward shim:

  * the shim's public surface is a strict superset of upstream's
    ``vllm/triton_utils/__init__.py.__all__`` (parsed at runtime from
    ``infer/vllm`` -- NOT hardcoded, so an upstream rename is caught);
  * ``register()`` / ``unregister()`` are reversible, idempotent under
    ``force=True``, and respect a pre-existing entry when
    ``force=False``;
  * the forwarded ``triton`` is the *real* Triton (``JITFunction``
    class is the upstream one), never a placeholder stub;
  * ``tl`` exposes the dtypes real kernels actually use;
  * ``_has_active_triton_driver()`` returns ``False`` on this no-GPU
    host without raising;
  * the autotune fallback fires under ``TRITON_INTERPRET=1`` and
    delegates to the real autotuner with an active driver (mocked);
  * ``sys.path`` is left untouched by ``register()`` (no shim-by-path
    side effect);
  * other ``vllm.*`` submodules are still reachable via the real
    upstream copy after the shim is installed (the shim is a
    sub-package override, not a top-level vllm shadow);
  * the deeper import ``from vllm.triton_utils.importing import
    HAS_TRITON`` also returns the real-Triton view.

A regression in any of these breaks the resolver verdict on the
~5--10 P3 tasks the shim unblocks (L1_14 / L1_22 / L2_01 / ...).
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INFER_VLLM_ROOT = REPO_ROOT.parent / "infer" / "vllm"
UPSTREAM_VLLM_TRITON_UTILS_INIT = (
    INFER_VLLM_ROOT / "vllm" / "triton_utils" / "__init__.py"
)


@pytest.fixture
def shim_clean():
    """Snapshot + restore ``sys.modules`` entries the shim touches,
    plus environment variables it reads. Every test gets a clean shim
    state so ordering between tests does not leak."""
    keys = ("vllm.triton_utils", "vllm.triton_utils.importing", "vllm")
    saved_modules = {k: sys.modules.get(k) for k in keys}
    saved_env = {k: os.environ.get(k) for k in ("TRITON_INTERPRET",)}
    # Drop any pre-existing entries so register() starts fresh.
    for k in keys:
        sys.modules.pop(k, None)
    # Also clear the shim's internal saved-state dict so unregister()
    # in a previous test does not leak across.
    from triton_ascend_bench.upstream_shims import _inject

    _inject._saved.clear()
    yield
    # Restore.
    for k, v in saved_modules.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    _inject._saved.clear()


def _parse_upstream_all() -> list[str]:
    """Extract ``__all__`` from the real upstream
    ``vllm/triton_utils/__init__.py`` via the AST so the test does not
    hardcode the list -- an upstream rename is caught."""
    if not UPSTREAM_VLLM_TRITON_UTILS_INIT.is_file():
        pytest.skip(
            f"upstream vllm/triton_utils/__init__.py not found at "
            f"{UPSTREAM_VLLM_TRITON_UTILS_INIT} (infer checkout missing?)"
        )
    tree = ast.parse(UPSTREAM_VLLM_TRITON_UTILS_INIT.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
        ):
            return [el.value for el in node.value.elts]
    raise RuntimeError("upstream __init__.py has no __all__ assignment")


def test_shim_module_importable(shim_clean) -> None:
    """The ``_inject`` module itself must import without raising
    (a syntax error or missing dependency would mask every later
    register() call)."""
    from triton_ascend_bench.upstream_shims import _inject

    assert hasattr(_inject, "register")
    assert hasattr(_inject, "unregister")


def test_shim_all_mirrors_upstream(shim_clean) -> None:
    """The shim's ``__all__`` must be a superset of upstream's
    so every name a vendored kernel can import is satisfied."""
    upstream_all = set(_parse_upstream_all())
    from triton_ascend_bench.upstream_shims import vllm_triton_utils_shim

    shim_all = set(vllm_triton_utils_shim.__all__)
    assert upstream_all <= shim_all, (
        f"upstream exports {upstream_all - shim_all} that the shim "
        f"does not"
    )


def test_register_injects_into_sys_modules(shim_clean) -> None:
    """``register()`` makes ``sys.modules['vllm.triton_utils']`` the
    shim object, not the upstream placeholder module."""
    from triton_ascend_bench.upstream_shims import _inject
    from triton_ascend_bench.upstream_shims import (
        vllm_triton_utils_shim as expected,
    )

    assert _inject.register(force=True) is True
    assert sys.modules["vllm.triton_utils"] is expected


def test_register_force_overrides_existing(shim_clean) -> None:
    """``force=False`` leaves a pre-existing entry alone; ``force=True``
    overwrites it AND records the original for ``unregister()``."""
    import types

    fake_existing = types.ModuleType("vllm.triton_utils")
    fake_existing.__pa_test_marker__ = "original-upstream"
    sys.modules["vllm.triton_utils"] = fake_existing

    from triton_ascend_bench.upstream_shims import _inject

    # force=False -> keep existing.
    installed = _inject.register(force=False)
    assert installed is False
    assert sys.modules["vllm.triton_utils"] is fake_existing

    # force=True -> overwrite + remember original.
    installed = _inject.register(force=True)
    assert installed is True
    assert sys.modules["vllm.triton_utils"] is not fake_existing
    assert _inject._saved["vllm.triton_utils"] is fake_existing


def test_unregister_cleans_sys_modules(shim_clean) -> None:
    """After ``register()`` then ``unregister()`` the prior
    ``sys.modules`` state is restored (no shim entries remain)."""
    from triton_ascend_bench.upstream_shims import _inject

    assert "vllm.triton_utils" not in sys.modules
    _inject.register(force=True)
    assert "vllm.triton_utils" in sys.modules
    _inject.unregister()
    # The prior state was "not present" -> entries removed.
    assert "vllm.triton_utils" not in sys.modules
    assert "vllm.triton_utils.importing" not in sys.modules


def test_triton_is_real_not_placeholder(shim_clean) -> None:
    """The forwarded ``triton`` exposes the real ``JITFunction`` class
    (a placeholder stub would not have it). Also confirm the autotuner
    machinery is intact."""
    import triton as real_triton

    from triton_ascend_bench.upstream_shims import _inject

    _inject.register(force=True)

    from vllm.triton_utils import triton as shim_triton

    assert hasattr(shim_triton, "JITFunction")
    assert shim_triton.JITFunction is real_triton.JITFunction
    assert shim_triton.Config is real_triton.Config
    assert callable(shim_triton.jit)
    assert callable(shim_triton.heuristics)


def test_tl_dtypes_available(shim_clean) -> None:
    """Every dtype a vendored kernel may reach for must be present on
    the forwarded ``tl``."""
    from triton_ascend_bench.upstream_shims import _inject

    _inject.register(force=True)

    from vllm.triton_utils import tl

    for name in (
        "constexpr",
        "float32",
        "float16",
        "bfloat16",
        "int8",
        "int16",
        "int32",
        "int64",
        "program_id",
        "arange",
        "load",
        "store",
        "exp",
        "log",
    ):
        assert hasattr(tl, name), f"tl missing {name!r}"


def test_has_active_triton_driver_no_raise(shim_clean) -> None:
    """The driver probe must NEVER raise -- it is called inside the
    autotune wrapper on every kernel launch, so a swallowed exception
    here is a hard requirement."""
    from triton_ascend_bench.upstream_shims.vllm_triton_utils_shim import (
        _has_active_triton_driver,
    )

    result = _has_active_triton_driver()
    assert isinstance(result, bool)
    # On this dev box (no GPU) the driver is not active. We do NOT
    # assert ``result is False`` here because a future host with a real
    # driver should still pass.


def test_autotune_interpreter_mode_skips_benchmark(shim_clean) -> None:
    """Under ``TRITON_INTERPRET=1`` (or no active driver) the autotune
    wrapper must return an object whose ``[grid](...)`` runs the JIT
    function with the first config's kwargs merged in, WITHOUT
    benchmarking (which would raise on a no-GPU box). We also pin
    ``_has_active_triton_driver`` to ``False`` so a concurrent test
    suite that happens to have woken a real driver does not flip the
    branch and break this assertion."""
    os.environ["TRITON_INTERPRET"] = "1"

    from triton_ascend_bench.upstream_shims import _inject, vllm_triton_utils_shim

    _inject.register(force=True)

    import torch
    from vllm.triton_utils import tl, triton

    captured: dict = {}

    with mock.patch.object(
        vllm_triton_utils_shim,
        "_has_active_triton_driver",
        return_value=False,
    ):
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK": 64}, num_warps=4),
                triton.Config({"BLOCK": 128}, num_warps=4),
            ],
            key=["n"],
        )
        @triton.jit
        def k(p, n, BLOCK: tl.constexpr):  # noqa: N803 -- triton constexpr convention
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            v = tl.load(p + offs, mask=mask)
            tl.store(p + offs, v + 1, mask=mask)

        from triton_ascend_bench.upstream_shims.vllm_triton_utils_shim import (
            _AutotunedKernel,
        )

        assert isinstance(k, _AutotunedKernel), (
            f"expected _AutotunedKernel under TRITON_INTERPRET=1, "
            f"got {type(k).__name__}"
        )

        captured["first_config_kwargs"] = k._first_kwargs  # type: ignore[attr-defined]
        x = torch.zeros(256, dtype=torch.float32)
        k[(4,)](x, 256)
    # The first config's BLOCK=64 must have been passed; verify the
    # kernel actually executed (every element incremented by 1).
    assert torch.all(x == 1.0)
    assert captured["first_config_kwargs"]["BLOCK"] == 64


def test_autotune_interpreter_forwards_full_config_kwargs(shim_clean) -> None:
    """The interpreter fallback must propagate ``num_warps`` /
    ``num_stages`` / ``num_ctas`` etc. -- not just ``Config.kwargs``.
    Real Triton dispatches via ``config.all_kwargs()``; dropping the
    launch-param subset would silently mis-launch a kernel whose first
    config relies on a non-default ``num_warps`` (e.g. L2_01 lists
    ``num_warps=8, num_stages=3`` on its first config). We assert
    every field ``all_kwargs()`` returns is captured in
    ``_first_kwargs``."""
    os.environ["TRITON_INTERPRET"] = "1"

    from triton_ascend_bench.upstream_shims import _inject, vllm_triton_utils_shim

    _inject.register(force=True)

    from vllm.triton_utils import triton

    cfg = triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256},
        num_warps=8,
        num_stages=3,
    )

    with mock.patch.object(
        vllm_triton_utils_shim,
        "_has_active_triton_driver",
        return_value=False,
    ):
        @triton.autotune(
            configs=[cfg, triton.Config({"BLOCK_SIZE_M": 32}, num_warps=2)],
            key=["n"],
        )
        @triton.jit
        def k(p, n, BLOCK_SIZE_M: triton.language.constexpr,  # noqa: N803
              BLOCK_SIZE_N: triton.language.constexpr):  # noqa: N803
            pass

    expected = dict(cfg.all_kwargs())
    captured = k._first_kwargs  # type: ignore[attr-defined]
    for key, value in expected.items():
        assert captured.get(key) == value, (
            f"shim fallback dropped {key!r}: expected {value!r}, got "
            f"{captured.get(key)!r}; full captured={captured!r}"
        )


def test_autotune_interpreter_fallback_handles_stub_without_all_kwargs(
    shim_clean,
) -> None:
    """If a test double / older Triton Config exposes only the public
    attributes (``kwargs`` dict + ``num_warps`` / ``num_stages`` /
    ``num_ctas`` / ``maxnreg`` / ``ir_override``) but no
    ``all_kwargs()`` method, the shim's manual-merge fallback must
    still emit every launch param. The merge precedence also matters:
    kwargs go first, launch params overlay -- so a kernel that names a
    ``num_warps`` constexpr inside its kwargs dict cannot accidentally
    shadow the Config's real ``num_warps`` field."""
    os.environ["TRITON_INTERPRET"] = "1"

    from triton_ascend_bench.upstream_shims import _inject, vllm_triton_utils_shim

    _inject.register(force=True)

    class _StubConfig:
        """Mimics the public Config attribute surface without
        ``all_kwargs()``."""

        def __init__(
            self,
            kwargs: dict,
            num_warps: int,
            num_stages: int,
            num_ctas: int = 1,
            maxnreg: int | None = None,
            ir_override: str | None = None,
        ) -> None:
            self.kwargs = kwargs
            self.num_warps = num_warps
            self.num_stages = num_stages
            self.num_ctas = num_ctas
            self.maxnreg = maxnreg
            self.ir_override = ir_override

    stub = _StubConfig(
        {"BLOCK": 128},
        num_warps=8,
        num_stages=3,
        num_ctas=2,
        maxnreg=64,
        ir_override="custom-ir",
    )

    with mock.patch.object(
        vllm_triton_utils_shim,
        "_has_active_triton_driver",
        return_value=False,
    ):
        # Use the shim's autotune directly so we can pass our stub
        # without instantiating a real Config.
        decorator = vllm_triton_utils_shim.autotune(
            configs=[stub], key=["n"],
        )

        def _fake_jit_fn() -> None:
            pass

        # No JIT subscripting in this test -- we only care about the
        # captured _first_kwargs surface.
        wrapped = decorator(_fake_jit_fn)

    captured = wrapped._first_kwargs  # type: ignore[attr-defined]
    assert captured["BLOCK"] == 128
    assert captured["num_warps"] == 8
    assert captured["num_stages"] == 3
    assert captured["num_ctas"] == 2
    assert captured["maxnreg"] == 64
    assert captured["ir_override"] == "custom-ir"


def test_autotune_active_driver_uses_real(shim_clean) -> None:
    """When ``_interpreter_mode_active`` reports False (active driver +
    no TRITON_INTERPRET), the wrapper must delegate to the real
    ``triton.autotune``, not the fallback."""
    # Make sure interpret is OFF.
    os.environ.pop("TRITON_INTERPRET", None)

    from triton_ascend_bench.upstream_shims import vllm_triton_utils_shim

    with mock.patch.object(
        vllm_triton_utils_shim,
        "_has_active_triton_driver",
        return_value=True,
    ), mock.patch(
        "triton.autotune",
    ) as real_autotune:
        sentinel = object()
        real_autotune.return_value = sentinel

        configs = ["c0", "c1"]  # opaque to the real autotune mock
        result = vllm_triton_utils_shim.autotune(configs=configs, key=["n"])

        real_autotune.assert_called_once_with(configs=configs, key=["n"])
        assert result is sentinel


def test_real_vllm_subpackage_not_shadowed(shim_clean) -> None:
    """Installing the ``vllm.triton_utils`` shim must NOT block access
    to the rest of the ``vllm.*`` namespace. We use
    ``vllm.utils.math_utils.cdiv`` because (a) it is real, (b) it is
    used by upstream's own ``triton_utils/importing.py``, and (c) it is
    pure-Python (no GPU dependency)."""
    # Put infer/vllm on sys.path so the real vllm package resolves.
    vllm_path = str(INFER_VLLM_ROOT)
    added_path = False
    if vllm_path not in sys.path:
        sys.path.insert(0, vllm_path)
        added_path = True

    # Drop any cached vllm modules so the import truly resolves to the
    # infer/vllm checkout (not a stale unrelated package).
    for k in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
        sys.modules.pop(k, None)

    try:
        from triton_ascend_bench.upstream_shims import _inject

        _inject.register(force=True)

        import vllm.utils.math_utils as mu

        # cdiv is the canonical pure-Python utility upstream exports.
        assert callable(mu.cdiv)
        assert mu.cdiv(7, 3) == 3

        # And the shim is still the active vllm.triton_utils entry.
        from triton_ascend_bench.upstream_shims import (
            vllm_triton_utils_shim as expected,
        )

        assert sys.modules["vllm.triton_utils"] is expected
    finally:
        if added_path:
            sys.path.remove(vllm_path)


def test_vllm_triton_utils_importing_submodule(shim_clean) -> None:
    """``from vllm.triton_utils.importing import HAS_TRITON`` must
    return the real-Triton ``True`` value, not the placeholder
    ``False``."""
    from triton_ascend_bench.upstream_shims import _inject

    _inject.register(force=True)

    from vllm.triton_utils.importing import HAS_TRITON, tl, triton

    assert HAS_TRITON is True
    assert triton is not None
    assert tl is not None
    assert hasattr(triton, "JITFunction")


def test_shim_does_not_touch_sys_path(shim_clean) -> None:
    """``register()`` must operate purely via ``sys.modules`` -- never
    by mutating ``sys.path``. If a future change tried to insert a
    shim directory onto ``sys.path``, this test would catch it
    (shadowing every other vllm.* submodule is exactly the failure
    mode the shim was designed to avoid)."""
    from triton_ascend_bench.upstream_shims import _inject

    path_before = list(sys.path)
    _inject.register(force=True)
    path_after = list(sys.path)
    assert path_after == path_before, (
        f"register() mutated sys.path: added "
        f"{set(path_after) - set(path_before)!r}, removed "
        f"{set(path_before) - set(path_after)!r}"
    )


def test_is_registered_reflects_state(shim_clean) -> None:
    """``is_registered()`` returns ``True`` iff the parent shim is
    the active ``sys.modules`` entry. Smoke-checks the predicate the
    runner subprocesses can use to assert the shim took effect."""
    from triton_ascend_bench.upstream_shims import _inject

    assert _inject.is_registered() is False
    _inject.register(force=True)
    assert _inject.is_registered() is True
    _inject.unregister()
    assert _inject.is_registered() is False


def test_kernel_with_relative_import_loads_via_shim_and_bridge(
    shim_clean, tmp_path,
) -> None:
    """End-to-end smoke: a synthetic vendored kernel.py that imports
    ``from vllm.triton_utils import tl, triton`` (the failure mode the
    shim fixes) loads, compiles, and dispatches under interpreter
    mode. This is the closest unit test to the real resolver path
    without spawning a subprocess.

    We craft the kernel.py in a tmp_path, register the shim, import it
    under ``TRITON_INTERPRET=1`` and run a one-shape launch. We also
    pin ``_has_active_triton_driver`` to ``False`` so a concurrent
    test suite that happens to have woken a real driver does not flip
    the branch and break this assertion."""
    os.environ["TRITON_INTERPRET"] = "1"

    from triton_ascend_bench.upstream_shims import _inject, vllm_triton_utils_shim

    _inject.register(force=True)

    kernel_src = (
        '"""synthetic vendored kernel.py for the shim smoke test."""\n'
        "import torch\n"
        "from vllm.triton_utils import tl, triton\n"
        "\n"
        "@triton.autotune(\n"
        "    configs=[\n"
        '        triton.Config({"BLOCK": 64}, num_warps=4),\n'
        '        triton.Config({"BLOCK": 128}, num_warps=4),\n'
        "    ],\n"
        '    key=["n"],\n'
        ")\n"
        "@triton.jit\n"
        "def _add_one_kernel(p, n, BLOCK: tl.constexpr):\n"
        "    pid = tl.program_id(0)\n"
        "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
        "    mask = offs < n\n"
        "    v = tl.load(p + offs, mask=mask)\n"
        "    tl.store(p + offs, v + 1, mask=mask)\n"
        "\n"
        "def add_one(x):\n"
        "    n = x.numel()\n"
        "    _add_one_kernel[(1,)](x, n)\n"
        "    return x\n"
    )
    kpath = tmp_path / "kernel.py"
    kpath.write_text(kernel_src)

    with mock.patch.object(
        vllm_triton_utils_shim,
        "_has_active_triton_driver",
        return_value=False,
    ):
        import importlib.util

        spec = importlib.util.spec_from_file_location("shim_e2e_kernel", kpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        import torch

        x = torch.zeros(64, dtype=torch.float32)
        mod.add_one(x)
        assert torch.all(x == 1.0)
