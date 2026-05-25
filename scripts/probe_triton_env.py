"""One-shot diagnostic for the deploy host's Triton install.

Why this script exists: on the user's `/inspire/...` deploy host, the
resolver's oracle subprocess fails on the six sglang LoRA tasks
(L2_04 ~ L2_09) with::

    AssertionError at sglang/srt/utils/common.py:3816
    assert isinstance(fn, triton.runtime.jit.JITFunction)

vllm has also flagged "Triton is installed, but triton.backends could
not be imported. Disabling Triton" during the same setup step. Both
symptoms point at a partial / forked Triton install on that host, but
we cannot tell from the dev box exactly what's wrong without probing.

This script answers three questions in one run:

1. What does ``@triton.jit`` actually return on this host? (the class,
   its module path, its MRO, and whether sglang's isinstance check
   against ``triton.runtime.jit.JITFunction`` would pass).
2. What does the Python env think Triton is? (version, file path,
   what's in ``triton.backends``, which Triton-named PyPI packages
   are installed).
3. Whether the missing-leaves we already shimmed into ``[infer]`` are
   complete (re-check imports of cachetools / pybase64 / cbor2 / etc.).

Run it on the SAME python the bench resolver uses -- that's the
``python3`` ``tab_quickstart.sh:289`` invokes. If multiple pythons
exist on the host, find it with::

    which python3 && python3 -c 'import sys; print(sys.executable)'

Then::

    python3 /path/to/scripts/probe_triton_env.py

Paste the output back so we can decide whether the fix is
``pip install -U triton``, an env-side replacement of triton-ascend,
or a deeper plumbing change in the bench.
"""
from __future__ import annotations

import sys


def _hdr(title: str) -> None:
    print(f"\n=== {title} ===")


def probe_python() -> None:
    _hdr("python")
    print(f"executable: {sys.executable}")
    print(f"version:    {sys.version.split()[0]}")


def probe_triton() -> None:
    _hdr("triton")
    try:
        import triton
    except Exception as exc:
        print(f"import triton FAIL: {type(exc).__name__}: {exc}")
        return
    print(f"triton.__version__: {triton.__version__}")
    print(f"triton.__file__:    {triton.__file__}")
    # JITFunction class identity
    try:
        from triton.runtime.jit import JITFunction
        print(f"JITFunction class:  {JITFunction}")
        print(f"JITFunction module: {JITFunction.__module__}")
    except Exception as exc:
        print(
            f"import triton.runtime.jit.JITFunction FAIL: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    # backends — vllm checks this exact path
    try:
        import triton.backends as tb
        backends = getattr(tb, "backends", None)
        if backends is None:
            print("triton.backends imported but has no .backends attr")
        else:
            print(f"triton.backends.backends keys: {list(backends.keys())}")
            actives = []
            for name, b in backends.items():
                drv = getattr(b, "driver", None)
                is_active = bool(drv and drv.is_active())
                actives.append((name, is_active))
            print(f"backend is_active: {actives}")
    except Exception as exc:
        print(
            f"import triton.backends FAIL (this is the vllm 'Disabling Triton' trigger): "
            f"{type(exc).__name__}: {exc}"
        )


def _load_synthetic_kernel(src: str, mod_name: str):
    """Write ``src`` to a temp .py file and importlib-load it. Triton
    3.x requires ``@jit`` functions to live in a real source file
    (``inspect.getsource`` check inside JITFunction.__init__), so a
    bare ``exec()`` raises ``ValueError``.
    """
    import importlib.util
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".py", prefix=f"{mod_name}_")
    try:
        with open(fd, "w") as f:
            f.write(src)
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        import os
        try:
            os.unlink(path)
        except OSError:
            pass


def probe_jit_decoration() -> None:
    _hdr("@triton.jit decoration result")
    try:
        import triton  # noqa: F401  (verified at top, ensure same env)
        from triton.runtime.jit import JITFunction
    except Exception as exc:
        print(f"prerequisite import FAIL: {type(exc).__name__}: {exc}")
        return

    # Plain @triton.jit
    src_plain = (
        "import triton\n"
        "@triton.jit\n"
        "def f(x_ptr):\n"
        "    return\n"
    )
    try:
        mod = _load_synthetic_kernel(src_plain, "_probe_plain")
        f = mod.f
        print(f"@triton.jit:           type={type(f).__module__}.{type(f).__qualname__}")
        print(f"  isinstance JITFunction: {isinstance(f, JITFunction)}")
        print(
            "  mro: "
            + ", ".join(c.__module__ + "." + c.__qualname__ for c in type(f).__mro__)
        )
    except Exception as exc:
        print(f"@triton.jit FAIL: {type(exc).__name__}: {exc}")

    # @triton.jit(do_not_specialize=...) — this is what L2_04/L2_05 use
    src_kw = (
        "import triton\n"
        "@triton.jit(do_not_specialize=['num_segs'])\n"
        "def g(x_ptr, num_segs):\n"
        "    return\n"
    )
    try:
        mod = _load_synthetic_kernel(src_kw, "_probe_kw")
        g = mod.g
        is_jit = isinstance(g, JITFunction)
        print(
            f"@triton.jit(do_not_specialize=...): "
            f"type={type(g).__module__}.{type(g).__qualname__}"
        )
        print(f"  isinstance JITFunction: {is_jit}")
        if not is_jit:
            print(
                "  ^^ FALSE -- this is the exact failure sglang's "
                "CachedKernel.__init__ check sees on this host."
            )
    except Exception as exc:
        print(
            f"@triton.jit(do_not_specialize=...) FAIL: "
            f"{type(exc).__name__}: {exc}"
        )


def probe_pip_triton() -> None:
    _hdr("pip-installed triton packages")
    import subprocess
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "list", "--format=freeze"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        print(f"pip list FAIL: {type(exc).__name__}: {exc}")
        return
    hits = [
        line for line in out.splitlines()
        if line.lower().startswith(("triton", "pytorch-triton"))
    ]
    if not hits:
        print("(no triton-prefixed packages found)")
    for line in hits:
        print(f"  {line}")


def probe_leaf_deps() -> None:
    _hdr("vllm/sglang leaf deps (must all import OK)")
    for mod in (
        "cbor2", "pybase64", "zmq", "msgspec", "cachetools",
        "cpuinfo",
    ):
        try:
            __import__(mod)
            print(f"  {mod}: OK")
        except Exception as exc:
            print(f"  {mod}: FAIL -- {type(exc).__name__}: {exc}")


def probe_sglang_assertion() -> None:
    """Reproduce the exact sglang assertion path inline so we see the
    real failure mode (without needing to import all of sglang)."""
    _hdr("sglang CachedKernel assertion reproduction")
    try:
        import triton
        from triton.runtime.jit import JITFunction
    except Exception as exc:
        print(f"prerequisite FAIL: {type(exc).__name__}: {exc}")
        return
    src = (
        "import triton\n"
        "@triton.jit(do_not_specialize=['num_segs'])\n"
        "def k(x_ptr, num_segs):\n"
        "    return\n"
    )
    try:
        mod = _load_synthetic_kernel(src, "_probe_assert")
    except Exception as exc:
        print(f"jit decoration FAIL: {type(exc).__name__}: {exc}")
        return
    fn = mod.k
    try:
        assert isinstance(fn, JITFunction)
        print(
            "assert PASS -- sglang's check would succeed; the LoRA tasks "
            "should NOT be blocked by this assert. If they still are, the "
            "issue is downstream (multi-import / sys.path / class identity "
            "across two triton copies)."
        )
    except AssertionError:
        print(
            "assert FAIL -- this reproduces the /inspire/ failure. "
            f"Decorated fn is type {type(fn).__module__}.{type(fn).__qualname__} "
            f"but sglang checks against {JITFunction.__module__}.JITFunction. "
            "Fix the triton install on this host."
        )


def main() -> int:
    probe_python()
    probe_triton()
    probe_jit_decoration()
    probe_sglang_assertion()
    probe_pip_triton()
    probe_leaf_deps()
    print("\n=== done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
