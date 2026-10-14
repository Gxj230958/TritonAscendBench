"""U5: the R2 import-deny gate -- the runtime never imports vllm/sglang.

Three complementary checks (the bench has no general CI beyond
``.github/workflows/ci.yml``'s ``pytest tests/unit_tests/`` step, which installs
``.[dev,automation]`` -- a clean vllm/sglang-free venv -- so these run there):

1. Import-surface purity: in a subprocess that BLOCKS vllm/sglang at import,
   discover + import every runtime submodule (excluding the vendored
   ``frozen.deps`` tree and the guarded ``frozen.snapshots`` data) and assert
   none pulled upstream into ``sys.modules``.
2. ``tab run`` purity: run an actual ``tab run`` with vllm/sglang blocked and
   assert it never attempts an upstream import.
3. Static backstop: a module-top ``import vllm``/``sglang`` scan over the shipped
   importable tree + task runtime files (covers ``frozen.deps``, which is
   vendored and not all standalone-importable).
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Subprocess body: block vllm/sglang at import, walk + import the runtime
# surface (excluding vendored frozen.deps + guarded frozen.snapshots), assert
# none leaked. The meta_path block must be installed BEFORE ascendbench is
# imported, hence a fresh subprocess (the parent pytest may already hold real
# vllm from an external infer/ tree on a dev box).
_IMPORT_SURFACE_PROBE = textwrap.dedent(
    """
    import importlib, importlib.abc, pkgutil, sys

    FORBIDDEN = {"vllm", "sglang"}
    EXCLUDE = {"ascendbench.frozen.deps", "ascendbench.frozen.snapshots"}

    class _Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname.split(".")[0] in FORBIDDEN:
                raise ImportError("blocked upstream import: " + fullname)
            return None

    sys.meta_path.insert(0, _Block())

    def _onerror(name):
        # descent errors inside the vendored / guarded subtrees are expected
        # (frozen.deps has a vendored module with an upstream-style SyntaxError;
        # frozen.snapshots raises a deliberate guard). Re-raise anything else.
        if name.startswith("ascendbench.frozen.deps") or name.startswith(
            "ascendbench.frozen.snapshots"
        ):
            return
        raise

    import ascendbench

    failures = []
    for info in pkgutil.walk_packages(
        ascendbench.__path__, ascendbench.__name__ + ".", onerror=_onerror
    ):
        name = info.name
        if any(name == e or name.startswith(e + ".") for e in EXCLUDE):
            continue
        try:
            importlib.import_module(name)
        except ImportError as exc:
            msg = str(exc)
            if "provenance archives" in msg:
                continue  # snapshots guard -- expected, not a leak
            if "blocked upstream import" in msg:
                failures.append("LEAK: " + name + " -> " + msg)
            else:
                failures.append("IMPORT-FAIL: " + name + " -> " + msg)
        except Exception as exc:  # noqa: BLE001
            failures.append("ERROR: " + name + " -> " + type(exc).__name__ + ": " + str(exc))

    leaked = sorted(m for m in sys.modules if m.split(".")[0] in FORBIDDEN)
    if leaked:
        failures.append("sys.modules leaked: " + repr(leaked))
    if failures:
        print("\\n".join(failures))
        sys.exit(1)
    print("OK")
    """
)


def test_runtime_import_surface_has_no_upstream():
    """Import the full ascendbench runtime surface (minus vendored frozen.deps +
    guarded frozen.snapshots) with vllm/sglang blocked; nothing may pull them."""
    r = subprocess.run(
        [sys.executable, "-c", _IMPORT_SURFACE_PROBE],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, (
        "runtime import surface pulled upstream or failed to import:\n"
        f"{r.stdout}\n{r.stderr}"
    )


def test_tab_run_does_not_import_upstream(tmp_path):
    """Run an actual ``tab run`` (skeleton solution) with vllm/sglang blocked at
    import; it must never attempt an upstream import. This covers the cli + eval +
    static-check ENTRY path's import purity -- the skeleton bails at static check
    before grade execution (eval.py returns early); full candidate/oracle
    EXECUTION purity over the 101 tasks is covered by the U6 GPU-daemon
    regression."""
    # 1. create a skeleton solution (init-solution does not grade -> no blocker)
    sol_dir = tmp_path / "sol"
    init = subprocess.run(
        [sys.executable, "-m", "ascendbench.engine.cli",
         "init-solution", "L1_12_rms_norm", "--out", str(sol_dir)],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert init.returncode == 0, f"init-solution failed: {init.stderr}"

    out_path = tmp_path / "run.json"
    probe = textwrap.dedent(
        f"""
        import importlib.abc, sys
        FORBIDDEN = {{"vllm", "sglang"}}
        hits = []
        class _Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname.split(".")[0] in FORBIDDEN:
                    hits.append(fullname)
                    raise ImportError("blocked upstream import: " + fullname)
                return None
        sys.meta_path.insert(0, _Block())
        sys.argv = ["tab", "run", "L1_12_rms_norm",
                    "--solution", {str(sol_dir / "solution.py")!r},
                    "--backend", "gpu", "--out", {str(out_path)!r}]
        from ascendbench.engine.cli import main
        try:
            rc = main()
        except SystemExit as e:
            rc = e.code
        if hits:
            print("UPSTREAM IMPORT ATTEMPTED: " + repr(hits))
            sys.exit(1)
        print("RC", rc)
        """
    )
    r = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, (
        f"tab run attempted an upstream import or crashed:\n{r.stdout}\n{r.stderr}"
    )


def test_no_moduletop_upstream_import_in_shipped_tree():
    """U5.3 static backstop: no shipped importable module (ascendbench/ minus the
    guarded snapshots) and no task runtime file may have a module-top
    ``import vllm``/``sglang``. Covers vendored frozen/deps which can't all be
    dynamically imported. Uses ERE (a BRE ``^(from|import)`` would be vacuous)."""
    r = subprocess.run(
        ["git", "grep", "-nE", r"^(from|import) (vllm|sglang)\b", "--",
         "ascendbench", "tasks"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    # rc 0 = matches (the snapshots will match -- filter them out); rc 1 = no
    # matches. Any other rc (128 = git missing / not a repo) must fail loudly,
    # not pass vacuously on empty stdout.
    assert r.returncode in (0, 1), (
        f"git grep failed (rc={r.returncode}); cannot verify import purity:\n{r.stderr}"
    )
    # Anything left after dropping the guarded snapshots is a leak.
    hits = [ln for ln in r.stdout.splitlines() if "frozen/snapshots/" not in ln]
    assert not hits, "module-top upstream imports in shipped tree:\n" + "\n".join(hits)
