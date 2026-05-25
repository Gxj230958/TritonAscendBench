"""Contract tests for the upstream-sibling loader bridge.

The vendored kernel.py files frequently use ``from .X import Y`` to
import a helper that lives next to the upstream source. Without a
parent package the bare loader raises ``ImportError: attempted
relative import with no known parent package`` -- the ~18-task P2
bucket of ``BLOCKED(upstream-not-invocable-here:
relative-import-packaging)``.

The fix: ``_pa_oracle_check_runner._load`` (and the mirror
``_pa_subproc_runner._load_kernel_with_optional_bridge``) parses the
``Source path:`` header the freeze tool stamps into every vendored
kernel.py, locates the upstream sibling directory under one of the
configured upstream roots (``infer/vllm`` / ``infer/sglang/python``)
and builds a per-task synthetic package chain so the kernel's
``from .X`` resolves into the upstream sibling.

These tests FAIL ON REVERT to either the single-dir synthetic package
(P2 bucket) or to a silent fallback that hides why the import broke.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPTS = REPO / "scripts"
_ORACLE_RUNNER = SCRIPTS / "_pa_oracle_check_runner.py"
_SUBPROC_RUNNER = SCRIPTS / "_pa_subproc_runner.py"

# Load both runner modules so the unit-test cases can poke their
# bridge primitives directly. The runners ship duplicated copies of the
# loader on purpose (the oracle anchor runs in a subprocess with infer/
# on path; the S2 runner runs ``--side original`` with infer/ on path
# AND ``--side severed`` with infer/ explicitly OFF -- a shared library
# import would couple the two surfaces). Both copies MUST be exercised
# so a regression in either stays caught.
_ocr_spec = importlib.util.spec_from_file_location(
    "_pa_oracle_check_runner_bridge", _ORACLE_RUNNER,
)
ocr = importlib.util.module_from_spec(_ocr_spec)
sys.modules["_pa_oracle_check_runner_bridge"] = ocr
_ocr_spec.loader.exec_module(ocr)

_sub_spec = importlib.util.spec_from_file_location(
    "_pa_subproc_runner_bridge", _SUBPROC_RUNNER,
)
sub = importlib.util.module_from_spec(_sub_spec)
sys.modules["_pa_subproc_runner_bridge"] = sub
_sub_spec.loader.exec_module(sub)


def _mk_upstream_root(tmp_path: Path) -> Path:
    """Build a tiny upstream tree::

        upstream_root/
          mypkg/sub/
            __init__.py
            target.py        (declares ``def helper(): return 7``)
            utils.py         (declares ``HELPER = 'sibling-from-upstream'``)
            parent_helper... lives one dir up so ``from ..X`` is testable
          mypkg/
            __init__.py
            parent_helper.py (declares ``PARENT = 'two-dots-up'``)

    Returns the upstream root (the first ``infer/vllm``-style dir)."""
    root = tmp_path / "upstream_root"
    sub = root / "mypkg" / "sub"
    sub.mkdir(parents=True)
    (root / "mypkg" / "__init__.py").write_text("")
    (sub / "__init__.py").write_text("")
    (sub / "target.py").write_text(
        "def helper():\n    return 7\n"
    )
    (sub / "utils.py").write_text(
        "HELPER = 'sibling-from-upstream'\n"
        "def input_guard(x):\n    return x\n"
    )
    (root / "mypkg" / "parent_helper.py").write_text(
        "PARENT = 'two-dots-up'\n"
    )
    return root


def _mk_vendored_kernel(
    task_dir: Path, source_relpath: str, kernel_body: str,
) -> Path:
    """Write a fake vendored kernel.py with the freeze-stamped header
    pointing at ``source_relpath`` and the provided body."""
    task_dir.mkdir(parents=True, exist_ok=True)
    kp = task_dir / "kernel.py"
    kp.write_text(
        '"""Vendored kernel.\n\n'
        f"Source path: `{source_relpath}`\n"
        "Upstream repo: test\n"
        '"""\n'
        + kernel_body
    )
    return kp


# ---------------------------------------------------------------------
# UNIT 1: legacy single-dir path is preserved for kernels without
# any relative import (back-compat for L1_12 / L1_16 / L1_26 PASSes).
# ---------------------------------------------------------------------
def test_task_without_relative_import_loads_unchanged(tmp_path):
    """A kernel with NO ``from .X`` import must take the legacy path:
    ``__path__`` is just the task dir, no upstream lookup attempted.
    Reverting the rel-depth==0 branch would route every task through
    the bridge and pointlessly require a Source-path header on every
    file -- breaking task-local kernels that have no upstream copy."""
    task_dir = tmp_path / "L_legacy"
    task_dir.mkdir()
    kp = task_dir / "kernel.py"
    kp.write_text(
        '"""No relative imports here."""\n'
        "def hello():\n    return 42\n"
    )
    # Pass upstream_roots=None on purpose -- this is the bare path the
    # old loader took and must keep working.
    mod = ocr._load(kp, "legacy_loadcheck_a", upstream_roots=None)
    assert mod.hello() == 42
    # Passing roots but with no relative import in the file: same
    # legacy branch (the rel-depth gate short-circuits).
    mod2 = ocr._load(
        kp, "legacy_loadcheck_b",
        upstream_roots=[tmp_path / "nonexistent"],
    )
    assert mod2.hello() == 42


# ---------------------------------------------------------------------
# UNIT 2: the canonical fix -- ``from .X`` resolves via the upstream
# sibling that the freeze-stamp header points at.
# ---------------------------------------------------------------------
def test_task_with_relative_import_via_sibling_bridge(tmp_path):
    """A vendored kernel with ``from .utils import HELPER`` must
    import successfully when the upstream sibling dir (containing
    ``utils.py``) is reachable under one of the configured upstream
    roots. FAILS ON REVERT to the old single-dir synthetic package
    (``ImportError: attempted relative import``)."""
    upstream_root = _mk_upstream_root(tmp_path)
    task_dir = tmp_path / "tasks" / "L_bridge"
    _mk_vendored_kernel(
        task_dir,
        source_relpath="mypkg/sub/target.py",
        kernel_body=(
            "from .utils import HELPER\n"
            "def fetched():\n    return HELPER\n"
        ),
    )
    mod = ocr._load(
        task_dir / "kernel.py", "bridge_one_dot",
        upstream_roots=[upstream_root],
    )
    assert mod.fetched() == "sibling-from-upstream"


# ---------------------------------------------------------------------
# UNIT 3: the Source-path header is the authoritative input to the
# bridge -- the loader MUST read it to find the sibling dir.
# ---------------------------------------------------------------------
def test_sibling_bridge_uses_upstream_source_path_from_kernel_header(
    tmp_path,
):
    """A kernel whose header says ``Source path:
    mypkg/sub/target.py`` must resolve ``from .utils`` to
    ``<upstream_root>/mypkg/sub/utils.py`` -- NOT to any other
    similarly-named utils.py elsewhere. The header IS the binding."""
    upstream_root = _mk_upstream_root(tmp_path)
    # Decoy ``utils.py`` at a wrong location: ``mypkg/utils.py`` with a
    # different value. If the loader resolves the wrong dir we will
    # import the decoy.
    (upstream_root / "mypkg" / "utils.py").write_text(
        "HELPER = 'WRONG-decoy-one-level-up'\n"
    )
    task_dir = tmp_path / "tasks" / "L_header"
    _mk_vendored_kernel(
        task_dir,
        source_relpath="mypkg/sub/target.py",
        kernel_body=(
            "from .utils import HELPER\n"
            "def fetched():\n    return HELPER\n"
        ),
    )
    mod = ocr._load(
        task_dir / "kernel.py", "bridge_header_path",
        upstream_roots=[upstream_root],
    )
    # The Source-path-derived sibling won (``mypkg/sub/utils.py``);
    # the wrong-level decoy did NOT win.
    assert mod.fetched() == "sibling-from-upstream", mod.fetched()


# ---------------------------------------------------------------------
# UNIT 4: when vllm misses, sglang is probed.
# ---------------------------------------------------------------------
def test_sibling_bridge_falls_back_to_sglang_if_vllm_not_found(tmp_path):
    """``upstream_roots`` is a list, probed in order. A kernel whose
    ``Source path`` does NOT exist under the first root must be
    located under the second (the sglang half of the live infer/
    checkout). FAILS ON REVERT to a single hard-coded root."""
    vllm_root = tmp_path / "vllm_root_empty"
    vllm_root.mkdir()  # exists but does not contain the source file
    sglang_root = _mk_upstream_root(tmp_path / "sglang_layout")
    task_dir = tmp_path / "tasks" / "L_fallback"
    _mk_vendored_kernel(
        task_dir,
        source_relpath="mypkg/sub/target.py",
        kernel_body=(
            "from .utils import HELPER\n"
            "def fetched():\n    return HELPER\n"
        ),
    )
    mod = ocr._load(
        task_dir / "kernel.py", "bridge_fallback",
        upstream_roots=[vllm_root, sglang_root],
    )
    assert mod.fetched() == "sibling-from-upstream"


# ---------------------------------------------------------------------
# UNIT 5: failure mode -- when the sibling cannot be located, the
# loader raises a CLEAR error rather than silently falling back to
# the legacy single-dir path (which would re-emit the misleading
# "attempted relative import with no known parent package").
# ---------------------------------------------------------------------
def test_sibling_bridge_missing_upstream_raises_clear_error(tmp_path):
    """A vendored kernel that DOES carry a Source-path header but the
    upstream file is absent under EVERY configured root must fail
    loud with a message naming the missing source_path and the probed
    roots. A silent fallback would hide why the import broke -- this
    test FAILS ON REVERT to a silent fallback."""
    fake_root = tmp_path / "empty_root"
    fake_root.mkdir()  # exists, but no mypkg/sub/target.py beneath it
    task_dir = tmp_path / "tasks" / "L_missing"
    _mk_vendored_kernel(
        task_dir,
        source_relpath="mypkg/sub/target.py",
        kernel_body=(
            "from .utils import HELPER\n"
            "def fetched():\n    return HELPER\n"
        ),
    )
    with pytest.raises(RuntimeError) as ei:
        ocr._load(
            task_dir / "kernel.py", "bridge_missing",
            upstream_roots=[fake_root],
        )
    msg = str(ei.value)
    assert "upstream sibling not found" in msg, msg
    assert "mypkg/sub/target.py" in msg, msg
    assert str(fake_root) in msg, msg


# ---------------------------------------------------------------------
# UNIT 6: the synthetic package chain is per-task -- loading two
# different tasks must not bleed their __path__ into each other.
# ---------------------------------------------------------------------
def test_synthetic_package_isolated_per_task(tmp_path):
    """The synthetic chain is keyed by the caller-supplied ``name``.
    Loading task A and task B back-to-back must produce two distinct
    keyed chains so a third reload of A cannot accidentally pick up
    B's ``__path__`` (or vice versa). FAILS ON REVERT to a single
    global synthetic package."""
    upstream_root_a = _mk_upstream_root(tmp_path / "root_a")
    upstream_root_b = _mk_upstream_root(tmp_path / "root_b")
    # Make root_b's sibling utils.py declare a different value so we
    # can tell which root won the import.
    (
        upstream_root_b / "mypkg" / "sub" / "utils.py"
    ).write_text("HELPER = 'root-B-value'\n")
    task_a = tmp_path / "tasks" / "L_iso_A"
    task_b = tmp_path / "tasks" / "L_iso_B"
    _mk_vendored_kernel(
        task_a, "mypkg/sub/target.py",
        kernel_body=(
            "from .utils import HELPER\n"
            "def v():\n    return HELPER\n"
        ),
    )
    _mk_vendored_kernel(
        task_b, "mypkg/sub/target.py",
        kernel_body=(
            "from .utils import HELPER\n"
            "def v():\n    return HELPER\n"
        ),
    )
    mod_a = ocr._load(
        task_a / "kernel.py", "iso_a",
        upstream_roots=[upstream_root_a],
    )
    mod_b = ocr._load(
        task_b / "kernel.py", "iso_b",
        upstream_roots=[upstream_root_b],
    )
    assert mod_a.v() == "sibling-from-upstream", mod_a.v()
    assert mod_b.v() == "root-B-value", mod_b.v()
    # Their synthetic packages live under DIFFERENT keys.
    pkgs = [
        k for k in sys.modules
        if k.startswith("_pa_pkg_iso_a") or k.startswith("_pa_pkg_iso_b")
    ]
    a_keys = [k for k in pkgs if "iso_a" in k]
    b_keys = [k for k in pkgs if "iso_b" in k]
    assert a_keys and b_keys, pkgs
    assert not (set(a_keys) & set(b_keys)), pkgs


# ---------------------------------------------------------------------
# UNIT 7: ``from ..X`` (two-dot relative) is supported.
# ---------------------------------------------------------------------
def test_kernel_with_dot_dot_import_handled(tmp_path):
    """A vendored kernel using ``from ..parent_helper import PARENT``
    (an upstream that lives one directory ABOVE the source file) must
    resolve correctly -- the synthetic chain walks 2 levels up under
    the upstream sibling root. FAILS ON REVERT to a one-level chain
    (which is what the original ``_load`` provided)."""
    upstream_root = _mk_upstream_root(tmp_path)
    task_dir = tmp_path / "tasks" / "L_two_dots"
    _mk_vendored_kernel(
        task_dir,
        source_relpath="mypkg/sub/target.py",
        kernel_body=(
            "from ..parent_helper import PARENT\n"
            "def fetched():\n    return PARENT\n"
        ),
    )
    mod = ocr._load(
        task_dir / "kernel.py", "bridge_two_dot",
        upstream_roots=[upstream_root],
    )
    assert mod.fetched() == "two-dots-up"


# ---------------------------------------------------------------------
# UNIT 8: the loader is read-only with respect to the task and its
# meta -- a loaded kernel must not have its source bytes modified.
# ---------------------------------------------------------------------
def test_loader_does_not_modify_kernel_file_or_meta(tmp_path):
    """Whatever the loader does in memory, the task directory's
    kernel.py bytes must be byte-exact unchanged after a load -- the
    bench's frozen-asset contract requires it. FAILS ON REVERT to a
    fix that rewrote kernel.py to inject an import statement."""
    upstream_root = _mk_upstream_root(tmp_path)
    task_dir = tmp_path / "tasks" / "L_readonly"
    kp = _mk_vendored_kernel(
        task_dir,
        source_relpath="mypkg/sub/target.py",
        kernel_body=(
            "from .utils import HELPER\n"
            "def fetched():\n    return HELPER\n"
        ),
    )
    bytes_before = kp.read_bytes()
    sibling_before = (
        upstream_root / "mypkg" / "sub" / "utils.py"
    ).read_bytes()
    _ = ocr._load(
        kp, "readonly_check", upstream_roots=[upstream_root],
    )
    assert kp.read_bytes() == bytes_before
    assert (
        upstream_root / "mypkg" / "sub" / "utils.py"
    ).read_bytes() == sibling_before


# ---------------------------------------------------------------------
# UNIT 9: end-to-end through the oracle runner subprocess -- the
# bridge wiring in main() actually plumbs upstream_roots into _load.
# This guards against a regression where _load is fixed but main()
# forgets to pass the roots (which would re-break every P2 task).
# ---------------------------------------------------------------------
def test_oracle_runner_subprocess_uses_bridge_end_to_end(tmp_path):
    """Drive the FULL ``_pa_oracle_check_runner`` subprocess on a
    minimal task whose kernel uses ``from .utils`` and whose oracle
    matches. The spec's ``vllm_root`` / ``sglang_root`` carry the
    test upstream; the runner must use them, build the bridge,
    import the kernel successfully, and report ``consistent: true``.
    FAILS ON REVERT if main() does not pass upstream_roots into
    _check_one_shape / _load."""
    upstream_root = _mk_upstream_root(tmp_path)
    task_dir = tmp_path / "tasks" / "L_e2e"
    task_dir.mkdir(parents=True)
    (task_dir / "kernel.py").write_text(
        '"""Vendored kernel for the bridge end-to-end test.\n\n'
        "Source path: `mypkg/sub/target.py`\n"
        "Upstream repo: test\n"
        '"""\n'
        "from .utils import HELPER\n"
        "def cand(x):\n"
        "    assert HELPER == 'sibling-from-upstream'\n"
        "    return {'y': x + 1.0}\n"
    )
    (task_dir / "oracle_pytorch.py").write_text(
        "import torch\n"
        "def run(inp):\n"
        "    return {'y': inp['x'] + 1.0}\n"
    )
    (task_dir / "meta.yaml").write_text(
        yaml.safe_dump(
            {
                "task_id": "E2E",
                "tolerance_class": "elementwise",
                "oracle": {"gpu_triton": "kernel.py::cand"},
            }
        )
    )
    (task_dir / "shapes.yaml").write_text(
        yaml.safe_dump(
            {
                "size_knobs": {"D": 4},
                "public_shapes": [
                    {"id": "S0", "knobs": {"D": 4}},
                    {"id": "S1", "knobs": {"D": 8}},
                ],
                "inputs": {
                    "x": {
                        "dims": ["D"], "dtype": "float32",
                        "dist": "randn",
                    }
                },
            }
        )
    )
    work = Path(tempfile.mkdtemp(prefix="pa_anchor_bridge_e2e_"))
    spec = {
        "task_dir": str(task_dir),
        "entrypoint": "cand",
        "out_json": str(work / "out.json"),
        "vllm_root": str(upstream_root),  # test upstream as the vllm root
        "sglang_root": str(tmp_path / "empty_sglang"),
        "repo_root": str(REPO),
    }
    (tmp_path / "empty_sglang").mkdir()
    (work / "spec.json").write_text(json.dumps(spec))
    proc = subprocess.run(
        [sys.executable, str(_ORACLE_RUNNER), "--spec",
         str(work / "spec.json")],
        capture_output=True, text=True,
    )
    out = json.loads((work / "out.json").read_text())
    assert out.get("consistent") is True, (
        f"end-to-end bridge failed: {out!r} (stderr={proc.stderr[-500:]})"
    )
    assert "anchored on 2/2 shape(s)" in out.get("detail", ""), out


# ---------------------------------------------------------------------
# UNIT 10: an honest header without a Source path that uses relative
# imports must produce a clear error (not a silent fallback).
# ---------------------------------------------------------------------
def test_relative_import_without_source_path_header_raises(tmp_path):
    """If a kernel uses ``from .X import Y`` but its docstring carries
    NO ``Source path:`` header, the loader cannot guess the upstream
    sibling. It must raise with a message naming both facts so the
    operator immediately knows where to look. FAILS ON REVERT to a
    silent fallback that would hide why the import broke."""
    task_dir = tmp_path / "tasks" / "L_no_header"
    task_dir.mkdir(parents=True)
    kp = task_dir / "kernel.py"
    # No Source-path header at all.
    kp.write_text(
        '"""A vendored kernel that forgot the freeze header."""\n'
        "from .utils import HELPER\n"
        "def fn():\n    return HELPER\n"
    )
    with pytest.raises(RuntimeError) as ei:
        ocr._load(
            kp, "no_header_check",
            upstream_roots=[tmp_path / "anything"],
        )
    assert "Source path" in str(ei.value), str(ei.value)


# ---------------------------------------------------------------------
# UNIT 11: the S2 subprocess runner ships a DUPLICATED bridge
# implementation (``_load_kernel_with_optional_bridge``). The two
# copies exist on purpose -- the oracle runner sets infer/ on sys.path
# unconditionally, while the S2 runner toggles it per side -- but
# they MUST stay behaviourally identical for the original-side
# vendored kernel.py because both load the SAME bytes. A regression in
# the S2 copy would silently break the S2-PASS pipeline for every P2
# task, so the next four tests exercise the S2 copy directly.
# ---------------------------------------------------------------------
def test_subproc_runner_bridge_resolves_relative_import(tmp_path):
    """Direct test of ``_pa_subproc_runner._load_kernel_with_optional_
    bridge`` -- the duplicated bridge in the S2 runner must also turn
    a ``from .utils`` into an upstream-sibling import. FAILS ON REVERT
    of the S2 copy to the bare ``spec_from_file_location`` form."""
    upstream_root = _mk_upstream_root(tmp_path)
    task_dir = tmp_path / "tasks" / "L_sub_one_dot"
    kp = _mk_vendored_kernel(
        task_dir,
        source_relpath="mypkg/sub/target.py",
        kernel_body=(
            "from .utils import HELPER\n"
            "def fetched():\n    return HELPER\n"
        ),
    )
    mod = sub._load_kernel_with_optional_bridge(
        kp, "sub_one_dot_check", upstream_roots=[upstream_root],
    )
    assert mod.fetched() == "sibling-from-upstream"


def test_subproc_runner_bridge_handles_two_dot_relative(tmp_path):
    """``from ..X`` works on the S2 side too -- chain depth must be
    walked correctly in both copies (a hand-copied loader can drift)."""
    upstream_root = _mk_upstream_root(tmp_path)
    task_dir = tmp_path / "tasks" / "L_sub_two_dots"
    kp = _mk_vendored_kernel(
        task_dir,
        source_relpath="mypkg/sub/target.py",
        kernel_body=(
            "from ..parent_helper import PARENT\n"
            "def fetched():\n    return PARENT\n"
        ),
    )
    mod = sub._load_kernel_with_optional_bridge(
        kp, "sub_two_dot_check", upstream_roots=[upstream_root],
    )
    assert mod.fetched() == "two-dots-up"


def test_subproc_runner_severed_side_takes_legacy_path(tmp_path):
    """Severed-side kernels are the STAGED standalone candidates --
    they have no relative imports and MUST stay on the severed surface
    (the bridge would resolve into infer/ and the leak guard would
    rightfully reject it). The S2 runner gates the bridge on
    ``side == 'original'`` and passes ``upstream_roots=None`` for
    severed; the loader's rel-depth==0 OR roots-is-None branch must
    therefore take the legacy single-dir path. FAILS ON REVERT to a
    fix that always engages the bridge."""
    task_dir = tmp_path / "tasks" / "L_severed"
    task_dir.mkdir(parents=True)
    kp = task_dir / "kernel.py"
    kp.write_text(
        '"""Staged candidate -- no relative imports allowed here."""\n'
        "def run(inputs):\n    return {'y': inputs['x']}\n"
    )
    mod = sub._load_kernel_with_optional_bridge(
        kp, "severed_legacy", upstream_roots=None,
    )
    out = mod.run({"x": "passthrough"})
    assert out == {"y": "passthrough"}, out


def test_subproc_runner_bridge_missing_upstream_raises_clear_error(
    tmp_path,
):
    """The S2 copy of the bridge must also fail loud when the upstream
    sibling cannot be located -- a silent fallback on the S2 side
    would re-emit the misleading
    ``attempted relative import with no known parent package`` and
    re-bucket the task into the P2 blocked tier even though the bridge
    succeeded on the oracle side. Both copies MUST be consistent."""
    fake_root = tmp_path / "empty_for_sub"
    fake_root.mkdir()
    task_dir = tmp_path / "tasks" / "L_sub_missing"
    kp = _mk_vendored_kernel(
        task_dir,
        source_relpath="mypkg/sub/target.py",
        kernel_body=(
            "from .utils import HELPER\n"
            "def fn():\n    return HELPER\n"
        ),
    )
    with pytest.raises(RuntimeError) as ei:
        sub._load_kernel_with_optional_bridge(
            kp, "sub_missing_check", upstream_roots=[fake_root],
        )
    msg = str(ei.value)
    assert "upstream sibling not found" in msg, msg
    assert "mypkg/sub/target.py" in msg, msg


# ---------------------------------------------------------------------
# UNIT 12: contract symmetry -- the two duplicated bridges must
# agree on the public surface. If one ships a new keyword arg the
# other does not, the S2 runner's call would TypeError. A direct
# signature comparison guards against the duplicates drifting apart.
# ---------------------------------------------------------------------
def test_both_runner_bridges_share_helper_contract():
    """The four bridge primitives (Source-path extraction, depth scan,
    sibling resolution, synthetic chain build) MUST exist on both
    runner modules with the same callable signature. A regression that
    renames or re-shapes one without the other would silently leave
    one runner on the broken path."""
    import inspect

    names = [
        "_extract_source_path",
        "_max_relative_depth",
        "_resolve_upstream_sibling",
        "_build_synthetic_pkg_chain",
    ]
    for n in names:
        assert hasattr(ocr, n), f"oracle runner lost {n}"
        assert hasattr(sub, n), f"subproc runner lost {n}"
        s_o = inspect.signature(getattr(ocr, n))
        s_s = inspect.signature(getattr(sub, n))
        assert str(s_o) == str(s_s), (
            f"signature drift on {n}: oracle={s_o!r} vs "
            f"subproc={s_s!r} -- the duplicated bridges must agree"
        )
