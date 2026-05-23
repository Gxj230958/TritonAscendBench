"""Smoke tests for ``scripts/automation/`` package skeleton (M0 Unit 0.1).

Verifies:
  * ``automation`` is importable as a package (i.e. the package
    layout + ``__init__.py`` files exist).
  * ``automation.__doc__`` carries the ``INTERNAL`` marker so any
    downstream reviewer can spot the boundary at-a-glance.
  * ``automation.lib`` and ``automation.prompts`` are sub-packages.
  * ``automation`` extras are declared in ``pyproject.toml`` (so the
    CI install step ``pip install -e ".[dev,automation]"`` resolves).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"


@pytest.fixture(autouse=True)
def _ensure_scripts_on_path():
    """Insert ``<repo>/scripts`` on ``sys.path`` so ``import automation``
    resolves the package under ``scripts/automation/``. CLI entry-points
    achieve the same via the KTD-11 bootstrap header; tests do it once
    in the fixture."""
    inserted = False
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
        inserted = True
    # Drop any stale cached package so re-imports honour the path tweak.
    for mod_name in ("automation", "automation.lib", "automation.prompts"):
        sys.modules.pop(mod_name, None)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(str(_SCRIPTS))
            except ValueError:
                pass


def test_automation_package_importable():
    automation = importlib.import_module("automation")
    assert automation.__doc__ is not None, "automation package must have a docstring"
    assert "INTERNAL" in automation.__doc__, (
        "automation/__init__.py must mark itself INTERNAL so the release "
        "boundary is obvious to any reader"
    )


def test_automation_lib_is_subpackage():
    lib = importlib.import_module("automation.lib")
    assert lib.__name__ == "automation.lib"


def test_automation_prompts_is_subpackage():
    prompts = importlib.import_module("automation.prompts")
    assert prompts.__name__ == "automation.prompts"


def test_automation_directory_layout():
    auto = _SCRIPTS / "automation"
    assert (auto / "__init__.py").is_file()
    assert (auto / "lib" / "__init__.py").is_file()
    assert (auto / "prompts" / "__init__.py").is_file()
    assert (auto / "README.md").is_file()


def test_readme_does_not_name_external_projects():
    """The README must describe the directory generically and not point
    at any external project (KTD-10 release boundary -- bench repo /
    automation README do not name external projects)."""
    text = (_SCRIPTS / "automation" / "README.md").read_text(encoding="utf-8")
    lowered = text.lower()
    for forbidden in ("kernelpilot", "kp_"):
        assert forbidden not in lowered, (
            f"automation/README.md must not reference {forbidden!r} "
            "(KTD-10 release-boundary rule)"
        )


def test_pyproject_declares_automation_extras():
    pyproject = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # Pure string check -- avoids pulling in a TOML parser dependency
    # for a smoke test. The exact key/value here matches Unit 0.1.
    assert "automation = [" in pyproject, (
        "pyproject.toml must declare [project.optional-dependencies].automation"
    )
    assert "requests>=2.31" in pyproject, (
        "automation extras must pin requests>=2.31 (orchestrator runtime)"
    )
