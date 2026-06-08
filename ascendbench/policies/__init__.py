"""ascendbench.policies — bundled config data + a resolver.

Holds the tolerance / ceiling / sanity / floor / trust policy files. `policy_path`
resolves a bundled file in both an installed wheel (via importlib.resources) and a
repo-tree / editable checkout (via this directory), so loaders work either way.
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path


def policy_path(name: str) -> Path:
    """Filesystem path to a bundled policy file (e.g. ``tolerance_policy.yaml``)."""
    try:
        cand = Path(str(resources.files(__name__) / name))
        if cand.is_file():
            return cand
    except (ModuleNotFoundError, FileNotFoundError, TypeError, NotADirectoryError):
        pass
    return Path(__file__).resolve().parent / name
