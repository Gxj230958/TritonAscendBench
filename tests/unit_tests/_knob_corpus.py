"""A knob-equivalent doc for a task, reconstructed from its CURRENT committed compact-v2 shapes.yaml.

The corpus was migrated to the compact v2 explicit format (U9). Tests that exercise the knob->v2 / knob->v1
machinery (round-trip, knob-vs-explicit parity, the materializer) still need a KNOB-shaped doc. ``v2_to_knob_spec``
reconstructs one LOSSLESSLY from the committed v2 (``generate_compact_shapes(v2_to_knob_spec(v2)) == v2``, proven
101/101), so the source is the working-tree shapes.yaml -- NO git-history / pinned-commit dependency (it survives
a squash-merge of the feature branch) and it reflects any authoring edit. The reconstructed doc is a NORMALISED
knob stand-in (row-0 env as ``size_knobs``, full per-row ``knobs``), not the original authored bytes; that is
faithful for these machinery tests, which compare independent generation paths on a valid knob.

Tests that instead assert "the CORPUS is X" should read the working-tree shapes.yaml directly and assert the v2
reality, NOT use this module.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.corpus.v2_authoring import load_v2, v2_to_knob_spec  # noqa: E402


def _resolve_rel(task: str) -> str:
    """Normalise a task reference to the repo-relative dir, e.g. 'tasks/level1/L1_01_bias'.

    Accepts 'L1_01_bias', 'level2/L2_01_...', 'tasks/level3/L3_..', or an absolute/relative dir path.
    """
    p = Path(task)
    if p.is_absolute() or (p.exists() and p.is_dir()):
        return str(Path(task).resolve().relative_to(_REPO))
    parts = Path(task).parts
    if parts and parts[0] == "tasks":
        return str(Path(*parts))
    if len(parts) >= 2 and parts[0] in ("level1", "level2", "level3"):
        return str(Path("tasks", *parts))
    # bare task id -> find it under tasks/
    matches = list((_REPO / "tasks").glob(f"*/{Path(task).name}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"cannot resolve task {task!r} to a unique tasks/ dir (got {matches})")
    return str(matches[0].relative_to(_REPO))


def knob_doc(task: str) -> dict:
    """A knob-equivalent doc reconstructed from the task's current committed compact-v2 shapes.yaml.

    Fail-closed via load_v2 (rejects a non-v2 / malformed file) and v2_to_knob_spec (rejects struct /
    per-row-varying inputs). The result always carries ``size_knobs`` -- the ``doc.get("size_knobs")``-gated
    test loops never silently skip every row.
    """
    rel = _resolve_rel(task)
    return v2_to_knob_spec(load_v2(_REPO / rel / "shapes.yaml"))


def all_knob_task_dirs() -> list[Path]:
    """The 101 real corpus task dirs (working-tree paths). Excludes the v2-only tasks/_canary fixture."""
    return sorted(p.parent for p in (_REPO / "tasks").glob("level*/*/shapes.yaml"))
