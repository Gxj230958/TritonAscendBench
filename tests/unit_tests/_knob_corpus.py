"""Read a task's pre-flip KNOB shapes.yaml from git history.

The corpus was migrated to the compact v2 explicit format (U9). Tests that exercise the knob->v2 / knob->v1
machinery (round-trip, parity, the materializer) still need the KNOB source, which no longer lives in the
working tree. It is in git at the all-knob ref below, so these helpers read it from there -- keeping such tests
decoupled from the working-tree corpus format. Tests that instead assert "the CORPUS is X" should read the
working-tree shapes.yaml directly and assert the v2 reality, NOT use this module.
"""
from __future__ import annotations

import functools
import subprocess
from pathlib import Path

import yaml

#: the M2-writer commit -- the last state where all 101 tasks are still knob-format (before the flip).
_ALL_KNOB_REF = "e563ac2"
_REPO = Path(__file__).resolve().parents[2]


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


@functools.cache
def knob_text(task: str) -> str:
    """The knob-format shapes.yaml TEXT for a task, read from the all-knob git ref."""
    rel = _resolve_rel(task)
    out = subprocess.run(
        ["git", "-C", str(_REPO), "show", f"{_ALL_KNOB_REF}:{rel}/shapes.yaml"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise FileNotFoundError(f"no knob shapes.yaml for {rel} at {_ALL_KNOB_REF}: {out.stderr.strip()}")
    return out.stdout


def knob_doc(task: str) -> dict:
    """The parsed knob-format shapes.yaml for a task (from git history).

    Fail-closed: the all-knob ref must yield an actual knob doc. If it ever drifts to a non-knob (explicit) doc,
    raise here rather than let a ``doc.get("size_knobs")``-gated test loop silently skip every row (a vacuous
    pass). Every task at the all-knob ref carries size_knobs by construction.
    """
    doc = yaml.safe_load(knob_text(task))
    if not (isinstance(doc, dict) and doc.get("size_knobs")):
        raise AssertionError(f"{task}: doc at {_ALL_KNOB_REF} is not knob-format (no size_knobs) -- ref drift?")
    return doc


def all_knob_task_dirs() -> list[Path]:
    """The 101 real corpus task dirs (working-tree paths; their shapes.yaml are now v2, use knob_doc for the
    knob form). Excludes the v2-only tasks/_canary fixture, which has no knob baseline at the all-knob ref."""
    return sorted(p.parent for p in (_REPO / "tasks").glob("level*/*/shapes.yaml"))
