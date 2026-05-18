"""eligibility.py -- single source of truth for the score-eligible set.

Extracted from the inline logic previously in ``cli.py`` (~L597). A
task is *score-eligible* iff it has a real working oracle (not a
``raise NotImplementedError`` stub) and its ``shapes.yaml`` is not
``PHASE-C-MANUAL``. Counting stub/manual tasks as failures would
systematically depress valid submissions, so they are excluded from
the denominator (NOT counted as failures).

This module is **side-effect-light** by design (plan U7):

  * pure functions only;
  * file reads happen at **call time**, never at import time;
  * ZERO import-time I/O.

Both ``cli.py`` (analyze / release_blocked) and
``scripts/validate_tasks.py`` (release gate) import this module so the
score-eligible computation has a single source of truth and cannot
drift between the two call sites (plan: "API surface parity").
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


class _TaskLike(Protocol):
    task_id: str

    # `task_dir` is a pathlib.Path in practice; kept structural so this
    # module never imports dataset (avoids an import cycle / I/O).
    @property
    def task_dir(self): ...


def is_score_eligible(task: _TaskLike) -> bool:
    """True iff `task` counts toward the scored denominator.

    A task is NOT eligible when its oracle is a stub
    (``raise NotImplementedError``) or its ``shapes.yaml`` is
    ``PHASE-C-MANUAL``. Any OS error reading the task is treated as
    NOT eligible (the same conservative behaviour as the original
    inline ``cli.py`` code, which ``continue``-d on ``OSError``).
    """
    try:
        osrc = (task.task_dir / "oracle_pytorch.py").read_text()
        shp = (task.task_dir / "shapes.yaml").read_text()
    except OSError:
        return False
    if "raise NotImplementedError" in osrc or "PHASE-C-MANUAL" in shp:
        return False
    return True


def compute_score_eligible(tasks: Iterable[_TaskLike]) -> set[str]:
    """Return the set of score-eligible ``task_id``s.

    This is the single authority used by both ``cli.py analyze`` and
    ``scripts/validate_tasks.py --release-gate``. Behaviour is
    byte-identical to the previous inline ``cli.py`` loop.
    """
    return {t.task_id for t in tasks if is_score_eligible(t)}
