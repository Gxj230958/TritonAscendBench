"""dataset.py — TaskDef loader.

Reads `tasks/level{1,2,3}/<task_id>/` per `specs/task_api_contract.md`:
- meta.yaml      (mandatory metadata)
- shapes.yaml    (public shapes)
- contract.md    (human-readable contract)
- kernel.py      (frozen GPU reference)
- oracle_pytorch.py
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ascendbench.corpus.provenance import (
    ProvenanceGoldenSpec,
    parse_provenance_golden,
)
from ascendbench.perf.cost_model import CostModel, CostModelError, parse_cost_model

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TASKS_ROOT = REPO_ROOT / "tasks"
# Canary trap tasks live here (W2.5 / DESIGN-V2 §6.8). `iter_all_tasks()`
# deliberately skips every `_`-prefixed directory, so canaries are NEVER
# score-eligible. They are discovered ONLY through `iter_canary_tasks()`.
# TAB_CANARY_ROOT lets the sealed (gitignored) maintainer canary copy live
# outside the repo so a submitter-visible public clone has no trap; defaults
# to tasks/_canary for a maintainer checkout.
CANARY_ROOT = Path(os.environ.get("TAB_CANARY_ROOT") or (TASKS_ROOT / "_canary"))


@dataclass
class ShapeSpec:
    id: str
    name: str
    params: dict[str, Any]


@dataclass
class TaskDef:
    task_id: str
    level: int
    task_dir: Path
    meta: dict[str, Any]
    public_shapes: list[ShapeSpec]
    contract_md: str
    kernel_py: str
    kernel_py_sha256: str
    oracle_module_name: str = "oracle_pytorch"
    # Normalised performance cost model (DESIGN-V2 §5.6). ``None`` when the
    # task's meta.yaml omits `cost_model` — the loader stays backward
    # compatible so pre-W3.1 tasks (most of the first 100, no cost_model yet)
    # still load; the red→green transition is carried by `validate_tasks`,
    # never by the loader.
    cost_model: CostModel | None = None
    # Operator input contract (WA.1, `specs/task_api_contract.md §5.2`).
    # Stored as the **raw dict** straight from `meta.yaml.param_constraints`
    # — the loader does NOT parse it into a structured object and does NOT
    # validate it. ``None`` when the block is absent. A malformed block is
    # kept verbatim (no raise) so it can reach the WA.4 `task_status=blocked`
    # gate instead of crashing `load_task`. The default keeps direct
    # ``TaskDef(...)`` construction sites (canary adapter, test fixtures)
    # backward compatible. `loader-lenient / validator-strict`: presence and
    # schema are enforced only by `scripts/validate_tasks.py`.
    param_constraints: dict[str, Any] | None = None
    # V9 plan M1.2: typed view of nested ``meta.provenance.golden:``
    # (the baseline trust chain). ``None`` when the nested block is
    # absent (legacy task; grace period until R10/R5 capture has run).
    # The full ``provenance:`` dict (including the legacy top-level
    # keys ``added_in`` / ``authors`` / ``last_modified``) remains
    # reachable via ``self.meta['provenance']``; this field is the
    # typed view of only the new nested ``golden:`` sub-block. Same
    # loader-lenient / validator-strict split as ``cost_model``:
    # presence and closed-enum shape are enforced by
    # ``scripts.validate_tasks``, never by the loader.
    provenance_golden: ProvenanceGoldenSpec | None = None

    @property
    def tolerance_class(self) -> str:
        # spec schema: top-level `tolerance_class`
        return self.meta.get("tolerance_class") or self.meta["tolerance"]["class"]

    @property
    def oracle_tier(self) -> str:
        # spec schema: top-level `oracle_simplified` (string, not nested .tier)
        v = self.meta.get("oracle_simplified")
        if isinstance(v, str):
            return v
        return v["tier"]                                  # back-compat fallback

    @property
    def upstream_sha(self) -> str:
        return self.meta["upstream"]["commit_sha"]


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_task(task_id: str) -> TaskDef:
    """Locate `tasks/level{1,2,3}/<task_id>/` and parse it."""
    task_dir: Path | None = None
    for level in ("level1", "level2", "level3"):
        candidate = TASKS_ROOT / level / task_id
        if candidate.is_dir():
            task_dir = candidate
            break
    if task_dir is None:
        raise FileNotFoundError(f"Task {task_id!r} not found under {TASKS_ROOT}/")

    meta = yaml.safe_load((task_dir / "meta.yaml").read_text())
    shapes_doc = yaml.safe_load((task_dir / "shapes.yaml").read_text())
    contract_md = (task_dir / "contract.md").read_text()
    kernel_py = (task_dir / "kernel.py").read_text()

    public_shapes = [
        ShapeSpec(
            id=s["id"],
            name=s["name"],
            params={k: v for k, v in s.items() if k not in {"id", "name"}},
        )
        for s in shapes_doc["public_shapes"]
    ]

    raw_level = meta["level"]
    if isinstance(raw_level, str):
        level_str = raw_level.upper().lstrip("L")
        level_int = int(level_str) if level_str.isdigit() else 0
    else:
        level_int = int(raw_level)

    # `cost_model` is optional at the loader level (backward compatible):
    # a missing block yields `TaskDef.cost_model=None`; a malformed block
    # raises CostModelError so a typo never silently produces a bad baseline.
    cost_model: CostModel | None = None
    raw_cost_model = meta.get("cost_model")
    if raw_cost_model is not None:
        try:
            cost_model = parse_cost_model(raw_cost_model)
        except CostModelError as e:
            raise CostModelError(f"{task_id}: {e}") from e

    # `param_constraints` is FULLY lenient at the loader level (WA.1,
    # `specs/task_api_contract.md §5.2`). Unlike `cost_model` above — which
    # eager-parses and raises on a malformed block — `param_constraints` is
    # kept as the raw dict and NEVER validated / parsed here:
    #   * missing  -> `TaskDef.param_constraints = None`;
    #   * present  -> stored verbatim, even when schema-illegal, so a
    #     malformed block reaches the WA.4 `task_status=blocked` gate
    #     instead of crashing `load_task`.
    # This is a deliberate departure from `cost_model`'s eager-parse-then-raise
    # (see `cost_model` handling just above). Presence + schema enforcement
    # is the job of `scripts/validate_tasks.py` ("validator-strict").
    raw_param_constraints = meta.get("param_constraints")
    param_constraints: dict[str, Any] | None = None
    if raw_param_constraints is not None:
        if isinstance(raw_param_constraints, dict):
            param_constraints = raw_param_constraints
        else:
            # A non-dict `param_constraints` (e.g. a stray list / scalar) is
            # still malformed-but-present: keep it reachable by the validator
            # rather than dropping it. `TaskDef.param_constraints` is typed
            # `dict | None`, so wrap a non-dict payload under a sentinel key.
            logger.warning(
                "%s: param_constraints is not a mapping (%s); kept raw for "
                "validate_tasks", task_id, type(raw_param_constraints).__name__,
            )
            param_constraints = {"__raw__": raw_param_constraints}

    # V9 plan M1.2: typed view of the nested ``provenance.golden:`` sub-
    # block. Returns ``None`` when the block is absent -- the validator
    # gates presence, not the loader.
    provenance_golden = parse_provenance_golden(meta)

    return TaskDef(
        task_id=task_id,
        level=level_int,
        task_dir=task_dir,
        meta=meta,
        public_shapes=public_shapes,
        contract_md=contract_md,
        kernel_py=kernel_py,
        kernel_py_sha256=_sha256_text(kernel_py),
        cost_model=cost_model,
        param_constraints=param_constraints,
        provenance_golden=provenance_golden,
    )


def iter_all_tasks() -> list[TaskDef]:
    out: list[TaskDef] = []
    for level in ("level1", "level2", "level3"):
        level_dir = TASKS_ROOT / level
        if not level_dir.is_dir():
            continue
        for d in sorted(level_dir.iterdir()):
            if d.is_dir() and not d.name.startswith("_"):
                out.append(load_task(d.name))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Canary discovery (W2.5 — DESIGN-V2 §6.8 / §8)
# ─────────────────────────────────────────────────────────────────────────────
# Canary trap tasks live under `tasks/_canary/`. Because `iter_all_tasks()`
# skips `_`-prefixed directories by design, a canary can never leak into the
# score-eligible denominator. They are loaded ONLY by the dedicated enumerator
# below and fed to `adversarial.check_canaries` through an independent path.


@dataclass
class CanaryTaskDef:
    """A loaded canary trap task (`tasks/_canary/<canary_id>/`).

    A canary mirrors a real task's interface but carries an intentionally wrong
    (trap) oracle. It is NOT score-eligible. Distinct from :class:`TaskDef`:

    * ``source_task_id`` binds the canary to a real level1/2/3 task — the
      runner reuses the submission's ``solution.py`` for THAT task to run the
      canary (the submitter writes no canary-specific solution).
    * ``canary_trap_type`` is the trap class (1 of 10; DESIGN-V2 §8).
    * ``expected_honest_pass`` is whether an honest submission should PASS the
      canary; ``check_canaries`` flags any ``got != expected_honest_pass``.

    A canary carries no ``kernel.py`` (it has no own solution), so unlike
    :class:`TaskDef` there is no ``kernel_py`` field.
    """

    canary_id: str
    task_dir: Path
    meta: dict[str, Any]
    public_shapes: list[ShapeSpec]
    contract_md: str
    source_task_id: str
    canary_trap_type: str
    expected_honest_pass: bool
    oracle_module_name: str = "oracle_pytorch"

    def as_task_def(self) -> TaskDef:
        """Adapt this canary into a :class:`TaskDef` for the grading machinery.

        `eval.evaluate(task_def=...)` grades a task purely from its `task_dir`
        / `meta` / `tolerance_class` / `cost_model`; it never reads
        `kernel_py`. A canary has no `kernel.py`, so the adapter sets an empty
        `kernel_py` (and its sha) — every field the runner actually touches is
        the canary's own. `level=0` because a canary is not a level1/2/3 task.
        """
        return TaskDef(
            task_id=self.canary_id,
            level=0,
            task_dir=self.task_dir,
            meta=self.meta,
            public_shapes=list(self.public_shapes),
            contract_md=self.contract_md,
            kernel_py="",
            kernel_py_sha256=_sha256_text(""),
            cost_model=None,
        )


# Files a canary task directory must contain. A canary has NO `kernel.py` and
# NO submitter solution — `source_task_id` binds it to a real task instead.
CANARY_REQUIRED_FILES = frozenset(
    {"meta.yaml", "shapes.yaml", "contract.md", "oracle_pytorch.py"}
)


def load_canary_task(canary_id: str) -> CanaryTaskDef:
    """Load a single canary task from `tasks/_canary/<canary_id>/`.

    Raises ``FileNotFoundError`` if the directory is absent and ``KeyError`` if
    a required canary-schema field is missing. Static schema enforcement (incl.
    ``source_task_id`` pointing at a real task) is the job of
    ``scripts/validate_tasks.py``'s `_canary` entry point — this loader stays
    lenient enough to surface a malformed canary to the validator.
    """
    task_dir = CANARY_ROOT / canary_id
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Canary {canary_id!r} not found under {CANARY_ROOT}/")

    meta = yaml.safe_load((task_dir / "meta.yaml").read_text())
    shapes_doc = yaml.safe_load((task_dir / "shapes.yaml").read_text())
    contract_md = (task_dir / "contract.md").read_text()

    public_shapes = [
        ShapeSpec(
            id=s["id"],
            name=s["name"],
            params={k: v for k, v in s.items() if k not in {"id", "name"}},
        )
        for s in (shapes_doc.get("public_shapes") or [])
    ]

    return CanaryTaskDef(
        canary_id=canary_id,
        task_dir=task_dir,
        meta=meta,
        public_shapes=public_shapes,
        contract_md=contract_md,
        source_task_id=meta["source_task_id"],
        canary_trap_type=meta["canary_trap_type"],
        expected_honest_pass=bool(meta["expected_honest_pass"]),
    )


def iter_canary_tasks() -> list[CanaryTaskDef]:
    """Discover and load every canary trap task under `tasks/_canary/`.

    This is the canary-discovery enumerator (W2.5). It is the ONLY path that
    surfaces canaries — `iter_all_tasks()` skips them by design. Canaries
    discovered here are fed to `adversarial.check_canaries` and never enter the
    score-eligible set computed by `eligibility.compute_score_eligible`.
    """
    out: list[CanaryTaskDef] = []
    if not CANARY_ROOT.is_dir():
        return out
    for d in sorted(CANARY_ROOT.iterdir()):
        if not d.is_dir():
            continue
        # A canary directory has a meta.yaml; skip stray scratch dirs.
        if not (d / "meta.yaml").is_file():
            logger.warning("skipping non-canary dir under _canary/: %s", d.name)
            continue
        out.append(load_canary_task(d.name))
    return out
