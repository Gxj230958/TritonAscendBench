"""Triage summary metrics (Unit 3.2) -- 4 independent backlog dials.

INTERNAL bench-team automation. CLI entry-point that consumes a
``triage_state.json`` (Unit 1.4 / Unit 1.9 owned) plus per-task
``meta.yaml`` files and emits two artifacts under
``build/reports/``:

  * ``summary-metrics-<ISO_DATE>.json`` -- machine-readable per-task
    flags + aggregate counts.
  * ``summary-metrics-<ISO_DATE>.md``   -- human markdown report.

The four metrics are **independent** by design (R4 settled). A single
task may satisfy zero, one, or all of them simultaneously; the sum can
therefore exceed the total task count. Do NOT add a
"sum <= total" invariant -- it is wrong on purpose.

Metric definitions (the canonical formulas):

1. ``reverify_nonblocked`` --
   ``task.last_consumed_verdict in {"PASS", "SKIPPED_BY_DESIGN"}``.
   Reads the common-audit field on the task entry, NOT ``state == DONE``;
   this counts every task whose last bench/reverify consumption cleared
   the BLOCKED gate, regardless of whether the orchestrator has since
   transitioned the entry to a follow-up state (e.g.
   ``BACKLOG_PENDING_GOLDEN``).
2. ``phase_a_promoted`` -- ``meta.yaml.phase_a_prime.status == "PASS"``.
   Does NOT include ``SKIPPED_BY_DESIGN`` (that is the phase B
   delegation lane, counted by ``golden_ready`` when the capture lands).
   Does NOT exclude ``release_scope.excluded`` -- a task can be both
   ``phase_a_promoted`` AND ``manual_backlog`` simultaneously.
3. ``golden_ready`` -- ``meta.yaml.phase_b_capture.status == "DONE"``
   OR the legacy ``tasks/<group>/<task_id>/golden/`` directory is
   complete (every declared public + hidden shape has a USABLE
   ``<shape_id>.safetensors`` artifact, reusing the
   ``eligibility.golden_artifact_usable`` predicate so this matches
   ``validate_tasks.py``'s static golden gate).
4. ``manual_backlog`` -- ``meta.yaml.release_scope.excluded == True``.
   The single source of truth is ``meta.release_scope.excluded``, NOT
   the task ``state`` enum -- operators mark exclusion in meta and the
   eligibility / release-gate layer derives the rest.

All four metrics are independent: a single task may satisfy zero, one,
or all four. ``release_scope.excluded`` does NOT remove a task from
``phase_a_promoted`` / ``golden_ready`` -- the overlap is the point.

Public API
----------
* ``load_state(state_path)``
* ``load_task_meta(tasks_root, task_id)``
* ``compute_metrics(state, tasks_root)``
* ``compute_per_task_flags(state, tasks_root)``
* ``render_markdown(metrics, *, date_iso, total, per_state_count,
  state_path, tasks_root)``
* ``render_json(metrics, per_task, *, date_iso, total,
  per_state_count, state_path, tasks_root)``
* ``main(argv=None)`` -- CLI entry-point.
"""
from __future__ import annotations

# ---- KTD-11 sys.path bootstrap (CLI entry-point) -------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
# --------------------------------------------------------------------------

import argparse  # noqa: E402
import logging  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Any  # noqa: E402

import yaml  # noqa: E402

from automation.fix_orchestrator import load_existing_triage_state  # noqa: E402
from automation.lib.atomic import atomic_write_json, atomic_write_text  # noqa: E402

logger = logging.getLogger("automation.summary_metrics")

# Verdict values that count as "non-blocked" for ``reverify_nonblocked``.
# PASS = the canonical happy path; SKIPPED_BY_DESIGN = phase A' opted
# out (delegation to phase B golden), which is also non-blocking.
REVERIFY_NONBLOCKED_VERDICTS: frozenset[str] = frozenset({
    "PASS", "SKIPPED_BY_DESIGN",
})

METRIC_NAMES: tuple[str, ...] = (
    "reverify_nonblocked",
    "phase_a_promoted",
    "golden_ready",
    "manual_backlog",
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_state(state_path: Path) -> dict[str, Any]:
    """Load the orchestrator triage state via the canonical strict reader.

    Delegates to ``fix_orchestrator.load_existing_triage_state`` so we
    share the same schema-version + per-task validation guard. Raises
    the same ``StateFileNotFoundError`` / ``StateFileCorruptError`` the
    orchestrator does.
    """
    return load_existing_triage_state(state_path)


def _find_task_dir(tasks_root: Path, task_id: str) -> Path | None:
    """Locate ``tasks/<group>/<task_id>/`` under ``tasks_root``.

    Tasks live one level deep under ``tasks_root`` (e.g. ``level1/``,
    ``level2/``, ``_canary/``, ``_vendored/``). We accept either layout:
    a direct ``tasks_root/<task_id>/`` for tests with a flat fixture,
    or any ``tasks_root/*/<task_id>/`` for the production layout.
    Returns the resolved task dir or ``None`` if not found.
    """
    direct = tasks_root / task_id
    if direct.is_dir():
        return direct
    for child in tasks_root.iterdir():
        if not child.is_dir():
            continue
        candidate = child / task_id
        if candidate.is_dir():
            return candidate
    return None


def load_task_meta(tasks_root: Path, task_id: str) -> dict[str, Any]:
    """Load ``tasks/<...>/<task_id>/meta.yaml`` and return its parsed
    dict. Returns ``{}`` (with a warning log) when the file is missing
    or unparseable -- callers must tolerate missing meta gracefully so
    that a single stale task does not abort the whole report run.
    """
    task_dir = _find_task_dir(tasks_root, task_id)
    if task_dir is None:
        logger.warning(
            "load_task_meta: task dir for %s not found under %s",
            task_id, tasks_root,
        )
        return {}
    meta_path = task_dir / "meta.yaml"
    if not meta_path.is_file():
        logger.warning(
            "load_task_meta: meta.yaml missing for %s at %s",
            task_id, meta_path,
        )
        return {}
    try:
        data = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        logger.warning(
            "load_task_meta: meta.yaml unparseable for %s (%s)",
            task_id, exc,
        )
        return {}
    if data is None:
        return {}
    if not isinstance(data, dict):
        logger.warning(
            "load_task_meta: meta.yaml for %s is not a mapping (got %s)",
            task_id, type(data).__name__,
        )
        return {}
    return data


# ---------------------------------------------------------------------------
# Legacy golden completeness (mirrors validate_tasks.py / eligibility)
# ---------------------------------------------------------------------------

def _legacy_golden_complete(tasks_root: Path, task_id: str) -> bool:
    """True iff ``tasks/<group>/<task_id>/golden/`` carries a USABLE
    artifact for every declared shape (public + hidden), reusing the
    same predicate ``validate_tasks.py``'s static golden gate and
    ``eligibility.golden_complete`` apply. Returns False on a missing
    task dir, missing ``shapes.yaml``, missing ``golden/`` directory,
    empty shape set, or any unloadable artifact.
    """
    task_dir = _find_task_dir(tasks_root, task_id)
    if task_dir is None:
        return False
    shapes_path = task_dir / "shapes.yaml"
    if not shapes_path.is_file():
        return False
    try:
        doc = yaml.safe_load(shapes_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return False
    if not isinstance(doc, dict):
        return False
    from ascendbench.input_spec.shape_format import (
        ShapeFormatError,
        hidden_ids,
        public_ids,
    )
    try:   # adapter handles either format; a malformed explicit doc fails closed (return False)
        shape_ids = [str(s) for s in (list(public_ids(doc)) + list(hidden_ids(doc)))]
    except ShapeFormatError:
        return False
    if not shape_ids:
        return False
    golden_dir = task_dir / "golden"
    if not golden_dir.is_dir():
        return False
    # Reuse the shared "exists AND safetensors loads" predicate so the
    # runtime loader, the static validator, and this metric all agree
    # on what counts as a usable golden artifact.
    from ascendbench.release_gate.eligibility import golden_artifact_usable
    return all(
        golden_artifact_usable(golden_dir / f"{sid}.safetensors")
        for sid in shape_ids
    )


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _meta_phase_a_pass(meta: dict[str, Any]) -> bool:
    block = meta.get("phase_a_prime")
    if not isinstance(block, dict):
        return False
    return block.get("status") == "PASS"


def _meta_phase_b_done(meta: dict[str, Any]) -> bool:
    block = meta.get("phase_b_capture")
    if not isinstance(block, dict):
        return False
    return block.get("status") == "DONE"


def _meta_release_scope_excluded(meta: dict[str, Any]) -> bool:
    block = meta.get("release_scope")
    if not isinstance(block, dict):
        return False
    return block.get("excluded") is True


def _is_reverify_nonblocked(entry: dict[str, Any]) -> bool:
    """``task.last_consumed_verdict`` lives in the common-audit fields
    set by ``state_machine.consume_result`` -- independent of the
    current ``state`` enum. A task whose most-recent bench / reverify
    consumption returned PASS or SKIPPED_BY_DESIGN counts toward this
    metric even if it has since moved into a follow-up state (e.g.
    ``BACKLOG_PENDING_GOLDEN`` for SKIPPED_BY_DESIGN delegation)."""
    return entry.get("last_consumed_verdict") in REVERIFY_NONBLOCKED_VERDICTS


def _is_phase_a_promoted(meta: dict[str, Any]) -> bool:
    """``meta.phase_a_prime.status == "PASS"``. Independent of task
    state and of ``release_scope.excluded`` (overlap is intentional)."""
    return _meta_phase_a_pass(meta)


def _is_golden_ready(
    meta: dict[str, Any], tasks_root: Path, task_id: str,
) -> bool:
    """``meta.phase_b_capture.status == "DONE"`` OR the legacy
    ``golden/`` directory carries a usable artifact for every declared
    shape -- the two paths are equivalent ways of declaring "the
    upstream-kernel golden has been captured / shipped for this
    task"."""
    if _meta_phase_b_done(meta):
        return True
    return _legacy_golden_complete(tasks_root, task_id)


def _is_manual_backlog(meta: dict[str, Any]) -> bool:
    """``meta.release_scope.excluded == True`` -- the operator-marked
    exclusion field is the canonical source of truth, NOT the task
    state enum (state may legally still be DONE / BACKLOG_* / etc.;
    the eligibility & release-gate layers derive everything else off
    this single meta field)."""
    return _meta_release_scope_excluded(meta)


def compute_per_task_flags(
    state: dict[str, Any],
    tasks_root: Path,
) -> dict[str, dict[str, bool]]:
    """Return ``{task_id: {metric_name: bool}}`` for every task in the
    state file. Used both for the JSON audit payload and as the input
    to ``compute_metrics``.

    Missing ``meta.yaml`` collapses meta-dependent metrics to ``False``
    (with a warning) but never raises.
    """
    out: dict[str, dict[str, bool]] = {}
    tasks: dict[str, Any] = state.get("tasks", {})
    for task_id, entry in tasks.items():
        meta = load_task_meta(tasks_root, task_id)
        out[task_id] = {
            "reverify_nonblocked": _is_reverify_nonblocked(entry),
            "phase_a_promoted": _is_phase_a_promoted(meta),
            "golden_ready": _is_golden_ready(meta, tasks_root, task_id),
            "manual_backlog": _is_manual_backlog(meta),
        }
    return out


def compute_metrics(
    state: dict[str, Any],
    tasks_root: Path,
) -> dict[str, int]:
    """Compute the 4 independent metric counts.

    NOTE: the 4 metrics are independent. A single task may count toward
    multiple metrics simultaneously. We deliberately do NOT enforce a
    ``sum(metrics) <= total`` invariant -- that bound is incorrect
    given overlap.
    """
    flags = compute_per_task_flags(state, tasks_root)
    counts = dict.fromkeys(METRIC_NAMES, 0)
    for per_task in flags.values():
        for name in METRIC_NAMES:
            if per_task.get(name):
                counts[name] += 1
    return counts


def compute_per_state_count(state: dict[str, Any]) -> dict[str, int]:
    """Per-state tally for the markdown breakdown table."""
    tasks: dict[str, Any] = state.get("tasks", {})
    counter: Counter[str] = Counter()
    for entry in tasks.values():
        counter[str(entry.get("state", ""))] += 1
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _pct(numer: int, denom: int) -> str:
    if denom <= 0:
        return "0.0%"
    return f"{(numer / denom) * 100:.1f}%"


_METRIC_DEFINITIONS: dict[str, str] = {
    "reverify_nonblocked":
        "last_consumed_verdict in {PASS, SKIPPED_BY_DESIGN}",
    "phase_a_promoted":
        "meta.phase_a_prime.status == PASS",
    "golden_ready":
        "meta.phase_b_capture.status == DONE OR legacy golden/ complete",
    "manual_backlog":
        "meta.release_scope.excluded == True",
}


def render_markdown(
    metrics: dict[str, int],
    *,
    date_iso: str,
    total: int,
    per_state_count: dict[str, int] | None = None,
    state_path: Path | str = "",
    tasks_root: Path | str = "",
) -> str:
    """Render the markdown summary. ``per_state_count`` is optional
    (the breakdown table is omitted when empty)."""
    lines: list[str] = []
    lines.append(f"# Triage Summary Metrics -- {date_iso}")
    lines.append("")
    lines.append("Generated by: scripts/automation/summary_metrics.py")
    lines.append(f"State file: {state_path}")
    lines.append(f"Tasks root: {tasks_root}")
    lines.append(f"Total tasks: {total}")
    lines.append("")
    lines.append("## Metrics (4 independent, may overlap)")
    lines.append("")
    lines.append("| Metric | Count | / Total | Definition |")
    lines.append("|---|---|---|---|")
    for name in METRIC_NAMES:
        count = metrics.get(name, 0)
        lines.append(
            f"| {name} | {count} | {_pct(count, total)} | "
            f"{_METRIC_DEFINITIONS[name]} |"
        )
    lines.append("")
    lines.append(
        "> Metrics are INDEPENDENT by design. A single task may count "
        "toward multiple"
    )
    lines.append(
        "> metrics (e.g., a `phase_a_promoted` task whose operator has "
        "also marked"
    )
    lines.append(
        "> `release_scope.excluded = true` counts toward BOTH "
        "phase_a_promoted AND"
    )
    lines.append(
        "> manual_backlog). Do NOT expect "
        "`reverify_nonblocked + phase_a_promoted +"
    )
    lines.append(
        "> golden_ready + manual_backlog == total` -- the sum may "
        "legally exceed total."
    )
    lines.append("")

    if per_state_count:
        lines.append("## Per-state Breakdown")
        lines.append("")
        lines.append("| State | Count |")
        lines.append("|---|---|")
        for state_name, count in per_state_count.items():
            lines.append(f"| {state_name} | {count} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def render_json(
    metrics: dict[str, int],
    per_task: dict[str, dict[str, bool]],
    *,
    date_iso: str,
    total: int,
    per_state_count: dict[str, int] | None = None,
    state_path: Path | str = "",
    tasks_root: Path | str = "",
) -> dict[str, Any]:
    """Return the JSON payload dict (caller is responsible for writing
    it to disk through ``atomic_write_json``)."""
    return {
        "generated_at": _iso_utc_now_seconds(),
        "date_iso": date_iso,
        "state_path": str(state_path),
        "tasks_root": str(tasks_root),
        "total": total,
        "metrics": dict(metrics),
        "per_state_count": dict(per_state_count or {}),
        "per_task_flags": per_task,
    }


def _iso_utc_now_seconds() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="summary_metrics",
        description=(
            "Compute 4 independent triage backlog dials "
            "(reverify_nonblocked / phase_a_promoted / golden_ready / "
            "manual_backlog) and emit markdown + JSON summaries under "
            "build/reports/."
        ),
    )
    p.add_argument(
        "--state-path",
        required=True,
        type=Path,
        help="Path to triage_state.json (Unit 1.4 / Unit 1.9 owned).",
    )
    p.add_argument(
        "--tasks-root",
        type=Path,
        default=Path("./tasks"),
        help="Root dir holding per-task meta.yaml (default: ./tasks).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./build/reports"),
        help=(
            "Output dir for summary-metrics-<DATE>.{json,md} "
            "(default: ./build/reports)."
        ),
    )
    p.add_argument(
        "--format",
        choices=("markdown", "json", "both"),
        default="both",
        help="Which artifacts to write (default: both).",
    )
    p.add_argument(
        "--date-iso",
        default=None,
        help=(
            "ISO date stamp for filenames (default: today UTC, "
            "YYYY-MM-DD)."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="stdlib logging level (default: INFO).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    state_path: Path = args.state_path.resolve()
    tasks_root: Path = args.tasks_root.resolve()
    out_dir: Path = args.out_dir.resolve()
    date_iso: str = args.date_iso or _today_iso()

    logger.info(
        "summary_metrics: state=%s tasks_root=%s out_dir=%s date=%s "
        "format=%s",
        state_path, tasks_root, out_dir, date_iso, args.format,
    )

    state = load_state(state_path)
    total = len(state.get("tasks", {}))
    per_task = compute_per_task_flags(state, tasks_root)
    # Aggregate from per_task to avoid re-reading meta files.
    metrics = dict.fromkeys(METRIC_NAMES, 0)
    for flags in per_task.values():
        for name in METRIC_NAMES:
            if flags.get(name):
                metrics[name] += 1
    per_state_count = compute_per_state_count(state)

    md_path = out_dir / f"summary-metrics-{date_iso}.md"
    json_path = out_dir / f"summary-metrics-{date_iso}.json"

    if args.format in ("markdown", "both"):
        md_text = render_markdown(
            metrics,
            date_iso=date_iso,
            total=total,
            per_state_count=per_state_count,
            state_path=state_path,
            tasks_root=tasks_root,
        )
        atomic_write_text(md_path, md_text)
        logger.info("wrote markdown: %s", md_path)

    if args.format in ("json", "both"):
        payload = render_json(
            metrics,
            per_task,
            date_iso=date_iso,
            total=total,
            per_state_count=per_state_count,
            state_path=state_path,
            tasks_root=tasks_root,
        )
        atomic_write_json(json_path, payload)
        logger.info("wrote json: %s", json_path)

    logger.info(
        "metrics: reverify_nonblocked=%d phase_a_promoted=%d "
        "golden_ready=%d manual_backlog=%d (total=%d)",
        metrics["reverify_nonblocked"],
        metrics["phase_a_promoted"],
        metrics["golden_ready"],
        metrics["manual_backlog"],
        total,
    )
    return 0


__all__ = [
    "METRIC_NAMES",
    "REVERIFY_NONBLOCKED_VERDICTS",
    "compute_metrics",
    "compute_per_state_count",
    "compute_per_task_flags",
    "load_state",
    "load_task_meta",
    "main",
    "render_json",
    "render_markdown",
]


if __name__ == "__main__":
    raise SystemExit(main())
