"""Backlog report generator (Unit 3.1).

CLI entry-point. Reads the orchestrator triage state file
(``triage_state.json``, Unit 1.9 strict schema) plus, optionally, each
task's last ``result.json`` produced under ``results/<run_id>/`` (Unit
1.7 strict 9-field schema), partitions every non-DONE task by canonical
state, and writes a human-readable markdown report plus a JSON sidecar
to ``build/reports/backlog-report-<ISO_DATE>.{md,json}``.

The report is **read-only** -- it never mutates the state file. It is
the produce-once, review-many artifact that the bench team and the
orchestrator's M3 review loop both consume.

Schema invariants enforced here
-------------------------------
* state file: strict via ``load_existing_triage_state`` (Unit 1.9 SoT
  loader). Missing file => exit non-zero. Corrupt => loader quarantines
  + raises; backlog report does not generate.
* state_data: defensive ``.get(...) or default`` reads. A task in
  transition (e.g. ``BENCH_RUNNING`` mid-write) renders with ``n/a``
  placeholders rather than failing the whole report.
* output directory: ``build/reports/`` is gitignored -- generated
  artifacts MUST stay there so they do not leak into the repo or
  release tarballs.
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
import json  # noqa: E402
import logging  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Any  # noqa: E402

from automation.fix_orchestrator import (  # noqa: E402
    StateFileCorruptError,
    StateFileNotFoundError,
    load_existing_triage_state_relaxed,
)
from automation.lib import atomic  # noqa: E402

logger = logging.getLogger("automation.backlog_report")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# States that count as "backlog" -- anything pending human / system action
# beyond a clean DONE.
BACKLOG_STATES: tuple[str, ...] = (
    "BACKLOG_PENDING_GOLDEN",
    "BACKLOG_UNEXPECTED",
    "UNFIXABLE_BY_LLM",
    "MANUAL_BACKLOG",
    "HUMAN_REVIEW_PENDING",
    "HUMAN_REVIEW_EXPIRED",
    "DEFERRAL_ACCEPTED",
)

# Human-estimate hours per verdict_class (used in markdown table).
# ``smoke-fail`` is manual-only (upstream invocation crashed during the
# smoke phase): no LLM remediation attempt, the entry sits in
# BACKLOG_UNEXPECTED until an operator triages -- 2 hr matches the
# unexpected-blocked estimate because the diagnosis surface is similar.
_VERDICT_HOURS: dict[str, str] = {
    "shapes-incomplete": "0.25",
    "oracle-mismatch": "0.5",
    "upstream-not-invocable-here": "1.0",
    "unexpected-blocked": "2.0",
    "smoke-fail": "2.0",
}


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_state(state_path: Path) -> dict[str, Any]:
    """Read-only loader: preserves the top-level schema invariants
    (schema_version, tasks dict shape, COMMON_FIELDS, verdict_class
    enum, history kind/at invariants) but tolerates a partial-write
    ``state_data`` (e.g. a BENCH_RUNNING entry whose
    ``pending_run_id`` was not yet flushed to disk). The downstream
    renderers use ``.get(..., "n/a")`` so a transitional entry surfaces
    as ``n/a`` placeholders rather than crashing the whole report.

    Raises ``StateFileNotFoundError`` if the file is missing and
    ``StateFileCorruptError`` if a top-level invariant is violated.
    """
    return load_existing_triage_state_relaxed(state_path)


def _safe_state_data(entry: dict[str, Any]) -> dict[str, Any]:
    sd = entry.get("state_data") or {}
    if not isinstance(sd, dict):
        return {}
    return sd


def partition_tasks(
    state: dict[str, Any], *, include_done: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Bucket every task entry by its canonical ``state``.

    Returns a dict mapping state name -> ``[{"task_id": ..., "entry": ...}, ...]``.
    By default only the canonical backlog states (``BACKLOG_STATES``)
    are included. ``include_done=True`` also adds a ``DONE`` bucket.
    Tasks in active in-flight states (``QUEUED`` / ``PROMPTING`` /
    ``BENCH_RUNNING`` / ...) are never bucketed: they are work-in-flight,
    not backlog.
    """
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in BACKLOG_STATES}
    if include_done:
        out["DONE"] = []
    tasks = state.get("tasks") or {}
    for tid, entry in tasks.items():
        st = entry.get("state")
        if st in out:
            out[st].append({"task_id": tid, "entry": entry})
    # Stable order: sort each bucket by task_id for diff-friendliness.
    for bucket in out.values():
        bucket.sort(key=lambda r: r["task_id"])
    return out


def enrich_with_result(
    task_entry: dict[str, Any], results_root: Path,
) -> dict[str, Any]:
    """Attach the latest ``result.json`` content for a task entry.

    Reads ``<results_root>/<last_consumed_run_id>/result.json`` and
    merges its fields onto a shallow copy of ``task_entry``. If the
    task has no ``last_consumed_run_id`` (never reverified) or the
    file is missing / unreadable, returns the entry with ``result_*``
    fields set to ``None`` -- never raises.

    The merged keys (all prefixed ``result_``) are:
      * ``result_verdict``
      * ``result_verdict_class``
      * ``result_reason``
      * ``result_produced_at``
      * ``result_producer``
      * ``result_cmd``
    """
    out = dict(task_entry)
    out["result_verdict"] = None
    out["result_verdict_class"] = None
    out["result_reason"] = None
    out["result_produced_at"] = None
    out["result_producer"] = None
    out["result_cmd"] = None
    run_id = task_entry.get("last_consumed_run_id")
    if not run_id:
        return out
    result_path = Path(results_root) / str(run_id) / "result.json"
    if not result_path.is_file():
        return out
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning(
            "failed to read result.json at %s; skipping enrichment",
            result_path,
        )
        return out
    if not isinstance(payload, dict):
        return out
    out["result_verdict"] = payload.get("verdict")
    out["result_verdict_class"] = payload.get("verdict_class")
    out["result_reason"] = payload.get("reason")
    out["result_produced_at"] = payload.get("produced_at")
    out["result_producer"] = payload.get("producer")
    out["result_cmd"] = payload.get("cmd")
    return out


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _verdict_class_mix(entries: list[dict[str, Any]]) -> str:
    counter: Counter[str] = Counter()
    for row in entries:
        vc = row["entry"].get("verdict_class") or "unknown"
        counter[vc] += 1
    if not counter:
        return "-"
    parts = [f"{k}:{v}" for k, v in sorted(counter.items())]
    return ", ".join(parts)


def _history_tail(entry: dict[str, Any], n: int = 3) -> list[dict[str, Any]]:
    history = entry.get("history") or []
    if not isinstance(history, list):
        return []
    return list(history[-n:])


def _count_history_kind(entry: dict[str, Any], kind: str) -> int:
    history = entry.get("history") or []
    if not isinstance(history, list):
        return 0
    return sum(1 for h in history if isinstance(h, dict) and h.get("kind") == kind)


def _count_done_via_promotion(state: dict[str, Any]) -> tuple[int, int]:
    """Return (promoted_count, total_done_skipped_by_design).

    Counts ``DONE`` tasks whose ``final_verdict == "SKIPPED_BY_DESIGN"``.
    Of those, ``promoted_count`` is the subset whose history contains a
    ``capture_done_promoted`` event (rule b: sweep promote). The
    remainder are rule a (immediate DONE at consume time).
    """
    promoted = 0
    total = 0
    tasks = state.get("tasks") or {}
    for entry in tasks.values():
        if entry.get("state") != "DONE":
            continue
        sd = _safe_state_data(entry)
        if sd.get("final_verdict") != "SKIPPED_BY_DESIGN":
            continue
        total += 1
        if _count_history_kind(entry, "capture_done_promoted") > 0:
            promoted += 1
    return promoted, total


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def _truncate(s: str | None, limit: int = 200) -> str:
    if s is None:
        return "n/a"
    s = str(s)
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def _md_escape_inline(s: str) -> str:
    # Cheap inline escape: pipe + newline are the only chars that wreck
    # a markdown table cell. Truncation already prevents huge cells.
    return s.replace("|", "\\|").replace("\n", " ")


def _render_task_md(row: dict[str, Any]) -> str:
    tid = row["task_id"]
    entry = row["entry"]
    sd = _safe_state_data(entry)
    enriched = row.get("enriched") or {}
    last_verdict = entry.get("last_consumed_verdict") or "n/a"
    last_at = entry.get("last_consumed_at") or "n/a"
    last_run = entry.get("last_consumed_run_id") or "n/a"
    attempt = entry.get("attempt", 0)
    verdict_class = entry.get("verdict_class") or "unknown"
    hours = _VERDICT_HOURS.get(verdict_class, "?")
    result_verdict = enriched.get("result_verdict") or "n/a"
    result_reason = _truncate(enriched.get("result_reason"))

    lines: list[str] = []
    lines.append(f"### {tid}")
    lines.append("")
    lines.append(f"- verdict_class: `{verdict_class}` (estimated_hours={hours})")
    lines.append(f"- attempt: {attempt}")
    lines.append(f"- last_consumed_at: {last_at}")
    lines.append(f"- last_consumed_run_id: `{last_run}`")
    lines.append(f"- last_consumed_verdict: `{last_verdict}`")
    lines.append(f"- result_verdict (from result.json): `{result_verdict}`")
    lines.append(f"- result_reason: {result_reason}")
    if sd:
        # Sort keys for stable rendering.
        for key in sorted(sd.keys()):
            val = sd[key]
            lines.append(f"- state_data.{key}: `{_truncate(val, 160)}`")
    tail = _history_tail(entry)
    if tail:
        lines.append("- history_tail:")
        for rec in tail:
            kind = rec.get("kind", "?") if isinstance(rec, dict) else "?"
            at = rec.get("at", "?") if isinstance(rec, dict) else "?"
            lines.append(f"  - {at} [{kind}]")
    lines.append("")
    return "\n".join(lines)


def render_markdown(
    partitioned: dict[str, list[dict[str, Any]]],
    *,
    date_iso: str,
    state_path: Path,
    state: dict[str, Any],
    include_done: bool,
) -> str:
    """Render the full markdown report."""
    tasks = state.get("tasks") or {}
    total = len(tasks)
    done = sum(1 for e in tasks.values() if e.get("state") == "DONE")
    backlog = sum(
        len(partitioned[s]) for s in BACKLOG_STATES if s in partitioned
    )
    promoted, total_skipped = _count_done_via_promotion(state)

    lines: list[str] = []
    lines.append(f"# Triage Backlog Report -- {date_iso}")
    lines.append("")
    lines.append("Generated by: `scripts/automation/backlog_report.py`")
    lines.append(f"State file: `{state_path}`")
    lines.append(f"Total tasks: {total} ({backlog} backlog + {done} done)")
    lines.append("")
    lines.append(
        f"defer-to-golden: {promoted} of {total_skipped} SKIPPED_BY_DESIGN "
        "tasks reached DONE via sweep promote (capture_done_promoted)"
    )
    lines.append("")
    lines.append("## Backlog Summary")
    lines.append("")
    lines.append("| State | Count | Verdict Class Mix |")
    lines.append("|---|---|---|")
    for st in BACKLOG_STATES:
        rows = partitioned.get(st, [])
        mix = _verdict_class_mix(rows)
        lines.append(f"| {st} | {len(rows)} | {_md_escape_inline(mix)} |")
    if include_done:
        rows = partitioned.get("DONE", [])
        lines.append(
            f"| DONE | {len(rows)} | {_md_escape_inline(_verdict_class_mix(rows))} |"
        )
    lines.append("")

    # Per-state sections.
    section_order = list(BACKLOG_STATES)
    if include_done:
        section_order.append("DONE")
    for st in section_order:
        rows = partitioned.get(st, [])
        lines.append(f"## {st} ({len(rows)} tasks)")
        lines.append("")
        if not rows:
            lines.append("_no tasks in this state_")
            lines.append("")
            continue
        for row in rows:
            lines.append(_render_task_md(row))
    return "\n".join(lines).rstrip() + "\n"


def render_json(
    partitioned: dict[str, list[dict[str, Any]]],
    *,
    date_iso: str,
    state_path: Path,
    state: dict[str, Any],
    include_done: bool,
) -> dict[str, Any]:
    """Render the JSON sidecar (machine-consumable)."""
    tasks = state.get("tasks") or {}
    total = len(tasks)
    done = sum(1 for e in tasks.values() if e.get("state") == "DONE")
    backlog = sum(
        len(partitioned[s]) for s in BACKLOG_STATES if s in partitioned
    )
    promoted, total_skipped = _count_done_via_promotion(state)

    backlog_json: dict[str, list[dict[str, Any]]] = {}
    section_order = list(BACKLOG_STATES)
    if include_done:
        section_order.append("DONE")
    for st in section_order:
        rows = partitioned.get(st, [])
        out_rows: list[dict[str, Any]] = []
        for row in rows:
            entry = row["entry"]
            enriched = row.get("enriched") or {}
            out_rows.append({
                "task_id": row["task_id"],
                "verdict_class": entry.get("verdict_class"),
                "attempt": entry.get("attempt", 0),
                "last_consumed_at": entry.get("last_consumed_at"),
                "last_consumed_run_id": entry.get("last_consumed_run_id"),
                "last_consumed_verdict": entry.get("last_consumed_verdict"),
                "state_data": _safe_state_data(entry),
                "history_tail": _history_tail(entry),
                "patch_validation_failed_count": _count_history_kind(
                    entry, "patch_validation_failed",
                ),
                "result_consumed_count": _count_history_kind(
                    entry, "result_consumed",
                ),
                "result_verdict": enriched.get("result_verdict"),
                "result_verdict_class": enriched.get("result_verdict_class"),
                "result_reason": enriched.get("result_reason"),
                "result_produced_at": enriched.get("result_produced_at"),
                "result_producer": enriched.get("result_producer"),
                "result_cmd": enriched.get("result_cmd"),
            })
        backlog_json[st] = out_rows

    return {
        "schema_version": "1.0",
        "generated_at": _iso_utc_now(),
        "date_iso": date_iso,
        "state_path": str(state_path),
        "totals": {
            "total": total,
            "backlog": backlog,
            "done": done,
            "skipped_by_design_done": total_skipped,
            "promoted_via_sweep": promoted,
        },
        "backlog": backlog_json,
    }


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="backlog_report",
        description=(
            "Generate a backlog report from the orchestrator triage state. "
            "Writes markdown + JSON to build/reports/ (gitignored)."
        ),
    )
    p.add_argument(
        "--state-path", required=True, type=Path,
        help="Path to triage_state.json (Unit 1.9 strict schema).",
    )
    p.add_argument(
        "--results-root", type=Path, default=Path("./results"),
        help="Directory containing per-run-id result.json files "
             "(default: ./results).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("./build/reports"),
        help="Output directory for backlog report artifacts "
             "(default: ./build/reports).",
    )
    p.add_argument(
        "--format", choices=("markdown", "json", "both"), default="both",
        help="Which artifact(s) to write (default: both).",
    )
    p.add_argument(
        "--include-done", action="store_true", default=False,
        help="If set, include DONE tasks in the report (default: backlog only).",
    )
    p.add_argument(
        "--date-iso", type=str, default=None,
        help="ISO date YYYY-MM-DD used in filenames + header "
             "(default: today UTC).",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: INFO).",
    )
    return p


def _enrich_partition(
    partitioned: dict[str, list[dict[str, Any]]], results_root: Path,
) -> None:
    """Mutate ``partitioned`` in-place: attach an ``enriched`` dict to
    every row carrying result.json fields (or None placeholders)."""
    for rows in partitioned.values():
        for row in rows:
            row["enriched"] = enrich_with_result(row["entry"], results_root)


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    state_path: Path = args.state_path.resolve()
    try:
        state = load_state(state_path)
    except StateFileNotFoundError as exc:
        print(
            f"ERROR: state file not found at {state_path}; "
            "backlog report refuses to operate on empty state",
            file=sys.stderr, flush=True,
        )
        logger.error("%s", exc)
        return 2
    except StateFileCorruptError as exc:
        print(
            f"ERROR: state file at {state_path} is corrupt; "
            "see quarantine sibling .corrupt.<ts>.bak",
            file=sys.stderr, flush=True,
        )
        logger.error("%s", exc)
        return 3

    date_iso = args.date_iso or _iso_utc_today()
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    partitioned = partition_tasks(state, include_done=args.include_done)
    _enrich_partition(partitioned, args.results_root)

    print(
        f"[backlog_report] state={state_path} tasks={len(state.get('tasks') or {})} "
        f"date={date_iso}",
        flush=True,
    )

    if args.format in ("markdown", "both"):
        md = render_markdown(
            partitioned,
            date_iso=date_iso,
            state_path=state_path,
            state=state,
            include_done=args.include_done,
        )
        md_path = out_dir / f"backlog-report-{date_iso}.md"
        atomic.atomic_write_text(md_path, md)
        print(f"[backlog_report] wrote {md_path}", flush=True)

    if args.format in ("json", "both"):
        js = render_json(
            partitioned,
            date_iso=date_iso,
            state_path=state_path,
            state=state,
            include_done=args.include_done,
        )
        js_path = out_dir / f"backlog-report-{date_iso}.json"
        atomic.atomic_write_json(js_path, js)
        print(f"[backlog_report] wrote {js_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BACKLOG_STATES",
    "enrich_with_result",
    "load_state",
    "main",
    "partition_tasks",
    "render_json",
    "render_markdown",
]
