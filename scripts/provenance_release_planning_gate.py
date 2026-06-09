#!/usr/bin/env python3
"""provenance_release_planning_gate.py -- plan (R13c).

Compute the 4-state R13c go/no-go gate from the trusted census +
``provenance.golden`` view of the corpus.

Decision tree (plan):

    projected = projected_eligible(corpus,
                                   exclude_r11_pending=True,
                                   exclude_r17_high=True)

    PROCEED                        = |projected| >= 60
                                     AND |distinct_categories| >= 8
                                     AND rare_class_coverage_OK
    PROVISIONAL                    = |projected| >= 30
                                     AND |distinct_categories| >= 6
                                     AND rare_class_coverage_OK
    PROVISIONAL_MISSING_RARE       = size OK but rare-class coverage
                                     (convolution / spec_decode /
                                      normalization) < 50%
    STOP                           = |projected| < 30 OR < 6 categories

Rare-class set (corrected -- disjoint from R17 high-risk set):
``convolution`` (3 tasks), ``spec_decode`` (1 task), ``normalization``
(5 tasks). ``norm`` (1 task) is in the R17 high-risk set so it's
already leaderboard-excluded; it does NOT enter the rare set.

The script's exit code is the gate state (0=PROCEED, 1=PROVISIONAL,
2=PROVISIONAL_MISSING_RARE, 3=STOP) so CI pipelines can branch on it.
``validate_tasks --release-gate`` is intentionally NOT modified -- it
keeps its binary release-gate contract; this script is the separate
planning gate.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = REPO_ROOT / "tasks"

# Make the package importable when run from anywhere.
sys.path.insert(0, str(REPO_ROOT))

from ascendbench.release_gate.eligibility import (  # noqa: E402
    projected_eligible,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("provenance_release_planning_gate")


# ─────────────────────────────────────────────────────────────────────
# Gate-state vocabulary
# ─────────────────────────────────────────────────────────────────────

STATE_PROCEED = "PROCEED"
STATE_PROVISIONAL = "PROVISIONAL"
STATE_PROVISIONAL_MISSING_RARE = "PROVISIONAL_MISSING_RARE"
STATE_STOP = "STOP"

_EXIT_CODE = {
    STATE_PROCEED: 0,
    STATE_PROVISIONAL: 1,
    STATE_PROVISIONAL_MISSING_RARE: 2,
    STATE_STOP: 3,
}

# Rare-class set (final). DISJOINT from R17 high-
# risk -- `norm` is high-risk so leaderboard-excluded; the rare set
# only carries the small-population, NON-high-risk categories.
RARE_CLASS_SET = frozenset({"convolution", "spec_decode", "normalization"})


# Thresholds. Exposed as module constants so tests can reference them
# instead of duplicating magic numbers.
THRESH_PROCEED_SIZE = 60
THRESH_PROCEED_CATEGORIES = 8
THRESH_PROVISIONAL_SIZE = 30
THRESH_PROVISIONAL_CATEGORIES = 6
THRESH_RARE_CLASS_COVERAGE_RATIO = 0.5  # 50%


@dataclass
class _MiniTask:
    task_id: str
    meta: dict = field(default_factory=dict)


def _load_corpus() -> list[_MiniTask]:
    out: list[_MiniTask] = []
    for level in ("level1", "level2", "level3"):
        ld = TASKS_ROOT / level
        if not ld.is_dir():
            continue
        for td in sorted(ld.iterdir()):
            if td.is_dir() and not td.name.startswith("_"):
                meta_path = td / "meta.yaml"
                if not meta_path.is_file():
                    continue
                try:
                    meta = yaml.safe_load(meta_path.read_text()) or {}
                except yaml.YAMLError:
                    continue
                out.append(_MiniTask(task_id=td.name, meta=meta))
    return out


@dataclass
class GateOutcome:
    state: str
    projected_size: int
    distinct_categories: int
    rare_class_coverage: dict[str, int]
    rare_class_total_in_corpus: dict[str, int]
    rare_class_ratio: float
    reasons: list[str] = field(default_factory=list)


def _rare_class_corpus_counts(tasks: list[_MiniTask]) -> dict[str, int]:
    """Count rare-class tasks in the WHOLE corpus (denominator for the
    rare-class coverage ratio)."""
    out: dict[str, int] = {c: 0 for c in RARE_CLASS_SET}
    for t in tasks:
        cat = t.meta.get("category")
        if cat in out:
            out[cat] += 1
    return out


def _rare_class_projected_counts(
    tasks: list[_MiniTask], projected_ids: set[str]
) -> dict[str, int]:
    out: dict[str, int] = {c: 0 for c in RARE_CLASS_SET}
    proj_set = projected_ids
    for t in tasks:
        if t.task_id not in proj_set:
            continue
        cat = t.meta.get("category")
        if cat in out:
            out[cat] += 1
    return out


def compute_gate(tasks: list[_MiniTask]) -> GateOutcome:
    """Run the R13c decision tree on the materialised corpus list."""
    projected = projected_eligible(
        tasks, exclude_r11_pending=True, exclude_r17_high=True
    )
    projected_tasks = [t for t in tasks if t.task_id in projected]
    distinct_categories = {
        t.meta.get("category")
        for t in projected_tasks
        if t.meta.get("category")
    }

    rare_corpus = _rare_class_corpus_counts(tasks)
    rare_proj = _rare_class_projected_counts(tasks, projected)

    # Rare-class coverage ratio = sum_proj / sum_corpus across the rare
    # set. We exclude rare classes whose corpus count is 0 (cannot
    # cover something that doesn't exist).
    rare_denom = sum(rare_corpus.values())
    rare_numer = sum(rare_proj.values())
    rare_ratio = (rare_numer / rare_denom) if rare_denom > 0 else 1.0
    rare_ok = rare_ratio >= THRESH_RARE_CLASS_COVERAGE_RATIO

    reasons: list[str] = []
    size = len(projected)
    n_categories = len(distinct_categories)

    if size < THRESH_PROVISIONAL_SIZE:
        reasons.append(
            f"projected_eligible size {size} < {THRESH_PROVISIONAL_SIZE} "
            "(STOP threshold)"
        )
        state = STATE_STOP
    elif n_categories < THRESH_PROVISIONAL_CATEGORIES:
        reasons.append(
            f"distinct categories {n_categories} < "
            f"{THRESH_PROVISIONAL_CATEGORIES} (STOP threshold)"
        )
        state = STATE_STOP
    elif (
        size >= THRESH_PROCEED_SIZE
        and n_categories >= THRESH_PROCEED_CATEGORIES
        and rare_ok
    ):
        state = STATE_PROCEED
    elif (
        size >= THRESH_PROVISIONAL_SIZE
        and n_categories >= THRESH_PROVISIONAL_CATEGORIES
        and not rare_ok
    ):
        reasons.append(
            f"rare-class coverage {rare_ratio:.0%} < "
            f"{THRESH_RARE_CLASS_COVERAGE_RATIO:.0%}"
        )
        state = STATE_PROVISIONAL_MISSING_RARE
    else:
        if size < THRESH_PROCEED_SIZE:
            reasons.append(
                f"projected_eligible {size} < {THRESH_PROCEED_SIZE}"
            )
        if n_categories < THRESH_PROCEED_CATEGORIES:
            reasons.append(
                f"distinct categories {n_categories} < "
                f"{THRESH_PROCEED_CATEGORIES}"
            )
        state = STATE_PROVISIONAL

    return GateOutcome(
        state=state,
        projected_size=size,
        distinct_categories=n_categories,
        rare_class_coverage=rare_proj,
        rare_class_total_in_corpus=rare_corpus,
        rare_class_ratio=rare_ratio,
        reasons=reasons,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--json-out",
        default=str(
            REPO_ROOT / "docs" / "triage_backlog"
            / "2026-05-27-r13c-gate.json"
        ),
    )
    args = p.parse_args(argv)

    tasks = _load_corpus()
    outcome = compute_gate(tasks)

    log.info("=== R13c gate outcome ===")
    log.info("  state:                  %s", outcome.state)
    log.info("  projected_eligible:     %d", outcome.projected_size)
    log.info("  distinct_categories:    %d", outcome.distinct_categories)
    log.info("  rare-class coverage:")
    for cls in sorted(RARE_CLASS_SET):
        n_proj = outcome.rare_class_coverage.get(cls, 0)
        n_total = outcome.rare_class_total_in_corpus.get(cls, 0)
        log.info("    %-15s %d / %d", cls + ":", n_proj, n_total)
    log.info(
        "  rare-class ratio:       %.2f (threshold %.2f)",
        outcome.rare_class_ratio, THRESH_RARE_CLASS_COVERAGE_RATIO,
    )
    for r in outcome.reasons:
        log.info("  reason: %s", r)

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(asdict(outcome), indent=2))
    log.info("json -> %s", args.json_out)

    return _EXIT_CODE[outcome.state]


if __name__ == "__main__":
    sys.exit(main())
