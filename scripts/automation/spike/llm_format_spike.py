"""LLM output format A/B spike (M0 Unit 0.4).

Validates the KTD-1 decision (file-blocks output format) by running a
small set of real shapes-incomplete tasks through the LLM twice each --
once asking for file-blocks output, once asking for unified diff -- and
comparing three success-rate metrics:

    1. Parse success      -- can the response be lexed without error?
    2. Apply success      -- can the parsed output be written to the
                             task directory without conflict?
    3. Semantic correctness -- after apply, does ``resolve_upstream_entrypoints``
                             report PASS on the task?

Output: a Markdown file with one statistic block per format plus a
decision record (keep KTD-1 / escalate to user / re-do prompt template).

Status: SKELETON. The two probe functions (``call_llm_file_blocks`` /
``call_llm_unified_diff``) are NOT implemented; each raises
``NotImplementedError``. Running this script with ``--llm-endpoint``
exits non-zero until the probes are filled in. Running without
``--llm-endpoint`` also exits non-zero -- the M0 plan forbids fabricated
statistics, so a dry-run path that emits a markdown with ``parse_ok =
apply_ok = semantic_ok = False`` placeholders is NOT allowed.

See plan §Unit 0.4 for required probe semantics.

Usage::

    python3 scripts/automation/spike/llm_format_spike.py \\
        --tasks T01,T02,T03,T04,T05 \\
        --llm-endpoint http://llm:8000/v1/chat/completions \\
        --repeats 3 \\
        --out docs/release/internal-llm-format-spike-results.md

This file is INTERNAL (export-ignored). Spike runs are one-shot but the
script remains available for re-running on a new LLM endpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

# --- KTD-11 sys.path bootstrap (CLI entry-point) ---------------------------
# spike/ lives under scripts/automation/spike/, so parents[3] == <repo_root>.
# The bootstrap MUST run before any ``from automation...`` import below,
# hence the ``E402`` noqa on the subsequent stdlib imports.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
# Subsequent imports (e.g. ``from automation.lib import ...``) resolve once
# the bootstrap above runs. This spike calls the LLM endpoint directly and
# does not depend on lib/ yet.
# ---------------------------------------------------------------------------

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Any  # noqa: E402


@dataclass
class FormatStat:
    """One (format, task, repeat) probe -- the aggregator collapses
    these into per-format statistic blocks."""

    format: str  # "file-blocks" or "unified-diff"
    task_id: str
    repeat_idx: int
    parse_ok: bool = False
    apply_ok: bool = False
    semantic_ok: bool = False
    latency_ms: float = 0.0
    raw_response: str = ""
    notes: str = ""


@dataclass
class FormatSummary:
    """Aggregated metrics for a single format across all tasks / repeats."""

    format: str
    samples: int = 0
    parse_success_rate: float = 0.0
    apply_success_rate: float = 0.0
    semantic_success_rate: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    anomalies: list[dict[str, Any]] = field(default_factory=list)

    def to_markdown_block(self) -> str:
        lines = [
            f"### Format: `{self.format}`",
            "",
            "| Metric                  | Value          |",
            "|-------------------------|----------------|",
            f"| samples                 | {self.samples} |",
            f"| parse_success_rate      | {self.parse_success_rate:.2%} |",
            f"| apply_success_rate      | {self.apply_success_rate:.2%} |",
            f"| semantic_success_rate   | {self.semantic_success_rate:.2%} |",
            f"| p50_latency_ms          | {self.p50_latency_ms:.1f} |",
            f"| p95_latency_ms          | {self.p95_latency_ms:.1f} |",
            f"| anomaly_count           | {len(self.anomalies)} |",
        ]
        if self.anomalies:
            lines.append("")
            lines.append("Anomalies (first 5):")
            lines.append("```json")
            lines.append(json.dumps(self.anomalies[:5], indent=2, sort_keys=True))
            lines.append("```")
        lines.append("")
        return "\n".join(lines)


def _percentile(samples_ms: list[float], p: float) -> float:
    if not samples_ms:
        return 0.0
    s = sorted(samples_ms)
    if len(s) == 1:
        return s[0]
    idx = (len(s) - 1) * (p / 100.0)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] + (s[hi] - s[lo]) * frac


# --- Probe protocol --------------------------------------------------------


_PROBE_NOT_IMPLEMENTED = (
    "Probe {name} is not implemented yet -- see plan §Unit 0.4 for the "
    "required LLM-call semantics. This script is a skeleton; running it "
    "MUST fail loud until the probes are filled in."
)


def call_llm_file_blocks(task_id: str, endpoint: str, repeat_idx: int) -> FormatStat:
    """Ask the LLM to emit ``### FILE: <path>\\n<content>\\n### END\\n`` blocks.

    NOT IMPLEMENTED. Raises ``NotImplementedError`` so the main entry
    point surfaces the gap rather than emitting a fake-zero markdown.
    """
    raise NotImplementedError(_PROBE_NOT_IMPLEMENTED.format(name="file-blocks"))


def call_llm_unified_diff(task_id: str, endpoint: str, repeat_idx: int) -> FormatStat:
    """Ask the LLM to emit a ``### PATCH ... ### END`` unified-diff block.

    NOT IMPLEMENTED. See ``call_llm_file_blocks``.
    """
    raise NotImplementedError(_PROBE_NOT_IMPLEMENTED.format(name="unified-diff"))


def run_spike(tasks: list[str], endpoint: str, repeats: int) -> list[FormatStat]:
    """Run N task x 2 format x R repeat = 2*N*R LLM calls; collect FormatStat
    records. Aggregation lives in ``summarize`` below."""
    stats: list[FormatStat] = []
    for task_id in tasks:
        for repeat_idx in range(repeats):
            stats.append(call_llm_file_blocks(task_id, endpoint, repeat_idx))
            stats.append(call_llm_unified_diff(task_id, endpoint, repeat_idx))
    return stats


def summarize(stats: list[FormatStat]) -> dict[str, FormatSummary]:
    """Group by format and compute per-metric success rates plus
    latency percentiles."""
    summaries: dict[str, FormatSummary] = {}
    by_fmt: dict[str, list[FormatStat]] = {}
    for s in stats:
        by_fmt.setdefault(s.format, []).append(s)
    for fmt, group in by_fmt.items():
        n = len(group)
        if n == 0:
            continue
        latencies = [s.latency_ms for s in group if s.latency_ms > 0]
        anomalies = [
            {"task": s.task_id, "repeat": s.repeat_idx, "notes": s.notes}
            for s in group
            if not s.parse_ok or s.notes.startswith("not_run")
        ]
        summaries[fmt] = FormatSummary(
            format=fmt,
            samples=n,
            parse_success_rate=sum(1 for s in group if s.parse_ok) / n,
            apply_success_rate=sum(1 for s in group if s.apply_ok) / n,
            semantic_success_rate=sum(1 for s in group if s.semantic_ok) / n,
            p50_latency_ms=_percentile(latencies, 50),
            p95_latency_ms=_percentile(latencies, 95),
            anomalies=anomalies,
        )
    return summaries


def decide(summaries: dict[str, FormatSummary]) -> str:
    """Decision record string. Three branches per the M0 plan:

      * file-blocks >= unified-diff on every metric  -> keep KTD-1
      * unified-diff clearly better (especially semantic) -> escalate user
      * both formats < 50% semantic                       -> re-do prompt
    """
    fb = summaries.get("file-blocks")
    ud = summaries.get("unified-diff")
    if not fb or not ud:
        return "INDETERMINATE -- spike did not produce both formats; re-run."
    if fb.samples == 0 or ud.samples == 0:
        return "INDETERMINATE -- one format had 0 samples; re-run."

    fb_better = (
        fb.parse_success_rate >= ud.parse_success_rate
        and fb.apply_success_rate >= ud.apply_success_rate
        and fb.semantic_success_rate >= ud.semantic_success_rate
    )
    if fb.semantic_success_rate < 0.5 and ud.semantic_success_rate < 0.5:
        return (
            "RE-DO-PROMPT: both formats < 50% semantic correctness; "
            "the prompt template needs work before M0 can ship."
        )
    if fb_better:
        return (
            "KEEP-KTD-1: file-blocks >= unified-diff on parse / apply / "
            "semantic. KTD-1 stays as-is."
        )
    return (
        "ESCALATE-USER: unified-diff outperforms file-blocks on at "
        "least one metric. Surface to user before flipping KTD-1."
    )


def render_results_markdown(
    summaries: dict[str, FormatSummary],
    tasks: list[str],
    endpoint: str,
    repeats: int,
    decision: str,
) -> str:
    header = [
        "# INTERNAL: LLM output format A/B spike results",
        "",
        "> M0 Unit 0.4 -- empirical KTD-1 (file-blocks output) validation",
        "> against the unified-diff alternative.",
        "",
        f"- tasks: `{', '.join(tasks)}`",
        f"- endpoint: `{endpoint}`",
        f"- repeats per (task x format): {repeats}",
        f"- timestamp (epoch): {int(time.time())}",
        "",
        "## Per-format metrics",
        "",
    ]
    blocks = [s.to_markdown_block() for s in summaries.values()]
    footer = [
        "## Decision",
        "",
        f"**{decision}**",
        "",
    ]
    return "\n".join(header + blocks + footer)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM output format A/B spike (M0 Unit 0.4)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        required=False,
        default="",
        help="Comma-separated task IDs (e.g. T01,T02,...). 5-10 tasks "
        "covering L1 / L2 / L3 each is the M0 minimum.",
    )
    parser.add_argument(
        "--llm-endpoint",
        type=str,
        default="",
        help="LLM endpoint URL (e.g. http://llm:8000/v1/chat/completions). "
        "REQUIRED -- omitting it exits non-zero (the spike has no offline "
        "fallback; the M0 plan forbids fabricated statistics, so a fake-zero "
        "markdown would be worse than no markdown).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats per (task, format) pair (default 3).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output Markdown path. Defaults to "
        "docs/release/internal-llm-format-spike-results.md under the repo root.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out = args.out or (
        _REPO_ROOT / "docs" / "release" / "internal-llm-format-spike-results.md"
    )

    # Fail loud #1: the spike by definition needs an LLM endpoint. The M0
    # plan forbids fabricated statistics, so a dry-run path that emits a
    # markdown with all-zero success rates is NOT allowed.
    if not args.llm_endpoint:
        print(
            "ERROR: llm_format_spike requires --llm-endpoint. This script is a "
            "skeleton -- the file-blocks / unified-diff probes are not yet "
            "implemented. See plan §Unit 0.4 for required probe semantics.",
            file=sys.stderr,
        )
        return 2
    if not tasks:
        print(
            "ERROR: llm_format_spike requires --tasks (comma-separated task IDs).",
            file=sys.stderr,
        )
        return 2

    # Fail loud #2: probes are unimplemented. Surface the exact probe
    # that failed; do NOT emit a fake-zero markdown.
    try:
        stats = run_spike(tasks, args.llm_endpoint, args.repeats)
    except NotImplementedError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    summaries = summarize(stats)
    decision = decide(summaries)
    md = render_results_markdown(
        summaries, tasks, args.llm_endpoint, args.repeats, decision
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")

    print(f"spike complete -- results written to {out}")
    print(f"decision: {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
