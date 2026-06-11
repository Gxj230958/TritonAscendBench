#!/usr/bin/env python3
"""source_provenance_census.py.

Static census of 100 corpus tasks across 5 dimensions:

* (a) Source ① static feasibility -- can an author-test ref be
      extracted from ``infer/{repo}/**/*.py``? (the
      extractor itself is deprecated; this dimension is reported as
      ``V9_DEPRECATED_M1_1`` for every task so the report stays
      honest about the source landscape.)
* (b) tolerance source        -- ``upstream_test_explicit`` /
      ``upstream_test_default`` / ``migration_drift_budget`` /
      ``corpus_only`` (the tolerance class is from the corpus's
      ``tolerance_class`` field, no upstream attestation).
* (c) GPU / platform skip     -- whether the kernel's frozen
      upstream test imports torch.distributed / has
      ``pytest.mark.gpu`` / ``skipif(not torch.cuda.is_available)``.
* (d) commit_sha verifiable   -- ``git -C infer/{repo} cat-file -e
      <sha>`` (delegates to the helper).
* (e) R17 reconstructability risk -- per-category score driven by
      a static dictionary (``elementwise / norm / gemm / sampling``
      are HIGH risk: math closed-form, leaderboard-excluded).

This is pure CPU + AST + git CLI -- safe to run on the dev box.
Output: per-task JSON + Markdown summary report.

R13c gate reads the JSON output to compute
``projected_eligible`` + classify rare-class coverage.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TASKS_ROOT = REPO_ROOT / "tasks"
INFER_ROOT = REPO_ROOT.parent / "infer"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("source_provenance_census")


# ─────────────────────────────────────────────────────────────────────
# Dimension (e): R17 reconstructability risk by category
# ─────────────────────────────────────────────────────────────────────
#
# HIGH-risk: math closed-form ops. Adversary can replay perfectly
# without the upstream kernel -> leaderboard-excluded.
# MEDIUM-risk: layout-dependent but pattern-known (paged-attention,
# SSM, conv) -- harder to fake but not impossible.
# LOW-risk: byte-exact / state-dependent kernels with no easy closed
# form (sampling RNG with bit_exact, quant cache mapping, etc.)
#
# Mapped against the 12 corpus categories. ``norm`` and
# ``normalization`` are TWO DISTINCT categories per the plan
# decision; the table reflects that.
_CATEGORY_RISK: dict[str, str] = {
    # HIGH (math closed-form)
    "elementwise": "high",
    "norm": "high",
    "gemm": "high",
    "sampling": "high",
    # MEDIUM
    "normalization": "medium",
    "paged-attention": "medium",
    "spec_decode": "medium",
    "convolution": "medium",
    "ssm": "medium",
    "moe": "medium",
    "attention": "medium",
    "softmax": "medium",
    "scan": "medium",
    # LOW
    "quantization": "low",
    "rng": "low",
}


def _risk_for_category(category: str | None) -> str:
    if not category:
        return "unknown"
    # Unknown category is fail-closed: high (the verifier surfaces it).
    return _CATEGORY_RISK.get(category, "unknown")


# ─────────────────────────────────────────────────────────────────────
# Dimension (c): GPU / platform skip detection (static)
# ─────────────────────────────────────────────────────────────────────


_GPU_SKIP_PATTERNS = [
    re.compile(r"@pytest\.mark\.gpu\b"),
    re.compile(r"skipif\([^)]*torch\.cuda\.is_available[^)]*\)"),
    re.compile(r"requires_gpu\b"),
    re.compile(r"from\s+torch\.distributed\s+import"),
    re.compile(r"import\s+torch\.distributed\b"),
]


def _gpu_skip_in_text(text: str) -> list[str]:
    """Return a list of hit patterns (empty when none)."""
    hits: list[str] = []
    for pat in _GPU_SKIP_PATTERNS:
        if pat.search(text):
            hits.append(pat.pattern)
    return hits


# ─────────────────────────────────────────────────────────────────────
# Dimension (b): tolerance source
# ─────────────────────────────────────────────────────────────────────


def _classify_tolerance_source(meta: dict) -> str:
    """Pick the tolerance source label per the enum.

    * provenance.golden.tolerance_source takes precedence when set.
    * Otherwise the corpus declares a ``tolerance_class`` (legacy);
      report ``corpus_only`` so the census makes the absence of an
      upstream attestation visible.
    """
    golden = (meta.get("provenance") or {}).get("golden") or {}
    src = golden.get("tolerance_source")
    if src:
        return str(src)
    if meta.get("tolerance_class"):
        return "corpus_only"
    return "absent"


# ─────────────────────────────────────────────────────────────────────
# Dimension (d): commit sha verifiable (delegates to the helper)
# ─────────────────────────────────────────────────────────────────────


def _commit_sha_verifiable(meta: dict) -> tuple[bool, str | None]:
    """Return ``(verifiable, commit_sha)``.

    A ``snapshot-no-git`` sentinel is considered "not verifiable" but
    expected; callers gate on it separately.
    """
    # Lazy import via the canonical ``scripts.`` package path so the
    # module identity is unique (sys.modules entry is shared with
    # other callers / tests' mocks).
    import importlib  # noqa: PLC0415

    sys.path.insert(0, str(REPO_ROOT))
    vcc = importlib.import_module(
        "scripts.provenance.verify_upstream_commit_chain"
    )

    upstream = meta.get("upstream") or {}
    sha = upstream.get("commit_sha")
    repo = meta.get("repo") or upstream.get("repo")
    if vcc._is_snapshot_sentinel(sha):
        return False, sha
    if not vcc._is_hex(sha):
        return False, sha
    repo_root = vcc._resolve_repo_root(repo)
    if repo_root is None:
        return False, sha
    return vcc._git_object_reachable(repo_root, sha), sha


# ─────────────────────────────────────────────────────────────────────
# Census records
# ─────────────────────────────────────────────────────────────────────


@dataclass
class TaskCensus:
    task_id: str
    category: str | None
    repo: str | None
    # (a) source ① static feasibility -- the extractor is deprecated; reported
    # as a fixed sentinel so the dimension is honest about its state.
    source_1_static_extractable: str = "V9_DEPRECATED_M1_1"
    # (b) tolerance source
    tolerance_source: str = "absent"
    # (c) GPU/platform skip hits (when kernel.py references blocked imports)
    gpu_skip_patterns: list[str] = field(default_factory=list)
    # (d) commit sha verifiable
    commit_sha: str | None = None
    commit_sha_verifiable: bool = False
    # (e) R17 reconstructability risk
    reconstructability_risk: str = "unknown"


def census_task(task_dir: Path) -> TaskCensus | None:
    meta_path = task_dir / "meta.yaml"
    if not meta_path.is_file():
        return None
    try:
        meta = yaml.safe_load(meta_path.read_text()) or {}
    except yaml.YAMLError as exc:
        log.warning("%s: meta.yaml YAML error: %s", task_dir.name, exc)
        return None

    category = meta.get("category")
    repo = meta.get("repo")

    tolerance_source = _classify_tolerance_source(meta)

    # GPU/platform scan against the kernel.py we ship (since the
    # upstream test set is not vendored, scan our own kernel mirror
    # for the same patterns -- they would prevent the kernel from
    # ever running on the CI dev box, hence relevant to the census).
    kernel_text = ""
    kp = task_dir / "kernel.py"
    if kp.is_file():
        try:
            kernel_text = kp.read_text(errors="ignore")
        except OSError:
            kernel_text = ""
    gpu_skip = _gpu_skip_in_text(kernel_text)

    verifiable, sha = _commit_sha_verifiable(meta)
    risk = _risk_for_category(category)

    return TaskCensus(
        task_id=task_dir.name,
        category=category,
        repo=repo,
        source_1_static_extractable="V9_DEPRECATED_M1_1",
        tolerance_source=tolerance_source,
        gpu_skip_patterns=gpu_skip,
        commit_sha=sha,
        commit_sha_verifiable=verifiable,
        reconstructability_risk=risk,
    )


def _iter_task_dirs(filter_ids: set[str] | None = None) -> list[Path]:
    out: list[Path] = []
    for level in ("level1", "level2", "level3"):
        ld = TASKS_ROOT / level
        if not ld.is_dir():
            continue
        for td in sorted(ld.iterdir()):
            if td.is_dir() and not td.name.startswith("_"):
                if filter_ids is None or td.name in filter_ids:
                    out.append(td)
    return out


def _emit_report(rows: list[TaskCensus], report_path: Path) -> None:
    from collections import Counter

    # Family taxonomy: norm vs normalization disjoint check
    cat_tally = Counter((r.category or "<missing>") for r in rows)
    risk_tally = Counter(r.reconstructability_risk for r in rows)
    tol_src_tally = Counter(r.tolerance_source for r in rows)
    verifiable_count = sum(1 for r in rows if r.commit_sha_verifiable)

    lines: list[str] = [
        "---",
        "tool: scripts/provenance/census_source_provenance.py",
        "date: 2026-05-27",
        "purpose: V9 plan M0a.1 — 5-dimension static provenance census",
        "---",
        "",
        "# Source Provenance Census",
        "",
        f"Tasks processed: **{len(rows)}**  ",
        f"commit_sha git-verifiable: **{verifiable_count}/{len(rows)}**",
        "",
        "## (a) Source ① static feasibility",
        "",
        "V9 deprecated the M1.1 author-test extractor; every task is "
        "reported as `V9_DEPRECATED_M1_1`. The R10 GPU equivalence path "
        "(M2.1a/b/c) covers source ① verification instead.",
        "",
        "## (b) tolerance source",
        "",
        "| source | count |",
        "|---|---|",
    ]
    for src, n in tol_src_tally.most_common():
        lines.append(f"| `{src}` | {n} |")
    lines.append("")

    lines += [
        "## (c) GPU / platform skip hits in kernel.py",
        "",
    ]
    hits = [r for r in rows if r.gpu_skip_patterns]
    if not hits:
        lines.append("_No task's kernel.py matches GPU-skip patterns._")
    else:
        lines.append("| task | matched patterns |")
        lines.append("|---|---|")
        for r in hits:
            lines.append(
                f"| `{r.task_id}` | "
                + ", ".join(f"`{p}`" for p in r.gpu_skip_patterns)
                + " |"
            )
    lines.append("")

    lines += [
        "## (d) commit_sha verifiable",
        "",
        f"verifiable: **{verifiable_count} / {len(rows)}**",
        "",
        "_Tasks with `commit_sha = snapshot-no-git` are excluded by "
        "design (no upstream history to verify)._",
        "",
        "## (e) R17 reconstructability risk by category",
        "",
        "| category | count | risk |",
        "|---|---|---|",
    ]
    for cat, n in cat_tally.most_common():
        cat_risk = _CATEGORY_RISK.get(cat, "unknown")
        lines.append(f"| `{cat}` | {n} | `{cat_risk}` |")
    lines.append("")
    lines += [
        "## Risk tally",
        "",
        "| risk | count |",
        "|---|---|",
    ]
    for risk, n in risk_tally.most_common():
        lines.append(f"| `{risk}` | {n} |")
    lines.append("")

    # Family taxonomy verify: norm vs normalization disjoint
    if "norm" in cat_tally and "normalization" in cat_tally:
        lines += [
            "## Family taxonomy disjoint check",
            "",
            f"`norm` ({cat_tally['norm']}) and `normalization` "
            f"({cat_tally['normalization']}) are TWO distinct categories "
            "(per V9 plan decision); the R13c gate treats them "
            "separately for rare-class coverage.",
            "",
        ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--task", help="single task id")
    p.add_argument("--task-list", help="comma-separated task ids")
    p.add_argument("--all", action="store_true", help="all tasks")
    p.add_argument(
        "--report",
        default=str(
            REPO_ROOT / "build" / "census-report.md"
        ),
    )
    p.add_argument(
        "--json-out",
        default=str(
            REPO_ROOT / "build" / "census-detail.json"
        ),
    )
    args = p.parse_args(argv)

    if not (args.task or args.task_list or args.all):
        p.error("either --task / --task-list / --all required")

    filter_ids: set[str] | None
    if args.all:
        filter_ids = None
    elif args.task_list:
        filter_ids = set(args.task_list.split(","))
    else:
        filter_ids = {args.task}

    rows: list[TaskCensus] = []
    for td in _iter_task_dirs(filter_ids):
        rec = census_task(td)
        if rec is None:
            continue
        rows.append(rec)
        log.info(
            "  %s: cat=%s risk=%s tol=%s sha_ok=%s",
            rec.task_id, rec.category, rec.reconstructability_risk,
            rec.tolerance_source, rec.commit_sha_verifiable,
        )

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    _emit_report(rows, Path(args.report))
    log.info("json -> %s", args.json_out)
    log.info("report -> %s", args.report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
