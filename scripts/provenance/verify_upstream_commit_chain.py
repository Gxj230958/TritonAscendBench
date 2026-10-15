#!/usr/bin/env python3
"""verify_upstream_commit_chain.py -- (R12).

Asserts that the kernel.py commit SHA (``meta.upstream.commit_sha``)
matches the author-test capture commit SHA
(``provenance.golden.test_capture_sha``). When the two anchors agree
the source ① equivalence path is fully verified (R12 PASSes); when
they disagree but both refs are reachable in their respective
``infer/{repo}/.git`` the result is ``NEEDS_MANUAL`` (a human
reviewer decides whether the drift is acceptable); when either ref
is unreachable the task is routed to source ② / BACKLOG_PENDING_GOLDEN.

Outputs (per task) one verdict line into a Markdown report + a JSON
machine-readable summary. The 4-state classification (``VERIFIED`` /
``NEEDS_MANUAL`` / ``BACKLOG_PENDING_GOLDEN`` / ``EXCLUDED_SNAPSHOT``)
is the canonical R12 vocabulary downstream of the schema (which
allows ``oracle_kernel_equivalence: NEEDS_MANUAL`` so we can write
this through the apply script later).

This is pure git CLI + YAML I/O -- no GPU, no torch import. Safe to
run on the dev box; no provider privileges required.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TASKS_ROOT = REPO_ROOT / "tasks"
INFER_ROOT = REPO_ROOT.parent / "infer"

VLLM_DIR = INFER_ROOT / "vllm"
SGLANG_DIR = INFER_ROOT / "sglang"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("verify_upstream_commit_chain")


# Closed verdict set.
VERDICT_VERIFIED = "VERIFIED"
VERDICT_NEEDS_MANUAL = "NEEDS_MANUAL"
VERDICT_BACKLOG = "BACKLOG_PENDING_GOLDEN"
VERDICT_EXCLUDED_SNAPSHOT = "EXCLUDED_SNAPSHOT"
VERDICT_NO_TEST_SHA = "NO_TEST_CAPTURE_SHA"

ALL_VERDICTS = frozenset({
    VERDICT_VERIFIED,
    VERDICT_NEEDS_MANUAL,
    VERDICT_BACKLOG,
    VERDICT_EXCLUDED_SNAPSHOT,
    VERDICT_NO_TEST_SHA,
})


@dataclass
class Verdict:
    task_id: str
    verdict: str
    kernel_commit_sha: str | None = None
    test_capture_sha: str | None = None
    kernel_sha_reachable: bool = False
    test_sha_reachable: bool = False
    repo_root: str | None = None
    reason: str = ""


def _git_object_reachable(git_repo: Path, sha: str) -> bool:
    """``git -C <repo> cat-file -e <sha>`` -- True iff the object exists.

    Returns ``False`` on missing object, missing repo, missing
    ``git`` binary, or any subprocess error -- conservatively fail-
    closed (cannot prove reachability -> not reachable).
    """
    if not git_repo.is_dir():
        return False
    if shutil.which("git") is None:
        return False
    try:
        proc = subprocess.run(
            ["git", "-C", str(git_repo), "cat-file", "-e", sha],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("git cat-file failed for %s in %s: %s", sha, git_repo, exc)
        return False
    return proc.returncode == 0


def _resolve_repo_root(repo_field: str | None) -> Path | None:
    """Map ``meta.repo`` -> infer/<dir> root.

    ``meta.repo`` values seen in the corpus: ``vLLM`` / ``SGLang`` /
    ``FBGEMM`` / ``Bytedance_internal`` / ``PrimusTurbo``. Only the
    vLLM and SGLang families currently have a vendored ``.git`` under
    ``infer/`` on the dev box; others are snapshot-only by design.
    """
    if not repo_field:
        return None
    rf = repo_field.lower()
    if rf == "vllm":
        return VLLM_DIR
    if rf == "sglang":
        return SGLANG_DIR
    return None


def _is_snapshot_sentinel(sha: str | None) -> bool:
    """Recognise the ``snapshot-no-git`` sentinel used for tasks whose
    upstream is a frozen snapshot (no associated git history)."""
    if not isinstance(sha, str):
        return True  # missing == treat like snapshot for routing purposes
    return sha in {"snapshot-no-git", ""} or sha.startswith("snapshot")


def _is_hex(sha: str | None, length: int = 40) -> bool:
    if not isinstance(sha, str):
        return False
    if len(sha) < 7:  # accept short hashes
        return False
    return all(c in "0123456789abcdef" for c in sha.lower())


def verify_task(task_dir: Path) -> Verdict:
    """Compute the R12 verdict for one task directory."""
    task_id = task_dir.name
    meta_path = task_dir / "meta.yaml"
    if not meta_path.is_file():
        return Verdict(
            task_id=task_id, verdict=VERDICT_BACKLOG,
            reason="meta.yaml missing",
        )
    try:
        meta = yaml.safe_load(meta_path.read_text()) or {}
    except yaml.YAMLError as exc:
        return Verdict(
            task_id=task_id, verdict=VERDICT_BACKLOG,
            reason=f"meta.yaml YAML error: {exc}",
        )

    upstream = meta.get("upstream") or {}
    kernel_sha = upstream.get("commit_sha")
    repo_field = meta.get("repo") or upstream.get("repo")

    if _is_snapshot_sentinel(kernel_sha):
        return Verdict(
            task_id=task_id, verdict=VERDICT_EXCLUDED_SNAPSHOT,
            kernel_commit_sha=kernel_sha,
            reason="snapshot-no-git: R12 N/A (force source ② path)",
        )

    golden = (meta.get("provenance") or {}).get("golden") or {}
    test_sha = golden.get("test_capture_sha")

    repo_root = _resolve_repo_root(repo_field)
    repo_root_str = str(repo_root) if repo_root else None

    if not _is_hex(test_sha):
        # Source ① test capture hasn't been recorded yet (the author-test
        # extractor was deprecated, so this is the expected state for the
        # current corpus). Report cleanly + classify so callers can
        # surface "R12 not yet evaluable" cleanly.
        kernel_ok = bool(repo_root) and _is_hex(kernel_sha) and (
            _git_object_reachable(repo_root, kernel_sha)
        )
        return Verdict(
            task_id=task_id, verdict=VERDICT_NO_TEST_SHA,
            kernel_commit_sha=kernel_sha,
            kernel_sha_reachable=kernel_ok,
            repo_root=repo_root_str,
            reason=(
                "provenance.golden.test_capture_sha not yet recorded "
                "(M1.1 author-test extractor deprecated in V9; R10 "
                "GPU equivalence path covers source ① verification)"
            ),
        )

    if not repo_root:
        return Verdict(
            task_id=task_id, verdict=VERDICT_BACKLOG,
            kernel_commit_sha=kernel_sha,
            test_capture_sha=test_sha,
            reason=(
                f"meta.repo={repo_field!r} has no vendored .git under "
                f"infer/; cannot verify either anchor"
            ),
        )

    kernel_reachable = _git_object_reachable(repo_root, kernel_sha)
    test_reachable = _git_object_reachable(repo_root, test_sha)

    if not (kernel_reachable and test_reachable):
        return Verdict(
            task_id=task_id, verdict=VERDICT_BACKLOG,
            kernel_commit_sha=kernel_sha,
            test_capture_sha=test_sha,
            kernel_sha_reachable=kernel_reachable,
            test_sha_reachable=test_reachable,
            repo_root=repo_root_str,
            reason=(
                "one or both anchors unreachable in "
                f"{repo_root}; routing to source ②"
            ),
        )

    if kernel_sha == test_sha:
        return Verdict(
            task_id=task_id, verdict=VERDICT_VERIFIED,
            kernel_commit_sha=kernel_sha,
            test_capture_sha=test_sha,
            kernel_sha_reachable=True,
            test_sha_reachable=True,
            repo_root=repo_root_str,
        )

    return Verdict(
        task_id=task_id, verdict=VERDICT_NEEDS_MANUAL,
        kernel_commit_sha=kernel_sha,
        test_capture_sha=test_sha,
        kernel_sha_reachable=True,
        test_sha_reachable=True,
        repo_root=repo_root_str,
        reason=(
            f"kernel sha {kernel_sha[:8]} != test sha {test_sha[:8]} "
            "(both reachable -- possible acceptable drift, human review)"
        ),
    )


def _iter_task_dirs(filter_ids: set[str] | None = None) -> Iterable[Path]:
    for level in ("level1", "level2", "level3"):
        ld = TASKS_ROOT / level
        if not ld.is_dir():
            continue
        for td in sorted(ld.iterdir()):
            if td.is_dir() and not td.name.startswith("_"):
                if filter_ids is None or td.name in filter_ids:
                    yield td


def _emit_report(verdicts: list[Verdict], report_path: Path) -> None:
    from collections import Counter

    tally = Counter(v.verdict for v in verdicts)
    lines: list[str] = [
        "---",
        "tool: scripts/provenance/verify_upstream_commit_chain.py",
        "date: 2026-05-27",
        "purpose: V9 plan M2.0.5 — R12 SHA-chain integrity verdict",
        "---",
        "",
        "# R12 SHA-chain Integrity Report",
        "",
        f"Tasks processed: **{len(verdicts)}**",
        "",
        "## Verdict tally",
        "",
        "| verdict | count |",
        "|---|---|",
    ]
    for v, n in tally.most_common():
        lines.append(f"| `{v}` | {n} |")
    lines.append("")

    for v_name in (
        VERDICT_VERIFIED,
        VERDICT_NEEDS_MANUAL,
        VERDICT_BACKLOG,
        VERDICT_EXCLUDED_SNAPSHOT,
        VERDICT_NO_TEST_SHA,
    ):
        subset = [v for v in verdicts if v.verdict == v_name]
        if not subset:
            continue
        lines.append(f"## {v_name} ({len(subset)})")
        lines.append("")
        for v in subset:
            parts = [f"- `{v.task_id}`"]
            if v.kernel_commit_sha:
                parts.append(f"kernel={v.kernel_commit_sha[:8]}")
            if v.test_capture_sha:
                parts.append(f"test={v.test_capture_sha[:8]}")
            if v.reason:
                parts.append(f"reason: {v.reason}")
            lines.append("  ".join(parts))
        lines.append("")

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
            REPO_ROOT / "build" / "upstream-commit-chain.md"
        ),
        help="markdown report path",
    )
    p.add_argument(
        "--json-out",
        default="/tmp/r12_sha_chain.json",
        help="json verdict output",
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

    verdicts: list[Verdict] = []
    for td in _iter_task_dirs(filter_ids):
        v = verify_task(td)
        verdicts.append(v)
        log.info("  %s: %s%s", v.task_id, v.verdict,
                 f"  ({v.reason[:120]})" if v.reason else "")

    from collections import Counter
    tally = Counter(v.verdict for v in verdicts)
    log.info("--- tally ---")
    for vname, n in tally.most_common():
        log.info("  %s: %d", vname, n)

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps([asdict(v) for v in verdicts], indent=2)
    )
    _emit_report(verdicts, Path(args.report))
    log.info("json -> %s", args.json_out)
    log.info("report -> %s", args.report)

    # Exit code: 0 if no NEEDS_MANUAL nor BACKLOG entries (full pass);
    # else 1 (something needs human triage). EXCLUDED_SNAPSHOT and
    # NO_TEST_CAPTURE_SHA are NOT treated as failure (expected states).
    needs_attention = sum(
        1 for v in verdicts
        if v.verdict in {VERDICT_NEEDS_MANUAL, VERDICT_BACKLOG}
    )
    return 0 if needs_attention == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
