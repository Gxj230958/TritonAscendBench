#!/usr/bin/env python3
"""freeze_source.py — lock upstream commit SHA + create vendored snapshot.

Per v4.13 Phase 1.2 (Source drift mitigation).

Workflow:
1. Read tasks/level{1,2,3}/<task_id>/meta.yaml
2. Read commit_sha + upstream.path
3. Copy upstream snapshot to ascendbench/frozen/snapshots/<task_id>_<sha7>.py
4. Update meta.yaml.upstream.file_sha256 + vendored_snapshot path
5. Commit (provider repo)

Usage:
    python scripts/internal/freeze_upstream_snapshot.py --task L1_12_rms_norm
    python scripts/internal/freeze_upstream_snapshot.py --all
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def freeze_task(task_dir: Path, vendored_root: Path) -> dict:
    meta_path = task_dir / "meta.yaml"
    if not meta_path.exists():
        sys.exit(f"ERROR: meta.yaml missing in {task_dir}")

    meta = yaml.safe_load(meta_path.read_text())
    task_id = meta["task_id"]
    upstream = meta["upstream"]
    upstream_path = upstream["path"]
    commit_sha = upstream["commit_sha"]

    # Reject any angle-bracket placeholder.
    if (
        not isinstance(commit_sha, str)
        or commit_sha.startswith("<")
        or commit_sha.endswith(">")
        or commit_sha in {"unknown", "TBD"}
    ):
        sys.exit(
            f"ERROR: {task_id} commit_sha is a placeholder ({commit_sha!r}); "
            "run extract_kernel first against a real git checkout"
        )

    # Find upstream source (search common locations)
    candidate_paths = [
        REPO_ROOT.parent / "infer" / "vllm" / upstream_path,
        REPO_ROOT.parent / "infer" / "sglang" / upstream_path,
        Path(upstream_path),  # absolute
    ]
    source_file = next((p for p in candidate_paths if p.exists()), None)
    if not source_file:
        sys.exit(
            f"ERROR: cannot locate upstream source for {task_id}; "
            f"tried: {[str(p) for p in candidate_paths]}"
        )

    # Compute file sha256
    file_sha256 = hashlib.sha256(source_file.read_bytes()).hexdigest()

    # Vendored snapshot path
    sha7 = commit_sha[:7]
    vendored_dst = vendored_root / f"{task_id}_{sha7}.py"
    vendored_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, vendored_dst)
    logger.info("Vendored %s → %s", source_file, vendored_dst)

    # Update meta.yaml
    meta["upstream"]["file_sha256"] = file_sha256
    meta["upstream"]["vendored_snapshot"] = str(
        vendored_dst.relative_to(REPO_ROOT)
    )
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
    logger.info("Updated meta.yaml.upstream.file_sha256 = %s", file_sha256)

    return {
        "task_id": task_id,
        "source": str(source_file),
        "vendored": str(vendored_dst),
        "commit_sha": commit_sha,
        "file_sha256": file_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", help="Single task ID")
    parser.add_argument("--all", action="store_true", help="Freeze all tasks")
    parser.add_argument(
        "--vendored-root",
        default="ascendbench/frozen/snapshots",
        help="Vendored snapshot directory (relative to repo root)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    vendored_root = REPO_ROOT / args.vendored_root
    tasks_root = REPO_ROOT / "tasks"

    if args.all:
        results = []
        for level in ["level1", "level2", "level3"]:
            level_dir = tasks_root / level
            if not level_dir.exists():
                continue
            for task_dir in sorted(level_dir.iterdir()):
                if task_dir.is_dir() and not task_dir.name.startswith("_"):
                    try:
                        results.append(freeze_task(task_dir, vendored_root))
                    except SystemExit as e:
                        logger.error("Failed for %s: %s", task_dir.name, e)
        logger.info("Frozen %d tasks", len(results))
    elif args.task:
        task_dir = None
        for level in ["level1", "level2", "level3"]:
            candidate = tasks_root / level / args.task
            if candidate.exists():
                task_dir = candidate
                break
        if not task_dir:
            sys.exit(f"ERROR: task {args.task} not found in tasks/")
        result = freeze_task(task_dir, vendored_root)
        logger.info("Done: %s", result)
    else:
        parser.error("Specify --task or --all")

    return 0


if __name__ == "__main__":
    sys.exit(main())
