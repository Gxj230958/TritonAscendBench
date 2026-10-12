"""Bundle reproducible scalar evidence and frozen sources, excluding caches/tensors."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import shutil


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(data)
    return value.hexdigest()


def merge_attempts(row, supplement):
    row = dict(row)
    attempts = [{**attempt, "evidence_root": row["evidence_root"]}
                for attempt in row["attempts"]]
    if supplement:
        if supplement["required_shapes"] != row["required_shapes"]:
            raise RuntimeError("supplement shape coverage differs from primary inventory")
        attempts.extend({**attempt, "evidence_root": "supplement"}
                        for attempt in supplement["attempts"])
    row["attempts"] = attempts
    row["complete"] = any(attempt["complete"] for attempt in attempts)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--supplement", type=Path)
    parser.add_argument("--extra", type=Path, action="append", default=[])
    args = parser.parse_args()
    for root in (args.suite, args.replay, *([args.supplement] if args.supplement else [])):
        subprocess.run([sys.executable, str(Path(__file__).with_name("streaming_summary.py")),
                        str(root)], check=True)
    suite = json.loads((args.suite / "summary.json").read_text())
    replay = json.loads((args.replay / "summary.json").read_text())
    replacements = {row["task"]: row for row in replay["records"]}
    supplements = ({row["task"]: row for row in json.loads(
        (args.supplement / "summary.json").read_text())["records"]}
        if args.supplement else {})
    rows = []
    categories = Counter()
    for original in suite["records"]:
        row = dict(replacements.get(original["task"], original))
        row["evidence_root"] = "replay" if original["task"] in replacements else "suite"
        row = merge_attempts(row, supplements.get(original["task"]))
        attempts = row["attempts"]
        chosen = next((a for a in attempts if a["category"] == "complete_verified"),
                      attempts[-1] if attempts else None)
        row["final_category"] = chosen["category"] if chosen else "not_executed"
        row["verified_pass"] = any(a["category"] == "complete_verified" for a in attempts)
        row["observed_device_shapes"] = sorted({sid for a in attempts
                                               for phase in ("correctness", "timing")
                                               for sid in a.get(f"observed_{phase}_shapes", [])})
        categories[row["final_category"]] += 1
        rows.append(row)
    summary = {"tasks": len(rows), "required_shapes": suite["required_shapes"],
               "attempted_tasks": sum(bool(r["attempts"]) for r in rows),
               "complete_device_tasks": sum(r["complete"] for r in rows),
               "verified_pass_tasks": sum(r["verified_pass"] for r in rows),
               "complete_device_shapes": sum(r["required_shapes"] for r in rows if r["complete"]),
               "observed_device_shapes": sum(len(r["observed_device_shapes"]) for r in rows),
               "categories": dict(categories), "records": rows}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    table = ["# Full Task Results", "", "The first seven tasks use frozen-code replay evidence.", "",
             "| Task | Required Shapes | Observed Device Shapes | Full Device Coverage | Final Category | Attempts |",
             "|---|---:|---:|---|---|---:|"]
    for row in rows:
        table.append(f'| {row["task"]} | {row["required_shapes"]} | '
                     f'{len(row["observed_device_shapes"])} | '
                     f'{"complete" if row["complete"] else "incomplete"} | '
                     f'{row["final_category"]} | {len(row["attempts"])} |')
    (args.out / "task-results.md").write_text("\n".join(table) + "\n")
    roots = [("suite", args.suite), ("replay", args.replay)]
    if args.supplement:
        roots.append(("supplement", args.supplement))
    roots.extend((p.name.removeprefix("tab-"), p) for p in args.extra)
    files = []
    for label, root in roots:
        paths = sorted(root.rglob("*")) if root.is_dir() else [root]
        for path in paths:
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root) if root.is_dir() else Path(path.name)
            if any(part in {"home", "scratch", "cache", "prof", "__pycache__"}
                   for part in relative.parts):
                continue
            if path.suffix not in {".json", ".jsonl", ".log", ".txt", ".xml", ".py"}:
                continue
            files.append((f"{label}/{relative}", path))
    # Preserve reference metadata; golden tensor bytes remain externally owned.
    inventory = json.loads((args.suite / "inventory.json").read_text())
    for entry in inventory:
        task = Path(entry["task_dir"])
        for path in [task / "meta.yaml", task / "oracle_pytorch.py", task / "shapes.yaml",
                     *sorted((task / "golden").glob("*.json"))]:
            if path.is_file():
                files.append((f'references/{entry["task"]}/{path.relative_to(task)}', path))
    candidate_inventories = [("", inventory)]
    if args.supplement:
        candidate_inventories.append(("supplement/", json.loads(
            (args.supplement / "inventory.json").read_text())))
    for prefix, entries in candidate_inventories:
        for entry in entries:
            for index, candidate in enumerate(entry["candidates"]):
                solution = Path(candidate["path"])
                if digest(solution) != candidate["sha256"]:
                    raise RuntimeError(f"inventoried candidate changed: {solution}")
                files.append((f'candidates/{prefix}{entry["task"]}/{index}/solution.py', solution))
                if candidate.get("evidence"):
                    evidence = Path(candidate["evidence"])
                    if digest(evidence) != candidate["evidence_sha256"]:
                        raise RuntimeError(f"historical evidence changed: {evidence}")
                    files.append((f'candidates/{prefix}{entry["task"]}/{index}/verified_result.json', evidence))
    manifest_path = args.out / "manifest.json"
    archive = args.out / "raw-evidence.tar.gz"
    with tempfile.TemporaryDirectory(prefix="evidence-snapshot-", dir=args.out) as temporary:
        snapshots = []
        for name, path in files:
            snapshot = Path(temporary) / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, snapshot)
            snapshots.append((name, snapshot))
        manifest = [{"path": name, "bytes": path.stat().st_size, "sha256": digest(path)}
                    for name, path in snapshots]
        manifest_path.write_text(json.dumps(manifest, indent=2))
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(manifest_path, arcname="manifest.json")
            for name, path in snapshots:
                handle.add(path, arcname=name, recursive=False)
    (args.out / "archive-sha256.txt").write_text(digest(archive) + "  raw-evidence.tar.gz\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}))


if __name__ == "__main__":
    main()
