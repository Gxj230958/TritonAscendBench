"""Audit frozen campaign receipts and every entry in the evidence archive."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
import tarfile
from collections import Counter
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalized(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                body.pop(0)
    return ast.dump(tree, include_attributes=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--runtime-commit", default="f940554",
                        help="Frozen runtime revision whose AST is compared with campaign sources.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    roots = {name: getattr(args, name) for name in ("suite", "replay", "supplement")}
    inventories = {name: {entry["task"]: entry for entry in json.loads(
        (path / "inventory.json").read_text())} for name, path in roots.items()}
    summary = json.loads((args.evidence / "summary.json").read_text())
    attempts = Counter()
    complete_shapes = 0
    profiler_roots = set()
    runtime_hashes = {}
    for row in summary["records"]:
        for attempt in row["attempts"]:
            origin = attempt["evidence_root"]
            receipt = roots[origin] / attempt["result"]
            run = json.loads(receipt.read_text())
            entry = inventories[origin][row["task"]]
            assert run["candidate"]["sha256"] in {c["sha256"] for c in entry["candidates"]}
            assert sha(Path(run["candidate"]["path"]).read_bytes()) == run["candidate"]["sha256"]
            for name, digest in run["reference_sha256"].items():
                assert sha((Path(entry["task_dir"]) / name).read_bytes()) == digest
            commit = "f940554" if origin == "supplement" else "2092615"
            for name, digest in run["code_sha256"].items():
                key = (commit, name)
                if key not in runtime_hashes:
                    source = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
                    reference = subprocess.check_output(
                        ["git", "show", f"{args.runtime_commit}:{name}"], cwd=root)
                    assert normalized(source) == normalized(reference)
                    runtime_hashes[key] = sha(source)
                assert runtime_hashes[key] == digest
            attempts[attempt["category"]] += 1
            for path in receipt.parent.glob("protocol/*.json"):
                protocol = json.loads(path.read_text())
                assert protocol["devices"]["ASCEND_RT_VISIBLE_DEVICES"] == str(run["device"])
                records = protocol["result"]["per_shape"]
                for rec in records.values():
                    if rec.get("npu_prof_raw_dir"):
                        profiler_roots.add(str(Path(rec["npu_prof_raw_dir"]).parent))
                if attempt["category"] != "complete_verified":
                    continue
                assert set(records) == set(entry["public"] + entry["hidden"])
                assert protocol["trials_released"] == len(records) * (
                    111 if protocol["pass"] == "timing" else 2)
                for rec in records.values():
                    assert rec["ran"] and rec["backend_active"] and rec["kernel_event_present"]
                    if protocol["pass"] == "timing":
                        for key in ("durations", "event_ns"):
                            samples = rec[key]
                            assert len(samples) == 99 and all(math.isfinite(x) and x > 0 for x in samples)
                        complete_shapes += 1
    assert complete_shapes == summary["complete_device_shapes"]
    assert all(not Path(path).exists() for path in profiler_roots)
    manifest = json.loads((args.evidence / "manifest.json").read_text())
    archive = args.evidence / "raw-evidence.tar.gz"
    assert sha(archive.read_bytes()) == (args.evidence / "archive-sha256.txt").read_text().split()[0]
    with tarfile.open(archive) as handle:
        names = handle.getnames()
        assert len(names) == len(set(names)) == len(manifest) + 1
        for entry in manifest:
            assert not any(part in {"home", "scratch", "cache", "prof", "__pycache__"}
                           for part in Path(entry["path"]).parts)
            data = handle.extractfile(entry["path"]).read()
            assert len(data) == entry["bytes"] and sha(data) == entry["sha256"]
    print(json.dumps({"attempts": dict(attempts), "complete_timing_shapes": complete_shapes,
                      "removed_profiler_roots": len(profiler_roots),
                      "archive_entries_checked": len(manifest),
                      "archive_sha256": sha(archive.read_bytes()),
                      "runtime_ast_equivalence": True,
                      "runtime_comparison_commit": args.runtime_commit}, indent=2))


if __name__ == "__main__":
    main()
