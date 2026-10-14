"""Freeze additional existing candidates after the primary verified campaign."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ascendbench.integrity import static_checker  # noqa: E402 - source checkout bootstrap
from scripts.internal.streaming_validation import sha  # noqa: E402


def rank(candidate):
    return (candidate["correctness"] == "ok", candidate["runtime"] == "ok",
            candidate["correct_shapes"])


def select_candidates(candidates):
    compatible = sorted((c for c in candidates if not c["static_blocked"]),
                        key=rank, reverse=True)
    proven = [c for c in compatible if c["correctness"] == "ok"]
    return proven or compatible[:1]


def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.out / "inventory.json").exists():
        raise RuntimeError("supplement inventory already frozen")
    original = {r["task"]: r for r in json.loads((args.suite / "inventory.json").read_text())}
    summary = json.loads(args.summary.read_text())
    failed = {r["task"] for r in summary["records"] if not r["verified_pass"]
              and r["final_category"] != "artifact_or_contract_blocked"}
    found = {task: {} for task in failed}
    for exp in sorted(args.experiments.iterdir()):
        for base in (exp / "tasks", exp / "bare/tasks", exp / "pipeline/tasks"):
            for task in sorted(failed):
                seen = {c["sha256"] for c in original[task]["candidates"]}
                for solution in sorted((base / task / "rounds").glob("*/solution.py")):
                    digest = sha(solution)
                    if digest in seen:
                        continue
                    receipt = solution.parent / "verified_result.json"
                    try:
                        evidence = json.loads(receipt.read_text())
                    except (OSError, ValueError):
                        evidence = {}
                    candidate = {"path": str(solution), "sha256": digest,
                                 "evidence": str(receipt) if receipt.exists() else None,
                                 "evidence_sha256": sha(receipt) if receipt.exists() else None,
                                 "correctness": evidence.get("correctness_status"),
                                 "runtime": evidence.get("runtime_status"),
                                 "correct_shapes": sum(s.get("correctness_passed", False)
                                                       for s in evidence.get("shape_results", []))}
                    previous = found[task].get(digest)
                    if previous is None or rank(candidate) > rank(previous):
                        found[task][digest] = candidate
    screening, inventory = [], []
    for task in sorted(failed):
        entry = original[task]
        meta = yaml.safe_load((Path(entry["task_dir"]) / "meta.yaml").read_text())
        checks = meta.get("runner_checks", {}).get("static_checks", [])
        candidates = list(found[task].values())
        for candidate in candidates:
            source = Path(candidate["path"])
            if sha(source) != candidate["sha256"]:
                raise RuntimeError("candidate changed during screening")
            report = static_checker.check_solution(source.read_text(), required_checks=checks)
            candidate["static_blocked"] = report.blocked
            candidate["static_findings"] = [asdict(f) for f in report.findings]
        selected = select_candidates(candidates)
        screening.append({"task": task, "candidates": candidates,
                          "selected_sha256": [c["sha256"] for c in selected]})
        if selected:
            inventory.append({**entry, "candidates": selected})
    args.out.mkdir(parents=True)
    (args.out / "inventory.json").write_text(json.dumps(inventory, indent=2))
    (args.out / "candidate-screening.json").write_text(json.dumps({
        "policy": "all unused statically compatible historically-correct candidates; otherwise strongest remaining compatible candidate",
        "original_inventory_sha256": sha(args.suite / "inventory.json"),
        "records": screening}, indent=2))
    print(json.dumps({"tasks_screened": len(screening),
                      "distinct_unused_sources": sum(len(r["candidates"]) for r in screening),
                      "tasks_selected": len(inventory),
                      "candidates_selected": sum(len(r["candidates"]) for r in inventory)}))


if __name__ == "__main__":
    main()
