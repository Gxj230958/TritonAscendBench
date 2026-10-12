"""Summarize actual device coverage, keeping attempted and completed distinct."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    inventory = json.loads((args.root / "inventory.json").read_text())
    tasks = []
    for entry in inventory:
        required = set(entry["public"] + entry["hidden"])
        attempts = []
        for file in sorted((args.root / entry["task"]).glob("*/result.json")):
            run = json.loads(file.read_text())
            result = run["result"]
            phases = {}
            protocol_errors = []
            for receipt in file.parent.glob("protocol/*.json"):
                protocol = json.loads(receipt.read_text())
                phase = protocol["pass"]
                shape_records = protocol["result"]["per_shape"]
                phases[phase] = {
                    sid for sid, rec in shape_records.items()
                    if rec.get("ran") and rec.get("backend_active") is True
                    and rec.get("kernel_event_present") is True
                    and (phase != "timing" or len(rec.get("durations", [])) == 99)}
                if protocol["result"].get("load_error"):
                    protocol_errors.append(protocol["result"]["load_error"])
            complete = all(phases.get(p, set()) == required for p in ("correctness", "timing"))
            reasons = [s.get("correctness_reason") for s in result.get("shape_results", [])
                       if s.get("correctness_reason")]
            text = " ".join([result.get("error_reason") or "", *reasons, *protocol_errors])
            if complete and result.get("correctness_status") == "ok":
                category = ("complete_verified" if result.get("runtime_status") == "ok"
                            and result.get("performance_status") == "ok"
                            else "complete_performance_flagged")
            elif result.get("static_checks_passed") is False:
                category = "candidate_static_rejected"
            elif "golden" in text.lower() or result.get("task_status") in ("blocked", "unsupported"):
                category = "artifact_or_contract_blocked"
            elif any(word in text.lower() for word in ("compile", "compilation", "llvm", "triton.compiler")):
                category = "candidate_compile_failure"
            elif any(word in text.lower() for word in ("verification failed", "mismatch", "missing output", "max_abs", "tolerance", "bit_exact equality failed")):
                category = "candidate_numerical_failure"
            else:
                category = "incomplete_needs_diagnosis"
            attempts.append({"candidate_sha256": run["candidate"]["sha256"],
                             "result": str(file.relative_to(args.root)), "category": category,
                             "complete": complete, "correctness": result.get("correctness_status"),
                             "runtime": result.get("runtime_status"), "performance": result.get("performance_status"),
                             "executed_correctness_shapes": sorted(phases.get("correctness", set())),
                             "executed_timing_shapes": sorted(phases.get("timing", set())),
                             "elapsed_s": run["elapsed_s"], "reason": text[:3000]})
        tasks.append({"task": entry["task"], "required_shapes": len(required), "attempts": attempts,
                      "complete": any(a["complete"] for a in attempts)})
    summary = {"tasks": len(tasks), "required_shapes": sum(t["required_shapes"] for t in tasks),
               "attempted_tasks": sum(bool(t["attempts"]) for t in tasks),
               "complete_device_tasks": sum(t["complete"] for t in tasks), "records": tasks}
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}))


if __name__ == "__main__":
    main()
