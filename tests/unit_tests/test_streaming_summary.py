import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/internal/streaming_summary.py"


def write_run(root, timing_trials=99, timing_ran=True, runtime="ok", performance="ok"):
    ids = ["P1", "H1"]
    (root / "inventory.json").write_text(json.dumps([
        {"task": "task", "public": ["P1"], "hidden": ["H1"]}]))
    attempt = root / "task/0"
    protocol = attempt / "protocol"
    protocol.mkdir(parents=True)
    (attempt / "result.json").write_text(json.dumps({
        "candidate": {"sha256": "frozen"}, "elapsed_s": 1,
        "result": {"static_checks_passed": True, "correctness_status": "ok",
                   "runtime_status": runtime, "performance_status": performance,
                   "shape_results": []}}))
    for phase in ("correctness", "timing"):
        records = {sid: {"ran": phase == "correctness" or timing_ran,
                         "backend_active": True, "kernel_event_present": True,
                         "durations": [1] * timing_trials if phase == "timing" else []}
                   for sid in ids}
        (protocol / f"{phase}.json").write_text(json.dumps({
            "pass": phase, "result": {"per_shape": records, "load_error": None}}))
    subprocess.run([sys.executable, str(SCRIPT), str(root)], check=True, capture_output=True)
    return json.loads((root / "summary.json").read_text())


def test_partial_timing_preserves_observation_without_full_coverage(tmp_path):
    summary = write_run(tmp_path, timing_trials=0, timing_ran=False,
                        runtime="error", performance="error")
    assert summary["attempted_tasks"] == 1
    assert summary["complete_device_tasks"] == 0
    attempt = summary["records"][0]["attempts"][0]
    assert attempt["executed_timing_shapes"] == []
    assert attempt["observed_timing_shapes"] == ["H1", "P1"]


def test_complete_device_run_with_invalid_performance_is_not_verified_pass(tmp_path):
    summary = write_run(tmp_path, runtime="invalid_timing", performance="quarantined")
    assert summary["complete_device_tasks"] == 1
    attempt = summary["records"][0]["attempts"][0]
    assert attempt["category"] == "complete_performance_flagged"
    assert attempt["executed_timing_shapes"] == ["H1", "P1"]
