import json
import subprocess
import sys
from pathlib import Path

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


def test_device_crash_is_separate_from_an_earlier_numerical_mismatch(tmp_path):
    write_run(tmp_path, timing_trials=0, timing_ran=False,
              runtime="error", performance="error")
    file = tmp_path / "task/0/result.json"
    run = json.loads(file.read_text())
    run["result"]["shape_results"] = [{"correctness_reason": "shape mismatch"}]
    run["result"]["error_reason"] = "closed channel: vector core exception, code 507035"
    file.write_text(json.dumps(run))
    subprocess.run([sys.executable, str(SCRIPT), str(tmp_path)], check=True, capture_output=True)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["complete_device_tasks"] == 0
    assert summary["records"][0]["attempts"][0]["category"] == "device_execution_failure"


def test_missing_input_key_during_compile_warmup_is_a_contract_failure(tmp_path):
    write_run(tmp_path, timing_trials=0, timing_ran=False,
              runtime="error", performance="error")
    file = tmp_path / "task/0/result.json"
    run = json.loads(file.read_text())
    run["result"]["error_reason"] = "candidate P1/compile: KeyError: 'cb'"
    file.write_text(json.dumps(run))
    subprocess.run([sys.executable, str(SCRIPT), str(tmp_path)], check=True, capture_output=True)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["complete_device_tasks"] == 0
    assert summary["records"][0]["attempts"][0]["category"] == "candidate_contract_failure"
