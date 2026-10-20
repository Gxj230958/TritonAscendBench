import json

from scripts.internal.streaming_validation import device_idle, pending_attempts


def test_busy_or_unhealthy_device_is_not_admitted():
    healthy = "| 3 910B2C | OK |\n| No running processes found in NPU 3 |"
    assert device_idle(healthy, 3)
    assert not device_idle(healthy, 2)
    assert not device_idle(healthy.replace("OK", "ERROR"), 3)
    assert not device_idle(healthy.replace("No running processes found in NPU 3", "3 0 123 python"), 3)


def test_resume_retries_first_unexecuted_candidate(tmp_path):
    records = [{"task": "task", "candidates": [{}, {}, {}]}]
    first = tmp_path / "task/0"
    first.mkdir(parents=True)
    (first / "result.json").write_text(json.dumps({"result": {
        "runtime_status": "error", "correctness_status": "fail"}}))
    busy = tmp_path / "task/1"
    busy.mkdir()
    (busy / "run.log").write_text("device busy")
    assert pending_attempts(records, tmp_path) == [("task", 1)]
    (busy / "result.json").write_text(json.dumps({"result": {
        "runtime_status": "error", "correctness_status": "fail"}}))
    assert pending_attempts(records, tmp_path) == [("task", 2)]


def test_resume_skips_successful_later_candidate(tmp_path):
    records = [{"task": "task", "candidates": [{}, {}, {}]}]
    final = tmp_path / "task/2"
    final.mkdir(parents=True)
    (final / "result.json").write_text(json.dumps({"result": {
        "runtime_status": "ok", "correctness_status": "ok"}}))
    assert pending_attempts(records, tmp_path) == []
