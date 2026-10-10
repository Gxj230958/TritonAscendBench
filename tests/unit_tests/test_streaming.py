import os
import socket
import weakref
import time

import pytest
import torch

from ascendbench.integrity import stream_transport as wire
from ascendbench.integrity.streaming import InputSource, run_streaming_two_pass


@pytest.mark.parametrize("transport", ["memfd", "file"])
def test_stream_lifecycle(tmp_path, transport):
    sol = tmp_path / "solution.py"
    sol.write_text('import torch\ndef npu_triton_impl(inputs):\n'
                   '    inputs["x"].add_(inputs["scale"])\n'
                   '    return {"y": inputs["x"]}\n')
    pending = []
    pending_outputs = []
    received = []
    seeds = []
    baseline_fds = len(os.listdir("/proc/self/fd"))

    def build(sid, seed):
        assert all(ref() is None for ref in pending)
        assert all(ref() is None for ref in pending_outputs)
        x = torch.randn(128, generator=torch.Generator().manual_seed(seed))
        pending.append(weakref.ref(x))
        seeds.append(seed)
        return {"x": x, "scale": 2.0, "axes": [1, 2], "optional": None}

    def consume(phase, sid, trial, seed, output):
        assert all(ref() is None for ref in pending)
        expected = torch.randn(128, generator=torch.Generator().manual_seed(seed)) + 2
        torch.testing.assert_close(output["y"], expected)
        pending_outputs.append(weakref.ref(output["y"]))
        received.append((phase, sid, trial, seed))
        assert len(os.listdir("/proc/self/fd")) <= baseline_fds + 5

    source = InputSource(tuple(f"s{i}" for i in range(4)), build, 2, 6)
    result = run_streaming_two_pass(sol, source, consume, backend="gpu", transport=transport,
                                   sample_size=3, timeout_s=90)
    assert result.correctness.load_error is None
    assert result.timing.load_error is None
    assert len(seeds) == 4 * (1 + 2 + 6)
    assert len(received) == 4 * (1 + 3)
    assert all(len(r.durations) == 5 for r in result.timing.per_shape.values())
    assert all(not r.outputs and r.sampled_outputs is None
               for p in (result.correctness, result.timing) for r in p.per_shape.values())
    assert len(os.listdir("/proc/self/fd")) == baseline_fds


def test_offline_samples_use_seed_zero(tmp_path):
    sol = tmp_path / "solution.py"
    sol.write_text('def npu_triton_impl(inputs):\n    return {"y": inputs["x"]}\n')
    seeds, checked = [], []
    def build(sid, seed):
        seeds.append(seed)
        return {"x": torch.tensor([seed])}
    def consume(phase, sid, trial, seed, outputs):
        checked.append(seed)
        assert seed == 0
        assert outputs["y"].item() == 0
    result = run_streaming_two_pass(sol, InputSource(("s",), build, 2, 8, True), consume,
                                   sample_size=3, backend="gpu")
    assert result.timing.load_error is None
    assert len(checked) == 4
    assert seeds.count(0) == 4
    assert len(set(seeds) - {0, 1, 2}) == 5


@pytest.mark.parametrize("body", ['raise RuntimeError("broken")', 'while True: pass'])
def test_child_failure_cleans_resources(tmp_path, body):
    sol = tmp_path / "solution.py"
    sol.write_text(f'def npu_triton_impl(inputs):\n    {body}\n')
    result = run_streaming_two_pass(sol, InputSource(("s",), lambda s, t: {"x": torch.ones(2)}),
                                   lambda *args: None, timeout_s=15, backend="gpu")
    assert result.correctness.load_error
    assert not result.timing.ran_any
    assert not any(r.durations for r in result.correctness.per_shape.values())


def test_reject_unsealed_and_wrong_identity(tmp_path):
    fd = os.memfd_create("unsealed", os.MFD_ALLOW_SEALING)
    os.write(fd, b"0123456789")
    with pytest.raises(ValueError, match="unsealed"):
        wire.load_payload(fd, {"bytes": 10, "storage": "memfd"})
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        wire.send(a, wire.message("released", id=3))
        with pytest.raises(ValueError, match="unexpected"):
            wire.expect(b, "released", 2)
    finally:
        a.close()
        b.close()


def test_two_processes_and_no_prefetch(tmp_path):
    sol = tmp_path / "solution.py"
    sol.write_text('import os, torch\n_calls = 0\ndef npu_triton_impl(inputs):\n'
                   '    global _calls\n    _calls += 1\n'
                   '    return {"pid": torch.tensor([os.getpid()]), "calls": torch.tensor([_calls])}\n')
    events = []
    pids = {}
    def build(sid, seed):
        events.append(("build", seed))
        return {"x": torch.ones(1)}
    def consume(phase, sid, trial, seed, output):
        pids.setdefault(phase, int(output["pid"].item()))
        assert pids[phase] == output["pid"].item()
        assert events[-1] == ("build", seed)
        events.append(("consume", seed))
        if phase == "correctness":
            assert output["calls"].item() == 1
        else:
            assert output["calls"].item() == trial + 3
    result = run_streaming_two_pass(sol, InputSource(("s",), build, 2, 5), consume,
                                   backend="gpu", sample_size=5)
    assert result.timing.load_error is None
    assert pids["correctness"] != pids["timing"]
    assert all(not os.path.exists(f"/proc/{pid}") for pid in pids.values())


def test_preparation_timeout_does_not_leave_worker(tmp_path):
    sol = tmp_path / "solution.py"
    sol.write_text('def npu_triton_impl(inputs):\n    return {}\n')
    def build(sid, seed):
        time.sleep(5)
    result = run_streaming_two_pass(sol, InputSource(("s",), build), lambda *a: None,
                                   backend="gpu", preparation_timeout_s=0.05)
    assert result.correctness.timed_out
    assert "preparation" in result.correctness.load_error
    assert not result.timing.ran_any


@pytest.mark.parametrize("failure", ["missing", "crash", "validation"])
def test_failed_output_cannot_produce_timing(tmp_path, failure):
    sol = tmp_path / "solution.py"
    body = 'return {}' if failure != "crash" else 'os._exit(7)'
    sol.write_text(f'import os\ndef npu_triton_impl(inputs):\n    {body}\n')
    def consume(*args):
        if failure == "missing":
            assert "y" in args[-1], "missing output"
        return False
    result = run_streaming_two_pass(sol, InputSource(("s",), lambda *a: {"x": torch.ones(1)}),
                                   consume, backend="gpu")
    assert not result.timing.ran_any
    assert not any(r.durations for r in result.timing.per_shape.values())


def test_golden_index_does_not_retain_tensors(tmp_path):
    from safetensors.torch import save_file
    from ascendbench.engine.eval import _OfflineGolden
    paths = {}
    for i in range(20):
        paths[str(i)] = tmp_path / f"{i}.safetensors"
        save_file({"x": torch.ones(1024)}, str(paths[str(i)]))
    golden = _OfflineGolden(paths)
    refs = []
    for sid in golden:
        value = golden[sid]
        refs.append(weakref.ref(value["x"]))
        del value
        assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("offline", [False, True])
def test_evaluate_streams_and_checks_every_sample(tmp_path, monkeypatch, offline):
    from types import SimpleNamespace
    import yaml
    from safetensors.torch import save_file
    from tests.unit_tests.test_eval_timing_integration_pins import _make_task, _TENSOR_INPUTS
    from ascendbench.engine import eval as engine
    from ascendbench.input_spec.shape_format import build_inputs_for
    from ascendbench.integrity import process_isolation as pi

    task, sol = _make_task(tmp_path, inputs_spec=_TENSOR_INPUTS)
    sol.write_text('def npu_triton_impl(inputs):\n'
                   '    inputs["x"].add_(1)\n    return {"y": inputs["x"]}\n')
    monkeypatch.setattr(engine.static_checker, "check_solution",
                        lambda *a, **k: SimpleNamespace(blocked=False, findings=[]))
    monkeypatch.setattr(pi, "DEFAULT_N_WARMUP", 1)
    monkeypatch.setattr(pi, "DEFAULT_N_TRIALS", 3)
    if offline:
        task.meta["oracle_simplified"] = "upstream_oracle_required"
        golden = tmp_path / "golden"
        golden.mkdir()
        shapes = yaml.safe_load((tmp_path / "shapes.yaml").read_text())
        for sid in ("P1", "P2"):
            save_file({"y": build_inputs_for(shapes, sid, seed=0)["x"] + 1},
                      str(golden / f"{sid}.safetensors"))
        (tmp_path / "oracle_pytorch.py").write_text('raise RuntimeError("must not import")\n')
    result = engine.evaluate(task.task_id, sol, backend="gpu", task_def=task, mode="verified")
    assert result.error_reason is None
    assert result.correctness_status == "ok", result.shape_results
    assert result.runtime_status == "ok"
    assert len(result.shape_results) == 2
    assert all(s.sampled_outputs_checked == 3 and s.trials == 2 for s in result.shape_results)


def test_device_selection_is_explicit_and_consistent(monkeypatch):
    from ascendbench.integrity.streaming import device_environment
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,7")
    monkeypatch.setenv("DEVICE_ID", "1")
    monkeypatch.delenv("ASCEND_DEVICE_ID", raising=False)
    assert device_environment("npu") == {"ASCEND_RT_VISIBLE_DEVICES": "7", "DEVICE_ID": "0"}
    monkeypatch.setenv("ASCEND_DEVICE_ID", "0")
    with pytest.raises(ValueError, match="conflicting"):
        device_environment("npu")
    monkeypatch.delenv("ASCEND_DEVICE_ID")
    monkeypatch.setenv("DEVICE_ID", "2")
    with pytest.raises(ValueError, match="outside"):
        device_environment("npu")


def test_bad_json_and_extra_descriptors_are_closed(tmp_path):
    import array
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    fd = os.open(tmp_path, os.O_RDONLY)
    count = len(os.listdir("/proc/self/fd"))
    try:
        a.sendmsg([b'{"v":1,"v":1}'], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))])
        with pytest.raises(ValueError, match="duplicate"):
            wire.receive(b, payload=True)
        assert len(os.listdir("/proc/self/fd")) == count
        a.sendmsg([b'{"v":1}'], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd, fd]))])
        with pytest.raises(ValueError, match="descriptor count"):
            wire.receive(b, payload=True)
        assert len(os.listdir("/proc/self/fd")) == count
    finally:
        a.close()
        b.close()
        os.close(fd)
