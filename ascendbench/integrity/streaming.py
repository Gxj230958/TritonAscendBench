"""One outstanding trial, two task-level processes, no accumulated tensors."""
from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from ascendbench.input_spec.inputs import split_inputs
from ascendbench.integrity import stream_transport as wire

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputSource(Mapping):
    shape_ids: tuple[str, ...]
    build: Callable[[str, int], dict]
    n_warmup: int = 10
    n_trials: int = 100
    offline: bool = False

    def __iter__(self):
        return iter(self.shape_ids)

    def __len__(self):
        return len(self.shape_ids)

    def __getitem__(self, sid):
        """Compatibility view of correctness inputs, materialized on lookup."""
        if sid not in self.shape_ids:
            raise KeyError(sid)
        return self.build(sid, 0)

    def __post_init__(self):
        if (not self.shape_ids or len(set(self.shape_ids)) != len(self.shape_ids)
                or self.n_warmup < 0 or self.n_trials < 1):
            raise ValueError("invalid streaming input schedule")


@contextlib.contextmanager
def preparation_deadline(seconds):
    """Interrupt trusted synchronous preparation; never abandon a tensor thread."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("streaming preparation requires the main thread")
    def expired(signum, frame):
        raise TimeoutError("input preparation/validation deadline exceeded")
    previous = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    if old_timer != (0.0, 0.0):
        raise RuntimeError("streaming cannot replace an active SIGALRM timer")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def device_environment(backend):
    names = (("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_DEVICE_ID", "DEVICE_ID")
             if backend == "npu" else ("CUDA_VISIBLE_DEVICES",))
    result = {}
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            pattern = r"\d+(,\d+)*" if name.endswith("VISIBLE_DEVICES") else r"\d+"
            if not re.fullmatch(pattern, value):
                raise ValueError(f"invalid {name}")
            result[name] = value
    if backend == "npu":
        ids = [result[k] for k in ("ASCEND_DEVICE_ID", "DEVICE_ID") if k in result]
        if len(set(ids)) > 1:
            raise ValueError("conflicting device IDs")
        if ids:
            selected = int(ids[0])
            visible = result.get("ASCEND_RT_VISIBLE_DEVICES")
            if visible is not None:
                devices = visible.split(",")
                if selected >= len(devices):
                    raise ValueError("device ID outside visible device list")
                selected = int(devices[selected])
            result["ASCEND_RT_VISIBLE_DEVICES"] = str(selected)
            for name in ("ASCEND_DEVICE_ID", "DEVICE_ID"):
                if name in result:
                    result[name] = "0"
    return result


def _schedule(source, pass_kind, samples, compile_needed):
    for sid in source.shape_ids:
        if compile_needed:
            yield sid, "compile", -1, 0, False, False
        if pass_kind == "correctness":
            yield sid, "correctness", 0, 0, True, True
        else:
            for w in range(source.n_warmup):
                yield sid, "warmup", w - source.n_warmup, 1 + w, False, False
            profiled = min(samples[sid]) if samples[sid] else 0
            for t in range(source.n_trials):
                sampled = t in samples[sid]
                seed = 0 if source.offline and sampled else 1000 + t
                yield sid, "measured", t, seed, sampled, t == profiled


def run_streaming_pass(solution, source, consumer, *, pass_kind, samples,
                       timeout_s, backend, transport="memfd",
                       preparation_timeout_s=300, transfer_timeout_s=300):
    from ascendbench.integrity import process_isolation as pi
    result = pi.CandidateRunResult(False, False, None, pass_kind=pass_kind, streamed=True)
    started = time.monotonic()
    metrics = {"pass": pass_kind, "input_bytes": 0, "max_input_bytes": 0,
               "trials_released": 0, "first_execute_s": None}
    devices = device_environment(backend)
    with tempfile.TemporaryDirectory(prefix="tab-stream-") as root:
        root = Path(root)
        for name in ("candidate", "cache", "scratch", "prof", "home"):
            (root / name).mkdir()
        candidate_dir = root / "candidate"
        shutil.copyfile(solution, candidate_dir / "solution.py")
        child_py = candidate_dir / "_child.py"
        child_py.write_text(pi._CHILD)
        parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        proc = None
        parser_proc = None
        parser_sock = None
        stderr_tail = deque(maxlen=8)
        stderr_thread = None
        try:
            args = dict(protocol_version=pi.PROTOCOL_VERSION,
                        status_schema_version=pi.STATUS_SCHEMA_VERSION,
                        solution=str(candidate_dir / "solution.py"), out_dir=str(candidate_dir),
                        status_json=str(candidate_dir / "status.json"), pass_kind=pass_kind,
                        inputs_mode="stream", shapes=[], backend=backend,
                        stream_fd=child_sock.fileno(), transport=transport)
            args_path = candidate_dir / "_args.json"
            args_path.write_text(json.dumps(args))
            env = {"PATH": os.environ.get("PATH", ""), "HOME": str(root / "home"),
                   "PYTHONPATH": str(pi.REPO_ROOT), "PYTHONHASHSEED": "0",
                   "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(root / "scratch"),
                   "TRITON_CACHE_DIR": str(root / "cache"), "AB_NPU_PROF_ROOT": str(root / "prof"),
                   "OMP_NUM_THREADS": "1", **devices}
            if backend == "gpu":
                env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
            if backend == "npu":
                env.update(pi._audited_npu_env((
                    "LD_LIBRARY_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH",
                    "ASCEND_OPP_PATH", "ASCEND_AICPU_PATH", "CMAKE_PREFIX_PATH", "TOOLCHAIN_HOME")))
            # DEVNULL prevents candidate logging from accumulating or filling a pipe.
            proc = subprocess.Popen([sys.executable, str(child_py), str(args_path)],
                                    cwd=candidate_dir, env=env, pass_fds=(child_sock.fileno(),),
                                    start_new_session=True, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE)
            def drain_stderr():
                while chunk := proc.stderr.read(4096):
                    stderr_tail.append(chunk)
            stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
            stderr_thread.start()
            child_sock.close()
            parent_sock.settimeout(timeout_s)
            ready, _ = wire.expect(parent_sock, "ready")
            if ready.get("error"):
                raise RuntimeError(ready["error"])
            compile_needed = ready.get("compile") is True
            per_shape_count = (1 if pass_kind == "correctness" else source.n_warmup + source.n_trials)
            count = len(source.shape_ids) * (per_shape_count + int(compile_needed))
            previous_sid = None
            execution_remaining = float(timeout_s)
            for identity, (sid, phase, trial, seed, output, profile) in enumerate(
                    _schedule(source, pass_kind, samples, compile_needed)):
                if sid != previous_sid:
                    logger.info("stream %s shape %s (%d/%d trials)", pass_kind, sid, identity, count)
                    previous_sid = sid
                scr = result.per_shape.setdefault(sid, pi.ShapeCandidateResult(sid, False))
                with preparation_deadline(preparation_timeout_s):
                    built = source.build(sid, seed)
                    tensors, scalars = split_inputs(built)
                    # Only the current trial exists here; close sender fd before execution.
                    with wire.payload_file(tensors, root / "scratch", transport) as (fd, meta):
                        metrics["input_bytes"] += meta["bytes"]
                        metrics["max_input_bytes"] = max(metrics["max_input_bytes"], meta["bytes"])
                        parent_sock.settimeout(transfer_timeout_s)
                        wire.send(parent_sock, wire.message(
                            "trial", id=identity, phase=phase, output=output, profile=profile,
                            scalars=scalars, last=identity == count - 1, **meta), fd)
                    del built, tensors, scalars
                parent_sock.settimeout(transfer_timeout_s)
                wire.expect(parent_sock, "loaded", identity)
                if execution_remaining <= 0:
                    raise TimeoutError("candidate pass execution budget exceeded")
                parent_sock.settimeout(execution_remaining)
                start = time.monotonic()
                if metrics["first_execute_s"] is None:
                    metrics["first_execute_s"] = start - started
                wire.send(parent_sock, wire.message("execute", id=identity))
                wire.expect(parent_sock, "executed", identity)
                execution_remaining -= time.monotonic() - start
                parent_sock.settimeout(transfer_timeout_s)
                msg, fd = wire.expect(parent_sock, "result", identity, payload=True)
                with preparation_deadline(preparation_timeout_s):
                    outputs = wire.load_payload(fd, msg)
                    rec = msg.get("record")
                    if not isinstance(rec, dict) or rec.get("error") or rec.get("ran") is not True:
                        raise RuntimeError(f"candidate {sid}/{phase}: {rec}")
                    durations = rec.get("durations")
                    expected_count = int(phase == "measured" and not profile)
                    if (not isinstance(durations, list) or len(durations) != expected_count
                            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                                   for v in durations)):
                        raise ValueError("invalid duration record")
                    events = rec.get("event_ns")
                    if events is not None and (not isinstance(events, list)
                            or len(events) != expected_count or any(
                                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                                for v in events)):
                        raise ValueError("invalid event record")
                    if output:
                        if consumer(pass_kind, sid, trial, seed, outputs) is False:
                            scr.error = "output validation failed"
                    del outputs
                    for name in ("kernel_event_present", "stream_nondefault", "fallback_signal",
                                 "backend_active", "npu_prof_raw_dir"):
                        if rec.get(name) is not None:
                            setattr(scr, name, rec[name])
                    if backend == "npu" and rec.get("npu_prof_raw_dir"):
                        if parser_proc is None:
                            parser_sock, parser_child = socket.socketpair(
                                socket.AF_UNIX, socket.SOCK_SEQPACKET)
                            try:
                                parser_proc = subprocess.Popen(
                                    [sys.executable, "-m", "ascendbench.integrity.profile_worker",
                                     str(parser_child.fileno())], env=env,
                                    pass_fds=(parser_child.fileno(),), start_new_session=True,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            finally:
                                parser_child.close()
                            parser_sock.settimeout(pi._ANALYSE_TIMEOUT_S)
                            wire.expect(parser_sock, "ready")
                        wire.send(parser_sock, wire.message("parse", id=identity,
                                  raw=rec["npu_prof_raw_dir"], root=str(root / "prof")))
                        parsed, _ = wire.expect(parser_sock, "parsed", identity)
                        scr.kernel_event_present = parsed["kernel"]
                        scr.fallback_signal = parsed["fallback"]
                        if parsed["kernel"] is None:
                            raise RuntimeError("profiler parse unverified")
                        # Trace lifetime is one profiled trial, not the whole task.
                        trace = Path(rec["npu_prof_raw_dir"]).resolve()
                        if trace.is_relative_to((root / "prof").resolve()) and trace != root / "prof":
                            shutil.rmtree(trace, ignore_errors=True)
                    scr.durations.extend(rec.get("durations") or [])
                    if rec.get("event_ns"):
                        if scr.event_ns is None:
                            scr.event_ns = []
                        scr.event_ns.extend(rec["event_ns"])
                parent_sock.settimeout(transfer_timeout_s)
                wire.send(parent_sock, wire.message("consumed", id=identity))
                wire.expect(parent_sock, "released", identity)
                metrics["trials_released"] += 1
                if phase == "correctness" or (phase == "measured" and trial == source.n_trials - 1):
                    scr.ran = True
            proc.wait(timeout=transfer_timeout_s)
            if proc.returncode != 0:
                raise RuntimeError(f"candidate exit {proc.returncode}")
            result.ran_any = all(s.ran for s in result.per_shape.values())
        except Exception as exc:
            result.timed_out = isinstance(exc, (TimeoutError, subprocess.TimeoutExpired))
            result.load_error = f"stream {pass_kind}: {type(exc).__name__}: {exc}"
            logger.error(result.load_error)
            # A partial pass must never produce usable performance scalars.
            for scr in result.per_shape.values():
                scr.ran = False
                scr.error = result.load_error
                scr.durations.clear()
                scr.event_ns = None
        finally:
            if parser_sock is not None:
                parser_sock.close()
            if parser_proc is not None:
                pi.kill_process_group(parser_proc.pid)
                parser_proc.wait(timeout=10)
            parent_sock.close()
            child_sock.close()
            if proc is not None:
                pi.kill_process_group(proc.pid)
                proc.wait(timeout=10)
                stderr_thread.join(timeout=5)
                proc.stderr.close()
                if result.load_error:
                    detail = b"".join(stderr_tail).decode(errors="replace")[-4000:]
                    result.load_error += "\n" + detail
            report_dir = os.environ.get("TAB_STREAM_REPORT_DIR")
            if report_dir:
                report = Path(report_dir)
                report.mkdir(parents=True, exist_ok=True)
                metrics.update(elapsed_s=time.monotonic() - started, devices=devices,
                               pid=proc.pid if proc else None, result=asdict(result))
                (report / f"{pass_kind}-{os.getpid()}.json").write_text(json.dumps(metrics, indent=2))
    return result


def run_streaming_two_pass(solution, source, consumer, *, timeout_s=90,
                           sample_size=10, backend="npu", **kwargs):
    from ascendbench.integrity import process_isolation as pi
    if consumer is None:
        raise ValueError("streaming requires an immediate result consumer")
    samples = {sid: sorted(secrets.SystemRandom().sample(
        range(source.n_trials), min(sample_size, source.n_trials))) for sid in source.shape_ids}
    correctness = run_streaming_pass(solution, source, consumer, pass_kind="correctness",
                                    samples=samples, timeout_s=timeout_s, backend=backend, **kwargs)
    if correctness.load_error or correctness.timed_out or any(
            scr.error for scr in correctness.per_shape.values()):
        timing = pi.CandidateRunResult(False, False, "correctness pass failed", pass_kind="timing")
    else:
        timing = run_streaming_pass(solution, source, consumer, pass_kind="timing",
                                   samples=samples, timeout_s=timeout_s, backend=backend, **kwargs)
    return pi.TwoPassResult(correctness, timing, ["owned streaming pass resources"])
