"""Reproducible streaming experiment inventory and verified device execution."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal
import shutil
import shlex
import re

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def device_idle(status, device):
    return (f"No running processes found in NPU {device} " in status and
            re.search(rf"\|\s*{device}\s+910B\S*\s*\|\s*OK\s*\|", status) is not None)


def pending_attempts(records, root, task=None):
    queue = []
    for entry in records:
        if task and entry["task"] != task:
            continue
        missing = None
        for index in range(len(entry["candidates"])):
            path = root / entry["task"] / str(index) / "result.json"
            if not path.exists():
                if missing is None:
                    missing = index
                continue
            result = json.loads(path.read_text())["result"]
            if result.get("runtime_status") == "ok" and result.get("correctness_status") == "ok":
                missing = None
                break
        if missing is not None:
            queue.append((entry["task"], missing))
    return queue


def inventory(args):
    import yaml
    from ascendbench.input_spec.inputs import shape_ids
    records = []
    for task_dir in sorted(args.tasks.glob("level*/*")):
        if not (task_dir / "meta.yaml").is_file():
            continue
        candidates = []
        for exp in args.experiments.iterdir():
            for base in (exp / "tasks", exp / "bare/tasks", exp / "pipeline/tasks"):
                rounds = base / task_dir.name / "rounds"
                for solution in rounds.glob("*/solution.py"):
                    result_file = solution.parent / "verified_result.json"
                    try:
                        evidence = json.loads(result_file.read_text())
                    except (OSError, ValueError):
                        evidence = {}
                    good = sum(s.get("correctness_passed", False)
                               for s in evidence.get("shape_results", []))
                    candidates.append({"path": str(solution), "sha256": sha(solution),
                                       "evidence": str(result_file) if result_file.exists() else None,
                                       "evidence_sha256": sha(result_file) if result_file.exists() else None,
                                       "correct_shapes": good,
                                       "correctness": evidence.get("correctness_status"),
                                       "runtime": evidence.get("runtime_status")})
        candidates.sort(key=lambda c: (c["correctness"] == "ok", c["runtime"] == "ok",
                                       c["correct_shapes"]), reverse=True)
        # Keep distinct existing candidates for compiler/numerical retries.
        unique = {c["sha256"]: c for c in reversed(candidates)}
        candidates = sorted(unique.values(), key=lambda c: (
            c["correctness"] == "ok", c["runtime"] == "ok", c["correct_shapes"]), reverse=True)
        shapes_path = task_dir / "shapes.yaml"
        shapes = yaml.safe_load(shapes_path.read_text())
        pub, hid = shape_ids(shapes)
        records.append({"task": task_dir.name, "task_dir": str(task_dir),
                        "shapes_sha256": sha(shapes_path), "public": pub, "hidden": hid,
                        "shapes": shapes, "candidates": candidates[:3]})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "inventory.json").write_text(json.dumps(records, indent=2))
    print(json.dumps({"tasks": len(records), "with_candidates": sum(bool(r["candidates"]) for r in records),
                      "shapes": sum(len(r["public"]) + len(r["hidden"]) for r in records)}))


def run_task(args):
    # Health/occupancy is checked immediately before imports that initialize NPU.
    status = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True).stdout
    if not device_idle(status, args.device):
        print(f"device {args.device} is not healthy and idle; retry admission", flush=True)
        raise SystemExit(75)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device)
    os.environ.pop("ASCEND_DEVICE_ID", None)
    os.environ.pop("DEVICE_ID", None)
    os.environ["OMP_NUM_THREADS"] = "1"
    from ascendbench.corpus import dataset
    from ascendbench.engine.eval import evaluate
    records = json.loads((args.out / "inventory.json").read_text())
    entry = next(r for r in records if r["task"] == args.task)
    task_dir = Path(entry["task_dir"])
    if sha(task_dir / "shapes.yaml") != entry["shapes_sha256"]:
        raise RuntimeError("shapes changed since inventory")
    candidate = entry["candidates"][args.candidate]
    solution = Path(candidate["path"])
    if sha(solution) != candidate["sha256"]:
        raise RuntimeError("candidate changed since inventory")
    dataset.TASKS_ROOT = task_dir.parent.parent
    task = dataset.load_task(args.task)
    destination = args.out / args.task / str(args.candidate)
    destination.mkdir(parents=True, exist_ok=True)
    frozen_solution = destination / "solution.py"
    shutil.copyfile(solution, frozen_solution)
    if sha(frozen_solution) != candidate["sha256"]:
        raise RuntimeError("candidate changed during snapshot")
    solution = frozen_solution
    os.environ["TAB_STREAM_REPORT_DIR"] = str(destination / "protocol")
    (destination / "device-before.txt").write_text(status)
    versions = {}
    for package in ("torch", "torch-npu", "triton", "triton-ascend", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    instrumentation = []
    if args.baseline_root:
        from ascendbench.integrity import process_isolation as pi
        # Legacy children strip device selection. A tiny interpreter adapter
        # supplies the exact same selected card without changing kernel code.
        launcher = destination / "selected-python.sh"
        launcher.write_text("#!/bin/sh\nexport ASCEND_RT_VISIBLE_DEVICES=" + str(args.device)
                            + "\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
        launcher.chmod(0o700)
        pi.sys.executable = str(launcher)
        for name in ("home", "scratch"):
            (destination / name).mkdir(exist_ok=True)
        os.environ["HOME"] = str(destination / "home")
        os.environ["TMPDIR"] = str(destination / "scratch")
        import tempfile
        tempfile.tempdir = str(destination / "scratch")
        pi.scrub_between_passes = lambda: []
        original = pi.run_candidate_isolated
        def record_pass(*pos, **kw):
            kw["timeout_s"] = int(os.environ.get("TAB_PASS_TIMEOUT_S", "1800"))
            response = original(*pos, **kw)
            records = {}
            for sid, rec in response.per_shape.items():
                records[sid] = {field.name: getattr(rec, field.name) for field in dataclasses.fields(rec)
                                if field.name not in ("outputs", "sampled_outputs")}
            target = destination / "protocol"
            target.mkdir(exist_ok=True)
            (target / f"{response.pass_kind}-{os.getpid()}.json").write_text(json.dumps({
                "pass": response.pass_kind, "result": {"per_shape": records,
                "load_error": response.load_error, "timed_out": response.timed_out}}, indent=2))
            return response
        pi.run_candidate_isolated = record_pass
        instrumentation = ["selected-device interpreter adapter", "private HOME/TMPDIR",
                           "global scrub disabled", "explicit 1800s child timeout", "scalar diagnostic receipt"]
    started = time.monotonic()
    code_hashes = {str(p.relative_to(ROOT)): sha(p) for p in (
        ROOT / "ascendbench/engine/eval.py", ROOT / "ascendbench/integrity/process_isolation.py",
        ROOT / "ascendbench/integrity/streaming.py", ROOT / "ascendbench/integrity/stream_transport.py",
        ROOT / "ascendbench/integrity/profile_worker.py") if p.is_file()}
    reference_hashes = {str(p.relative_to(task_dir)): sha(p) for p in (
        task_dir / "meta.yaml", task_dir / "oracle_pytorch.py", task_dir / "shapes.yaml",
        *sorted((task_dir / "golden").glob("*"))) if p.is_file()}
    result = evaluate(args.task, solution, backend="npu", task_def=task, mode="verified")
    output = {"result": dataclasses.asdict(result), "elapsed_s": time.monotonic() - started,
              "versions": versions, "candidate": candidate, "device": args.device, "code_sha256": code_hashes,
              "reference_sha256": reference_hashes,
              "baseline_instrumentation": instrumentation,
              "command": sys.argv, "commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (destination / "result.json").write_text(json.dumps(output, indent=2))
    print(json.dumps({"task": args.task, "elapsed_s": output["elapsed_s"],
                      "correctness": result.correctness_status, "runtime": result.runtime_status,
                      "shapes": len(result.shape_results), "error": result.error_reason}))


def suite(args):
    """Supervise only owned workers; include system file cache conservatively."""
    import psutil
    records = json.loads((args.out / "inventory.json").read_text())
    queue = pending_attempts(records, args.out, args.task)
    if args.limit:
        queue = queue[:args.limit]
    active = {}
    started = time.monotonic()
    peak = 0
    samples = (args.out / f"resources-{args.workers}.jsonl").open("a")
    outcomes = []
    last_admission_check = 0
    last_status = ""
    last_notice = 0
    session = str(time.time_ns())
    def stop_owned(proc):
        try:
            descendants = psutil.Process(proc.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            descendants = []
        for child in reversed(descendants):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    try:
        while queue or active:
            rss, io_read, io_write = psutil.Process().memory_info().rss, 0, 0
            for proc, _, _, _ in active.values():
                try:
                    tree = [psutil.Process(proc.pid), *psutil.Process(proc.pid).children(recursive=True)]
                    for p in tree:
                        try:
                            rss += p.memory_info().rss
                            io = p.io_counters()
                            io_read += io.read_bytes
                            io_write += io.write_bytes
                        except psutil.Error:
                            pass
                except psutil.Error:
                    pass
            mem = psutil.virtual_memory()
            # Global cache is an overestimate of this experiment's cache footprint.
            accounted = rss + mem.cached + mem.buffers
            peak = max(peak, accounted)
            samples.write(json.dumps({"s": time.monotonic() - started, "rss": rss,
                                      "global_file_cache": mem.cached + mem.buffers,
                                      "accounted": accounted, "io_read": io_read,
                                      "io_write": io_write, "workers": len(active)}) + "\n")
            samples.flush()
            if accounted >= 900_000_000_000:
                raise RuntimeError("900 GB abort watermark reached")
            for device in list(active):
                proc, log, task, candidate = active[device]
                if proc.poll() is None:
                    continue
                log.close()
                destination = args.out / task / str(candidate) / "result.json"
                result = json.loads(destination.read_text())["result"] if destination.exists() else {}
                outcomes.append({"task": task, "candidate": candidate, "exit": proc.returncode,
                                 "correctness": result.get("correctness_status"),
                                 "runtime": result.get("runtime_status")})
                record = next(r for r in records if r["task"] == task)
                if proc.returncode == 75:
                    queue.append((task, candidate))
                elif (result.get("runtime_status") != "ok" or result.get("correctness_status") != "ok"):
                    if candidate + 1 < len(record["candidates"]):
                        queue.append((task, candidate + 1))
                print(json.dumps(outcomes[-1]), flush=True)
                del active[device]
            if accounted < 800_000_000_000:
                now = time.monotonic()
                if queue and now - last_admission_check >= 5:
                    last_status = subprocess.check_output(["npu-smi", "info"], text=True)
                    last_admission_check = now
                for device in range(args.device, args.device + args.workers):
                    if device in active or not queue or not device_idle(last_status, device):
                        continue
                    task, candidate = queue.pop(0)
                    dest = args.out / task / str(candidate)
                    if (dest / "result.json").exists():
                        continue
                    dest.mkdir(parents=True, exist_ok=True)
                    old_log = dest / "run.log"
                    if old_log.exists():
                        old_log.rename(dest / f"run-before-{session}.log")
                    log = (dest / "run.log").open("w")
                    cmd = [sys.executable, str(Path(__file__).resolve()), "task", "--out", str(args.out),
                           "--task", task, "--candidate", str(candidate), "--device", str(device)]
                    if args.baseline_root:
                        cmd.extend(["--baseline-root", str(args.baseline_root)])
                    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True)
                    active[device] = proc, log, task, candidate
            if time.monotonic() - last_notice >= 30:
                print(json.dumps({"pending": len(queue), "active": len(active),
                                  "accounted_bytes": accounted}), flush=True)
                last_notice = time.monotonic()
            time.sleep(1)
    finally:
        for proc, log, _, _ in active.values():
            stop_owned(proc)
            log.close()
        samples.close()
        target = args.out / f"supervisor-{args.workers}.json"
        if target.exists():
            target.rename(args.out / f"supervisor-{args.workers}-before-{session}.json")
        target.write_text(json.dumps({
            "elapsed_s": time.monotonic() - started, "peak_accounted_bytes": peak,
            "cgroup_hard_limit": False, "outcomes": outcomes}, indent=2))


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["inventory", "task", "suite", "campaign"])
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--experiments", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--candidate", type=int, default=0)
    parser.add_argument("--workers", type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--baseline-root", type=Path)
    args = parser.parse_args()
    if args.baseline_root:
        ROOT = args.baseline_root.resolve()
        sys.path.insert(0, str(ROOT))
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.action == "campaign":
        for workers in (1, 2, 4, 8):
            args.workers = workers
            args.limit = workers if workers < 8 else 0
            suite(args)
    else:
        {"inventory": inventory, "task": run_task, "suite": suite}[args.action](args)


if __name__ == "__main__":
    main()
