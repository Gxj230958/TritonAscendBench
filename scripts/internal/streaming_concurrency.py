"""Repeat one frozen full task at 1/2/4/8 workers with owned-process accounting."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", type=int, default=8)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    entry = next(r for r in inventory if r["task"] == args.task)
    args.out.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for count in (1, 2, 4, 8):
        cohort = args.out / str(count)
        cohort.mkdir(exist_ok=True)
        owned = []
        waiting = time.monotonic()
        last_notice = 0
        while True:
            status = subprocess.check_output(["npu-smi", "info"], text=True)
            if all(f"No running processes found in NPU {device} " in status
                   and re.search(rf"\|\s*{device}\s+910B\S*\s*\|\s*OK\s*\|", status)
                   for device in range(args.device, args.device + count)):
                (cohort / "devices-before.txt").write_text(status)
                break
            elapsed = time.monotonic() - waiting
            if elapsed > 3600:
                raise RuntimeError("selected cohort did not become healthy and idle")
            if elapsed - last_notice >= 30:
                print(json.dumps({"waiting_workers": count, "wait_s": elapsed}), flush=True)
                last_notice = elapsed
            time.sleep(5)
        wait_s = time.monotonic() - waiting
        started = time.monotonic()
        peak = 0
        try:
            with (cohort / "resources.jsonl").open("w") as samples:
                launched = 0
                while launched < count or any(p.poll() is None for p, _ in owned):
                    rss = psutil.Process().memory_info().rss
                    for proc, _ in owned:
                        if proc.poll() is not None:
                            continue
                        try:
                            tree = [psutil.Process(proc.pid), *psutil.Process(proc.pid).children(recursive=True)]
                            for child in tree:
                                try:
                                    rss += child.memory_info().rss
                                except psutil.Error:
                                    pass
                        except psutil.Error:
                            pass
                    mem = psutil.virtual_memory()
                    accounted = rss + mem.cached + mem.buffers
                    peak = max(peak, accounted)
                    samples.write(json.dumps({"s": time.monotonic() - started,
                                              "rss": rss, "accounted": accounted,
                                              "global_file_cache": mem.cached + mem.buffers}) + "\n")
                    samples.flush()
                    if accounted >= 900_000_000_000:
                        raise RuntimeError("900 GB abort watermark reached")
                    if launched < count and accounted < 800_000_000_000:
                        root = cohort / str(launched)
                        root.mkdir(exist_ok=True)
                        (root / "inventory.json").write_text(json.dumps([entry], indent=2))
                        log = (root / "run.log").open("w")
                        proc = subprocess.Popen([
                            sys.executable, str(Path(__file__).with_name("streaming_validation.py")),
                            "task", "--out", str(root), "--task", args.task,
                            "--device", str(args.device + launched)],
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        owned.append((proc, log))
                        launched += 1
                    time.sleep(1)
        finally:
            for proc, log in owned:
                if proc.poll() is None:
                    try:
                        for child in reversed(psutil.Process(proc.pid).children(recursive=True)):
                            try:
                                child.kill()
                            except psutil.Error:
                                pass
                    except psutil.Error:
                        pass
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                log.close()
        outcome = {"workers": count, "elapsed_s": time.monotonic() - started,
                   "wait_for_idle_s": wait_s, "peak_accounted_bytes": peak,
                   "exits": [p.returncode for p, _ in owned]}
        outcomes.append(outcome)
        (args.out / "supervisor.json").write_text(json.dumps(outcomes, indent=2))
        print(json.dumps(outcome), flush=True)
        if any(outcome["exits"]):
            raise RuntimeError("cohort did not complete; inspect per-worker logs")


if __name__ == "__main__":
    main()
