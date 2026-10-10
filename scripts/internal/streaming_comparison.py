"""Controlled CPU lifecycle comparison, not a kernel performance benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=["main", "improvement-1", "streaming"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shapes", type=int, default=4)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--elements", type=int, default=1048576)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))
    import psutil
    import torch
    from ascendbench.integrity import process_isolation as pi
    pi.sys.executable = str(Path(__file__).with_name("cpu_only_python.sh"))
    torch.set_num_threads(1)
    stats = {"peak_tree_rss": 0, "peak_temporary_bytes": 0, "peak_global_file_cache": 0,
             "peak_fds": 0, "peak_tree_write_bytes": 0}
    done = threading.Event()
    with tempfile.TemporaryDirectory(prefix="tab-comparison-") as directory:
        root = Path(directory)
        tempfile.tempdir = directory
        marker = root / "first-call"
        solution = root / "solution.py"
        solution.write_text(
            "import torch, time\n_seen = False\n"
            "def npu_triton_impl(inputs):\n"
            "    global _seen\n"
            "    if not _seen:\n"
            "        try:\n"
            "            with open(inputs['marker'], 'x') as f: f.write(str(time.monotonic()))\n"
            "        except FileExistsError: pass\n"
            "        _seen = True\n"
            "    return {'y': inputs['x'] + 1}\n")
        def monitor():
            while not done.wait(0.05):
                tree = [psutil.Process(), *psutil.Process().children(recursive=True)]
                rss = fds = written = 0
                for p in tree:
                    try:
                        rss += p.memory_info().rss
                        fds += p.num_fds()
                        written += p.io_counters().write_bytes
                    except psutil.Error:
                        pass
                stats["peak_tree_rss"] = max(stats["peak_tree_rss"], rss)
                stats["peak_fds"] = max(stats["peak_fds"], fds)
                stats["peak_tree_write_bytes"] = max(stats["peak_tree_write_bytes"], written)
                stats["peak_global_file_cache"] = max(stats["peak_global_file_cache"],
                                                       psutil.virtual_memory().cached)
                size = 0
                for path in root.rglob("*"):
                    try:
                        if path.is_file():
                            size += path.stat().st_size
                    except FileNotFoundError:
                        pass
                stats["peak_temporary_bytes"] = max(stats["peak_temporary_bytes"], size)
        watcher = threading.Thread(target=monitor)
        watcher.start()
        started = time.monotonic()
        def build(sid, seed):
            return {"x": torch.randn(args.elements, generator=torch.Generator().manual_seed(seed)),
                    "marker": str(marker)}
        def trial_seed(i):
            return 1 + i if i < 2 else 1000 + i - 2
        ids = tuple(f"s{i}" for i in range(args.shapes))
        try:
            if args.kind == "streaming":
                from ascendbench.integrity.streaming import InputSource, run_streaming_two_pass
                checked = []
                def consume(phase, sid, trial, seed, output):
                    torch.testing.assert_close(output["y"], build(sid, seed)["x"] + 1)
                    checked.append((phase, sid))
                two = run_streaming_two_pass(solution, InputSource(ids, build, 2, args.trials), consume,
                                            backend="gpu", timeout_s=300, sample_size=3)
                assert two.correctness.ran_any and two.timing.ran_any, (two.correctness, two.timing)
                timed = two.timing
            else:
                if args.kind == "main":
                    correctness = {sid: build(sid, 0) for sid in ids}
                    timing = {sid: [build(sid, trial_seed(i)) for i in range(args.trials + 2)] for sid in ids}
                else:
                    correctness = {sid: pi.LazyInput(lambda sid=sid: build(sid, 0)) for sid in ids}
                    timing = {sid: pi.LazyTrials(args.trials + 2, lambda i, sid=sid: build(sid, trial_seed(i)))
                              for sid in ids}
                # No global scrub: legacy cleanup targets are shared with other users.
                corr = pi.run_candidate_isolated(solution, correctness, backend="gpu", timeout_s=300)
                timed = pi.run_candidate_isolated(solution, timing, backend="gpu", timeout_s=300,
                                                  pass_kind="timing", n_warmup=2,
                                                  sampled_indices={sid: [0, 1, 2] for sid in ids})
                assert corr.ran_any and timed.ran_any, (corr, timed)
            stats["elapsed_s"] = time.monotonic() - started
            stats["first_call_s"] = float(marker.read_text()) - started
            stats["synthetic_host_ns"] = {sid: rec.durations for sid, rec in timed.per_shape.items()}
        finally:
            done.set()
            watcher.join()
        stats.update(kind=args.kind, shapes=args.shapes, trials=args.trials, elements=args.elements,
                     commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                    cwd=args.root, text=True).strip())
        stats["code_sha256"] = {str(path.relative_to(args.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (args.root / "ascendbench/integrity").glob("*.py")}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats))


if __name__ == "__main__":
    main()
