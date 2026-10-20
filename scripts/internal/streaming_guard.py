"""Aggregate memory guard for multiple owned experiment supervisors."""
import argparse
import json
import signal
import time

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pids", type=int, nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    roots = [psutil.Process(pid) for pid in args.pids]
    paused = set()
    started = time.monotonic()
    try:
        with open(args.out, "w") as output:
            while any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in roots):
                tree = {}
                for root in roots:
                    try:
                        if root.is_running():
                            tree[root.pid] = root
                            tree.update((p.pid, p) for p in root.children(recursive=True))
                    except psutil.Error:
                        pass
                rss = psutil.Process().memory_info().rss
                for proc in tree.values():
                    try:
                        rss += proc.memory_info().rss
                    except psutil.Error:
                        pass
                mem = psutil.virtual_memory()
                accounted = rss + mem.cached + mem.buffers
                output.write(json.dumps({"s": time.monotonic() - started, "rss": rss,
                                         "accounted": accounted, "global_file_cache": mem.cached + mem.buffers,
                                         "owned_processes": len(tree), "supervisors": args.pids}) + "\n")
                output.flush()
                if accounted >= 900_000_000_000:
                    for proc in reversed(list(tree.values())):
                        try:
                            proc.kill()
                        except psutil.Error:
                            pass
                    raise RuntimeError("aggregate 900 GB abort watermark reached")
                # Pausing supervisors stops admission; already running workers
                # keep making progress while the independent guard watches them.
                for root in roots:
                    try:
                        if accounted >= 800_000_000_000 and root.pid not in paused:
                            root.send_signal(signal.SIGSTOP)
                            paused.add(root.pid)
                        elif accounted < 800_000_000_000 and root.pid in paused:
                            root.send_signal(signal.SIGCONT)
                            paused.remove(root.pid)
                    except psutil.Error:
                        pass
                time.sleep(1)
    finally:
        for root in roots:
            if root.pid in paused:
                try:
                    root.send_signal(signal.SIGCONT)
                except psutil.Error:
                    pass


if __name__ == "__main__":
    main()
