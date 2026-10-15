"""Reproduce failed seed-0 shapes through the original isolated runner."""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    status = subprocess.check_output(["npu-smi", "info"], text=True)
    if (f"No running processes found in NPU {args.device} " not in status or
            not re.search(rf"\|\s*{args.device}\s+910B\S*\s*\|\s*OK\s*\|", status)):
        raise RuntimeError("selected device is not healthy and idle")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.pop("ASCEND_DEVICE_ID", None)
    os.environ.pop("DEVICE_ID", None)
    sys.path.insert(0, str(args.baseline_root))
    import torch
    import yaml

    from ascendbench.correctness import tolerance
    from ascendbench.input_spec.shape_format import build_inputs_for
    from ascendbench.integrity import process_isolation as pi
    previous = json.loads(args.result.read_text())
    task_id = previous["result"]["task_id"]
    entry = next(r for r in json.loads(args.inventory.read_text()) if r["task"] == task_id)
    task = Path(entry["task_dir"])
    shapes = yaml.safe_load((task / "shapes.yaml").read_text())
    meta = yaml.safe_load((task / "meta.yaml").read_text())
    offline = meta.get("oracle_simplified") == "upstream_oracle_required"
    if not offline:
        spec = importlib.util.spec_from_file_location("_oracle_check", task / "oracle_pytorch.py")
        oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle)
    selected = [s["shape_id"] for s in previous["result"]["shape_results"]
                if not s["correctness_passed"]][:3]
    if not selected:
        raise RuntimeError("no failed correctness shapes")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "device-before.txt").write_text(status)
    launcher = args.out / "selected-python.sh"
    launcher.write_text("#!/bin/sh\nexport ASCEND_RT_VISIBLE_DEVICES=" + str(args.device)
                        + "\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
    launcher.chmod(0o700)
    pi.sys.executable = str(launcher)
    os.environ["HOME"] = str(args.out)
    solution = args.result.parent / "solution.py"
    if hashlib.sha256(solution.read_bytes()).hexdigest() != previous["candidate"]["sha256"]:
        raise RuntimeError("candidate hash mismatch")
    response = pi.run_candidate_isolated(solution, {
        sid: build_inputs_for(shapes, sid, seed=0) for sid in selected},
        backend="npu", pass_kind="correctness", timeout_s=180)
    policy = tolerance.load_policy()[meta["tolerance_class"]]
    findings = {}
    for sid in selected:
        rec = response.per_shape.get(sid)
        if offline:
            from safetensors.torch import load_file
            expected = load_file(str(task / "golden" / f"{sid}.safetensors"))
        else:
            expected = oracle.run(build_inputs_for(shapes, sid, seed=0))
        checks = {}
        for key, value in expected.items():
            if not torch.is_tensor(value):
                continue
            if rec is None or key not in rec.outputs:
                checks[key] = {"passed": False, "reason": response.load_error or "missing output"}
            else:
                compared = tolerance.compare(rec.outputs[key], value, policy,
                                             require_output_dtype_match=True)
                checks[key] = {"passed": compared.passed, "reason": compared.reason}
        findings[sid] = checks
    result = {"task": task_id, "candidate_sha256": previous["candidate"]["sha256"],
              "baseline_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=args.baseline_root, text=True).strip(),
              "command": sys.argv, "load_error": response.load_error,
              "reference_kind": "offline" if offline else "oracle",
              "scope": "diagnostic correctness-only subset; not formal coverage", "checks": findings}
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
