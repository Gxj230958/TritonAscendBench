#!/usr/bin/env python3
"""Lint shapes.yaml against known host-wrapper / Triton constexpr contracts.

This is intentionally narrow and fail-closed on the patterns we have
already seen in Phase A':

* int8 ``tl.dot`` kernels must not launch with ``BLOCK_SIZE_K < 32``;
* the common W8A8 block-matmul family must keep ``As`` / ``Bs`` group
  dimensions consistent with ``block_size``.

The script is standalone so it can be run in CI or by operators before
GPU verify:

    python3 scripts/lint_shapes_against_kernel.py
    python3 scripts/lint_shapes_against_kernel.py --task L2_17_w8a8_block_int8_matmul
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = REPO_ROOT / "tasks"

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triton_ascend_bench.inputs import build_inputs  # noqa: E402


@dataclass
class Finding:
    task_id: str
    shape_id: str
    code: str
    detail: str


def _task_dirs(task_id: str | None) -> list[Path]:
    if task_id:
        matches = list(TASKS_ROOT.glob(f"**/{task_id}"))
        return [p for p in matches if p.is_dir()]
    return sorted(p.parent for p in TASKS_ROOT.glob("**/kernel.py"))


def _shape_ids(shapes_doc: dict) -> list[str]:
    ids: list[str] = []
    for section in ("public_shapes", "hidden_shapes"):
        for row in shapes_doc.get(section) or []:
            sid = row.get("id")
            if sid:
                ids.append(str(sid))
    return ids


def lint_task(task_dir: Path) -> list[Finding]:
    kernel_path = task_dir / "kernel.py"
    shapes_path = task_dir / "shapes.yaml"
    if not kernel_path.is_file() or not shapes_path.is_file():
        return []

    kernel_src = kernel_path.read_text(encoding="utf-8", errors="replace")
    shapes_doc = yaml.safe_load(shapes_path.read_text()) or {}
    inputs = shapes_doc.get("inputs") or {}
    task_id = task_dir.name
    findings: list[Finding] = []

    block_size_spec = inputs.get("block_size")
    block_size = (
        block_size_spec.get("scalar_seq")
        if isinstance(block_size_spec, dict)
        else None
    )

    # Pattern 1: int8 tl.dot compile-time K contract.
    if "tl.dot(" in kernel_src and isinstance(block_size, list) and len(block_size) >= 2:
        block_k = int(block_size[1])
        int8_inputs = [
            name for name, spec in inputs.items()
            if isinstance(spec, dict) and spec.get("dtype") == "int8"
        ]
        if int8_inputs and block_k < 32:
            findings.append(
                Finding(
                    task_id=task_id,
                    shape_id="*",
                    code="int8_tl_dot_block_k_lt_32",
                    detail=(
                        f"tl.dot is present, int8 inputs={int8_inputs}, but "
                        f"block_size[1]={block_k} < 32"
                    ),
                )
            )

    # Pattern 2: W8A8 block-matmul group-shape contract.
    required = {"A", "B", "As", "Bs", "block_size"}
    if required.issubset(inputs):
        for sid in _shape_ids(shapes_doc):
            built = build_inputs(shapes_doc, sid, seed=0)
            block_n, block_k = built["block_size"]
            a = built["A"]
            b = built["B"]
            as_ = built["As"]
            bs = built["Bs"]

            expected_as_k = math.ceil(a.shape[-1] / block_k)
            expected_bs_n = math.ceil(b.shape[0] / block_n)
            expected_bs_k = math.ceil(b.shape[-1] / block_k)

            if as_.shape[-1] != expected_as_k:
                findings.append(
                    Finding(
                        task_id=task_id,
                        shape_id=sid,
                        code="w8a8_as_group_mismatch",
                        detail=(
                            f"As.shape[-1]={as_.shape[-1]} but "
                            f"ceil(A.K/block_k)={expected_as_k} "
                            f"(A.shape={tuple(a.shape)}, block_k={block_k})"
                        ),
                    )
                )
            if bs.shape[0] != expected_bs_n or bs.shape[1] != expected_bs_k:
                findings.append(
                    Finding(
                        task_id=task_id,
                        shape_id=sid,
                        code="w8a8_bs_group_mismatch",
                        detail=(
                            f"Bs.shape={tuple(bs.shape)} but expected "
                            f"({expected_bs_n}, {expected_bs_k}) from "
                            f"B.shape={tuple(b.shape)} and block_size="
                            f"{tuple(built['block_size'])}"
                        ),
                    )
                )

    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", help="restrict to one task id")
    args = ap.parse_args(argv)

    findings: list[Finding] = []
    for task_dir in _task_dirs(args.task):
        findings.extend(lint_task(task_dir))

    if findings:
        print(json.dumps([asdict(f) for f in findings], indent=2))
        return 1
    print("[]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
