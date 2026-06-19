#!/usr/bin/env python3
"""Reproducible shape/dtype feasibility report (Unit 1 of the shape-coverage-expansion plan).

The first plan draft asserted feasibility counts ("61 dtype-locked / 22 tight") derived from
``meta.dtype`` -- which is WRONG: e.g. L2_20's meta says ``float32`` but its kernel carries a
``configs = {torch.bfloat16: ..., torch.float16: ..., torch.float32: ...}`` dispatch dict, so the
KERNEL is multi-dtype capable. This script replaces those asserted numbers with a CHECKED-IN report
derived from the KERNEL SOURCE + ``param_constraints`` + launcher asserts, with the evidence cited
per task so every verdict is spot-checkable. The plan ships the report; it asserts no counts.

Classification (heuristic, evidence-backed -- NOT a proof):

  dtype_class:
    * ``multi_dtype``     -- kernel carries a dtype-dispatch config dict keyed by >=2 torch float
                             dtypes, OR references >=2 distinct float dtypes without a quant token.
                             Candidate for the Unit-4 dtype sweep.
    * ``single_dtype``    -- exactly one float dtype token; likely locked to it.
    * ``quant_locked``    -- fp8 / int8 / e4m3 / e5m2 quant tokens; dtype is not freely swappable.
    * ``unknown``         -- no clear dtype signal (manual check needed).

  shape_class (from param_constraints.dims + tensor_invariants + launcher asserts):
    * ``loose``   -- only lower/upper-bound dim constraints (``X >= 1``); free to scale.
    * ``medium``  -- a few structural constraints (one equality / divisibility / invariant).
    * ``tight``   -- many structural constraints (``%``, ``==`` across dims) or launcher asserts;
                     new shapes must satisfy a per-family valid lattice.

Run: ``python3 scripts/corpus/shape_feasibility_report.py --regen`` (writes the md below).
The report is deterministic: same corpus -> same bytes.
"""
from __future__ import annotations

import argparse
import ast
import glob
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = REPO / "docs" / "reviews" / "2026-06-15-shape-feasibility.md"

_FLOAT_DTYPE_TOKENS = ("bfloat16", "float16", "float32", "float64")
_QUANT_TOKENS = ("float8", "fp8", "e4m3", "e5m2", "int8", "uint8", "int4", "w8a8", "w4a8", "awq")


@dataclass
class TaskFeasibility:
    task_id: str
    level: str
    dtype_class: str = "unknown"
    dtype_evidence: list[str] = field(default_factory=list)
    shape_class: str = "loose"
    shape_evidence: list[str] = field(default_factory=list)
    n_declared_shapes: int = 0
    meta_dtype: str = ""
    has_launcher: bool = False
    n_launcher_asserts: int = 0


def _dtype_config_dict_dtypes(tree: ast.AST) -> set[str]:
    """Find Dict literals keyed by ``torch.<floatdtype>`` (the L2_20 dispatch-config pattern) and
    return the set of float-dtype names used as keys."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        key_dtypes: set[str] = set()
        for k in node.keys:
            if (
                isinstance(k, ast.Attribute)
                and isinstance(k.value, ast.Name)
                and k.value.id in ("torch", "tl")
                and k.attr in _FLOAT_DTYPE_TOKENS
            ):
                key_dtypes.add(k.attr)
        if len(key_dtypes) >= 2:
            found |= key_dtypes
    return found


def _classify_dtype(kernel_src: str) -> tuple[str, list[str]]:
    ev: list[str] = []
    try:
        tree = ast.parse(kernel_src)
    except SyntaxError:
        tree = None

    config_dtypes = _dtype_config_dict_dtypes(tree) if tree is not None else set()
    if config_dtypes:
        ev.append(f"dtype-dispatch config dict keyed by {sorted(config_dtypes)}")

    quant_hits = sorted({t for t in _QUANT_TOKENS if re.search(rf"\b{re.escape(t)}\b", kernel_src, re.I)})
    float_hits = sorted({
        t for t in _FLOAT_DTYPE_TOKENS
        if re.search(rf"(torch|tl)\.{t}\b", kernel_src)
    })
    if float_hits:
        ev.append(f"float dtype tokens: {float_hits}")
    if quant_hits:
        ev.append(f"quant tokens: {quant_hits}")

    # Quant takes PRECEDENCE over the multi-float-token rule: a quant kernel (w8a8/fp8/awq) carries
    # float tokens too (fp32 accumulator, fp16/bf16 output), but its INPUT dtype is locked to the
    # quant type -- it is NOT a free dtype-sweep candidate. Classifying it multi_dtype on the
    # incidental accumulator/output floats would be a false candidate.
    if quant_hits:
        return "quant_locked", ev
    if config_dtypes or len(float_hits) >= 2:
        return "multi_dtype", ev
    if len(float_hits) == 1:
        return "single_dtype", ev
    return "unknown", ev


def _classify_shape(meta: dict, launcher_src: str | None) -> tuple[str, list[str], int]:
    ev: list[str] = []
    pc = meta.get("param_constraints") or {}
    dims = pc.get("dims") or []
    invariants = pc.get("tensor_invariants") or []
    # structural constraints = equalities / divisibility (a per-family lattice), vs plain bounds.
    structural = [d for d in dims if isinstance(d, str) and ("%" in d or "==" in d)]
    bounds = [d for d in dims if isinstance(d, str) and d not in structural]
    if bounds:
        ev.append(f"{len(bounds)} bound dim constraint(s)")
    if structural:
        ev.append(f"{len(structural)} structural dim constraint(s): {structural[:4]}")
    if invariants:
        ev.append(f"{len(invariants)} tensor_invariant(s)")

    n_asserts = 0
    if launcher_src:
        n_asserts = len(re.findall(r"^\s*assert\b", launcher_src, re.M))
        if n_asserts:
            ev.append(f"{n_asserts} launcher assert(s)")

    score = 2 * len(structural) + len(invariants) + (1 if n_asserts >= 3 else 0)
    if score == 0:
        cls = "loose"
    elif score <= 2:
        cls = "medium"
    else:
        cls = "tight"
    return cls, ev, n_asserts


def _shape_ids(shapes_doc: dict) -> list[str]:
    from ascendbench.input_spec.shape_format import hidden_ids, public_ids
    return list(public_ids(shapes_doc)) + list(hidden_ids(shapes_doc))


def assess_task(task_dir: Path) -> TaskFeasibility:
    tid = task_dir.name
    level = task_dir.parent.name
    f = TaskFeasibility(task_id=tid, level=level)

    meta = yaml.safe_load((task_dir / "meta.yaml").read_text()) or {}
    f.meta_dtype = str((meta.get("param_constraints") or {}).get("dtypes") or meta.get("dtype") or "")

    kp = task_dir / "kernel.py"
    kernel_src = kp.read_text(errors="replace") if kp.is_file() else ""
    f.dtype_class, f.dtype_evidence = _classify_dtype(kernel_src)

    launcher = task_dir / "_pa_launcher.py"
    f.has_launcher = launcher.is_file()
    launcher_src = launcher.read_text(errors="replace") if f.has_launcher else None
    f.shape_class, f.shape_evidence, f.n_launcher_asserts = _classify_shape(meta, launcher_src)

    try:
        shapes_doc = yaml.safe_load((task_dir / "shapes.yaml").read_text()) or {}
        f.n_declared_shapes = len(_shape_ids(shapes_doc))
    except Exception:
        f.n_declared_shapes = 0
    return f


def build_report(results: list[TaskFeasibility]) -> str:
    from collections import Counter

    dt = Counter(r.dtype_class for r in results)
    sh = Counter(r.shape_class for r in results)
    lines: list[str] = [
        "---",
        "title: TAB shape/dtype feasibility (Unit 1 -- kernel-source-derived, reproducible)",
        "date: 2026-06-15",
        "tool: scripts/corpus/shape_feasibility_report.py --regen",
        "---",
        "",
        "# Shape / dtype feasibility report",
        "",
        "Per-task dtype- and shape-flexibility derived from the KERNEL SOURCE + `param_constraints`",
        "+ launcher asserts (NOT `meta.dtype`). Heuristic + evidence-backed: every verdict cites the",
        "tokens it matched so it is spot-checkable. Regenerate with `--regen` (deterministic).",
        "",
        f"Tasks: **{len(results)}**",
        "",
        "## dtype_class distribution",
        "",
        "| class | count | meaning |",
        "|---|---|---|",
        f"| `multi_dtype` | {dt.get('multi_dtype', 0)} | kernel dispatches on / accepts >=2 float dtypes -- Unit-4 dtype-sweep candidate |",
        f"| `single_dtype` | {dt.get('single_dtype', 0)} | one float dtype token; likely locked |",
        f"| `quant_locked` | {dt.get('quant_locked', 0)} | fp8/int8 quant; dtype not freely swappable |",
        f"| `unknown` | {dt.get('unknown', 0)} | no clear dtype signal (manual check) |",
        "",
        "## shape_class distribution",
        "",
        "| class | count | meaning |",
        "|---|---|---|",
        f"| `loose` | {sh.get('loose', 0)} | bound-only dim constraints; free to scale |",
        f"| `medium` | {sh.get('medium', 0)} | a few structural constraints |",
        f"| `tight` | {sh.get('tight', 0)} | many structural constraints / launcher asserts; needs a per-family lattice |",
        "",
        "> These counts are DERIVED (reproducible), not hand-asserted. The expansion (Unit 2/3/4)",
        "> validates every candidate shape by a real build_inputs + oracle dry-run regardless of class.",
        "",
        "## Per-task",
        "",
        "| task | level | dtype_class | shape_class | #shapes | launcher | evidence |",
        "|------|-------|-------------|-------------|---------|----------|----------|",
    ]
    for r in sorted(results, key=lambda x: x.task_id):
        ev = "; ".join(r.dtype_evidence + r.shape_evidence) or "-"
        ev = ev.replace("|", "\\|")
        lines.append(
            f"| {r.task_id} | {r.level} | {r.dtype_class} | {r.shape_class} | "
            f"{r.n_declared_shapes} | {'Y' if r.has_launcher else '-'} | {ev} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--regen", action="store_true", help="write the report to its canonical path")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="report output path")
    p.add_argument("--task", help="assess a single task id (prints to stdout, no write)")
    args = p.parse_args(argv)

    task_dirs = sorted(
        Path(mp).parent for mp in glob.glob(str(REPO / "tasks" / "level*" / "*" / "meta.yaml"))
    )
    if args.task:
        td = next((d for d in task_dirs if d.name == args.task), None)
        if td is None:
            print(f"task {args.task} not found")
            return 2
        r = assess_task(td)
        print(f"{r.task_id}: dtype={r.dtype_class} shape={r.shape_class} "
              f"shapes={r.n_declared_shapes} launcher={r.has_launcher}")
        print("  dtype evidence:", "; ".join(r.dtype_evidence) or "-")
        print("  shape evidence:", "; ".join(r.shape_evidence) or "-")
        return 0

    results = [assess_task(td) for td in task_dirs]
    report = build_report(results)
    if args.regen:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"wrote {args.out} ({len(results)} tasks)")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
