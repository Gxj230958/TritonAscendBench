#!/usr/bin/env python3
"""U9 migration M2: flip a task's knob shapes.yaml to the COMPACT v2 explicit form.

Writes the schema-2 doc generate_compact_shapes() produces with a READABLE serialization -- each tensor input
on ONE line ``- {name: x, kind: tensor, shape: [4, 128], dtype: float32, axes: [B, N], dist: randn}`` and the
row ``env`` inline -- so a migrated shapes.yaml is compact (~320 lines for L1_01 vs the knob's 941 and the
block-style dump's 4443). The expand_compact()->v1 path the adapter reads is unchanged (gated in M1).

Rollback is ``git checkout -- <shapes.yaml>``. The committed v2 manifest differs from the knob manifest (a
real format change), so the daemon re-grade (S2/C2/golden/roofline) re-stamps each meta -- run it from the box.

Usage:
  migrate_to_v2.py --task tasks/level1/L1_01_bias [--write]   # default prints; --write replaces shapes.yaml
  migrate_to_v2.py --all [--write]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ascendbench.input_spec.compact_shapes import generate_compact_shapes  # noqa: E402


class _Flow(dict):
    """A mapping rendered in YAML FLOW style (``{a: 1, b: 2}``) -- used for each input + the row env."""


class _FlowList(list):
    """A sequence rendered in YAML FLOW style (``[1, 2]``) -- used for shape / axes."""


def _dumper():
    d = yaml.SafeDumper

    def _map(dumper, data):
        return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True)

    def _seq(dumper, data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    d.add_representer(_Flow, _map)
    d.add_representer(_FlowList, _seq)
    return d


def _flow_row(row: dict) -> dict:
    """Wrap a v2 row's leaf collections so they serialise on one line (env inline, each input inline)."""
    out: dict = {"id": row["id"]}
    if row.get("env") is not None:
        out["env"] = _Flow(row["env"])
    inputs = []
    for inp in row.get("inputs") or []:
        f = _Flow()
        for k, v in inp.items():
            f[k] = _FlowList(v) if isinstance(v, list) else v
        inputs.append(f)
    out["inputs"] = inputs
    return out


def to_yaml(v2_doc: dict) -> str:
    """Serialise a v2 doc with compact one-line-per-input rows."""
    doc = {
        "schema_version": v2_doc["schema_version"],
        "public": [_flow_row(r) for r in v2_doc.get("public") or []],
        "hidden": [_flow_row(r) for r in v2_doc.get("hidden") or []],
    }
    return yaml.dump(doc, Dumper=_dumper(), sort_keys=False, default_flow_style=False, width=4096)


def migrate(task_dir: Path, write: bool) -> int:
    sp = task_dir / "shapes.yaml"
    knob = yaml.safe_load(sp.read_text())
    if not isinstance(knob, dict) or not knob.get("size_knobs"):
        print(f"  skip {task_dir.name}: not a knob-format shapes.yaml", file=sys.stderr)
        return 0
    text = to_yaml(generate_compact_shapes(knob))
    if write:
        sp.write_text(text)
        print(f"  wrote {task_dir.name}/shapes.yaml ({len(text.splitlines())} lines)")
    else:
        print(text)
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Flip a knob shapes.yaml to the compact v2 explicit form.")
    ap.add_argument("--task", help="single task dir")
    ap.add_argument("--all", action="store_true", help="every task under tasks/")
    ap.add_argument("--write", action="store_true", help="replace shapes.yaml (default: print to stdout)")
    args = ap.parse_args(argv)
    if args.all:
        n = sum(migrate(sp.parent, args.write) for sp in sorted((REPO / "tasks").rglob("shapes.yaml")))
        print(f"migrated {n} task(s)", file=sys.stderr)
    elif args.task:
        migrate(Path(args.task), args.write)
    else:
        ap.error("pass --task <dir> or --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
