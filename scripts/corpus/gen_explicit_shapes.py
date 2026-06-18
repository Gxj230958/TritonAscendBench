#!/usr/bin/env python3
"""Phase-2 U6: generate EXPLICIT resolved-literal shape rows from a task's shapes.yaml.

This is the real generator the explicit format rides on -- it emits explicit rows (ordered inputs with
literal shapes + declared dtype + named axes + resolved dist args), not a knob grid. Every row produced
here reproduces ``build_inputs`` byte-for-byte (proven by the U6 equivalence gate / ``build_from_explicit``),
so it is the safe source for U9's per-task migration.

Scope boundary (codex-vetted): U6 RENDERS the sweep output that ``extend_task_shapes.py`` already
enumerated into the explicit format -- it does NOT re-derive the value-ladder / coprime-grid / edge sweep
from a hand-authored per-op spec (that authoring surface + the 101-task migration are U9), and it does NOT
rewrite the format-reading consumers (build_inputs / IPC / manifest / sandbox -- that is U8). It is a
library (``generate_explicit_shapes``) plus a dump CLI for inspection.

Usage: ``gen_explicit_shapes.py --task tasks/level1/L1_01_bias`` (prints the explicit doc as YAML).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ascendbench.input_spec.explicit_build import explicit_row  # noqa: E402
from ascendbench.input_spec.inputs import shape_ids  # noqa: E402
from ascendbench.input_spec.row_ir import SCHEMA_VERSION  # noqa: E402


def generate_explicit_shapes(shapes_doc: dict) -> dict:
    """Return the explicit-format doc for a task: ``{schema_version, public: [...], hidden: [...]}``.

    Each row is ``{id, role, inputs: [...]}`` from :func:`explicit_build.explicit_row`, with ``role``
    distinguishing public from hidden (the hidden rows stay sealed from the candidate sandbox; the
    sealing itself is enforced by the consumers in U8, not here).
    """
    pub_ids, hid_ids = shape_ids(shapes_doc)
    public = [{**explicit_row(shapes_doc, sid), "role": "public"} for sid in pub_ids]
    hidden = [{**explicit_row(shapes_doc, sid), "role": "hidden"} for sid in hid_ids]
    return {"schema_version": SCHEMA_VERSION, "public": public, "hidden": hidden}


def _load(task_dir: Path) -> dict:
    return yaml.safe_load((task_dir / "shapes.yaml").read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate explicit resolved-literal shape rows.")
    ap.add_argument("--task", required=True, help="task dir containing shapes.yaml")
    args = ap.parse_args(argv)
    doc = generate_explicit_shapes(_load(Path(args.task)))
    print(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
