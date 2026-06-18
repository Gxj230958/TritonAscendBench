#!/usr/bin/env python3
"""Phase-2 U6: lint explicit shape rows -- axis-link consistency + per-tensor validity.

Axis-link semantics are STRICT (codex-vetted), with no loose "shared-name OR declared-link" ambiguity:

  1. Same unqualified axis NAME across tensors in the SAME row => MUST be equal extent. Axis names are
     canonical row-environment symbols, so two axes that can legitimately differ must NOT share a name
     (otherwise the later cost-model re-base gets an ambiguous symbol). This holds by construction in the
     knob era (an axis name is a dim expression and knobs are row-global), so a violation means a
     hand-authored row broke the invariant.
  2. A declared ``axis_links`` entry (e.g. ``q.head_dim == k.head_dim``) enforces equality between
     DIFFERENTLY named/qualified axes -- the only way to link axes that do not share a name.

Per tensor it also enforces ``len(axes) == len(shape)`` and a registered dtype/dist via the U5 typed IR
(:class:`row_ir.TensorSpec`).

Usage: ``lint_explicit_shapes.py --task tasks/level1/L1_01_bias`` (exit 0 = clean, 1 = violations).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ascendbench.input_spec import row_ir  # noqa: E402
from scripts.corpus.gen_explicit_shapes import generate_explicit_shapes  # noqa: E402


def _qualified(link: str) -> tuple[str, str]:
    """Parse a ``q.head_dim == k.head_dim`` axis-link into its two ``tensor.axis`` operands."""
    lhs, _, rhs = link.partition("==")
    return lhs.strip(), rhs.strip()


def _axis_extent(row: dict, qualified: str) -> int | None:
    """Resolve ``tensor.axis`` to its extent in a row, or None if absent."""
    tname, _, axis = qualified.partition(".")
    for inp in row["inputs"]:
        if inp["name"] == tname and inp["kind"] == "tensor":
            for ax, ext in zip(inp["axes"], inp["shape"], strict=False):
                if ax == axis:
                    return ext
    return None


def lint_row(row: dict, axis_links: list[str] | None = None) -> list[str]:
    """Return a list of violation strings for one explicit row (empty = clean)."""
    out: list[str] = []
    name_to_extent: dict[str, int] = {}
    for inp in row["inputs"]:
        if inp["kind"] != "tensor":
            continue
        # per-tensor validity via the typed IR (rank match + registered dtype/dist)
        try:
            row_ir.TensorSpec(shape=inp["shape"], dtype=inp["dtype"], axes=inp["axes"],
                              dist=inp["dist"]).validate()
        except Exception as e:   # surface any IR validation failure as a lint violation
            out.append(f"row {row['id']} tensor {inp['name']}: {e}")
        # rule 1: a shared axis NAME must carry a single extent within the row
        for ax, ext in zip(inp["axes"], inp["shape"], strict=False):
            if ax in name_to_extent and name_to_extent[ax] != ext:
                out.append(f"row {row['id']}: axis {ax!r} has conflicting extents "
                           f"{name_to_extent[ax]} vs {ext} (input {inp['name']})")
            else:
                name_to_extent.setdefault(ax, ext)
    # rule 2: declared cross-name links must hold
    for link in (axis_links or []):
        lhs, rhs = _qualified(link)
        le, re_ = _axis_extent(row, lhs), _axis_extent(row, rhs)
        if le is None or re_ is None:
            out.append(f"row {row['id']}: axis_link {link!r} references an unknown tensor.axis")
        elif le != re_:
            out.append(f"row {row['id']}: axis_link {link!r} violated ({le} != {re_})")
    return out


def lint_task(explicit_doc: dict, axis_links: list[str] | None = None) -> list[str]:
    """Lint every public + hidden row of a task's explicit doc; return all violations."""
    violations: list[str] = []
    for row in (explicit_doc.get("public") or []) + (explicit_doc.get("hidden") or []):
        violations.extend(lint_row(row, axis_links))
    return violations


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Lint explicit shape rows.")
    ap.add_argument("--task", required=True, help="task dir containing shapes.yaml")
    args = ap.parse_args(argv)
    doc = generate_explicit_shapes(yaml.safe_load((Path(args.task) / "shapes.yaml").read_text()))
    violations = lint_task(doc)
    if violations:
        for v in violations:
            print(f"FAIL: {v}")
        return 1
    print("OK: explicit rows lint clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
