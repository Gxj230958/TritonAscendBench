#!/usr/bin/env python3
"""Phase-1 down-payment: annotate each shapes.yaml with a generated, non-authoritative `explicit_view`
+ per-row `provenance`, WITHOUT changing the manifest hash (no re-grade).

`canonical_manifest` hashes only `inputs` + `size_knobs` + per-row `id`/`dtype`/`knobs`; an extra
top-level `explicit_view:` key is ignored, so adding it is manifest-neutral. The view is a STRUCTURED
field (not a comment) because the patch/rewrite path (`file_blocks.py`) does not preserve comments.
The view mirrors `build_inputs` via the shared materializer, giving an at-a-glance, decoder-free read
of every shape row while the authoritative format stays the knob grid until the Phase-2 cutover.

Usage: `annotate_explicit_view.py --task tasks/level1/L1_01_bias [--apply]` (default = dry-run +
manifest-neutral check).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ascendbench.corpus.shape_manifest import shape_manifest_sha256  # noqa: E402
from ascendbench.input_spec.inputs import shape_ids  # noqa: E402
from ascendbench.input_spec.materialize import materialize_row  # noqa: E402

_BASELINE_NAMES = {"tiny", "small", "medium", "large", "big", "huge"}
_EDGE_HINTS = ("edge", "aspect", "singleton", "all_min", "all_max", "min", "max")


def _provenance(row: dict) -> str:
    """Best-effort Phase-1 provenance tag from the row name (refined later from generation metadata)."""
    name = str(row.get("name") or "").lower()
    if any(h in name for h in _EDGE_HINTS):
        return "edge"
    if name in _BASELINE_NAMES:
        return "baseline"
    return "synthetic"


def _view_entry(shapes_doc: dict, sid: str, row: dict) -> dict:
    desc = materialize_row(shapes_doc, sid)
    tensors, params, structs = {}, {}, {}
    for name, d in desc.items():
        if d["kind"] == "tensor":
            dist = d["dist"]
            # keep the FULL dist (kind + args), not just the kind, so the view stays a faithful
            # decoder-free descriptor (e.g. offsets:to=E*L, randint:0:E, randn_inf_excl:<ref>).
            dist_repr = dist["kind"] if not dist.get("args") else {"kind": dist["kind"], "args": dist["args"]}
            tensors[name] = {"shape": d["shape"], "dtype": d["dtype"], "axes": d["axes"], "dist": dist_repr}
        elif d["kind"] == "struct":
            structs[name] = d.get("struct")
        elif d["kind"] == "optional":
            params[name] = None
        else:
            params[name] = d.get("value")
    entry: dict = {"provenance": _provenance(row), "tensors": tensors}
    if params:
        entry["params"] = params
    if structs:
        entry["structs"] = structs
    return entry


def _strip_existing_view(raw: str) -> str:
    """Remove a previously-appended top-level ``explicit_view:`` block (idempotent re-apply). The
    block is always the LAST top-level key (we append it), so truncating at its column-0 line is safe."""
    lines = raw.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if ln.startswith("explicit_view:"):
            return "".join(lines[:i])
    return raw


def annotate(task_dir: Path, apply: bool) -> dict:
    sp = task_dir / "shapes.yaml"
    raw = sp.read_text()
    doc = yaml.safe_load(raw)
    before = shape_manifest_sha256(doc)
    pub, hid = shape_ids(doc)
    rows = {r["id"]: r for r in (doc.get("public_shapes") or []) + (doc.get("hidden_shapes") or [])}
    view = {sid: _view_entry(doc, sid, rows[sid]) for sid in pub + hid}

    # BYTE-PRESERVING append: keep the existing file bytes (comments included) verbatim and append a
    # top-level `explicit_view:` block, rather than a full yaml round-trip (which would drop comments
    # and reformat every file). `explicit_view` is not in canonical_manifest, so the hash is unchanged.
    base = _strip_existing_view(raw).rstrip("\n")
    block = yaml.safe_dump({"explicit_view": view}, sort_keys=False, allow_unicode=True)
    new_text = base + "\n\n" + block

    new_doc = yaml.safe_load(new_text)                 # parse-back: must stay valid YAML
    after = shape_manifest_sha256(new_doc)
    manifest_neutral = (before == after and new_doc.get("explicit_view") == view)

    if apply and manifest_neutral:
        sp.write_text(new_text)
    return {"task": task_dir.name, "rows": len(view),
            "manifest_neutral": manifest_neutral, "before": before[:12], "after": after[:12]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, help="path to tasks/<lvl>/<task>")
    ap.add_argument("--apply", action="store_true", help="write (default: dry-run)")
    args = ap.parse_args()
    rep = annotate(Path(args.task).resolve(), args.apply)
    print(json.dumps(rep, indent=1))
    if not rep["manifest_neutral"]:
        print("ERROR: annotation changed the manifest hash -- refusing to apply", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
