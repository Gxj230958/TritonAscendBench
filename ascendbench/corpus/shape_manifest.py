"""Canonical shape-manifest hash: the freshness signal for a task's declared shape set.

``phase_a_prime`` already pins ``kernel_sha256`` (a kernel.py byte change invalidates a stale S2/C2
PASS). The shape set had no equivalent: editing an EXISTING shape's knobs/dims/dtype in place (same
id-set) left ``equivalence_per_shape`` keys unchanged, so a stale PASS over the edited shape went
uncaught. This module's hash closes that gap -- ANY change to what ``build_inputs`` would materialise
for a declared shape (add/remove a shape row, edit a knob/dim/dtype/dist, change a size_knob default,
reorder the ``inputs`` mapping) changes the hash, invalidating the recorded PASS exactly like a
kernel byte change does.

Order matters for the ``inputs`` mapping (and nested struct ``fields``): ``build_inputs`` iterates
them in authored order with ONE seeded generator, and coupled dists / struct refs require the earlier
sibling to materialise FIRST. So those mappings are encoded as ordered key/value pair lists, NOT
sorted -- a reorder is a real semantic change and must change the hash. By-name knob mappings
(``size_knobs`` / per-shape ``knobs``) are consumed by name and stay plain dicts. Shape ROWS are
addressed by id, so they are sorted by (role, id) for a stable order. The hash is taken over the
PARSED-YAML projection, so comment / whitespace / quote-style churn does not invalidate it.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from ascendbench.input_spec.shape_format import (
    explicit_canonical_projection,
    is_explicit_shapes_doc,
)


def _ord(value: Any) -> Any:
    """Recursively order-preserve every mapping as a list of ``[key, value]`` pairs.

    Used for the ``inputs`` block, where the generator consumes inputs (and struct fields) in
    authored order with cross-sibling RNG/ref coupling. A list of pairs is immune to the
    ``sort_keys=True`` JSON dump, so authored order is preserved end to end. (A reorder of the
    by-name keys WITHIN a single tensor spec -- ``dims``/``dtype``/``dist`` -- also changes the hash;
    that is a harmless false-RED, never a false-GREEN: it can only force a re-verify, not hide a
    changed shape.)
    """
    if isinstance(value, dict):
        return [[k, _ord(v)] for k, v in value.items()]
    if isinstance(value, list):
        return [_ord(v) for v in value]
    return value


def canonical_manifest(shapes_doc: dict | None) -> dict:
    """Build the canonical (hash-input) projection of a shapes.yaml document.

    Dual-format (U8): an EXPLICIT-format doc projects via the adapter's full-row projection (so any
    semantic edit moves the hash); a knob-format doc keeps the projection below BYTE-IDENTICAL (the
    explicit branch is gated on the strict discriminator, which returns False for every knob doc -- the
    all-101 re-hash == HEAD is the manifest-neutrality gate).
    """
    d = shapes_doc if isinstance(shapes_doc, dict) else {}
    if is_explicit_shapes_doc(d):
        return explicit_canonical_projection(d)
    shapes: list[dict] = []
    for role in ("public", "hidden"):
        for s in (d.get(f"{role}_shapes") or []):
            if isinstance(s, dict) and s.get("id"):
                shapes.append(
                    {
                        "role": role,
                        "id": str(s["id"]),
                        # per-shape dtype override (Unit-4 dtype sweep); None today
                        "dtype": s.get("dtype"),
                        # by-name knobs -> plain dict (sort_keys handles it)
                        "knobs": s.get("knobs") or {},
                    }
                )
    # Shape rows are addressed by id -> a stable (role, id) order is canonical.
    shapes.sort(key=lambda x: (x["role"], x["id"]))
    return {
        # ORDER-PRESERVED: input declaration order drives RNG draw order + ref resolution.
        "inputs": _ord(d.get("inputs") or {}),
        # by-name knob defaults -> plain dict (sort_keys handles it).
        "size_knobs": d.get("size_knobs") or {},
        "shapes": shapes,
    }


def shape_manifest_sha256(shapes_doc: dict | None) -> str:
    """sha256 of the canonical declared-shape manifest (see module docstring)."""
    blob = json.dumps(
        canonical_manifest(shapes_doc),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def shape_manifest_sha256_for_task(task_dir) -> str | None:
    """Convenience: load ``task_dir/shapes.yaml`` and hash it. Returns None on read/parse error
    (callers treat None as "cannot determine" -- conservative, never a spurious match)."""
    from pathlib import Path

    import yaml

    p = Path(task_dir) / "shapes.yaml"
    try:
        doc = yaml.safe_load(p.read_text())
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(doc, dict):
        return None
    return shape_manifest_sha256(doc)


def _backfill_cli(argv: list[str] | None = None) -> int:
    """One-time backfill: stamp ``shape_manifest_sha256`` (from the CURRENT shapes.yaml) into every
    task's ``phase_a_prime`` block, and into ``phase_b_capture`` where its status is DONE.

    The current shapes.yaml IS the shape set the existing green S2/C2/golden PASS was verified over,
    so this is a truthful backfill (NOT a tune-to-pass): it just makes the freshness gate active for
    the whole corpus, so a LATER in-place shape edit flips the task RED until re-verify. Idempotent.
    """
    import argparse
    import glob
    from pathlib import Path

    import yaml

    repo = Path(__file__).resolve().parent.parent.parent
    ap = argparse.ArgumentParser(description="backfill phase_a_prime/phase_b_capture shape_manifest_sha256")
    ap.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = ap.parse_args(argv)

    changed = 0
    for mp in sorted(glob.glob(str(repo / "tasks" / "level*" / "*" / "meta.yaml"))):
        td = Path(mp).parent
        manifest = shape_manifest_sha256_for_task(td)
        if not manifest:
            print(f"  SKIP {td.name}: shapes.yaml unreadable")
            continue
        meta = yaml.safe_load(Path(mp).read_text()) or {}
        dirty = False
        pap = meta.get("phase_a_prime")
        if isinstance(pap, dict) and pap.get("shape_manifest_sha256") != manifest:
            pap["shape_manifest_sha256"] = manifest
            dirty = True
        pbc = meta.get("phase_b_capture")
        if (
            isinstance(pbc, dict)
            and pbc.get("status") == "DONE"
            and pbc.get("shape_manifest_sha256") != manifest
        ):
            pbc["shape_manifest_sha256"] = manifest
            dirty = True
        if dirty:
            changed += 1
            print(f"  {'WOULD UPDATE' if args.dry_run else 'UPDATE'} {td.name} -> {manifest[:12]}")
            if not args.dry_run:
                tmp = Path(mp).with_suffix(".yaml.tmp")
                tmp.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
                tmp.replace(Path(mp))
    print(f"{'(dry-run) ' if args.dry_run else ''}backfilled {changed} task meta(s)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_backfill_cli())
