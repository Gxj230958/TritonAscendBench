#!/usr/bin/env python3
"""Corpus self-consistency gate (authoring-side).

The hard gate that prevents the metadata-drift class that accumulated before it
existed: stale severed-kernel hashes on a changed kernel.py, declared
param_constraints drifting from the materialized fixtures, C2 equivalence records
drifting from the shape set, and C2 promotion without a fresh S2. It asserts that
the TRACKED metadata (provenance.golden verdict, phase_a_prime hashes, shape
coverage, ceiling status) is internally consistent with the on-disk kernel.py +
shapes.yaml -- i.e. that a task's metadata does not lie about itself.

Authoring-side scope: it does NOT require the gitignored, GPU-host offline
``golden/<shape>.safetensors`` reference artifacts to be present here (the
separate release gate asserts those via ``golden_complete``).

It asserts the MEANINGFUL invariant, not a brittle hardcoded count: every
non-deprecated task is precision-aligned (VERIFIED strict-pass OR allowlisted
CEILING_ACCEPTED dtype-ceiling), and the deprecated tasks are exactly the
known-upstream-bug ones. The strict-pass-vs-dtype-ceiling split is REPORTED (it
is just where the sub-ULP-tight strict bar happens to fall), never asserted as a
fixed number.

Run: python3 scripts/corpus/validate_corpus_consistency.py [--quiet]
Exit 0 iff zero violations.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ""


def _ceiling_allowlist() -> set[str]:
    cfg = yaml.safe_load((REPO / "ascendbench" / "policies" / "ceiling_acceptance.yaml").read_text()) or {}
    names: set[str] = set(cfg.get("tol_units_tasks") or [])
    names |= set((cfg.get("physical_tasks") or {}).keys())
    return names


def _shape_ids(shapes_doc: dict) -> set[str]:
    out: set[str] = set()
    for grp in ("public_shapes", "hidden_shapes"):
        for s in shapes_doc.get(grp) or []:
            if s.get("id"):
                out.add(s["id"])
    return out


def check_task(task_dir: Path, allowlist: set[str]) -> tuple[str, list[str]]:
    """Return (oke, violations) for one task."""
    tid = task_dir.name
    v: list[str] = []
    meta = yaml.safe_load((task_dir / "meta.yaml").read_text()) or {}
    g = (meta.get("provenance") or {}).get("golden") or {}
    pap = meta.get("phase_a_prime") or {}
    up = meta.get("upstream") or {}
    oke = g.get("oracle_kernel_equivalence")
    ksha = _sha256(task_dir / "kernel.py")
    poc = bool(up.get("hand_curated_poc"))

    if oke not in ("VERIFIED", "CEILING_ACCEPTED", "BLOCKED_UPSTREAM_BUG"):
        v.append(f"oracle_kernel_equivalence={oke!r} not in the aligned/blocked set")
        return oke or "MISSING", v

    # equivalence_per_shape must cover the current shape set (no stale 2-anchor record).
    if oke in ("VERIFIED", "CEILING_ACCEPTED"):
        try:
            want = _shape_ids(yaml.safe_load((task_dir / "shapes.yaml").read_text()) or {})
        except Exception as e:
            v.append(f"shapes.yaml unreadable: {e}")
            want = set()
        got = set((g.get("equivalence_per_shape") or {}).keys())
        if want and want != got:
            v.append(f"equivalence_per_shape {sorted(got)} != shapes {sorted(want)} (stale/partial grade)")

    # standalone hash must match the on-disk kernel.py (no stale sliced hash on a changed file).
    sa = pap.get("standalone_sha256")
    if sa and not any(p in str(sa) for p in ("placeholder", "scaffold", "TODO", "PENDING", "no-")):
        if sa != ksha:
            v.append(f"phase_a_prime.standalone_sha256 stale (meta {str(sa)[:12]} != kernel.py {ksha[:12]})")

    if oke == "VERIFIED":
        if g.get("golden_source") != "oracle_pytorch_verified":
            v.append(f"VERIFIED golden_source={g.get('golden_source')!r} (want oracle_pytorch_verified)")
        if not g.get("golden_sha256"):
            v.append("VERIFIED missing golden_sha256")
        ks = pap.get("kernel_sha256")
        if not poc and ks and ks != ksha:
            v.append(f"VERIFIED S2 not fresh: kernel_sha256 {str(ks)[:12]} != kernel.py {ksha[:12]}")
    elif oke == "CEILING_ACCEPTED":
        if tid not in allowlist:
            v.append(f"CEILING_ACCEPTED but {tid} not on the ceiling allowlist")
        if g.get("golden_source") != "dtype_ceiling_accepted":
            v.append(f"CEILING_ACCEPTED golden_source={g.get('golden_source')!r} (want dtype_ceiling_accepted)")
        if g.get("golden_sha256"):
            v.append("CEILING_ACCEPTED must NOT carry golden_sha256 (never promoted to truth)")
        ks = pap.get("kernel_sha256")
        if not poc and ks and ks != ksha:
            v.append(f"CEILING_ACCEPTED S2 not fresh: kernel_sha256 {str(ks)[:12]} != kernel.py {ksha[:12]}")
    elif oke == "BLOCKED_UPSTREAM_BUG":
        if up.get("known_kernel_bug") is not True:
            v.append("BLOCKED_UPSTREAM_BUG without upstream.known_kernel_bug: true")
    return oke, v


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    allowlist = _ceiling_allowlist()
    tally = {"VERIFIED": 0, "CEILING_ACCEPTED": 0, "BLOCKED_UPSTREAM_BUG": 0, "OTHER": 0}
    violations: list[tuple[str, str]] = []
    for mp in sorted(glob.glob(str(REPO / "tasks" / "level*" / "*" / "meta.yaml"))):
        td = Path(mp).parent
        oke, vs = check_task(td, allowlist)
        tally[oke if oke in tally else "OTHER"] = tally.get(oke if oke in tally else "OTHER", 0) + 1
        for one in vs:
            violations.append((td.name, one))

    aligned = tally["VERIFIED"] + tally["CEILING_ACCEPTED"]
    blocked = tally["BLOCKED_UPSTREAM_BUG"]
    if not args.quiet:
        print(f"tally: {tally['VERIFIED']} VERIFIED + {tally['CEILING_ACCEPTED']} CEILING_ACCEPTED "
              f"= {aligned} aligned; {blocked} BLOCKED_UPSTREAM_BUG; {tally['OTHER']} other")
    # Meaningful invariant (NOT a hardcoded 82): every non-deprecated task is aligned,
    # deprecated == the known-bug set.
    inv_ok = True
    if tally["OTHER"]:
        inv_ok = False
    if violations:
        inv_ok = False
        print(f"\n{len(violations)} self-consistency violations:")
        for tid, msg in violations[:60]:
            print(f"  [{tid}] {msg}")
    if inv_ok:
        print(">>> corpus self-consistency: GREEN")
        return 0
    print(">>> corpus self-consistency: VIOLATIONS")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
