"""Phase-2 U9 approach A: the explicit format carries a row-invariant ``dist_symbolic`` beside the per-row
resolved ``dist``.

This unlocks (1) symbolic dist-parity in ``param_constraints`` (the resolved-only compare misses a symbol
disagreement that coincidentally resolves equal) and (2) dist-param recovery for the cost universe (a later
unit). These tests pin the format field, the fail-closed discriminator checks, the row-invariance signature,
the symbolic dists check, and the manifest projection.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from ascendbench.input_spec import shape_format as sf  # noqa: E402
from ascendbench.input_spec.explicit_build import resolve_dist_args  # noqa: E402
from ascendbench.input_spec.inputs import parse_dist_spec  # noqa: E402
from ascendbench.input_spec.param_constraints import check_param_constraints  # noqa: E402
from scripts.corpus.gen_explicit_shapes import SCHEMA_VERSION, generate_explicit_shapes  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _knob_corpus import knob_doc  # noqa: E402

# a task whose tensors carry several dist kinds incl a symbolic-arg dist (randint:0:N)
_SYMBOLIC_DIST_TASK = "tasks/level1/L1_02_compute_slot_mappings"


def _task(task: str):
    d = REPO / task
    if not (d / "shapes.yaml").exists():
        pytest.skip(f"{task} absent")
    meta = yaml.safe_load((d / "meta.yaml").read_text()) or {}
    doc = knob_doc(task)   # corpus is v2; the knob form (for generate_explicit_shapes) is reconstructed from the committed v2
    return meta.get("param_constraints"), doc


def _synthetic_doc(dsym: dict, resolved: dict, env: dict) -> dict:
    """One-row explicit doc with a single tensor x carrying (dsym, resolved) under env."""
    return {"schema_version": SCHEMA_VERSION, "public": [
        {"id": "P1", "role": "public", "env": env, "inputs": [
            {"name": "x", "kind": "tensor", "shape": [env.get("N", 8)], "dtype": "float32",
             "axes": ["N"], "dist": resolved, "dist_symbolic": dsym},
        ]}], "hidden": []}


def test_explicit_tensor_carries_row_invariant_dist_symbolic():
    # every tensor input of a generated explicit doc has a well-formed dist_symbolic that is IDENTICAL across
    # all rows (only the resolved dist/shape move per row).
    _raw, knob = _task(_SYMBOLIC_DIST_TASK)
    exp = generate_explicit_shapes(knob)
    rows = exp["public"] + exp["hidden"]
    assert rows, "task should generate rows"
    seen: dict[str, dict] = {}
    for row in rows:
        for inp in row["inputs"]:
            if inp["kind"] != "tensor":
                continue
            ds = inp.get("dist_symbolic")
            assert isinstance(ds, dict) and isinstance(ds.get("kind"), str) and isinstance(ds.get("args"), dict), \
                f"tensor {inp['name']} dist_symbolic malformed: {ds!r}"
            if inp["name"] in seen:
                assert ds == seen[inp["name"]], f"dist_symbolic not row-invariant for {inp['name']}"
            seen[inp["name"]] = ds
    assert seen, "task should have at least one tensor input"


def test_is_explicit_rejects_tensor_missing_dist_symbolic():
    bad = {"schema_version": SCHEMA_VERSION, "public": [
        {"id": "P1", "inputs": [{"name": "x", "kind": "tensor", "dtype": "float32", "axes": ["N"]}]}]}
    with pytest.raises(sf.ShapeFormatError, match="dist_symbolic"):
        sf.is_explicit_shapes_doc(bad)


def test_is_explicit_rejects_non_string_dist_args_keys():
    # a hostile doc with mixed-type dist_symbolic.args keys must fail closed with ShapeFormatError, NOT leak a
    # raw TypeError from the row-invariance signature's sort.
    bad = {"schema_version": SCHEMA_VERSION, "public": [
        {"id": "P1", "inputs": [{"name": "x", "kind": "tensor", "dtype": "float32", "axes": ["N"],
                                 "dist_symbolic": {"kind": "randint", "args": {1: "a", "b": "c"}}}]}]}
    with pytest.raises(sf.ShapeFormatError, match="args keys must all be strings"):
        sf.is_explicit_shapes_doc(bad)


def test_is_explicit_rejects_tensor_missing_axes():
    bad = {"schema_version": SCHEMA_VERSION, "public": [
        {"id": "P1", "inputs": [{"name": "x", "kind": "tensor", "dtype": "float32",
                                 "dist_symbolic": {"kind": "randn", "args": {}}}]}]}
    with pytest.raises(sf.ShapeFormatError, match="axes"):
        sf.is_explicit_shapes_doc(bad)


def _tensor(ds: dict) -> dict:
    return {"name": "x", "kind": "tensor", "dtype": "float32", "axes": ["N"], "dist_symbolic": ds}


def test_sig_catches_dist_symbolic_drift():
    # two rows whose tensor x declares a DIFFERENT symbolic dist must fail closed (row-invariance), even
    # though name/kind/dtype/axes agree -- a silent symbolic-dist change between rows is a real defect.
    drift = {"schema_version": SCHEMA_VERSION, "public": [
        {"id": "P1", "inputs": [_tensor({"kind": "randn", "args": {}})]},
        {"id": "P2", "inputs": [_tensor({"kind": "rand", "args": {}})]},
    ]}
    with pytest.raises(sf.ShapeFormatError, match="inconsistent input forms"):
        sf.is_explicit_shapes_doc(drift)


def test_validate_dists_explicit_catches_symbol_disagreement():
    # THE weakness approach A closes: constraint declares the LITERAL `randint:0:8`; the row materialises the
    # SYMBOLIC `randint:0:N` with N==8. The values resolve EQUAL (so a resolved-only compare PASSES) but the
    # symbolic forms disagree -- the symbolic-first compare must catch it. (dists values are {kind, args}
    # mappings, like a real meta -- parse_dist_spec normalises a DSL string to that form.)
    dsym = parse_dist_spec("randint:0:N")
    resolved = resolve_dist_args(dsym, {"N": 8})
    doc = _synthetic_doc(dsym, resolved, {"N": 8})
    raw = {"dims": [], "dtypes": {}, "dists": {"x": parse_dist_spec("randint:0:8")},
           "scalars": {}, "tensor_invariants": [], "scaffold": False}
    res = check_param_constraints(raw, doc, tid="sym-disagree")
    assert res.ok is False, "symbolic disagreement (literal 8 vs symbol N) must be caught"
    assert any("symbolic dist" in r for r in res.reasons), res.reasons
    # sanity: the resolved values DO coincide (so only the symbolic compare catches it)
    assert resolve_dist_args(parse_dist_spec("randint:0:8"), {"N": 8}) == resolved


def test_validate_dists_explicit_passes_on_symbol_agreement():
    # the matching symbolic constraint (`randint:0:N`) passes both the symbolic and resolved compares.
    dsym = parse_dist_spec("randint:0:N")
    doc = _synthetic_doc(dsym, resolve_dist_args(dsym, {"N": 8}), {"N": 8})
    raw = {"dims": [], "dtypes": {}, "dists": {"x": parse_dist_spec("randint:0:N")},
           "scalars": {}, "tensor_invariants": [], "scaffold": False}
    assert check_param_constraints(raw, doc, tid="sym-agree").ok is True


def test_dist_symbolic_change_moves_explicit_manifest_hash():
    # dist_symbolic rides in the explicit canonical projection, so changing it (a semantic change) moves the
    # explicit manifest hash. (Knob-format hashes are unaffected -- proven by the all-101 neutrality gate.)
    _raw, knob = _task(_SYMBOLIC_DIST_TASK)
    exp = generate_explicit_shapes(knob)
    base = sf.explicit_canonical_projection(exp)
    mutated = copy.deepcopy(exp)
    for row in mutated["public"] + mutated["hidden"]:
        for inp in row["inputs"]:
            if inp["kind"] == "tensor":
                inp["dist_symbolic"] = {"kind": "zzz_other", "args": {}}
    assert sf.explicit_canonical_projection(mutated) != base
