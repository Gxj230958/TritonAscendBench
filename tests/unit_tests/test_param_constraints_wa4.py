"""WA.4: param_constraints semantic validation + status resolver + eval gate.

Covers the WA.4 deliverables (`docs/plans/2026-05-21-operator-param-validity-impl.md`):

* ① the five-subsection **semantic** checks — `dims` predicates against the
  knob universe, `dtypes` / `dists` agreement with `shapes.yaml.inputs`,
  `scalars` value-domain predicates, and materialised-tensor
  `tensor_invariants` (each enumerated type);
* ②(a) the shared `check_param_constraints` helper — the `scaffold: true`
  short-circuit runs *before* any schema / semantic check;
* ②(b) the fixed-priority `resolve_task_status` arbiter — all 5 tiers, in
  particular that tiers 2/3 are NOT overridden by the param_constraints gate;
* ②(b) `eval.evaluate` sets the resolver verdict before every early return;
* ③ `build_inputs` fail-loud on an unknown dtype;
* the cross-layer integration — `eval` and `cli analyze` derive the SAME
  `task_status` through the one shared `_authoritative_task_status` /
  `resolve_task_status` authority.

0 skipped — every test runs.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from triton_ascend_bench import param_constraints as pc
from triton_ascend_bench import scoring
from triton_ascend_bench.dataset import TaskDef
from triton_ascend_bench.eval import _resolve_eval_task_status, evaluate
from triton_ascend_bench.inputs import build_inputs

REPO = Path(__file__).resolve().parent.parent.parent

# load validate_tasks.py (a script, not a package module).
_spec = importlib.util.spec_from_file_location(
    "validate_tasks", REPO / "scripts" / "validate_tasks.py")
vt = importlib.util.module_from_spec(_spec)
sys.modules["validate_tasks"] = vt
_spec.loader.exec_module(vt)


# ---------------------------------------------------------------------------
# A reusable shapes.yaml document with a tensor, an offsets vector, and a
# scalar — enough surface for all five semantic subsections.
# ---------------------------------------------------------------------------

def _block(**parts) -> dict:
    """Build a schema-closed ``scaffold: false`` param_constraints block.

    R4-1: a non-scaffold block must declare all five semantic subsections.
    Tests that exercise one subsection in isolation pass only that subsection;
    this helper backfills the others with empty containers (the explicit
    "no constraint of this kind" form) so the block is schema-closed and the
    schema-shape layer admits it through to the semantic layer.

    Args:
        **parts: any subset of ``dims`` / ``dtypes`` / ``dists`` / ``scalars``
            / ``tensor_invariants`` overrides; ``scaffold`` is forced ``False``.

    Returns:
        A schema-closed ``param_constraints`` mapping.
    """
    block: dict = {
        "scaffold": False,
        "dims": [], "dtypes": {}, "dists": {}, "scalars": {},
        "tensor_invariants": [],
    }
    block.update(parts)
    return block


def _shapes(*, dist_x: str = "randn", dtype_x: str = "float32") -> dict:
    return {
        "size_knobs": {"B": 2, "D": 16},
        "public_shapes": [
            {"id": "P1", "name": "tiny", "knobs": {"B": 1, "D": 8}},
            {"id": "P2", "name": "small", "knobs": {"B": 2, "D": 16}},
        ],
        "hidden_shapes": [
            {"id": "H1", "name": "hid", "knobs": {"B": 3, "D": 32}},
        ],
        "inputs": {
            "x": {"dims": ["B", "D"], "dtype": dtype_x, "dist": dist_x},
            "probs": {"dims": ["B", "D"], "dtype": "float32",
                      "dist": "softmax_rows"},
            "cu": {"dims": ["B"], "dtype": "int64", "dist": "offsets:to=D"},
            "eps": {"scalar": 1.0e-6},
        },
    }


# ===========================================================================
# WA.4 ① — five-subsection semantic checks: dims
# ===========================================================================

def test_dims_predicate_holds_for_all_rows_passes():
    block = _block(dims= ["B >= 1", "D >= 8"])
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert res.ok, res.reasons


def test_dims_predicate_false_for_one_row_fails():
    # `D >= 32` is false for the P1 row (D = 8).
    block = _block(dims= ["D >= 32"])
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert not res.ok
    assert any("dims predicate" in r and "false" in r for r in res.reasons)


def test_dims_predicate_unevaluable_fails():
    # `Z` is not a knob — the predicate is unevaluable.
    block = _block(dims= ["Z >= 1"])
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert not res.ok
    assert any("not evaluable" in r for r in res.reasons)


def test_dims_helper_predicates_evaluate():
    # `is_pow2` / `gcd` predicate helpers work against the knob rows
    # (every D in the fixture is a power of two).
    block = _block(dims= ["is_pow2(D)"])
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert res.ok, res.reasons


# ===========================================================================
# WA.4 ① — dtypes
# ===========================================================================

def test_dtypes_match_passes():
    block = _block(dtypes= {"x": "float32"})
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert res.ok, res.reasons


def test_dtypes_mismatch_fails():
    # x materialises float32; the constraint declares int8.
    block = _block(dtypes= {"x": "int8"})
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert not res.ok
    assert any("dtypes['x']" in r and "materialises" in r
               for r in res.reasons)


def test_dtypes_alias_spelling_accepted():
    # `fp32` and `float32` name the same torch dtype — either is accepted.
    block = _block(dtypes= {"x": "fp32"})
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert res.ok, res.reasons


def test_dtypes_unknown_input_fails():
    block = _block(dtypes= {"nonexist": "float32"})
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert not res.ok
    assert any("no such tensor input" in r for r in res.reasons)


# ===========================================================================
# WA.4 ① — dists
# ===========================================================================

def test_dists_match_passes():
    block = _block(dists= {"x": {"kind": "randn"},
                       "probs": {"kind": "softmax_rows"}})
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert res.ok, res.reasons


def test_dists_wrong_but_supported_fails():
    # probs materialises softmax_rows; declaring plain randn must be rejected.
    block = _block(dists= {"probs": {"kind": "randn"}})
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert not res.ok
    assert any("dists['probs']" in r and "requires" in r
               for r in res.reasons)


def test_dists_string_dsl_normalised_for_comparison():
    # The shapes.yaml `cu` dist is the string DSL `offsets:to=D`; the
    # constraint uses the structured form — both normalise identically.
    block = _block(dists= {"cu": {"kind": "offsets", "args": {"to": "D"}}})
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert res.ok, res.reasons


def test_dists_args_mismatch_fails():
    # `offsets:to=D` vs a declared `offsets:to=B` — different args.
    block = _block(dists= {"cu": {"kind": "offsets", "args": {"to": "B"}}})
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert not res.ok
    assert any("dists['cu']" in r for r in res.reasons)


# ===========================================================================
# WA.4 ① — scalars
# ===========================================================================

def test_scalars_predicate_holds_passes():
    block = _block(scalars= {"eps": "0 < eps <= 1e-2"})
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert res.ok, res.reasons


def test_scalars_predicate_false_fails():
    block = _block(scalars= {"eps": "eps > 1"})
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert not res.ok
    assert any("scalars['eps']" in r and "false" in r for r in res.reasons)


def test_scalars_seq_predicate_with_len_and_sum():
    # A scalar_seq input + a predicate using len() / sum().
    shapes = _shapes()
    shapes["inputs"]["sec"] = {"scalar_seq": [4, 4]}
    block = _block(scalars= {"sec": "len(sec) == 2 and sum(sec) == 8"})
    res = pc.check_param_constraints(block, shapes, tid="T")
    assert res.ok, res.reasons
    # And a violation is caught.
    block_bad = _block(scalars= {"sec": "sum(sec) == 99"})
    res_bad = pc.check_param_constraints(block_bad, shapes, tid="T")
    assert not res_bad.ok


def test_scalars_derive_rule_syntax_checked():
    shapes = _shapes()
    shapes["inputs"]["scale"] = {"derive": "1/sqrt(D)"}
    block = _block(scalars= {"scale": {"derive": "1/sqrt(D)"}})
    res = pc.check_param_constraints(block, shapes, tid="T")
    assert res.ok, res.reasons
    # A malformed derive expression is rejected.
    block_bad = _block(scalars= {"scale": {"derive": "1 +"}})
    res_bad = pc.check_param_constraints(block_bad, shapes, tid="T")
    assert not res_bad.ok


# --- R1-2: scalar / scalar_seq predicates resolve knob symbols -------------

def test_scalars_predicate_resolves_knob_symbol():
    # A scalar_seq constraint referencing a knob (`D`) must evaluate — the
    # knob symbol table is merged into the predicate's symbols (R1-2). Here
    # `D == 16` on the size_knobs default + P2 rows, so sum(sec)==D//2 holds
    # there; P1 (D=8) and H1 (D=32) make it false.
    shapes = _shapes()
    shapes["inputs"]["sec"] = {"scalar_seq": [4, 4]}
    block = _block(scalars= {"sec": "sum(sec) == D // 2"})
    res = pc.check_param_constraints(block, shapes, tid="T")
    assert not res.ok
    # The knob D resolved — failures are row-specific `is false`, NOT an
    # `unknown symbol 'D'` evaluation error.
    assert not any("unknown symbol" in r for r in res.reasons)
    assert any("is false for row P1" in r for r in res.reasons)
    assert not any("row P2" in r for r in res.reasons)


def test_scalars_predicate_knob_referencing_constraint_passes():
    # A knob-referencing scalar_seq constraint that holds for every row.
    shapes = _shapes()
    shapes["inputs"]["sec"] = {"scalar_seq": [1, 2, 3]}
    block = _block(scalars= {"sec": "len(sec) == 3 and sum(sec) < D"})
    res = pc.check_param_constraints(block, shapes, tid="T")
    assert res.ok, res.reasons


def test_scalars_scalar_predicate_resolves_knob_symbol():
    # A plain `scalar` (not a seq) predicate may also reference a knob.
    shapes = _shapes()
    shapes["inputs"]["k_eps"] = {"scalar": 8}
    block = _block(scalars= {"k_eps": "k_eps <= D"})
    res = pc.check_param_constraints(block, shapes, tid="T")
    assert res.ok, res.reasons


def test_scalars_index_constraint_on_scalar_seq():
    # R1-3 end to end: a `scalar_seq` index constraint validated through
    # the param_constraints helper.
    shapes = _shapes()
    shapes["inputs"]["sec"] = {"scalar_seq": [8, 8]}
    block = _block(scalars= {"sec": "sec[0] == 8 and sec[-1] == 8"})
    res = pc.check_param_constraints(block, shapes, tid="T")
    assert res.ok, res.reasons
    block_bad = _block(scalars= {"sec": "sec[0] == 99"})
    res_bad = pc.check_param_constraints(block_bad, shapes, tid="T")
    assert not res_bad.ok
    assert any("is false" in r for r in res_bad.reasons)


# --- R1-4: scalars.derive reconciles with shapes.yaml.inputs ---------------

def test_scalars_derive_reconciles_with_shapes_derive():
    # The `param_constraints` derive formula matches the `shapes.yaml`
    # derive formula (whitespace-insensitive) — accepted.
    shapes = _shapes()
    shapes["inputs"]["scale"] = {"derive": "1 / sqrt(D)"}
    block = _block(scalars= {"scale": {"derive": "1/sqrt(D)"}})
    res = pc.check_param_constraints(block, shapes, tid="T")
    assert res.ok, res.reasons


def test_scalars_derive_rejected_when_shapes_materialises_fixed_scalar():
    # The reported gap: `param_constraints` declares `scale` derived but
    # `shapes.yaml` still pins it to a fixed `{scalar: 0.125}` — must fail.
    shapes = _shapes()
    shapes["inputs"]["scale"] = {"scalar": 0.125}
    block = _block(scalars= {"scale": {"derive": "1/sqrt(D)"}})
    res = pc.check_param_constraints(block, shapes, tid="T")
    assert not res.ok
    assert any("derive rule" in r and "not honoured" in r
               for r in res.reasons)


def test_scalars_derive_rejected_when_formulas_differ():
    # Both sides are derive forms but the formulas disagree — must fail.
    shapes = _shapes()
    shapes["inputs"]["scale"] = {"derive": "1/sqrt(D)"}
    block = _block(scalars= {"scale": {"derive": "2/sqrt(D)"}})
    res = pc.check_param_constraints(block, shapes, tid="T")
    assert not res.ok
    assert any("differs from" in r for r in res.reasons)


def test_scalars_derive_rejected_when_shapes_materialises_seq():
    # `shapes.yaml` materialises the input as a `scalar_seq` — not a derive.
    shapes = _shapes()
    shapes["inputs"]["scale"] = {"scalar_seq": [1, 2]}
    block = _block(scalars= {"scale": {"derive": "1/sqrt(D)"}})
    res = pc.check_param_constraints(block, shapes, tid="T")
    assert not res.ok
    assert any("scalar_seq" in r and "not honoured" in r
               for r in res.reasons)


def test_scalars_derive_malformed_formula_rejected_before_reconcile():
    # A syntactically invalid derive formula fails before reconciliation.
    shapes = _shapes()
    shapes["inputs"]["scale"] = {"derive": "1/sqrt(D)"}
    block = _block(scalars= {"scale": {"derive": "1 +"}})
    res = pc.check_param_constraints(block, shapes, tid="T")
    assert not res.ok
    assert any("derive expression is invalid" in r for r in res.reasons)


# ===========================================================================
# WA.4 ① — tensor_invariants (each enumerated type, post-materialisation)
# ===========================================================================

def test_tensor_invariant_monotonic_passes():
    # `cu` is an offsets vector — monotonic non-decreasing by construction.
    block = _block(tensor_invariants= [
        {"type": "monotonic_nondecreasing", "args": {"input": "cu"}}])
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert res.ok, res.reasons


def test_tensor_invariant_first_eq_passes_and_fails():
    # An offsets vector always starts at 0.
    ok = pc.check_param_constraints(
        _block(tensor_invariants= [
            {"type": "first_eq", "args": {"input": "cu", "value": 0}}]),
        _shapes(), tid="T")
    assert ok.ok, ok.reasons
    bad = pc.check_param_constraints(
        _block(tensor_invariants= [
            {"type": "first_eq", "args": {"input": "cu", "value": 7}}]),
        _shapes(), tid="T")
    assert not bad.ok
    assert any("first_eq" in r for r in bad.reasons)


def test_tensor_invariant_last_eq_against_knob_expr():
    # `offsets:to=D` pins the final element to D — last_eq(cu, "D") holds
    # when the offsets vector has at least 2 elements (B >= 2 every row).
    shapes = _shapes()
    shapes["public_shapes"] = [
        {"id": "P1", "name": "tiny", "knobs": {"B": 2, "D": 8}},
        {"id": "P2", "name": "small", "knobs": {"B": 3, "D": 16}},
    ]
    shapes["hidden_shapes"] = [
        {"id": "H1", "name": "hid", "knobs": {"B": 4, "D": 32}},
    ]
    ok = pc.check_param_constraints(
        _block(tensor_invariants= [
            {"type": "last_eq", "args": {"input": "cu", "value": "D"}}]),
        shapes, tid="T")
    assert ok.ok, ok.reasons


def test_tensor_invariant_length_eq():
    # len(cu) == B (cu has dims [B]).
    ok = pc.check_param_constraints(
        _block(tensor_invariants= [
            {"type": "length_eq", "args": {"input": "cu", "value": "B"}}]),
        _shapes(), tid="T")
    assert ok.ok, ok.reasons
    bad = pc.check_param_constraints(
        _block(tensor_invariants= [
            {"type": "length_eq", "args": {"input": "cu", "value": "D"}}]),
        _shapes(), tid="T")
    assert not bad.ok


def test_tensor_invariant_bounded_above_by():
    # Every offsets element is <= the final value D.
    ok = pc.check_param_constraints(
        _block(tensor_invariants= [
            {"type": "bounded_above_by", "args": {"input": "cu",
                                                  "value": "D"}}]),
        _shapes(), tid="T")
    assert ok.ok, ok.reasons
    bad = pc.check_param_constraints(
        _block(tensor_invariants= [
            {"type": "bounded_above_by", "args": {"input": "cu",
                                                  "value": 0}}]),
        _shapes(), tid="T")
    assert not bad.ok


def test_tensor_invariant_index_in_range():
    # cu values lie in [0, D] — index_in_range with value D+1 holds; with a
    # too-small bound it fails.
    ok = pc.check_param_constraints(
        _block(tensor_invariants= [
            {"type": "index_in_range", "args": {"input": "cu",
                                                "value": "D + 1"}}]),
        _shapes(), tid="T")
    assert ok.ok, ok.reasons
    bad = pc.check_param_constraints(
        _block(tensor_invariants= [
            {"type": "index_in_range", "args": {"input": "cu",
                                                "value": 1}}]),
        _shapes(), tid="T")
    assert not bad.ok


def test_tensor_invariant_non_tensor_input_fails():
    # `eps` is a scalar, not a materialised tensor.
    res = pc.check_param_constraints(
        _block(tensor_invariants= [
            {"type": "monotonic_nondecreasing", "args": {"input": "eps"}}]),
        _shapes(), tid="T")
    assert not res.ok
    assert any("not a materialised tensor" in r for r in res.reasons)


# ===========================================================================
# WA.4 ②(a) — shared helper: scaffold short-circuit
# ===========================================================================

def test_scaffold_short_circuits_before_semantic_checks():
    # A `scaffold: true` block with an OTHERWISE semantically-broken body
    # (dims references a missing knob) yields ONLY the scaffold failure —
    # the semantic layer never runs.
    block = {"scaffold": True, "dims": ["NONEXISTENT >= 99"]}
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert res.scaffold is True
    assert not res.ok
    assert len(res.reasons) == 1
    assert "scaffold" in res.reasons[0]
    assert not any("dims" in r for r in res.reasons)


def test_scaffold_short_circuits_with_placeholder_shapes():
    # A scaffold task's shapes.yaml is a placeholder (no size_knobs/inputs);
    # the short-circuit means that placeholder never triggers noise.
    res = pc.check_param_constraints(
        {"scaffold": True}, {"placeholder": True}, tid="T")
    assert res.scaffold is True
    assert len(res.reasons) == 1


def test_missing_block_is_a_failure():
    res = pc.check_param_constraints(None, _shapes(), tid="T")
    assert res.missing is True
    assert not res.ok


def test_schema_illegal_block_fails_before_semantics():
    # `dims` as a string is schema-illegal — reported before any semantics.
    res = pc.check_param_constraints(
        _block(dims= "D >= 8"), _shapes(), tid="T")
    assert not res.ok
    assert any("dims must be a list" in r for r in res.reasons)


def test_clean_block_is_ok():
    block = _block(dims= ["B >= 1"],
             dtypes= {"x": "float32"})
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert res.ok
    assert res.reasons == []


# --- R5-1: a `scaffold: false` block with no shapes_doc fails closed -------

def test_scaffold_false_block_missing_shapes_doc_fails_closed():
    # R5-1 — a non-scaffold block carries the operator input contract and
    # MUST have its semantics verified. When `shapes_doc` is None (the task's
    # shapes.yaml is missing / unreadable / unparseable) the semantic checks
    # cannot run, so the verdict must be a FAILURE — never a silent pass.
    # A silent pass would let a task with a corrupt shapes.yaml resolve to
    # `evaluated` instead of `blocked` (the WA.4 fail-open hole).
    res = pc.check_param_constraints(_block(), None, tid="T")
    assert not res.ok
    assert res.scaffold is False
    assert res.missing is False
    assert any("cannot be verified" in r and "shapes.yaml" in r
               for r in res.reasons)


def test_scaffold_false_block_non_mapping_shapes_doc_fails_closed():
    # A shapes.yaml that parses to a non-mapping (e.g. a YAML list) is just
    # as unverifiable — also fail-closed.
    res = pc.check_param_constraints(_block(), ["not", "a", "mapping"],
                                     tid="T")
    assert not res.ok
    assert any("not a mapping" in r for r in res.reasons)


def test_scaffold_true_block_missing_shapes_doc_still_short_circuits():
    # A `scaffold: true` block is exempt: the scaffold short-circuit runs
    # before any shapes_doc is consulted, so `None` is fine here — the verdict
    # is the single clean `scaffold` failure, not a shapes.yaml complaint.
    res = pc.check_param_constraints({"scaffold": True}, None, tid="T")
    assert res.scaffold is True
    assert not res.ok
    assert len(res.reasons) == 1
    assert "scaffold" in res.reasons[0]
    assert not any("shapes.yaml" in r for r in res.reasons)


# --- R4-1: a `scaffold: false` block must be schema-closed ----------------

@pytest.mark.parametrize(
    "dropped", ["dims", "dtypes", "dists", "scalars", "tensor_invariants"])
def test_scaffold_false_missing_subsection_fails_helper(dropped):
    # The shared helper rejects a `scaffold: false` block that is missing any
    # one of the five semantic subsections — the schema-shape layer fires
    # before any semantic check runs.
    block = {
        "scaffold": False,
        "dims": [], "dtypes": {}, "dists": {}, "scalars": {},
        "tensor_invariants": [],
    }
    del block[dropped]
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert not res.ok
    assert any("missing required subsection" in r and dropped in r
               for r in res.reasons)


def test_scaffold_false_schema_closed_block_passes_helper():
    # The same block with every subsection present (empty containers) — the
    # explicit "no constraint" form — passes the helper.
    res = pc.check_param_constraints(_block(), _shapes(), tid="T")
    assert res.ok, res.reasons


# ===========================================================================
# WA.4 ②(b) — resolve_task_status: the fixed 5-tier priority
# ===========================================================================

def test_resolver_tier1_canary_exempt():
    # is_canary wins over everything — even when every other gate trips.
    assert pc.resolve_task_status(
        is_canary=True, self_reported_unsupported=True,
        verified_golden_incomplete=True, param_constraints_failed=True,
    ) == "evaluated"


def test_resolver_tier2_self_reported_unsupported_not_overridden():
    # A self-reported upstream task stays `unsupported` even when its
    # param_constraints fails — the gate must NOT pull it back to `blocked`.
    assert pc.resolve_task_status(
        is_canary=False, self_reported_unsupported=True,
        verified_golden_incomplete=False, param_constraints_failed=True,
    ) == "unsupported"


def test_resolver_tier3_verified_golden_incomplete_not_overridden():
    # Verified-mode incomplete golden stays `blocked` (the existing W4.3
    # verdict) — and param_constraints failing does not change it.
    assert pc.resolve_task_status(
        is_canary=False, self_reported_unsupported=False,
        verified_golden_incomplete=True, param_constraints_failed=True,
    ) == "blocked"


def test_resolver_tier4_param_constraints_failed_blocks():
    assert pc.resolve_task_status(
        is_canary=False, self_reported_unsupported=False,
        verified_golden_incomplete=False, param_constraints_failed=True,
    ) == "blocked"


def test_resolver_tier5_clean_task_evaluated():
    assert pc.resolve_task_status(
        is_canary=False, self_reported_unsupported=False,
        verified_golden_incomplete=False, param_constraints_failed=False,
    ) == "evaluated"


def test_resolver_tier_precedence_2_beats_3_and_4():
    # When tier 2 and 3 both trip, tier 2 (`unsupported`) wins.
    assert pc.resolve_task_status(
        is_canary=False, self_reported_unsupported=True,
        verified_golden_incomplete=True, param_constraints_failed=False,
    ) == "unsupported"


# ===========================================================================
# WA.4 ③ — build_inputs fail-loud on unknown dtype / dist
# ===========================================================================

def test_build_inputs_unknown_dtype_fails_loud():
    shapes = _shapes(dtype_x="flaot32")  # typo
    with pytest.raises(TypeError, match="unknown dtype"):
        build_inputs(shapes, "P1", seed=0)


def test_build_inputs_known_dtype_materialises():
    # The control: a correct dtype materialises without raising.
    out = build_inputs(_shapes(), "P1", seed=0)
    assert out["x"].dtype.is_floating_point


def test_build_inputs_unknown_dist_fails_loud():
    # WA.3 already made unknown dist fail-loud — confirm it still holds.
    from triton_ascend_bench.inputs import DistSpecError

    shapes = _shapes(dist_x="gaussian")  # not a known dist kind
    with pytest.raises(DistSpecError):
        build_inputs(shapes, "P1", seed=0)


# ===========================================================================
# WA.4 ① — validator wiring: validate_tasks reports semantic violations
# ===========================================================================

_DEF_KERNEL = "@triton.jit\ndef run(i):\n    return {}\n"


def _write_task(td: Path, *, param_constraints, shapes_inputs=None,
                is_canary=False) -> None:
    """Materialise a minimal schema-valid task dir under ``td``."""
    (td / "kernel.py").write_text(_DEF_KERNEL)
    sha = hashlib.sha256(_DEF_KERNEL.encode()).hexdigest()
    (td / "oracle_pytorch.py").write_text("def run(i):\n    return {}\n")
    knobs = {"B": 2, "D": 16}
    shapes_doc = {
        "size_knobs": knobs,
        "public_shapes": [{"id": "P1", "name": "tiny",
                           "knobs": {"B": 1, "D": 8}}],
    }
    if shapes_inputs is not None:
        shapes_doc["inputs"] = shapes_inputs
    (td / "shapes.yaml").write_text(yaml.safe_dump(shapes_doc))
    (td / "contract.md").write_text("# Contract — WA.4 fixture\n")
    meta = {
        "schema_version": "1.0", "task_id": "L1_99_wa4_fixture",
        "level": "L1", "idx": 1,
        "name": "k", "repo": "vLLM", "dtype_class": "VV",
        "category": "norm", "npu_difficulty": "low",
        "upstream": {"file_sha256": sha},
        "interface": {}, "oracle": {"gpu_triton": "kernel.py::placeholder"},
        "tolerance_class": "_t",
        "oracle_simplified": "simple_pytorch_ok", "runner_checks": {},
        "is_canary": is_canary, "is_hidden_only": False,
        "cost_model": {
            "flops": "B*D", "flops_class": "vector",
            "bytes": "2*B*D*dtype_size", "bytes_resource": "hbm",
        },
        "param_constraints": param_constraints,
    }
    (td / "meta.yaml").write_text(yaml.safe_dump(meta))


def test_validate_tasks_reports_dims_violation(tmp_path):
    # `D >= 32` is false for the P1 knob row (D = 8).
    _write_task(tmp_path, param_constraints=_block(dims= ["D >= 32"]))
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert any("dims predicate" in e and "false" in e for e in errs), errs


def test_validate_tasks_reports_scaffold(tmp_path):
    _write_task(tmp_path, param_constraints={"scaffold": True})
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert any("scaffold is true" in e for e in errs), errs


def test_validate_tasks_reports_dtype_violation(tmp_path):
    _write_task(
        tmp_path,
        param_constraints=_block(dtypes= {"x": "int8"}),
        shapes_inputs={"x": {"dims": ["B", "D"], "dtype": "float32",
                             "dist": "randn"}})
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert any("dtypes['x']" in e and "materialises" in e
               for e in errs), errs


def test_validate_tasks_clean_semantic_block_no_error(tmp_path):
    _write_task(
        tmp_path,
        param_constraints=_block(dims= ["D >= 8", "B >= 1"],
                           dtypes= {"x": "float32"},
                           dists= {"x": {"kind": "randn"}}),
        shapes_inputs={"x": {"dims": ["B", "D"], "dtype": "float32",
                             "dist": "randn"}})
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert not any("param_constraints" in e for e in errs), errs


def test_validate_tasks_canary_exempt_from_semantic_gate(tmp_path):
    # A canary with a scaffold block is NOT reported (canary exemption).
    _write_task(tmp_path, param_constraints={"scaffold": True},
                is_canary=True)
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert not any("param_constraints" in e for e in errs), errs


def test_validate_tasks_schema_error_not_double_reported(tmp_path):
    # A schema-shape error (dims as a string) is reported exactly once —
    # the semantic step suppresses the schema reasons Step 7b' already owns.
    _write_task(tmp_path, param_constraints=_block(dims= "D >= 8"))
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    dims_errs = [e for e in errs if "dims must be a list" in e]
    assert len(dims_errs) == 1, dims_errs


# ===========================================================================
# WA.4 ②(b) — eval.evaluate sets the resolver verdict before early returns
# ===========================================================================

def _synthetic_task(tmp_path: Path, *, param_constraints,
                    oracle_tier="simple_pytorch_ok") -> TaskDef:
    """A synthetic non-upstream TaskDef with a runnable shapes.yaml."""
    tdir = tmp_path / "task"
    tdir.mkdir()
    (tdir / "oracle_pytorch.py").write_text(
        "def run(inputs):\n    return {'y': inputs['x'] * 2.0}\n")
    (tdir / "kernel.py").write_text("# frozen\n")
    (tdir / "contract.md").write_text("# synthetic\n")
    shapes = {
        "size_knobs": {"B": 4},
        "public_shapes": [{"id": "P1", "name": "tiny", "knobs": {"B": 4}}],
        "hidden_shapes": [],
        "inputs": {"x": {"dims": ["B"], "dtype": "float32",
                         "dist": "randn"}},
    }
    (tdir / "shapes.yaml").write_text(yaml.safe_dump(shapes))
    meta = {
        "task_id": "SYN_wa4", "level": "L1",
        "tolerance_class": "elementwise_exact_or_inf_safe",
        "oracle_simplified": oracle_tier, "runner_checks": {},
        "upstream": {"commit_sha": "deadbeef"},
        "param_constraints": param_constraints,
    }
    (tdir / "meta.yaml").write_text(yaml.safe_dump(meta))
    return TaskDef(
        task_id="SYN_wa4", level=1, task_dir=tdir, meta=meta,
        public_shapes=[], contract_md="# synthetic\n",
        kernel_py="# frozen\n", kernel_py_sha256="0" * 64,
        param_constraints=param_constraints)


_CLEAN_PC = _block(dtypes= {"x": "float32"},
             dists= {"x": {"kind": "randn"}})
_SOLUTION_SRC = (
    "def run(inputs):\n"
    "    return {'y': inputs['x'] * 2.0}\n")


@pytest.fixture
def static_pass(monkeypatch):
    """Make the static checker pass so grading reaches phase-2b."""
    from triton_ascend_bench import eval as _eval
    from triton_ascend_bench import static_checker
    monkeypatch.setattr(_eval.static_checker, "check_solution",
                        lambda *a, **k: static_checker.CheckReport())


def test_eval_resolver_helper_scaffold_is_blocked(tmp_path):
    task = _synthetic_task(tmp_path, param_constraints={"scaffold": True})
    assert _resolve_eval_task_status(task, "self-reported") == "blocked"


def test_eval_resolver_helper_clean_is_evaluated(tmp_path):
    task = _synthetic_task(tmp_path, param_constraints=_CLEAN_PC)
    assert _resolve_eval_task_status(task, "self-reported") == "evaluated"


def test_eval_resolver_helper_corrupt_shapes_yaml_is_blocked(tmp_path):
    # R5-1 — a `scaffold: false` task whose shapes.yaml is unparseable must
    # resolve to `blocked` through the status resolver, NOT `evaluated`. The
    # resolver loads shapes.yaml, fails to parse it, passes `shapes_doc=None`
    # to the shared helper, and the helper now fails closed for a non-scaffold
    # block — so the fail-open hole is shut end-to-end.
    task = _synthetic_task(tmp_path, param_constraints=_CLEAN_PC)
    # Corrupt the on-disk shapes.yaml so yaml.safe_load raises YAMLError.
    (task.task_dir / "shapes.yaml").write_text("inputs: {x: [unterminated\n")
    assert _resolve_eval_task_status(task, "self-reported") == "blocked"


def test_eval_resolver_helper_missing_shapes_yaml_is_blocked(tmp_path):
    # The same fail-closed verdict when shapes.yaml is absent entirely.
    task = _synthetic_task(tmp_path, param_constraints=_CLEAN_PC)
    (task.task_dir / "shapes.yaml").unlink()
    assert _resolve_eval_task_status(task, "self-reported") == "blocked"


def test_eval_scaffold_task_blocked_through_static_early_return(
        tmp_path, monkeypatch):
    # A static-check block is an EARLY return — it must still carry the
    # resolver verdict (`blocked` for a scaffold task).
    from triton_ascend_bench import eval as _eval
    from triton_ascend_bench import static_checker

    blocked_report = static_checker.CheckReport()
    # `CheckReport.blocked` is derived from a `block`-severity finding.
    blocked_report.add("L0", "block", "synthetic static block", None)
    monkeypatch.setattr(_eval.static_checker, "check_solution",
                        lambda *a, **k: blocked_report)
    task = _synthetic_task(tmp_path, param_constraints={"scaffold": True})
    sol = tmp_path / "solution.py"
    sol.write_text(_SOLUTION_SRC)
    result = evaluate("SYN_wa4", sol, task_def=task, mode="self-reported")
    assert result.task_status == "blocked"


def test_eval_scaffold_task_blocked_through_grading_path(
        tmp_path, static_pass):
    # A scaffold non-upstream task still grades (oracle path), but its
    # task_status stays `blocked` from the resolver — grading does NOT
    # silently restore it to `evaluated`.
    task = _synthetic_task(tmp_path, param_constraints={"scaffold": True})
    sol = tmp_path / "solution.py"
    sol.write_text(_SOLUTION_SRC)
    result = evaluate("SYN_wa4", sol, task_def=task, mode="self-reported")
    assert result.task_status == "blocked"


def test_eval_clean_task_is_evaluated(tmp_path, static_pass):
    task = _synthetic_task(tmp_path, param_constraints=_CLEAN_PC)
    sol = tmp_path / "solution.py"
    sol.write_text(_SOLUTION_SRC)
    result = evaluate("SYN_wa4", sol, task_def=task, mode="self-reported")
    assert result.task_status == scoring.TASK_STATUS_EVALUATED


def test_eval_upstream_self_reported_stays_unsupported_not_blocked(
        tmp_path, static_pass):
    # Tier 2: a self-reported upstream task is `unsupported` even though its
    # param_constraints is a scaffold — the gate must NOT make it `blocked`.
    task = _synthetic_task(tmp_path, param_constraints={"scaffold": True},
                           oracle_tier="upstream_oracle_required")
    sol = tmp_path / "solution.py"
    sol.write_text(_SOLUTION_SRC)
    result = evaluate("SYN_wa4", sol, task_def=task, mode="self-reported")
    assert result.task_status == "unsupported"


# ===========================================================================
# WA.4 System-Wide — eval and `cli analyze` derive the SAME task_status
# ===========================================================================

def _run_tab(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "triton_ascend_bench.cli", *args],
        cwd=REPO, capture_output=True, text=True)


@pytest.mark.parametrize("task_id,expected", [
    # A reconciled corpus task — both `evaluated`.
    ("L1_16_silu_and_mul", "evaluated"),
])
def test_eval_and_cli_analyze_agree_on_task_status(
        tmp_path, task_id, expected):
    """System-wide integration (real objects, no mock): `eval` and
    `cli analyze` route through the SAME `_authoritative_task_status` /
    `resolve_task_status` authority, so the `task_status` they assign to a
    real corpus task is identical.

    Only the `evaluated` real-corpus case runs here: after WB.1 reconciled
    every `formula_exact_manual` task, no score-eligible corpus task resolves
    to a non-`evaluated` repo-authoritative status, so the cli-side mismatch
    counter (which only counts score-eligible tasks) has nothing to fire on.
    The non-`evaluated` repo-authority paths are covered by the synthetic-task
    resolver tests above (`test_resolver_tier*`,
    `test_eval_resolver_helper_scaffold_is_blocked`) and by the cli-analyze
    mismatch-counter test in `test_cli_w1_4.py`."""
    from triton_ascend_bench.dataset import load_task

    # ── eval side ──────────────────────────────────────────────────────
    task = load_task(task_id)
    eval_status = _resolve_eval_task_status(task, "self-reported")
    assert eval_status == expected, (task_id, eval_status)

    # ── cli analyze side ───────────────────────────────────────────────
    # A suite reporting `evaluated` for the task; analyze recomputes the
    # repo-authoritative status and counts a mismatch when it disagrees.
    suite = {
        "mode": "self-reported",
        "results": [
            {"task_id": task_id,
             "static_checks_passed": True, "correctness_status": "ok",
             "runtime_status": "ok", "performance_status": "ok",
             "performance_flags": [], "fallback_detected": False,
             "aggregate_score": 1.0, "s_t": 1.0,
             "task_status": "evaluated"},
        ],
    }
    suite_path = tmp_path / "suite.json"
    out_path = tmp_path / "score.json"
    suite_path.write_text(json.dumps(suite))
    r = _run_tab("analyze", "--verified-results", str(suite_path),
                 "--submitter", "t", "--out", str(out_path))
    assert r.returncode == 0, f"analyze failed: {r.stdout} {r.stderr}"
    doc = json.loads(out_path.read_text())
    # The suite reported `evaluated`; if the repo authority differs the
    # mismatch counter fires — that counter is the cli-side projection of
    # `_authoritative_task_status`.
    if expected == "evaluated":
        assert doc["task_status_mismatch_count"] == 0
    else:
        assert doc["task_status_mismatch_count"] == 1, (
            f"{task_id}: cli analyze did not derive {expected!r}")


# ===========================================================================
# R7 — validator schema-strictness fixes (codex review R7).
#
#   R7-1  `validate_schema_shape` requires the `scaffold` key on every block;
#         a missing key (silently treated as non-scaffold) is a schema error.
#   R7-2  `dtypes` only constrains real tensor inputs — a non-tensor (scalar)
#         input declared under `dtypes` is rejected, not accepted via the
#         default-dtype fallback.
#   R7-3  `dists` likewise — a non-tensor input declared under `dists` is
#         rejected, not accepted via the missing-`dist` fallback to `randn`.
# ===========================================================================

# --- R7-1: `scaffold` key is mandatory on every param_constraints block ----

def test_r7_1_missing_scaffold_key_is_schema_error():
    # A block with all five semantic subsections but NO `scaffold` key. Before
    # R7-1 `raw.get("scaffold") is not True` treated the absent key as
    # non-scaffold and never flagged it.
    block = {
        "dims": [], "dtypes": {}, "dists": {}, "scalars": {},
        "tensor_invariants": [],
    }
    errs = pc.validate_schema_shape(block, tid="T")
    assert any("missing the required 'scaffold' key" in e for e in errs), errs


def test_r7_1_missing_scaffold_key_fails_helper():
    # The full helper rejects it too — a block with no `scaffold` key never
    # resolves to `ok`.
    block = {
        "dims": [], "dtypes": {}, "dists": {}, "scalars": {},
        "tensor_invariants": [],
    }
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert not res.ok
    assert any("'scaffold' key" in r for r in res.reasons), res.reasons


def test_r7_1_non_bool_scaffold_still_rejected():
    # The existing non-bool guard still fires (a string `scaffold`).
    block = {
        "scaffold": "true",
        "dims": [], "dtypes": {}, "dists": {}, "scalars": {},
        "tensor_invariants": [],
    }
    errs = pc.validate_schema_shape(block, tid="T")
    assert any("scaffold must be a boolean" in e for e in errs), errs


def test_r7_1_present_bool_scaffold_passes():
    # The control: an explicit `scaffold: false` (or true) is fine.
    assert not any(
        "scaffold" in e
        for e in pc.validate_schema_shape(_block(), tid="T"))
    assert not any(
        "scaffold" in e and "missing" in e
        for e in pc.validate_schema_shape({"scaffold": True}, tid="T"))


def test_r7_1_validate_tasks_reports_missing_scaffold(tmp_path):
    # End-to-end: `validate_tasks` flags a corpus task whose
    # `param_constraints` block omits the `scaffold` key.
    _write_task(tmp_path, param_constraints={
        "dims": [], "dtypes": {}, "dists": {}, "scalars": {},
        "tensor_invariants": [],
    })
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert any("'scaffold' key" in e for e in errs), errs


# --- R7-2: `dtypes` only constrains tensor inputs --------------------------

def test_r7_2_dtypes_on_scalar_input_rejected():
    # `eps` is a `{scalar: ...}` non-tensor input in `_shapes()`. Declaring it
    # under `dtypes` is a schema misuse — a scalar has no materialised dtype.
    # Pre-fix it slipped through via the default-dtype fallback.
    block = _block(dtypes={"eps": "float32"})
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert not res.ok
    assert any("dtypes" in r and "eps" in r and "non-tensor" in r
               for r in res.reasons), res.reasons


def test_r7_2_dtypes_on_derive_input_rejected():
    # A `{derive: ...}` non-tensor input declared under `dtypes` is rejected.
    shapes = _shapes()
    shapes["inputs"]["scale"] = {"derive": "1/sqrt(D)"}
    block = _block(dtypes={"scale": "float32"})
    res = pc.check_param_constraints(block, shapes, tid="T")
    assert not res.ok
    assert any("dtypes" in r and "scale" in r and "non-tensor" in r
               for r in res.reasons), res.reasons


def test_r7_2_dtypes_on_tensor_input_still_passes():
    # The control: a real tensor input under `dtypes` is still accepted.
    res = pc.check_param_constraints(
        _block(dtypes={"x": "float32"}), _shapes(), tid="T")
    assert res.ok, res.reasons


def test_r7_2_validate_tasks_reports_scalar_under_dtypes(tmp_path):
    # End-to-end through `validate_tasks`.
    _write_task(
        tmp_path,
        param_constraints=_block(dtypes={"eps": "float32"}),
        shapes_inputs={"x": {"dims": ["B", "D"], "dtype": "float32",
                             "dist": "randn"},
                       "eps": {"scalar": 1.0e-6}})
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert any("dtypes" in e and "eps" in e and "non-tensor" in e
               for e in errs), errs


# --- R7-3: `dists` only constrains tensor inputs ---------------------------

def test_r7_3_dists_on_scalar_input_rejected():
    # `eps` is a non-tensor input — declaring it under `dists` is a schema
    # misuse. Pre-fix it slipped through via the missing-`dist` randn fallback.
    block = _block(dists={"eps": {"kind": "randn"}})
    res = pc.check_param_constraints(block, _shapes(), tid="T")
    assert not res.ok
    assert any("dists" in r and "eps" in r and "non-tensor" in r
               for r in res.reasons), res.reasons


def test_r7_3_dists_on_scalar_seq_input_rejected():
    # A `{scalar_seq: ...}` non-tensor input declared under `dists` is
    # rejected too.
    shapes = _shapes()
    shapes["inputs"]["block_size"] = {"scalar_seq": [8, 8]}
    block = _block(dists={"block_size": {"kind": "randn"}})
    res = pc.check_param_constraints(block, shapes, tid="T")
    assert not res.ok
    assert any("dists" in r and "block_size" in r and "non-tensor" in r
               for r in res.reasons), res.reasons


def test_r7_3_dists_on_tensor_input_still_passes():
    # The control: a real tensor input under `dists` is still accepted.
    res = pc.check_param_constraints(
        _block(dists={"probs": {"kind": "softmax_rows"}}), _shapes(),
        tid="T")
    assert res.ok, res.reasons


def test_r7_3_validate_tasks_reports_scalar_under_dists(tmp_path):
    # End-to-end through `validate_tasks`.
    _write_task(
        tmp_path,
        param_constraints=_block(dists={"eps": {"kind": "randn"}}),
        shapes_inputs={"x": {"dims": ["B", "D"], "dtype": "float32",
                             "dist": "randn"},
                       "eps": {"scalar": 1.0e-6}})
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert any("dists" in e and "eps" in e and "non-tensor" in e
               for e in errs), errs
