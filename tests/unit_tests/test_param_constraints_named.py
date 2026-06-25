"""Phase-2 U8 commit 4a: param_constraints dual-format (knob + explicit named row environment).

The strongest dual-format test is VERDICT PARITY: a constraint's intent is format-invariant, so for a real
task `check_param_constraints(raw, knob_doc).ok` must equal `check_param_constraints(raw, explicit_doc).ok`.
Knob-format byte-neutrality is proven separately (all-101 patched==HEAD diff). These tests exercise the
explicit path per subsection, including the per-row dist revalidation codex required.
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
from ascendbench.input_spec.param_constraints import check_param_constraints  # noqa: E402
from scripts.corpus.gen_explicit_shapes import generate_explicit_shapes  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _knob_corpus import knob_doc  # noqa: E402

# diverse passing tasks: dists / scalars / tensor_invariants / dtypes / dims + randn_inf_excl + optional-null
_PARITY_TASKS = [
    "tasks/level1/L1_01_bias",
    "tasks/level1/L1_02_compute_slot_mappings",
    "tasks/level1/L1_09_quantize_k_cache_fast",
    "tasks/level1/L1_15_merge_state",                  # randn_inf_excl (ref-bearing dist)
    "tasks/level3/L3_13_kernel_unified_attention_3d",  # optional/null q_descale
    "tasks/level2/L2_01_layer_norm_fwd",
]


def _task(task: str):
    d = REPO / task
    if not (d / "shapes.yaml").exists():
        pytest.skip(f"{task} absent")
    meta = yaml.safe_load((d / "meta.yaml").read_text()) or {}
    doc = knob_doc(task)   # the corpus is v2 now; the knob form (for the knob-vs-explicit parity) is reconstructed from the committed v2
    return meta.get("param_constraints"), doc


@pytest.mark.parametrize("task", _PARITY_TASKS)
def test_explicit_verdict_matches_knob(task):
    raw, knob = _task(task)
    exp = generate_explicit_shapes(knob)
    knob_v = check_param_constraints(raw, knob, tid=task)
    exp_v = check_param_constraints(raw, exp, tid=task)
    assert knob_v.ok == exp_v.ok, (task, knob_v.reasons, exp_v.reasons)
    assert exp_v.ok, f"{task} explicit verdict not ok: {exp_v.reasons}"   # all parity tasks pass on knob


def test_explicit_dist_mismatch_caught_symmetrically():
    # mutate a real dists constraint to a WRONG dist kind; both knob and explicit must catch it (the
    # explicit path resolves the symbolic constraint per row and compares to the row's resolved dist).
    raw, knob = _task("tasks/level1/L1_02_compute_slot_mappings")
    assert raw and raw.get("dists"), "fixture needs a dists constraint"
    bad = copy.deepcopy(raw)
    name = next(iter(bad["dists"]))
    bad["dists"][name] = "softmax_rows"   # a wrong-but-valid dist kind for that tensor
    exp = generate_explicit_shapes(knob)
    assert check_param_constraints(bad, knob, tid="m").ok is False
    assert check_param_constraints(bad, exp, tid="m").ok is False     # caught in explicit too


def test_input_specs_optional_is_non_dict():
    # an optional/null input must reconstruct as a NON-DICT (None), matching a knob-format null input, so a
    # tensor-detector never mistakes it for a tensor (codex fix).
    _raw, knob = _task("tasks/level3/L3_13_kernel_unified_attention_3d")
    exp = generate_explicit_shapes(knob)
    specs = sf.input_specs(exp)
    knob_specs = sf.input_specs(knob)
    nulls = [n for n, s in knob_specs.items() if s is None]
    assert nulls, "L3_13 should declare an optional/null input"
    for n in nulls:
        assert specs.get(n) is None, f"explicit optional {n} should be None, got {specs.get(n)!r}"


def test_malformed_explicit_doc_returns_clean_blocked_not_crash():
    # check_param_constraints is the single pass/fail entrypoint -- a malformed explicit doc must return a
    # fail-closed verdict (ok=False), never raise ShapeFormatError up into validate_tasks/eval. Use a
    # schema-complete raw so validation reaches the semantic layer where the adapter discriminator runs.
    raw, _knob = _task("tasks/level1/L1_01_bias")
    malformed = {"schema_version": 1, "public": [{"id": "P1"}]}   # versioned but row missing its 'inputs'
    res = check_param_constraints(raw, malformed, tid="bad")
    assert res.ok is False
    assert any("malformed explicit" in r for r in res.reasons), res.reasons


def test_malformed_explicit_input_name_returns_clean_blocked_not_crash():
    # A truthy but unhashable input name used to pass is_explicit_shapes_doc, then crash input_specs() when
    # it tried to use the list as a dict key. The discriminator must reject it as ShapeFormatError so the
    # check_param_constraints entrypoint can fail closed.
    raw = {"dims": [], "dtypes": {"x": "float32"}, "dists": {}, "scalars": {},
           "tensor_invariants": [], "scaffold": False}
    malformed = {"schema_version": 1, "public": [
        {"id": "P1", "inputs": [{"name": ["x"], "kind": "tensor", "dtype": "float32"}]},
    ]}
    res = check_param_constraints(raw, malformed, tid="bad-name")
    assert res.ok is False
    assert any("malformed explicit" in r for r in res.reasons), res.reasons


def test_malformed_explicit_env_returns_clean_blocked_not_crash():
    # A non-mapping row env used to crash effective_env() through dict(env) before the entrypoint could
    # return a clean blocked verdict.
    raw = {"dims": ["B >= 1"], "dtypes": {}, "dists": {}, "scalars": {},
           "tensor_invariants": [], "scaffold": False}
    malformed = {"schema_version": 1, "public": [{"id": "P1", "env": ["bad"], "inputs": []}]}
    res = check_param_constraints(raw, malformed, tid="bad-env")
    assert res.ok is False
    assert any("malformed explicit" in r for r in res.reasons), res.reasons


def test_randn_inf_excl_ref_preserved_on_explicit():
    # L1_15's randn_inf_excl dist carries a tensor-NAME ref; the explicit per-row dist revalidation must
    # keep it a name (not resolve it), so the task still validates clean on the explicit form.
    raw, knob = _task("tasks/level1/L1_15_merge_state")
    exp = generate_explicit_shapes(knob)
    assert check_param_constraints(raw, exp, tid="L1_15").ok is True
