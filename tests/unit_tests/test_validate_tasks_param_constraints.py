"""WA.1: param_constraints schema — loader + validator tests.

Covers the WA.1 deliverables (`docs/plans/2026-05-21-operator-param-validity-impl.md`):

* the dataset loader is **fully lenient** — a missing block yields
  ``TaskDef.param_constraints = None``; a malformed block is kept verbatim as
  a raw dict and never raises (deliberately unlike ``cost_model``);
* ``TaskDef`` carries the new ``param_constraints`` field with a default;
* ``validate_tasks`` lists ``param_constraints`` as a required top-level
  ``meta.yaml`` key, with the ``is_canary`` exemption;
* the WA.1 validator layer checks only the **schema shape** of a present
  block (six subsections, container types) — not semantics (that is WA.4);
* the two hand-reconciled sample tasks (``L1_12``, ``L1_16``) carry a
  ``scaffold: false`` ``param_constraints`` block that passes the WA.1
  schema-shape check.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from ascendbench.corpus.dataset import load_task

REPO = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location(
    "validate_tasks", REPO / "scripts" / "validate_tasks.py"
)
vt = importlib.util.module_from_spec(_spec)
sys.modules["validate_tasks"] = vt
_spec.loader.exec_module(vt)


_OMIT = object()  # sentinel: leave the param_constraints block out entirely


def _write_task(td: Path, *, param_constraints=_OMIT, is_canary=False) -> None:
    """Materialise a minimal schema-valid task dir under ``td``.

    ``param_constraints`` defaults to :data:`_OMIT` (block omitted); pass an
    explicit value to inject one. ``is_canary`` flips ``meta.is_canary``.
    """
    kernel = "@triton.jit\ndef run(i):\n    return {}\n"
    (td / "kernel.py").write_text(kernel)
    sha = hashlib.sha256(kernel.encode()).hexdigest()
    (td / "oracle_pytorch.py").write_text("def run(i):\n    return {}\n")
    knobs = {"B": 2, "N": 4}
    # The fixture shapes carry one tensor input + one scalar so the WA.4
    # semantic checks have something to validate against.
    (td / "shapes.yaml").write_text(
        yaml.safe_dump(
            {
                "size_knobs": knobs,
                "public_shapes": [{"id": "P1", "name": "tiny", "knobs": knobs}],
                "inputs": {
                    "x": {"dims": ["B", "N"], "dtype": "float32",
                          "dist": "randn"},
                    "eps": {"scalar": 1.0e-6},
                },
            }
        )
    )
    (td / "contract.md").write_text("# Contract — L1_99 param-constraints fixture\n")
    meta = {
        "schema_version": "1.0", "task_id": "L1_99_param_constraints_fixture",
        "level": "L1", "idx": 1,
        "name": "k", "repo": "vLLM", "dtype_class": "VV",
        "category": "norm", "npu_difficulty": "low",
        "upstream": {"file_sha256": sha},
        "interface": {}, "oracle": {"gpu_triton": "kernel.py::placeholder"},
        "tolerance_class": "_t",
        "oracle_simplified": "simple_pytorch_ok", "runner_checks": {},
        "is_canary": is_canary, "is_hidden_only": False,
        # cost_model is required too — include a legal block so the only
        # error surface under test is param_constraints.
        "cost_model": {
            "flops": "B*N", "flops_class": "vector",
            "bytes": "2*B*N*dtype_size", "bytes_resource": "hbm",
        },
    }
    if param_constraints is not _OMIT:
        meta["param_constraints"] = param_constraints
    (td / "meta.yaml").write_text(yaml.safe_dump(meta))


# A schema-shape-valid full param_constraints block (all six subsections).
_GOOD_BLOCK = {
    "scaffold": False,
    "dims": ["D >= 8", "B >= 1"],
    "dtypes": {"x": "float32", "a": "int8"},
    "dists": {
        "x": {"kind": "randn"},
        "idx": {"kind": "randint", "args": {"lo": 0, "hi": "N"}},
    },
    "scalars": {"eps": "0 < eps <= 1e-2"},
    "tensor_invariants": [
        {"type": "monotonic_nondecreasing", "args": {"input": "cu"}},
        {"type": "length_eq", "args": {"input": "seqused", "value": "E"}},
    ],
}


# ---------------------------------------------------------------------------
# TaskDef field + loader leniency
# ---------------------------------------------------------------------------

def test_param_constraints_missing_is_none(tmp_path, monkeypatch):
    # Backward compat: a task whose meta.yaml has no `param_constraints`
    # block loads fine with `TaskDef.param_constraints = None`. Uses a
    # synthetic task dir — the WA.5 generator seeds a `scaffold: true`
    # block into every real corpus task, so no corpus task is block-free.
    td = tmp_path / "level1" / "L1_99_param_constraints_fixture"
    td.mkdir(parents=True)
    _write_task(td)  # param_constraints omitted
    import ascendbench.corpus.dataset as ds
    monkeypatch.setattr(ds, "TASKS_ROOT", tmp_path)
    task = ds.load_task("L1_99_param_constraints_fixture")
    assert task.param_constraints is None


def test_param_constraints_sample_task_loaded_as_raw_dict():
    # The hand-reconciled sample task carries a real block; the loader
    # stores it verbatim as a raw dict (no structured object).
    task = load_task("L1_12_rms_norm")
    assert isinstance(task.param_constraints, dict)
    assert task.param_constraints["scaffold"] is False
    assert task.param_constraints["dtypes"] == {
        "input": "bfloat16", "weight": "bfloat16"
    }


def test_loader_lenient_on_malformed_block(tmp_path, monkeypatch):
    # A malformed (schema-illegal) `param_constraints` must NOT raise inside
    # load_task — unlike `cost_model`, the loader keeps it as a raw dict so it
    # can reach the WA.4 task_status=blocked gate.
    td = tmp_path / "level1" / "L1_99_param_constraints_fixture"
    td.mkdir(parents=True)
    # `dims` is a string, not a list — schema-illegal but loader-lenient.
    _write_task(td, param_constraints={"dims": "D >= 8", "scaffold": "yes"})
    # The loader resolves tasks under tasks/level{1,2,3}; point it here.
    import ascendbench.corpus.dataset as ds
    monkeypatch.setattr(ds, "TASKS_ROOT", tmp_path)
    task = ds.load_task("L1_99_param_constraints_fixture")
    # raw dict preserved verbatim, no raise, no validation
    assert task.param_constraints == {"dims": "D >= 8", "scaffold": "yes"}


def test_loader_wraps_non_mapping_param_constraints(tmp_path, monkeypatch):
    # A `param_constraints` that is not a mapping at all (a stray list) is
    # still kept reachable: the loader wraps it under a sentinel key rather
    # than dropping it or raising.
    td = tmp_path / "level1" / "L1_99_param_constraints_fixture"
    td.mkdir(parents=True)
    _write_task(td, param_constraints=["not", "a", "mapping"])
    import ascendbench.corpus.dataset as ds
    monkeypatch.setattr(ds, "TASKS_ROOT", tmp_path)
    task = ds.load_task("L1_99_param_constraints_fixture")
    assert task.param_constraints == {"__raw__": ["not", "a", "mapping"]}


# ---------------------------------------------------------------------------
# validate_tasks: param_constraints is required (with canary exemption)
# ---------------------------------------------------------------------------

def test_missing_param_constraints_reported(tmp_path):
    # WA.1: `param_constraints` is a required top-level meta key — a task
    # with no block is a validation error.
    _write_task(tmp_path)  # param_constraints omitted
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert any("missing keys" in e and "param_constraints" in e
               for e in errs), errs


def test_canary_exempt_from_required_param_constraints(tmp_path):
    # `is_canary: true` tasks are exempt — a missing block is NOT reported.
    _write_task(tmp_path, is_canary=True)  # param_constraints omitted
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert not any("param_constraints" in e for e in errs), errs


def test_canary_still_validates_other_required_keys(tmp_path):
    # The canary exemption removes ONLY param_constraints from the required
    # set; every other required key is still enforced for a canary.
    _write_task(tmp_path, is_canary=True, param_constraints={"scaffold": True})
    meta = yaml.safe_load((tmp_path / "meta.yaml").read_text())
    del meta["cost_model"]
    (tmp_path / "meta.yaml").write_text(yaml.safe_dump(meta))
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert any("missing keys" in e and "cost_model" in e for e in errs), errs


# A reconciled (`scaffold: false`) block that is BOTH schema-shape valid AND
# semantically consistent with the `_write_task` fixture shapes.yaml — its
# `dims` predicates hold for the {B:2, N:4} knob row, its `dtypes` / `dists`
# match the `x` tensor input, and its `scalars` predicate holds for `eps`.
_SEMANTIC_GOOD_BLOCK = {
    "scaffold": False,
    "dims": ["B >= 1", "N >= 1"],
    "dtypes": {"x": "float32"},
    "dists": {"x": {"kind": "randn"}},
    "scalars": {"eps": "0 < eps <= 1e-2"},
    "tensor_invariants": [],
}


def test_present_param_constraints_not_double_reported(tmp_path):
    # A present, schema-shape-valid AND semantically-consistent block (WA.4)
    # produces no param_constraints error from the full validator.
    _write_task(tmp_path, param_constraints=_SEMANTIC_GOOD_BLOCK)
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert not any("param_constraints" in e for e in errs), errs


# ---------------------------------------------------------------------------
# WA.1 validator layer: schema-shape checks only
# ---------------------------------------------------------------------------

def test_good_block_passes_schema_shape():
    errs = vt._validate_param_constraints("T", {"param_constraints": _GOOD_BLOCK})
    assert errs == [], errs


def test_scaffold_only_block_passes_schema_shape():
    # A bare `{scaffold: true}` draft is schema-shape-valid at the WA.1 layer
    # (the scaffold *gate* — rejecting it — is WA.4, not WA.1).
    errs = vt._validate_param_constraints("T", {"param_constraints": {"scaffold": True}})
    assert errs == [], errs


def test_missing_block_skipped_by_layer():
    # The WA.1 layer skips a missing block (the required-key check owns it).
    errs = vt._validate_param_constraints("T", {})
    assert errs == []


def test_non_mapping_block_reported():
    errs = vt._validate_param_constraints("T", {"param_constraints": ["x"]})
    assert any("must be a mapping" in e for e in errs), errs


def test_unknown_subsection_reported():
    errs = vt._validate_param_constraints(
        "T", {"param_constraints": {"scaffold": True, "bogus": 1}})
    assert any("unknown subsections" in e and "bogus" in e for e in errs), errs


def test_dims_must_be_list_of_strings():
    errs = vt._validate_param_constraints(
        "T", {"param_constraints": {"dims": "D >= 8"}})
    assert any("dims must be a list" in e for e in errs), errs
    errs = vt._validate_param_constraints(
        "T", {"param_constraints": {"dims": ["D >= 8", 16]}})
    assert any("dims must be a list of" in e for e in errs), errs


def test_dtypes_unknown_dtype_reported():
    # The dtype value is bound to the closed `build_inputs` alias enum — a
    # typo (`flaot32`) is statically rejected.
    errs = vt._validate_param_constraints(
        "T", {"param_constraints": {"dtypes": {"x": "flaot32"}}})
    assert any("dtypes['x']" in e and "not in" in e for e in errs), errs


def test_dtypes_known_dtype_accepted():
    # A partial draft block (`scaffold: true`) so the schema-shape layer
    # exercises ONLY the dtypes sub-check; the R4-1 all-five-subsections rule
    # applies to `scaffold: false` blocks and is not under test here.
    errs = vt._validate_param_constraints(
        "T", {"param_constraints": {"scaffold": True,
                                    "dtypes": {"x": "bfloat16", "a": "int8"}}})
    assert errs == [], errs


def test_dtypes_value_must_be_string():
    errs = vt._validate_param_constraints(
        "T", {"param_constraints": {"dtypes": {"x": 32}}})
    assert any("dtypes['x']" in e and "dtype string" in e for e in errs), errs


def test_dists_unknown_kind_reported():
    errs = vt._validate_param_constraints(
        "T", {"param_constraints": {"dists": {"x": {"kind": "gaussian"}}}})
    assert any("dists['x']" in e and "kind" in e and "not in" in e
               for e in errs), errs


def test_dists_missing_kind_reported():
    errs = vt._validate_param_constraints(
        "T", {"param_constraints": {"dists": {"x": {"args": {}}}}})
    assert any("dists['x']" in e and "missing string 'kind'" in e
               for e in errs), errs


def test_dists_bad_args_and_extra_key_reported():
    errs = vt._validate_param_constraints(
        "T", {"param_constraints": {
            "dists": {"x": {"kind": "randn", "args": [1], "extra": 2}}}})
    assert any("args must be a mapping" in e for e in errs), errs
    assert any("unknown keys" in e for e in errs), errs


def test_dists_good_kinds_accepted():
    # Partial draft block — only the dists sub-check is under test (see
    # `test_dtypes_known_dtype_accepted` for the `scaffold: true` rationale).
    errs = vt._validate_param_constraints(
        "T", {"param_constraints": {"scaffold": True, "dists": {
            "x": {"kind": "softmax_rows"},
            "idx": {"kind": "randint", "args": {"lo": 0, "hi": "N"}},
        }}})
    assert errs == [], errs


def test_scalars_predicate_string_accepted():
    errs = vt._validate_param_constraints(
        "T", {"param_constraints": {"scaffold": True,
                                    "scalars": {"eps": "0 < eps <= 1e-2"}}})
    assert errs == [], errs


def test_scalars_derive_mapping_accepted():
    errs = vt._validate_param_constraints(
        "T", {"param_constraints": {
            "scaffold": True,
            "scalars": {"scale": {"derive": "1/sqrt(D)"}}}})
    assert errs == [], errs


def test_scalars_bad_form_reported():
    # A mapping without `derive`, or a non-string/non-mapping rule.
    errs = vt._validate_param_constraints(
        "T", {"param_constraints": {"scalars": {"eps": {"min": 0}}}})
    assert any("scalars['eps']" in e and "derive" in e for e in errs), errs
    errs = vt._validate_param_constraints(
        "T", {"param_constraints": {"scalars": {"eps": 5}}})
    assert any("scalars['eps']" in e for e in errs), errs


def test_tensor_invariants_unknown_type_reported():
    errs = vt._validate_param_constraints(
        "T", {"param_constraints": {"tensor_invariants": [
            {"type": "is_sorted", "args": {"input": "cu"}}]}})
    assert any("tensor_invariants[0]" in e and "not in" in e for e in errs), errs


def test_tensor_invariants_must_be_list():
    errs = vt._validate_param_constraints(
        "T", {"param_constraints": {"tensor_invariants": {"type": "first_eq"}}})
    assert any("tensor_invariants must be" in e for e in errs), errs


def test_tensor_invariants_missing_type_and_extra_key():
    errs = vt._validate_param_constraints(
        "T", {"param_constraints": {"tensor_invariants": [
            {"args": {"input": "cu"}, "bogus": 1}]}})
    assert any("missing string 'type'" in e for e in errs), errs
    assert any("unknown keys" in e for e in errs), errs


def test_tensor_invariants_good_types_accepted():
    # Partial draft block — only the tensor_invariants sub-check is under test.
    errs = vt._validate_param_constraints(
        "T", {"param_constraints": {"scaffold": True, "tensor_invariants": [
            {"type": "monotonic_nondecreasing", "args": {"input": "cu"}},
            {"type": "first_eq", "args": {"input": "cu", "value": 0}},
            {"type": "bounded_above_by", "args": {"input": "x", "value": "N"}},
            {"type": "index_in_range", "args": {"input": "x", "value": "E"}},
        ]}})
    assert errs == [], errs


def test_scaffold_must_be_boolean():
    errs = vt._validate_param_constraints(
        "T", {"param_constraints": {"scaffold": "true"}})
    assert any("scaffold must be a boolean" in e for e in errs), errs


# ---------------------------------------------------------------------------
# R4-1: a `scaffold: false` block must be schema-closed — every one of the
# five semantic subsections must be present (an empty container is fine, a
# silently-absent key is not). A `scaffold: true` draft is exempt.
# ---------------------------------------------------------------------------

_FIVE_SUBSECTIONS = ("dims", "dtypes", "dists", "scalars", "tensor_invariants")


@pytest.mark.parametrize("dropped", _FIVE_SUBSECTIONS)
def test_scaffold_false_missing_one_subsection_rejected(dropped):
    # Drop exactly one of the five semantic subsections from an otherwise
    # schema-closed `scaffold: false` block — the validator must reject it.
    block = {
        "scaffold": False,
        "dims": [], "dtypes": {}, "dists": {}, "scalars": {},
        "tensor_invariants": [],
    }
    del block[dropped]
    errs = vt._validate_param_constraints("T", {"param_constraints": block})
    assert any("missing required subsection" in e and dropped in e
               for e in errs), errs


def test_scaffold_false_all_subsections_empty_containers_accepted():
    # A schema-closed block where every semantic subsection is an EMPTY
    # container is accepted — an empty container is the explicit "no
    # constraint of this kind" declaration, not a missing key.
    block = {
        "scaffold": False,
        "dims": [], "dtypes": {}, "dists": {}, "scalars": {},
        "tensor_invariants": [],
    }
    errs = vt._validate_param_constraints("T", {"param_constraints": block})
    assert errs == [], errs


def test_scaffold_true_block_exempt_from_schema_closure():
    # A `scaffold: true` draft is NOT required to be schema-closed — the
    # all-five rule applies only once the block has been reconciled.
    errs = vt._validate_param_constraints(
        "T", {"param_constraints": {"scaffold": True}})
    assert errs == [], errs


# ---------------------------------------------------------------------------
# end-to-end: the WA.1 sample tasks pass the schema-shape check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_id", ["L1_12_rms_norm", "L2_04_silu_and_mul"])
def test_sample_tasks_param_constraints_pass_schema_shape(task_id):
    # The two hand-reconciled WA.1 sample tasks carry a `scaffold: false`
    # block that passes the WA.1 schema-shape validator layer cleanly.
    level_dir = f"level{task_id[1]}"  # L1_* -> level1, L2_* -> level2
    meta = yaml.safe_load(
        (REPO / "tasks" / level_dir / task_id / "meta.yaml").read_text()
    )
    assert "param_constraints" in meta
    assert meta["param_constraints"]["scaffold"] is False
    errs = vt._validate_param_constraints(task_id, meta)
    assert errs == [], errs
