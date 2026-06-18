"""Phase-2 U7: cost_model re-base onto the named row environment.

Two guarantees:
  * CHARACTERIZATION (blocking): the re-base changes which symbols are LEGAL, never what a formula
    EVALUATES to. The committed baseline (`data/cost_baseline.json`) must reproduce exactly -- a mismatch
    means a formula value moved, which would move Roofline t_ref / perf scoring (never tune-to-pass).
  * NAMED UNIVERSE: a cost symbol must be a tensor-axis/scalar/dist input, a declared row_only_param, or a
    derived symbol; a homeless symbol fails (typo guard), and a row_only_param must be real + truly homeless.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from ascendbench.perf.cost_model import CostModelError, parse_cost_model  # noqa: E402
from scripts.corpus.cost_environment import check_cost_symbols, named_cost_universe  # noqa: E402
from scripts.perf.snapshot_cost_outcomes import snapshot_corpus  # noqa: E402

_BASELINE = REPO / "tests/unit_tests/data/cost_baseline.json"


def test_characterization_cost_values_unchanged():
    # the re-based universe must not move any flops/bytes value (incl the L1_21 div-by-zero outcomes)
    baseline = json.loads(_BASELINE.read_text())
    now = snapshot_corpus()
    assert now == baseline, "cost_model values changed vs the committed baseline -- perf scoring would move"


def test_baseline_records_failures_as_first_class_outcomes():
    # the snapshot captures an error outcome, not a crash (L1_21:P16 has L=1 and a L-1 denominator)
    baseline = json.loads(_BASELINE.read_text())
    p16 = baseline["L1_21_create_flashmla_kv_indices_triton"]["P16"]["bytes"]["hbm"]
    assert isinstance(p16, dict) and "error" in p16 and "division by zero" in p16["error"]


# --- named-universe check -------------------------------------------------------------------------

def _doc(inputs=None, knobs=None):
    return {"size_knobs": knobs or {"B": 2, "N": 4},
            "inputs": inputs or {"x": {"dims": ["B", "N"], "dtype": "float32", "dist": "randn"}},
            "public_shapes": [{"id": "P1", "knobs": {}}]}


def test_homed_cost_symbols_pass():
    meta = {"cost_model": {"flops": "B*N", "flops_class": "vector",
                           "bytes": "B*N*dtype_size", "bytes_resource": "hbm"}}
    assert check_cost_symbols(_doc(), meta) == []
    assert {"B", "N", "dtype_size"} <= named_cost_universe(_doc(), meta)


def test_homeless_cost_symbol_fails():
    # G appears in the cost formula but no tensor axis / scalar / dist input -- and is not declared
    meta = {"cost_model": {"flops": "B*N*G", "flops_class": "vector",
                           "bytes": "B*N*dtype_size", "bytes_resource": "hbm"}}
    v = check_cost_symbols(_doc(), meta)
    assert any("'G'" in s and "homeless" in s for s in v)


def test_declared_row_only_param_passes():
    # G is a declared size_knob used only in the cost formula (no tensor axis) -> a real row-only param
    doc = _doc(knobs={"B": 2, "N": 4, "G": 8})
    meta = {"cost_model": {"flops": "B*N*G", "flops_class": "vector",
                           "bytes": "B*N*dtype_size", "bytes_resource": "hbm",
                           "row_only_params": ["G"]}}
    assert check_cost_symbols(doc, meta) == []


def test_declared_row_only_param_not_in_formula_flagged():
    meta = {"cost_model": {"flops": "B*N", "flops_class": "vector",
                           "bytes": "B*N*dtype_size", "bytes_resource": "hbm",
                           "row_only_params": ["G"]}}
    assert any("dead declaration" in s for s in check_cost_symbols(_doc(), meta))


def test_declared_row_only_param_with_input_home_flagged():
    # B has a tensor-axis home, so declaring it row-only is wrong
    meta = {"cost_model": {"flops": "B*N", "flops_class": "vector",
                           "bytes": "B*N*dtype_size", "bytes_resource": "hbm",
                           "row_only_params": ["B"]}}
    assert any("has an input home" in s for s in check_cost_symbols(_doc(), meta))


def test_parse_cost_model_row_only_params_type_checked():
    base = {"flops": "B", "flops_class": "vector", "bytes": "B*dtype_size", "bytes_resource": "hbm"}
    assert parse_cost_model({**base, "row_only_params": ["D"]}).row_only_params == ("D",)
    assert parse_cost_model(base).row_only_params == ()
    with pytest.raises(CostModelError, match="row_only_params"):
        parse_cost_model({**base, "row_only_params": "D"})       # str, not a list
    with pytest.raises(CostModelError, match="row_only_params"):
        parse_cost_model({**base, "row_only_params": [1]})       # non-string element


def test_free_symbols_does_not_drop_helper_named_value_symbol():
    # codex blind-spot fix: an AST extractor must report a symbol used as a VALUE even if it shares a
    # helper's spelling (a regex-minus-funcs scan would mis-drop it), while still excluding a real call.
    from ascendbench.input_spec.expr import free_symbols
    assert free_symbols("B*min") == {"B", "min"}        # min as a value -> a symbol
    assert free_symbols("min(B, N)") == {"B", "N"}      # min as a call -> not a symbol
    # a name used as BOTH a call func and a value in one formula must still report the value (exclude by
    # node identity, not globally) -- the residual blind spot codex caught.
    assert free_symbols("min(B, N) + min") == {"B", "N", "min"}
    # and the universe check therefore catches such a homeless symbol rather than silently dropping it
    meta = {"cost_model": {"flops": "B*min", "flops_class": "vector",
                           "bytes": "B*dtype_size", "bytes_resource": "hbm"}}
    assert any("'min'" in s and "homeless" in s for s in check_cost_symbols(_doc(), meta))


def test_three_corpus_row_only_tasks_declared_and_clean():
    import yaml
    for task, sym in [("tasks/level1/L1_09_quantize_k_cache_fast", "D"),
                      ("tasks/level3/L3_13_kernel_unified_attention_3d", "CAUSAL"),
                      ("tasks/level3/L3_15_chunk_state_varlen", "G")]:
        d = REPO / task
        doc = yaml.safe_load((d / "shapes.yaml").read_text())
        meta = yaml.safe_load((d / "meta.yaml").read_text())
        assert sym in parse_cost_model(meta["cost_model"]).row_only_params
        assert check_cost_symbols(doc, meta) == []
