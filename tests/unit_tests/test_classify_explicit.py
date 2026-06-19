"""Phase-2 U9 c4b: classify + the cost-symbol universe are format-invariant (knob == explicit).

The explicit format resolves dist args to literals, so a dist-param symbol that homes ONLY in a dist (e.g.
L1_21:N, L2_26:T -- both used in a cost formula) was unrecoverable until approach A added a row-invariant
``dist_symbolic``. These tests pin (1) classify verdict parity, (2) cost-universe parity, and (3) that the
dist-param-in-cost symbols are no longer homeless on the explicit form.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.corpus.classify_symbols import classify  # noqa: E402
from scripts.corpus.cost_environment import (  # noqa: E402
    check_cost_symbols,
    cost_formula_symbols,
    named_cost_universe,
)
from scripts.corpus.gen_explicit_shapes import generate_explicit_shapes  # noqa: E402

_TASKS = [
    "tasks/level1/L1_01_bias",
    "tasks/level1/L1_09_quantize_k_cache_fast",      # declared cost row_only_param D
    "tasks/level1/L1_21_create_flashmla_kv_indices_triton",  # dist-param N in cost
    "tasks/level2/L2_26_fwd_kernel_ep_scatter_1",    # dist-param T in cost
    "tasks/level3/L3_13_kernel_unified_attention_3d",
    "tasks/level3/L3_15_chunk_state_varlen",
]


def _docs(task: str):
    d = REPO / task
    if not (d / "shapes.yaml").exists():
        pytest.skip(f"{task} absent")
    knob = yaml.safe_load((d / "shapes.yaml").read_text())
    meta = yaml.safe_load((d / "meta.yaml").read_text()) or {}
    return knob, meta, generate_explicit_shapes(knob)


@pytest.mark.parametrize("task", _TASKS)
def test_classify_verdict_parity(task):
    knob, meta, exp = _docs(task)
    assert classify(knob, meta) == classify(exp, meta)


@pytest.mark.parametrize("task", _TASKS)
def test_cost_universe_parity(task):
    knob, meta, exp = _docs(task)
    assert check_cost_symbols(knob, meta) == check_cost_symbols(exp, meta)


@pytest.mark.parametrize(("task", "sym"), [
    ("tasks/level1/L1_21_create_flashmla_kv_indices_triton", "N"),
    ("tasks/level2/L2_26_fwd_kernel_ep_scatter_1", "T"),
])
def test_dist_param_in_cost_recovered_on_explicit(task, sym):
    # the symbol homes only in a dist arg yet appears in a cost formula -- on the explicit form it must be
    # recovered (via dist_symbolic) into the cost universe, NOT flagged homeless.
    _knob, meta, exp = _docs(task)
    assert sym in cost_formula_symbols(meta), f"{task} cost formula should reference {sym}"
    assert sym in named_cost_universe(exp, meta), f"{sym} must be recovered into the explicit cost universe"
    assert classify(exp, meta).get(sym) == "dist-param"
    assert check_cost_symbols(exp, meta) == [], "explicit cost universe must be clean"


def test_classify_dispatches_explicit_path():
    # an explicit doc must classify off the rows (env keys), not return {} from the absent size_knobs.
    knob, meta, exp = _docs("tasks/level1/L1_01_bias")
    ce = classify(exp, meta)
    assert ce, "explicit classify must not be empty"
    assert set(ce) == set((knob.get("size_knobs") or {}).keys())


def test_plain_scalar_symbol_recovered_on_explicit():
    # a plain `scalar` input whose value is a SYMBOL string must classify scalar-input on the explicit form
    # (the knob path scans the scalar field, so the explicit path must scan the raw value for parity). No
    # corpus task exercises this today; this pins the latent edge so a future hand-authored doc is covered.
    doc = {"schema_version": 1, "public": [
        {"id": "P1", "role": "public", "env": {"M": 4},
         "inputs": [{"name": "s", "kind": "scalar", "value": "M"}]}], "hidden": []}
    assert classify(doc, {}).get("M") == "scalar-input"


def test_corpus_wide_classify_and_cost_parity():
    # the strong guarantee: across EVERY task, classify + the cost universe are byte-identical knob vs explicit.
    cls_mism, cost_mism = [], []
    for sp in sorted((REPO / "tasks").rglob("shapes.yaml")):
        knob = yaml.safe_load(sp.read_text())
        if not isinstance(knob, dict) or not knob.get("size_knobs"):
            continue
        meta = yaml.safe_load((sp.parent / "meta.yaml").read_text()) or {}
        exp = generate_explicit_shapes(knob)
        if classify(knob, meta) != classify(exp, meta):
            cls_mism.append(sp.parent.name)
        if check_cost_symbols(knob, meta) != check_cost_symbols(exp, meta):
            cost_mism.append(sp.parent.name)
    assert not cls_mism, f"classify mismatches: {cls_mism}"
    assert not cost_mism, f"cost-universe mismatches: {cost_mism}"
