"""Unit 0 (shape-coverage expansion, Phase 0): fail-closed pipeline + coverage gate + shape-manifest
freshness. These gates must ship BEFORE any shape is added so a new/edited shape can never be
silently ungraded or leave a stale PASS.
"""
from __future__ import annotations

import copy
import sys
import types
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from ascendbench.corpus.shape_manifest import shape_manifest_sha256  # noqa: E402
from ascendbench.release_gate.eligibility import (  # noqa: E402
    fresh_s2_pass,
    golden_coverage_exact,
)

# ── shape-manifest hash: stable to cosmetics, sensitive to anything build_inputs consumes ────────

_BASE_SHAPES = {
    "size_knobs": {"B": 8, "L": 16, "D": 32},
    "inputs": {
        "a": {"dims": ["B", "L"], "dtype": "float32", "dist": "randn"},
        "b": {"dims": ["B", "D"], "dtype": "float16", "dist": "randn"},
    },
    "public_shapes": [
        {"id": "P1", "name": "tiny", "knobs": {"B": 1, "L": 4, "D": 8}},
        {"id": "P2", "name": "small", "knobs": {"B": 2, "L": 8, "D": 16}},
    ],
    "hidden_shapes": [{"id": "H1", "name": "odd", "knobs": {"B": 3, "L": 6, "D": 12}}],
}


def test_manifest_stable_to_name_rename():
    d = copy.deepcopy(_BASE_SHAPES)
    d["public_shapes"][0]["name"] = "RENAMED"
    assert shape_manifest_sha256(d) == shape_manifest_sha256(_BASE_SHAPES)


def test_manifest_changes_on_added_shape():
    d = copy.deepcopy(_BASE_SHAPES)
    d["public_shapes"].append({"id": "P3", "knobs": {"B": 4}})
    assert shape_manifest_sha256(d) != shape_manifest_sha256(_BASE_SHAPES)


def test_manifest_changes_on_inplace_knob_edit():
    # The core gap: same id-set, edited dim -> must change (equivalence_per_shape keys would NOT).
    d = copy.deepcopy(_BASE_SHAPES)
    d["public_shapes"][0]["knobs"]["D"] = 999
    assert shape_manifest_sha256(d) != shape_manifest_sha256(_BASE_SHAPES)


def test_manifest_changes_on_input_dtype_edit():
    d = copy.deepcopy(_BASE_SHAPES)
    d["inputs"]["a"]["dtype"] = "bfloat16"
    assert shape_manifest_sha256(d) != shape_manifest_sha256(_BASE_SHAPES)


def test_manifest_changes_on_input_reorder():
    # build_inputs iterates inputs in authored order with one shared RNG -> a reorder is a real
    # semantic change and MUST change the hash.
    d = copy.deepcopy(_BASE_SHAPES)
    d["inputs"] = {"b": _BASE_SHAPES["inputs"]["b"], "a": _BASE_SHAPES["inputs"]["a"]}
    assert shape_manifest_sha256(d) != shape_manifest_sha256(_BASE_SHAPES)


def test_manifest_changes_on_size_knob_default_edit():
    d = copy.deepcopy(_BASE_SHAPES)
    d["size_knobs"]["B"] = 777
    assert shape_manifest_sha256(d) != shape_manifest_sha256(_BASE_SHAPES)


# ── fresh_s2_pass: shape-manifest freshness (backward compatible) ─────────────────────────────────

def test_fresh_s2_pass_backward_compatible_two_args():
    m = {"phase_a_prime": {"status": "PASS", "kernel_sha256": "abc"}}
    assert fresh_s2_pass(m, "abc") is True
    assert fresh_s2_pass(m, "different") is False


def test_fresh_s2_pass_manifest_stale_blocks_pass_branch():
    m = {"phase_a_prime": {"status": "PASS", "shape_manifest_sha256": "OLD"}}
    assert fresh_s2_pass(m, None, "OLD") is True
    assert fresh_s2_pass(m, None, "NEW") is False


def test_fresh_s2_pass_manifest_absent_does_not_block():
    # Legacy meta without the field: forward-compatible (no manifest clause fires).
    m = {"phase_a_prime": {"status": "PASS"}}
    assert fresh_s2_pass(m, None, "ANY") is True


def test_fresh_s2_pass_skipped_branch_uses_phase_b_manifest():
    m = {
        "phase_a_prime": {"status": "SKIPPED_BY_DESIGN", "shape_manifest_sha256": "PAP"},
        "phase_b_capture": {"status": "DONE", "shape_manifest_sha256": "PBC"},
    }
    # SKIPPED branch keys on phase_b_capture, NOT phase_a_prime.
    assert fresh_s2_pass(m, None, "PBC") is True
    assert fresh_s2_pass(m, None, "PAP") is False


# ── golden_coverage_exact: strict declared == golden-stems (both directions) ──────────────────────

def _mk_task(tmp_path: Path, declared: list[str], golden_stems: list[str]):
    td = tmp_path / "t"
    td.mkdir(exist_ok=True)
    (td / "shapes.yaml").write_text(
        yaml.safe_dump({"public_shapes": [{"id": s} for s in declared]})
    )
    gdir = td / "golden"
    gdir.mkdir(exist_ok=True)
    for s in golden_stems:
        # a real safetensors header so golden_artifact_usable passes
        import torch
        from safetensors.torch import save_file
        save_file({"x": torch.zeros(1)}, str(gdir / f"{s}.safetensors"))
    return types.SimpleNamespace(task_id="t", task_dir=td)


def test_golden_coverage_exact_ok(tmp_path):
    t = _mk_task(tmp_path, ["P1", "P2"], ["P1", "P2"])
    ok, reason = golden_coverage_exact(t)
    assert ok is True, reason


def test_golden_coverage_exact_missing(tmp_path):
    t = _mk_task(tmp_path, ["P1", "P2"], ["P1"])
    ok, reason = golden_coverage_exact(t)
    assert ok is False and "missing" in reason


def test_golden_coverage_exact_stale_extra(tmp_path):
    # A stale golden for a removed shape -- subset-only golden_complete would MISS this.
    t = _mk_task(tmp_path, ["P1"], ["P1", "P_old"])
    ok, reason = golden_coverage_exact(t)
    assert ok is False and "stale" in reason


# ── precompute fail-closed: only opt-in sweep extras may silently skip ────────────────────────────

def test_precompute_extra_flag_only_for_opt_in_sweep(monkeypatch):
    import importlib

    pre = importlib.import_module("scripts.oracle.precompute_oracle_reference")
    shapes_doc = copy.deepcopy(_BASE_SHAPES)
    declared = pre._declared_shape_ids(shapes_doc)
    assert declared == {"P1", "P2", "H1"}

    # No opt-in sweep set -> nothing is "extra"; every declared shape is fail-closed.
    monkeypatch.delenv("TAB_SWEEP_EXTRA_SHAPES", raising=False)
    assert pre._sweep_extra_ids() == set()

    # An opt-in sweep extra is recognised + appended to the sweep list.
    monkeypatch.setenv("TAB_SWEEP_EXTRA_SHAPES", "Z9, Z8")
    assert pre._sweep_extra_ids() == {"Z9", "Z8"}
    swept = pre._sweep_shape_ids(shapes_doc)
    assert "Z9" in swept and "Z8" in swept
    # declared shapes still come first and are NOT in the sweep-extra (lenient) set
    assert "P1" in swept and "H1" in swept
    assert not (declared & pre._sweep_extra_ids())


# ── apply: SHAPE_COVERAGE_GAP mapping + no lingering truth-source ──────────────────────────────────

def test_apply_shape_coverage_gap_clears_stale_truth_source(tmp_path, monkeypatch):
    import importlib

    apply_mod = importlib.import_module("scripts.oracle.apply_oracle_equivalence_verdict")

    # A task whose meta previously said VERIFIED/promoted-to-truth.
    td = tmp_path / "tasks" / "level1" / "L1_x"
    td.mkdir(parents=True)
    (td / "kernel.py").write_text("def run(inputs):\n    return {}\n")
    (td / "shapes.yaml").write_text(yaml.safe_dump({"public_shapes": [{"id": "P1"}]}))
    (td / "meta.yaml").write_text(yaml.safe_dump({
        "provenance": {"golden": {
            "oracle_kernel_equivalence": "VERIFIED",
            "golden_source": "oracle_pytorch_verified",
            "golden_sha256": "deadbeef",
        }},
    }))
    verdict = td.parent.parent / "verdict.json"
    import json
    verdict.write_text(json.dumps({
        "task_id": "L1_x", "overall": "SHAPE_COVERAGE_GAP",
        "shapes": [{"shape_id": "P1", "status": "SHAPE_COVERAGE_GAP"}],
    }))

    monkeypatch.setattr(apply_mod, "_find_task_dir", lambda tid: td)
    res = apply_mod._apply_one(verdict)
    meta = yaml.safe_load((td / "meta.yaml").read_text())
    g = meta["provenance"]["golden"]
    assert g["oracle_kernel_equivalence"] == "SHAPE_COVERAGE_GAP"
    assert g["golden_source"] == "blocked_shape_coverage_gap"  # overwritten, not lingering
    assert "golden_sha256" not in g  # cleared on non-VERIFIED
    assert res.promoted_to_truth is False


# ── run_release_gate coverage pass: closes the subset-escape ──────────────────────────────────────

def _write_task(base: Path, tid: str, *, oke: str, manifest: str | None, fresh_manifest: bool):
    """Materialise a minimal on-disk task whose meta is VERIFIED/CEILING + S2 PASS, with a
    shape_manifest that is either fresh or stale vs the on-disk shapes.yaml."""
    import hashlib

    td = base / tid
    td.mkdir(parents=True, exist_ok=True)
    src = "def run(inputs):\n    return {}\n"
    (td / "kernel.py").write_text(src)
    ksha = hashlib.sha256(src.encode()).hexdigest()
    shapes = {"public_shapes": [{"id": "P1", "knobs": {"B": 1}}]}
    (td / "shapes.yaml").write_text(yaml.safe_dump(shapes))
    on_disk = shape_manifest_sha256(shapes)
    pap_manifest = on_disk if fresh_manifest else (manifest or "STALE")
    meta = {
        "tolerance_class": "vector_reduce",
        "phase_a_prime": {
            "status": "PASS", "kernel_sha256": ksha,
            "standalone_sha256": ksha, "shape_manifest_sha256": pap_manifest,
        },
        "provenance": {"golden": {
            "oracle_kernel_equivalence": oke,
            **({"golden_source": "oracle_pytorch_verified", "golden_sha256": "x"}
               if oke == "VERIFIED" else {"golden_source": "blocked_shape_coverage_gap"}),
            "equivalence_per_shape": {"P1": {"status": "VERIFIED"}},
        }},
    }
    (td / "meta.yaml").write_text(yaml.safe_dump(meta))
    return types.SimpleNamespace(task_id=tid, task_dir=td, meta=meta)


def test_release_gate_red_on_stale_manifest(tmp_path):
    import importlib
    vt = importlib.import_module("scripts.corpus.validate_tasks")
    good = _write_task(tmp_path, "good", oke="VERIFIED", manifest=None, fresh_manifest=True)
    stale = _write_task(tmp_path, "stale", oke="VERIFIED", manifest=None, fresh_manifest=False)
    # both eligible + S2 PASS -> compute_release_blocking empty; only the coverage pass catches
    # the stale-manifest task.
    rc = vt.run_release_gate(tasks=[good, stale], eligible_ids={"good", "stale"})
    assert rc == 1


def test_release_gate_red_on_coverage_gap_dropped_from_eligible(tmp_path):
    # The subset-escape: a SHAPE_COVERAGE_GAP task is NOT provenance-eligible, so it would drop out
    # of the eligible denominator. The coverage pass runs over ALL tasks, so it is still caught RED.
    import importlib
    vt = importlib.import_module("scripts.corpus.validate_tasks")
    good = _write_task(tmp_path, "good", oke="VERIFIED", manifest=None, fresh_manifest=True)
    gap = _write_task(tmp_path, "gap", oke="SHAPE_COVERAGE_GAP", manifest=None, fresh_manifest=True)
    # gap is excluded from eligible_ids (mirrors compute_release_ready_eligible dropping it).
    rc = vt.run_release_gate(tasks=[good, gap], eligible_ids={"good"})
    assert rc == 1


def test_release_gate_green_when_all_fresh(tmp_path):
    import importlib
    vt = importlib.import_module("scripts.corpus.validate_tasks")
    a = _write_task(tmp_path, "a", oke="VERIFIED", manifest=None, fresh_manifest=True)
    b = _write_task(tmp_path, "b", oke="VERIFIED", manifest=None, fresh_manifest=True)
    rc = vt.run_release_gate(tasks=[a, b], eligible_ids={"a", "b"})
    assert rc == 0


# ── Unit 1 feasibility classifier: quant takes precedence over multi-float ─────────────────────────

def test_feasibility_quant_precedes_multi_dtype():
    import importlib
    rep = importlib.import_module("scripts.corpus.shape_feasibility_report")
    # A w8a8 quant kernel that also uses an fp32 accumulator + fp16 output: the incidental float
    # tokens must NOT make it a multi_dtype (dtype-sweep) candidate -- its input dtype is int8-locked.
    quant_src = (
        "import torch\n"
        "def run(inputs):\n"
        "    acc = torch.zeros(1, dtype=torch.float32)\n"
        "    out = acc.to(torch.float16)\n"
        "    # w8a8 int8 block-scaled matmul\n"
        "    return {'out': out}\n"
    )
    cls, ev = rep._classify_dtype(quant_src)
    assert cls == "quant_locked", (cls, ev)

    # A genuine multi-dtype float kernel (no quant token) stays multi_dtype.
    multi_src = (
        "import torch\n"
        "configs = {torch.bfloat16: 1, torch.float16: 2, torch.float32: 3}\n"
        "def run(inputs):\n    return {}\n"
    )
    cls2, _ = rep._classify_dtype(multi_src)
    assert cls2 == "multi_dtype", cls2
