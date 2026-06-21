"""Unit 2: per-task append-only shape-extension tool (scripts/corpus/extend_task_shapes.py).

Covers the mechanics that must be correct before any shape is appended: active-knob detection,
NPUKernelBench dim-mapping (codex residual #4), non-pow2/mod-respecting value generation, canonical
dedup, canonical v2 re-emit (the append path), and determinism. The CPU-validation == pending_gpu contract is
exercised on a synthetic CPU-runnable task.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

ext = importlib.import_module("scripts.corpus.extend_task_shapes")

from ascendbench.input_spec.compact_shapes import generate_compact_shapes  # noqa: E402
from scripts.corpus.migrate_to_v2 import to_yaml  # noqa: E402
from scripts.corpus.v2_authoring import v2_to_knob_spec  # noqa: E402


def _v2_and_base(knob_yaml: str | None = None):
    """A (committed v2 doc, reconstructed knob base_doc) pair from a knob fixture."""
    knob = yaml.safe_load(knob_yaml if knob_yaml is not None else _SHAPES_YAML)
    v2 = generate_compact_shapes(knob)
    return v2, v2_to_knob_spec(v2)


# ── active-knob detection (document order, only inputs-referenced knobs) ───────────────────────────

def test_active_knobs_in_order():
    doc = {
        "size_knobs": {"B": 1, "L": 1, "D": 1, "N": 1, "Z": 1},
        "inputs": {
            "a": {"dims": ["B", "D"]},
            "b": {"dims": ["D", "N"]},
        },
    }
    # Z is inert (not referenced); order follows first appearance in inputs.
    assert ext._active_knobs_in_order(doc) == ["B", "D", "N"]


def test_active_knobs_expression_dim():
    doc = {"size_knobs": {"B": 1, "D": 1}, "inputs": {"x": {"dims": ["B", "2*D"]}}}
    assert ext._active_knobs_in_order(doc) == ["B", "D"]


def test_active_knobs_scans_dist_only_knob():
    # A knob (N) that appears ONLY inside a dist expression must be detected as active --
    # regression for the dist-scan gap (omitting it mis-classified the knob as dead).
    doc = {
        "size_knobs": {"B": 1, "D": 1, "N": 1, "E": 1, "Z": 1},
        "inputs": {
            "x": {"dims": ["B", "D"]},
            "idx": {"dims": ["E"], "dist": "randint:0:E"},
            "offs": {"dims": ["E"], "dist": "offsets:to=B*N"},  # N only here
            "seq": {"dims": ["E"], "dist": "full:D"},
        },
    }
    active = ext._active_knobs_in_order(doc)
    assert "N" in active          # dist-only knob now detected
    assert "Z" not in active      # genuinely-inert knob still excluded
    assert set(active) == {"B", "D", "E", "N"}


def test_active_knobs_dist_dict_form():
    # dist may also be a dict; str() rendering must still surface the knob reference.
    doc = {"size_knobs": {"D": 1, "M": 1}, "inputs": {"x": {"dims": ["D"], "dist": {"kind": "randint", "high": "M"}}}}
    assert set(ext._active_knobs_in_order(doc)) == {"D", "M"}


# ── value generation: min + mod respected, non-pow2 ───────────────────────────────────────────────

def test_round_valid_min_and_mod():
    assert ext._round_valid(1, {"min": 4, "mod": None}) == 4
    assert ext._round_valid(7, {"min": 1, "mod": 2}) == 8      # round up to multiple of 2
    assert ext._round_valid(193, {"min": 1, "mod": 2}) == 194  # odd -> next even


def test_nonpow2_near_is_not_power_of_two():
    v = ext._nonpow2_near(8192, {"min": 1, "mod": None})
    assert not ext._is_pow2(v)
    assert v >= 8192  # scale-relative (near 1.5x)
    v2 = ext._nonpow2_near(1024, {"min": 1, "mod": 2})
    assert not ext._is_pow2(v2) and v2 % 2 == 0


# ── NPUKernelBench dim-mapping (codex residual #4: explicit mapping tests) ─────────────────────────

def test_borrow_layernorm_mapping():
    # x=[1,128,4096], normalized_shape=[4096] (1-D) -> B=prod(1,128)=128, D=4096
    out = ext._borrow_layernorm([[1, 128, 4096]], [{"name": "normalized_shape", "value": [4096]}])
    assert out is not None and out[0] == {"B": 128, "D": 4096}


def test_borrow_layernorm_rejects_multidim_normalize():
    # normalized_shape=[4096,64] (2-D) does not map to TAB x=[B,D] -> filtered out
    out = ext._borrow_layernorm([[1, 4096, 64]], [{"name": "normalized_shape", "value": [4096, 64]}])
    assert out is None


def test_borrow_bmm_mapping_and_filter():
    # A=[2,16,32] @ B=[2,32,16] -> B=2,L=16,D=32,N=16
    out = ext._borrow_bmm([[2, 16, 32], [2, 32, 16]], [])
    assert out is not None and out[0] == {"B": 2, "L": 16, "D": 32, "N": 16}
    # k mismatch (A last dim 32 != B middle dim 8) -> rejected
    assert ext._borrow_bmm([[2, 16, 32], [2, 8, 16]], []) is None
    # non-3D -> rejected
    assert ext._borrow_bmm([[16, 32], [32, 16]], []) is None


def test_borrow_sampling_is_diverse_and_deterministic():
    # only run if the real NPUKernelBench json is present
    if not (ext.NPUKB_ROOT / "level1/10_LayerNorm.json").is_file():
        return
    bounds = {"B": {"min": 1, "mod": None}, "D": {"min": 2, "mod": None}}
    a = ext._gen_borrow_candidates("L2_01_layer_norm_fwd", ["B", "D"], bounds, cap=6)
    b = ext._gen_borrow_candidates("L2_01_layer_norm_fwd", ["B", "D"], bounds, cap=6)
    assert [c.knobs for c in a] == [c.knobs for c in b]  # deterministic
    sizes = [c.knobs["B"] * c.knobs["D"] for c in a]
    assert sizes == sorted(sizes) and len(set(sizes)) > 1  # spread small->large, not clustered


# ── canonical dedup key (effective active knobs, inert/formatting ignored) ─────────────────────────

def test_active_key_ignores_inert_and_uses_size_knob_fallback():
    doc = {"size_knobs": {"B": 8, "D": 16, "Z": 99}}
    # row overrides only B; D falls back to size_knobs; Z is inert (not active)
    k1 = ext._active_key(["B", "D"], ext._effective(doc, {"B": 4}))
    k2 = ext._active_key(["B", "D"], ext._effective(doc, {"B": 4, "Z": 1}))  # inert Z differs
    assert k1 == k2 == (("B", 4), ("D", 16))


# ── canonical v2 append (_write_extended_v2) ───────────────────────────────────────────────────────

_SHAPES_YAML = """# header comment kept verbatim
size_knobs:
  B: 8
  D: 16
public_shapes:
- id: P1
  name: tiny
  knobs:
    B: 1
    D: 2
hidden_shapes:
- id: H1
  name: odd
  knobs:
    B: 3
    D: 6
inputs:
  x:
    dims:
    - B
    - D
    dtype: bfloat16
    dist: randn
"""


def test_write_extended_v2_preserves_existing_rows(tmp_path):
    orig_v2, base = _v2_and_base()
    p = tmp_path / "shapes.yaml"
    new_v2 = ext._write_extended_v2(
        orig_v2, base,
        pub_rows=[{"id": "P2", "knobs": {"B": 128, "D": 4096}}],
        hid_rows=[{"id": "H2", "knobs": {"B": 1, "D": 2}}],
        shapes_path=p)
    # existing rows re-materialise byte-identical (the v2 row dict is unchanged).
    assert next(r for r in new_v2["public"] if r["id"] == "P1") == \
        next(r for r in orig_v2["public"] if r["id"] == "P1")
    assert {r["id"] for r in new_v2["public"]} == {"P1", "P2"}
    assert {r["id"] for r in new_v2["hidden"]} == {"H1", "H2"}
    # the new public row materialises the candidate env (B=128, D=4096) into its tensor shape.
    p2 = next(r for r in new_v2["public"] if r["id"] == "P2")
    assert p2["env"]["B"] == 128 and p2["env"]["D"] == 4096
    assert next(i for i in p2["inputs"] if i["name"] == "x")["shape"] == [128, 4096]
    # the written file is canonical v2.
    assert p.read_text() == to_yaml(new_v2)


def test_write_extended_v2_byte_preservation_guard(tmp_path, monkeypatch):
    # if the re-emit ever changed an existing row, the guard must raise rather than write a corrupted corpus.
    orig_v2, base = _v2_and_base()
    real = ext.generate_compact_shapes

    def _corrupt(doc):
        out = real(doc)
        out["public"][0]["env"]["B"] = 999   # mutate an EXISTING row (P1) on re-emit
        return out

    monkeypatch.setattr(ext, "generate_compact_shapes", _corrupt)
    with pytest.raises(RuntimeError, match="byte-preservation FAILED"):
        ext._write_extended_v2(orig_v2, base, [{"id": "P2", "knobs": {"B": 4, "D": 8}}], [],
                               tmp_path / "shapes.yaml")


# ── end-to-end on a synthetic CPU-runnable task: survivors are pending_gpu, append-only ────────────

def _mk_synth_task(tmp_path: Path) -> Path:
    td = tmp_path / "tasks" / "level1" / "L1_synth"
    td.mkdir(parents=True)
    # the corpus is compact v2; extend_task load_v2's it, so the synth task ships its shapes.yaml as v2.
    (td / "shapes.yaml").write_text(to_yaml(generate_compact_shapes(yaml.safe_load(_SHAPES_YAML))))
    (td / "meta.yaml").write_text(yaml.safe_dump({
        "param_constraints": {
            "scaffold": False, "dims": ["B >= 1", "D >= 2"],
            "dtypes": {"x": "bfloat16"}, "dists": {"x": {"kind": "randn"}},
            "scalars": {}, "tensor_invariants": [],
        },
    }, sort_keys=False))
    (td / "oracle_pytorch.py").write_text(
        "def run(inputs):\n    return {'out': inputs['x'] + 1}\n")
    return td


def test_gen_self_candidates_constant_knob_no_crash():
    # codex residual #1: a knob with FEWER unique values than rows must not crash the base-row index.
    active = ["B", "D"]
    existing = [{"B": 4, "D": 8}, {"B": 4, "D": 8}, {"B": 4, "D": 16}]  # B constant (1 unique val)
    bounds = {"B": {"min": 1, "mod": None}, "D": {"min": 1, "mod": None}}
    cands = ext._gen_self_candidates(active, existing, bounds)
    assert cands, "should still generate candidates for a constant-knob task"


def test_write_extended_v2_rejects_id_collision(tmp_path):
    orig_v2, base = _v2_and_base()
    p = tmp_path / "shapes.yaml"
    # a new id "P1" collides with the existing P1 -> fail-closed, not a silent duplicate.
    with pytest.raises(RuntimeError, match="collide"):
        ext._write_extended_v2(orig_v2, base, [{"id": "P1", "knobs": {"B": 9, "D": 9}}], [], p)
    # a duplicate id among the new rows themselves -> also raises.
    with pytest.raises(RuntimeError, match="duplicate id"):
        ext._write_extended_v2(orig_v2, base,
                               [{"id": "P2", "knobs": {"B": 9, "D": 9}},
                                {"id": "P2", "knobs": {"B": 7, "D": 7}}], [], p)


def test_extend_task_rejects_struct(tmp_path, monkeypatch):
    td = tmp_path / "tasks" / "level1" / "L1_struct"
    td.mkdir(parents=True)
    # a compact-v2 doc carrying a struct input row -- extend must skip fail-closed (the element cap does not
    # model struct tensors, and v2_to_knob_spec cannot reconstruct struct).
    (td / "shapes.yaml").write_text(yaml.safe_dump({
        "schema_version": 2,
        "public": [{"id": "P1", "env": {"B": 4, "D": 8}, "inputs": [
            {"name": "x", "kind": "tensor", "shape": [4, 8], "dtype": "float32", "axes": ["B", "D"],
             "dist": "randn"},
            {"name": "meta", "kind": "struct", "struct": {"fields": {"y": 1}}}]}],
        "hidden": [],
    }))
    (td / "meta.yaml").write_text(yaml.safe_dump({"param_constraints": {"scaffold": False,
        "dims": ["B >= 1"], "dtypes": {}, "dists": {}, "scalars": {}, "tensor_invariants": []}}))
    (td / "oracle_pytorch.py").write_text("def run(inputs):\n    return {}\n")
    res = ext.extend_task(td, target_public=2, target_hidden=1, apply=False)
    assert "struct" in res.skipped_reason


def test_extend_task_end_to_end_pending_gpu(tmp_path, monkeypatch):
    td = _mk_synth_task(tmp_path)
    # point the tool's task-dir resolver at our synthetic tree
    monkeypatch.setattr(ext, "_task_dir", lambda tid: td)
    res = ext.extend_task(td, target_public=4, target_hidden=2, apply=False, no_borrow=True)
    surv = [c for c in res.candidates if c.status == "CPU_VALID_PENDING_GPU"]
    assert res.active_knobs == ["B", "D"]
    assert surv, "expected at least one CPU-valid candidate"
    assert all(c.status == "CPU_VALID_PENDING_GPU" for c in surv)
    # dry-run must NOT have written shapes.yaml (still the single original v2 public row).
    after = yaml.safe_load((td / "shapes.yaml").read_text())
    assert [r["id"] for r in after["public"]] == ["P1"]


def test_extend_task_apply_writes_canonical_v2(tmp_path, monkeypatch):
    td = _mk_synth_task(tmp_path)
    monkeypatch.setattr(ext, "_task_dir", lambda tid: td)
    before = yaml.safe_load((td / "shapes.yaml").read_text())
    res = ext.extend_task(td, target_public=3, target_hidden=0, apply=True, no_borrow=True)
    surv_pub = [c for c in res.candidates if c.status == "CPU_VALID_PENDING_GPU" and c.role == "public"]
    assert surv_pub
    after = yaml.safe_load((td / "shapes.yaml").read_text())
    # the original P1 row is byte-preserved + the survivors appended.
    assert after["public"][0] == before["public"][0]
    assert len(after["public"]) == 1 + len(surv_pub)
    # the written file is canonical + self-consistent (passes the corpus invariant).
    assert (td / "shapes.yaml").read_text() == to_yaml(generate_compact_shapes(v2_to_knob_spec(after)))
    # the meta shape_coverage followed the survivor counts (the pre-existing H1 hidden row is untouched).
    cov = yaml.safe_load((td / "meta.yaml").read_text())["runner_checks"]["shape_coverage"]
    assert cov["public"] == len(after["public"]) and cov["hidden"] == len(after["hidden"])


def test_value_ladder_diverse_and_bounded():
    # the ladder should span small->large with pow2 + non-pow2, respect min/mod/cap, be sorted+unique
    lad = ext._value_ladder({"min": 2, "mod": None}, 8192)
    assert lad == sorted(set(lad)), "ladder sorted+unique"
    assert all(2 <= v <= ext.DIM_CAP for v in lad), "within min..DIM_CAP"
    assert len(lad) >= 10, "diverse (>=10 values)"
    assert any(not ext._is_pow2(v) for v in lad), "has non-power-of-2 LLM dims"
    assert max(lad) >= 8192, "reaches large dims"
    # mod-constrained ladder: all even
    lad2 = ext._value_ladder({"min": 4, "mod": 2}, 1024)
    assert all(v % 2 == 0 for v in lad2), "mod respected"


def test_enhanced_generation_many_diverse_candidates():
    # 2-knob task -> single-axis sweep + grid should yield many (>=30) deduped candidates
    active = ["B", "D"]
    existing = [{"B": 4, "D": 64}, {"B": 16, "D": 256}, {"B": 32, "D": 1024}]
    bounds = {"B": {"min": 1, "mod": None}, "D": {"min": 2, "mod": None}}
    cands = ext._gen_self_candidates(active, existing, bounds)
    keys = {tuple(sorted(c.knobs.items())) for c in cands}
    assert len(keys) >= 30, f"expected >=30 distinct candidates, got {len(keys)}"
    # spans a wide D range
    ds = sorted({c.knobs.get("D") for c in cands if "D" in c.knobs})
    assert max(ds) / max(min(ds), 1) >= 50, "wide D range covered"
