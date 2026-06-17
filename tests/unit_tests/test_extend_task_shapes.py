"""Unit 2: per-task append-only shape-extension tool (scripts/corpus/extend_task_shapes.py).

Covers the mechanics that must be correct before any shape is appended: active-knob detection,
NPUKernelBench dim-mapping (codex residual #4), non-pow2/mod-respecting value generation, canonical
dedup, byte-preserving text splice, and determinism. The CPU-validation == pending_gpu contract is
exercised on a synthetic CPU-runnable task.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

ext = importlib.import_module("scripts.corpus.extend_task_shapes")


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


# ── byte-preserving text splice ───────────────────────────────────────────────────────────────────

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


def test_splice_append_preserves_existing_bytes(tmp_path):
    p = tmp_path / "shapes.yaml"
    p.write_text(_SHAPES_YAML)
    ext._splice_append(
        p,
        public_rows=[{"id": "P2", "name": "borrow_x", "knobs": {"B": 128, "D": 4096}}],
        hidden_rows=[{"id": "H2", "name": "edge", "knobs": {"B": 1, "D": 2}}],
    )
    new = p.read_text()
    # original lines all still present verbatim (append-only, comment preserved)
    for line in _SHAPES_YAML.splitlines():
        assert line in new.splitlines(), f"existing line lost: {line!r}"
    doc = yaml.safe_load(new)
    assert {s["id"] for s in doc["public_shapes"]} == {"P1", "P2"}
    assert {s["id"] for s in doc["hidden_shapes"]} == {"H1", "H2"}
    # the new rows materialise correctly
    assert doc["public_shapes"][1]["knobs"] == {"B": 128, "D": 4096}
    # inputs + size_knobs untouched
    assert doc["inputs"]["x"]["dtype"] == "bfloat16"
    assert doc["size_knobs"] == {"B": 8, "D": 16}


def test_splice_append_detects_corruption(tmp_path, monkeypatch):
    # if the splice were to alter an existing row, the parse-back guard must raise.
    p = tmp_path / "shapes.yaml"
    p.write_text(_SHAPES_YAML)
    orig_row_text = ext._yaml_row_text

    def _bad_row_text(row):
        # simulate a buggy renderer that also mutates an existing id's knobs in the new block
        return orig_row_text(row).replace("P2", "P1")  # collide with existing id -> mismatch

    monkeypatch.setattr(ext, "_yaml_row_text", _bad_row_text)
    try:
        ext._splice_append(p, public_rows=[{"id": "P2", "name": "x", "knobs": {"B": 9, "D": 9}}],
                           hidden_rows=[])
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "byte-preservation guard must reject a splice that changes an existing row"


# ── end-to-end on a synthetic CPU-runnable task: survivors are pending_gpu, append-only ────────────

def _mk_synth_task(tmp_path: Path) -> Path:
    td = tmp_path / "tasks" / "level1" / "L1_synth"
    td.mkdir(parents=True)
    (td / "shapes.yaml").write_text(_SHAPES_YAML)
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


def test_splice_rejects_id_collision(tmp_path):
    p = tmp_path / "shapes.yaml"
    p.write_text(_SHAPES_YAML)
    # new id "P1" collides with the existing P1 -> must raise (fail-closed), not silently duplicate.
    try:
        ext._splice_append(p, public_rows=[{"id": "P1", "name": "x", "knobs": {"B": 9, "D": 9}}],
                           hidden_rows=[])
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    # duplicate id among the new rows themselves -> also raises.
    try:
        ext._splice_append(p, public_rows=[{"id": "P2", "name": "x", "knobs": {"B": 9, "D": 9}},
                                           {"id": "P2", "name": "y", "knobs": {"B": 7, "D": 7}}],
                           hidden_rows=[])
        raised2 = False
    except RuntimeError:
        raised2 = True
    assert raised2


def test_extend_task_rejects_struct(tmp_path, monkeypatch):
    td = tmp_path / "tasks" / "level1" / "L1_struct"
    td.mkdir(parents=True)
    (td / "shapes.yaml").write_text(yaml.safe_dump({
        "size_knobs": {"B": 4, "D": 8},
        "public_shapes": [{"id": "P1", "knobs": {"B": 1, "D": 2}}],
        "inputs": {"meta": {"struct": True, "fields": {"x": {"dims": ["B", "D"]}}}},
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
    # dry-run must NOT have written shapes.yaml
    assert yaml.safe_load((td / "shapes.yaml").read_text())["public_shapes"] == [
        {"id": "P1", "name": "tiny", "knobs": {"B": 1, "D": 2}}]


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
