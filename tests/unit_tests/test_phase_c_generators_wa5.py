"""WA.5: generator updates — phase_c_docs.py + generate_task_scaffolds.py.

Covers the WA.5 deliverables
(WA.5):

* ``phase_c_docs`` merges a ``scaffold: true`` ``param_constraints`` skeleton
  into ``meta.yaml`` without disturbing any other field;
* the **meta-skip** path triggers only for a human-reconciled block
  (``scaffold == false``); a missing block or one with ``scaffold == true``
  is (re-)seeded;
* the contract.md constraint mirror is a marker-fenced section, rewritten in
  place on every run — even for a meta-skipped task;
* a legacy hand-written ``## Constraints`` section is migrated in place;
* the contract.md mirror refresh is idempotent;
* the ``_S`` literal emitter routes a list to ``scalar_seq`` (not ``scalar``);
* ``generate_task_scaffolds`` emits a schema-conforming ``scaffold: true``
  ``param_constraints`` block, a constant-formula ``cost_model`` placeholder
  that passes the validator, and a marker-fenced contract.md mirror.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent.parent


def _load_module(name: str, rel: str):
    """Import a ``scripts/`` module by file path (it is not a package)."""
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pcd = _load_module("phase_c_docs", "scripts/corpus/generate_shapes_and_contract.py")
gts = _load_module("generate_task_scaffolds", "scripts/corpus/generate_task_scaffolds.py")
vt = _load_module("validate_tasks", "scripts/corpus/validate_tasks.py")


# --------------------------- fixtures ---------------------------

def _base_meta() -> dict:
    """A minimal meta.yaml mapping with a handful of non-param fields."""
    return {
        "schema_version": "1.0",
        "task_id": "L9_99_demo",
        "level": "L9",
        "idx": 99,
        "name": "demo_kernel",
        "repo": "vLLM",
        "tolerance_class": "elementwise",
        "oracle_simplified": "simple_pytorch_ok",
        "upstream": {"repo": "vllm-project/vllm", "path": "x/y.py"},
        "is_canary": False,
    }


def _write_meta(td: Path, meta: dict) -> Path:
    p = td / "meta.yaml"
    p.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
    return p


def _write_contract(td: Path, text: str) -> Path:
    p = td / "contract.md"
    p.write_text(text)
    return p


# ------------------ _S — list routes to scalar_seq ------------------

def test_S_scalar_for_plain_value():
    assert pcd._S(8) == {"scalar": 8}
    assert pcd._S(0.125) == {"scalar": 0.125}
    assert pcd._S("tanh") == {"scalar": "tanh"}


def test_S_scalar_seq_for_list():
    """A list literal must be emitted under ``scalar_seq``, not ``scalar``."""
    assert pcd._S([8, 8]) == {"scalar_seq": [8, 8]}
    assert pcd._S([3, 3, 2]) == {"scalar_seq": [3, 3, 2]}
    assert pcd._S((1, 2)) == {"scalar_seq": [1, 2]}
    # The shape overrides for L1_31 / L2_10 route their list literals through
    # _S, so the generated shapes.yaml carries scalar_seq.
    block_size = pcd.SHAPE_OVERRIDES["L2_18_w8a8_triton_block_scaled_mm"]
    assert "scalar_seq" not in str(block_size)  # no list literal there
    mrope = pcd.SHAPE_OVERRIDES["L1_19_triton_mrope_forward"]["mrope_section"]
    assert mrope == {"scalar_seq": [3, 3, 2]}


# ------------------ param_constraints skeleton ------------------

def test_scaffold_skeleton_is_schema_shape_valid():
    """The seeded skeleton passes the WA.1 validator schema-shape layer."""
    skel = pcd._param_constraints_scaffold()
    assert skel["scaffold"] is True
    assert skel["dims"] == [] and skel["dtypes"] == {}
    assert skel["dists"] == {} and skel["scalars"] == {}
    assert skel["tensor_invariants"] == []
    errs = vt._validate_param_constraints("L9_99", {"param_constraints": skel})
    assert errs == []


def test_is_reconciled_only_for_scaffold_false():
    assert pcd._is_reconciled({"scaffold": False}) is True
    assert pcd._is_reconciled({"scaffold": True}) is False
    assert pcd._is_reconciled({}) is False           # missing scaffold key
    assert pcd._is_reconciled({"scaffold": None}) is False
    assert pcd._is_reconciled(None) is False          # block absent


# ------------------ meta.yaml merge / meta-skip ------------------

def test_seed_meta_merges_without_clobbering_other_fields(tmp_path):
    """Seeding adds param_constraints; every other meta field is preserved."""
    meta = _base_meta()
    meta["runner_checks"] = {"shape_coverage": {"public": 5}}
    meta_path = _write_meta(tmp_path, meta)

    result = pcd._seed_meta_param_constraints(tmp_path, dry=False)
    assert result == "meta-seeded"

    after = yaml.safe_load(meta_path.read_text())
    # param_constraints injected as a scaffold skeleton.
    assert after["param_constraints"]["scaffold"] is True
    # every pre-existing field is byte-identical.
    for k, v in meta.items():
        assert after[k] == v


def test_seed_meta_skip_protects_reconciled_block(tmp_path):
    """A scaffold:false block is never overwritten (meta-skip path)."""
    meta = _base_meta()
    human = {
        "scaffold": False,
        "dims": ["D >= 8"],
        "dtypes": {"x": "float32"},
    }
    meta["param_constraints"] = human
    meta_path = _write_meta(tmp_path, meta)

    result = pcd._seed_meta_param_constraints(tmp_path, dry=False)
    assert result == "meta-skip"

    after = yaml.safe_load(meta_path.read_text())
    assert after["param_constraints"] == human  # untouched


def test_seed_meta_reseeds_scaffold_true_block(tmp_path):
    """A scaffold:true block is re-seeded (it is an un-reconciled draft)."""
    meta = _base_meta()
    meta["param_constraints"] = {"scaffold": True}  # partial / stale draft
    meta_path = _write_meta(tmp_path, meta)

    result = pcd._seed_meta_param_constraints(tmp_path, dry=False)
    assert result == "meta-seeded"

    after = yaml.safe_load(meta_path.read_text())
    # re-seeded to the full skeleton.
    assert after["param_constraints"] == pcd._param_constraints_scaffold()


def test_seed_meta_seeds_missing_block(tmp_path):
    """A task with no param_constraints block at all is seeded."""
    meta = _base_meta()
    assert "param_constraints" not in meta
    meta_path = _write_meta(tmp_path, meta)

    result = pcd._seed_meta_param_constraints(tmp_path, dry=False)
    assert result == "meta-seeded"
    after = yaml.safe_load(meta_path.read_text())
    assert after["param_constraints"]["scaffold"] is True


def test_seed_meta_dry_run_writes_nothing(tmp_path):
    meta = _base_meta()
    meta_path = _write_meta(tmp_path, meta)
    before = meta_path.read_text()

    result = pcd._seed_meta_param_constraints(tmp_path, dry=True)
    assert result == "meta-dry-seed"
    assert meta_path.read_text() == before


# ------------------ contract.md constraint mirror ------------------

def test_mirror_render_has_markers_and_subsections():
    skel = pcd._param_constraints_scaffold()
    mirror = pcd._render_constraint_mirror(skel)
    assert mirror.startswith(pcd._MIRROR_BEGIN)
    assert mirror.endswith(pcd._MIRROR_END)
    for heading in ("dims", "dtypes", "dists", "scalars", "tensor_invariants"):
        assert f"(`{heading}`)" in mirror
    assert "scaffold: true" in mirror


def test_mirror_render_reflects_reconciled_content():
    block = {
        "scaffold": False,
        "dims": ["D >= 8", "H % G == 0"],
        "dtypes": {"x": "float32"},
        "dists": {"q": {"kind": "randn"}},
        "scalars": {"eps": "0 < eps <= 1e-2"},
        "tensor_invariants": [
            {"type": "monotonic_nondecreasing", "args": {"input": "cu"}}
        ],
    }
    mirror = pcd._render_constraint_mirror(block)
    assert "`D >= 8`" in mirror
    assert "`H % G == 0`" in mirror
    assert "`x`: `float32`" in mirror
    assert "`q`: `randn`" in mirror
    assert "`eps`" in mirror
    assert "monotonic_nondecreasing" in mirror
    assert "scaffold: false" in mirror


def test_mirror_first_landing_appends_marker_block(tmp_path):
    """A contract.md with no marker and no legacy section gets the block
    appended at end of file, fenced by markers."""
    meta = _base_meta()
    meta["param_constraints"] = pcd._param_constraints_scaffold()
    _write_meta(tmp_path, meta)
    _write_contract(tmp_path, "# Contract\n\n## Identity\n\nfoo\n")

    result = pcd._refresh_contract_mirror(tmp_path, dry=False)
    assert result == "mirror-rewritten"

    text = (tmp_path / "contract.md").read_text()
    assert text.count(pcd._MIRROR_BEGIN) == 1
    assert text.count(pcd._MIRROR_END) == 1
    assert "## Identity" in text  # original content preserved
    assert text.index("## Identity") < text.index(pcd._MIRROR_BEGIN)


def test_mirror_migrates_legacy_constraints_section(tmp_path):
    """A legacy hand-written `## Constraints` section is replaced in place;
    the file ends with exactly one constraint section."""
    meta = _base_meta()
    meta["param_constraints"] = pcd._param_constraints_scaffold()
    _write_meta(tmp_path, meta)
    legacy = (
        "# Contract\n\n"
        "## Required ops\n\n- two dots\n\n"
        "## Constraints\n\n"
        "- `head_dim <= 128` at v0.1.\n"
        "- `BLOCK_M` is a constexpr.\n\n"
        "## Metamorphic checks\n\n1. zero V.\n"
    )
    _write_contract(tmp_path, legacy)

    result = pcd._refresh_contract_mirror(tmp_path, dry=False)
    assert result == "mirror-rewritten"

    text = (tmp_path / "contract.md").read_text()
    # legacy hand-written content is gone.
    assert "head_dim <= 128" not in text
    assert "BLOCK_M` is a constexpr" not in text
    # exactly one `## Constraints` section, and it is the marker mirror.
    assert text.count("## Constraints") == 1
    assert text.count(pcd._MIRROR_BEGIN) == 1
    # surrounding sections survive.
    assert "## Required ops" in text
    assert "## Metamorphic checks" in text


def test_mirror_rewrites_existing_marker_block_in_place(tmp_path):
    """An existing marker block is rewritten from current meta — content
    outside the markers is untouched."""
    meta = _base_meta()
    meta["param_constraints"] = pcd._param_constraints_scaffold()
    _write_meta(tmp_path, meta)
    stale_mirror = (
        f"{pcd._MIRROR_BEGIN}\n\n## Constraints\n\nSTALE TEXT\n\n"
        f"{pcd._MIRROR_END}"
    )
    _write_contract(
        tmp_path, f"# Contract\n\n## Identity\n\nkeep me\n\n{stale_mirror}\n")

    result = pcd._refresh_contract_mirror(tmp_path, dry=False)
    assert result == "mirror-rewritten"

    text = (tmp_path / "contract.md").read_text()
    assert "STALE TEXT" not in text
    assert "keep me" in text
    assert text.count(pcd._MIRROR_BEGIN) == 1


def test_mirror_refresh_runs_even_for_reconciled_task(tmp_path):
    """meta-skip protects meta.param_constraints only — the contract mirror
    is still refreshed from the (reconciled) block."""
    meta = _base_meta()
    meta["param_constraints"] = {
        "scaffold": False,
        "dims": ["D >= 16"],
        "dtypes": {}, "dists": {}, "scalars": {}, "tensor_invariants": [],
    }
    _write_meta(tmp_path, meta)
    _write_contract(tmp_path, "# Contract\n\n## Identity\n\nx\n")

    pc = pcd.process_param_constraints(tmp_path, dry=False)
    meta_token, mirror_token = pc.split("/")
    assert meta_token == "meta-skip"
    assert mirror_token == "mirror-rewritten"
    text = (tmp_path / "contract.md").read_text()
    assert "`D >= 16`" in text  # reconciled content reflected


def test_mirror_refresh_is_idempotent(tmp_path):
    """A second refresh with unchanged meta is a no-op."""
    meta = _base_meta()
    meta["param_constraints"] = pcd._param_constraints_scaffold()
    _write_meta(tmp_path, meta)
    _write_contract(tmp_path, "# Contract\n\n## Identity\n\nx\n")

    assert pcd._refresh_contract_mirror(tmp_path, dry=False) == "mirror-rewritten"
    first = (tmp_path / "contract.md").read_text()
    assert pcd._refresh_contract_mirror(tmp_path, dry=False) == "mirror-unchanged"
    assert (tmp_path / "contract.md").read_text() == first


# ------------------ generate_task_scaffolds ------------------

def _demo_row():
    return gts.RawRow(
        md_id="L9-99", level="L9", idx=99, name="_demo_kernel",
        repo_raw="vLLM", cv_vv="VV", rationale="demo",
        difficulty_raw="中", source_path="x/y.py",
    )


def test_scaffold_meta_has_schema_conforming_param_constraints():
    meta = yaml.safe_load(gts.render_meta(_demo_row(), "deadbeef"))
    pc = meta["param_constraints"]
    assert pc["scaffold"] is True
    assert pc["dims"] == [] and pc["dtypes"] == {}
    assert pc["dists"] == {} and pc["scalars"] == {}
    assert pc["tensor_invariants"] == []
    errs = vt._validate_param_constraints("L9_99", meta)
    assert errs == []


def test_scaffold_meta_has_validator_passing_cost_model(tmp_path):
    """The scaffold cost_model placeholder is a constant-formula block that
    passes both the schema parse and the formula-evaluability check."""
    meta = yaml.safe_load(gts.render_meta(_demo_row(), "deadbeef"))
    cm = meta["cost_model"]
    assert cm["flops_class"] == "vector"
    assert cm["bytes_resource"] == "hbm"
    # validator: schema parse + formula evaluability against an empty knob
    # universe (the scaffold shapes.yaml declares no size_knobs).
    errs = vt._validate_cost_model("L9_99", meta, tmp_path)
    assert errs == []


def test_scaffold_contract_has_marker_fenced_mirror():
    contract = gts.PLACEHOLDER_CONTRACT.format(
        task_id="L9_99_demo", md_id="L9-99", repo="vLLM",
        name="_demo_kernel", source_path="x/y.py", cv_vv="VV",
        difficulty="medium", category="other", rationale="demo",
        tolerance_class="elementwise",
    )
    assert contract.count(pcd._MIRROR_BEGIN) == 1
    assert contract.count(pcd._MIRROR_END) == 1
    assert "## Constraints" in contract


def test_scaffold_contract_mirror_is_rewritable_by_phase_c(tmp_path):
    """The marker block generate_task_scaffolds emits is a valid target for
    phase_c_docs' in-place rewrite (the two generators stay symmetric)."""
    contract = gts.PLACEHOLDER_CONTRACT.format(
        task_id="L9_99_demo", md_id="L9-99", repo="vLLM",
        name="_demo_kernel", source_path="x/y.py", cv_vv="VV",
        difficulty="medium", category="other", rationale="demo",
        tolerance_class="elementwise",
    )
    meta = yaml.safe_load(gts.render_meta(_demo_row(), "deadbeef"))
    # hand-reconcile the block so the rewrite has new content to render.
    meta["param_constraints"] = {
        "scaffold": False, "dims": ["L >= 16"],
        "dtypes": {}, "dists": {}, "scalars": {}, "tensor_invariants": [],
    }
    _write_meta(tmp_path, meta)
    _write_contract(tmp_path, contract)

    result = pcd._refresh_contract_mirror(tmp_path, dry=False)
    assert result == "mirror-rewritten"
    text = (tmp_path / "contract.md").read_text()
    assert text.count(pcd._MIRROR_BEGIN) == 1
    assert "`L >= 16`" in text


# ------------------ corpus-level mirror/meta consistency ------------------

def _corpus_task_dirs() -> list[Path]:
    """All 101 real corpus task directories (canary tasks excluded).

    ``tasks/_canary/`` is skipped via the underscore-prefix rule — canary
    tasks are exempt from ``param_constraints`` (WA.1) and therefore from the
    contract.md constraint mirror.
    """
    return list(pcd._iter_task_dirs())


def _extract_mirror_section(contract_text: str) -> str:
    """Return the marker-fenced mirror section of a contract.md, inclusive.

    Args:
        contract_text: the full contract.md text.

    Returns:
        The text from :data:`pcd._MIRROR_BEGIN` through :data:`pcd._MIRROR_END`
        inclusive, or ``""`` when no marker pair is present.
    """
    begin = contract_text.find(pcd._MIRROR_BEGIN)
    end = contract_text.find(pcd._MIRROR_END)
    if begin == -1 or end == -1 or end <= begin:
        return ""
    return contract_text[begin:end + len(pcd._MIRROR_END)]


def test_corpus_count():
    """Guards the corpus-consistency test below from silently degrading to a
    no-op if task discovery breaks. Corpus is 101 scored tasks (canary
    excluded); update this if the corpus intentionally changes."""
    assert len(_corpus_task_dirs()) == 101


@pytest.mark.parametrize(
    "task_dir", _corpus_task_dirs(), ids=lambda d: d.name)
def test_corpus_contract_mirror_matches_meta(task_dir: Path):
    """Every corpus task's contract.md mirror equals the rendering of its
    ``meta.param_constraints``.

    WA.5 invariant: ``meta.yaml`` is the single authoritative source and the
    contract.md constraint section is a pure derived mirror. A mismatch means
    ``meta.param_constraints`` was edited (e.g. a WB.1 reconcile) without
    re-running the ``phase_c_docs.py --refresh-mirror-only`` mirror refresh —
    this test fails so that drift is caught.
    """
    contract_path = task_dir / "contract.md"
    meta_path = task_dir / "meta.yaml"
    assert contract_path.is_file(), f"{task_dir.name}: missing contract.md"
    assert meta_path.is_file(), f"{task_dir.name}: missing meta.yaml"

    meta = yaml.safe_load(meta_path.read_text()) or {}
    expected = pcd._render_constraint_mirror(meta.get("param_constraints"))
    actual = _extract_mirror_section(contract_path.read_text())

    assert actual, (
        f"{task_dir.name}: contract.md has no param_constraints mirror "
        f"marker — run `python3 scripts/corpus/generate_shapes_and_contract.py --refresh-mirror-only`")
    assert actual == expected, (
        f"{task_dir.name}: contract.md mirror is stale vs "
        f"meta.param_constraints — run "
        f"`python3 scripts/corpus/generate_shapes_and_contract.py --refresh-mirror-only`")


def test_emit_shapes_yaml_is_canonical_v2():
    # U10: a scaffolded NEW task is born as compact v2 (canonical, comment-free), not knob -- so it never
    # diverges from the migrated corpus / the self-consistency test.
    sys.path.insert(0, str(REPO))
    from ascendbench.input_spec import shape_format as sf
    from ascendbench.input_spec.compact_shapes import generate_compact_shapes
    from scripts.corpus.migrate_to_v2 import to_yaml
    from scripts.corpus.v2_authoring import v2_to_knob_spec
    spec = {"x": {"dims": ["B", "D"], "dtype": "float32", "dist": "randn"},
            "idx": {"dims": ["B"], "dtype": "int64", "dist": "randint:0:N"},
            "eps": {"scalar": 1e-5}}
    txt = pcd._emit_shapes_yaml(spec)
    doc = yaml.safe_load(txt)
    assert doc["schema_version"] == 2
    assert sf.is_explicit_shapes_doc(doc) is True
    assert to_yaml(doc) == txt                                            # canonical, comment-free
    assert to_yaml(generate_compact_shapes(v2_to_knob_spec(doc))) == txt  # self-consistent
    assert "#" not in txt                                                 # no comment header / manual note
    assert list(sf.public_ids(doc)) == ["P1", "P2", "P3", "P4", "P5"]


def test_manual_scaffold_gated_by_valid_release_scope(tmp_path):
    # U10: a phase-c-MANUAL scaffold's shapes.yaml is comment-free v2, so the gate moves to a structured
    # meta.release_scope.excluded marker -- which must be a COMPLETE record (validate_tasks requires
    # excluded/reason/excluded_at/excluded_by when excluded is True).
    td = tmp_path / "L2_11_rejection_sample"   # the hardcoded _MANUAL task id
    td.mkdir()
    (td / "oracle_pytorch.py").write_text("def run(inputs):\n    x = inputs['x']\n    return {'out': x}\n")
    (td / "meta.yaml").write_text(yaml.safe_dump({"task_id": "L2_11_rejection_sample"}))
    status = pcd.process_body(td, dry=False)
    assert "[MANUAL]" in status
    m = yaml.safe_load((td / "meta.yaml").read_text())
    rs = m["release_scope"]
    assert rs["excluded"] is True and rs["reason"] and rs["excluded_by"]
    assert vt._validate_release_scope("L2_11_rejection_sample", m) == []   # a complete, valid record
    assert "PHASE-C-MANUAL" not in (td / "shapes.yaml").read_text()        # gated via meta, not a comment


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
