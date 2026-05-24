"""Tests for ``automation.lib.file_blocks`` (Unit 1.6).

Covers parser (META + FILE), Layer 1..5 validators, validate_patch /
apply_patch contract, byte-equal round-trip with snapshot/rollback.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"


@pytest.fixture(autouse=True)
def _ensure_scripts_on_path():
    inserted = False
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
        inserted = True
    yield
    if inserted:
        try:
            sys.path.remove(str(_SCRIPTS))
        except ValueError:
            pass


TASK_ID = "L1_06_swiglu"
LEVEL = "L1"


def _file_block(name: str, content: str) -> str:
    return f"### FILE: tasks/{LEVEL}/{TASK_ID}/{name}\n{content}\n### END\n"


def _meta_block(payload: dict) -> str:
    import json as _j
    return f"### META\n{_j.dumps(payload)}\n### END\n"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_parse_meta_and_blocks_in_order():
    from automation.lib import file_blocks as fb

    meta = _meta_block({"decision": "patch-shapes", "rationale": "x", "evidence": {}})
    f1 = _file_block("shapes.yaml", "public_shapes:\n- {id: s1}\n")
    f2 = _file_block("meta.yaml", "schema_version: 1.0\n")
    f3 = _file_block("oracle_pytorch.py", "def oracle(x):\n    return x * 2\n")
    text = meta + "\n" + f1 + "\n" + f2 + "\n" + f3
    m = fb.parse_meta(text)
    assert m is not None
    assert m.decision == "patch-shapes"
    blocks = fb.parse_blocks(text)
    assert [b.path.name for b in blocks] == ["shapes.yaml", "meta.yaml", "oracle_pytorch.py"]


def test_parse_order_insensitive_with_prose():
    """META can come anywhere; arbitrary prose outside blocks is ignored."""
    from automation.lib import file_blocks as fb

    text = (
        "This patch is needed because the resolver flagged missing param size_M.\n\n"
        + _file_block("shapes.yaml", "public_shapes: []\n")
        + "\nrationale prose explaining the decision\n\n"
        + _meta_block({"decision": "patch-shapes", "rationale": "x", "evidence": {}})
        + "\nmore prose at the end\n"
    )
    m = fb.parse_meta(text)
    assert m is not None
    blocks = fb.parse_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].path.name == "shapes.yaml"


def test_parse_zero_meta_returns_none():
    from automation.lib import file_blocks as fb

    text = _file_block("shapes.yaml", "x: 1\n")
    assert fb.parse_meta(text) is None


def test_parse_multiple_meta_raises():
    from automation.lib import file_blocks as fb

    text = _meta_block({"decision": "patch-shapes"}) + _meta_block({"decision": "patch-shapes"})
    with pytest.raises(fb.MultipleMetaBlocksError):
        fb.parse_meta(text)


# ---------------------------------------------------------------------------
# validate_meta
# ---------------------------------------------------------------------------

def test_validate_meta_unknown_decision_error():
    from automation.lib import file_blocks as fb

    m = fb.MetaBlock(decision="unknown-banana", payload={"decision": "unknown-banana"})
    errs = fb.validate_meta(m, expected_decision_set={"patch-shapes"}, expected_keys=set())
    assert any(e.code == "bad_decision" for e in errs)


def test_validate_meta_missing_keys_error():
    from automation.lib import file_blocks as fb

    m = fb.MetaBlock(decision="patch-shapes", payload={"decision": "patch-shapes"})
    errs = fb.validate_meta(
        m, expected_decision_set={"patch-shapes"},
        expected_keys={"decision", "rationale", "evidence"},
    )
    codes = [e.code for e in errs]
    assert "missing_keys" in codes


def test_validate_meta_oracle_action_placeholder_error():
    from automation.lib import file_blocks as fb

    m = fb.MetaBlock(
        decision="rewrite-primitives",
        payload={"decision": "rewrite-primitives", "oracle_action": "placeholder"},
    )
    errs = fb.validate_meta(
        m, expected_decision_set={"rewrite-primitives"},
        expected_keys={"decision"},
    )
    assert any(e.code == "bad_oracle_action" for e in errs)


def test_validate_meta_unknown_field_is_warning_not_error():
    from automation.lib import file_blocks as fb

    m = fb.MetaBlock(
        decision="patch-shapes",
        payload={"decision": "patch-shapes", "future_field": "x"},
    )
    errs = fb.validate_meta(
        m, expected_decision_set={"patch-shapes"}, expected_keys={"decision"},
    )
    assert errs == []


# ---------------------------------------------------------------------------
# Layer 1 -- path whitelist
# ---------------------------------------------------------------------------

def test_layer1_path_traversal_rejected():
    from automation.lib import file_blocks as fb

    # Build a manual block since our _file_block helper rebases path
    raw = (
        _meta_block({"decision": "patch-shapes", "rationale": "x", "evidence": {}})
        + "### FILE: ../../etc/passwd\nbad\n### END\n"
    )
    m = fb.parse_meta(raw)
    blocks = fb.parse_blocks(raw)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"patch-shapes"}, expected_keys={"decision"},
    )
    assert any(e.layer == 1 for e in errs)


def test_layer1_wrong_task_id_rejected():
    from automation.lib import file_blocks as fb

    raw = (
        _meta_block({"decision": "patch-shapes", "rationale": "x", "evidence": {}})
        + f"### FILE: tasks/{LEVEL}/OTHER_TASK/shapes.yaml\nx: 1\n### END\n"
    )
    m = fb.parse_meta(raw)
    blocks = fb.parse_blocks(raw)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"patch-shapes"}, expected_keys={"decision"},
    )
    assert any(e.code == "path_task_id" for e in errs)


def test_layer1_wrong_filename_rejected():
    from automation.lib import file_blocks as fb

    raw = (
        _meta_block({"decision": "patch-shapes", "rationale": "x", "evidence": {}})
        + _file_block("notes.md", "do not write here\n")
    )
    m = fb.parse_meta(raw)
    blocks = fb.parse_blocks(raw)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"patch-shapes"}, expected_keys={"decision"},
    )
    assert any(e.code == "path_filename" for e in errs)


# ---------------------------------------------------------------------------
# Layer 2 -- import leak
# ---------------------------------------------------------------------------

def test_layer2_vllm_import_rejected():
    from automation.lib import file_blocks as fb

    text = _meta_block({"decision": "rewrite-primitives", "oracle_action": "rewrite-primitives"}) + _file_block(
        "oracle_pytorch.py", "from vllm import attention\n\ndef oracle(x):\n    return x + 1\n",
    )
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"rewrite-primitives"}, expected_keys={"decision"},
    )
    assert any(e.code == "forbidden_import" for e in errs)


def test_layer2_sglang_import_rejected():
    from automation.lib import file_blocks as fb

    text = _meta_block({"decision": "rewrite-primitives", "oracle_action": "rewrite-primitives"}) + _file_block(
        "oracle_pytorch.py", "import sglang\n\ndef oracle(x):\n    return x + 1\n",
    )
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"rewrite-primitives"}, expected_keys={"decision"},
    )
    assert any(e.code == "forbidden_import" for e in errs)


# ---------------------------------------------------------------------------
# Layer 3 -- stub signature
# ---------------------------------------------------------------------------

def test_layer3_pass_only_body_rejected():
    from automation.lib import file_blocks as fb

    text = _meta_block({"decision": "rewrite-primitives", "oracle_action": "rewrite-primitives"}) + _file_block(
        "oracle_pytorch.py", "def oracle(x):\n    pass\n",
    )
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"rewrite-primitives"}, expected_keys={"decision"},
    )
    assert any(e.code == "stub_pass" for e in errs)


def test_layer3_not_implemented_only_rejected():
    from automation.lib import file_blocks as fb

    text = _meta_block({"decision": "rewrite-primitives", "oracle_action": "rewrite-primitives"}) + _file_block(
        "oracle_pytorch.py",
        "def oracle(x):\n    raise NotImplementedError('stub')\n",
    )
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"rewrite-primitives"}, expected_keys={"decision"},
    )
    assert any(e.code == "stub_not_implemented" for e in errs)


def test_layer3_empty_like_only_rejected():
    from automation.lib import file_blocks as fb

    text = _meta_block({"decision": "rewrite-primitives", "oracle_action": "rewrite-primitives"}) + _file_block(
        "oracle_pytorch.py", "import torch\n\ndef oracle(x):\n    return torch.empty_like(x)\n",
    )
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"rewrite-primitives"}, expected_keys={"decision"},
    )
    assert any(e.code == "stub_empty_like" for e in errs)


# ---------------------------------------------------------------------------
# Layer 4 -- META x FILE consistency
# ---------------------------------------------------------------------------

def test_layer4_patch_shapes_requires_shapes_yaml():
    from automation.lib import file_blocks as fb

    text = _meta_block({"decision": "patch-shapes", "rationale": "x", "evidence": {}}) + _file_block(
        "meta.yaml", "schema_version: 1.0\n",
    )
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"patch-shapes"}, expected_keys={"decision"},
    )
    assert any(e.code == "missing_shapes_yaml" for e in errs)


def test_layer4_tolerance_bump_rejects_oracle():
    from automation.lib import file_blocks as fb

    text = (
        _meta_block({"decision": "tolerance-bump", "rationale": "x"})
        + _file_block("meta.yaml", "schema_version: 1.0\n")
        + _file_block("oracle_pytorch.py", "def f(x):\n    return x + 1\n")
    )
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"tolerance-bump"}, expected_keys={"decision"},
    )
    assert any(e.code == "unexpected_oracle" for e in errs)


def test_layer4_defer_unchanged_rejects_oracle():
    from automation.lib import file_blocks as fb

    defer_meta = {
        "phase_b_capture_required": True,
        "phase_b_capture": {
            "status": "PENDING",
            "upstream_callable": {
                "module": "vllm.model_executor.layers.layernorm",
                "function": "rms_norm",
            },
        },
        "phase_a_prime": {
            "skip_invocation": True,
            "status": "SKIPPED_BY_DESIGN",
            "reason": "upstream gating",
        },
        "oracle_simplified": "upstream_oracle_required",
    }
    meta_yaml_text = yaml.safe_dump(defer_meta, sort_keys=False)
    text = (
        _meta_block({"decision": "defer-to-upstream-oracle", "oracle_action": "unchanged"})
        + _file_block("meta.yaml", meta_yaml_text)
        + _file_block("oracle_pytorch.py", "def f(x):\n    return x + 1\n")
    )
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"defer-to-upstream-oracle"}, expected_keys={"decision"},
    )
    assert any(e.code == "unexpected_oracle_for_unchanged" for e in errs)


# ---------------------------------------------------------------------------
# Layer 5 -- defer-to-* field validation
# ---------------------------------------------------------------------------

def _ok_defer_meta() -> dict:
    return {
        "phase_b_capture_required": True,
        "phase_b_capture": {
            "status": "PENDING",
            "upstream_callable": {
                "module": "vllm.model_executor.layers.layernorm",
                "function": "rms_norm",
            },
        },
        "phase_a_prime": {
            "skip_invocation": True,
            "status": "SKIPPED_BY_DESIGN",
            "reason": "upstream gating; cannot invoke isolated.",
        },
        "oracle_simplified": "upstream_oracle_required",
    }


def _defer_text(meta_dict: dict, oracle_action="unchanged",
                decision="defer-to-golden") -> str:
    return (
        _meta_block({"decision": decision, "oracle_action": oracle_action,
                     "upstream_callable": meta_dict["phase_b_capture"]["upstream_callable"]})
        + _file_block("meta.yaml", yaml.safe_dump(meta_dict, sort_keys=False))
    )


def test_layer5_defer_happy_full_5_fields():
    from automation.lib import file_blocks as fb

    text = _defer_text(_ok_defer_meta())
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"defer-to-golden"}, expected_keys={"decision"},
    )
    assert errs == [], f"expected clean validation, got: {errs}"


def test_layer5_missing_phase_b_capture_required():
    from automation.lib import file_blocks as fb

    bad = _ok_defer_meta()
    del bad["phase_b_capture_required"]
    text = _defer_text(bad)
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"defer-to-golden"}, expected_keys={"decision"},
    )
    assert any(e.code == "bad_phase_b_capture_required" for e in errs)


def test_layer5_missing_oracle_simplified():
    from automation.lib import file_blocks as fb

    bad = _ok_defer_meta()
    del bad["oracle_simplified"]
    text = _defer_text(bad, decision="defer-to-upstream-oracle")
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"defer-to-upstream-oracle"}, expected_keys={"decision"},
    )
    assert any(e.code == "bad_oracle_simplified" for e in errs)


@pytest.mark.parametrize("bad_module", [
    "vllm..layernorm",
    "vllm/layernorm",
    "3vllm.layernorm",
    ".vllm.layernorm",
    "vllm. layernorm",
])
def test_layer5_module_non_canonical(bad_module):
    from automation.lib import file_blocks as fb

    bad = _ok_defer_meta()
    bad["phase_b_capture"]["upstream_callable"]["module"] = bad_module
    text = _defer_text(bad)
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"defer-to-golden"}, expected_keys={"decision"},
    )
    assert any(e.code == "bad_upstream_module" for e in errs)


@pytest.mark.parametrize("bad_fn", ["rms-norm", "3norm", "fn name", ""])
def test_layer5_function_non_identifier(bad_fn):
    from automation.lib import file_blocks as fb

    bad = _ok_defer_meta()
    bad["phase_b_capture"]["upstream_callable"]["function"] = bad_fn
    text = _defer_text(bad)
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"defer-to-golden"}, expected_keys={"decision"},
    )
    assert any(e.code == "bad_upstream_function" for e in errs)


def test_layer5_skip_invocation_false():
    from automation.lib import file_blocks as fb

    bad = _ok_defer_meta()
    bad["phase_a_prime"]["skip_invocation"] = False
    text = _defer_text(bad)
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"defer-to-golden"}, expected_keys={"decision"},
    )
    assert any(e.code == "bad_skip_invocation" for e in errs)


def test_layer5_phase_b_capture_status_unknown():
    from automation.lib import file_blocks as fb

    bad = _ok_defer_meta()
    bad["phase_b_capture"]["status"] = "UNKNOWN"
    text = _defer_text(bad)
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"defer-to-golden"}, expected_keys={"decision"},
    )
    assert any(e.code == "bad_phase_b_capture_status" for e in errs)


def test_layer5_candidate_meta_yaml_syntax_error():
    from automation.lib import file_blocks as fb

    text = (
        _meta_block({"decision": "defer-to-golden"})
        + _file_block("meta.yaml", "phase_b_capture: [\n")
    )
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"defer-to-golden"}, expected_keys={"decision"},
    )
    assert any(e.code == "meta_yaml_syntax" for e in errs)


# ---------------------------------------------------------------------------
# Layer 5 -- rewrite-primitives high-level op scan
# ---------------------------------------------------------------------------

def test_layer5_rewrite_primitives_f_gelu_rejected():
    from automation.lib import file_blocks as fb

    text = _meta_block({"decision": "rewrite-primitives", "oracle_action": "rewrite-primitives"}) + _file_block(
        "oracle_pytorch.py",
        "import torch\nimport torch.nn.functional as F\n\n"
        "def oracle(x):\n    return F.gelu(x)\n",
    )
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"rewrite-primitives"}, expected_keys={"decision"},
    )
    assert any(e.code == "high_level_op" for e in errs)


def test_layer5_rewrite_primitives_nn_silu_class_rejected():
    from automation.lib import file_blocks as fb

    text = _meta_block({"decision": "rewrite-primitives", "oracle_action": "rewrite-primitives"}) + _file_block(
        "oracle_pytorch.py",
        "import torch\n\ndef oracle(x):\n    return torch.nn.SiLU()(x)\n",
    )
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"rewrite-primitives"}, expected_keys={"decision"},
    )
    assert any(e.code == "high_level_nn_class" for e in errs)


def test_layer5_rewrite_primitives_softmax_rejected():
    from automation.lib import file_blocks as fb

    text = _meta_block({"decision": "rewrite-primitives", "oracle_action": "rewrite-primitives"}) + _file_block(
        "oracle_pytorch.py",
        "import torch.nn.functional as F\n\n"
        "def oracle(x):\n    return F.softmax(x, dim=-1)\n",
    )
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"rewrite-primitives"}, expected_keys={"decision"},
    )
    assert any(e.code == "high_level_op" for e in errs)


def test_layer5_defer_to_golden_with_rewrite_oracle_has_leak():
    from automation.lib import file_blocks as fb

    ok = _ok_defer_meta()
    text = (
        _meta_block({"decision": "defer-to-golden", "oracle_action": "rewrite-primitives",
                     "upstream_callable": ok["phase_b_capture"]["upstream_callable"]})
        + _file_block("meta.yaml", yaml.safe_dump(ok, sort_keys=False))
        + _file_block(
            "oracle_pytorch.py",
            "import torch.nn.functional as F\n\n"
            "def oracle(x):\n    return F.layer_norm(x, (16,))\n",
        )
    )
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"defer-to-golden"}, expected_keys={"decision"},
    )
    assert any(e.code == "high_level_op" for e in errs)


def test_layer5_rewrite_primitives_low_level_passes():
    """A genuinely low-level math oracle should not flag Layer 5."""
    from automation.lib import file_blocks as fb

    text = _meta_block({"decision": "rewrite-primitives", "oracle_action": "rewrite-primitives"}) + _file_block(
        "oracle_pytorch.py",
        "import torch\n\n"
        "def oracle(x, eps=1e-5):\n"
        "    norm = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)\n"
        "    return x / norm\n",
    )
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"rewrite-primitives"}, expected_keys={"decision"},
    )
    assert errs == [], f"expected clean validation, got: {errs}"


def test_layer5_candidate_oracle_syntax_error():
    from automation.lib import file_blocks as fb

    text = _meta_block({"decision": "rewrite-primitives", "oracle_action": "rewrite-primitives"}) + _file_block(
        "oracle_pytorch.py", "def run(:\n    return x\n",
    )
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, _REPO_ROOT,
        expected_decision_set={"rewrite-primitives"}, expected_keys={"decision"},
    )
    assert any(e.code == "syntax_error" for e in errs)


# ---------------------------------------------------------------------------
# Round-trip: validate_patch (clean) -> snapshot -> apply_patch -> rollback
# ---------------------------------------------------------------------------

def _make_task_dir(repo_root: Path, level: str = LEVEL, task_id: str = TASK_ID) -> Path:
    td = repo_root / "tasks" / level / task_id
    td.mkdir(parents=True)
    (td / "meta.yaml").write_text("schema_version: 1.0\nname: original\n", encoding="utf-8")
    (td / "shapes.yaml").write_text("public_shapes: []\n", encoding="utf-8")
    (td / "oracle_pytorch.py").write_text("def oracle(x):\n    return x\n", encoding="utf-8")
    return td


def test_validate_fail_does_not_mutate_task_dir(tmp_path):
    import hashlib

    from automation.lib import file_blocks as fb

    repo = tmp_path
    td = _make_task_dir(repo)
    sigs_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in td.iterdir()}

    # decision=patch-shapes but FILE has no shapes.yaml -> consistency error
    text = _meta_block({"decision": "patch-shapes", "rationale": "x", "evidence": {}}) + _file_block(
        "meta.yaml", "schema_version: 1.0\nname: mutated\n",
    )
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, repo,
        expected_decision_set={"patch-shapes"}, expected_keys={"decision"},
    )
    assert errs != []
    # caller MUST not call apply_patch on non-empty errors
    sigs_after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in td.iterdir()}
    assert sigs_before == sigs_after


def test_apply_patch_writes_files_atomically(tmp_path):
    from automation.lib import file_blocks as fb

    repo = tmp_path
    _make_task_dir(repo)
    new_shapes = "public_shapes:\n  - {id: s1, name: small, knobs: {M: 16}}\n"
    text = _meta_block({"decision": "patch-shapes", "rationale": "x", "evidence": {}}) + _file_block(
        "shapes.yaml", new_shapes,
    )
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, repo,
        expected_decision_set={"patch-shapes"}, expected_keys={"decision"},
    )
    assert errs == []
    fb.apply_patch(m, blocks, TASK_ID, repo)
    parsed = yaml.safe_load(
        (repo / "tasks" / LEVEL / TASK_ID / "shapes.yaml").read_text(encoding="utf-8")
    )
    assert parsed["public_shapes"][0]["id"] == "s1"


def test_apply_patch_rollback_round_trip(tmp_path, monkeypatch):
    from automation.lib import atomic
    from automation.lib import file_blocks as fb

    repo = tmp_path
    td = _make_task_dir(repo)
    snap_tar = tmp_path / "snap" / "baseline.tar.gz"
    snap_manifest = tmp_path / "snap" / "manifest.json"
    atomic.snapshot_task_baseline(td, snap_tar, snap_manifest)

    # Build a patch that should validate cleanly...
    new_meta_content = "schema_version: 1.0\nname: ATTEMPTED\n"
    new_shapes = "public_shapes: [{id: s1, name: small, knobs: {M: 16}}]\n"
    text = (
        _meta_block({"decision": "patch-shapes", "rationale": "x", "evidence": {}})
        + _file_block("shapes.yaml", new_shapes)
        + _file_block("meta.yaml", new_meta_content)
    )
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, repo,
        expected_decision_set={"patch-shapes"}, expected_keys={"decision"},
    )
    assert errs == []

    # ...but make the second write fail to provoke a partial apply.
    real_atomic_write_yaml = atomic.atomic_write_yaml
    call_count = {"n": 0}

    def maybe_boom(path, data):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise OSError("simulated disk full")
        real_atomic_write_yaml(path, data)

    monkeypatch.setattr("automation.lib.file_blocks.atomic.atomic_write_yaml", maybe_boom)
    with pytest.raises(OSError):
        fb.apply_patch(m, blocks, TASK_ID, repo)
    # Caller's responsibility: rollback now.
    monkeypatch.undo()
    atomic.rollback_task(td, snap_tar, snap_manifest)
    assert (td / "meta.yaml").read_text(encoding="utf-8") == "schema_version: 1.0\nname: original\n"


def test_no_meta_block_returns_missing_meta_error(tmp_path):
    from automation.lib import file_blocks as fb

    text = _file_block("shapes.yaml", "public_shapes: []\n")
    m = fb.parse_meta(text)
    blocks = fb.parse_blocks(text)
    errs = fb.validate_patch(
        m, blocks, TASK_ID, tmp_path,
        expected_decision_set={"patch-shapes"}, expected_keys={"decision"},
    )
    assert any(e.code == "missing_meta" for e in errs)
