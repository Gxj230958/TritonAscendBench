"""Tests for ``automation.prompts.shapes_incomplete`` (Unit 1.10)."""
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


def _scaffold(tmp_path: Path, task_id: str = "L1_06_swiglu", level: str = "L1") -> Path:
    td = tmp_path / "tasks" / level / task_id
    td.mkdir(parents=True)
    (td / "meta.yaml").write_text(
        yaml.safe_dump({"schema_version": "1.0", "name": "swiglu", "kind": "triton"}),
        encoding="utf-8",
    )
    (td / "shapes.yaml").write_text("public_shapes: []\n", encoding="utf-8")
    (td / "oracle_pytorch.py").write_text("def oracle(x):\n    return x\n", encoding="utf-8")
    vd = tmp_path / "tasks" / "_vendored"
    vd.mkdir(parents=True)
    (vd / f"{task_id}_wrapper.py").write_text(
        "def swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:\n"
        "    ...\n",
        encoding="utf-8",
    )
    infer = tmp_path / "infer" / "vllm"
    infer.mkdir(parents=True)
    (infer / "model.py").write_text(
        "from x import y\n# call site: swiglu(x, gate)\nout = swiglu(x, gate)\n",
        encoding="utf-8",
    )
    return td


def test_build_prompt_happy(tmp_path):
    from automation.prompts import shapes_incomplete

    _scaffold(tmp_path)
    messages = shapes_incomplete.build_prompt("L1_06_swiglu", repo_root=tmp_path)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "patch-shapes" in messages[0]["content"]
    assert "### META" in messages[0]["content"]
    assert "L1_06_swiglu" in messages[1]["content"]
    assert "swiglu" in messages[1]["content"]


def test_build_prompt_includes_high_level_op_warning(tmp_path):
    from automation.prompts import shapes_incomplete

    _scaffold(tmp_path)
    messages = shapes_incomplete.build_prompt("L1_06_swiglu", repo_root=tmp_path)
    sys_content = messages[0]["content"]
    # At least the famous gelu / silu / softmax names must be listed.
    for name in ("silu", "gelu", "softmax", "layer_norm", "rms_norm"):
        assert name in sys_content


def test_build_prompt_includes_caller_evidence(tmp_path):
    from automation.prompts import shapes_incomplete

    _scaffold(tmp_path)
    messages = shapes_incomplete.build_prompt("L1_06_swiglu", repo_root=tmp_path)
    assert "infer/vllm/model.py" in messages[1]["content"]


def test_build_prompt_missing_wrapper_raises(tmp_path):
    from automation.prompts import shapes_incomplete

    task_id = "L1_06_swiglu"
    td = tmp_path / "tasks" / "L1" / task_id
    td.mkdir(parents=True)
    (td / "meta.yaml").write_text("schema_version: 1.0\n", encoding="utf-8")
    # No _vendored/ dir at all.
    with pytest.raises(shapes_incomplete.WrapperSourceMissingError):
        shapes_incomplete.build_prompt(task_id, repo_root=tmp_path)


def test_build_prompt_no_caller_does_not_block(tmp_path):
    from automation.prompts import shapes_incomplete

    task_id = "L1_06_swiglu"
    td = tmp_path / "tasks" / "L1" / task_id
    td.mkdir(parents=True)
    (td / "meta.yaml").write_text("schema_version: 1.0\n", encoding="utf-8")
    (td / "shapes.yaml").write_text("public_shapes: []\n", encoding="utf-8")
    vd = tmp_path / "tasks" / "_vendored"
    vd.mkdir(parents=True)
    (vd / f"{task_id}_w.py").write_text(
        "def swiglu(x, y): ...\n", encoding="utf-8",
    )
    # No infer/ dir; should still build.
    msgs = shapes_incomplete.build_prompt(task_id, repo_root=tmp_path)
    assert "no caller hits" in msgs[1]["content"] or "no infer" in msgs[1]["content"]


def test_build_prompt_task_dir_missing_raises(tmp_path):
    from automation.prompts import shapes_incomplete

    with pytest.raises(FileNotFoundError):
        shapes_incomplete.build_prompt("L1_99_unknown", repo_root=tmp_path)


def test_build_prompt_under_32k_chars(tmp_path):
    """Rough token cap: the prompt should be << 32k chars (~8k tokens)."""
    from automation.prompts import shapes_incomplete

    _scaffold(tmp_path)
    messages = shapes_incomplete.build_prompt("L1_06_swiglu", repo_root=tmp_path)
    total = sum(len(m["content"]) for m in messages)
    assert total < 32_000


def test_build_prompt_missing_param_evidence_passed_through(tmp_path):
    from automation.prompts import shapes_incomplete

    _scaffold(tmp_path)
    messages = shapes_incomplete.build_prompt(
        "L1_06_swiglu", repo_root=tmp_path,
        missing_params_evidence="param 'gate' has no value at L342",
    )
    assert "param 'gate' has no value at L342" in messages[1]["content"]


def test_shapes_incomplete_prompt_lists_patch_shapes_keys(tmp_path):
    """The shapes_incomplete system prompt must enumerate the 4
    ``patch-shapes`` META keys -- the validator rejects underspecified
    META and burns the retry budget if any of these is missing from
    the prompt."""
    from automation.prompts import shapes_incomplete

    _scaffold(tmp_path)
    messages = shapes_incomplete.build_prompt(
        "L1_06_swiglu", repo_root=tmp_path,
    )
    sys_content = messages[0]["content"]
    for key in ("decision", "missing_params", "rationale", "evidence"):
        assert key in sys_content, (
            f"shapes_incomplete system prompt missing patch-shapes META "
            f"key {key!r}"
        )
    # The decision label itself must appear in the per-decision table.
    assert "patch-shapes" in sys_content


def test_shared_block_grammar_block_protocol_unchanged():
    """The block grammar carries the META + FILE syntax envelope that
    ``automation.lib.file_blocks`` parses. A future edit to the grammar
    must keep both block markers visible -- otherwise every prompt
    silently breaks the parser."""
    from automation.prompts import shared

    assert "### META" in shared.BLOCK_GRAMMAR
    assert "### END" in shared.BLOCK_GRAMMAR
    assert "### FILE" in shared.BLOCK_GRAMMAR
    # The decision-agnostic grammar must NOT bake any specific
    # decision into the syntax description -- per-decision keys live
    # in render_meta_keys_table.
    for dec in (
        "patch-shapes", "tolerance-bump", "rewrite-primitives",
        "defer-to-upstream-oracle", "defer-to-golden",
    ):
        assert dec not in shared.BLOCK_GRAMMAR, (
            f"BLOCK_GRAMMAR should not hard-code decision {dec!r}; "
            "use render_meta_keys_table for per-decision content."
        )


def test_shared_meta_keys_by_decision_all_5_decisions():
    """The shared META-keys table MUST stay in lockstep with the
    orchestrator's ``_DECISION_TO_REQUIRED_KEYS`` -- if either side
    drifts, the LLM ships a META the validator rejects."""
    from automation import fix_orchestrator
    from automation.prompts import shared

    expected_decisions = {
        "patch-shapes",
        "tolerance-bump",
        "rewrite-primitives",
        "defer-to-upstream-oracle",
        "defer-to-golden",
    }
    assert set(shared.META_KEYS_BY_DECISION.keys()) == expected_decisions

    for dec, keys in shared.META_KEYS_BY_DECISION.items():
        orchestrator_keys = fix_orchestrator._DECISION_TO_REQUIRED_KEYS[dec]
        assert set(keys) == set(orchestrator_keys), (
            f"shared.META_KEYS_BY_DECISION[{dec!r}] drifted from "
            f"fix_orchestrator._DECISION_TO_REQUIRED_KEYS: "
            f"{set(keys)} != {set(orchestrator_keys)}"
        )
