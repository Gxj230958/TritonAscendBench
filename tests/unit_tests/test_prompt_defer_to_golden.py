"""Tests for ``automation.prompts.defer_to_golden`` (Unit 2.7).

The defer-to-golden prompt is the SOLE remediation path for
``upstream-not-invocable-here``. The system message must clamp the
decision menu to one value, surface the 5-field defer schema, and the
user message must splice the existing meta.yaml plus resolver hint /
blocker reason verbatim so the LLM has a faithful baseline.
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


_MARKER_LICENSE = "Apache-2.0-MARKER-FOR-TEST"


def _scaffold(
    tmp_path: Path,
    task_id: str = "L1_01_bias",
    level: str = "level1",
) -> Path:
    td = tmp_path / "tasks" / level / task_id
    td.mkdir(parents=True)
    meta = {
        "schema_version": "1.0",
        "task_id": task_id,
        "name": "logit_bias",
        "kind": "triton",
        "upstream": {
            "repo": "vllm-project/vllm",
            "path": "vllm/v1/worker/gpu/sample/logit_bias.py",
            "license": _MARKER_LICENSE,
        },
    }
    (td / "meta.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=False), encoding="utf-8",
    )
    (td / "oracle_pytorch.py").write_text(
        "import torch\n"
        "def oracle(x: torch.Tensor) -> torch.Tensor:\n"
        "    return x + 1\n",
        encoding="utf-8",
    )
    vd = tmp_path / "tasks" / "_vendored"
    vd.mkdir(parents=True)
    (vd / f"{task_id}_wrapper.py").write_text(
        "import torch\n"
        "def logit_bias(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:\n"
        "    ...\n",
        encoding="utf-8",
    )
    return td


def test_build_prompt_returns_two_messages(tmp_path):
    from automation.prompts import defer_to_golden

    _scaffold(tmp_path)
    messages = defer_to_golden.build_prompt(
        "L1_01_bias", tmp_path,
    )
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_user_prompt_contains_meta_yaml(tmp_path):
    """The full current meta.yaml body must appear in the user message
    so the LLM can produce a minimal-diff patch."""
    from automation.prompts import defer_to_golden

    _scaffold(tmp_path)
    messages = defer_to_golden.build_prompt(
        "L1_01_bias", tmp_path,
    )
    user_content = messages[1]["content"]
    # A distinctive license string from the scaffolded meta.yaml -- if
    # this leaks through, the full body was spliced in.
    assert _MARKER_LICENSE in user_content
    assert "vllm/v1/worker/gpu/sample/logit_bias.py" in user_content


def test_decision_set_only_defer_to_golden(tmp_path):
    """System prompt must clamp the decision menu to exactly one value.
    The retired `defer-to-upstream-oracle` is explicitly named only as
    a forbidden alternative."""
    from automation.prompts import defer_to_golden

    _scaffold(tmp_path)
    messages = defer_to_golden.build_prompt(
        "L1_01_bias", tmp_path,
    )
    sys_content = messages[0]["content"]
    assert "defer-to-golden" in sys_content
    # The "ONE valid value only" clamp must be explicit.
    assert "ONE valid value only" in sys_content
    # Layer 4 schema reference: the retired alternative is named only
    # to forbid it, never offered as a choice.
    assert "retired" in sys_content


def test_upstream_callable_hint_in_user_prompt(tmp_path):
    """A non-empty resolver hint must be spliced into the user content
    so the LLM does not invent a dotted path from scratch."""
    from automation.prompts import defer_to_golden

    _scaffold(tmp_path)
    messages = defer_to_golden.build_prompt(
        "L1_01_bias", tmp_path,
        upstream_callable_hint={
            "module": "vllm.v1.worker.gpu.sample.logit_bias",
            "function": "apply_logit_bias",
        },
    )
    user_content = messages[1]["content"]
    assert "vllm.v1.worker.gpu.sample.logit_bias" in user_content
    assert "apply_logit_bias" in user_content


def test_blocker_reason_in_user_prompt(tmp_path):
    """The blocker reason becomes the candidate phase_a_prime.reason --
    it must surface in the user content so the LLM copies it verbatim
    (a 5-field defer patch without a non-empty reason is Layer 5 reject)."""
    from automation.prompts import defer_to_golden

    _scaffold(tmp_path)
    reason = "C++ binding requires real GPU runtime; not importable here"
    messages = defer_to_golden.build_prompt(
        "L1_01_bias", tmp_path,
        blocker_reason=reason,
    )
    user_content = messages[1]["content"]
    assert reason in user_content


def test_missing_meta_file_raises(tmp_path):
    """If the task dir exists but meta.yaml is missing, fail loud --
    asking the LLM to patch a meta we cannot show is meaningless."""
    from automation.prompts import defer_to_golden

    task_id = "L1_01_bias"
    td = tmp_path / "tasks" / "level1" / task_id
    td.mkdir(parents=True)
    # No meta.yaml created on purpose.
    with pytest.raises(FileNotFoundError):
        defer_to_golden.build_prompt(task_id, tmp_path)


def test_task_dir_missing_raises(tmp_path):
    """Unknown task_id -> FileNotFoundError."""
    from automation.prompts import defer_to_golden

    with pytest.raises(FileNotFoundError):
        defer_to_golden.build_prompt("L1_99_unknown", tmp_path)


def test_phase_b_capture_status_enum_documented(tmp_path):
    """The system prompt must explicitly enumerate PENDING|DONE|FAILED
    for ``phase_b_capture.status`` -- earlier drafts of the spec used
    'CAPTURED', which the Layer 5 validator rejects, so the prompt has
    to teach the LLM the canonical enum.
    """
    from automation.prompts import defer_to_golden

    _scaffold(tmp_path)
    messages = defer_to_golden.build_prompt(
        "L1_01_bias", tmp_path,
    )
    sys_content = messages[0]["content"]
    # All three enum values present.
    for value in ("PENDING", "DONE", "FAILED"):
        assert value in sys_content
    # And an explicit warning about the obsolete 'CAPTURED' value.
    assert "CAPTURED" in sys_content


def test_defer_to_golden_prompt_lists_decision_keys(tmp_path):
    """The defer_to_golden system prompt must enumerate the 5 keys
    required for the ``defer-to-golden`` decision: the 3 common ones
    (decision, rationale, evidence) plus the two decision-specific
    fields ``upstream_callable`` and ``blocker_reason``."""
    from automation.prompts import defer_to_golden

    _scaffold(tmp_path)
    messages = defer_to_golden.build_prompt(
        "L1_01_bias", tmp_path,
    )
    sys_content = messages[0]["content"]
    # Block grammar invariants.
    assert "### META" in sys_content
    assert "### FILE" in sys_content
    # All 5 required keys for the defer-to-golden decision row.
    for key in (
        "decision", "rationale", "evidence",
        "upstream_callable", "blocker_reason",
    ):
        assert key in sys_content, f"missing defer-to-golden META key {key!r}"
    # The decision label itself appears in the per-decision table.
    assert "defer-to-golden" in sys_content
