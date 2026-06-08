"""W2.1 — hidden-shape injection and sealing.

Pins the W2.1 contract on top of W1.0/W1.3/W2.5:

  * ``evaluate(..., mode="verified")`` grades the public AND the sealed
    ``shapes.yaml.hidden_shapes`` shapes; ``mode="self-reported"`` (the
    default) grades public shapes only and never materialises hidden.
  * the candidate-facing shapes view (``process_isolation`` sanitiser)
    has the ``hidden_shapes`` section removed.
  * the deprecated ``--shapes-overlay`` flag has no residual semantics:
    it is suppressed from ``--help`` and ignored in the suite output.

Dev machine has no NPU — these verify the shape-selection plumbing and
the sealing contract, not NPU numeric reality.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ascendbench.integrity import process_isolation as pi
from ascendbench.engine.eval import evaluate

REPO = Path(__file__).resolve().parents[2]
_TASK = "L1_01_bias"
_SOLUTION = REPO / "solutions" / _TASK / "solution.py"

# L1_01_bias ships 5 public (P1..P5) + 4 hidden (H1..H4) shapes.
_PUBLIC_IDS = {"P1", "P2", "P3", "P4", "P5"}
_HIDDEN_IDS = {"H1", "H2", "H3", "H4"}


def _run_tab(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ascendbench.engine.cli", *args],
        cwd=REPO, capture_output=True, text=True,
    )


# ---------------------------------------------------------------------------
# evaluate(mode=...) shape selection.
# ---------------------------------------------------------------------------

def test_self_reported_mode_grades_public_only():
    """The default self-reported mode grades public shapes only — no
    hidden shape is materialised or graded."""
    assert _SOLUTION.exists(), f"fixture missing: {_SOLUTION}"
    result = evaluate(_TASK, _SOLUTION, mode="self-reported")

    assert result.error_reason is None, result.error_reason
    graded = {r.shape_id for r in result.shape_results}
    assert graded, "self-reported grading produced no shape results"
    assert graded <= _PUBLIC_IDS, (
        f"self-reported mode graded a non-public shape: {graded - _PUBLIC_IDS}"
    )
    assert graded.isdisjoint(_HIDDEN_IDS), (
        f"self-reported mode leaked hidden shapes: {graded & _HIDDEN_IDS}"
    )


def test_default_mode_is_self_reported():
    """Omitting `mode` is identical to self-reported — no hidden shapes."""
    default = evaluate(_TASK, _SOLUTION)
    explicit = evaluate(_TASK, _SOLUTION, mode="self-reported")
    assert {r.shape_id for r in default.shape_results} == (
        {r.shape_id for r in explicit.shape_results}
    )
    assert {r.shape_id for r in default.shape_results}.isdisjoint(_HIDDEN_IDS)


def test_verified_mode_grades_public_and_hidden():
    """Verified mode grades the public shapes AND the sealed
    `shapes.yaml.hidden_shapes` shapes — the runner materialises and
    injects the hidden shapes' inputs down the W1.0 IPC."""
    result = evaluate(_TASK, _SOLUTION, mode="verified")

    assert result.error_reason is None, result.error_reason
    graded = {r.shape_id for r in result.shape_results}
    assert _PUBLIC_IDS <= graded, (
        f"verified mode dropped public shapes: {_PUBLIC_IDS - graded}"
    )
    assert _HIDDEN_IDS <= graded, (
        f"verified mode did not grade the hidden shapes: "
        f"{_HIDDEN_IDS - graded}"
    )


def test_verified_grades_strictly_more_shapes_than_self_reported():
    """Verified mode's graded set is a strict superset of self-reported's:
    proof the hidden shapes are an *addition*, not a replacement."""
    sr = {r.shape_id for r in evaluate(_TASK, _SOLUTION,
                                       mode="self-reported").shape_results}
    ve = {r.shape_id for r in evaluate(_TASK, _SOLUTION,
                                       mode="verified").shape_results}
    assert sr < ve, f"verified ({ve}) is not a strict superset of self-reported ({sr})"
    assert (ve - sr) == _HIDDEN_IDS


def test_unknown_mode_rejected():
    """An unknown mode string fails loudly rather than silently grading
    public-only."""
    with pytest.raises(ValueError, match="mode must be"):
        evaluate(_TASK, _SOLUTION, mode="overlay")


# ---------------------------------------------------------------------------
# Candidate view: the hidden_shapes section is sealed.
# ---------------------------------------------------------------------------

def test_sanitised_shapes_view_strips_hidden_section():
    """`process_isolation.sanitised_shapes_view` removes the
    `hidden_shapes` section while keeping the public view intact."""
    full = {
        "size_knobs": {"B": 2, "L": 4, "D": 8},
        "public_shapes": [{"id": "P1", "knobs": {"B": 1}}],
        "hidden_shapes": [{"id": "H1", "knobs": {"B": 3, "L": 13}}],
        "inputs": {"x": {"dims": ["B"], "dtype": "float32"}},
    }
    view = pi.sanitised_shapes_view(full)

    assert "hidden_shapes" not in view, "hidden_shapes leaked into candidate view"
    # The public view is preserved verbatim.
    assert view["public_shapes"] == full["public_shapes"]
    assert view["size_knobs"] == full["size_knobs"]
    assert view["inputs"] == full["inputs"]
    # The original document is not mutated.
    assert "hidden_shapes" in full


def test_candidate_view_has_no_hidden_knobs():
    """No hidden shape's knob row may appear anywhere in the sanitised
    view — a stricter assertion than just key absence."""
    full = {
        "public_shapes": [{"id": "P1", "knobs": {"B": 1}}],
        "hidden_shapes": [{"id": "H_secret", "knobs": {"B": 999}}],
    }
    view = pi.sanitised_shapes_view(full)
    blob = json.dumps(view)
    assert "H_secret" not in blob
    assert "999" not in blob


def test_run_two_pass_never_carries_shapes_document():
    """The candidate IPC carries only materialised inputs — the child is
    never handed a shapes document, so the hidden_shapes section is
    structurally absent from the candidate process (W1.0/W2.1)."""
    src = pi._CHILD
    # The child script must not reference build_inputs / shapes.yaml /
    # hidden_shapes — it works purely off injected safetensors + sidecar.
    assert "build_inputs" not in src
    assert "shapes.yaml" not in src
    assert "hidden_shapes" not in src
    assert "repo_root" not in src.lower()


# ---------------------------------------------------------------------------
# --shapes-overlay deprecation: no residual semantics.
# ---------------------------------------------------------------------------

def test_shapes_overlay_suppressed_from_help():
    """The deprecated `--shapes-overlay` flag is suppressed from the
    run-suite help output — it is no longer an advertised interface."""
    r = _run_tab("run-suite", "--help")
    out = r.stdout + r.stderr
    assert "--shapes-overlay" not in out, (
        "deprecated --shapes-overlay still advertised in run-suite --help"
    )
    # --mode is the supported verified-mode switch.
    assert "--mode" in out


def test_run_has_mode_flag():
    """`tab run` (the single-task bring-up entrypoint) advertises
    `--mode` so W2.1 / W4.3 single-task verified runs are possible."""
    r = _run_tab("run", "--help")
    out = r.stdout + r.stderr
    assert "--mode" in out
    assert "self-reported" in out and "verified" in out


def test_shapes_overlay_flag_ignored_and_not_emitted(tmp_path):
    """Passing the deprecated `--shapes-overlay` neither errors nor leaves
    any residual semantics: the suite output JSON has no `shapes_overlay`
    key, and the flag's value does not appear anywhere in the result."""
    sol_root = tmp_path / "solutions"
    (sol_root / _TASK).mkdir(parents=True)
    (sol_root / _TASK / "solution.py").write_text(_SOLUTION.read_text())
    out_path = tmp_path / "suite.json"

    r = _run_tab(
        "run-suite", "--level", "L1",
        "--solutions-root", str(sol_root),
        "--shapes-overlay", "/nonexistent/overlay/dir",
        "--out", str(out_path),
    )
    assert r.returncode == 0, f"run-suite failed: {r.stdout} {r.stderr}"
    doc = json.loads(out_path.read_text())
    # The deprecated flag must not be echoed into the suite contract.
    assert "shapes_overlay" not in doc, (
        "deprecated --shapes-overlay leaked into suite output"
    )
    assert "/nonexistent/overlay/dir" not in json.dumps(doc), (
        "the --shapes-overlay value has residual semantics in the output"
    )
    # `mode` (the real switch) is still emitted.
    assert "mode" in doc
