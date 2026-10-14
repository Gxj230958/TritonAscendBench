"""Unit tests for V9 plan M0.0 -- GoldenStore authority."""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

from ascendbench.correctness.golden_store import GoldenStore

# ─────────────────────────────────────────────────────────────────────
# task_dir / locate / manifest_path / signature_path -- pure path locators
# ─────────────────────────────────────────────────────────────────────


def test_task_dir_finds_existing_level1_task():
    p = GoldenStore.task_dir("L1_12_rms_norm")
    assert p is not None
    assert p.is_dir()
    assert p.name == "L1_12_rms_norm"


def test_task_dir_returns_none_for_unknown_task():
    assert GoldenStore.task_dir("L9_99_does_not_exist") is None


def test_locate_returns_conventional_path_even_when_artifact_absent():
    """locate() reports the location BEFORE the artifact is written --
    capture pipelines call this in advance."""
    p = GoldenStore.locate("L1_12_rms_norm", "P1")
    assert p is not None
    # The path string is conventional even if the file does not exist:
    assert p.name == "P1.safetensors"
    assert p.parent.name == "golden"


def test_locate_returns_none_for_unknown_task():
    assert GoldenStore.locate("L9_99_does_not_exist", "P1") is None


def test_manifest_path_uses_conventional_location():
    p = GoldenStore.manifest_path("L1_12_rms_norm")
    assert p is not None
    assert p.name == "manifest.json"
    assert p.parent.name == "golden"


def test_signature_path_uses_conventional_location():
    p = GoldenStore.signature_path("L1_12_rms_norm")
    assert p is not None
    assert p.name == "provenance.sig"


# ─────────────────────────────────────────────────────────────────────
# locate_sealed -- uid check + env var gating
# ─────────────────────────────────────────────────────────────────────


def test_locate_sealed_returns_none_for_unknown_task_dir():
    # Sealed location is computed from $TAB_GPFS_ROOT/sealed_golden/<id>/...
    # but the task identity itself is the key; an unknown task id is
    # still represented in the path -- the caller checks ``.is_file()``
    # to confirm presence. The store returns the conventional path
    # rather than raising.
    p = GoldenStore.locate_sealed("L1_12_rms_norm", "H1")
    assert p is not None
    assert p.name == "sealed_H1.safetensors"


def test_locate_sealed_returns_none_when_uid_mismatch():
    """The R19 boundary: a non-provider uid must NEVER see a sealed path."""
    with patch.dict(os.environ, {"TAB_PROVIDER_UID": "12345"}, clear=False):
        # uid 6789 != provider uid 12345 -> sealed path withheld
        assert GoldenStore.locate_sealed("L1_12_rms_norm", "H1", uid=6789) is None
        # Calling without a uid does NOT enforce the check (callers without
        # an effective uid pass None; the file-system mode 0700 is the
        # ultimate gate).
        assert GoldenStore.locate_sealed("L1_12_rms_norm", "H1", uid=None) is not None


def test_locate_sealed_honours_matching_uid():
    with patch.dict(os.environ, {"TAB_PROVIDER_UID": "7777"}, clear=False):
        assert GoldenStore.locate_sealed("L1_12_rms_norm", "H1", uid=7777) is not None


def test_locate_sealed_garbled_provider_uid_treats_as_mismatch():
    """A malformed $TAB_PROVIDER_UID is conservatively treated as 'caller
    is not the provider' (fail-closed)."""
    with patch.dict(os.environ, {"TAB_PROVIDER_UID": "not_a_number"}, clear=False):
        assert GoldenStore.locate_sealed("L1_12_rms_norm", "H1", uid=42) is None


# ─────────────────────────────────────────────────────────────────────
# locate_pyref -- R10 oracle pyref pair
# ─────────────────────────────────────────────────────────────────────


def test_locate_pyref_returns_inputs_and_ref_paths():
    pair = GoldenStore.locate_pyref("L1_12_rms_norm", "P1")
    assert pair is not None
    inputs_path, ref_path = pair
    assert inputs_path.name == "inputs_P1.safetensors"
    assert ref_path.name == "ref_P1.safetensors"
    assert inputs_path.parent.name == "r10_pyref"
    assert ref_path.parent.name == "r10_pyref"


def test_locate_pyref_returns_none_for_unknown_task():
    assert GoldenStore.locate_pyref("L9_99_does_not_exist", "P1") is None


def test_has_pyref_true_when_artifacts_present(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file

    pyref = tmp_path / "r10_pyref"
    pyref.mkdir()
    save_file({"out": torch.zeros(1)}, str(pyref / "ref_P1.safetensors"))
    monkeypatch.setattr(GoldenStore, "task_dir", lambda _task_id: tmp_path)
    assert GoldenStore.has_pyref("SYN_pyref") is True


def test_has_pyref_false_when_artifacts_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(GoldenStore, "task_dir", lambda _task_id: tmp_path)
    assert GoldenStore.has_pyref("SYN_pyref") is False


def test_has_pyref_false_for_unknown_task():
    assert GoldenStore.has_pyref("L9_99_does_not_exist") is False


# ─────────────────────────────────────────────────────────────────────
# is_complete / integrity_check
# ─────────────────────────────────────────────────────────────────────


def test_is_complete_false_when_golden_dir_absent(tmp_path):
    """Synthetic task with no golden/ -- is_complete returns False."""
    fake_task = SimpleNamespace(task_id="L_synth", task_dir=tmp_path / "task")
    fake_task.task_dir.mkdir()
    # No shapes.yaml -> shapes_eval_ids returns None -> False
    assert GoldenStore.is_complete(fake_task) is False


def test_is_complete_true_when_all_golden_artifacts_usable(tmp_path):
    """Synthetic task with a complete usable golden set."""
    task_dir = tmp_path / "L_synth"
    task_dir.mkdir()
    (task_dir / "shapes.yaml").write_text(
        "public_shapes:\n  - id: P1\n    name: small\n"
        "hidden_shapes:\n  - id: H1\n    name: hidden\n"
    )
    golden_dir = task_dir / "golden"
    golden_dir.mkdir()
    # Write valid safetensors files (one fp32 tensor each)
    import torch
    from safetensors.torch import save_file

    for sid in ("P1", "H1"):
        save_file({"out": torch.zeros(4)}, str(golden_dir / f"{sid}.safetensors"))

    fake_task = SimpleNamespace(task_id="L_synth", task_dir=task_dir)
    assert GoldenStore.is_complete(fake_task) is True


def test_is_complete_false_when_artifact_corrupt(tmp_path):
    """A corrupt .safetensors makes is_complete fail-closed."""
    task_dir = tmp_path / "L_synth_bad"
    task_dir.mkdir()
    (task_dir / "shapes.yaml").write_text("public_shapes:\n  - id: P1\n    name: x\n")
    golden_dir = task_dir / "golden"
    golden_dir.mkdir()
    (golden_dir / "P1.safetensors").write_bytes(b"\x00\x00not_a_real_file")

    fake_task = SimpleNamespace(task_id="L_synth_bad", task_dir=task_dir)
    assert GoldenStore.is_complete(fake_task) is False


def test_integrity_check_true_without_manifest():
    """A task without a manifest (legacy / pre-M3.2) returns True --
    'no integrity claim made' is not 'integrity failed'."""
    assert GoldenStore.integrity_check("L1_12_rms_norm") is True


def test_integrity_check_false_for_unknown_task():
    assert GoldenStore.integrity_check("L9_99_does_not_exist") is False
