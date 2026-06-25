"""Unit tests for V9 plan M4.1 -- eval golden-load via GoldenStore.

Covers:

* ``eligibility.golden_integrity_chain_ok`` returns True for a clean
  pre-M3.2 task (no manifest) -- grace period.
* The chain check returns True when a manifest matches every artifact's
  SHA-256 -- tamper-free baseline.
* The chain check returns False when an artifact is mutated after the
  manifest is written -- tamper-evident.
* ``GoldenIntegrityError`` is raised (rather than ``blocked``-routing)
  when ``_resolve_golden`` is called against a task whose chain fails.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from safetensors.torch import save_file


def _make_task_dir(tmp_path: Path) -> Path:
    """Build a minimal task dir with a shapes.yaml + golden/ + meta.yaml."""
    td = tmp_path / "tasks_root" / "level1" / "L_synth"
    td.mkdir(parents=True)
    (td / "shapes.yaml").write_text(
        "public_shapes:\n  - id: P1\n    name: small\n"
    )
    (td / "meta.yaml").write_text(yaml.safe_dump({
        "task_id": "L_synth",
        "level": "L1",
        "tolerance_class": "elementwise",
        "category": "elementwise",
        "oracle_simplified": "upstream_oracle_required",
        "provenance": {
            "added_in": "v0.1",
            "golden": {
                "oracle_kernel_equivalence": "VERIFIED",
            },
        },
    }))
    return td


def _write_golden(td: Path, tensor: torch.Tensor) -> None:
    golden_dir = td / "golden"
    golden_dir.mkdir(exist_ok=True)
    save_file({"out": tensor}, str(golden_dir / "P1.safetensors"))


def _write_manifest(td: Path) -> None:
    """Write a manifest.json that matches the on-disk artifact's SHA."""
    artifact = td / "golden" / "P1.safetensors"
    h = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "task_id": "L_synth",
        "tolerance_class": "elementwise",
        "artifacts": [
            {"shape_id": "P1", "sha256": h, "tolerance_class": "elementwise"}
        ],
    }
    (td / "golden" / "manifest.json").write_text(json.dumps(manifest))


# ─────────────────────────────────────────────────────────────────────
# golden_integrity_chain_ok
# ─────────────────────────────────────────────────────────────────────


def test_chain_ok_when_no_manifest(tmp_path, monkeypatch):
    """Pre-M3.2 corpus (no manifest) is graced -> chain ok."""
    td = _make_task_dir(tmp_path)
    _write_golden(td, torch.tensor([0.1, 0.2]))

    from ascendbench.correctness.golden_store import GoldenStore
    from ascendbench.release_gate import eligibility

    monkeypatch.setattr(GoldenStore, "task_dir", staticmethod(lambda tid: td))
    assert eligibility.golden_integrity_chain_ok(td, ["P1"]) is True


def test_chain_ok_with_matching_manifest(tmp_path, monkeypatch):
    td = _make_task_dir(tmp_path)
    _write_golden(td, torch.tensor([0.1, 0.2]))
    _write_manifest(td)

    from ascendbench.correctness.golden_store import GoldenStore
    from ascendbench.release_gate import eligibility

    monkeypatch.setattr(GoldenStore, "task_dir", staticmethod(lambda tid: td))
    assert eligibility.golden_integrity_chain_ok(td, ["P1"]) is True


def test_chain_fails_after_tamper(tmp_path, monkeypatch):
    """If the artifact bytes change after the manifest is written, the
    chain check must reject."""
    td = _make_task_dir(tmp_path)
    _write_golden(td, torch.tensor([0.1, 0.2]))
    _write_manifest(td)
    # tamper: write different tensor
    save_file({"out": torch.tensor([99.0, 99.0])},
              str(td / "golden" / "P1.safetensors"))

    from ascendbench.correctness.golden_store import GoldenStore
    from ascendbench.release_gate import eligibility

    monkeypatch.setattr(GoldenStore, "task_dir", staticmethod(lambda tid: td))
    assert eligibility.golden_integrity_chain_ok(td, ["P1"]) is False


def test_chain_fails_when_manifest_missing_shape(tmp_path, monkeypatch):
    """A graded shape not covered by the manifest fails the chain."""
    td = _make_task_dir(tmp_path)
    _write_golden(td, torch.tensor([0.1, 0.2]))
    _write_manifest(td)

    from ascendbench.correctness.golden_store import GoldenStore
    from ascendbench.release_gate import eligibility

    monkeypatch.setattr(GoldenStore, "task_dir", staticmethod(lambda tid: td))
    # Asking about a shape that isn't in the manifest
    assert eligibility.golden_integrity_chain_ok(
        td, ["NOT_IN_MANIFEST"]
    ) is False


def test_chain_fails_for_unknown_task_dir(tmp_path):
    from ascendbench.release_gate import eligibility

    nonexistent = tmp_path / "no_such_dir"
    assert eligibility.golden_integrity_chain_ok(
        nonexistent, ["P1"]
    ) is False


# ─────────────────────────────────────────────────────────────────────
# _resolve_golden integration -- raises GoldenIntegrityError
# ─────────────────────────────────────────────────────────────────────


def test_resolve_golden_raises_on_tamper(tmp_path, monkeypatch):
    """When the chain check fails, _resolve_golden raises
    GoldenIntegrityError (not the silent 'blocked' routing)."""
    td = _make_task_dir(tmp_path)
    _write_golden(td, torch.tensor([0.1, 0.2]))
    _write_manifest(td)
    save_file({"out": torch.tensor([99.0, 99.0])},
              str(td / "golden" / "P1.safetensors"))

    from ascendbench.correctness.golden_store import GoldenStore
    from ascendbench.engine import eval as eval_mod

    monkeypatch.setattr(GoldenStore, "task_dir", staticmethod(lambda tid: td))

    fake_task = SimpleNamespace(
        task_id="L_synth",
        task_dir=td,
        meta={"oracle_simplified": "upstream_oracle_required"},
        oracle_tier="upstream_oracle_required",
    )
    with pytest.raises(eval_mod.GoldenIntegrityError):
        eval_mod._resolve_golden(fake_task, "verified", ["P1"])


def test_resolve_golden_passes_with_clean_chain(tmp_path, monkeypatch):
    """Clean SHA chain + manifest -> evaluated path."""
    td = _make_task_dir(tmp_path)
    _write_golden(td, torch.tensor([0.1, 0.2]))
    _write_manifest(td)

    from ascendbench.correctness.golden_store import GoldenStore
    from ascendbench.engine import eval as eval_mod
    from ascendbench.perf import scoring

    monkeypatch.setattr(GoldenStore, "task_dir", staticmethod(lambda tid: td))

    fake_task = SimpleNamespace(
        task_id="L_synth",
        task_dir=td,
        meta={"oracle_simplified": "upstream_oracle_required"},
        oracle_tier="upstream_oracle_required",
    )
    status, kind, offline = eval_mod._resolve_golden(
        fake_task, "verified", ["P1"]
    )
    assert status == scoring.TASK_STATUS_EVALUATED
    assert kind == "offline"
    assert "P1" in offline
