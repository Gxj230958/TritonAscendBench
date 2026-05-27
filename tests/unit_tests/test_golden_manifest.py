"""Unit tests for V9 plan M3.2 -- golden manifest + signed envelope.

Covers:

* canonical-JSON encoding is stable across dict insertion orders
* build_manifest computes SHA-256 of every golden artifact + tolerance
* envelope hashes change when ANY anchor changes (tamper-evidence)
* sign/verify round-trip uses ed25519 with the trust_roots.json fixture
* sign refuses to run with a placeholder key path / missing trust root
* atomic_write_bytes is a true bytes writer with mode bits honoured
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

# Ensure scripts/ is on sys.path (matches how the script is invoked).
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2]),
)

from scripts.golden_manifest import (
    _canonical_json_bytes,
    build_envelope,
    build_manifest,
    build_provenance_commit,
    sign_envelope,
    verify_envelope,
)
from scripts.automation.lib.atomic import atomic_write_bytes


# ─────────────────────────────────────────────────────────────────────
# atomic_write_bytes promotion
# ─────────────────────────────────────────────────────────────────────


def test_atomic_write_bytes_writes_raw_bytes(tmp_path):
    p = tmp_path / "sub" / "file.bin"
    payload = b"\x00\x01\x02golden"
    atomic_write_bytes(p, payload)
    assert p.read_bytes() == payload


def test_atomic_write_bytes_honors_mode(tmp_path):
    p = tmp_path / "secret.bin"
    atomic_write_bytes(p, b"sealed", mode=0o600)
    # umask may further restrict but cannot loosen; assert at least
    # owner-only (no group/other read).
    stat_mode = os.stat(p).st_mode & 0o777
    assert stat_mode & 0o077 == 0, (
        f"sealed artifact must not be group/world readable; mode={stat_mode:o}"
    )


def test_atomic_write_bytes_rejects_str_payload(tmp_path):
    with pytest.raises(TypeError):
        atomic_write_bytes(tmp_path / "file", "not bytes")


# ─────────────────────────────────────────────────────────────────────
# canonical JSON encoding
# ─────────────────────────────────────────────────────────────────────


def test_canonical_json_is_order_independent():
    a = {"b": 1, "a": [3, 1, 2], "c": {"y": 1, "x": 2}}
    b = {"a": [3, 1, 2], "c": {"x": 2, "y": 1}, "b": 1}
    assert _canonical_json_bytes(a) == _canonical_json_bytes(b)


def test_canonical_json_compact_no_whitespace():
    out = _canonical_json_bytes({"a": 1, "b": [2, 3]})
    assert b" " not in out
    assert b"\n" not in out


# ─────────────────────────────────────────────────────────────────────
# build_manifest
# ─────────────────────────────────────────────────────────────────────


def test_build_manifest_hashes_each_artifact(tmp_path):
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    for sid in ("P1", "P2"):
        save_file(
            {"out": torch.tensor([1.0, 2.0])},
            str(golden_dir / f"{sid}.safetensors"),
        )
    m = build_manifest(
        "L_synth",
        golden_dir=golden_dir,
        shape_ids=["P1", "P2"],
        tolerance_class="elementwise",
    )
    assert m.task_id == "L_synth"
    assert m.tolerance_class == "elementwise"
    assert len(m.artifacts) == 2
    for a in m.artifacts:
        assert a["shape_id"] in {"P1", "P2"}
        assert len(a["sha256"]) == 64
        assert a["tolerance_class"] == "elementwise"


def test_build_manifest_missing_artifact_raises(tmp_path):
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        build_manifest(
            "L_synth",
            golden_dir=golden_dir,
            shape_ids=["P1"],
            tolerance_class="elementwise",
        )


# ─────────────────────────────────────────────────────────────────────
# build_envelope -- tamper evidence
# ─────────────────────────────────────────────────────────────────────


def _write_anchor_artifacts(tmp_path: Path) -> dict[str, Path]:
    manifest = tmp_path / "manifest.json"
    pc = tmp_path / "provenance_commit.json"
    sb = tmp_path / "sanity_bounds.yaml"
    manifest.write_text(json.dumps({"task_id": "L1", "artifacts": []}))
    pc.write_text(json.dumps({"task_id": "L1"}))
    sb.write_text("by_category: {}\n")
    return {"manifest": manifest, "pc": pc, "sb": sb}


def test_envelope_changes_when_manifest_changes(tmp_path):
    anchors = _write_anchor_artifacts(tmp_path)
    e1 = build_envelope(
        manifest_path=anchors["manifest"],
        provenance_commit_path=anchors["pc"],
        sanity_bounds_path=anchors["sb"],
        meta_provenance_golden={"source": "upstream_author_test"},
    )
    # tamper manifest
    anchors["manifest"].write_text(json.dumps({"task_id": "TAMPERED"}))
    e2 = build_envelope(
        manifest_path=anchors["manifest"],
        provenance_commit_path=anchors["pc"],
        sanity_bounds_path=anchors["sb"],
        meta_provenance_golden={"source": "upstream_author_test"},
    )
    assert e1 != e2
    assert e1["anchors"]["manifest_sha256"] != e2["anchors"]["manifest_sha256"]


def test_envelope_changes_when_meta_provenance_changes(tmp_path):
    anchors = _write_anchor_artifacts(tmp_path)
    e1 = build_envelope(
        manifest_path=anchors["manifest"],
        provenance_commit_path=anchors["pc"],
        sanity_bounds_path=anchors["sb"],
        meta_provenance_golden={"source": "upstream_author_test"},
    )
    e2 = build_envelope(
        manifest_path=anchors["manifest"],
        provenance_commit_path=anchors["pc"],
        sanity_bounds_path=anchors["sb"],
        meta_provenance_golden={"source": "gpu_kernel_rerun"},
    )
    assert e1["anchors"]["meta_provenance_golden_sha256"] != \
        e2["anchors"]["meta_provenance_golden_sha256"]


# ─────────────────────────────────────────────────────────────────────
# Sign / verify round-trip (skip if `cryptography` is unavailable)
# ─────────────────────────────────────────────────────────────────────


def _maybe_skip_no_crypto():
    try:
        import cryptography  # noqa: F401
    except ImportError:
        pytest.skip("cryptography not installed -- skip sign/verify test")


def test_sign_verify_round_trip(tmp_path, monkeypatch):
    _maybe_skip_no_crypto()
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    from cryptography.hazmat.primitives import serialization

    # Generate a key pair just for this test.
    priv = Ed25519PrivateKey.generate()
    pub_bytes = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    pub_hex = pub_bytes.hex()
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    # Place the private key under a synthetic GPFS root.
    gpfs = tmp_path / "gpfs"
    (gpfs / "keys").mkdir(parents=True)
    (gpfs / "keys" / "v_test.priv").write_bytes(priv_pem)
    monkeypatch.setenv("TAB_GPFS_ROOT", str(gpfs))

    # Patch the trust roots to contain only our test key.
    from scripts import golden_manifest as gm

    fake_roots = {
        "active_keys": [
            {"key_id": "v_test", "pubkey_ed25519_hex": pub_hex}
        ],
        "revoked_keys": [],
    }
    monkeypatch.setattr(gm, "load_trust_roots", lambda: fake_roots)

    envelope = {"envelope_schema": "v1", "anchors": {"x": "y" * 64}}
    sig = sign_envelope(envelope, key_id="v_test")
    assert isinstance(sig, bytes) and len(sig) > 0
    assert verify_envelope(envelope, sig, key_id="v_test") is True

    # Tamper -> verify fails
    bad_envelope = {**envelope, "anchors": {"x": "z" * 64}}
    assert verify_envelope(bad_envelope, sig, key_id="v_test") is False


def test_sign_refuses_to_use_placeholder_path(tmp_path, monkeypatch):
    _maybe_skip_no_crypto()
    monkeypatch.setenv("TAB_GPFS_ROOT", str(tmp_path / "no_such"))
    with pytest.raises(RuntimeError, match="not found"):
        sign_envelope({"x": 1}, key_id="never_existed")


# ─────────────────────────────────────────────────────────────────────
# provenance_commit shape
# ─────────────────────────────────────────────────────────────────────


def test_build_provenance_commit_includes_all_anchors():
    pc = build_provenance_commit(
        "L_synth",
        meta_upstream_commit_sha="a" * 40,
        kernel_py_sha256="b" * 64,
        test_capture_sha="c" * 40,
        upstream_closure_fingerprint="d" * 64,
    )
    assert pc["task_id"] == "L_synth"
    assert pc["meta_upstream_commit_sha"] == "a" * 40
    assert pc["kernel_py_sha256"] == "b" * 64
    assert pc["test_capture_sha"] == "c" * 40
    assert pc["upstream_closure_fingerprint"] == "d" * 64


def test_build_provenance_commit_allows_none_test_anchors():
    """Source ② tasks have no test_capture_sha -- pass through cleanly."""
    pc = build_provenance_commit(
        "L_synth",
        meta_upstream_commit_sha="a" * 40,
        kernel_py_sha256="b" * 64,
        test_capture_sha=None,
        upstream_closure_fingerprint=None,
    )
    assert pc["test_capture_sha"] is None
    assert pc["upstream_closure_fingerprint"] is None
