#!/usr/bin/env python3
"""golden_manifest.py -- V9 plan M3.2 (R5 + R18 signed-manifest writer).

Builds the per-task signed-manifest bundle that anchors the baseline
trust chain:

* ``tasks/<id>/golden/manifest.json``
      ``{shape_id, sha256, tolerance_class}`` per golden artifact.
* ``tasks/<id>/provenance_commit.json``
      ``{task_id, meta_upstream_commit_sha, kernel_py_sha256,
        test_capture_sha, upstream_closure_fingerprint}`` -- the R12
      commit-chain proof. Including ``meta_upstream_commit_sha`` here
      means R18 signature covers the R12 gate input.
* ``ascendbench/policies/sanity_bounds.yaml`` -- global, M3.0 ship.
* ``meta.yaml.provenance.golden`` canonical-JSON view -- per-task
  provenance fields.

All four artifacts feed into a **single detached signature** stored at
``tasks/<id>/provenance.sig``. The signature payload is the SHA-256
of a canonical-JSON envelope listing the four anchor digests.

Trust roots live in ``ascendbench/policies/trust_roots.json``
(not a Python constant; supports key rotation). The private key is
stored under ``$TAB_GPFS_ROOT/keys/<key_id>.priv`` (mode 0400, owner
provider) -- the dev box has neither, so signing fails-loud rather
than silently using a placeholder key.

This module is signing-library-agnostic at the interface boundary --
the canonical bytes + digest pipeline is pure stdlib; the ed25519
sign / verify pair lazy-imports ``cryptography`` only when actually
invoked (so unit tests of the canonicalisation path don't need it
installed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.automation.lib.atomic import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
)
from ascendbench.correctness.golden_store import GoldenStore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("golden_manifest")


TRUST_ROOTS = REPO_ROOT / "ascendbench" / "policies" / "trust_roots.json"


# ─────────────────────────────────────────────────────────────────────
# Canonical JSON envelope -- inputs to the detached signature
# ─────────────────────────────────────────────────────────────────────


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_bytes(obj: Any) -> bytes:
    """Stable JSON encoding for hashing/signing.

    Uses ``sort_keys=True`` + ``separators=(",",":")`` so different
    Python dict insertion orders / whitespace do not change the
    digest.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


@dataclass
class GoldenManifest:
    """V9 plan M3.2 manifest body."""

    task_id: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    tolerance_class: str | None = None


def build_manifest(
    task_id: str,
    *,
    golden_dir: Path,
    shape_ids: list[str],
    tolerance_class: str,
) -> GoldenManifest:
    """Scan ``golden_dir`` for the listed shape artifacts and build
    the manifest body."""
    artifacts: list[dict[str, Any]] = []
    for sid in shape_ids:
        path = golden_dir / f"{sid}.safetensors"
        if not path.is_file():
            raise FileNotFoundError(
                f"golden artifact for shape={sid} missing at {path}"
            )
        artifacts.append({
            "shape_id": sid,
            "sha256": _sha256_file(path),
            "tolerance_class": tolerance_class,
        })
    return GoldenManifest(
        task_id=task_id,
        artifacts=artifacts,
        tolerance_class=tolerance_class,
    )


def build_provenance_commit(
    task_id: str,
    *,
    meta_upstream_commit_sha: str,
    kernel_py_sha256: str,
    test_capture_sha: str | None,
    upstream_closure_fingerprint: str | None,
) -> dict[str, Any]:
    """R12 commit-chain proof JSON. ``test_capture_sha`` /
    ``upstream_closure_fingerprint`` may be ``None`` for tasks routed
    to source ② (no author-test anchor); the verifier handles that."""
    return {
        "task_id": task_id,
        "meta_upstream_commit_sha": meta_upstream_commit_sha,
        "kernel_py_sha256": kernel_py_sha256,
        "test_capture_sha": test_capture_sha,
        "upstream_closure_fingerprint": upstream_closure_fingerprint,
    }


def build_envelope(
    *,
    manifest_path: Path,
    provenance_commit_path: Path,
    sanity_bounds_path: Path,
    meta_provenance_golden: dict[str, Any],
) -> dict[str, Any]:
    """Compute the canonical-JSON envelope whose SHA-256 is signed.

    The envelope lists the four anchor digests so a single signature
    covers all four artifacts. Any tampering with any anchor changes
    its digest, which changes the envelope, which invalidates the
    signature -- the verify step fails closed.
    """
    return {
        "envelope_schema": "v1",
        "anchors": {
            "manifest_sha256":
                _sha256_file(manifest_path) if manifest_path.is_file() else None,
            "provenance_commit_sha256":
                _sha256_file(provenance_commit_path)
                if provenance_commit_path.is_file() else None,
            "sanity_bounds_sha256":
                _sha256_file(sanity_bounds_path)
                if sanity_bounds_path.is_file() else None,
            "meta_provenance_golden_sha256":
                _sha256_hex(_canonical_json_bytes(meta_provenance_golden)),
        },
    }


# ─────────────────────────────────────────────────────────────────────
# Trust roots + ed25519 sign / verify
# ─────────────────────────────────────────────────────────────────────


def load_trust_roots() -> dict[str, Any]:
    """Load ``ascendbench/policies/trust_roots.json``.

    Returns an empty doc when the file is absent (allows the verifier
    to fail-closed cleanly without crashing -- "no trust roots" is
    surfaced as a verify failure, not an exception).
    """
    if not TRUST_ROOTS.is_file():
        return {"active_keys": [], "revoked_keys": []}
    try:
        return json.loads(TRUST_ROOTS.read_text())
    except json.JSONDecodeError as exc:
        log.error("trust_roots.json parse error: %s", exc)
        return {"active_keys": [], "revoked_keys": []}


def _resolve_key_path(key_id: str) -> Path:
    gpfs = Path(os.environ.get("TAB_GPFS_ROOT", "/var/empty"))
    return gpfs / "keys" / f"{key_id}.priv"


def sign_envelope(envelope: dict[str, Any], *, key_id: str) -> bytes:
    """Sign the envelope's canonical-JSON SHA-256 using the named key.

    Raises ``RuntimeError`` when the trust roots / private key cannot
    be located -- the dev box never has these, so a developer
    invoking this function locally gets an immediate clear error
    instead of producing a fake signature.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
        from cryptography.hazmat.primitives import serialization
    except ImportError as exc:
        raise RuntimeError(
            "golden_manifest.sign_envelope requires the `cryptography` "
            "package (`pip install cryptography`)"
        ) from exc

    key_path = _resolve_key_path(key_id)
    if not key_path.is_file():
        raise RuntimeError(
            f"golden_manifest.sign_envelope: private key for {key_id!r} "
            f"not found at {key_path}; refusing to sign with a placeholder"
        )
    key_bytes = key_path.read_bytes()
    priv = serialization.load_pem_private_key(key_bytes, password=None)
    if not isinstance(priv, Ed25519PrivateKey):
        raise RuntimeError(
            f"golden_manifest.sign_envelope: key {key_id!r} is not ed25519"
        )
    payload = _canonical_json_bytes(envelope)
    return priv.sign(payload)


def verify_envelope(
    envelope: dict[str, Any],
    signature: bytes,
    *,
    key_id: str,
) -> bool:
    """Verify a detached signature against the loaded trust roots."""
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
        from cryptography.exceptions import InvalidSignature
    except ImportError:
        return False

    roots = load_trust_roots()
    active = roots.get("active_keys") or []
    pubkey_hex: str | None = None
    for k in active:
        if k.get("key_id") == key_id and k.get("pubkey_ed25519_hex"):
            pubkey_hex = k["pubkey_ed25519_hex"]
            break
    if pubkey_hex is None:
        return False
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
        pub.verify(signature, _canonical_json_bytes(envelope))
        return True
    except (InvalidSignature, ValueError):
        return False


# ─────────────────────────────────────────────────────────────────────
# End-to-end freeze + sign for one task
# ─────────────────────────────────────────────────────────────────────


def freeze_and_sign(
    task_id: str,
    *,
    shape_ids: list[str],
    tolerance_class: str,
    meta_upstream_commit_sha: str,
    kernel_py_sha256: str,
    test_capture_sha: str | None,
    upstream_closure_fingerprint: str | None,
    meta_provenance_golden: dict[str, Any],
    key_id: str = "v1",
    sign: bool = True,
) -> dict[str, Any]:
    """Write the four anchor files + optional detached signature for
    one task. Returns a summary dict for the caller to log."""
    td = GoldenStore.task_dir(task_id)
    if td is None:
        raise FileNotFoundError(f"task {task_id} not present in tasks/")
    golden_dir = td / "golden"
    if not golden_dir.is_dir():
        raise FileNotFoundError(
            f"task {task_id}: golden/ dir missing; capture stage has not run"
        )

    manifest_body = build_manifest(
        task_id,
        golden_dir=golden_dir,
        shape_ids=shape_ids,
        tolerance_class=tolerance_class,
    )
    manifest_path = golden_dir / "manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "task_id": manifest_body.task_id,
            "tolerance_class": manifest_body.tolerance_class,
            "artifacts": manifest_body.artifacts,
        },
    )

    pc_body = build_provenance_commit(
        task_id,
        meta_upstream_commit_sha=meta_upstream_commit_sha,
        kernel_py_sha256=kernel_py_sha256,
        test_capture_sha=test_capture_sha,
        upstream_closure_fingerprint=upstream_closure_fingerprint,
    )
    pc_path = td / "provenance_commit.json"
    atomic_write_json(pc_path, pc_body)

    sanity_bounds_path = REPO_ROOT / "ascendbench" / "policies" / "sanity_bounds.yaml"
    envelope = build_envelope(
        manifest_path=manifest_path,
        provenance_commit_path=pc_path,
        sanity_bounds_path=sanity_bounds_path,
        meta_provenance_golden=meta_provenance_golden,
    )

    sig_path = td / "provenance.sig"
    sig_info: dict[str, Any] = {"signed": False}
    if sign:
        sig_bytes = sign_envelope(envelope, key_id=key_id)
        atomic_write_bytes(sig_path, sig_bytes, mode=0o644)
        sig_info = {
            "signed": True,
            "key_id": key_id,
            "sig_path": str(sig_path),
            "sig_bytes": len(sig_bytes),
        }

    return {
        "task_id": task_id,
        "manifest_path": str(manifest_path),
        "provenance_commit_path": str(pc_path),
        "envelope": envelope,
        **sig_info,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--task", required=True, help="task id")
    p.add_argument("--verify", action="store_true",
                   help="verify the signature for the named task")
    p.add_argument("--key-id", default="v1")
    args = p.parse_args(argv)

    td = GoldenStore.task_dir(args.task)
    if td is None:
        log.error("task %s not found", args.task)
        return 2

    if args.verify:
        manifest_path = td / "golden" / "manifest.json"
        pc_path = td / "provenance_commit.json"
        sb_path = REPO_ROOT / "ascendbench" / "policies" / "sanity_bounds.yaml"
        sig_path = td / "provenance.sig"
        if not sig_path.is_file():
            log.error("signature %s missing", sig_path)
            return 2

        # Reconstruct the envelope from on-disk artifacts.
        try:
            meta_provenance = json.loads(
                (td / "meta_provenance_golden.cached.json").read_text()
            )
        except OSError:
            meta_provenance = {}
        envelope = build_envelope(
            manifest_path=manifest_path,
            provenance_commit_path=pc_path,
            sanity_bounds_path=sb_path,
            meta_provenance_golden=meta_provenance,
        )
        ok = verify_envelope(
            envelope, sig_path.read_bytes(), key_id=args.key_id
        )
        log.info("verify: %s", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    log.error("--verify is the only invocation mode supported from CLI")
    return 2


if __name__ == "__main__":
    sys.exit(main())
