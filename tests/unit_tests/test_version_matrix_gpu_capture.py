"""U6: [gpu_capture] schema tests.

Covers: all required fields present, reuses [gpu] cuda/torch/triton/py,
pending placeholder allowed by schema but means NOT release-PASS,
einops Class-D pin present.
"""
from __future__ import annotations

import tomllib  # Python >= 3.11 (pinned in infra/version_matrix.toml)
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def _matrix():
    return tomllib.loads((REPO / "infra" / "version_matrix.toml").read_text())


def test_gpu_capture_section_exists_with_required_fields():
    m = _matrix()
    assert "gpu_capture" in m
    gc = m["gpu_capture"]
    for field in (
        "cuda", "torch", "triton", "python",
        "device_identity", "compute_capability", "fp8_semantics",
        "extra_pins",
    ):
        assert field in gc, f"[gpu_capture] missing {field}"


def test_gpu_capture_reuses_gpu_pins():
    m = _matrix()
    for k in ("cuda", "torch", "triton", "python"):
        assert m["gpu_capture"][k] == m["gpu"][k], f"{k} must equal [gpu].{k}"


def test_device_dependent_fields_are_pending_not_fabricated():
    """device_identity / compute_capability / fp8_semantics MUST be the
    literal pending placeholder until backfilled on the capture host
    (plan U6: NEVER fabricate)."""
    gc = _matrix()["gpu_capture"]
    for f in ("device_identity", "compute_capability", "fp8_semantics"):
        assert gc[f].startswith("pending: filled-on-capture-host"), (
            f"{f}={gc[f]!r} must be the pending placeholder, not a "
            f"fabricated literal"
        )


def test_pending_means_not_release_pass():
    """A pending capture value must NOT be a release-PASS. We assert
    the documented invariant text is present (S6/U7 treats pending as
    non-releasable)."""
    raw = (REPO / "infra" / "version_matrix.toml").read_text()
    assert "NOT release-PASS" in raw or "not release-PASS" in raw


def test_class_d_einops_pin_present():
    gc = _matrix()["gpu_capture"]
    assert any(p.startswith("einops==") for p in gc["extra_pins"])
