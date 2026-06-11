"""U6: [gpu_capture] schema tests.

Covers: all required fields present, reuses [gpu] cuda/torch/triton/py,
pending placeholder allowed by schema but means NOT release-PASS,
einops declared in pyproject.toml so extra_pins is empty.
"""
from __future__ import annotations

import re
import tomllib  # Python >= 3.11 stdlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def _matrix():
    return tomllib.loads((REPO / "ascendbench" / "policies" / "version_matrix.toml").read_text())


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
    raw = (REPO / "ascendbench" / "policies" / "version_matrix.toml").read_text()
    assert "NOT release-PASS" in raw or "not release-PASS" in raw


def test_einops_declared_in_pyproject_not_extra_pins():
    """einops is a bench runtime dependency: it belongs in pyproject.toml
    [project].dependencies (installed by `pip install -e .`), not pinned
    in [gpu_capture].extra_pins. extra_pins is empty as a result."""
    gc = _matrix()["gpu_capture"]
    assert gc["extra_pins"] == [], (
        f"extra_pins must be empty (einops moved to pyproject.toml); "
        f"got {gc['extra_pins']!r}"
    )
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    deps = pyproject["project"]["dependencies"]
    dep_names = {
        re.split(r"[<>=~!;\s\[]", d, maxsplit=1)[0].strip().lower() for d in deps
    }
    assert "einops" in dep_names, (
        f"einops must be in [project].dependencies; got {sorted(dep_names)}"
    )
