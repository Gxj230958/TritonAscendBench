"""Unit tests for V9 plan M2.0.5 -- R12 SHA-chain integrity check.

Covers:

* ``verify_task`` returns the 5-state verdict (VERIFIED /
  NEEDS_MANUAL / BACKLOG_PENDING_GOLDEN / EXCLUDED_SNAPSHOT /
  NO_TEST_CAPTURE_SHA) given different combinations of
  ``meta.upstream.commit_sha`` and ``provenance.golden.test_capture_sha``.
* The snapshot-no-git sentinel routes to ``EXCLUDED_SNAPSHOT``.
* When both anchors are absent in ``infer/<repo>/.git``, the verdict
  is ``BACKLOG_PENDING_GOLDEN`` (force source ②).
* The git-reachability check is mocked so the unit test doesn't depend
  on the dev box's vendored ``infer/`` checkout.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from scripts.verify_upstream_commit_chain import (
    VERDICT_BACKLOG,
    VERDICT_EXCLUDED_SNAPSHOT,
    VERDICT_NEEDS_MANUAL,
    VERDICT_NO_TEST_SHA,
    VERDICT_VERIFIED,
    verify_task,
)


# ─────────────────────────────────────────────────────────────────────
# Fixture helper: write a fake task dir with controllable meta
# ─────────────────────────────────────────────────────────────────────


def _write_task(
    tmp_path: Path,
    *,
    repo: str = "vLLM",
    commit_sha: str = "a" * 40,
    test_capture_sha: str | None = None,
    add_provenance: bool = True,
) -> Path:
    td = tmp_path / "L_synth"
    td.mkdir()
    meta: dict = {
        "task_id": "L_synth",
        "repo": repo,
        "upstream": {"commit_sha": commit_sha},
    }
    if add_provenance:
        golden: dict = {}
        if test_capture_sha is not None:
            golden["test_capture_sha"] = test_capture_sha
        meta["provenance"] = {"golden": golden} if golden else {"added_in": "v0.1"}
    (td / "meta.yaml").write_text(yaml.safe_dump(meta))
    return td


# ─────────────────────────────────────────────────────────────────────
# Snapshot sentinel routing
# ─────────────────────────────────────────────────────────────────────


def test_snapshot_no_git_routes_to_excluded(tmp_path):
    td = _write_task(tmp_path, commit_sha="snapshot-no-git")
    v = verify_task(td)
    assert v.verdict == VERDICT_EXCLUDED_SNAPSHOT
    assert "snapshot" in v.reason.lower()


def test_missing_commit_sha_routes_to_excluded(tmp_path):
    td = _write_task(tmp_path, commit_sha="")
    v = verify_task(td)
    assert v.verdict == VERDICT_EXCLUDED_SNAPSHOT


# ─────────────────────────────────────────────────────────────────────
# NO_TEST_CAPTURE_SHA -- M1.1 deprecated in V9 (expected state)
# ─────────────────────────────────────────────────────────────────────


def test_missing_test_capture_sha_returns_no_test_verdict(tmp_path):
    """Current corpus state: M1.1 was deprecated; test_capture_sha
    isn't written -> NO_TEST_CAPTURE_SHA (not a failure)."""
    td = _write_task(tmp_path, commit_sha="a" * 40, test_capture_sha=None)
    with patch(
        "scripts.verify_upstream_commit_chain._git_object_reachable",
        return_value=True,
    ):
        v = verify_task(td)
    assert v.verdict == VERDICT_NO_TEST_SHA
    assert v.kernel_commit_sha == "a" * 40


def test_missing_test_capture_sha_reports_kernel_reachability(tmp_path):
    """Even without a test_capture_sha, the kernel sha reachability
    is reported so callers can surface unreachable-upstream cases."""
    td = _write_task(tmp_path, commit_sha="a" * 40, test_capture_sha=None)
    with patch(
        "scripts.verify_upstream_commit_chain._git_object_reachable",
        return_value=False,
    ):
        v = verify_task(td)
    assert v.verdict == VERDICT_NO_TEST_SHA
    assert v.kernel_sha_reachable is False


# ─────────────────────────────────────────────────────────────────────
# VERIFIED -- kernel sha == test sha, both reachable
# ─────────────────────────────────────────────────────────────────────


def test_matching_shas_both_reachable_returns_verified(tmp_path):
    sha = "1234567890" * 4  # 40 chars
    td = _write_task(tmp_path, commit_sha=sha, test_capture_sha=sha)
    with patch(
        "scripts.verify_upstream_commit_chain._git_object_reachable",
        return_value=True,
    ):
        v = verify_task(td)
    assert v.verdict == VERDICT_VERIFIED
    assert v.kernel_sha_reachable is True
    assert v.test_sha_reachable is True


# ─────────────────────────────────────────────────────────────────────
# NEEDS_MANUAL -- kernel sha != test sha, both reachable
# ─────────────────────────────────────────────────────────────────────


def test_diverging_shas_both_reachable_returns_needs_manual(tmp_path):
    kernel = "1" * 40
    test = "2" * 40
    td = _write_task(tmp_path, commit_sha=kernel, test_capture_sha=test)
    with patch(
        "scripts.verify_upstream_commit_chain._git_object_reachable",
        return_value=True,
    ):
        v = verify_task(td)
    assert v.verdict == VERDICT_NEEDS_MANUAL
    assert "human review" in v.reason or "drift" in v.reason


# ─────────────────────────────────────────────────────────────────────
# BACKLOG_PENDING_GOLDEN -- one or both anchors unreachable
# ─────────────────────────────────────────────────────────────────────


def test_diverging_shas_one_unreachable_returns_backlog(tmp_path):
    """When the test sha is not in the upstream tree, the chain claim
    cannot be verified -> force source ②."""
    kernel = "1" * 40
    test = "2" * 40
    td = _write_task(tmp_path, commit_sha=kernel, test_capture_sha=test)

    def fake_reach(repo, sha):
        return sha == kernel  # only kernel sha is reachable

    with patch(
        "scripts.verify_upstream_commit_chain._git_object_reachable",
        side_effect=fake_reach,
    ):
        v = verify_task(td)
    assert v.verdict == VERDICT_BACKLOG
    assert v.kernel_sha_reachable is True
    assert v.test_sha_reachable is False
    assert "unreachable" in v.reason


def test_repo_with_no_vendored_git_returns_backlog(tmp_path):
    """meta.repo=FBGEMM has no vendored .git; the verifier returns
    BACKLOG with a clear reason."""
    td = _write_task(
        tmp_path,
        repo="FBGEMM",
        commit_sha="a" * 40,
        test_capture_sha="a" * 40,
    )
    v = verify_task(td)
    assert v.verdict == VERDICT_BACKLOG
    assert "FBGEMM" in v.reason or "no vendored" in v.reason


# ─────────────────────────────────────────────────────────────────────
# Error handling -- malformed meta
# ─────────────────────────────────────────────────────────────────────


def test_missing_meta_yaml_returns_backlog(tmp_path):
    td = tmp_path / "L_no_meta"
    td.mkdir()
    # no meta.yaml
    v = verify_task(td)
    assert v.verdict == VERDICT_BACKLOG
    assert "meta.yaml missing" in v.reason


def test_malformed_meta_yaml_returns_backlog(tmp_path):
    td = tmp_path / "L_bad_meta"
    td.mkdir()
    (td / "meta.yaml").write_text("not: : valid: yaml: at: all: [")
    v = verify_task(td)
    assert v.verdict == VERDICT_BACKLOG
    assert "YAML" in v.reason
