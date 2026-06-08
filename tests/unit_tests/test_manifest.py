"""Unit tests for SolutionManifest round-trip."""
from __future__ import annotations

import pytest
import yaml

from ascendbench.manifest import SolutionManifest


def test_round_trip_dict():
    doc = {
        "submission_id": "01HXXX",
        "task_id": "L1_12_rms_norm",
        "submitter": {
            "org_identity": "sha256:abc",
            "contact": "alice@example.com",
        },
        "triton_ascend_version": "3.2.0",
        "cann_version": "8.5.0",
        "declared_mode": "self-reported",
        "declared_static_checks_pass": True,
        "declared_metamorphic_checks_pass": False,
    }
    m = SolutionManifest.from_dict(doc)
    assert m.submission_id == "01HXXX"
    assert m.task_id == "L1_12_rms_norm"
    assert m.submitter_org == "sha256:abc"
    assert m.declared_mode == "self-reported"

    rt = m.to_dict()
    assert rt["submission_id"] == doc["submission_id"]
    assert rt["submitter"]["org_identity"] == doc["submitter"]["org_identity"]


def test_load_from_yaml(tmp_path):
    doc = {
        "submission_id": "01HYYY",
        "task_id": "L2_17_w8a8_block_int8_matmul",
        "submitter": {"org_identity": "fp:1234", "contact": "bob@example.com"},
        "triton_ascend_version": "3.2.0",
        "cann_version": "8.5.0",
    }
    p = tmp_path / "manifest.yaml"
    p.write_text(yaml.safe_dump(doc))
    m = SolutionManifest.load(p)
    assert m.task_id == "L2_17_w8a8_block_int8_matmul"
    assert m.submitter_contact == "bob@example.com"
    assert m.declared_mode == "self-reported"  # default
