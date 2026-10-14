"""manifest.py — solution submission manifest.

Per spec, every submission ships as a tar.gz containing:
    manifest.yaml
    solution.py               (the submitter's triton-ascend kernel)
    runtime_env/Dockerfile    (optional pinned environment)
    notes.md                  (optional)

manifest.yaml schema:

    submission_id: <ulid>
    task_id: <task_id>
    submitter:
      org_identity: <hashed-gpg-or-email>
      contact: <email>
    triton_ascend_version: 3.2.0
    cann_version: 8.5.0
    container_digest: <sha256-of-docker-image-if-verified-mode>
    declared_mode: self-reported | verified
    declared_static_checks_pass: bool
    declared_metamorphic_checks_pass: bool
    notes_md: optional path
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SolutionManifest:
    submission_id: str
    task_id: str
    submitter_org: str
    submitter_contact: str
    triton_ascend_version: str
    cann_version: str
    declared_mode: str = "self-reported"
    container_digest: str | None = None
    declared_static_checks_pass: bool = False
    declared_metamorphic_checks_pass: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> SolutionManifest:
        sub = doc["submitter"]
        return cls(
            submission_id=doc["submission_id"],
            task_id=doc["task_id"],
            submitter_org=sub["org_identity"],
            submitter_contact=sub["contact"],
            triton_ascend_version=doc["triton_ascend_version"],
            cann_version=doc["cann_version"],
            declared_mode=doc.get("declared_mode", "self-reported"),
            container_digest=doc.get("container_digest"),
            declared_static_checks_pass=bool(
                doc.get("declared_static_checks_pass", False)
            ),
            declared_metamorphic_checks_pass=bool(
                doc.get("declared_metamorphic_checks_pass", False)
            ),
            raw=doc,
        )

    @classmethod
    def load(cls, path: Path) -> SolutionManifest:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "task_id": self.task_id,
            "submitter": {
                "org_identity": self.submitter_org,
                "contact": self.submitter_contact,
            },
            "triton_ascend_version": self.triton_ascend_version,
            "cann_version": self.cann_version,
            "declared_mode": self.declared_mode,
            "container_digest": self.container_digest,
            "declared_static_checks_pass": self.declared_static_checks_pass,
            "declared_metamorphic_checks_pass": self.declared_metamorphic_checks_pass,
        }
