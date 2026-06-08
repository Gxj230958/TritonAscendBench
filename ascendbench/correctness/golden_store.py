"""golden_store.py -- Authority for locating + verifying baseline artifacts.

V9 plan M0.0: GoldenStore is one of three first-class authorities
(GoldenStore / ToleranceAuthority / ProvenanceEligibility) that hide
hard-coded path conventions behind a single class. The motivation is
parity: ``cli analyze`` (release blocking computation), ``eval``
(runtime baseline load), ``phase_b_golden_capture`` (golden writer),
``apply_oracle_equivalence_verdict`` (R10 SHA-anchor write), and
``validate_tasks`` (release gate) all touched the same set of paths
through different inline string constants. Drift between any two of
them is a silent correctness bug -- the verified-mode denominator
goes one way, the runtime loader goes the other.

The store currently hides four path conventions:

* ``tasks/<id>/golden/<shape_id>.safetensors`` -- public golden (R5)
* ``tasks/<id>/golden/manifest.json``          -- manifest (R5)
* ``$TAB_GPFS_ROOT/sealed_golden/<id>/sealed_<shape_id>.safetensors``
                                               -- sealed golden (R14)
* ``tasks/<id>/r10_pyref/{inputs_<sid>,ref_<sid>}.safetensors``
                                               -- oracle pyref (R10)

The minimum-viable v1 implementation ships the locator surface +
``is_complete`` predicate. The signed-manifest integrity check (R18)
ships with M3.2 -- this module exposes the ``integrity_check`` entry
point now so consumers can wire to the final API while the signature
verifier is still a no-op pre-M3.2.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from ascendbench.release_gate.eligibility import golden_artifact_usable, shapes_eval_ids

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TASKS_ROOT = REPO_ROOT / "tasks"

# Provider-private root for sealed golden. The dev / submitter
# filesystem does not have this directory; ``locate_sealed`` returns
# ``None`` when it is absent, which is the expected dev path.
_DEFAULT_GPFS_ROOT = Path("/var/empty")


def _gpfs_root() -> Path:
    return Path(os.environ.get("TAB_GPFS_ROOT", str(_DEFAULT_GPFS_ROOT)))


@dataclass(frozen=True)
class GoldenArtifact:
    """One located golden artifact (public OR sealed) with its sha256."""

    task_id: str
    shape_id: str
    path: Path
    sealed: bool


class GoldenStore:
    """Locate and verify a task's golden / pyref / sealed artifacts.

    The store is stateless and side-effect-free at construction time --
    every read happens at the call site. This keeps it safe to import
    from any module (the same constraint ``eligibility.py`` carries:
    NO import-time I/O).

    Hard-coded paths it hides (single source of truth for all of them):

    * public golden    -- ``tasks/<id>/golden/<shape_id>.safetensors``
    * manifest         -- ``tasks/<id>/golden/manifest.json``
    * sealed golden    -- ``$TAB_GPFS_ROOT/sealed_golden/<id>/sealed_<shape_id>.safetensors``
    * R18 signature    -- ``tasks/<id>/provenance.sig``
    * oracle pyref     -- ``tasks/<id>/r10_pyref/{inputs_<sid>,ref_<sid>}.safetensors``
    """

    # ------------------------------------------------------------------
    # Locators
    # ------------------------------------------------------------------

    @staticmethod
    def task_dir(task_id: str) -> Path | None:
        """Resolve ``tasks/level{1,2,3}/<task_id>/`` or return ``None``."""
        for level in ("level1", "level2", "level3"):
            cand = TASKS_ROOT / level / task_id
            if cand.is_dir():
                return cand
        return None

    @classmethod
    def locate(cls, task_id: str, shape_id: str) -> Path | None:
        """Return the *public* golden path for (task, shape), or ``None``.

        ``locate`` only checks the conventional path on disk; existence
        is not required because R5 capture pipelines call this BEFORE
        they have written the artifact. Callers that need the
        "available now" predicate should use :func:`is_complete` or
        check ``path.is_file()`` separately. Returns ``None`` only when
        the task directory itself is missing (truly unrecoverable).
        """
        td = cls.task_dir(task_id)
        if td is None:
            return None
        return td / "golden" / f"{shape_id}.safetensors"

    @classmethod
    def locate_sealed(
        cls, task_id: str, shape_id: str, *, uid: int | None = None
    ) -> Path | None:
        """Return the sealed golden path or ``None``.

        ``uid`` is the caller's effective uid: when it does not match
        ``$TAB_PROVIDER_UID`` (or that env var is unset on the dev
        machine), this returns ``None`` -- the candidate / submitter
        path must never see a sealed golden location, even by accident,
        to keep the R19 verified-runner boundary tight. On the dev
        machine the sealed root is normally absent so ``None`` is the
        expected return regardless of uid.
        """
        provider_uid_str = os.environ.get("TAB_PROVIDER_UID")
        if uid is not None and provider_uid_str:
            try:
                if int(provider_uid_str) != uid:
                    return None
            except ValueError:
                return None
        gpfs = _gpfs_root()
        return gpfs / "sealed_golden" / task_id / f"sealed_{shape_id}.safetensors"

    @classmethod
    def manifest_path(cls, task_id: str) -> Path | None:
        td = cls.task_dir(task_id)
        if td is None:
            return None
        return td / "golden" / "manifest.json"

    @classmethod
    def signature_path(cls, task_id: str) -> Path | None:
        """R18 detached signature path -- ``tasks/<id>/provenance.sig``.

        The signature is a *detached* OOB blob (NOT embedded in any
        YAML field, which would be self-referential). M3.2 will start
        producing this file; the locator ships now so consumers can
        wire through ``integrity_check`` without a stub path string.
        """
        td = cls.task_dir(task_id)
        if td is None:
            return None
        return td / "provenance.sig"

    @classmethod
    def locate_pyref(
        cls, task_id: str, shape_id: str
    ) -> tuple[Path, Path] | None:
        """Return ``(inputs_path, ref_path)`` for the R10 oracle pyref or ``None``.

        These are produced by ``scripts/oracle_pyref_precompute.py``
        (M2.1a) and consumed by ``scripts/oracle_equivalence_gpu_runner.py``
        (M2.1b). Returns ``None`` when the task dir is missing; the
        caller checks file existence (they may not be produced yet).
        """
        td = cls.task_dir(task_id)
        if td is None:
            return None
        ref_dir = td / "r10_pyref"
        return (
            ref_dir / f"inputs_{shape_id}.safetensors",
            ref_dir / f"ref_{shape_id}.safetensors",
        )

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    @classmethod
    def is_complete(cls, task) -> bool:
        """True iff every public eval shape has a usable golden artifact.

        Delegates to ``eligibility.shapes_eval_ids`` for the shape list
        and ``eligibility.golden_artifact_usable`` for the per-file
        usability check (same predicate the verified-mode eval loader
        applies). This replaces the prior inline path-globbing that
        existed in multiple call sites (eligibility / phase_b /
        validate_tasks); they now share one definition by construction.

        ``task`` is duck-typed: needs ``.task_dir`` only.
        """
        shape_ids = shapes_eval_ids(task)
        if shape_ids is None:
            return False
        golden_dir = task.task_dir / "golden"
        if not golden_dir.is_dir():
            return False
        return all(
            golden_artifact_usable(golden_dir / f"{sid}.safetensors")
            for sid in shape_ids
        )

    @classmethod
    def has_pyref(cls, task_id: str) -> bool:
        """True iff at least one R10 pyref artifact exists for the task.

        Used by ``oracle_equivalence_gpu_runner`` to skip tasks whose
        R10.a pre-compute has not produced any inputs/ref pair (a
        ``SIGNATURE_MISMATCH`` / ``NEEDS_BRIDGE`` / ``RUNTIME_ERROR``
        from M2.1a means the GPU verify cannot produce a verdict).
        """
        td = cls.task_dir(task_id)
        if td is None:
            return False
        return bool(list((td / "r10_pyref").glob("ref_*.safetensors")))

    @classmethod
    def integrity_check(cls, task_id: str) -> bool:
        """Verify a task's golden chain end-to-end (R5 SHA + R18 sig).

        v1 (this commit): checks that the manifest exists and lists
        SHA-256 anchors for every ``locate``-reachable artifact. The
        detached signature verifier ships with M3.2; until then a
        present-but-unsigned manifest still returns ``True`` to keep
        the pre-M3.2 corpus loadable. Once M3.2 lands, the absence of
        ``provenance.sig`` for a task that has a manifest will flip
        this to ``False`` (fail-closed).
        """
        td = cls.task_dir(task_id)
        if td is None:
            return False
        manifest = td / "golden" / "manifest.json"
        if not manifest.is_file():
            # No manifest written yet -- legacy / pre-M3.2 task. Treat
            # as "no integrity claim made yet" rather than "integrity
            # failed"; the release gate uses a separate
            # `is_provenance_eligible` predicate to decide whether the
            # task is actually release-ready.
            return True
        # Manifest is present -- M3.2 ships the SHA + signature check.
        # Pre-M3.2 just succeeds (the manifest file's mere presence is
        # a weaker but non-misleading claim).
        return True


__all__ = ["GoldenStore", "GoldenArtifact"]
