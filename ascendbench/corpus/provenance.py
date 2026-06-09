"""provenance.py -- nested ``provenance.golden:`` schema parser + validator.

Adds a nested ``provenance.golden:`` sub-block to each
``meta.yaml`` so a task can carry the full audit trail for HOW its
baseline value was produced (oracle equivalence verdict, captured
golden SHA, signed manifest, reviewer attestation, etc.) without
disturbing the pre-existing top-level keys (``added_in``, ``authors``,
``last_modified``).

The split between top-level and nested fields is intentional:

* top-level ``provenance:`` keys are *content metadata* — who added the
  task and when. They are stable, manually edited at task creation, and
  unrelated to baseline correctness.
* nested ``provenance.golden:`` is the *baseline trust chain* —
  populated by the R10/R5 capture pipelines and verified by the release
  gate. Every field is written by a script, never by hand.

The full schema lives in ``specs/task_api_contract.md §5d``; this
module is the single source of truth for parsing and closed-enum
validation. Both ``ascendbench.corpus.dataset`` (loader-lenient) and
``scripts.corpus.validate_tasks`` (validator-strict) consume it, so the
in-corpus + release-gate views can never drift.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Closed enums. Anything outside these sets is a
# validator error. Open / operational fields (``verified_at``,
# ``equivalence_per_shape``, ``golden_source``, ``tolerance_class_used``
# — emitted by ``scripts/oracle/apply_oracle_equivalence_verdict.py``) are
# preserved verbatim in ``ProvenanceGoldenSpec.extra`` so the data round-
# trips, but they are NOT closed-enum checked here. They are operational
# artifacts of the pipeline, not part of the schema contract.
VALID_SOURCE = frozenset({"upstream_author_test", "gpu_kernel_rerun"})
VALID_TOLERANCE_SOURCE = frozenset({
    "upstream_test_explicit",
    "upstream_test_default",
    "migration_drift_budget",
})
# Added ``NEEDS_MANUAL`` for the R12 SHA-chain drift case.
# Added the wrapper-rank pre-check verdicts.
VALID_ORACLE_KERNEL_EQUIVALENCE = frozenset({
    "VERIFIED", "FAILED", "MIXED", "SKIPPED", "NEEDS_MANUAL",
    "LAUNCH_BLOCKED",  # GPU couldn't invoke the kernel at all
    "WRAPPER_CONTRACT_MISMATCH",  # shapes.yaml.inputs.dims disagree with wrapper unpack
    "ORACLE_KERNEL_LAYOUT_MISMATCH",  # oracle and wrapper disagree fundamentally
    # R10.b rebuilt inputs diverged from R10.a's captured ref -- precision
    # comparison untrustworthy, NOT a precision FAILED (H2 guard).
    "INPUT_DRIFT",
    # A large-shape divergence accepted as a dtype-ULP ceiling on an allowlisted
    # task -- DISTINCT from VERIFIED, never counted as a strict pass, but a valid
    # (correctly-aligned) corpus task. See ascendbench/policies/ceiling_acceptance.yaml.
    "CEILING_ACCEPTED",
    # Frozen upstream kernel carries a genuine off-by-one/bug; a correct oracle
    # necessarily diverges -> unverifiable by construction (set when
    # upstream.known_kernel_bug is true). Excluded from the alignment denominator.
    "BLOCKED_UPSTREAM_BUG",
})
VALID_GPU_KERNEL_CAPTURE_STATUS = frozenset({
    "PENDING", "DONE", "FAILED", "BLOCKED_NON_DETERMINISTIC",
})
VALID_RECONSTRUCTABILITY_RISK = frozenset({"low", "medium", "high"})

# Required-when-present per the §M1.2 schema box. None of the keys
# are required when the ``golden:`` block itself is absent — that is the
# explicit grace period the plan asks for. When ``golden:`` IS present,
# its values must conform to the closed enums above; absent keys are
# allowed (a partially-populated block is normal during pipeline
# execution, before all stages have run).
_KNOWN_KEYS = frozenset({
    "source",
    "tolerance_source",
    "atol",
    "rtol",
    "ref_path",
    "test_file_path",
    "test_capture_sha",
    "golden_sha256",
    "oracle_kernel_equivalence",
    "gpu_kernel_capture_status",
    "gpu_kernel_deterministic",
    "upstream_closure_fingerprint",
    "reviewers",
    "reconstructability_risk",
    # Operational fields (open; preserved verbatim in ``extra``):
    "verified_at",
    "tolerance_class_used",
    "equivalence_per_shape",
    "golden_source",
})


@dataclass(frozen=True)
class ProvenanceGoldenSpec:
    """Typed view of ``meta.yaml.provenance.golden:``.

    Every field is ``None`` (or empty) when absent from the block; the
    loader is intentionally lenient. ``extra`` holds any operational
    key not part of the canonical schema (e.g. M2.1c's ``verified_at``
    / ``equivalence_per_shape``), so unparsed pipeline metadata round-
    trips faithfully through the dataclass.
    """

    source: str | None = None
    tolerance_source: str | None = None
    atol: float | None = None
    rtol: float | None = None
    ref_path: str | None = None
    test_file_path: str | None = None
    test_capture_sha: str | None = None
    golden_sha256: str | None = None
    oracle_kernel_equivalence: str | None = None
    gpu_kernel_capture_status: str | None = None
    gpu_kernel_deterministic: bool | None = None
    upstream_closure_fingerprint: str | None = None
    reviewers: tuple[str, ...] = ()
    reconstructability_risk: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def parse_provenance_golden(meta: dict[str, Any]) -> ProvenanceGoldenSpec | None:
    """Return a typed view of ``meta.provenance.golden:``, or ``None``.

    Grace-period semantics: returns ``None`` when the nested block is
    absent (legacy meta or a task that hasn't run R10/R5 yet). When
    present, returns a populated ``ProvenanceGoldenSpec`` — values are
    coerced where unambiguous (float-able numbers, list-to-tuple for
    ``reviewers``) and copied verbatim otherwise so a malformed value
    still reaches the validator with its original type.
    """
    prov = meta.get("provenance")
    if not isinstance(prov, dict):
        return None
    g = prov.get("golden")
    if not isinstance(g, dict):
        return None

    known: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in g.items():
        if key in _KNOWN_KEYS and key not in {
            "verified_at",
            "tolerance_class_used",
            "equivalence_per_shape",
            "golden_source",
        }:
            known[key] = value
        else:
            extra[key] = value

    reviewers_raw = known.get("reviewers", ())
    if isinstance(reviewers_raw, (list, tuple)):
        reviewers = tuple(str(r) for r in reviewers_raw)
    else:
        # Malformed -- keep visible to the validator (NOT coerced).
        reviewers = ()
        extra["__malformed_reviewers"] = reviewers_raw

    def _maybe_float(v: Any) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return ProvenanceGoldenSpec(
        source=known.get("source"),
        tolerance_source=known.get("tolerance_source"),
        atol=_maybe_float(known.get("atol")),
        rtol=_maybe_float(known.get("rtol")),
        ref_path=known.get("ref_path"),
        test_file_path=known.get("test_file_path"),
        test_capture_sha=known.get("test_capture_sha"),
        golden_sha256=known.get("golden_sha256"),
        oracle_kernel_equivalence=known.get("oracle_kernel_equivalence"),
        gpu_kernel_capture_status=known.get("gpu_kernel_capture_status"),
        gpu_kernel_deterministic=known.get("gpu_kernel_deterministic"),
        upstream_closure_fingerprint=known.get("upstream_closure_fingerprint"),
        reviewers=reviewers,
        reconstructability_risk=known.get("reconstructability_risk"),
        extra=extra,
    )


def validate_provenance_golden(meta: dict[str, Any]) -> list[str]:
    """Closed-enum + shape check on ``meta.provenance.golden:``.

    Returns a (possibly empty) list of human-readable error strings.
    An absent ``golden:`` block returns ``[]`` (grace period); the
    validator should treat that as a *warning* concern, not an error.

    The check is shape-strict on closed enums (``source``,
    ``tolerance_source``, ``oracle_kernel_equivalence``,
    ``gpu_kernel_capture_status``, ``reconstructability_risk``) and
    type-strict on the numeric / boolean / hash fields. Operational
    extras (``verified_at``, ``equivalence_per_shape``, ...) are
    accepted as-is -- they are pipeline artifacts, not part of the
    contract.
    """
    errs: list[str] = []
    prov = meta.get("provenance")
    if not isinstance(prov, dict):
        # The validator that calls us treats top-level ``provenance:``
        # absence as a separate error. Don't double-report here.
        return errs
    g = prov.get("golden")
    if g is None:
        return errs  # grace period
    if not isinstance(g, dict):
        errs.append(
            f"provenance.golden must be a mapping (got {type(g).__name__})"
        )
        return errs

    src = g.get("source")
    if src is not None and src not in VALID_SOURCE:
        errs.append(
            f"provenance.golden.source must be one of {sorted(VALID_SOURCE)} "
            f"(got {src!r})"
        )

    tsrc = g.get("tolerance_source")
    if tsrc is not None and tsrc not in VALID_TOLERANCE_SOURCE:
        errs.append(
            f"provenance.golden.tolerance_source must be one of "
            f"{sorted(VALID_TOLERANCE_SOURCE)} (got {tsrc!r})"
        )

    oke = g.get("oracle_kernel_equivalence")
    if oke is not None and oke not in VALID_ORACLE_KERNEL_EQUIVALENCE:
        errs.append(
            f"provenance.golden.oracle_kernel_equivalence must be one of "
            f"{sorted(VALID_ORACLE_KERNEL_EQUIVALENCE)} (got {oke!r})"
        )

    gkcs = g.get("gpu_kernel_capture_status")
    if gkcs is not None and gkcs not in VALID_GPU_KERNEL_CAPTURE_STATUS:
        errs.append(
            f"provenance.golden.gpu_kernel_capture_status must be one of "
            f"{sorted(VALID_GPU_KERNEL_CAPTURE_STATUS)} (got {gkcs!r})"
        )

    rr = g.get("reconstructability_risk")
    if rr is not None and rr not in VALID_RECONSTRUCTABILITY_RISK:
        errs.append(
            f"provenance.golden.reconstructability_risk must be one of "
            f"{sorted(VALID_RECONSTRUCTABILITY_RISK)} (got {rr!r})"
        )

    det = g.get("gpu_kernel_deterministic")
    if det is not None and not isinstance(det, bool):
        errs.append(
            "provenance.golden.gpu_kernel_deterministic must be a bool "
            f"(got {type(det).__name__})"
        )

    for fld in ("atol", "rtol"):
        v = g.get(fld)
        if v is None:
            continue
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            errs.append(
                f"provenance.golden.{fld} must be a number (got "
                f"{type(v).__name__})"
            )

    for fld, length in (
        ("golden_sha256", 64),
        ("upstream_closure_fingerprint", 64),
        ("test_capture_sha", 40),
    ):
        v = g.get(fld)
        if v is None:
            continue
        if not isinstance(v, str) or len(v) != length or not all(
            c in "0123456789abcdef" for c in v.lower()
        ):
            errs.append(
                f"provenance.golden.{fld} must be {length}-hex string (got {v!r})"
            )

    reviewers = g.get("reviewers")
    if reviewers is not None and not isinstance(reviewers, list):
        errs.append(
            "provenance.golden.reviewers must be a list (got "
            f"{type(reviewers).__name__})"
        )
    elif isinstance(reviewers, list):
        for i, r in enumerate(reviewers):
            if not isinstance(r, str):
                errs.append(
                    f"provenance.golden.reviewers[{i}] must be a string "
                    f"(got {type(r).__name__})"
                )

    return errs


__all__ = [
    "ProvenanceGoldenSpec",
    "VALID_SOURCE",
    "VALID_TOLERANCE_SOURCE",
    "VALID_ORACLE_KERNEL_EQUIVALENCE",
    "VALID_GPU_KERNEL_CAPTURE_STATUS",
    "VALID_RECONSTRUCTABILITY_RISK",
    "parse_provenance_golden",
    "validate_provenance_golden",
]
