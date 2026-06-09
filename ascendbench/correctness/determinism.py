"""determinism.py -- R20 determinism verification, v1.

A kernel's captured golden is only trustworthy if the kernel produces
the same bytes every time it runs with the same inputs / seed / env.
R20 v1 ships a *deterministic-only* check: run N=3 times, require bit
identity. Tasks that fail this gate are marked
``gpu_kernel_capture_status=BLOCKED_NON_DETERMINISTIC,
gpu_kernel_deterministic=false`` and excluded from
``release_ready_with_provenance``. The band-mode (tolerance-bounded
non-determinism) variant is explicitly v2 / follow-up plan.

This module is GPU-agnostic by design: the caller passes a *function*
that runs one capture (returning the output tensor / dict of tensors),
plus the env-seed setup it uses. The gate runs the closure N=3 times
and computes the SHA-256 of the canonical output bytes, then asserts
bit identity across runs. The same gate can be exercised on CPU in
unit tests by passing a CPU-only closure -- no @pytest.mark.gpu
marker needed for the gate itself (GPU-tagged tests cover the wiring).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeterminismResult:
    """Verdict + diagnostic for an N=K determinism run."""

    deterministic: bool
    run_count: int
    digests: tuple[str, ...] = ()
    # Distinct digests observed (≥ 2 means non-deterministic).
    unique_digests: tuple[str, ...] = ()
    reason: str = ""


def _canonical_tensor_bytes(obj: Any) -> bytes:
    """Serialise a tensor (or nested dict/list of tensors) to canonical
    bytes for SHA-256 hashing.

    Lazy-imports torch so this module remains importable from any
    environment. The encoding is intentionally simple: each tensor is
    serialised as ``<dtype><shape><raw bytes>``; dicts walk keys in
    sorted order; lists / tuples walk in order.
    """
    import torch  # noqa: PLC0415

    if isinstance(obj, torch.Tensor):
        t = obj.detach().contiguous().cpu()
        header = (
            str(t.dtype).encode("utf-8")
            + b"|"
            + repr(tuple(t.shape)).encode("utf-8")
            + b"|"
        )
        return header + t.numpy().tobytes()

    if isinstance(obj, Mapping):
        parts: list[bytes] = []
        for key in sorted(obj.keys()):
            parts.append(b"K:" + str(key).encode("utf-8") + b"=")
            parts.append(_canonical_tensor_bytes(obj[key]))
        return b"".join(parts)

    if isinstance(obj, (list, tuple)):
        parts2: list[bytes] = [b"L:"]
        for v in obj:
            parts2.append(_canonical_tensor_bytes(v))
            parts2.append(b"|")
        return b"".join(parts2)

    # Scalars / strings: encode as repr (deterministic).
    return b"S:" + repr(obj).encode("utf-8")


def _sha256_of(obj: Any) -> str:
    return hashlib.sha256(_canonical_tensor_bytes(obj)).hexdigest()


def verify_determinism(
    run_once: Callable[[], Any], *, n_runs: int = 3,
) -> DeterminismResult:
    """Run ``run_once`` ``n_runs`` times; assert bit identity.

    ``run_once`` is a zero-argument closure that returns the tensor /
    nested dict produced by one capture. The caller is responsible
    for resetting seeds / env BEFORE this call -- ``verify_determinism``
    does NOT manage that (different capture pipelines have different
    determinism setups; coupling it here would be brittle).

    Returns a populated :class:`DeterminismResult`. ``n_runs < 2``
    raises ``ValueError`` -- single-run verification is meaningless.
    """
    if n_runs < 2:
        raise ValueError(f"verify_determinism: n_runs must be >= 2 (got {n_runs})")
    digests: list[str] = []
    for i in range(n_runs):
        out = run_once()
        d = _sha256_of(out)
        digests.append(d)
        logger.debug("determinism run %d -> %s", i, d[:16])
    unique = tuple(sorted(set(digests)))
    if len(unique) == 1:
        return DeterminismResult(
            deterministic=True,
            run_count=n_runs,
            digests=tuple(digests),
            unique_digests=unique,
        )
    return DeterminismResult(
        deterministic=False,
        run_count=n_runs,
        digests=tuple(digests),
        unique_digests=unique,
        reason=(
            f"R20: {n_runs} runs produced {len(unique)} distinct "
            f"output digests; kernel is non-deterministic"
        ),
    )


@dataclass(frozen=True)
class DeterminismVerdict:
    """Translation of :class:`DeterminismResult` into the V9 schema
    fields written to ``provenance.golden:``.

    * ``gpu_kernel_capture_status``: ``DONE`` if deterministic;
      ``BLOCKED_NON_DETERMINISTIC`` otherwise.
    * ``gpu_kernel_deterministic``: the bit-identity result.

    Callers (M3.2 / M2.1c) write these straight into ``meta.yaml``.
    """

    gpu_kernel_capture_status: str
    gpu_kernel_deterministic: bool
    reason: str = ""


def to_verdict(result: DeterminismResult) -> DeterminismVerdict:
    """Map a :class:`DeterminismResult` to the V9 schema verdict pair."""
    if result.deterministic:
        return DeterminismVerdict(
            gpu_kernel_capture_status="DONE",
            gpu_kernel_deterministic=True,
        )
    return DeterminismVerdict(
        gpu_kernel_capture_status="BLOCKED_NON_DETERMINISTIC",
        gpu_kernel_deterministic=False,
        reason=result.reason,
    )


__all__ = [
    "DeterminismResult",
    "DeterminismVerdict",
    "to_verdict",
    "verify_determinism",
]
