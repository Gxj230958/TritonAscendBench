"""peak_table.py — 910B Roofline π/β peak table loader.

Loads ``infra/peak_table_910b.toml`` and exposes the 910B per-compute-class
peak compute (π_c) and per-resource peak bandwidth (β_r) used by scoring's
Roofline theoretical-latency baseline (the design spec §5.6 / §9)::

    t_ref = max( max_c W_c / π_c , max_r M_r / β_r )

``W_c`` is per-compute-class FLOPs and ``M_r`` is per-resource bytes; both
come from each task's cost model. This module supplies π_c and β_r.

Style mirrors ``scripts/verify_version_matrix.py``: stdlib ``tomllib`` (with
a ``tomli`` fallback) and stdlib ``logging``.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PEAK_TABLE_PATH = REPO_ROOT / "infra" / "peak_table_910b.toml"

# Keys required by the design spec §5.6: cube fp16/bf16, int8, vector compute
# classes and HBM / cross-device-interconnect resources. The loader fails
# closed if the table omits any of these.
REQUIRED_COMPUTE_CLASSES = ("fp16_cube", "bf16_cube", "int8", "vector")
REQUIRED_RESOURCES = ("hbm", "interconnect")

# Per-compute-class peak value may be declared as TFLOP/s or TOP/s; the
# loader normalises both to FLOP/s (resp. OP/s) by this factor.
_TERA = 1e12
# Per-resource bandwidth is declared in GB/s; normalise to byte/s.
_GIGA = 1e9

# Provenance trusted for absolute "% of Roofline" claims. Ranking accepts any
# positive peak (t_ref cancels across candidates); academic-spec / estimate /
# unspecified are ranking-only.
ABSOLUTE_VERIFIED_PROVENANCES = frozenset(
    {"nvidia-official", "huawei-official", "on-device-measured"}
)


class PeakTableError(RuntimeError):
    """Raised when the 910B peak table is missing keys or holds an
    unverified (non-positive) placeholder needed for scoring."""


def _load_toml(path: Path) -> dict:
    if not path.exists():
        raise PeakTableError(f"peak table not found: {path}")
    try:
        return tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:  # type: ignore[attr-defined]
        raise PeakTableError(f"cannot parse peak table {path}: {e}") from e


def load_compute_peaks(
    path: Path | str | None = None, *, require_verified: bool = False
) -> dict[str, float]:
    """Return ``{compute_class: π_c}`` in FLOP/s (resp. OP/s for int8).

    Args:
        path: peak table TOML path; defaults to ``infra/peak_table_910b.toml``.
        require_verified: when True, raise :class:`PeakTableError` if any
            π_c is non-positive (an ``[unverified]`` placeholder). When
            False (default) such classes are still returned (value ``0.0``)
            so callers can inspect the table; scoring must pass True so an
            unverified peak never silently distorts ``t_ref``.

    Raises:
        PeakTableError: if the table is missing or omits a required class.
    """
    p = Path(path) if path is not None else DEFAULT_PEAK_TABLE_PATH
    data = _load_toml(p)
    classes = data.get("compute_class", {})
    if not isinstance(classes, dict):
        raise PeakTableError(f"{p}: [compute_class] is not a table")

    peaks: dict[str, float] = {}
    for name in REQUIRED_COMPUTE_CLASSES:
        entry = classes.get(name)
        if not isinstance(entry, dict):
            raise PeakTableError(
                f"{p}: missing required compute class [compute_class.{name}]"
            )
        if "peak_tflops" in entry:
            value = float(entry["peak_tflops"]) * _TERA
        elif "peak_tops" in entry:
            value = float(entry["peak_tops"]) * _TERA
        else:
            raise PeakTableError(
                f"{p}: [compute_class.{name}] has neither peak_tflops nor "
                f"peak_tops"
            )
        peaks[name] = value

    _check_verified(p, peaks, "compute class π", require_verified)
    return peaks


def load_resource_peaks(
    path: Path | str | None = None, *, require_verified: bool = False
) -> dict[str, float]:
    """Return ``{resource: β_r}`` in byte/s.

    Args:
        path: peak table TOML path; defaults to ``infra/peak_table_910b.toml``.
        require_verified: when True, raise :class:`PeakTableError` if any
            β_r is non-positive (an ``[unverified]`` placeholder).

    Raises:
        PeakTableError: if the table is missing or omits a required resource.
    """
    p = Path(path) if path is not None else DEFAULT_PEAK_TABLE_PATH
    data = _load_toml(p)
    resources = data.get("resource", {})
    if not isinstance(resources, dict):
        raise PeakTableError(f"{p}: [resource] is not a table")

    peaks: dict[str, float] = {}
    for name in REQUIRED_RESOURCES:
        entry = resources.get(name)
        if not isinstance(entry, dict):
            raise PeakTableError(
                f"{p}: missing required resource [resource.{name}]"
            )
        if "peak_bandwidth_gbps" not in entry:
            raise PeakTableError(
                f"{p}: [resource.{name}] has no peak_bandwidth_gbps"
            )
        peaks[name] = float(entry["peak_bandwidth_gbps"]) * _GIGA

    _check_verified(p, peaks, "resource β", require_verified)
    return peaks


def _check_verified(
    path: Path, peaks: dict[str, float], kind: str, require_verified: bool
) -> None:
    """Warn (or raise) on non-positive [unverified] placeholder peaks."""
    unverified = sorted(k for k, v in peaks.items() if v <= 0.0)
    if not unverified:
        return
    msg = (
        f"{path}: {kind} unverified/non-positive placeholders: "
        f"{', '.join(unverified)} — must be backfilled on a 910B host "
        f"before use in scoring"
    )
    if require_verified:
        raise PeakTableError(msg)
    logger.warning(msg)


def load_peak_table(
    path: Path | str | None = None, *, require_verified: bool = False
) -> tuple[dict[str, float], dict[str, float]]:
    """Convenience: return ``(compute_peaks, resource_peaks)`` together.

    See :func:`load_compute_peaks` / :func:`load_resource_peaks`.
    """
    return (
        load_compute_peaks(path, require_verified=require_verified),
        load_resource_peaks(path, require_verified=require_verified),
    )


def _load_provenance(
    data: dict, section: str, names: tuple[str, ...]
) -> dict[str, str]:
    """Per-key ``provenance`` string for ``[section.<name>]``; a key that is
    absent or carries no ``provenance`` field maps to ``"unspecified"``."""
    sect = data.get(section, {})
    out: dict[str, str] = {}
    for name in names:
        entry = sect.get(name) if isinstance(sect, dict) else None
        prov = entry.get("provenance") if isinstance(entry, dict) else None
        out[name] = str(prov) if prov else "unspecified"
    return out


def load_peak_provenance(
    path: Path | str | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """``(compute_provenance, resource_provenance)`` per-key provenance strings;
    an absent / no-provenance key maps to ``"unspecified"``."""
    p = Path(path) if path is not None else DEFAULT_PEAK_TABLE_PATH
    data = _load_toml(p)
    return (
        _load_provenance(data, "compute_class", REQUIRED_COMPUTE_CLASSES),
        _load_provenance(data, "resource", REQUIRED_RESOURCES),
    )


def is_ranking_usable(value: float) -> bool:
    """Positive-finite peak — enough to rank (peak scale cancels across candidates)."""
    return math.isfinite(value) and value > 0.0


def is_absolute_verified(value: float, provenance: str) -> bool:
    """Positive-finite and vendor-official / on-device-measured — needed for
    absolute efficiency claims, not just ranking."""
    return is_ranking_usable(value) and provenance in ABSOLUTE_VERIFIED_PROVENANCES
