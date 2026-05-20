"""peak_table.py — 910B Roofline π/β 峰值表 loader.

Loads ``infra/peak_table_910b.toml`` and exposes the 910B per-compute-class
peak compute (π_c) and per-resource peak bandwidth (β_r) used by scoring's
Roofline theoretical-latency baseline (DESIGN-V2 §5.6 / §9)::

    t_ref = max( max_c W_c / π_c , max_r M_r / β_r )

``W_c`` is per-compute-class FLOPs and ``M_r`` is per-resource bytes; both
come from each task's cost model. This module supplies π_c and β_r.

Style mirrors ``scripts/verify_version_matrix.py``: stdlib ``tomllib`` (with
a ``tomli`` fallback) and stdlib ``logging``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PEAK_TABLE_PATH = REPO_ROOT / "infra" / "peak_table_910b.toml"

# Keys required by DESIGN-V2 §5.6: cube fp16/bf16, int8, vector compute
# classes and HBM / cross-device-interconnect resources. The loader fails
# closed if the table omits any of these.
REQUIRED_COMPUTE_CLASSES = ("fp16_cube", "bf16_cube", "int8", "vector")
REQUIRED_RESOURCES = ("hbm", "interconnect")

# Per-compute-class peak value may be declared as TFLOP/s or TOP/s; the
# loader normalises both to FLOP/s (resp. OP/s) by this factor.
_TERA = 1e12
# Per-resource bandwidth is declared in GB/s; normalise to byte/s.
_GIGA = 1e9


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
) -> Dict[str, float]:
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

    peaks: Dict[str, float] = {}
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
) -> Dict[str, float]:
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

    peaks: Dict[str, float] = {}
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
    path: Path, peaks: Dict[str, float], kind: str, require_verified: bool
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
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Convenience: return ``(compute_peaks, resource_peaks)`` together.

    See :func:`load_compute_peaks` / :func:`load_resource_peaks`.
    """
    return (
        load_compute_peaks(path, require_verified=require_verified),
        load_resource_peaks(path, require_verified=require_verified),
    )
