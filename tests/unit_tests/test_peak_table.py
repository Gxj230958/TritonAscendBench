"""W1.1: 910B π/β peak table loader tests.

Covers: table loads, loader returns {compute_class: π} / {resource: β}
dicts, all DESIGN-V2 §5.6 compute-class / resource keys present, unit
normalisation (TFLOP/s & GB/s → FLOP/s & byte/s), [unverified] placeholder
handling via require_verified, and consistency with the coarse-grained
[hardware.910b] fields in version_matrix.toml.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from ascendbench.perf.peak_table import (
    ABSOLUTE_VERIFIED_PROVENANCES,
    DEFAULT_PEAK_TABLE_PATH,
    REQUIRED_COMPUTE_CLASSES,
    REQUIRED_RESOURCES,
    PeakTableError,
    is_absolute_verified,
    is_ranking_usable,
    load_compute_peaks,
    load_peak_provenance,
    load_peak_table,
    load_resource_peaks,
)

REPO = Path(__file__).resolve().parent.parent.parent


# --------------------------------------------------------------------------
# table presence & schema
# --------------------------------------------------------------------------

def test_peak_table_file_exists():
    assert DEFAULT_PEAK_TABLE_PATH.exists(), DEFAULT_PEAK_TABLE_PATH


def test_table_has_all_required_compute_class_sections():
    data = tomllib.loads(DEFAULT_PEAK_TABLE_PATH.read_text())
    for name in REQUIRED_COMPUTE_CLASSES:
        assert name in data["compute_class"], f"missing compute_class.{name}"


def test_table_has_all_required_resource_sections():
    data = tomllib.loads(DEFAULT_PEAK_TABLE_PATH.read_text())
    for name in REQUIRED_RESOURCES:
        assert name in data["resource"], f"missing resource.{name}"


def test_required_keys_match_design_v2_section_5_6():
    """DESIGN-V2 §5.6: cube fp16/bf16, int8, vector + HBM / interconnect."""
    assert set(REQUIRED_COMPUTE_CLASSES) == {
        "fp16_cube",
        "bf16_cube",
        "int8",
        "vector",
    }
    assert set(REQUIRED_RESOURCES) == {"hbm", "interconnect"}


# --------------------------------------------------------------------------
# loader return shape
# --------------------------------------------------------------------------

def test_load_compute_peaks_returns_dict_with_all_classes():
    peaks = load_compute_peaks()
    assert isinstance(peaks, dict)
    assert set(peaks) == set(REQUIRED_COMPUTE_CLASSES)
    for v in peaks.values():
        assert isinstance(v, float)


def test_load_resource_peaks_returns_dict_with_all_resources():
    peaks = load_resource_peaks()
    assert isinstance(peaks, dict)
    assert set(peaks) == set(REQUIRED_RESOURCES)
    for v in peaks.values():
        assert isinstance(v, float)


def test_load_peak_table_returns_both_dicts():
    compute, resource = load_peak_table()
    assert set(compute) == set(REQUIRED_COMPUTE_CLASSES)
    assert set(resource) == set(REQUIRED_RESOURCES)


# --------------------------------------------------------------------------
# unit normalisation
# --------------------------------------------------------------------------

def test_compute_peaks_normalised_to_flops_per_second():
    """fp16_cube 256 TFLOP/s → 2.56e14 FLOP/s; int8 512 TOP/s → 5.12e14."""
    peaks = load_compute_peaks()
    assert peaks["fp16_cube"] == pytest.approx(256.0 * 1e12)
    assert peaks["int8"] == pytest.approx(512.0 * 1e12)


def test_resource_peaks_normalised_to_bytes_per_second():
    """HBM 1600 GB/s → 1.6e12 byte/s."""
    peaks = load_resource_peaks()
    assert peaks["hbm"] == pytest.approx(1600.0 * 1e9)


def test_peaks_consistent_with_version_matrix_hardware_910b():
    """The refined table must agree with the coarse [hardware.910b] fields."""
    vm = tomllib.loads((REPO / "ascendbench" / "policies" / "version_matrix.toml").read_text())
    hw = vm["hardware"]["910b"]
    compute = load_compute_peaks()
    resource = load_resource_peaks()
    assert compute["fp16_cube"] == pytest.approx(hw["peak_fp16_tflops"] * 1e12)
    assert compute["int8"] == pytest.approx(hw["peak_int8_tops"] * 1e12)
    assert resource["hbm"] == pytest.approx(hw["hbm_bandwidth_gbps"] * 1e9)


# --------------------------------------------------------------------------
# [unverified] placeholder handling — fail-closed for scoring
# --------------------------------------------------------------------------

def test_require_verified_raises_on_unverified_compute_class(tmp_path):
    """A 0.0 compute peak is an [unverified] placeholder; require_verified
    fails closed. (The default table's vector is now filled, so use a tmp
    table that still has a 0.0 vector.)"""
    p = tmp_path / "pk.toml"
    p.write_text(
        "[compute_class.fp16_cube]\npeak_tflops = 256.0\n"
        "[compute_class.bf16_cube]\npeak_tflops = 256.0\n"
        "[compute_class.int8]\npeak_tops = 512.0\n"
        "[compute_class.vector]\npeak_tflops = 0.0\n"
        "[resource.hbm]\npeak_bandwidth_gbps = 1600.0\n"
        "[resource.interconnect]\npeak_bandwidth_gbps = 0.0\n")
    with pytest.raises(PeakTableError, match="vector"):
        load_compute_peaks(p, require_verified=True)


def test_require_verified_raises_on_unverified_resource():
    """interconnect β is an [unverified] 0.0 placeholder."""
    with pytest.raises(PeakTableError, match="interconnect"):
        load_resource_peaks(require_verified=True)


def test_default_mode_returns_unverified_placeholders_without_raising():
    """Without require_verified the loader returns the table for inspection;
    a remaining placeholder (interconnect) shows as 0.0."""
    compute = load_compute_peaks()
    resource = load_resource_peaks()
    # interconnect is still a 0.0 placeholder (no current task references it).
    assert resource["interconnect"] == 0.0
    # filled peaks are positive (vector now carries the academic-spec value).
    assert compute["vector"] > 0.0
    assert compute["fp16_cube"] > 0.0
    assert resource["hbm"] > 0.0


# --------------------------------------------------------------------------
# provenance: ranking_usable vs absolute_verified
# --------------------------------------------------------------------------

def _write_peak_table(tmp_path, vector_provenance=None):
    """Minimal peak table; vector gets a provenance field only if given."""
    lines = [
        "[compute_class.fp16_cube]", "peak_tflops = 256.0",
        'provenance = "huawei-official"',
        "[compute_class.bf16_cube]", "peak_tflops = 256.0",
        "[compute_class.int8]", "peak_tops = 512.0",
        'provenance = "huawei-official"',
        "[compute_class.vector]", "peak_tflops = 20.0",
    ]
    if vector_provenance:
        lines.append(f'provenance = "{vector_provenance}"')
    lines += [
        "[resource.hbm]", "peak_bandwidth_gbps = 1600.0",
        'provenance = "huawei-official"',
        "[resource.interconnect]", "peak_bandwidth_gbps = 0.0",
    ]
    p = tmp_path / "pk.toml"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_is_ranking_usable_positive_finite_only():
    assert is_ranking_usable(1.0) is True
    assert is_ranking_usable(20e12) is True
    assert is_ranking_usable(0.0) is False
    assert is_ranking_usable(-1.0) is False
    assert is_ranking_usable(float("inf")) is False
    assert is_ranking_usable(float("nan")) is False


def test_is_absolute_verified_requires_official_or_measured():
    # positive + official/measured -> absolute-verified.
    assert is_absolute_verified(67e12, "nvidia-official") is True
    assert is_absolute_verified(256e12, "huawei-official") is True
    assert is_absolute_verified(20e12, "on-device-measured") is True
    # positive but estimate/academic-spec -> ranking-only, NOT absolute.
    assert is_absolute_verified(20e12, "third-party-academic-spec") is False
    assert is_absolute_verified(16e12, "architecture-estimate") is False
    assert is_absolute_verified(20e12, "unspecified") is False
    # non-positive -> never absolute, even with official provenance.
    assert is_absolute_verified(0.0, "nvidia-official") is False


def test_absolute_verified_set_is_official_plus_measured():
    assert ABSOLUTE_VERIFIED_PROVENANCES == frozenset(
        {"nvidia-official", "huawei-official", "on-device-measured"})


def test_load_peak_provenance_parses_field(tmp_path):
    p = _write_peak_table(tmp_path, vector_provenance="third-party-academic-spec")
    compute_prov, resource_prov = load_peak_provenance(p)
    assert compute_prov["fp16_cube"] == "huawei-official"
    assert compute_prov["vector"] == "third-party-academic-spec"
    assert resource_prov["hbm"] == "huawei-official"


def test_missing_provenance_defaults_unspecified(tmp_path):
    p = _write_peak_table(tmp_path, vector_provenance=None)
    compute_prov, resource_prov = load_peak_provenance(p)
    # bf16_cube / vector / interconnect carry no provenance -> "unspecified".
    assert compute_prov["bf16_cube"] == "unspecified"
    assert compute_prov["vector"] == "unspecified"
    assert resource_prov["interconnect"] == "unspecified"


def test_academic_spec_vector_is_ranking_usable_not_absolute(tmp_path):
    """Headline invariant: an academic-spec vector peak (20T) is usable
    for RANKING but NOT for absolute-efficiency claims."""
    p = _write_peak_table(tmp_path, vector_provenance="third-party-academic-spec")
    compute, _ = load_peak_table(p)
    compute_prov, _ = load_peak_provenance(p)
    vec_val = compute["vector"]
    assert is_ranking_usable(vec_val) is True
    assert is_absolute_verified(vec_val, compute_prov["vector"]) is False


# --------------------------------------------------------------------------
# error paths
# --------------------------------------------------------------------------

def test_missing_table_raises():
    with pytest.raises(PeakTableError, match="not found"):
        load_compute_peaks(Path("/nonexistent/peak_table.toml"))


def test_table_missing_required_class_raises(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(
        "[compute_class.fp16_cube]\npeak_tflops = 256.0\n"
        "[resource.hbm]\npeak_bandwidth_gbps = 1600.0\n"
    )
    with pytest.raises(PeakTableError, match="bf16_cube"):
        load_compute_peaks(bad)


def test_table_missing_required_resource_raises(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(
        "[compute_class.fp16_cube]\npeak_tflops = 256.0\n"
        "[compute_class.bf16_cube]\npeak_tflops = 256.0\n"
        "[compute_class.int8]\npeak_tops = 512.0\n"
        "[compute_class.vector]\npeak_tflops = 1.0\n"
        "[resource.hbm]\npeak_bandwidth_gbps = 1600.0\n"
    )
    with pytest.raises(PeakTableError, match="interconnect"):
        load_resource_peaks(bad)


def test_compute_class_without_peak_field_raises(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(
        "[compute_class.fp16_cube]\ndescription = \"no peak\"\n"
        "[compute_class.bf16_cube]\npeak_tflops = 256.0\n"
        "[compute_class.int8]\npeak_tops = 512.0\n"
        "[compute_class.vector]\npeak_tflops = 1.0\n"
    )
    with pytest.raises(PeakTableError, match="peak_tflops"):
        load_compute_peaks(bad)
