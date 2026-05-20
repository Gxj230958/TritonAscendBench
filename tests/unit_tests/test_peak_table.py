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

from triton_ascend_bench.peak_table import (
    DEFAULT_PEAK_TABLE_PATH,
    REQUIRED_COMPUTE_CLASSES,
    REQUIRED_RESOURCES,
    PeakTableError,
    load_compute_peaks,
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
    """The细化 table must agree with the coarse [hardware.910b] fields."""
    vm = tomllib.loads((REPO / "infra" / "version_matrix.toml").read_text())
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
    """vector π is an [unverified] 0.0 placeholder; scoring must fail closed."""
    with pytest.raises(PeakTableError, match="vector"):
        load_compute_peaks(require_verified=True)


def test_require_verified_raises_on_unverified_resource():
    """interconnect β is an [unverified] 0.0 placeholder."""
    with pytest.raises(PeakTableError, match="interconnect"):
        load_resource_peaks(require_verified=True)


def test_default_mode_returns_unverified_placeholders_without_raising():
    """Without require_verified the loader still returns the table so
    callers can inspect it; the placeholder shows as 0.0."""
    compute = load_compute_peaks()
    resource = load_resource_peaks()
    assert compute["vector"] == 0.0
    assert resource["interconnect"] == 0.0
    # verified peaks are positive
    assert compute["fp16_cube"] > 0.0
    assert resource["hbm"] > 0.0


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
