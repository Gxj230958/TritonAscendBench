"""tolerance.py — apply numeric tolerance per `tolerance_policy.yaml`.

Tolerance classes (full list in `specs/tolerance_policy.md`):
    elementwise_fp16, elementwise_exact_or_inf_safe,
    vector_reduce_fp16, cube_matmul_fp16, cube_matmul_int8,
    flash_attention_fp16, int8_w8a8, fp8_software_emulated, ...

Returns (passed, summary) per pair of tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = REPO_ROOT / "triton_ascend_bench" / "tolerance_policy.yaml"


@dataclass
class ToleranceClass:
    name: str
    atol: float
    rtol: float
    inf_must_match: bool = True
    nan_must_match: bool = True
    sample_fraction: float = 1.0     # 1.0 == compare full tensor
    bit_exact: bool = False          # integer / bool / sampling_rng → torch.equal


def load_policy(path: Path = DEFAULT_POLICY) -> dict[str, ToleranceClass]:
    doc = yaml.safe_load(Path(path).read_text())
    out: dict[str, ToleranceClass] = {}
    for name, cfg in doc.get("classes", {}).items():
        out[name] = ToleranceClass(
            name=name,
            atol=float(cfg["atol"]),
            rtol=float(cfg["rtol"]),
            inf_must_match=bool(cfg.get("inf_must_match", True)),
            nan_must_match=bool(cfg.get("nan_must_match", True)),
            sample_fraction=float(cfg.get("sample_fraction", 1.0)),
            bit_exact=bool(cfg.get("bit_exact", False)),
        )
    return out


@dataclass
class ComparisonResult:
    passed: bool
    reason: str | None
    max_abs_diff: float
    max_rel_diff: float
    inf_mismatch_count: int
    nan_mismatch_count: int


def _check_full(
    a: torch.Tensor, e: torch.Tensor, cls: ToleranceClass
) -> ComparisonResult:
    """Full tensor inf/nan + allclose check (no sampling)."""
    inf_mask_a = torch.isinf(a)
    inf_mask_e = torch.isinf(e)
    nan_mask_a = torch.isnan(a)
    nan_mask_e = torch.isnan(e)

    inf_mismatch = int((inf_mask_a != inf_mask_e).sum().item())
    nan_mismatch = int((nan_mask_a != nan_mask_e).sum().item())

    if cls.inf_must_match and inf_mismatch > 0:
        return ComparisonResult(
            passed=False, reason="inf positions differ",
            max_abs_diff=float("inf"), max_rel_diff=float("inf"),
            inf_mismatch_count=inf_mismatch, nan_mismatch_count=nan_mismatch,
        )
    if cls.nan_must_match and nan_mismatch > 0:
        return ComparisonResult(
            passed=False, reason="nan positions differ",
            max_abs_diff=float("inf"), max_rel_diff=float("inf"),
            inf_mismatch_count=inf_mismatch, nan_mismatch_count=nan_mismatch,
        )

    finite_mask = ~(inf_mask_a | inf_mask_e | nan_mask_a | nan_mask_e)
    a_f = a[finite_mask]
    e_f = e[finite_mask]
    diff = (a_f - e_f).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = e_f.abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0

    passed = bool(torch.allclose(a_f, e_f, atol=cls.atol, rtol=cls.rtol))
    return ComparisonResult(
        passed=passed,
        reason=None if passed else f"atol/rtol: max_abs={max_abs:.3g}, max_rel={max_rel:.3g}",
        max_abs_diff=max_abs, max_rel_diff=max_rel,
        inf_mismatch_count=inf_mismatch, nan_mismatch_count=nan_mismatch,
    )


_INTEGER_DTYPES = {
    torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
}


def _bit_exact_compare(
    a: torch.Tensor, e: torch.Tensor
) -> ComparisonResult:
    """Integer / bool / sampling_rng exact equality (codex round-2 P1)."""
    if a.dtype != e.dtype:
        return ComparisonResult(
            passed=False,
            reason=f"dtype mismatch under bit_exact: {a.dtype} vs {e.dtype}",
            max_abs_diff=float("inf"), max_rel_diff=float("inf"),
            inf_mismatch_count=-1, nan_mismatch_count=-1,
        )
    passed = bool(torch.equal(a.detach().cpu(), e.detach().cpu()))
    return ComparisonResult(
        passed=passed,
        reason=None if passed else "bit_exact equality failed",
        max_abs_diff=0.0 if passed else float("inf"),
        max_rel_diff=0.0 if passed else float("inf"),
        inf_mismatch_count=0, nan_mismatch_count=0,
    )


def compare(
    actual: torch.Tensor,
    expected: torch.Tensor,
    cls: ToleranceClass,
) -> ComparisonResult:
    """Numeric comparison.

    Per codex P1-3: sampled fast-fail must NOT be the final verdict. If a
    sampled allclose detects a mismatch we fail-fast. If the sample passes,
    we still run the **full** allclose so that NaN/Inf or large drift
    outside the sample window cannot pass.

    Per codex round-2 P1: integer / bool dtypes (or any `bit_exact: true`
    class) get strict torch.equal — they MUST NOT go through float32 cast
    + allclose, which silently rounds e.g. int64(2^24+1) to 2^24.
    """
    if actual.shape != expected.shape:
        return ComparisonResult(
            passed=False,
            reason=f"shape mismatch: {actual.shape} vs {expected.shape}",
            max_abs_diff=float("inf"),
            max_rel_diff=float("inf"),
            inf_mismatch_count=-1,
            nan_mismatch_count=-1,
        )

    is_integer_actual = actual.dtype in _INTEGER_DTYPES
    is_integer_expected = expected.dtype in _INTEGER_DTYPES
    if cls.bit_exact or is_integer_actual or is_integer_expected:
        return _bit_exact_compare(actual, expected)

    a_full = actual.detach().to(torch.float32).cpu()
    e_full = expected.detach().to(torch.float32).cpu()

    if cls.sample_fraction < 1.0 and a_full.numel() > 1024:
        n_sample = max(1024, int(a_full.numel() * cls.sample_fraction))
        idx = torch.randperm(a_full.numel())[:n_sample]
        a_sample = a_full.reshape(-1)[idx]
        e_sample = e_full.reshape(-1)[idx]
        # Sampled fast-fail: reject early if even the sample fails.
        sample_result = _check_full(a_sample, e_sample, cls)
        if not sample_result.passed:
            return sample_result
        # Sample passed → still authoritative full check.

    return _check_full(a_full, e_full, cls)
