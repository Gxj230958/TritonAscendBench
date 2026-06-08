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

import torch
import yaml

from ascendbench.policies import policy_path

DEFAULT_POLICY = policy_path("tolerance_policy.yaml")


@dataclass
class ToleranceClass:
    name: str
    atol: float
    rtol: float
    inf_must_match: bool = True
    nan_must_match: bool = True
    sample_fraction: float = 1.0     # 1.0 == compare full tensor
    bit_exact: bool = False          # integer / bool / sampling_rng → torch.equal
    # V9 codex review 2026-05-27: per-class explicit integer-output
    # tolerance. When ``None`` (default), integer dtypes go through
    # bit-exact equality -- the historical safe behaviour for
    # sampling_rng / int64 ULP-sensitive ops. When set to a positive
    # int, integer outputs are compared with ``(actual - expected).abs()
    # <= int_atol`` instead, which is what int8 quantization rounding
    # actually needs (fp32 -> int8 banker's-rounding vs Triton native
    # rounding routinely differs by 1 step in ties).
    int_atol: int | None = None
    # fp8 quant outputs: max ordered representable-code distance. A value near an
    # e4m3 bucket midpoint may round to either neighbor (1 ULP), same as
    # int8_quant_round's int_atol. None -> fp8 goes through the float allclose.
    fp8_ulp: int | None = None


def load_policy(path: Path = DEFAULT_POLICY) -> dict[str, ToleranceClass]:
    doc = yaml.safe_load(Path(path).read_text())
    out: dict[str, ToleranceClass] = {}
    for name, cfg in doc.get("classes", {}).items():
        raw_int_atol = cfg.get("int_atol")
        int_atol: int | None = None
        if raw_int_atol is not None:
            int_atol = int(raw_int_atol)
            if int_atol < 0:
                raise ValueError(
                    f"tolerance class {name!r}: int_atol must be >= 0 "
                    f"(got {int_atol})"
                )
        raw_fp8_ulp = cfg.get("fp8_ulp")
        fp8_ulp: int | None = None
        if raw_fp8_ulp is not None:
            fp8_ulp = int(raw_fp8_ulp)
            if fp8_ulp < 0:
                raise ValueError(
                    f"tolerance class {name!r}: fp8_ulp must be >= 0 "
                    f"(got {fp8_ulp})"
                )
        out[name] = ToleranceClass(
            name=name,
            atol=float(cfg["atol"]),
            rtol=float(cfg["rtol"]),
            inf_must_match=bool(cfg.get("inf_must_match", True)),
            nan_must_match=bool(cfg.get("nan_must_match", True)),
            sample_fraction=float(cfg.get("sample_fraction", 1.0)),
            bit_exact=bool(cfg.get("bit_exact", False)),
            int_atol=int_atol,
            fp8_ulp=fp8_ulp,
        )
    return out


@dataclass(frozen=True)
class ToleranceSpec:
    """Unified tolerance spec for a task -- V9 plan M0.0 authority.

    ``for_task(task)`` consults the nested
    ``provenance.golden.tolerance_source`` (when present) to pick the
    right source:

    * ``upstream_test_explicit`` -- use the atol/rtol pinned in
      ``provenance.golden.atol`` / ``provenance.golden.rtol``
    * ``upstream_test_default`` -- fall back to the corpus
      ``tolerance_class`` (the historical default; was always the only
      source pre-M0.0)
    * ``migration_drift_budget`` -- R11 v2 follow-up; v1 not
      release-ready (``is_provenance_eligible`` excludes these tasks)

    When ``provenance.golden`` is absent the spec is built purely
    from ``meta.tolerance_class`` -- same behaviour as the previous
    direct ``load_policy()[task.tolerance_class]`` callers, so this
    is backward-compatible during the schema rollout.
    """

    tolerance_class_name: str
    cls: ToleranceClass
    # one of: upstream_test_explicit / upstream_test_default / migration_drift_budget
    source: str
    atol: float
    rtol: float
    # True iff source == upstream_test_explicit AND values were pinned
    pinned_from_upstream: bool


def for_task(
    task, *, policy: dict[str, ToleranceClass] | None = None
) -> ToleranceSpec:
    """Build the unified :class:`ToleranceSpec` for a task.

    ``task`` is duck-typed: needs ``.tolerance_class`` (str) and
    ``.meta`` (dict). Both ``ascendbench.corpus.dataset.TaskDef`` and
    raw-meta callers fit.

    Precedence (V9 plan M0.0):

    1. ``meta.provenance.golden.tolerance_source == upstream_test_explicit``
       AND ``atol`` / ``rtol`` present -> pinned from the upstream test;
       ``pinned_from_upstream = True``.
    2. Otherwise the task's declared ``tolerance_class`` is looked up
       in ``tolerance_policy.yaml`` and used as the default. The spec's
       ``source`` is reported faithfully (``upstream_test_default``
       when the schema declares it, also when the nested block is
       absent -- the pre-V9 schema always carried the default
       implicitly).
    3. ``migration_drift_budget`` is recognised as a source but its
       v1 implementation is "not release-ready"; the spec is still
       returned (callers gate on ``is_provenance_eligible`` separately).
    """
    if policy is None:
        policy = load_policy()
    cls_name = getattr(task, "tolerance_class", None) or (
        getattr(task, "meta", {}) or {}
    ).get("tolerance_class")
    if not cls_name or cls_name not in policy:
        raise KeyError(
            f"task {getattr(task, 'task_id', '?')!r} has no usable "
            f"tolerance_class (got {cls_name!r})"
        )
    base_cls = policy[cls_name]
    meta = getattr(task, "meta", {}) or {}
    golden = (meta.get("provenance") or {}).get("golden") or {}
    source = golden.get("tolerance_source")

    if source == "upstream_test_explicit":
        atol_pinned = golden.get("atol")
        rtol_pinned = golden.get("rtol")
        if atol_pinned is not None and rtol_pinned is not None:
            return ToleranceSpec(
                tolerance_class_name=cls_name,
                cls=ToleranceClass(
                    name=cls_name,
                    atol=float(atol_pinned),
                    rtol=float(rtol_pinned),
                    inf_must_match=base_cls.inf_must_match,
                    nan_must_match=base_cls.nan_must_match,
                    sample_fraction=base_cls.sample_fraction,
                    bit_exact=base_cls.bit_exact,
                    # Codex review 2026-05-27: clone *all* class fields so
                    # tasks that pin atol/rtol from an upstream test keep
                    # their integer-output ULP tolerance. Without this an
                    # explicitly pinned int8/int32 task would silently
                    # fall back to bit-exact even when the policy class
                    # declares int_atol.
                    int_atol=base_cls.int_atol,
                    fp8_ulp=base_cls.fp8_ulp,
                ),
                source="upstream_test_explicit",
                atol=float(atol_pinned),
                rtol=float(rtol_pinned),
                pinned_from_upstream=True,
            )
        # explicit declared but no values -> fall through to default;
        # report the actual effective source (default), not the
        # claimed-but-unfulfilled "explicit". The schema validator
        # accepts the partial declaration during pipeline progress;
        # the runtime reports what it actually used.
        source = None

    if source == "migration_drift_budget":
        # v1: not release-ready; spec still returned for visibility but
        # callers should gate on ``is_provenance_eligible``.
        return ToleranceSpec(
            tolerance_class_name=cls_name,
            cls=base_cls,
            source="migration_drift_budget",
            atol=base_cls.atol,
            rtol=base_cls.rtol,
            pinned_from_upstream=False,
        )

    return ToleranceSpec(
        tolerance_class_name=cls_name,
        cls=base_cls,
        source="upstream_test_default" if source is None else source,
        atol=base_cls.atol,
        rtol=base_cls.rtol,
        pinned_from_upstream=False,
    )


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


def _int_atol_compare(
    a: torch.Tensor, e: torch.Tensor, atol: int,
) -> ComparisonResult:
    """V9 codex review 2026-05-27: integer-typed compare with explicit
    ULP tolerance. ``|a - e| <= atol`` element-wise, with diff
    computed in widened int64 so the subtraction itself does not
    overflow the input dtype.
    """
    if a.dtype != e.dtype:
        return ComparisonResult(
            passed=False,
            reason=f"dtype mismatch under int_atol: {a.dtype} vs {e.dtype}",
            max_abs_diff=float("inf"), max_rel_diff=float("inf"),
            inf_mismatch_count=-1, nan_mismatch_count=-1,
        )
    a_i = a.detach().cpu().to(torch.int64)
    e_i = e.detach().cpu().to(torch.int64)
    diff = (a_i - e_i).abs()
    max_abs = int(diff.max().item()) if diff.numel() else 0
    passed = bool((diff <= atol).all().item())
    return ComparisonResult(
        passed=passed,
        reason=(
            None if passed
            else f"int_atol fail: max_abs={max_abs} > int_atol={atol}"
        ),
        max_abs_diff=float(max_abs),
        # rel_diff is meaningless for tiny-integer outputs (would
        # explode near zero); report 0 on pass / inf on fail.
        max_rel_diff=0.0 if passed else float("inf"),
        inf_mismatch_count=0, nan_mismatch_count=0,
    )


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


_FP8_DTYPES = {torch.float8_e4m3fn}


def _fp8_ulp_compare(
    a: torch.Tensor, e: torch.Tensor, ulp: int, max_frac: float = 0.01,
) -> ComparisonResult:
    """fp8 (e4m3) quant-output compare on the ordered representable-code.

    A value near an fp8 bucket midpoint may round to either neighbor when a tiny
    upstream precision differs (libdevice vs torch silu/exp), so the two fp8
    codes differ by 1 step -- the same 1-step quant ambiguity as
    int8_quant_round. Compares the ordered sign-magnitude e4m3 code (adjacent
    representable values are distance 1), canonicalising +/-0 and requiring NaN
    positions to match.
    """
    if a.dtype != e.dtype:
        return ComparisonResult(
            passed=False,
            reason=f"dtype mismatch under fp8_ulp: {a.dtype} vs {e.dtype}",
            max_abs_diff=float("inf"), max_rel_diff=float("inf"),
            inf_mismatch_count=-1, nan_mismatch_count=-1,
        )
    ba = a.detach().cpu().view(torch.uint8).to(torch.int16)
    be = e.detach().cpu().view(torch.uint8).to(torch.int16)
    nan_a, nan_e = (ba & 0x7F) == 0x7F, (be & 0x7F) == 0x7F
    nan_mismatch = int((nan_a != nan_e).sum().item())
    if nan_mismatch > 0:
        return ComparisonResult(
            passed=False, reason="fp8 nan positions differ",
            max_abs_diff=float("inf"), max_rel_diff=float("inf"),
            inf_mismatch_count=0, nan_mismatch_count=nan_mismatch,
        )

    def _code(b: torch.Tensor) -> torch.Tensor:
        mag = b & 0x7F
        c = torch.where(((b >> 7) & 1).bool(), -mag, mag)
        return torch.where(mag == 0, torch.zeros_like(c), c)  # canon +/-0

    fin = ~nan_a
    dist = (_code(ba)[fin] - _code(be)[fin]).abs()
    n = dist.numel()
    max_dist = int(dist.max().item()) if n else 0
    # Off-by-1 should be a rare boundary flip (a value within the silu/exp epsilon
    # of an fp8 midpoint); a LARGE off-by-1 fraction is a systematic rounding-mode
    # bias (a real bug), not quant ambiguity -- cap it.
    off_frac = float((dist > 0).float().mean().item()) if n else 0.0
    passed = (max_dist <= ulp) and (off_frac <= max_frac)
    if max_dist > ulp:
        reason = f"fp8_ulp fail: max ordered-code dist={max_dist} > {ulp}"
    elif off_frac > max_frac:
        reason = (f"fp8_ulp fail: {off_frac:.2%} of elements off by 1 "
                  f"> {max_frac:.2%} (systematic rounding-mode bias, not boundary)")
    else:
        reason = None
    return ComparisonResult(
        passed=passed,
        reason=reason,
        max_abs_diff=float(max_dist),
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
    if is_integer_actual or is_integer_expected:
        # Integer outputs: ``int_atol`` set on the class -> ULP-tolerant
        # compare (the quantization-rounding path); otherwise the
        # historical bit-exact path (sampling_rng / int64 ULP-sensitive
        # ops). ``cls.bit_exact=True`` always forces bit-exact even if
        # int_atol is set (explicit user intent wins).
        if cls.bit_exact or cls.int_atol is None:
            return _bit_exact_compare(actual, expected)
        return _int_atol_compare(actual, expected, cls.int_atol)
    # fp8 quant outputs: ordered-code ULP (boundary rounding ambiguity). Only
    # when BOTH are fp8 and the class opts in; scales/aux stay on float allclose.
    if cls.fp8_ulp is not None:
        a_fp8 = actual.dtype in _FP8_DTYPES
        e_fp8 = expected.dtype in _FP8_DTYPES
        if a_fp8 and e_fp8:
            return _fp8_ulp_compare(actual, expected, cls.fp8_ulp)
        if a_fp8 != e_fp8:
            # One side fp8, the other not: an fp8 graded output must be compared
            # as fp8, never silently fp32-cast to a pass.
            return ComparisonResult(
                passed=False,
                reason=(f"fp8 dtype mismatch under fp8_ulp: "
                        f"{actual.dtype} vs {expected.dtype}"),
                max_abs_diff=float("inf"), max_rel_diff=float("inf"),
                inf_mismatch_count=-1, nan_mismatch_count=-1,
            )
    if cls.bit_exact:
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
