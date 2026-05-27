"""sanity_gate.py -- V9 plan M3.0 (R16 output sanity gate).

Reads the central ``sanity_bounds.yaml`` table and applies the
per-category bounds to a captured golden tensor (or any candidate
output tensor). Used by ``phase_b_golden_capture`` BEFORE it signs a
manifest -- gross-corruption defence for the R16 trust chain.

This module is intentionally tiny and side-effect-free: load policy
on call, no import-time I/O. The full R5/R18 freeze + sign path is
in M3.2; the sanity gate ships independently so callers can wire it
through ``GoldenStore.put`` immediately.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BOUNDS = (
    REPO_ROOT / "triton_ascend_bench" / "sanity_bounds.yaml"
)


@dataclass(frozen=True)
class SanityBounds:
    """Bounds for one category. Loaded from ``sanity_bounds.yaml``."""

    category: str
    max_abs: float
    forbid_all_zero: bool
    allow_nan_rate: float
    allow_inf_rate: float
    must_match_input_rank: bool


@dataclass(frozen=True)
class SanityResult:
    """Per-tensor sanity-gate verdict."""

    passed: bool
    reason: str = ""
    # Diagnostic numerics (filled even when passed=True so callers can
    # log the actual bounds-touching values).
    max_abs_value: float = 0.0
    nan_rate: float = 0.0
    inf_rate: float = 0.0
    output_rank: int = 0
    input_rank: int | None = None


def load_bounds(path: Path = DEFAULT_BOUNDS) -> dict[str, SanityBounds]:
    """Load the central sanity-bounds YAML.

    Returns ``{category_name: SanityBounds}``. Raises ``KeyError`` /
    ``yaml.YAMLError`` on a malformed document -- the policy must be
    parse-clean for the gate to run (no silent skip).
    """
    doc = yaml.safe_load(Path(path).read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{path} root must be a YAML mapping")
    by_cat = doc.get("by_category")
    if not isinstance(by_cat, dict):
        raise KeyError(f"{path} missing `by_category` mapping")
    out: dict[str, SanityBounds] = {}
    for cat, cfg in by_cat.items():
        if not isinstance(cfg, dict):
            raise ValueError(
                f"{path} by_category.{cat} must be a mapping"
            )
        out[cat] = SanityBounds(
            category=cat,
            max_abs=float(cfg["max_abs"]),
            forbid_all_zero=bool(cfg.get("forbid_all_zero", False)),
            allow_nan_rate=float(cfg.get("allow_nan_rate", 0.0)),
            allow_inf_rate=float(cfg.get("allow_inf_rate", 0.0)),
            must_match_input_rank=bool(
                cfg.get("must_match_input_rank", False)
            ),
        )
    return out


def apply_bounds(
    output_tensor: Any,
    *,
    category: str,
    bounds: dict[str, SanityBounds] | None = None,
    input_rank: int | None = None,
) -> SanityResult:
    """Apply R16 sanity bounds to a single output tensor.

    ``output_tensor`` is a torch.Tensor (or any object with ``.shape``
    + numeric ``.abs()`` / ``isnan`` / ``isinf`` methods). The category
    look-up is fail-closed: an unknown category returns
    ``passed=False`` (forces the reviewer to register it).
    """
    if bounds is None:
        bounds = load_bounds()

    if category not in bounds:
        return SanityResult(
            passed=False,
            reason=(
                f"R16: no sanity bounds entry for category={category!r}; "
                "fail-closed (add entry to triton_ascend_bench/sanity_bounds.yaml)"
            ),
        )
    b = bounds[category]

    # Compute diagnostics first so we can populate them on either path.
    import torch  # local import keeps the module torch-optional at top

    if not isinstance(output_tensor, torch.Tensor):
        return SanityResult(
            passed=False,
            reason=(
                f"R16: output is not a torch.Tensor (got "
                f"{type(output_tensor).__name__})"
            ),
        )
    if output_tensor.numel() == 0:
        # Empty tensor: meaningless to check magnitude; pass-through
        # (the upstream gate that produced an empty tensor either
        # legitimately did so or already raised separately).
        return SanityResult(
            passed=True,
            output_rank=output_tensor.ndim,
            input_rank=input_rank,
        )

    finite_mask = torch.isfinite(output_tensor)
    n_total = output_tensor.numel()
    n_nan = int(torch.isnan(output_tensor).sum().item())
    n_inf = int(torch.isinf(output_tensor).sum().item())
    nan_rate = n_nan / n_total
    inf_rate = n_inf / n_total

    if finite_mask.any():
        max_abs = float(
            output_tensor[finite_mask].abs().max().item()
        )
    else:
        max_abs = float("inf")

    res = SanityResult(
        passed=True,
        max_abs_value=max_abs,
        nan_rate=nan_rate,
        inf_rate=inf_rate,
        output_rank=output_tensor.ndim,
        input_rank=input_rank,
    )

    # Apply each predicate, fail-fast on the first failure (so the
    # reason names the actual cause; we still report diagnostics).
    if nan_rate > b.allow_nan_rate:
        return SanityResult(
            **{**res.__dict__, "passed": False, "reason": (
                f"R16: NaN rate {nan_rate:.4f} > allow_nan_rate "
                f"{b.allow_nan_rate:.4f} for category {category}"
            )}
        )

    if inf_rate > b.allow_inf_rate:
        return SanityResult(
            **{**res.__dict__, "passed": False, "reason": (
                f"R16: Inf rate {inf_rate:.4f} > allow_inf_rate "
                f"{b.allow_inf_rate:.4f} for category {category}"
            )}
        )

    if b.forbid_all_zero and max_abs == 0.0:
        return SanityResult(
            **{**res.__dict__, "passed": False, "reason": (
                f"R16: output all-zero (forbidden for category {category})"
            )}
        )

    if max_abs > b.max_abs:
        return SanityResult(
            **{**res.__dict__, "passed": False, "reason": (
                f"R16: max_abs {max_abs:.3e} > bound {b.max_abs:.3e} "
                f"for category {category}"
            )}
        )

    if b.must_match_input_rank and input_rank is not None:
        if output_tensor.ndim != input_rank:
            return SanityResult(
                **{**res.__dict__, "passed": False, "reason": (
                    f"R16: output rank {output_tensor.ndim} != input "
                    f"rank {input_rank} for category {category}"
                )}
            )

    return res


__all__ = [
    "SanityBounds",
    "SanityResult",
    "apply_bounds",
    "load_bounds",
]
