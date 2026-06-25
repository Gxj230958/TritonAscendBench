"""Unit tests for V9 plan M3.1 -- R20 determinism verification.

These tests do NOT require a GPU. They use CPU-only torch closures
to simulate deterministic vs non-deterministic kernels (a separate
``test_r20_determinism_gpu.py`` exists for the GPU wiring path).
"""
from __future__ import annotations

import pytest
import torch

from ascendbench.correctness.determinism import (
    DeterminismResult,
    to_verdict,
    verify_determinism,
)

# ─────────────────────────────────────────────────────────────────────
# verify_determinism -- deterministic and non-deterministic paths
# ─────────────────────────────────────────────────────────────────────


def test_deterministic_closure_returns_pass():
    """A constant-output closure is bit-identical across runs."""
    out_tensor = torch.tensor([0.1, 0.2, 0.3])

    def run_once():
        return out_tensor.clone()

    res = verify_determinism(run_once)
    assert res.deterministic is True
    assert res.run_count == 3
    assert len(res.unique_digests) == 1


def test_non_deterministic_closure_returns_fail():
    """A closure that mutates its output each call fails the gate."""
    counter = [0]

    def run_once():
        counter[0] += 1
        return torch.tensor([float(counter[0])])

    res = verify_determinism(run_once)
    assert res.deterministic is False
    assert res.run_count == 3
    assert len(res.unique_digests) >= 2
    assert "non-deterministic" in res.reason


def test_dict_output_round_trips():
    """A dict of tensors is canonicalised consistently."""
    payload = {
        "out": torch.tensor([1.0, 2.0]),
        "aux": torch.tensor([0.5]),
    }

    def run_once():
        return {k: v.clone() for k, v in payload.items()}

    res = verify_determinism(run_once)
    assert res.deterministic is True


def test_nested_dict_keys_order_independent():
    """Different insertion order of dict keys still hashes the same."""
    t1 = torch.tensor([1.0])
    t2 = torch.tensor([2.0])

    state = {"order": 0}

    def run_once():
        state["order"] += 1
        if state["order"] % 2 == 0:
            return {"b": t2.clone(), "a": t1.clone()}
        return {"a": t1.clone(), "b": t2.clone()}

    res = verify_determinism(run_once)
    assert res.deterministic is True


def test_n_runs_must_be_at_least_2():
    with pytest.raises(ValueError):
        verify_determinism(lambda: torch.zeros(1), n_runs=1)


def test_supports_higher_n_runs():
    """N=5 still bit-identical for a deterministic closure."""
    res = verify_determinism(lambda: torch.zeros(2), n_runs=5)
    assert res.deterministic is True
    assert res.run_count == 5


# ─────────────────────────────────────────────────────────────────────
# to_verdict -- V9 schema field mapping
# ─────────────────────────────────────────────────────────────────────


def test_verdict_deterministic_maps_to_done():
    res = DeterminismResult(
        deterministic=True,
        run_count=3,
        digests=("a" * 64, "a" * 64, "a" * 64),
        unique_digests=("a" * 64,),
    )
    v = to_verdict(res)
    assert v.gpu_kernel_capture_status == "DONE"
    assert v.gpu_kernel_deterministic is True


def test_verdict_non_deterministic_maps_to_blocked():
    res = DeterminismResult(
        deterministic=False,
        run_count=3,
        digests=("a" * 64, "b" * 64, "c" * 64),
        unique_digests=("a" * 64, "b" * 64, "c" * 64),
        reason="R20: 3 distinct digests",
    )
    v = to_verdict(res)
    assert v.gpu_kernel_capture_status == "BLOCKED_NON_DETERMINISTIC"
    assert v.gpu_kernel_deterministic is False
    assert v.reason == "R20: 3 distinct digests"
