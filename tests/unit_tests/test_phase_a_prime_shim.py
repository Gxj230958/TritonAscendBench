"""U3: narrow Class-B shim (default-deny) tests.

Covers: no-op logger is inert + deterministic, B-rewrite-public
(tl/triton) REJECTED from shim, Class-A REJECTED, no-proof REJECTED,
shim has zero vllm/sglang imports.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location(
    "phase_a_prime", REPO / "scripts" / "phase_a_prime.py"
)
pap = importlib.util.module_from_spec(_spec)
sys.modules["phase_a_prime"] = pap
_spec.loader.exec_module(pap)


def _sc(cls, proof="p"):
    return pap.SymbolClass(
        symbol="vllm.logger.init_logger", module="vllm.logger",
        name="init_logger", cls=cls, proof=proof,
    )


def test_inert_b_with_proof_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(pap, "FROZEN_SHIM", tmp_path / "_frozen_shim")
    pap.register_shim_symbol(_sc(pap.CLASS_B_INERT_SHIM))
    assert (tmp_path / "_frozen_shim" / "__init__.py").exists()
    assert (tmp_path / "_frozen_shim" / "_logging.py").exists()


def test_b_rewrite_public_rejected_from_shim():
    """tl/triton re-exports must be rewritten to public import, never
    shimmed (plan U3 negative case)."""
    with pytest.raises(ValueError, match="rewritten to public"):
        pap.register_shim_symbol(_sc(pap.CLASS_B_REWRITE_PUBLIC))


def test_class_a_rejected_from_shim():
    with pytest.raises(ValueError, match=r"not.*B-inert-shim"):
        pap.register_shim_symbol(_sc(pap.CLASS_A))


def test_no_proof_rejected_from_shim():
    """B is default-deny: a symbol without a recorded proof is
    refused (plan U3: every shimmed symbol needs a proof)."""
    with pytest.raises(ValueError, match="no proof"):
        pap.register_shim_symbol(_sc(pap.CLASS_B_INERT_SHIM, proof=""))


def test_noop_logger_is_inert_and_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(pap, "FROZEN_SHIM", tmp_path / "_frozen_shim")
    pap.register_shim_symbol(_sc(pap.CLASS_B_INERT_SHIM))
    log_src = (tmp_path / "_frozen_shim" / "_logging.py").read_text()
    # zero vllm/sglang import in the shim
    assert "vllm" not in log_src and "sglang" not in log_src
    spec = importlib.util.spec_from_file_location(
        "_shimlog", tmp_path / "_frozen_shim" / "_logging.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lg = mod.init_logger("x")
    # Every call is a silent no-op returning None (deterministic).
    assert lg.info("a", 1, k=2) is None
    assert lg.warning("b") is None
    assert lg.anything_at_all("z") is None
    assert lg.isEnabledFor(10) is False
