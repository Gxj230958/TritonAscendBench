"""G7 — NPU software-stack provenance read off the bundled version matrix.

``version_matrix.toml`` is the single declared source of the software stack a
910B verdict is produced against. ``scripts/provenance/verify_version_matrix.py``
is the VERIFIER (asserts the [npu] cann / torch_npu / triton_ascend keys are
present); this module is the eval-side READER that emits the per-verdict
``software_hash`` in the ``scoring_spec.md`` shape::

    {"cann": "8.5.0", "torch_npu": "2.8.0.post1", "triton_ascend": "3.2.0"}

so every NPU verdict records the exact toolchain it was graded against. The
emitted value is whatever the matrix declares; a real 910B run must keep the
matrix in sync with the live host (the verifier guards that out of band).
"""
from __future__ import annotations

from functools import lru_cache

import tomllib

from ascendbench.policies import policy_path

# The keys that constitute the NPU software_hash, in the scoring_spec order.
_SOFTWARE_HASH_KEYS = ("cann", "torch_npu", "triton_ascend")


@lru_cache(maxsize=1)
def load_version_matrix() -> dict:
    """Parse the bundled ``version_matrix.toml`` (cached)."""
    with open(policy_path("version_matrix.toml"), "rb") as fh:
        return tomllib.load(fh)


def npu_software_hash() -> dict[str, str]:
    """Return the NPU ``software_hash`` ({cann, torch_npu, triton_ascend}) read
    from ``version_matrix.toml`` [npu]. Missing keys map to ``""`` rather than
    raising — emission is provenance, not a gate; ``verify_version_matrix.py``
    is the fail-closed completeness check run separately."""
    npu = load_version_matrix().get("npu", {})
    return {k: str(npu.get(k, "")) for k in _SOFTWARE_HASH_KEYS}


def software_hash_complete(sw: dict[str, str] | None = None) -> bool:
    """True iff every software_hash key is a non-empty string. Lets a caller
    surface an incomplete provenance stamp without re-parsing the matrix."""
    sw = sw if sw is not None else npu_software_hash()
    return all(bool(sw.get(k)) for k in _SOFTWARE_HASH_KEYS)
