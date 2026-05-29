"""Phase A' standalone shim for ``vllm.envs``.

When the rewriter severs a frozen kernel.py's ``import vllm.envs as envs``,
references like ``envs.VLLM_BATCH_INVARIANT`` must still resolve so the
standalone module imports without re-importing vllm. This module exposes
ONLY the verbatim upstream default values of the env flags the frozen
corpus actually reads, and LOUD-FAILS (AttributeError) on any other
attribute.

Why an allowlist + loud-fail (not a full ``os.getenv`` replication of
vllm.envs): most env flags are real semantic knobs that select a compute
path (tuned-config selection, workspace sizing, weight padding) and are
classified Class-S ("pin resolved_value per-task, NOT shim"). Silently
defaulting such a flag could mask a wrong-but-successful comparison. So
this shim exposes ONLY flags PROVEN inert for the verified entrypoint
(read only in module-scope setup, never on the compute path); every other
flag -- including the Class-S semantic knobs -- LOUD-FAILS. The loud-fail
IS the Class-S gate: a task whose standalone reads a semantic flag fails
at import (or at compute) and stays honestly BLOCKED, rather than being
silently shifted onto a default path. Such a flag must instead go through
the per-task Class-S resolved_value pinning, never this shim.

Verbatim default below mirrors the lambda table in
``infer/vllm/vllm/envs.py`` (the ``os.getenv(name, default)`` literal):
  - VLLM_BATCH_INVARIANT  getenv("VLLM_BATCH_INVARIANT", "0") -> bool(int) = False

Inertness (verified before admission):
  - VLLM_BATCH_INVARIANT gates the module-scope ``init_batch_invariance()``
    / ``enable_batch_invariant_mode()`` setup (monkeypatching torch.bmm,
    sizing tiles). The verified wrappers (e.g. bmm_batch_invariant,
    matmul_persistent) compute from their args + module-default globals
    and never read this flag, and the runner never auto-runs setup, so the
    default is inert for the verified path. (VLLM_TUNED_CONFIG_FOLDER /
    VLLM_ROCM_FP8_PADDING are deliberately NOT exposed -- they are
    compute-path semantic knobs (Class-S); they loud-fail here.)
"""
from __future__ import annotations

VLLM_BATCH_INVARIANT: bool = False

_ALLOWED = frozenset({"VLLM_BATCH_INVARIANT"})


def __getattr__(name: str):
    """PEP 562 module hook: fires only for names NOT defined above.
    Loud-fail so an un-vetted env flag is never silently defaulted."""
    raise AttributeError(
        f"vllm.envs shim: flag {name!r} is not in the verified-inert "
        f"allowlist {sorted(_ALLOWED)}. A standalone kernel reading an "
        "un-vetted env flag must be classified per-task (it may select a "
        "compute path); add it here only with proof it is inert for the "
        "verified entrypoint."
    )
