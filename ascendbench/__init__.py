"""ascendbench — the engine of the triton-ascend-bench benchmark.

A security-hardened adversarial grading engine for GPU Triton → Ascend 910B
kernel-port submissions: it severs an untrusted solution from the oracle/golden/
host, proves the port is an honest Triton kernel, and renders an oracle-equivalence
(correctness) + analytic-Roofline (performance) verdict.

Subpackages:
    engine        — eval orchestrator + cli (authoritative scoring / release gate)
    integrity     — anti-cheat: static_checker, process_isolation, adversarial, wrapper_contract
    correctness   — oracle-equivalence: tolerance, golden_store, determinism
    perf          — Roofline: timing, cost_model, peak_table, scoring
    input_spec    — adversarial fixture synthesis: inputs, expr, param_constraints
    corpus        — task loading + provenance
    release_gate  — eligibility + sanity gating
    policies      — tolerance / ceiling / sanity / floor / trust config data

Root: manifest (SolutionManifest), triton_compat; frozen/ + upstream_shims/ severance runtime.
The task DATA lives separately in tasks/. See docs/reference/ for the protocol.
"""

__version__ = "0.1.0"
__all__ = [
    "engine",
    "integrity",
    "correctness",
    "perf",
    "input_spec",
    "corpus",
    "release_gate",
    "manifest",
    "backends",
]
