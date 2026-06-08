"""ascendbench.frozen -- version-pinned upstream support materials.

shim/      inert default-deny symbol stubs the severed kernels import in
           place of live vllm/sglang (S2 severance).
deps/      content-addressed vendored dependency closures (h_<sha>/...).
snapshots/ verbatim upstream kernel snapshots (attribution / freeze
           reproducibility) -- data, NOT imported.

These live inside the installed package so a pip-installed eval can run the
S2 severance check; they were previously under tasks/_frozen_*/ which the
built wheel did not ship.
"""
