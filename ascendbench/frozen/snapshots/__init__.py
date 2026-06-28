"""ascendbench.frozen.snapshots -- verbatim upstream kernel snapshots (data).

These ``*.py`` files are raw upstream source kept for attribution and freeze
reproducibility (the per-task ``vendored_snapshot`` provenance). They still
carry their original ``import vllm`` / ``import sglang`` lines, so they are
DATA and must never be imported into the vllm/sglang-free runtime. This guard
fails any ``ascendbench.frozen.snapshots.*`` import closed: it runs when the
package is first imported, before any submodule body executes. Provenance
tooling reads these files as text by path, which this guard does not affect.
"""

raise ImportError(
    "ascendbench.frozen.snapshots are raw upstream provenance archives "
    "(data, not importable modules); read them as text by path instead"
)
