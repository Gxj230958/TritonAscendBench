# `tasks/_corrected/` — corrected-variant proof artifacts (NON-SCORED)

These are NOT part of the 100-task benchmark. The task discovery
(`phase_a_prime._iter_task_dirs`) and `validate_tasks.py` only walk
`tasks/level{1,2,3}/` and explicitly skip `_`-prefixed dirs, so nothing here is
score-eligible or release-eligible.

Each variant applies the MINIMAL upstream fix to a frozen kernel whose original
task is honestly BLOCKED by a real upstream bug, and demonstrates that the fix
verifies faithfully. The ORIGINAL task stays BLOCKED as the record of the bug;
the headline 100-task score is unchanged (97/100).

Manifest fields (meta.yaml `corrected_variant:`):
- `variant_of`            — original task id (stays BLOCKED)
- `original_kernel_sha256`/`corrected_kernel_sha256`
- `patch_summary`         — the exact one-line diff
- `discriminating_evidence` — where the fix is PROVEN needed (the daemon
  tolerance absorbs the bug, so a daemon PASS is confirmatory, NOT discriminating)
- `score_eligible: false`, `release_scope: {excluded: true}`
