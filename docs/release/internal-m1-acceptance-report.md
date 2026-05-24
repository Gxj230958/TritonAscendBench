# M1 Acceptance Report -- Tier 1 fix_orchestrator + shapes-incomplete pipeline

Status: **placeholder pending GPU host session**.

## Scope

M1 delivers the foundation for the LLM-driven fix orchestrator:

* `scripts/automation/lib/atomic.py` -- POSIX-atomic write helpers,
  baseline snapshot, `rsync --delete` rollback.
* `scripts/automation/lib/queue.py` -- directory-rename queue with the
  5-case mv state matrix, reaper, heartbeat, PID-reuse defense.
* `scripts/automation/lib/lock.py` -- mkdir-based atomic lock,
  three-stage steal, heartbeat eviction audit.
* `scripts/automation/lib/ssh_tunnel.py` -- local-port-forward SSH
  tunnel with health-probe supervisor.
* `scripts/automation/llm_client.py` -- OpenAI-compatible thin client
  with retry + budget tracking.
* `scripts/automation/lib/file_blocks.py` -- META + FILE protocol parser
  with the 5-layer validation pipeline (path whitelist, import leak,
  stub signature, META/FILE consistency, content-level semantic gate).
* `scripts/automation/extract_verdict.py` -- strict 9-field result.json
  writer + `_provenance.json` sidecar.
* `scripts/automation/tier1_reverify.py` -- GPU-side wrapper around
  resolver / phase_a_prime `--verify`.
* `scripts/automation/fix_orchestrator.py` + `state_machine.py` --
  canonical 17-state machine; full shapes-incomplete pipeline through
  validate -> snapshot -> apply -> pending reverify -> consume -> DONE.
* `scripts/automation/prompts/{shapes_incomplete,shared}.py` -- LLM
  prompt template for the shapes-incomplete fix.

## Mock end-to-end (CI-runnable)

`tests/unit_tests/test_e2e_shapes_incomplete.py` exercises five mock
scenarios against the full pipeline:

* Happy new -> done (verdict=PASS)
* First invalid LLM response -> retry -> PASS
* Blocked same-class retries -> UNFIXABLE_BY_LLM
* Verdict_class switch (shapes -> oracle-mismatch) -> re-enter NEW
* State file persists across orchestrator restarts (round-trip).

All five PASS without any GPU machine.

## Real end-to-end (DEFERRED to GPU session)

Per plan Unit 1.11 the real acceptance gate is:

1. Pick three real shapes-incomplete tasks (L1 / L2 / L3 each one).
2. Run `python3 scripts/automation/fix_orchestrator.py ... --apply` on
   the dev machine; capture state file + LLM prompt/response artifacts.
3. Copy/paste the three `python3 scripts/automation/tier1_reverify.py`
   commands from orchestrator stdout into the GPU host shell.
4. Wait for `result.json + _provenance.json + DONE` to land per task.
5. Re-run orchestrator on dev; assert at least one task transitions to
   `state=DONE, final_verdict=PASS`.

This is unblocked by Unit 1.11 -- the dev-side infrastructure is in
place. The acceptance run itself requires a GPU host session that has
not yet been scheduled.

When the GPU run completes, fill in:

* Tasks attempted (IDs + L-level):
* Tasks PASSED (count + IDs):
* Tasks BLOCKED same class (count):
* Tasks BLOCKED with class-switch (count):
* USD spent on LLM calls (median + p95 per task):
* Wall-clock per task (median + p95):
