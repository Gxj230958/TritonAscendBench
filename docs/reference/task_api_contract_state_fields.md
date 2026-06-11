# Task State Fields — bench-written meta.yaml state machine

> 中文版：[../zh/reference/task_api_contract_state_fields.md](../zh/reference/task_api_contract_state_fields.md)

> Full schema + writer/state-machine narrative for the bench-written
> `meta.yaml` state fields. The task contract (`task_api_contract.md` §6a)
> carries the thin read-only field reference; this document holds the
> maintainer-side writer detail.

## 6a. Task State Fields (maintainer-side writes)

The following `meta.yaml` fields are **optional, maintainer-side state**
written by the bench automation pipeline and the phase
runners (`scripts/verify/standalone_freeze_verifier.py`, `scripts/provenance/capture_golden_baselines.py`).
They drive the three-verdict pipeline (`shapes-incomplete` / `oracle-mismatch`
/ unfixable) and the release-scope eligibility gate. All four field groups
below are **optional with explicit defaults**: a task `meta.yaml` that
predates this section (the 100-task corpus authored earlier) loads and
validates exactly as before — the defaults are *implicit absent*, never an
error. The fields are written by automation, never hand-authored.

The fields are **not** added to `REQUIRED_META_TOP_LEVEL` (precedent: the
existing `phase_a_prime` block has always been optional). Schema-shape
checks for these fields, when they are present, live in the patch validator
(a bench-internal write-guard layer) — that layer rejects a
malformed write before it lands on disk. `scripts/corpus/validate_tasks.py`
ignores absent / well-formed blocks.

### 6a.1 `phase_a_prime.skip_invocation`

| key | value |
|-----|-------|
| **type** | `bool` |
| **required** | optional |
| **default** | `false` (treated as absent) |
| **written by** | the bench automation on a `defer-to-golden` decision under the `oracle-mismatch` verdict |
| **read by** | `scripts/verify/standalone_freeze_verifier.py` `s2_verify_task` early-return; eligibility gate (`ascendbench/release_gate/eligibility.py`) |

**Semantics.** When `true`, the phase A' GPU S2 verify step **does not**
invoke the operator in an isolated subprocess. Instead it records
`phase_a_prime.status = SKIPPED_BY_DESIGN` and treats the offline golden
capture (phase B, see §6a.3) as the sole truth source. This handles the
`upstream-not-invocable-here` case — an operator whose upstream entrypoint
cannot be called standalone on the GPU verify host (e.g. requires a
distributed launcher, a custom CUDA build, or a private dependency).

`skip_invocation: true` **must** be accompanied by
`phase_a_prime.status = SKIPPED_BY_DESIGN` and a non-empty
`phase_a_prime.reason`. A task with `skip_invocation: true` AND
`status: PASS` is a schema violation (cross-field invariant, §6a.6).

### 6a.2 `phase_a_prime.status` — `SKIPPED_BY_DESIGN` enum extension

| existing values | `PASS` / `BLOCKED` / `UNRUN` |
|-----------------|------------------------------|
| **new value** | `SKIPPED_BY_DESIGN` |
| **written by** | `phase_a_prime.update_meta_phase_a_prime` (existing writer) |
| **read by** | eligibility gate; `scripts/corpus/validate_tasks.py` golden-presence check |

**Semantics.** `SKIPPED_BY_DESIGN` is the truth-source delegation marker:
phase A' deliberately did not run isolated invocation; the offline phase B
golden (§6a.3) is authoritative. The eligibility gate treats this status
the same as `PASS` *only if* the corresponding `phase_b_capture.status`
records a successful capture (`DONE`).

When `phase_a_prime.status = SKIPPED_BY_DESIGN`, `phase_a_prime.reason`
(free-form string) is **required** and must explain why the operator
cannot be invoked standalone.

### 6a.3 `phase_b_capture` block

The audit-trail record of the offline phase B golden capture for a single
task.

| key | type | required | default | semantics |
|-----|------|----------|---------|-----------|
| `phase_b_capture.status` | string enum `PENDING` / `DONE` / `FAILED` | optional | absent (= no capture attempted) | capture state machine |
| `phase_b_capture.upstream_callable.module` | string, canonical dotted Python path | optional | absent | which upstream module exports the entrypoint to capture |
| `phase_b_capture.upstream_callable.function` | string, valid Python identifier | optional | absent | which function under that module to call |
| `phase_b_capture.reason` | string | optional; **required when** `status` is `FAILED` | absent | free-form failure / skip explanation |
| `phase_b_capture.captured_at` | ISO 8601 UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`) | optional; **required when** `status` is `DONE` | absent | when the capture succeeded |

**Writers.**

- The bench automation writes the
  initial `{status: PENDING, upstream_callable: {module, function},
  reason: …}` on a `defer-to-golden` patch. `upstream_callable.{module,
  function}` are sourced from the dual-anchor resolver evidence.
- `scripts/provenance/capture_golden_baselines.py` updates `status` to `DONE` (sets
  `captured_at`) on a successful capture child or `FAILED` (sets `reason`)
  on a capture failure. The `upstream_callable` subblock is not rewritten
  by the capture script — it is the orchestrator's authored intent.

**Readers.** Eligibility gate; `scripts/corpus/validate_tasks.py` golden-
directory presence check (a `DONE` capture implies a populated `golden/`
directory).

**`upstream_callable.module` format.** A canonical dotted Python module
path: a non-empty sequence of valid Python identifiers separated by
single dots, with no leading / trailing dot and no embedded whitespace
or slashes. Examples — legal: `vllm.model_executor.layers.layernorm`,
`sgl_kernel.activation`; illegal: `vllm/layernorm`, `vllm..layernorm`,
`3vllm.layernorm`, `vllm. layernorm`. Canonical-path enforcement is
performed by the Layer 5 patch validator
(a bench-internal write-guard).

**`upstream_callable.function` format.** A single valid Python
identifier (no dots, no whitespace, must satisfy `str.isidentifier()`).
A multi-segment dotted name belongs in `module`, not `function`.

### 6a.4 `phase_b_capture_required`

| key | value |
|-----|-------|
| **type** | `bool` |
| **required** | optional |
| **default** | `false` (treated as absent) |
| **written by** | orchestrator on a `defer-to-golden` patch under either the `oracle-mismatch` or `upstream-not-invocable-here` decision |
| **read by** | eligibility gate (`ascendbench/release_gate/eligibility.py::release_status_pass`) |

**Semantics.** When `true`, the task's release eligibility requires
`phase_b_capture.status == DONE` (i.e. the golden has been captured and
the operator's authoritative oracle is on disk). When `false` (default),
the eligibility predicate does not consult `phase_b_capture` at all.

This boolean is the explicit promotion of the offline golden into the
release-gate conjunctive predicate — without it, an
`oracle-mismatch` task could spuriously be `release_status = PASS` based
solely on the phase A' result while its golden is still missing.

### 6a.5 `release_scope` block

The explicit exclusion record for a task that the orchestrator has marked
unfixable by the LLM patch flow and removed from the release scope (the
`--mark-unfixable` flow).

| key | type | required | default | semantics |
|-----|------|----------|---------|-----------|
| `release_scope.excluded` | `bool` | optional | absent (= `false`) | sentinel — task is excluded from release scoring |
| `release_scope.reason` | string | optional; **required when** `excluded` is `true` | absent | free-form human-readable exclusion reason |
| `release_scope.excluded_at` | ISO 8601 UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`) | optional; **required when** `excluded` is `true` | absent | when the exclusion was recorded |
| `release_scope.excluded_by` | string identifying the actor (e.g. `fix_orchestrator/automation`, operator handle) | optional; **required when** `excluded` is `true` | absent | who recorded the exclusion |

The shipped writer constructs the four-key block atomically; partial
writes (e.g. `excluded: true` with no `reason`) are not produced by the
helper and are rejected by the Layer 5 patch validator. A task whose
`release_scope.excluded` is `true` is dropped from the eligibility
denominator by `is_score_eligible` (early-out) so it does not weigh on
the release pass-rate.

### 6a.6 Cross-field invariants

These invariants tie the four field groups together. They are enforced
at *write* time by the patch validator (Layer 5) and at *gate* time by
the eligibility predicate. The validator (`scripts/corpus/validate_tasks.py`)
does not re-derive them — a malformed cross-field combination cannot
land in `meta.yaml` because the patch is rejected at Layer 5 before
disk write.

1. `phase_a_prime.skip_invocation == true` **⇒**
   `phase_a_prime.status == "SKIPPED_BY_DESIGN"` **and**
   `phase_a_prime.reason` is a non-empty string.
2. `phase_a_prime.status == "SKIPPED_BY_DESIGN"` **⇒**
   `phase_a_prime.skip_invocation == true` **and**
   `phase_a_prime.reason` is a non-empty string.
3. `phase_b_capture.status == "DONE"` **⇒**
   `phase_b_capture.upstream_callable.module` is a canonical dotted
   Python path **and** `phase_b_capture.upstream_callable.function` is
   a valid Python identifier **and** `phase_b_capture.captured_at` is an
   ISO 8601 UTC timestamp.
4. `phase_b_capture.status == "FAILED"` **⇒**
   `phase_b_capture.reason` is a non-empty string.
5. `phase_b_capture_required == true` **and**
   `phase_b_capture.status != "DONE"` **⇒** the task is
   `release_status = BLOCKED` (not `PASS`); the eligibility gate enforces
   this.
6. `release_scope.excluded == true` **⇒**
   `release_scope.reason` is a non-empty string **and**
   `release_scope.excluded_at` is an ISO 8601 UTC timestamp **and**
   `release_scope.excluded_by` is a non-empty string. The task is
   excluded from the eligibility denominator (early-out in
   `is_score_eligible`).

**ISO 8601 timestamp format.** All `*_at` timestamps in this section are
written by the bench automation in
exactly the form `YYYY-MM-DDTHH:MM:SSZ` (UTC, second resolution,
trailing `Z`). The `±HHMM` numeric-offset form is **not** produced by
the bench — readers should accept either form per the ISO 8601 standard,
but the bench writes only the `Z` form.
