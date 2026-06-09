# `triton_ascend_bench/` package layout cleanup — plan

Status: PROPOSAL (no code changed). For codex deep-review then user scope decision.

## 1. Current state (facts)

`triton_ascend_bench/` is a **flat package**: 31 top-level `.py` modules (17k LOC)
+ 6 config files + 4 sub-packages, all in one directory.

```
top-level .py (LOC):  static_checker 4329 | cli 1554 | eval 1487 |
  process_isolation 1231 | inputs 1105 | param_constraints 858 |
  scoring 737 | adversarial 732 | eligibility 690 | expr 596 |
  wrapper_contract 513 | tolerance 457 | cost_model 432 |
  kernel_resolver 341 | dataset 333 | provenance 312 | golden_store 245 |
  timing 226 | sanity_gate 210 | peak_table 210 | triton_compat 166 |
  determinism 163 | manifest 86 | __init__ 24
config (flat): ceiling_acceptance.yaml | floor_dominated_acceptance.yaml |
  sanity_bounds.yaml | tolerance_policy.yaml | trust_roots.json | .run_cert
sub-packages: backends/ | upstream_shims/ | frozen/ | trusted_shim/
```

## 2. What's actually "messy" (5 dimensions)

1. **Flat sprawl** — 31 top-level modules, no functional grouping. A reader can't
   tell the eval engine from the grader from the input generator from the
   severance auditor; they're all one `ls`.
2. **Config scattered** — 6 yaml/json wedged between `.py` files, no `policies/`.
3. **Giant files** — 4 modules >1k LOC (static_checker 4329, cli 1554, eval 1487,
   process_isolation 1231). Splitting is a *separate, higher-risk* axis (see Scope C).
4. **`trusted_shim/`** — spec-planned host-clock primitive
   (`specs/timing_arbitration.md`, `specs/process_isolation.md`), currently
   imported by nobody. Not dead, but dormant — flag for decision.
5. **Public-API boundary blurred** — `__init__.__all__` lists 10 modules, but
   scripts/ and tests/ deep-import internals directly, bypassing the surface.

## 3. Hard constraints (the plan MUST honor)

| Constraint | Evidence | Implication |
|---|---|---|
| **`upstream_shims/` frozen** | 14 `kernel.py` import `triton_ascend_bench.upstream_shims.triton_autotune_interp` | Moving/renaming it changes 14 kernel SHAs → forces S2 rerun. **Do not move.** |
| **`frozen/` frozen** | 69 kernel.py import `triton_ascend_bench.frozen.*` (just migrated) | Same SHA lock. **Do not move.** |
| **`backends/` already clean** | gpu.py/npu.py, 1 sub-pkg | Leave as-is. |
| **No kernel.py imports eval-layer modules** | only the autotune shim above | Top-level eval-module regroup is **SHA-neutral** ✓ |
| **Config has hardcoded paths** | `tolerance.py:20` `REPO_ROOT/"triton_ascend_bench"/"tolerance_policy.yaml"`; `eligibility.py:243` `Path(__file__).parent/"sanity_bounds.yaml"` | Moving config touches ~15 load sites, mixed relative/absolute. |

### Blast radius of a top-level regroup (absolute imports everywhere)

| Consumer | Refs to top-level modules (non-frozen) |
|---|---|
| scripts/ | ~70 (16 modules: inputs 17, eligibility 9, dataset 6, kernel_resolver 5, …) |
| tests/ | ~150 (21 modules: dataset 30, eval 20, scoring 14, eligibility 12, cli 11, …) |
| internal cross-import | ~12 |
| config load sites | ~15 |

Every moved module = update all its import sites + a **full 101-task GPU
regression** (per project rule: layout change → whole-pipeline rerun).

## 4. Proposed target layout (Scope B)

Group by the bench's domain pipeline, not by technical layer. SHA-locked
sub-packages stay put.

```
triton_ascend_bench/
  __init__.py            # tightened public API (re-export key entrypoints)
  cli.py                 # entrypoint — stays top-level
  eval.py                # eval orchestrator engine — stays top-level (core)

  corpus/                # task/corpus loading + provenance
    dataset.py  manifest.py  kernel_resolver.py  golden_store.py  provenance.py
  inputgen/              # input synthesis (renamed to avoid inputs.py clash)
    inputs.py  expr.py  param_constraints.py
  grading/               # correctness + performance scoring + release gate
    tolerance.py  timing.py  scoring.py  cost_model.py  peak_table.py
    eligibility.py  sanity_gate.py
  severance/             # isolation + leak audit + anti-cheat
    static_checker.py  process_isolation.py  adversarial.py
    wrapper_contract.py  determinism.py
  policies/              # config data (was flat)
    ceiling_acceptance.yaml  floor_dominated_acceptance.yaml
    sanity_bounds.yaml  tolerance_policy.yaml  trust_roots.json

  triton_compat.py       # low-level shim — stays top-level (1 file, leaf)

  # UNTOUCHED (SHA-locked or already clean):
  backends/   upstream_shims/   frozen/   trusted_shim/
```

### Open grouping questions (for codex to arbitrate)

- **eval.py** top-level vs `grading/`? — proposed top-level (it's the orchestrator
  that drives grading, not a grading primitive).
- **cost_model.py / peak_table.py** in `grading/` vs a separate `perf/`? — proposed
  grading (both are perf scoring dims; avoid a 2-file package).
- **eligibility.py / sanity_gate.py** in `grading/` vs a separate `release/`? —
  proposed grading (tightly coupled to scoring; release-gate is a consumer view).
- **provenance.py** in `corpus/` vs `severance/`? — proposed corpus (imports
  dataset, is upstream attribution), but anti-cheat angle argues severance.
- **triton_compat.py** top-level vs a `_compat/`? — proposed top-level (single leaf
  file; a package for one module is over-structuring).
- **inputgen/** name — alternatives: `inputs_/`, `synth/`, `generators/`.

## 5. Scope options (cost / benefit — user picks)

| Scope | What | Import churn | GPU regression | Effort | Recommendation |
|---|---|---|---|---|---|
| **A** (conservative) | Only `policies/` config grouping + trusted_shim decision + tighten `__init__` API. No module moves. | ~15 config sites | smoke only | ~30 min | Solves dim 2+4+5, **leaves the core flat-sprawl (dim 1) unsolved** |
| **B** (recommended) | A + regroup 26 top-level modules into corpus/inputgen/grading/severance. | ~230 import sites | **full 101-task** | ~2 h | Solves dims 1-2-4-5. The real cleanup the user asked for. |
| **C** (aggressive) | B + split giant files (static_checker 4329, cli, eval, process_isolation). | ~230 + internal | full + heavy | ~half day | **Not this pass** — separate axis, high risk, defer. |

## 6. Why B (not A), honestly

The user's complaint is "太乱了" — that's dimension 1 (flat sprawl), which A does
not touch. A is cosmetic. B is the substantive fix. The cost of B is real
(~230 mechanical import rewrites + a full GPU regression) but it is *mechanical*
and *SHA-neutral* (no kernel touched), and it can be scripted + verified the same
way the frozen migration was. It is not avoidance-worthy work.

## 7. Execution outline (if B approved) — NOT YET DONE

1. `git mv` modules into sub-packages; add `__init__.py` per sub-package.
2. Rewrite imports: internal (~12) + scripts (~70) + tests (~150) via scripted
   `git grep -l | sed`, then verify 0 residual `triton_ascend_bench.<moved>` flat refs.
3. Move config to `policies/`; fix ~15 load sites (mixed `__file__`-relative +
   hardcoded `REPO_ROOT/"triton_ascend_bench"/"*.yaml"`).
4. Tighten `__init__` to re-export the documented public surface.
5. Resolve `trusted_shim/` (keep dormant w/ note, or drop per user).
6. Unit suite green → push to GPU → **full 101-task regression** (C1/C2 oracle-equiv
   + phase_a_prime/S2 + cost_model_xcheck + golden + release_gate + gpu-rank).
7. codex commit-review → user confirm → push.

## 8. Risks

- **Hidden dynamic imports** — `importlib`/string-based module refs that grep
  won't catch. Mitigation: grep for `import_module(`/`__import__`/`"triton_ascend_bench.` string literals before moving.
- **Config path breakage** — the hardcoded `REPO_ROOT/"triton_ascend_bench"/"*.yaml"`
  sites are the fragile ones; each must be repointed to `policies/`.
- **specs/ doc drift** — specs reference `triton_ascend_bench/<module>.yaml` paths in
  prose; update or note.
- **Re-review fatigue** — this is the 2nd big import-churn change after frozen;
  must be a single clean squashed commit + full regression, no half-states.
```
