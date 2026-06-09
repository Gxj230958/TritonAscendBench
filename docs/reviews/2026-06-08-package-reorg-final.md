# Package reorg + rename → `ascendbench` — final unified plan

Status: PROPOSAL (no code changed). Naming + layout converged across function-analysis
workflow + codex co-review. Awaiting user go to write/execute.

## 0. Decisions

| Axis | Decision | Rationale |
|---|---|---|
| import name | `ascendbench` (all-lowercase, PEP 8) | `<X>Bench` industry convention (KernelBench/SWE-bench family); single anchor; PyPI-free; repo name already carries `triton` |
| display/brand | `AscendBench` | mirrors KernelBench / SWE-bench display casing |
| repo / PyPI dist name | `triton-ascend-bench` (UNCHANGED) | benchmark full name / distribution identity |
| layout | flat (`ascendbench/` at repo root) | benchmark monorepo, not a pure lib; NOT src-layout |
| `tasks/` | stays at repo root, sibling of `ascendbench/` | data ≠ engine; 12G, never shipped in wheel |

## 1. Target subpackage layout (8 subpackages)

Grouped by FUNCTION in the eval pipeline (the package is an adversarial grading engine).

| Subpackage | Modules (from flat top-level) | What the group does |
|---|---|---|
| `engine/` | eval.py, cli.py | central driver: `eval.evaluate` runs one (task, solution, backend); cli `analyze` = authoritative final scoring + release gate |
| `integrity/` | static_checker.py, process_isolation.py, adversarial.py, wrapper_contract.py | anti-cheat trust boundary — **center of gravity** (static_checker 4329L). Static source defense, trusted-parent/untrusted-child isolation, canary traps |
| `correctness/` | tolerance.py, golden_store.py, determinism.py | C2 oracle-equivalence verdict: dtype-aware tolerance, golden trust chain, N=3 determinism gate |
| `perf/` | timing.py, cost_model.py, peak_table.py, scoring.py | Roofline performance dim: t_npu timing, cost-model formulas, 910B peak tables, efficiency/FastP aggregation |
| `input_spec/` | inputs.py, expr.py, param_constraints.py | shapes.yaml → seeded oracle-faithful adversarial fixtures; shared safe-eval (expr); input-contract gate |
| `corpus/` | dataset.py, provenance.py, kernel_resolver.py | task loading + chain-of-custody provenance + AST-only host-wrapper resolver |
| `release_gate/` | eligibility.py, sanity_gate.py | no-subset-escape denominator + release_ready verdict (C2 ∩ S2), re-derived from trusted repo state |
| `policies/` | ceiling_acceptance.yaml, floor_dominated_acceptance.yaml, sanity_bounds.yaml, tolerance_policy.yaml, trust_roots.json | config data (was flat). **`.run_cert` EXCLUDED** (runtime sentinel, not config) |

## 2. Stay at package root

| Item | Why |
|---|---|
| `__init__.py` | public API surface (re-exports key entrypoints) |
| `triton_compat.py` | single low-level load-time helper; a 1-file package is over-structuring |
| `manifest.py` | submission-side `SolutionManifest` schema — NOT task corpus (codex co-review) |
| `.run_cert` | runtime sentinel, not policy config |
| `backends/` | already a clean GPU/NPU subpackage |
| `frozen/` | **SHA-locked import path** — 70 kernel.py + 96 meta.yaml + 23 frozen_deps reference it; cannot move/nest |
| `upstream_shims/` | **SHA-locked import path** (14 kernels import the autotune interp); cannot move/nest |
| `trusted_shim/` | spec-planned host-clock primitive, dormant; keep, do not silently drop (codex co-review) |

## 3. Before / after top level (34 → 16 items)

BEFORE: 24 `.py` + 6 config + 4 dirs flat.
AFTER:
```
ascendbench/
  __init__.py  triton_compat.py  manifest.py  .run_cert     # 3 .py + 1 sentinel
  engine/ integrity/ correctness/ perf/ input_spec/ corpus/ release_gate/ policies/   # 8 new
  backends/ frozen/ upstream_shims/ trusted_shim/            # 4 unchanged
```

## 4. Execution steps (when approved)

1. **Create subpackages**: `git mv` each module into its subdir; add `__init__.py` per subpackage.
2. **Rename import root + subpackage paths** in ONE sweep:
   - `triton_ascend_bench` → `ascendbench` everywhere
   - moved modules → `ascendbench.<subpkg>.<mod>`
   - sites: internal (~12), scripts (~189 lines), tests (~293 lines), 70 kernel.py, 96 meta.yaml
   - guard string-literal pins (test asserts, error strings) that a plain sed misses
3. **frozen `h_<sha>`: KEEP names, do NOT recompute** (codex-verified). vendor_* hashes are
   computed at CREATION time; `validate_tasks` compares only task kernel.py hashes, never the
   frozen dir names — there is no runtime hash verifier. So just rewrite the import literals
   inside (`triton_ascend_bench.frozen`→`ascendbench.frozen`); meta.yaml `frozen_deps` +
   kernel.py imports update via the same rewrite. (Same pragmatic keep-name as the frozen migration.)
4. **Move config → `policies/`**: fix hardcoded paths in tolerance.py, sanity_gate.py,
   eligibility.py, scripts/golden_manifest.py, etc. **Also fix the pre-existing P0**: add
   `[tool.setuptools.package-data]` + switch loaders to `importlib.resources` so the wheel
   actually ships yaml (today it does NOT — pip-installed config loading is broken).
5. **pyproject**: `packages.find include` still matches `ascendbench*`; verify new subpackages
   discovered; keep `tasks*` excluded.
6. **gpfs sync**: rsync WITH `--delete` (orchestrator `--update` alone leaves stale flat `.py`
   on the daemon side — known trap).
7. **Verification = full GPU regression** over all 101 tasks: S2 rerun (re-derives kernel
   SHAs as a by-product, never hand-edit), C1/C2 oracle-equiv, phase_a_prime, cost_model
   xcheck, golden, release_gate, gpu-rank. Must be all-green before push.
8. **codex commit-review** → user confirm → push → auto-sync gpfs.

## 5. Blast radius (exact)

- import-rewrite sites: ~230 (scripts 189 + tests 293 lines touch ~16/21 modules; internal ~12)
- **SHA-affecting**: 70 kernel.py + 96 meta.yaml + 7 frozen `h_<sha>` dirs → full S2/GPU rerun
- config load sites: ~15
- This is a single import churn + single full regression — same class as the frozen migration.

## 6. Risks + mitigations

| Risk | Mitigation |
|---|---|
| sed misses dynamic/string-literal module refs | pre-grep `import_module`/`__import__`/`"triton_ascend_bench.` literals |
| config not shipped in wheel (pre-existing) | fix as part of step 4 (package-data + importlib.resources) |
| stale flat `.py` on gpfs daemon | rsync `--delete` |
| SHA hand-edit temptation | SHAs are the by-product of the GPU S2 rerun, never hand-set |
| second big churn after frozen | single clean squashed commit, full regression, no half-states |
