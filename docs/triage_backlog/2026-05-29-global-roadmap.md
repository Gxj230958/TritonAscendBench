# Global Roadmap: 100-task Corpus

## Where we are

`15/100` PASS. The gap is no longer "100 unrelated bugs"; it is concentrated into two dominant infrastructure walls plus a smaller tail of contract/oracle cleanup.

## Root-cause bucket table

| Bucket | Count | System root cause | Owner |
|---|---:|---|---|
| `00 PASS` | 15 | Already green. | `me-solo` |
| `10 rewriter-clean, need oracle/shapes only` | 16 | Imports are already cleared; failures are now wrapper-contract, oracle, shape, and input-generation mismatches, with a few bounded runtime gates. | `me-solo` |
| `20 sglang.srt.* Class-A wall` | 28 | `sglang` re-exports live CUDA/platform/custom-op symbols through `common.py`; there is no sound broad shim under the current inertness rules. | `bench-team-arch-decision` |
| `30 env package missing` | 3 | Required runtime packages are absent from the daemon environment: `flashinfer`, `flashinfer.gdn_prefill`, and `sglang.jit_kernel`. | `env-change` |
| `40 FLA relative-import package` | 18 | These tasks depend on a multi-file FLA ops package; current rewriting handles single-file/single-symbol cases, not a package vendoring boundary. | `me-solo` |
| `50 vllm deep stack` | 20 | Tasks pull deeper `vllm` subsystems (`config`, `worker`, `model_executor`, `_custom_ops`, platform helpers) where some imports are inert, but many are real runtime dependencies. | `me-solo` |

## The hard truth about the ceiling

The roadmap is dominated by two buckets: `sglang Class-A = 28` and `FLA = 18`. Those two lanes decide whether this stays a `~30 PASS` effort or becomes a serious run at the full corpus.

The current solo ceiling is still about `~30 PASS`, and that estimate is basically correct. A realistic solo-only path is:

- `15` already PASS.
- `+3` from the fp8 packed-dist lane (`L1_07/L1_08/L1_09`) if the input generator is extended correctly.
- `+3` from the bounded bridge lane (`L2_20`, `L2_21`, `L2_19`), which is the cleanest remaining manual work in bucket `10`.
- `+9 to +12` from the FLA lane after whole-package vendoring, not the full `+18`, because bucket data already warns that some FLA tasks will reveal secondary `vllm` or wrapper-scope blockers after the package wall is removed.

That yields `30-33 PASS` as the realistic solo range. I would not promise more than that without architectural help. In particular, the `28` `sglang` tasks are effectively frozen today, and the `3` env-missing tasks are also frozen unless the environment changes.

With bench-team decisions greenlit, the next milestone becomes materially larger. `sglang + env` decisions expose `31` currently blocked tasks, and a working FLA vendoring lane exposes another `18` on paper. That does not imply `100/100`; it means a realistic program milestone moves from `~30` to roughly `~70-75`, because the `20`-task deep-`vllm` bucket remains a separate hard tail even after `sglang`, FLA, and env work are addressed.

## Prioritized investment ladder

### Executable now

| Rank | Lever | Tasks unblocked | Effort | False-PASS risk | Owner |
|---|---|---:|---|---|---|
| 1 | FLA whole-package U2 vendoring | `18` max, `9-12` realistic immediate yield | `2-4` sessions | Medium | `me-solo` |
| 2 | fp8 packed-dist generator for `L1_07/08/09` | `3` | `1` session | Low if generator matches packing contract | `me-solo` |
| 3 | Bounded bridges: `L2_20` -> `L2_21` -> `L2_19` | `3` | `1-2` sessions | Low to medium | `me-solo` |
| 4 | Per-symbol inert `vllm` shim audits | `2-5` plausible | `1-2` sessions | Medium to high | `me-solo` |

### Gated lanes

| Rank | Lever | Tasks unblocked | Effort | False-PASS risk | Owner |
|---|---|---:|---|---|---|
| G1 | `sglang` Class-A strategy | `28` | One architecture decision, then implementation | Low if using real package or tightly scoped vendoring | `bench-team-arch-decision` |
| G2 | Install `flashinfer` and `sglang.jit_kernel` in daemon env | `3` | `<1` session once approved | Low | `env-change` |

## Arch decisions the bench team must make

| Decision | Tradeoff | Tasks |
|---|---|---:|
| Pick a real `sglang` Class-A strategy: either curated inert-subset pool vendoring, or install/use `sglang` in a dedicated venv. | Pool vendoring keeps the bench source-controlled but creates ongoing maintenance burden; venv install reduces false shims but adds env management and version pinning. Current broad-shim attempts should be treated as a dead end. | `28` |
| Install `flashinfer` and `sglang.jit_kernel` in the daemon environment. | Slightly heavier daemon env, but this is the lowest-risk path for those tasks; stubbing these packages is the wrong optimization. | `3` |
| Decide whether selective unfreeze/rescope exceptions are allowed for wrappers whose frozen contract is unsalvageable. | Preserves benchmark purity if refused, but may permanently strand a small tail. `L1_25` is already a confirmed case; `L3_05/L3_06` are likely rescope candidates rather than clean bridge work. | `1` confirmed, `up to 3` likely |

## Recommended next 2 weeks

1. Stop spending time on random bucket-`20` `sglang` tasks until the bench team makes the architecture call. That is churn, not progress.
2. Spend the next few sessions on the FLA lane first. It is the only large non-arch lane with real corpus-moving leverage. Land package vendoring, rerun all `18`, and immediately rebucket survivors by their second blocker.
3. In parallel or immediately after, close the fp8 generator lane (`+3`) and then the bounded bridge lane (`+3`) in the ranked order already identified: `L2_20`, then `L2_21`, then `L2_19`.
4. Ask the bench team this week for two binary decisions: `sglang` Class-A strategy and daemon env package installs. Those two calls determine whether the project remains a `~30 PASS` cleanup effort or becomes a `~70+ PASS` program.
5. After those decisions land, rerun the newly unblocked `sglang` and env buckets first to expose their true secondary blockers. Only then should time go into the remaining deep-`vllm` tail.

The realistic milestone framing is:

- `~30 PASS` is achievable solo without architectural decisions.
- `~70-75 PASS` is plausible if `sglang`, FLA infra, and daemon env changes are greenlit.
- `100/100` is not a credible near-term target without a separate strategy for the `20` deep-`vllm` tasks and a policy decision on the small frozen-wrapper tail.

No false-PASS shortcuts should be taken to make those numbers look better.
