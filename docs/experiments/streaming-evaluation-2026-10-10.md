# Streaming Evaluation Experiment Report

## Pull Request Description

This PR replaces eager trial serialization and accumulated evaluation outputs
with a complete streaming lifecycle. The runner constructs one deterministic
trial at a time, transfers safetensors through sealed Linux memfd descriptors
with bounded JSON control messages, validates returned outputs immediately, and
releases each trial before advancing. A per-trial temporary-file fallback is
available. Offline golden tensors are loaded only for the current shape, while
oracle inputs are independently reconstructed from their seeds. Live tensor
and transport resources therefore scale with the largest shape's working set
rather than the total number of shapes and trials.

The implementation preserves separate, sequential correctness and timing
processes, the existing evaluation JSON schema, and legacy dictionary/list
callers. Formal scoring remains 10 warmups, 100 measured calls, and 10 secret
sampled-output checks, with the profiled invocation excluded from latency
statistics. Input preparation, transfer, and parent validation remain outside
the kernel timing window. Device selection, private cache/scratch/profiler
directories, separate timeout budgets, protocol validation, and scoped cleanup
support concurrent execution without deleting other runs' resources. Failed
or incomplete streamed passes cannot produce valid performance results.

Validation includes 279 passing CPU regressions and one skip, controlled
comparisons against main and improvement-1, and verified 1/2/4/8-worker NPU
experiments. The 101-task campaign and supplemental retries produced 194
canonical attempts: 73 tasks completed correctness and timing on all 3,078
required shapes. The remaining 28 tasks retain explicit numerical, compilation,
contract, device-execution, or missing-golden outcomes. In the controlled
four-shape CPU comparison, peak owned temporary storage fell from approximately
419 MB to 24 KB. Frozen sources, raw receipts, resource measurements, and a
checksummed evidence archive accompany this report. These results establish
lifecycle improvements, not a kernel speedup or 101/101 acceptance. Offline
seed-0 sampling retains its documented anti-cache limitation, and sampled
memory accounting is not a hard cgroup guarantee.

## Experiment Status

Status: implementation, regressions, comparison/concurrency experiments, and
the primary 101-task campaign and supplemental candidate validation are complete.
**73 tasks / 3,078 shapes passed full verified device coverage.**
The remaining 28 tasks are explicit incomplete
coverage, not a 101/101 device acceptance claim. Experiments ran October 10 and
12, 2026; the resumed suite took 5,135.731 seconds.

## Implementation And Environment

- Baseline `main`: `61a7fcdc03ea3407667a20ee1e6991940fb790ad`.
- First materialization optimization: `8fa4755d68bd2364a7e2549b47f2e85c535fa584`.
- Streaming implementation: `2092615` on `fix/streaming-eval-inputs`.
- All 163 primary canonical attempt receipts match the five runtime source hashes at
  `2092615`. Supplemental runtime hashes match `f940554`; tooling also advanced
  to `18468d6` during that campaign. Later commits update experiment tooling and documentation; final
  runtime-file edits correct docstrings/comments only, with AST equivalence
  verified after removing docstrings.
- Host: Ubuntu 22.04.5 LTS, Linux 5.15.0-119-generic, Python 3.11.14;
  two Xeon Platinum 8468 sockets, 192 logical CPUs, two NUMA nodes.
- Device: Ascend 910B2C, 16 visible physical cards, driver `25.0.rc1.1`.
- CANN: `8.5.1`, inner version `V100R001C25SPC002B220` from
  `/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascend_toolkit_install.info`.
- PyTorch `2.9.0+cpu`, torch-npu `2.9.0.post1+gitdc51c2d`,
  triton-ascend `3.2.0.dev20260322`, safetensors `0.7.0`.
- `/sys/fs/cgroup` is mounted read-only. No independent hard cgroup limit was
  available. Admission/abort monitoring is observational, not hard enforcement.

The original working tree and its uncommitted timeout experiment were left
intact. Work started from `main` in `/tmp/TritonAscendBench-streaming`.
The timeout requirement was implemented independently. No unrelated probe or
experimental source files were carried into the branch.
The final worktree is retained at
`/inspire/sj-ssd3/project/qianghuaxuexi/s26045/TritonAscendBench-streaming`;
primary/replay/supplement raw records are also retained beneath
`/inspire/sj-ssd3/project/qianghuaxuexi/s26045/tab-streaming-evaluation-2026-10-10/`.

## CPU Regression

The final distinct regression total is **279 passed and 1 skipped**. The core
implementation matrix contributed 136 passes and one skip across the
streaming lifecycle, legacy isolation protocol, two-pass execution, golden
routing, secret sampled-output checking, timing/status integration, fixed
denominator, and profiler parser suites. The skipped test requires a corpus
fixture absent from the checkout. JUnit receipt: `tab-streaming-tests-2092615.xml`.

The disjoint input/tolerance/determinism matrix contributed 119 passes
(`tab-streaming-inputs.xml`), supervisor admission/resume added three passes
(`tab-streaming-validation-tests.xml`), and coverage-summary checks added four
passes (`tab-streaming-summary-tests.xml`). Observed partial execution and
flagged performance cannot be counted as a verified full pass.
Candidate-screening tests added three passes (`tab-streaming-supplement-tests.xml`),
evidence-merging tests two (`tab-streaming-evidence-tests.xml`), and release
boundary tests twelve (`tab-streaming-release-tests.xml`). These seven JUnit
receipts contain distinct cases; repeated runs are not counted again.
Tests cover original input mutation, output release before next construction,
seed reproducibility, mixed tensor/scalar inputs, offline golden release,
missing outputs, child crash, execution/preparation timeout, malformed JSON,
extra/unsealed descriptors, device selection and process reaping.

CPU tests used an isolated venv with the host packages exposed except
`torch_npu`, matching the repository's dev-machine tests. Direct testing in the
host environment initially failed because those tests assume no NPU runtime;
the host has torch-npu installed while the tool sandbox hides devices. The
sandbox also blocks inherited Unix-socket descriptor access. CPU IPC tests were
therefore run outside that sandbox with accelerator packages absent.

Two old tests assume an untracked `solutions/L1_01_bias/solution.py` fixture
that is missing in both the original checkout and the fresh worktree. They are
not part of the passing matrix and were not presented as passing.

## Controlled CPU Comparison

Synthetic workload: four shapes, 1,048,576 float32 input elements per shape
(4 MiB), two warmups, 20 measured calls, three checked outputs. This small
experiment measures lifecycle behavior, not formal NPU performance. Seeds and
candidate code are identical across implementations. The streaming path also
immediately verifies outputs. Legacy global cache deletion was not invoked.

| Metric | main | improvement-1 | streaming |
|---|---:|---:|---:|
| Peak process-tree RSS, bytes | 895,242,240 | 520,876,032 | 531,111,936 |
| Peak whole-host file cache, decimal GB | 126.924 | 127.110 | 126.756 |
| Peak owned temporary file bytes | 419,486,298 | 419,486,299 | 24,196 |
| Peak observed tree write_bytes | 453,873,664 | 453,873,664 | 86,016 |
| First candidate call wait, s | 1.654 | 1.097 | 1.070 |
| Total elapsed, s | 3.600 | 3.522 | 3.863 |

Interpretation: first-round laziness reduced tensor RSS but preserved cumulative
file materialization. Full streaming removed trial-scaled temporary payload
files. Its RSS is slightly above improvement-1 in this small workload, and its
elapsed time is longer: per-trial synchronization and bounded copying are real
costs. No kernel speedup is claimed. Process-tree RSS double-counts shared pages;
global cache was also recorded and is confounded by other host activity.

Holding input size fixed while increasing synthetic shapes from 1 to 16 yielded
511,377,408 versus 527,192,064 peak RSS bytes, 24,196 temporary bytes in both
cases, and 86,016 observed write_bytes in both. This supports bounded live
payload resources, not an assertion that allocators return all RSS immediately.
These two checks were repeated on October 12 under `f7ce419`, whose runtime
sources are identical to `2092615`; the receipts include source hashes.
Increasing further to 16 shapes and 100 measured trials per shape produced
528,220,160 peak RSS bytes, 24,196 temporary bytes, 14 observed tree descriptors,
and 86,016 observed write_bytes. Elapsed time was 21.850 seconds. This run used
the fixed implementation and checked sampled outputs immediately; its resource
bound holds across 1,600 measured invocations, without input reuse.

## Device Method

The inventory fixes 101 task IDs, 4,169 public/hidden shape entries
(3,493 public and 676 hidden), candidate
SHA-256 hashes, and available historical evidence hashes. Four tasks had no
historical verified JSON; existing source candidates were inventoried for them
with that absence made explicit. All attempts use the original complete shape
sets and formal 10 warmup / 100 measured / 10 secret sample defaults. The
profiled measured invocation is excluded, leaving 99 clean durations per shape.

For scale context, `L2_20_bmm/P4` contains 6,979,321,856 input tensor bytes.
Pre-serializing its 10 warmup plus 100 measured inputs alone would require
approximately 767,725,404,160 bytes (767.725 decimal GB), before other shapes,
correctness inputs, outputs, serializer copies, or references. This is an
arithmetic estimate from the frozen shape specification, not a measurement of
an attempted dangerous legacy run. Large original workloads were therefore
restricted to the controlled comparisons above.

The campaign ramps through 1, 2, 4 and 8 workers. Every worker checks the
selected physical card is healthy and idle immediately before runtime imports.
An attempt records candidate/reference/code hashes, versions, command, full
evaluation JSON and scalar protocol receipts. Failed tasks retry up to three
distinct existing candidates. File and profiler resources are private to each
pass, and only owned descendants may be stopped.

Memory accounting sums owned process-tree RSS and the whole host's cached and
buffered memory. Including unrelated cache and double-counted shared RSS is
conservative. Admission is below 800,000,000,000 bytes; abort is at
900,000,000,000 bytes. The requested acceptance ceiling is decimal
1,000,000,000,000 bytes. Polling cannot prove a hard instantaneous cap.

Complete device coverage requires every required shape to have both
correctness and timing receipts with an active backend, observed kernel events,
and 99 clean timing samples. Numerical, compilation, artifact/contract, and
harness failures are classified separately. A numerical failure remains a
candidate failure even when memory behavior is correct; a blocked artifact does
not become a device pass just because the CLI returned a JSON file.

NPU baseline comparisons use the same frozen `L1_01_bias` candidate and full
shape set. The old implementations require explicitly disclosed adapters for
device forwarding, private HOME/TMPDIR, a larger explicit child timeout, and
scalar diagnostic receipts. Global cache scrub is disabled to avoid touching
other users' resources. Their tensor construction, serialization, candidate
timing, and output accumulation remain the baseline implementation.

All three versions completed the full 40-shape `L1_01_bias` task with
correctness, runtime, and performance statuses `ok` on physical NPU 8.

| Metric | main | improvement-1 | streaming |
|---|---:|---:|---:|
| Peak process-tree RSS, bytes | 15,656,464,384 | 2,893,250,560 | 4,986,515,456 |
| Peak whole-host cache + buffers, decimal GB | 126.481 | 127.117 | 125.964 |
| Peak observed tree write_bytes | 3,794,771,968 | 3,794,948,096 | 309,399,552 |
| Full evaluation elapsed, s | 324.552 | 299.536 | 333.817 |
| Median of shape host medians, ns | 176,394.5 | 169,654.0 | 191,791.0 |
| Median of shape NPU-event medians, ns | 117,599.0 | 113,329.5 | 118,789.5 |
| Median of shape host CVs | 0.0892 | 0.1042 | 0.1352 |

These are single runs with background campaign activity on other cards, not
randomized repeated trials. The device-event figures are similar; neither the
elapsed times nor host timing differences establish a causal kernel speedup.
The memory figures show why the two optimizations address different resources:
the first removes eager tensor allocation, while the second removes accumulated
trial files and outputs. Streaming's extra receiver/parser workspace makes its
RSS higher than improvement-1 in this task. `write_bytes` is the peak sum for
live owned processes, not an exact cumulative total after exited children;
profiler/cache writes remain. Cached reads yielded zero observed block-device
read bytes in these NPU cohorts. Temporary-file size and first-call wait were
directly measured in the controlled CPU comparison; the NPU adapters did not
collect equivalent peak temporary-size or first-call receipts. No retrospective
NPU measurements for those two metrics are inferred.
The original stage-ramp campaign and the frozen-code replay are separate
cohorts. Their overlap, and overlap with the comparison runs, must be considered
when interpreting host contention. No card is shared between active cohorts.

Diagnostic baseline checks reproduced the first three failing shapes for both
`L1_12_rms_norm` and `L1_15_merge_state` through original `main`, with unchanged
candidate hashes and the same seed-0 reference route. The absolute and relative
error summaries matched the streaming failures. RMS norm used its PyTorch
oracle; merge-state used its offline golden artifacts. These checks support a
candidate numerical-failure classification and are not formal full-task coverage.

The two primary candidate passes retain their task-level ordering. Existing
metamorphic checks may run additional, separate isolated candidate invocations
after those passes; they are outside the timed evaluation and were preserved.
Profiler parser helpers likewise do not count as candidate passes.
The final summary also reports shape-level observed device events separately
from completed full-pass records. When a timing pass fails, its duration arrays
are invalidated even if earlier profiler evidence is present. Observed partial
execution therefore never promotes a task into full device coverage.

## Full Campaign Results

The canonical result set substitutes frozen-code replays for the initial seven
tasks. Every one of the 101 tasks has a formal result. There are 163 canonical
primary candidate attempts, and every non-passing task exhausted all three fixed
candidates. The primary 72 full verified passes cover 3,031 of 4,169 shapes (72.70%).
Supplemental validation recovered `L2_27_fused_moe_router_cudacore`, adding all
47 shapes: combined coverage is 73 tasks / 3,078 shapes (73.83%).
The combined set has 194 attempts. Across successful and failed attempts,
3,912 shapes have observed device kernel events (93.84%); this is diagnostic
execution evidence, not full-pass coverage.
The remaining 1,091 shapes lack complete correctness-and-timing coverage.

| Final Task Category | Tasks | Interpretation |
|---|---:|---|
| Complete verified | 73 | Both passes, all required shapes, 99 clean timings |
| Candidate numerical failure | 15 | Correctness or secret sampled output failed |
| Offline artifact / contract blocked | 5 | Missing trusted golden, no device acceptance |
| Candidate compile / launch failure | 6 | Last candidate fails toolchain compilation or launch limits |
| Candidate contract failure | 1 | Candidate requires an input key absent from the formal contract |
| Device execution failure | 1 | Vector-core exception; attribution requires further diagnosis |

Final failure categories describe the last attempt; all earlier outcomes are
retained as counts in the task table and full outcomes in JSON. At attempt level,
counts are 73 verified, 78 numerical failures, 15 static rejections, 12 artifact
blocks, 14 compile/launch failures, one candidate contract failure, and one device execution failure.
No attempt remains unclassified, and no transport/protocol/cleanup
regression was observed in this campaign. This is evidence for these workloads,
not a proof that the harness has no bugs. Candidate numerical failures were
also reproduced in controlled original-main diagnostics as described above.

All 73 complete tasks are rank eligible under the existing scoring
policy. Only 10 have `absolute_verified=true`; 63 retain unverified absolute
roofline parameters. Verified evaluation mode establishes execution/integrity
checks, not independent verification of every hardware peak-table parameter.
This change does not promote placeholder peak values into measured constants.

All 3,078 complete timing-shape records have 99 positive finite host durations
and 99 NPU-event durations. Release counts match the full schedule, including
compile warmup. All 259 recorded owned profiler roots were removed, and the
post-run process audit found no remaining campaign/candidate processes.

| Large Task | Shapes | Timing Input Bytes Transported | Maximum Payload Bytes | Released Timing Trials | Full Evaluation, s |
|---|---:|---:|---:|---:|---:|
| L2_20_bmm | 17 | 946,455,051,990 | 6,979,322,112 | 1,887 | 4,358.626 |
| L2_33_moe_mmk | 45 | 142,435,249,284 | 671,089,392 | 4,995 | 2,146.812 |
| L3_16_fwd | 34 | 123,860,848,944 | 805,307,096 | 3,774 | 2,818.305 |
| L3_21_expert_triton | 45 | 284,868,574,272 | 1,342,178,032 | 4,995 | 2,312.103 |

These byte counts are observed serialized input traffic, including compile
warmup and safetensors headers; they are not simultaneous residency. The BMM
timing pass transported about 946 GB over 4,157.355 seconds while releasing each
trial. Its first execute wait was 22.939 seconds. Correctness transported another
17,053,244,180 bytes and released 34 trials. Large random-input construction and
transfer remain substantial work outside the kernel timing window.

The scalar [task table](../../scripts/internal/evidence/streaming-evaluation-2026-10-10/task-results.md)
and [summary JSON](../../scripts/internal/evidence/streaming-evaluation-2026-10-10/summary.json)
contain each task's required/observed shapes, attempts and failure reasons.
The checksummed raw evidence archive, manifest, 303 primary and 45 selected
supplemental source entries, all 878 screened unused sources and available
historical receipts, reference metadata, resource samples, commands, environment
record and seven JUnit files live beside them under `scripts/internal/evidence/`.
The final [audit receipt](../../scripts/internal/evidence/streaming-evaluation-2026-10-10/audit.json)
checks 4,139 archived files against their byte sizes and SHA-256 hashes, source
hashes and runtime AST equivalence, schedule release counts, positive timing
samples, device selection, and profiler-root removal.
Input/output tensors and raw profiler traces are deliberately released rather
than accumulated; trusted parser receipts and event durations retain execution
evidence. Existing export-ignore rules exclude the internal evidence from release
archives. A source checkout of this branch is required to inspect those links.

## Fixed-Workload Concurrency

Each worker runs the same frozen `L1_07_dequantize_k_cache_fast` candidate and
its full 15-shape set with formal defaults. All 15 replicas across the four
cohorts returned correctness/runtime/performance `ok`. Cards are distinct,
private caches are cold, and admission waiting is excluded from elapsed time.

| Workers | Elapsed, s | Tasks/hour | Host Median, ns | NPU Event Median, ns | Host CV | Accounted Peak, GB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 150.942 | 23.85 | 180,285 | 111,400 | 0.2032 | 129.518 |
| 2 | 151.557 | 47.51 | 172,509 | 109,209.5 | 0.1618 | 139.880 |
| 4 | 153.923 | 93.55 | 174,938 | 110,209 | 0.1726 | 143.155 |
| 8 | 156.486 | 184.04 | 180,679 | 110,139.5 | 0.1820 | 166.227 |

Median/CV columns aggregate shape statistics across workers. Eight workers
waited 873.254 seconds for the replay card to become idle before starting.
The measured throughput ratio is 7.72x for eight versus one worker in this
particular short task. Background full-suite work ran on separate cards;
these single cohort measurements do not establish interference-free scaling
or a change in kernel implementation speed.

The aggregate guard observed a peak of 215.494 decimal GB across the overlapping
October 10 owned campaign/replay/concurrency process trees plus all host file
cache. Earlier cohorts and the October 12 resumed suite have separate resource
receipts. The resumed eight-worker suite peaked at **219,314,810,880 bytes
(219.315 decimal GB)**, including owned parent/child RSS and whole-host cache.
All recorded cohort peaks stayed below admission and abort watermarks.
The first supplemental cohort peaked at 282,967,035,904 bytes (282.967 GB).
The completed resumed supplemental cohort took 919.078 seconds and peaked at
214,189,350,912 bytes (214.189 GB). The earlier supplemental elapsed time,
1,068.478 seconds, includes the interval waiting for externally occupied cards.
This is below the 1000 GB ceiling at observed sampling points; the
read-only cgroup environment prevents a hard instantaneous-cap claim.

## Campaign Resumption

The initial October 10 campaign produced 55 task result files. Device admission
failures then affected other tasks and retries. Those failures happened before
candidate loading and do not count as numerical or compiler failures or actual
device coverage. The experiment supervisor initially consumed candidate retries
on such exits; this was a validation-tool defect, independent of trial IPC.
The October 12 correction checks healthy idle cards before dequeueing, retries
an admission race with the same candidate, and resumes the first unexecuted
candidate. Previous logs and supervisor receipts are preserved. Three focused
regression tests cover busy-card rejection and resume selection.

### Offline Artifact Gaps

Four tasks reached the missing-golden gate in the primary campaign. A fifth
reached it only after a supplemental candidate passed the static gate. Their
trusted offline artifact sets are absent:

| Task | Required Shapes |
|---|---:|
| L2_02_layer_norm_fwd_1pass | 37 |
| L2_17_w8a8_block_int8_matmul | 42 |
| L2_23_layer_norm_gated_fwd | 42 |
| L3_19_fused_moe | 40 |
| L3_07_fused_sigmoid_gating_delta_rule_update | 42 |

These 203 shapes are an explicit coverage gap (161 in the four primary blocks).
The existing capture report,
sibling benchmark worktrees, and experiment task directories supplied no
complete alternative artifact sets. Other inventoried candidates were tried;
`fused_moe` reached the missing-artifact gate after a prior static rejection.
The upstream-reference tier was preserved. A simplified PyTorch oracle is not
a substitute for the provider's offline golden and applicable integrity chain in verified mode.
Completing these tasks requires the missing trusted artifacts.
On October 12 the user explicitly requested that this issue be recorded and
temporarily skipped. The same artifact issue was reported for the fifth task
when it became reachable. Artifact repair is deferred; these results retain
their blocked status and are not counted as verified coverage.

### Supplemental Candidate Validation

After the three-candidate primary campaign, 878 unused distinct candidate
sources were screened for the other 25 failing tasks. Existing integrity checks
rejected 232. Selection retained all compatible candidates with historical
correctness evidence, or the strongest compatible candidate where none had
such evidence: 45 candidates in total. Nine attempts completed before an
unowned process occupied all 16 cards. No external process was stopped.
The idle supervisor was closed with no active workers, preserving resume state.
The full device-admission snapshot is archived. At that checkpoint, 36 selected
candidates across 16 tasks remained unexecuted. When all cards became healthy
and idle again, validation resumed and completed another 22 attempts.
All 25 tasks received supplemental validation: 31 candidates executed in total.
The router's second candidate passed all 47 shapes, so its remaining 14 selected
backups were unnecessary and were not executed. All six LoRA shrink candidates
failed independently seeded timing checks, despite seed-0 correctness passes.
No required retry remains queued. Historical seed-0 correctness alone did not
predict formal sampled-output success.

The supplemental GELU candidate processed a shape with 34,376,523,776 input
tensor bytes but failed a later shape's compiler UB allocation limit
(2,490,368 required bits versus 1,572,864 available on this toolchain).
It supplies large-input lifecycle evidence, not a successful task or performance
result. A rejection-sampling candidate encountered a vector-core exception
(`507035`) after an earlier numerical mismatch; this is reported separately
and is not attributed to the harness without a controlled diagnosis.

## Reproduction Commands

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python scripts/internal/streaming_validation.py inventory \
  --tasks /path/to/TritonAscendBench/tasks \
  --experiments /path/to/tab-experiments --out /path/to/evidence
TAB_PASS_TIMEOUT_S=1800 python scripts/internal/streaming_validation.py campaign \
  --out /path/to/evidence --device 0
python scripts/internal/streaming_summary.py /path/to/evidence

# Select additional existing candidates using a primary merged summary.
python scripts/internal/streaming_supplement.py \
  --suite /path/to/evidence --summary /path/to/primary-merged/summary.json \
  --experiments /path/to/tab-experiments --out /path/to/supplement
TAB_PASS_TIMEOUT_S=1800 python scripts/internal/streaming_validation.py suite \
  --out /path/to/supplement --workers 8 --device 0

python scripts/internal/streaming_evidence.py \
  --suite /path/to/evidence --replay /path/to/replay \
  --supplement /path/to/supplement --out /path/to/bundle
python scripts/internal/streaming_audit.py \
  --evidence /path/to/bundle --suite /path/to/evidence \
  --replay /path/to/replay --supplement /path/to/supplement

TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 \
  python scripts/internal/streaming_comparison.py \
  --root /path/to/checkout --kind streaming --out /path/to/comparison.json
```

The generated inventory must be retained; recreating it later can select newer
candidates. Reports disclose any attempt made before the implementation commit
was fixed. Conclusions about latency must account for concurrent host/device
load; changing input pacing can affect device idle gaps even though generation
is outside the timing window.
Replay requires the recorded external trusted golden tensor sets and task data;
the scalar archive deliberately does not duplicate those large tensors.
Inventory paths identify the original locations; frozen candidate sources and
reference metadata in the archive allow them to be checked or restored.

## Remaining Trust Limits

Sealed memfd and data-only IPC do not provide OS confinement against arbitrary
native code. Static/runtime checks and the existing compile audit window remain
the repository's threat model. Offline sampled trials reuse seed-0 content;
secret indices do not prevent content recognition or caching. Multi-seed golden
coverage, native-code confinement and stronger admission prediction are future
work, not completed results.
