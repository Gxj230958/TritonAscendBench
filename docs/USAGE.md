# Usage Guide

This guide takes you from a fresh checkout to a graded task. It is a
hands-on companion to the [README](../README.md) Quick Start; for *how the
bench works*, read [ARCHITECTURE.md](ARCHITECTURE.md) next.

## TL;DR — the whole loop

```bash
pip install -e .                                        # from a clone or release tarball
tab init-solution L2_04_silu_and_mul                    # scaffold solution.py
#   implement npu_triton_impl as a real Triton kernel   # see §3 for a full example
tab run L2_04_silu_and_mul \
    --solution solutions/L2_04_silu_and_mul/solution.py  # grade -> round_result.json
#   author manifest.yaml in the solution dir           # validate-submission requires it
tab package solutions/L2_04_silu_and_mul --out submission.tar.gz
tab validate-submission submission.tar.gz               # then: tab submit ... --endpoint <url>
```

Each step is detailed below.

## 1. Get the bench and install

`tasks/` (the task corpus) ships with the **git clone or the release
tarball** — it is not in the PyPI wheel. From a checkout:

```bash
pip install -e .            # core
pip install -e '.[infer]'   # optional: only for provenance/severance tooling
```

Python 3.10+ is required. Verify the CLI:

```bash
tab --help                  # or: python3 -m ascendbench.engine.cli --help
```

## 2. Scaffold a solution

```bash
tab init-solution L2_04_silu_and_mul
# -> scaffolded solutions/L2_04_silu_and_mul/solution.py
```

`init-solution` writes a **single `solution.py`** (no manifest is created —
you author `manifest.yaml` later, at package time). Use `--out <dir>` to
choose the location.

The stub defines the entrypoint you must implement:

```python
def npu_triton_impl(inputs: dict) -> dict:
    """`inputs` is built from the task's shapes; return a dict of output
    tensors with the keys the contract specifies."""
    ...
```

## 3. Implement your kernel

Read `tasks/<level>/<task_id>/contract.md` for the I/O dict keys, tolerance
class and shape coverage. Two rules the grader enforces (see
[reward_hack_defense.md](reference/reward_hack_defense.md)):

- The entrypoint **must be `npu_triton_impl(inputs) -> dict`**.
- Your solution **must be a real Triton kernel** — `@triton.jit` with
  `tl.load`/`tl.store`. A plain-PyTorch implementation is **rejected by the
  static checker** (it exists to stop "just do it in torch" submissions).

A complete, copy-pasteable `solution.py` for `L2_04_silu_and_mul` (output =
`silu(x1) * x3`, where `hidden_states` is split in half into `x1, x3`):

```python
import torch
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_kernel(x_ptr, out_ptr, hidden_dim, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < hidden_dim
    x1 = tl.load(x_ptr + row * n_cols + col, mask=mask, other=0.0)
    x3 = tl.load(x_ptr + row * n_cols + hidden_dim + col, mask=mask, other=0.0)
    x1f = x1.to(tl.float32)
    silu = x1f * tl.sigmoid(x1f)
    out = x3 * silu.to(x3.dtype)
    tl.store(out_ptr + row * hidden_dim + col, out, mask=mask)


def npu_triton_impl(inputs: dict) -> dict:
    hs = inputs["hidden_states"]                 # keys come from contract.md
    bs, n_cols = hs.shape
    hidden_dim = n_cols // 2
    out = torch.empty((bs, hidden_dim), dtype=hs.dtype, device=hs.device)
    BLOCK = 128
    grid = (bs, triton.cdiv(hidden_dim, BLOCK))
    _silu_and_mul_kernel[grid](hs, out, hidden_dim, n_cols, BLOCK=BLOCK)
    return {"output": out}                       # output key(s) from contract.md
```

Get the input/output keys (`hidden_states`, `output` here) and the tolerance
class from that task's `contract.md`; different tasks have different keys.

## 4. Grade one task

Point `--solution` at the **`solution.py` file** (not the directory):

```bash
tab run L2_04_silu_and_mul --solution solutions/L2_04_silu_and_mul/solution.py
# prints a one-line summary (score=… coverage=… static_ok=… → round_result.json)
```

The `round_result.json` records the per-dimension verdict:

| field | meaning |
|---|---|
| `static_checks_passed` | passed the anti-gaming static gate (real Triton kernel, no banned calls) |
| `correctness_status` | `ok` / `fail` / `error` — kernel output vs the PyTorch oracle within tolerance |
| `runtime_status` | `ok` / `error` / … — did the kernel execute |
| `performance_status` | `ok` / … — timing/roofline result |
| `task_status` | `evaluated` / `blocked` / … — overall |
| `coverage` | fraction of the task's shapes that passed |

(Status enums are broader than the common values shown; the above are the ones you meet first.)

> `VERIFIED` / `CEILING_ACCEPTED` are **corpus provenance** statuses (how the
> reference kernels were certified), not your `tab run` verdict. Your run
> reports the `*_status` fields above.

### Reading precision + performance detail

The top-level `*_status` fields are a summary. The **per-shape** detail lives in
`round_result.json` under `shape_results[]` — one entry per shape, keyed by `shape_id`:

**Precision** (per shape):
- `correctness_passed` — did this shape's output match the oracle within tolerance.
- `correctness_reason` — why it passed/failed (the tolerance detail, or the error).
- `metamorphic_results` — metamorphic-check outcomes.

**Performance** (per shape):
- `efficiency` — Roofline efficiency (achieved throughput vs peak).
- `t_ref` / `t_npu` — reference vs measured device time.
- `baseline_ns` / `measured_ns` / `median` / `p99` / `cv` / `trials` — latency statistics.
- `w_per_class` / `m_per_resource` — work and measured cost per resource class (vector / cube).
- `roofline_error` — any performance-measurement problem.

Read them with a few lines of Python:

```python
import json
d = json.load(open("round_result.json"))
for s in d["shape_results"]:
    print(s["shape_id"], "correct:", s["correctness_passed"],
          "| reason:", s["correctness_reason"], "| eff:", s["efficiency"])
```

An **excerpt** of a real per-shape entry — a CPU-only smoke of `L2_04_silu_and_mul`,
no device (the other emitted keys above are omitted here for brevity):

```json
{"shape_id": "P1", "correctness_passed": false,
 "correctness_reason": "RuntimeError: 0 active drivers ([]). There should only be one.",
 "t_ref": 1.28e-09, "t_npu": null, "efficiency": null, "measured_ns": null}
```

> **Real numbers need the target backend.** Correctness *can* be real on CPU when the
> Triton interpreter runs the kernel cleanly (tensors stay on host); without a working
> target backend the performance fields are `null`/`error` or host/interpreter-only
> diagnostics — **not** authoritative 910B numbers. Roofline performance requires real
> Ascend 910B hardware (README §Status). The example above shows the no-device case,
> where even correctness is `error` (`0 active drivers`).

**What an agent should check:** the run passed if `correctness_status == "ok"` and
`coverage == 1.0`; per shape, read `correctness_passed` (precision) and `efficiency`
(performance). The single `score` on the summary line is the emitted `aggregate_score`
(`s_t` when Roofline is available, otherwise a legacy diagnostic); `coverage` is reported
separately. Final scoring uses a correctness-coverage gate plus the Roofline performance
metric (see [scoring_spec.md](reference/scoring_spec.md)).

**What a real passing grade looks like.** For the `L2_04_silu_and_mul` reference kernel,
the committed metadata records **every shape `VERIFIED` with `max_abs = 0.0`** (exact match
to the oracle). Performance is measured on a reference CUDA-GPU grade (not the 910B target):
for a small vector op like this the per-shape Roofline `efficiency` is low — on the order of
`0.02` — at a few-microsecond device latency (measured with `do_bench`). Those performance
figures are illustrative reference-daemon numbers — not repo-verifiable, and not 910B
leaderboard scores. Roofline efficiency is judged per op, not against a fixed bar.

### Verify your kernel: precision + performance

Once `npu_triton_impl` runs, `tab run` answers two questions — is the output
**numerically aligned** with the reference, and how does it **perform**.

**Aligning precision.** Each task declares a **tolerance class** (`meta.yaml` /
`contract.md` §Tolerance class — `elementwise` for `L2_04`). Your output is compared to
the PyTorch oracle under that class: float `allclose` with the class's `atol`/`rtol`,
`int_atol` for int8-quant, or ULP for fp8. A shape's `correctness_passed` is `true` when it
is within tolerance; on a numeric miss, `correctness_reason` carries comparator detail —
floats report `max_abs`/`max_rel`, int-atol reports `max_abs > int_atol`, and fp8 reports the
max ordered-code distance vs the ULP cap; other failures report the specific mismatch/error.
The usual fix is to **match the oracle's numeric semantics** — the task's `contract.md`
flags the load-bearing parts (for `L2_04`: *"the cast-back to input dtype before the multiply
is load-bearing"* — compute SiLU in fp32, cast back, then multiply).
→ [tolerance_policy.md](reference/tolerance_policy.md), [correctness_model.md](reference/correctness_model.md).

**Confirming performance.** Per shape, `efficiency = min(cap, t_ref / t_npu)` — the
cost-model reference time over your measured device time (`t_npu` = median measured latency;
the GPU rank helper uses `triton.testing.do_bench` for its reference-proxy timings). Higher
`efficiency` is closer to the Roofline; `measured_ns` / `median` / `p99`
are the raw latency, and `s_t` aggregates the throughput score.
→ [timing_arbitration.md](reference/timing_arbitration.md), [scoring_spec.md](reference/scoring_spec.md).

> Precision aligns on **any working backend** — a real NPU/GPU, or the CPU interpreter for
> interpreter-clean kernels. Authoritative **performance** (Roofline efficiency vs the 910B
> peak) needs a real Ascend 910B (README §Status). Non-910B timings, when available, are
> reference proxies, not leaderboard numbers; CPU-only runs usually leave performance fields
> `null`/`error`.

### Run on the target NPU (Ascend 910B)

`tab run` and `run-suite` **default to `--backend npu`** — the NPU path runs your
triton-ascend kernel on a real Atlas 910B. (`--backend gpu` is a CUDA self-check path, not an
authoritative 910B run.)

**Environment.** A real NPU run needs `torch_npu` + `triton-ascend` + CANN with the driver
environment (`LD_LIBRARY_PATH`, `ASCEND_TOOLKIT_HOME`, `ASCEND_OPP_PATH`). The reference
versions are pinned in [`docker/Dockerfile.npu`](../docker/Dockerfile.npu) (CANN `8.5.0`,
`torch_npu 2.8.0.post1`, `triton_ascend 3.2.0`); bare-metal 910B setup lives outside this repo.

**No NPU means degraded, not a GPU fallback.** Without `torch_npu` the NPU backend cannot
place tensors on device; the isolated runner keeps them on CPU, device timing is unavailable,
and the result is marked `npu_antihack_unverified` — not a valid NPU proof. A real result needs
a real 910B; it does **not** silently fall back to the GPU.

**You cannot fake it with torch.** On the NPU a real triton-ascend kernel must launch a device
kernel; the runner parses the NPU trace (parent-side) for a kernel event. A torch-only path that
launches no kernel is flagged (`fallback_detected` / `torch_fallback`), which forces
`correctness_status = "fail"` and collapses the task to ~zero in final scoring.

**Results are read the same way** (`shape_results[]`, above). On the 910B the Roofline
`efficiency` is normalised against the packaged 910B peak table
(`ascendbench/policies/peak_table_910b.toml`). Some peak entries are estimates (`bf16_cube`
architectural, `vector` academic, `interconnect` a placeholder), so absolute efficiency is
gated by `absolute_verified` / `roofline_data_complete` — treat `efficiency` as a normalised
ranking signal, not a hardware-exact percentage.

**Practical notes.** A real NPU Triton run does a compile/JIT warmup before the graded calls;
`run-suite` also takes `--backend npu`; verified-mode grading of `upstream_oracle_required`
tasks may need their offline golden artifacts.

> **Sourcing.** This section is documented from the code and the shipped NPU runner. The exact
> bare-metal setup and on-device behavior are **not re-verified in this doc pass** (the 910B
> reachable here lacks `torch_npu`).

## 5. Grade the whole suite

```bash
tab run-suite --solutions-root solutions/    # one solution.py per task under the root
```

## 6. Package and submit

```bash
tab package solutions/L2_04_silu_and_mul --out submission.tar.gz
tab validate-submission submission.tar.gz     # validate the bundle locally
tab submit submission.tar.gz --endpoint <url> # validates the package; see below
```

`tab package` warns if `manifest.yaml` is missing (author it against
`ascendbench.manifest.SolutionManifest`); `validate-submission` then enforces
the schema. **`tab submit` validates and reports** — the hosted leaderboard
POST is the roadmap v0.1 service (README §Leaderboard), so no live endpoint
is required to prepare a submission.

## Worked example — CPU smoke

On a **CPU-only machine** you can exercise the workflow, the static gate, and
correctness (the runner keeps tensors on CPU when no accelerator is present);
authoritative **performance** (Roofline) needs the target 910B. Here is the
honest end-to-end on a dev box:

```bash
tab init-solution L2_04_silu_and_mul --out /tmp/demo/sol
```

**(a) A plain-torch attempt is blocked** — proof the anti-gaming gate works:

```text
static_checks_passed: False
findings: "no @triton.jit kernel found", "no tl.load found", "no tl.store found",
          "banned attribute path 'torch.nn.functional.silu'",
          "host-side Tensor BinOp (eager evaluation; use a triton kernel)"
```

**(b) A real Triton kernel passes the static gate**; run it under the Triton
interpreter for a CPU smoke:

```bash
TRITON_INTERPRET=1 tab run L2_04_silu_and_mul --solution /tmp/demo/sol/solution.py
# static_ok=True   correctness_status=fail   runtime_status=error   coverage=0.00
```

Here the static gate passes (`static_checks_passed: true` — a valid Triton
kernel), but this minimal kernel errored under the CPU interpreter
(`runtime_status=error`, so `correctness_status=fail`, `coverage=0`).
Correctness grading itself does run on CPU: a kernel that executes cleanly
under `TRITON_INTERPRET=1` yields a real `correctness_status=ok`. The smoke's
job is to validate the workflow + the static gate; authoritative
**performance** (Roofline) still needs real Ascend 910B hardware (README
§Status).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No such task` / missing `tasks/` | You installed the wheel; `tasks/` needs a **clone or release tarball**. |
| `IsADirectoryError` on `--solution` | Pass the **`solution.py` file path**, not the directory. |
| `static_checks_passed: false`, "no @triton.jit kernel" | Solution must be a **real Triton kernel**, not PyTorch. |
| `runtime_status: error` under `TRITON_INTERPRET=1` | The kernel didn't execute under the CPU interpreter; simplify it or run on the target NPU/GPU. Correctness itself does grade on CPU for interpreter-clean kernels. |
| `NotImplementedError: implement npu_triton_impl` | The scaffold stub — implement the entrypoint. |

## Next

- **How the bench works** → [ARCHITECTURE.md](ARCHITECTURE.md)
- **Design rationale** → [DESIGN.md](DESIGN.md)
- **Deep reference** (correctness, scoring, shapes, integrity, provenance) →
  [docs/reference/](reference/)
