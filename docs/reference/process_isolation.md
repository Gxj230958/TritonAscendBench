# Process Isolation Architecture

> Two isolation layers. The **shipped** dev-runnable layer
> (`ascendbench/integrity/process_isolation.py`) enforces the host-independent
> subset on any machine: subprocess isolation, safetensors-only ABI, and
> oracle/golden/shapes non-exposure. The **hardened verified-runner**
> confinement pictured below (Docker / seccomp / cgroup / NPU device
> allowlist) is the 910B deployment target — future work, see
> `docs/DESIGN.md §6`. Scoring model: `docs/DESIGN.md §4`.

## Three-Process Model

```
┌─── Trusted Runner Process (host) ────────────┐
│  • Loads oracle_pytorch.py                   │
│  • Computes golden_outputs                   │
│  • Holds tolerance_policy, scoring spec      │
│  • Reads sampled outputs from candidate      │
│  • Performs allclose check                   │
└──────────────────────────────────────────────┘
        │ fork() with restricted mount
        ↓
┌─── Candidate Process #1 (correctness sandbox) ─┐
│  • No network, read-only FS, seccomp           │
│  • mount: solution.py + materialised inputs    │
│  • NOT mount: oracle_pytorch.py, shapes.yaml    │
│  • NOT mount: golden_outputs        │
│  • NOT mount: cost-model GPU-measured baseline  │
│  • cgroup: 32 GB mem, 16 core                   │
│  • Trusted shim wraps solution.npu_triton_impl  │
│  • Writes output.safetensors (NOT .pt)          │
└─────────────────────────────────────────────────┘
        │ exit + SIGKILL process group
        ↓ (clear /tmp/triton-ascend-cache, device memory)
┌─── Candidate Process #2 (timing sandbox) ──────┐
│  • Same mounts as Process #1                   │
│  • Trusted shim runs warm-up + measured loops  │
│  • Uses CLOCK_MONOTONIC_RAW (host time)        │
│  • Cross-checks with torch.npu.Event           │
│  • Writes timing + sampled outputs to pipe     │
└─────────────────────────────────────────────────┘
        │ exit + SIGKILL
        ↓
Trusted Runner reads pipe, computes verdict
```

## Why Two Candidate Processes

Correctness pass and timing pass must NOT share state.

Otherwise: candidate could pre-compute answers in correctness pass, cache in `/tmp` or device memory, then look them up in timing pass (paying ~0 compute cost in measured trials).

Mitigations enforced:
- Different process per phase
- SIGKILL after each phase
- Full cache scrub between phases (`/tmp/triton-ascend-cache/*`, `/tmp/scratch`, device memory)

## Why Safetensors Output (NOT Pickle)

`.pt` / `.pkl` are pickle-based → arbitrary code execution on load.

If trusted runner does `torch.load(candidate_out.pt)`, candidate can craft malicious pickle.

Therefore:
- Candidate writes `output.safetensors` (data-only format)
- Runner reads with `safetensors.torch.load_file()` (no code execution)
- In the hardened verified-runner (future work): output metadata (dtype/stride/storage_offset/data_ptr/alias) signed by the trusted shim and sent via a runner-controlled pipe — checked BEFORE serialization

## Trusted Shim (hardened verified-runner — future work)

Runner-controlled binary mounted at `/runner/trusted_shim.so`, candidate cannot modify.

Shim responsibilities:
1. Receive `inputs` (read from /runner/inputs/)
2. Call `solution.npu_triton_impl(inputs)`
3. BEFORE serialization: check `output` dtype/shape/stride/storage_offset/data_ptr
4. Verify `output.data_ptr() != any input.data_ptr()` (no alias to inputs)
5. Send metadata via runner-controlled signed pipe
6. Serialize `output` to `safetensors`
7. Write to `/runner/output.safetensors`
8. Manage CLOCK_MONOTONIC_RAW timing windows
9. Forbid candidate from importing `torch.npu.Event` / `time.perf_counter` / `LD_PRELOAD`

## Sandbox Restrictions (hardened verified-runner — future work)

```bash
docker run --rm \
    --network=none \
    --read-only \
    --tmpfs /tmp:rw,size=256m \           # writable scratch only
    --device=/dev/davinci0 \              # NPU device allowlist
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --cap-drop=ALL \
    --security-opt seccomp=/runner/seccomp.json \
    --cgroup-rules='c 200:0 r' \
    --memory=32g --cpus=16 \              # cgroup limits
    --user=runner:runner \                # non-privileged
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /runner/inputs:/inputs:ro \
    -v /runner/solution:/solution:ro \
    -v /runner/kernel:/kernel:ro \
    -v /runner/output:/output:rw \
    -v /runner/shim:/runner_shim:ro \
    tab-npu:verified candidate_entry.py
```

## Forbidden Inside Candidate Sandbox

- Network (no `--network=none` override)
- File writes outside `/output/`
- `dlopen` of `.so` without runner broker signature
- `torch.npu.Event` / `time.perf_counter` / `LD_PRELOAD`
- Reading `/runner_shim/` (shim source)
- Reading `oracle_pytorch.py`, `golden_outputs`, or the cost-model GPU-measured baseline
- Network egress to ANY endpoint (NO upload exception — runner does upload from outside sandbox)
