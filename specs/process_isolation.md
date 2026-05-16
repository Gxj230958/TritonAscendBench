# Process Isolation Architecture

> v0 — per v4.13 §5b.1 + §5b.1a + §5b.2 + R11-P0-1/2 + R12-P0-1/2 + R15-P1-1/2

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
│  • mount: kernel.py, meta.yaml, shapes.yaml,   │
│           inputs.safetensors                    │
│  • NOT mount: oracle_pytorch.py                 │
│  • NOT mount: golden_outputs (R11-P0-1!)        │
│  • NOT mount: ascendc_baseline.yaml             │
│  • cgroup: 32 GB mem, 16 core                   │
│  • Trusted shim wraps solution.npu_triton_impl  │
│  • Writes output.safetensors (NOT .pt — R12)    │
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

Per R12-P0-2 + R16-P1: correctness pass and timing pass must NOT share state.

Otherwise: candidate could pre-compute answers in correctness pass, cache in `/tmp` or device memory, then look them up in timing pass (paying ~0 compute cost in measured trials).

Mitigations enforced:
- Different process per phase
- SIGKILL after each phase
- Full cache scrub between phases (`/tmp/triton-ascend-cache/*`, `/tmp/scratch`, device memory)

## Why Safetensors Output (NOT Pickle)

Per R12-P0-1: `.pt` / `.pkl` are pickle-based → arbitrary code execution on load.

If trusted runner does `torch.load(candidate_out.pt)`, candidate can craft malicious pickle.

Therefore:
- Candidate writes `output.safetensors` (data-only format)
- Runner reads with `safetensors.torch.load_file()` (no code execution)
- Output metadata (dtype/stride/storage_offset/data_ptr/alias) signed by trusted shim, sent via runner-controlled pipe — checked BEFORE serialization

## Trusted Shim

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

## Sandbox Restrictions

```bash
docker run --rm \
    --network=none \                      # R6-4
    --read-only \                          # R6-4
    --tmpfs /tmp:rw,size=256m \           # writable scratch only
    --device=/dev/davinci0 \              # NPU device allowlist (P2-2)
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --cap-drop=ALL \
    --security-opt seccomp=/runner/seccomp.json \
    --cgroup-rules='c 200:0 r' \
    --memory=32g --cpus=16 \              # P2-2 cgroup
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
- `dlopen` of `.so` without runner broker signature (R15-P1-1)
- `torch.npu.Event` / `time.perf_counter` / `LD_PRELOAD`
- Reading `/runner_shim/` (shim source)
- Reading `oracle_pytorch.py` or `ascendc_baseline.yaml`
- Network egress to ANY endpoint (NO upload exception — runner does upload from outside sandbox)
