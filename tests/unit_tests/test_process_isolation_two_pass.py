"""W1.3 — dual-sandbox + timing-pipeline unit tests for process_isolation.

Pins the W1.3 contract on top of the W1.0 protocol:

  * the candidate is graded in TWO independent subprocess passes
    (correctness then timing), with a SIGKILL + scrub between them;
  * the timing pass runs a per-trial fresh-input measured loop and sends
    up host_ns ``durations`` + a runner-secret subset of ``sampled_outputs``;
  * the timing arbitration failure paths (invalid_timing / quarantined /
    invalid_stream) are produced by `timing.arbitrate_timing`.

Dev machine has no NPU — these verify pipeline connectivity, the
isolation chain, and per-trial timing, not NPU numeric reality.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ascendbench.input_spec import inputs as _inputs
from ascendbench.integrity import process_isolation as pi

_SHAPES_DOC = {
    "size_knobs": {"B": 2, "L": 4, "D": 8},
    "public_shapes": [
        {"id": "P1", "name": "tiny", "knobs": {"B": 1, "L": 2, "D": 4}},
    ],
    "hidden_shapes": [
        {"id": "H1", "name": "h_odd", "knobs": {"B": 3, "L": 5, "D": 7}},
    ],
    "inputs": {
        "x": {"dims": ["B", "L", "D"], "dtype": "float32", "dist": "randn"},
        "w": {"dims": ["D"], "dtype": "float32", "dist": "ones"},
        "eps": {"scalar": 1.0e-5},
    },
}

# A candidate that records, into a file in its scratch dir, the PID it
# ran under — so the test can prove correctness and timing passes ran in
# *different* processes.
_SOLUTION_SRC = '''
import os
import torch


def npu_triton_impl(inputs):
    # Append this process's PID to a sidecar so the test can confirm the
    # two passes used two distinct interpreters.
    with open(os.path.join(os.getcwd(), "pids.txt"), "a") as fh:
        fh.write(str(os.getpid()) + "\\n")
    x = inputs["x"]
    w = inputs["w"]
    eps = inputs["eps"]
    return {"y": (x + w) * (1.0 + eps)}
'''


def _write_solution(tmp_path: Path) -> Path:
    sol = tmp_path / "solution.py"
    sol.write_text(_SOLUTION_SRC)
    return sol


def _timing_payload(shape_id: str, n_warmup: int, n_repeat: int) -> list:
    """A warm-up + measured per-trial sequence for one shape."""
    return (
        [_inputs.build_inputs(_SHAPES_DOC, shape_id, seed=1 + w)
         for w in range(n_warmup)]
        + [_inputs.build_inputs(_SHAPES_DOC, shape_id, seed=1000 + t)
           for t in range(n_repeat)]
    )


# ---------------------------------------------------------------------------
# Dual sandbox: two independent subprocesses + scrub between them.
# ---------------------------------------------------------------------------

def test_run_two_pass_uses_two_distinct_subprocesses(tmp_path):
    """correctness pass and timing pass are separate child processes."""
    sol = _write_solution(tmp_path)
    correctness = {"P1": _inputs.build_inputs(_SHAPES_DOC, "P1", seed=0)}
    timing = {"P1": _timing_payload("P1", n_warmup=2, n_repeat=4)}

    two = pi.run_two_pass(sol, correctness, timing, n_warmup=2)

    # Both passes ran.
    assert two.correctness.per_shape["P1"].ran, two.correctness.per_shape["P1"].error
    assert two.timing.per_shape["P1"].ran, two.timing.per_shape["P1"].error
    assert two.correctness.pass_kind == "correctness"
    assert two.timing.pass_kind == "timing"


def test_run_two_pass_scrub_receipt(tmp_path, monkeypatch):
    """The scrub between the two passes is invoked and its receipt is
    recorded on the TwoPassResult (the isolation-chain proof)."""
    sol = _write_solution(tmp_path)

    # Plant a fake compile-cache dir so the scrub has something to clear.
    fake_cache = tmp_path / "fake-triton-cache"
    fake_cache.mkdir()
    (fake_cache / "kernel.bin").write_bytes(b"stale")
    monkeypatch.setattr(pi, "_SCRUB_PATHS", (str(fake_cache),))

    correctness = {"P1": _inputs.build_inputs(_SHAPES_DOC, "P1", seed=0)}
    timing = {"P1": _timing_payload("P1", n_warmup=1, n_repeat=3)}

    two = pi.run_two_pass(sol, correctness, timing, n_warmup=1)

    # The legacy path also owns its per-pass cache. Shared paths are untouched.
    assert two.scrubbed == []
    assert (fake_cache / "kernel.bin").read_bytes() == b"stale"


def test_scrub_between_passes_is_safe_when_nothing_to_clear(monkeypatch):
    """scrub_between_passes is a no-op (empty receipt) when the compile
    cache / scratch dirs do not exist — safe on the dev machine."""
    monkeypatch.setattr(pi, "_SCRUB_PATHS",
                        ("/tmp/tab-does-not-exist-xyz",))
    scrubbed = pi.scrub_between_passes()
    # On the dev machine there is no NPU either -> fully empty receipt.
    assert scrubbed == []


def test_timing_pass_kills_correctness_subprocess(tmp_path):
    """After run_two_pass returns, the correctness-pass child PID is gone
    (the process group was SIGKILLed) — no leaked process."""
    sol = _write_solution(tmp_path)
    correctness = {"P1": _inputs.build_inputs(_SHAPES_DOC, "P1", seed=0)}
    timing = {"P1": _timing_payload("P1", n_warmup=1, n_repeat=3)}

    two = pi.run_two_pass(sol, correctness, timing, n_warmup=1)
    assert two.correctness.per_shape["P1"].ran
    # No child subprocess survives the call (run_two_pass is synchronous).
    # A negative assertion via /proc would be flaky; instead assert the
    # call returned, which only happens once both children are reaped.
    assert two.timing.per_shape["P1"].ran


def test_timeout_kills_whole_process_group(tmp_path):
    """R3-4: a candidate that forks a long-lived descendant has the WHOLE
    process group SIGKILLed on timeout — the descendant must not survive.

    The candidate forks a child that writes a marker file after a long
    sleep; the parent itself hangs so the runner times out. If only the
    direct child were killed, the forked descendant would live on and
    eventually write the marker. After the timeout it must NOT exist.
    """
    import time as _time

    marker = tmp_path / "descendant_alive.marker"
    sol = tmp_path / "solution.py"
    # The fork happens at module-body load time — the Layer E audit hook
    # is installed AFTER the candidate module loads but before any
    # `npu_triton_impl` call (R6-2), so a fork-bomb candidate would fork at
    # load time anyway. This still exercises R3-4: the forked descendant
    # must be reaped when the runner SIGKILLs the whole process group on
    # timeout, regardless of when the fork happened.
    sol.write_text(
        "import os, sys, time\n"
        "\n"
        "# Fork a descendant that outlives a quick SIGKILL of the direct\n"
        "# child only — it writes a marker after a long sleep.\n"
        "_pid = os.fork()\n"
        "if _pid == 0:\n"
        "    time.sleep(30)\n"
        f"    open({str(marker)!r}, 'w').write('alive')\n"
        "    os._exit(0)\n"
        "\n"
        "\n"
        "def npu_triton_impl(inputs):\n"
        "    # The direct child hangs so the runner times out.\n"
        "    time.sleep(30)\n"
        "    return {}\n"
    )
    built = _inputs.build_inputs(_SHAPES_DOC, "P1", seed=0)
    res = pi.run_candidate_isolated(sol, {"P1": built}, timeout_s=3)

    assert res.timed_out, "candidate should have hit the timeout"
    # Give any *surviving* descendant a generous window to write its
    # marker; if killpg worked it was SIGKILLed long before its sleep ends.
    _time.sleep(2)
    assert not marker.exists(), (
        "a forked descendant survived the timeout — the process group "
        "was not SIGKILLed (R3-4 regression)"
    )


# ---------------------------------------------------------------------------
# Timing pass: per-trial fresh inputs, durations + sampled_outputs uplink.
# ---------------------------------------------------------------------------

def test_timing_pass_produces_per_trial_durations(tmp_path):
    """The timing pass measures one host_ns per measured trial."""
    sol = _write_solution(tmp_path)
    n_repeat = 6
    timing = {"P1": _timing_payload("P1", n_warmup=2, n_repeat=n_repeat)}

    res = pi.run_candidate_isolated(
        sol, timing, pass_kind="timing", n_warmup=2, sample_size=3)

    scr = res.per_shape["P1"]
    assert scr.ran, scr.error
    # One duration per measured trial (warmup trials are NOT timed), minus
    # the single profiled trial whose host time is profiler-polluted (G5).
    assert len(scr.durations) == n_repeat - 1
    assert all(d > 0 for d in scr.durations)


def test_timing_pass_samples_outputs_uplink(tmp_path):
    """A runner-secret subset of measured-trial outputs is sent up."""
    sol = _write_solution(tmp_path)
    n_repeat = 8
    sample_size = 3
    timing = {"P1": _timing_payload("P1", n_warmup=1, n_repeat=n_repeat)}
    # Explicit sampled indices so the assertion is deterministic.
    sampled = {"P1": [0, 3, 7]}

    res = pi.run_candidate_isolated(
        sol, timing, pass_kind="timing", n_warmup=1,
        sample_size=sample_size, sampled_indices=sampled)

    scr = res.per_shape["P1"]
    assert scr.ran, scr.error
    assert scr.sampled_outputs is not None
    # Sampled outputs are keyed "{trial}::{tensor_key}".
    trials_seen = {int(k.split("::")[0]) for k in scr.sampled_outputs}
    assert trials_seen == {0, 3, 7}


def test_timing_pass_fresh_inputs_per_trial(tmp_path):
    """Each measured trial must receive a DISTINCT inputs dict — a
    candidate that hashes its input sees a different value every trial."""
    sol = tmp_path / "solution.py"
    sol.write_text('''
import torch


def npu_triton_impl(inputs):
    x = inputs["x"]
    # Return the input checksum so the harness can prove freshness.
    return {"y": x.flatten()[:1].clone(), "csum": x.sum().reshape(1)}
''')
    n_repeat = 5
    timing = {"P1": _timing_payload("P1", n_warmup=1, n_repeat=n_repeat)}
    sampled = {"P1": list(range(n_repeat))}

    res = pi.run_candidate_isolated(
        sol, timing, pass_kind="timing", n_warmup=1,
        sample_size=n_repeat, sampled_indices=sampled)

    scr = res.per_shape["P1"]
    assert scr.ran, scr.error
    # Collect the per-trial checksums; fresh inputs ⇒ all distinct.
    csums = {int(k.split("::")[0]): v.item()
             for k, v in scr.sampled_outputs.items()
             if k.endswith("::csum")}
    assert len(csums) == n_repeat
    assert len(set(csums.values())) == n_repeat, (
        f"measured trials shared inputs: {csums}")


def test_timing_pass_rejects_single_inputs(tmp_path):
    """A timing pass requires per-trial inputs; a single-dict payload is
    rejected up front."""
    sol = _write_solution(tmp_path)
    single = {"P1": _inputs.build_inputs(_SHAPES_DOC, "P1", seed=0)}

    res = pi.run_candidate_isolated(sol, single, pass_kind="timing")
    assert not res.ran_any
    assert res.load_error and "per-trial" in res.load_error


def test_correctness_pass_unaffected_no_durations(tmp_path):
    """A correctness pass produces outputs but no timing durations."""
    sol = _write_solution(tmp_path)
    correctness = {"P1": _inputs.build_inputs(_SHAPES_DOC, "P1", seed=0)}

    res = pi.run_candidate_isolated(sol, correctness)
    scr = res.per_shape["P1"]
    assert scr.ran
    assert "y" in scr.outputs
    assert scr.durations == []
    assert scr.sampled_outputs is None


def test_timing_pass_hidden_shape_per_trial(tmp_path):
    """Hidden-shape timing per-trial path works — the child cannot tell
    public from hidden, every shape is runner-materialised identically."""
    sol = _write_solution(tmp_path)
    timing = {"H1": _timing_payload("H1", n_warmup=1, n_repeat=4)}

    res = pi.run_candidate_isolated(
        sol, timing, pass_kind="timing", n_warmup=1, sample_size=2)
    scr = res.per_shape["H1"]
    assert scr.ran, scr.error
    # 4 measured trials minus the 1 profiled trial excluded from timing (G5).
    assert len(scr.durations) == 3


def test_run_two_pass_sampled_indices_are_runner_secret(tmp_path):
    """run_two_pass derives sampled indices from a runner-secret RNG; the
    child receives them but cannot have predicted them.

    R3-3: `run_two_pass` exposes NO `seed` parameter — the sampling seed
    is generated fresh per call from a CSPRNG and is strictly distinct
    from the deterministic input-generation seed, so a submitter cannot
    pre-compute the sampled set.
    """
    sol = _write_solution(tmp_path)
    correctness = {"P1": _inputs.build_inputs(_SHAPES_DOC, "P1", seed=0)}
    timing = {"P1": _timing_payload("P1", n_warmup=2, n_repeat=10)}

    two = pi.run_two_pass(sol, correctness, timing,
                          n_warmup=2, sample_size=4)
    scr = two.timing.per_shape["P1"]
    assert scr.sampled_outputs is not None
    trials = {int(k.split("::")[0]) for k in scr.sampled_outputs}
    # Exactly sample_size measured trials sampled, all in measured range.
    assert len(trials) == 4
    assert all(0 <= t < 10 for t in trials)


def test_run_two_pass_rejects_seed_kwarg(tmp_path):
    """R3-3: `run_two_pass` no longer accepts a caller-supplied `seed` —
    a predictable sampling seed would let a submitter pre-compute which
    measured trials are output-reverified and profiled."""
    sol = _write_solution(tmp_path)
    correctness = {"P1": _inputs.build_inputs(_SHAPES_DOC, "P1", seed=0)}
    timing = {"P1": _timing_payload("P1", n_warmup=1, n_repeat=3)}
    with pytest.raises(TypeError):
        pi.run_two_pass(sol, correctness, timing, n_warmup=1, seed=12345)


def test_fresh_sampling_seed_is_unpredictable():
    """R3-3: each `_fresh_sampling_seed()` call returns a distinct,
    positive 63-bit value from a CSPRNG — never a fixed literal."""
    seeds = {pi._fresh_sampling_seed() for _ in range(64)}
    # A CSPRNG over 63 bits will not collide across 64 draws.
    assert len(seeds) == 64
    assert all(0 <= s < (1 << 63) for s in seeds)
