"""Streaming input materialisation pins.

The runner used to build every graded shape's ``1 + n_warmup + n_repeat``
inputs up-front and hold them all in parent host memory
(O(shapes x 111)); large-shape tasks OOMed the host memcg and died as
exit 137 after a long silent materialisation. The lazy payload forms
(`LazyTrials` per-trial sequences, callable correctness inputs) make
`run_two_pass` / `run_two_pass_lazy` build / serialise / release ONE
input at a time while keeping the two-subprocess isolation, per-trial
fresh-input seeds, runner-secret sampled indices and offline-golden
sampled-slot overwrite semantics identical.

Dev machine has no NPU — these pins verify parent-side memory discipline,
wire equivalence with the materialised path, and the seed/overwrite
contract, not NPU numeric reality.
"""
from __future__ import annotations

from pathlib import Path

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

_SOLUTION_SRC = '''
def npu_triton_impl(inputs):
    x = inputs["x"]
    w = inputs["w"]
    eps = inputs["eps"]
    return {"y": (x + w) * (1.0 + eps)}
'''

_SHAPE_IDS = ["P1", "H1"]


def _write_solution(tmp_path: Path) -> Path:
    sol = tmp_path / "solution.py"
    sol.write_text(_SOLUTION_SRC)
    return sol


def _build(sid: str, seed: int):
    return _inputs.build_inputs(_SHAPES_DOC, sid, seed=seed)


def _timing_payload(shape_id: str, n_warmup: int, n_repeat: int) -> list:
    return (
        [_build(shape_id, 1 + w) for w in range(n_warmup)]
        + [_build(shape_id, 1000 + t) for t in range(n_repeat)]
    )


class _WireSpy:
    """Wrap `_materialise_inputs` and record every serialised payload.

    Records the materialisation ORDER and the exact bytes written, so the
    lazy and materialised paths can be diffed on their child-facing wire
    output (per-trial refs, trial indexing, input bytes, sidecar bytes).
    """

    def __init__(self, monkeypatch):
        self.captured = []
        orig = pi._materialise_inputs

        def spy(built, dst_dir, ref):
            tp, sp = orig(built, dst_dir, ref)
            self.captured.append(
                (ref, Path(tp).read_bytes(), Path(sp).read_bytes()))
            return tp, sp

        monkeypatch.setattr(pi, "_materialise_inputs", spy)


def _tracked_build_factory(live):
    class Tracked(dict):
        def __init__(self, src):
            super().__init__(src)
            live["n"] += 1
            live["peak"] = max(live["peak"], live["n"])

        def __del__(self):
            live["n"] -= 1

    def build_input(sid, seed):
        return Tracked(_build(sid, seed))

    return build_input


# ---------------------------------------------------------------------------
# Memory discipline — the whole point of the streaming path.
# ---------------------------------------------------------------------------

def test_lazy_path_holds_at_most_one_built_input(tmp_path):
    """In-process-oracle path: while the lazy two-pass run materialises its
    inputs, no more than ONE built inputs dict is alive in the parent at
    any moment; after the call, none remain."""
    live = {"n": 0, "peak": 0}
    build_input = _tracked_build_factory(live)

    sol = _write_solution(tmp_path)
    two = pi.run_two_pass_lazy(
        sol, _SHAPE_IDS, build_input, n_warmup=2, n_repeat=4, sample_size=2)

    assert two.correctness.per_shape["P1"].ran
    assert two.timing.per_shape["P1"].ran
    assert live["peak"] == 1, (
        f"peak of {live['peak']} concurrently-built inputs — the "
        f"streaming materialisation must serialise+release one input "
        f"before building the next")
    assert live["n"] == 0, "built inputs leaked past the end of the run"


def test_offline_golden_path_retains_only_the_fixed_inputs(tmp_path):
    """Offline-golden path: the per-shape seed-0 fixed input is the only
    deliberate retention (the golden the sampled trials must run on), so
    the peak is n_shapes fixed + one transiently-built trial — never the
    {shapes x trials} residence set."""
    live = {"n": 0, "peak": 0}
    build_input = _tracked_build_factory(live)

    sol = _write_solution(tmp_path)
    two = pi.run_two_pass_lazy(
        sol, _SHAPE_IDS, build_input,
        n_warmup=2, n_repeat=4, sample_size=2,
        golden_fixed_builder=lambda sid: build_input(sid, 0))

    assert two.correctness.per_shape["P1"].ran
    assert two.timing.per_shape["P1"].ran
    assert live["peak"] == len(_SHAPE_IDS) + 1, (
        f"peak {live['peak']} — expected {len(_SHAPE_IDS)} retained fixed "
        f"inputs + 1 transient build")
    # The retained fixed inputs are released once the run returns.
    assert live["n"] == 0


# ---------------------------------------------------------------------------
# Wire equivalence with the materialised legacy path.
# ---------------------------------------------------------------------------

def test_lazy_path_wire_output_matches_materialised_path(
        tmp_path, monkeypatch):
    """For the same sampling seed, the lazy path serialises the SAME byte
    stream (refs, trial indexing, per-trial input bytes, golden-fixed
    overwrite positions) as the legacy materialised-list path."""
    monkeypatch.setattr(pi, "_fresh_sampling_seed", lambda: 424242)
    n_warmup, n_repeat, sample_size = 2, 6, 2

    def run_legacy(sol):
        correctness = {sid: _build(sid, 0) for sid in _SHAPE_IDS}
        timing = {sid: _timing_payload(sid, n_warmup, n_repeat)
                  for sid in _SHAPE_IDS}
        golden_fixed = {sid: _build(sid, 0) for sid in _SHAPE_IDS}
        return pi.run_two_pass(
            sol, correctness, timing,
            n_warmup=n_warmup, sample_size=sample_size,
            golden_fixed_inputs=golden_fixed)

    sol = _write_solution(tmp_path)

    spy_legacy = _WireSpy(monkeypatch)
    two_legacy = run_legacy(sol)
    wire_legacy = list(spy_legacy.captured)

    spy_lazy = _WireSpy(monkeypatch)
    two = pi.run_two_pass_lazy(
        sol, _SHAPE_IDS, _build,
        n_warmup=n_warmup, n_repeat=n_repeat, sample_size=sample_size,
        golden_fixed_builder=lambda sid: _build(sid, 0))
    wire_lazy = list(spy_lazy.captured)

    assert two_legacy.timing.per_shape["H1"].ran
    assert two.correctness.per_shape["H1"].ran
    assert two.timing.per_shape["P1"].durations
    assert wire_lazy == wire_legacy, (
        "the streaming path must serialise exactly the same per-trial "
        "inputs in the same order as the materialised path (seeds, trial "
        "indices and golden-fixed overwrite positions included)")

    # Sanity: the golden-fixed overwrite actually happened — with the
    # sampling seed pinned, some measured trial carried the seed-0 input.
    golden = {(ref, tb, sb) for ref, tb, sb in wire_legacy if ref == "P1"}
    assert len(golden) == 1
    golden_tb_sb = next(iter(golden))[1:]
    assert any(ref.startswith("P1.t") and (tb, sb) == golden_tb_sb
               for ref, tb, sb in wire_legacy), (
        "no sampled measured trial carried the fixed seed-0 input")


# ---------------------------------------------------------------------------
# Seed + sampled-overwrite contract of the lazy builder itself.
# ---------------------------------------------------------------------------

def test_lazy_builder_seed_sequence_and_golden_retention(
        tmp_path, monkeypatch):
    """Golden path: the per-shape fixed input is built exactly ONCE (up
    front), warm trials are built at warm_seed+w and only the
    NON-sampled measured trials at meas_seed+t — the sampled-overwrite
    slots never build a fresh input at all."""
    monkeypatch.setattr(pi, "_fresh_sampling_seed", lambda: 777)
    n_warmup, n_repeat, sample_size = 3, 7, 2
    calls = []
    golden_calls = []

    def build_input(sid, seed):
        calls.append((sid, seed))
        return _build(sid, seed)

    def golden_fixed_builder(sid):
        golden_calls.append(sid)
        return _build(sid, 0)

    sol = _write_solution(tmp_path)
    pi.run_two_pass_lazy(
        sol, _SHAPE_IDS, build_input,
        n_warmup=n_warmup, n_repeat=n_repeat, sample_size=sample_size,
        warm_seed=1, meas_seed=1000,
        golden_fixed_builder=golden_fixed_builder)

    # Fixed inputs built once per shape, up front, in shape order.
    assert golden_calls == list(_SHAPE_IDS)

    # No correctness-pass seed-0 build via build_input (the correctness
    # pass reuses the retained fixed inputs).
    for sid in _SHAPE_IDS:
        per_shape = [seed for s, seed in calls if s == sid]
        # Warm-up: exactly n_warmup warm seeds, ascending.
        warm = [seed for seed in per_shape if seed < 1000]
        assert warm == [1 + w for w in range(n_warmup)]
        # Measured: only the non-sampled fresh meas seeds, ascending.
        meas = [seed for seed in per_shape if seed >= 1000]
        assert len(meas) == n_repeat - sample_size
        assert meas == sorted(meas)


def test_lazy_path_without_golden_fix_uses_only_fresh_seeds(tmp_path):
    """No golden_fixed_builder ⇒ every per-trial input is freshly seeded —
    the anti-cache defence stays fully intact (in-process-oracle path)."""
    calls = []

    def build_input(sid, seed):
        calls.append((sid, seed))
        return _build(sid, seed)

    sol = _write_solution(tmp_path)
    pi.run_two_pass_lazy(
        sol, _SHAPE_IDS, build_input, n_warmup=2, n_repeat=4)

    assert calls[:2] == [("P1", 0), ("H1", 0)]
    for sid in _SHAPE_IDS:
        per_shape = [seed for s, seed in calls[2:] if s == sid]
        assert per_shape == ([1 + w for w in range(2)]
                             + [1000 + t for t in range(4)])
