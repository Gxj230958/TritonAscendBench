"""Cross-test isolation for the Phase A' resolver / provenance suite.

WHY THIS EXISTS (confirmed cross-test global-state pollution bug)
-----------------------------------------------------------------
Several test files load the production scripts as fresh, file-local
module instances and register them under fixed ``sys.modules`` keys::

    pap = importlib.util.module_from_spec(...)   # file-local instance
    sys.modules["phase_a_prime"] = pap
    rue = importlib.util.module_from_spec(...)   # file-local instance
    sys.modules["resolve_upstream_entrypoints"] = rue

The ``sys.modules[...]`` assignment at *import (collection)* time is
benign, but a few tests RE-register a brand-new module instance at
*run* time, e.g.:

  * ``test_phase_a_prime_s2_soundness.py``'s ``_rue()`` helper, and
  * ``test_pa_fix1_*``'s inline ``_rue`` reload, and
  * ``test_resolve_upstream_entrypoints.py::test_r9_fix2_*`` part 3
    (re-imports a fresh ``phase_a_prime``).

``phase_a_prime._load_resolver_module()`` resolves the resolver via
``sys.modules["resolve_upstream_entrypoints"]``. So once an *earlier*
test clobbers that key with its own instance, a *later* test's
``monkeypatch.setattr(rue, ...)`` (which targets the later test file's
own module-level ``rue``) no longer affects the instance the
production code actually uses -- the patched ``build_infer_state`` /
``run_resolver`` spies are silently bypassed. This made
``test_unverifiable_before_missing_cache_no_regen`` pass in
isolation but FAIL in the full suite (its part-3
``pap_called["run_resolver"] == 0`` assertion fired because the
unpatched real ``build_infer_state`` saw a clean+pinned ``infer/`` and
let the missing-cache regen path run).

Secondary leak: ``phase_a_prime._RESOLVER_CACHE_MEM`` is a process
-global memo of loaded resolver verdicts. A prior provenance test can
populate it on a shared module instance; a later test that does not
reset the SAME instance would consume a stale memo.

THE FIX (shared, order-independent -- not a one-test band-aid)
--------------------------------------------------------------
An autouse fixture that, BEFORE every test in this directory, pins
``sys.modules["phase_a_prime"]`` / ``sys.modules["resolve_upstream_
entrypoints"]`` back to the *current test file's own* module-level
instances (``request.module.pap`` / ``request.module.rue`` when the
file defines them) and clears the ``_RESOLVER_CACHE_MEM`` memo on every
loaded ``phase_a_prime`` instance. AFTER the test it restores the
pre-test ``sys.modules`` entries so collection-time state is never
permanently corrupted. This makes EVERY provenance/resolver test
operate on the exact module instances its own ``monkeypatch`` calls
target, regardless of suite order -- the shared-state leak is fixed at
the isolation layer for the whole directory, so no provenance test can
regress on ordering again.

This is purely a TEST-ISOLATION fix: the production
``phase_a_prime`` / ``resolve_upstream_entrypoints`` code is correct;
the leak was entirely the tests' ``sys.modules`` surgery being
clobbered across files. No production code is changed.
"""
from __future__ import annotations

import sys

import pytest


@pytest.fixture
def tiny_eval_fixture(tmp_path, monkeypatch):
    """Self-contained grading inputs and CPU mock execution; no private solution."""
    import hashlib

    import torch
    import yaml

    from ascendbench.corpus.dataset import TaskDef
    from ascendbench.integrity import process_isolation as pi

    source = (
        "import triton\nimport triton.language as tl\n"
        "@triton.jit\n"
        "def kernel(x, y, BLOCK: tl.constexpr):\n"
        "    i = tl.arange(0, BLOCK)\n"
        "    tl.store(y + i, tl.load(x + i) + 1.0)\n"
        "def npu_triton_impl(inputs):\n"
        "    x = inputs['x']\n"
        "    y = x.new_empty(x.shape)\n"
        "    kernel[(1,)](x, y, BLOCK=4)\n"
        "    return {'y': y}\n"
    )
    shapes = {
        "size_knobs": {"N": 4},
        "public_shapes": [{"id": "P1", "knobs": {"N": 4}}],
        "hidden_shapes": [{"id": "H1", "knobs": {"N": 6}}],
        "inputs": {"x": {"dims": ["N"], "dtype": "float32", "dist": "randn"}},
    }
    (tmp_path / "shapes.yaml").write_text(yaml.safe_dump(shapes))
    (tmp_path / "oracle_pytorch.py").write_text("def run(inputs):\n    return {'y': inputs['x'] + 1.0}\n")
    solution = tmp_path / "solution.py"
    solution.write_text(source)
    task = TaskDef(
        task_id="SYN_cpu_eval", level=1, task_dir=tmp_path,
        meta={"tolerance_class": "elementwise", "oracle_simplified": "simple_pytorch_ok",
              "runner_checks": {}, "oracle": {"pytorch": "oracle_pytorch.py::run"},
              "param_constraints": {"scaffold": False, "dims": ["N >= 1"],
                                    "dtypes": {"x": "float32"},
                                    "dists": {"x": {"kind": "randn"}},
                                    "scalars": {}, "tensor_invariants": []}},
        public_shapes=[], contract_md="Synthetic CPU grading fixture.",
        kernel_py=source, kernel_py_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )
    task.param_constraints = task.meta["param_constraints"]

    def mock_execution(_solution, correctness_inputs, _timing_inputs, **_kwargs):
        correct = {}
        timed = {}
        for sid, inputs in correctness_inputs.items():
            tensor = inputs["x"]
            assert torch.is_tensor(tensor)
            correct[sid] = pi.ShapeCandidateResult(sid, True, outputs={"y": tensor + 1.0})
            timed[sid] = pi.ShapeCandidateResult(sid, True, durations=[5_000_000.0] * 99)
        return pi.TwoPassResult(
            pi.CandidateRunResult(True, False, None, correct),
            pi.CandidateRunResult(True, False, None, timed, pass_kind="timing"),
            scrubbed=[],
        )

    monkeypatch.setattr(pi, "run_two_pass", mock_execution)
    return task, solution


@pytest.fixture
def tiny_cli_fixture(tiny_eval_fixture, monkeypatch):
    """Exercise the actual argument parser and producer with CPU mock execution."""
    from ascendbench.corpus import dataset
    from ascendbench.engine import cli
    from ascendbench.engine import eval as evaluator

    task, solution = tiny_eval_fixture

    def load_task(task_id):
        assert task_id == task.task_id
        return task

    monkeypatch.setattr(dataset, "load_task", load_task)
    monkeypatch.setattr(evaluator, "load_task", load_task)
    monkeypatch.setattr(dataset, "iter_all_tasks", lambda: [task])

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["tab", *args])
        return cli.main()

    return task, solution, run


_GUARDED_KEYS = ("phase_a_prime", "resolve_upstream_entrypoints")


@pytest.fixture
def pinned_infer_roots(tmp_path, monkeypatch):
    """Opt-in clean Git roots for cache tests; no external checkout required."""
    import subprocess

    roots = []
    pins = {}
    for name in ("vllm", "sglang"):
        root = tmp_path / "infer" / name
        root.mkdir(parents=True)
        (root / "fixture.py").write_text("value = 1\n")
        for args in (
            ["init", "-q"], ["add", "fixture.py"],
            ["-c", "user.name=CPU fixture", "-c", "user.email=fixture@example.invalid",
             "commit", "-qm", "Pinned test fixture"],
        ):
            subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
        pins[name] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        roots.append((name, root))

    def configure(module):
        monkeypatch.setattr(module, "_INFER_ROOTS", tuple(roots))
        monkeypatch.setattr(module, "VLLM_ROOT", roots[0][1])
        monkeypatch.setattr(module, "SGLANG_ROOT", roots[1][1])
        monkeypatch.setattr(module, "expected_infer_pins", lambda: dict(pins))

    return configure


def _reset_resolver_memo() -> None:
    """Clear the ``_RESOLVER_CACHE_MEM`` process-global memo on EVERY
    loaded ``phase_a_prime`` instance (each test file may have its
    own), so a memo populated by a prior test cannot leak into the
    next one regardless of which instance ``sys.modules`` points at."""
    seen: set[int] = set()
    candidates = []
    mod = sys.modules.get("phase_a_prime")
    if mod is not None:
        candidates.append(mod)
    # Also sweep any other phase_a_prime instances still reachable via
    # already-imported test modules' module-level ``pap`` binding.
    # Exception-safe: some modules use a ``__getattr__`` that can raise
    # on an unknown attribute -- never let that break an unrelated test.
    for m in list(sys.modules.values()):
        try:
            pap = getattr(m, "pap", None)
            ok = pap is not None and getattr(
                pap, "__name__", None
            ) == "phase_a_prime"
        except Exception:
            continue
        if ok:
            candidates.append(pap)
    for pap in candidates:
        if id(pap) in seen:
            continue
        seen.add(id(pap))
        if hasattr(pap, "_RESOLVER_CACHE_MEM"):
            pap._RESOLVER_CACHE_MEM = None


@pytest.fixture(autouse=True)
def _isolate_resolver_global_state(request):
    """Pin the guarded ``sys.modules`` keys to the running test file's
    own module instances and reset the resolver memo, around every
    test. Restores the pre-test ``sys.modules`` entries afterwards."""
    saved = {k: sys.modules.get(k) for k in _GUARDED_KEYS}

    test_mod = request.module
    # The resolver test file binds ``rue``; the phase_a_prime test
    # files bind ``pap``. Re-pin sys.modules to whatever THIS file
    # registered at its own collection time, undoing any clobber a
    # prior test (in another file, run earlier) performed.
    own_pap = getattr(test_mod, "pap", None)
    if own_pap is not None and getattr(
        own_pap, "__name__", None
    ) == "phase_a_prime":
        sys.modules["phase_a_prime"] = own_pap
    own_rue = getattr(test_mod, "rue", None)
    if own_rue is not None and getattr(
        own_rue, "__name__", None
    ) == "resolve_upstream_entrypoints":
        sys.modules["resolve_upstream_entrypoints"] = own_rue

    _reset_resolver_memo()

    try:
        yield
    finally:
        _reset_resolver_memo()
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
