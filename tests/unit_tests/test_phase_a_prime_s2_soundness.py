"""Phase A' S2 SOUNDNESS tests (codex r2 HOLE A / B / C).

These exercise the *real* (GPU-free) S2 machinery -- S2-smoke runs on
CPU via ``TRITON_INTERPRET``, staging is pure temp-file work, and the
alias comparator is pure logic. Every test below FAILS if the
corresponding fix is reverted (they assert behaviour, not flags).

HOLE A -- S2-smoke must do REAL work + REAL comparison (not set a flag):
  a divergent severed candidate FAILS smoke (BLOCKED before the GPU
  gate); an equivalent one PASSES smoke and then honestly returns
  UNRUN(needs-gpu) (smoke NEVER yields PASS).
HOLE B -- non-circular build-staged-then-promote: build_staged_candidate
  produces a standalone+run() in a TEMP dir WITHOUT touching the real
  kernel.py; promotion is S2-PASS-gated only; end-to-end smoke works
  with a kernel that exposes only the upstream positional entrypoint
  (the old circular flow could never even run).
HOLE C -- alias soundness: storage-identity (not data_ptr) key groups
  same-storage-different-offset views; ALL inputs (not just mutated)
  enter the alias relation so output<->unmodified-input aliasing is
  detected even when values are identical.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location(
    "phase_a_prime", REPO / "scripts" / "phase_a_prime.py"
)
pap = importlib.util.module_from_spec(_spec)
sys.modules["phase_a_prime"] = pap
_spec.loader.exec_module(pap)

_SHAPES = {
    "size_knobs": {"D": 4},
    "public_shapes": [{"id": "P1", "knobs": {"D": 4}}],
    "inputs": {"x": {"dims": ["D"], "dtype": "float32", "dist": "randn"}},
}


def _mk_task(entry_src: str, gt: str = "kernel.py::ent") -> Path:
    """A minimal real task tree: kernel.py exposes ONLY a positional
    upstream entrypoint (NOT run()), meta.yaml names it via
    oracle.gpu_triton, shapes.yaml drives build_inputs."""
    td = Path(tempfile.mkdtemp(prefix="pa_fx_"))
    (td / "kernel.py").write_text(entry_src)
    (td / "meta.yaml").write_text(
        yaml.safe_dump(
            {
                "task_id": "FX",
                "upstream": {"file_sha256": "ORIG_UPSTREAM_SHA"},
                "oracle": {"gpu_triton": gt},
            }
        )
    )
    (td / "shapes.yaml").write_text(yaml.safe_dump(_SHAPES))
    return td


# ───────────────────────── HOLE B ─────────────────────────
def test_build_staged_candidate_never_touches_real_kernel():
    """build_staged_candidate produces standalone+run() into a TEMP
    dir; the real kernel.py is byte-for-byte UNTOUCHED and still has
    NO run() (HOLE B: staging is temp-only)."""
    src = "import torch\ndef ent(x):\n    x.add_(1.0)\n"
    td = _mk_task(src)
    tc = pap.classify_task("FX", src)
    staged = pap.build_staged_candidate(td, tc)

    assert staged.is_file()
    assert staged.resolve() != (td / "kernel.py").resolve()
    # real kernel.py byte-identical, still no run()
    assert (td / "kernel.py").read_text() == src
    assert "def run(" not in (td / "kernel.py").read_text()
    # staged candidate DOES synthesise run(inputs)->dict
    assert "def run(inputs: dict) -> dict:" in staged.read_text()
    assert "ent" in staged.read_text()


def test_promote_refuses_unless_s2_pass_real_kernel_untouched():
    """promote_staged_candidate REFUSES on UNRUN/BLOCKED and leaves the
    real kernel.py untouched (no fake PASS, plan §invariant 3)."""
    src = "import torch\ndef ent(x):\n    x.add_(1.0)\n"
    td = _mk_task(src)
    tc = pap.classify_task("FX", src)
    staged = pap.build_staged_candidate(td, tc)

    for st in ("UNRUN", "BLOCKED"):
        ok, msg = pap.promote_staged_candidate(
            staged, td, pap.S2Result(task_id="FX", status=st), tc
        )
        assert ok is False
        assert "refusing to promote" in msg
        assert (td / "kernel.py").read_text() == src  # untouched

    # A genuine PASS verdict DOES promote (operation wired, atomic).
    ok, msg = pap.promote_staged_candidate(
        staged, td,
        pap.S2Result(task_id="FX", status="PASS", reason="bit-exact",
                     s2_gpu_run_id="rid"),
        tc,
    )
    assert ok is True, msg
    assert "def run(inputs: dict) -> dict:" in (td / "kernel.py").read_text()
    m = yaml.safe_load((td / "meta.yaml").read_text())
    assert m["upstream"]["file_sha256"] == "ORIG_UPSTREAM_SHA"  # preserved
    assert m["phase_a_prime"]["status"] == "PASS"
    assert not (td / "kernel.py.tmp").exists()  # atomic


def test_s2_non_circular_equivalent_smoke_passes_then_unrun(monkeypatch):
    """The OLD flow was circular: S2 needed run() but run() only
    existed AFTER S2-PASS, so S2 could never even run. Now: a kernel
    exposing ONLY the upstream positional entrypoint runs end-to-end
    -- smoke PASSES (real CPU compare) then honestly UNRUN(needs-gpu);
    real kernel.py NEVER gets run() on a no-GPU host."""
    monkeypatch.setattr(pap, "_cuda_available", lambda: False)
    src = "import torch\ndef ent(x):\n    x.add_(1.0)\n"
    td = _mk_task(src)
    res = pap.s2_verify_task(td)

    assert res.smoke_ran is True
    assert res.status == "UNRUN"  # smoke passed, GPU verdict honest
    assert res.status != "PASS"  # smoke NEVER PASS
    assert "S2-smoke passed" in res.reason
    assert "needs-gpu" in res.reason or "cuda" in res.reason.lower()
    # nothing promoted: real kernel.py untouched
    assert (td / "kernel.py").read_text() == src
    assert "def run(" not in (td / "kernel.py").read_text()


def test_no_real_task_kernel_has_run_without_gpu_pass():
    """Hard invariant: NO real task kernel.py exposes top-level run()
    -- we never rewrite a real task without a genuine GPU S2-PASS
    (staging is temp only)."""
    import re

    pat = re.compile(r"^def run\(", re.M)
    offenders = [
        p
        for p in (REPO / "tasks").glob("level*/*/kernel.py")
        if pat.search(p.read_text(encoding="utf-8", errors="replace"))
    ]
    assert offenders == [], offenders


# ───────────────────────── HOLE A ─────────────────────────
def test_smoke_divergent_severed_is_blocked_before_gpu_gate(monkeypatch):
    """A deliberately divergent severed candidate FAILS the real
    CPU/TRITON_INTERPRET smoke comparison => early BLOCKED, never
    reaching (or needing) the GPU gate; real kernel.py untouched."""
    monkeypatch.setattr(pap, "_cuda_available", lambda: False)
    # original mutates +1.0; we override the staged builder to emit a
    # DIVERGENT candidate (+2.0) -- this is the "staged candidate
    # diverges" fixture HOLE B/A asks for.
    src = "import torch\ndef ent(x):\n    x.add_(1.0)\n"
    td = _mk_task(src)

    def _divergent_staged(task_dir, classification, staging_root=None):
        root = Path(tempfile.mkdtemp(prefix="pa_div_"))
        sp = root / "kernel.py"
        sp.write_text(
            "import torch\n"
            "def run(inputs):\n"
            "    x = inputs['x']\n"
            "    x.add_(2.0)  # DIVERGES from upstream +1.0\n"
            "    return {'x': x}\n"
        )
        return sp

    monkeypatch.setattr(pap, "build_staged_candidate", _divergent_staged)
    res = pap.s2_verify_task(td)

    assert res.status == "BLOCKED"
    assert res.status != "PASS"
    assert "S2-smoke MISMATCH" in res.reason
    assert "NEVER PASS" in res.reason or "untouched" in res.reason
    # real kernel.py untouched, nothing promoted
    assert (td / "kernel.py").read_text() == src


def test_smoke_does_real_comparison_not_a_flag():
    """_run_s2_pair (the smoke engine) does REAL capture + REAL strict
    compare on CPU: equivalent => equivalent True, divergent =>
    equivalent False with a bit-exact failure. (The reverted bug just
    set res.smoke_ran=True with NO subprocess and NO comparison.)"""
    orig = Path(tempfile.mkdtemp()) / "kernel.py"
    orig.write_text("import torch\ndef ent(x):\n    x.add_(1.0)\n")

    sev_ok = Path(tempfile.mkdtemp()) / "kernel.py"
    sev_ok.write_text(
        "import torch\ndef run(inputs):\n"
        "    x=inputs['x']\n    x.add_(1.0)\n    return {'x':x}\n"
    )
    eq, why, *_ = pap._run_s2_pair(
        orig, sev_ok, "ent", _SHAPES, ["P1"], 0, False, None, True
    )
    assert eq is True, why

    sev_bad = Path(tempfile.mkdtemp()) / "kernel.py"
    sev_bad.write_text(
        "import torch\ndef run(inputs):\n"
        "    x=inputs['x']\n    x.add_(2.0)\n    return {'x':x}\n"
    )
    eq2, why2, *_ = pap._run_s2_pair(
        orig, sev_bad, "ent", _SHAPES, ["P1"], 0, False, None, True
    )
    assert eq2 is False
    assert "bit-exact" in why2  # real value comparison happened


def test_smoke_uses_triton_interpret_env(monkeypatch):
    """S2-smoke must put TRITON_INTERPRET=1 in the ACTUAL subprocess
    environment (BLOCKER 3 -- codex r3 [confirmed]: the old test only
    spied that interpret=True was *passed*; it would still pass if the
    `env["TRITON_INTERPRET"]="1"` export at phase_a_prime.py were
    deleted). We spy on the real `subprocess.run` and assert the env
    dict the runner is invoked with carries the key/value, then
    delegate to the real run so the comparison is REAL."""
    import subprocess as _sp

    captured: list[tuple[str, dict]] = []
    real_run = _sp.run

    def _spy_run(cmd, *a, **kw):
        # cmd == [python, _pa_subproc_runner.py, --side, <side>, --spec, ..]
        side = cmd[cmd.index("--side") + 1] if "--side" in cmd else "?"
        captured.append((side, dict(kw.get("env") or {})))
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(_sp, "run", _spy_run)
    monkeypatch.setattr(pap, "_cuda_available", lambda: False)
    src = "import torch\ndef ent(x):\n    x.add_(1.0)\n"
    pap.s2_verify_task(_mk_task(src))

    assert captured, "no subprocess was spawned"
    # EVERY spawned side (smoke, no GPU so only smoke runs) must carry
    # TRITON_INTERPRET=1 in its real environment. This FAILS if the
    # env export is removed.
    for side, env in captured:
        assert env.get("TRITON_INTERPRET") == "1", (side, env)


def _load_subproc_runner():
    """Import scripts/_pa_subproc_runner.py as a module (it is not a
    package); cached in sys.modules so repeated calls are cheap."""
    import importlib.util as _ilu

    if "_pa_subproc_runner" in sys.modules:
        return sys.modules["_pa_subproc_runner"]
    rp = REPO / "scripts" / "_pa_subproc_runner.py"
    sp = _ilu.spec_from_file_location("_pa_subproc_runner", rp)
    mod = _ilu.module_from_spec(sp)
    sys.modules["_pa_subproc_runner"] = mod
    sp.loader.exec_module(mod)
    return mod


def _drive_run_side(device: str, monkeypatch, recorder: list):
    """Drive the REAL ``_pa_subproc_runner._run_side`` IN-PROCESS for a
    single side with the given ``device``, recording every
    ``Tensor.to(<dev>)`` device argument. ``torch.cuda.is_available`` is
    forced True (so the runner's CPU-fake guard does not short-circuit)
    and ``Tensor.to`` is stubbed to RECORD the device then return the
    tensor unchanged (no real GPU touched -- the rest of the manifest
    pipeline still runs on CPU tensors). Returns the runner rc."""
    import argparse as _ap
    import json as _json

    runner = _load_subproc_runner()

    work = Path(tempfile.mkdtemp(prefix="pa_moveto_"))
    (work / "kernel.py").write_text(
        "import torch\ndef ent(x):\n    x.add_(1.0)\n"
    )
    (work / "shapes.json").write_text(_json.dumps(_SHAPES))
    spec = {
        "kernel_path": str(work / "kernel.py"),
        "status_json": str(work / "status.json"),
        "shapes_json": str(work / "shapes.json"),
        "out_dir": str(work),
        "shape_ids": ["P1"],
        "seed": 0,
        "device": device,
        "entrypoint": "kernel.py::ent",
        "accel_disabled": False,
        "pinned_config": None,
    }
    (work / "spec.json").write_text(_json.dumps(spec))

    real_to = torch.Tensor.to

    def _spy_to(self, *a, **kw):
        # First positional / `device=` kw is the move target.
        dev = kw.get("device")
        if dev is None and a:
            dev = a[0]
        recorder.append(str(dev))
        # Never actually move to cuda (no GPU here) -- keep the tensor
        # on CPU so the rest of the manifest pipeline still runs. For a
        # genuine cpu move just delegate (no-op).
        if isinstance(dev, str) and dev.startswith("cuda"):
            return self
        return real_to(self, *a, **kw)

    monkeypatch.setattr(torch.Tensor, "to", _spy_to, raising=True)
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: True, raising=True
    )

    ns = _ap.Namespace(side="original", spec=str(work / "spec.json"))
    return runner._run_side(ns)


def test_authoritative_pass_actually_moves_inputs_to_cuda(monkeypatch):
    """BLOCKER 1 / HOLE 3 (codex r3+r4 [confirmed]): the
    S2-gpu-authoritative pass must ACTUALLY move every input tensor onto
    the cuda device BEFORE the entrypoint runs (real Triton GPU kernels
    no-run/error on CPU tensors -> S2-gpu-authoritative structurally
    cannot PASS without this).

    This drives the REAL ``_run_side`` in-process with a device-recording
    ``Tensor.to`` stub. It asserts the runner ACTUALLY calls
    ``.to('cuda...')`` on the input tensor -- it is NOT theatrical: if
    the ``inp = _move_to_device(inp, device)`` call at
    ``_pa_subproc_runner.py`` is deleted, NO ``cuda`` device is ever
    recorded and this test FAILS (proven by revert in the commit msg /
    report)."""
    rec: list = []
    rc = _drive_run_side("cuda", monkeypatch, rec)
    cuda_moves = [d for d in rec if str(d).startswith("cuda")]
    # The input tensor 'x' MUST have been moved to cuda before fn(inp).
    assert cuda_moves, (
        "runner did NOT move any input tensor to cuda -- "
        "_move_to_device(inp, device) was not effective "
        f"(recorded .to() targets: {rec}, rc={rc})"
    )
    # And the runner must have completed the real input path (rc=0 ->
    # it ran the entrypoint after the move, not aborted before line
    # 435), so the recorded cuda move is on the genuine input path.
    assert rc == 0, rc


def test_smoke_pass_stays_cpu_no_cuda_move(monkeypatch):
    """The CPU/``TRITON_INTERPRET`` S2-smoke pass passes device='cpu' so
    ``_move_to_device`` is a no-op there -- it must NEVER move an input
    tensor to cuda. Same in-process driver, device='cpu': zero cuda
    moves recorded (smoke genuinely stays on CPU; runs on a no-GPU
    box)."""
    rec: list = []
    rc = _drive_run_side("cpu", monkeypatch, rec)
    cuda_moves = [d for d in rec if str(d).startswith("cuda")]
    assert not cuda_moves, (
        f"smoke pass moved a tensor to cuda (must stay CPU): {rec}"
    )
    assert rc == 0, rc


def test_cuda_host_skips_smoke_by_default_only_gpu_spec(monkeypatch):
    """On a CUDA host, ``s2_verify_task`` skips the CPU/TRITON_INTERPRET
    smoke pre-filter by default (no device='cpu' spec spawned). Smoke
    gives no unique correctness coverage over S2-gpu-authoritative
    (same _run_s2_pair, severed-leak detection lives in the subprocess
    runner not in smoke-specific code), so paying for it on every CUDA
    host only widens the false-positive surface."""
    import json as _json
    import subprocess as _sp

    seen_specs: list[tuple[str, dict]] = []
    real_run = _sp.run

    def _spy_run(cmd, *a, **kw):
        side = cmd[cmd.index("--side") + 1] if "--side" in cmd else "?"
        spec_path = cmd[cmd.index("--spec") + 1]
        seen_specs.append((side, _json.loads(Path(spec_path).read_text())))
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(_sp, "run", _spy_run)
    monkeypatch.setattr(pap, "_cuda_available", lambda: True)
    monkeypatch.delenv("PA_REQUIRE_S2_SMOKE", raising=False)
    src = "import torch\ndef ent(x):\n    x.add_(1.0)\n"
    pap.s2_verify_task(_mk_task(src))

    smoke = [s for sd, s in seen_specs if s.get("device") == "cpu"]
    gpu = [s for sd, s in seen_specs if s.get("device") == "cuda"]
    assert not smoke, (
        f"smoke must not run by default on CUDA host: {[s.get('device') for _sd, s in seen_specs]}"
    )
    assert gpu, [s.get("device") for _sd, s in seen_specs]


def test_require_smoke_opt_in_brings_back_cpu_spec(monkeypatch):
    """``PA_REQUIRE_S2_SMOKE=1`` (set by --require-smoke) forces the CPU
    smoke spec to spawn alongside the CUDA spec, for operators who want
    the pre-filter despite running on a GPU host."""
    import json as _json
    import subprocess as _sp

    seen_specs: list[tuple[str, dict]] = []
    real_run = _sp.run

    def _spy_run(cmd, *a, **kw):
        spec_path = cmd[cmd.index("--spec") + 1]
        seen_specs.append(_json.loads(Path(spec_path).read_text()))
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(_sp, "run", _spy_run)
    monkeypatch.setattr(pap, "_cuda_available", lambda: True)
    monkeypatch.setenv("PA_REQUIRE_S2_SMOKE", "1")
    src = "import torch\ndef ent(x):\n    x.add_(1.0)\n"
    pap.s2_verify_task(_mk_task(src))

    smoke = [s for s in seen_specs if s.get("device") == "cpu"]
    gpu = [s for s in seen_specs if s.get("device") == "cuda"]
    assert smoke, [s.get("device") for s in seen_specs]
    assert gpu, [s.get("device") for s in seen_specs]


def test_authoritative_runner_refuses_cuda_when_absent_no_silent_fake():
    """BLOCKER 1 invariant: the subprocess runner REFUSES a
    device='cuda' spec when torch.cuda.is_available() is False (rc=2,
    explicit error) instead of silently running CPU tensors -- that
    would make S2-gpu-authoritative vacuously 'run' without a GPU (a
    silent fake we reject). This is exercised directly (no GPU here)."""
    import json as _json
    import subprocess as _sp

    work = Path(tempfile.mkdtemp())
    kp = work / "kernel.py"
    kp.write_text("import torch\ndef ent(x):\n    x.add_(1.0)\n")
    spec = {
        "kernel_path": str(kp),
        "status_json": str(work / "status.json"),
        "shapes_json": str(work / "shapes.json"),
        "out_dir": str(work),
        "shape_ids": ["P1"],
        "seed": 0,
        "entrypoint": "ent",
        "device": "cuda",
    }
    (work / "shapes.json").write_text(_json.dumps(_SHAPES))
    (work / "spec.json").write_text(_json.dumps(spec))
    runner = REPO / "scripts" / "_pa_subproc_runner.py"
    proc = _sp.run(
        [sys.executable, str(runner), "--side", "original",
         "--spec", str(work / "spec.json")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2, (proc.returncode, proc.stderr)
    st = _json.loads((work / "status.json").read_text())
    assert st.get("ran") is False
    assert "cuda" in (st.get("error") or "").lower()


def test_placeholder_meta_is_blocked_not_unrun(tmp_path):
    """BLOCKER 2 (codex r3 [confirmed]): a task whose
    meta.oracle.gpu_triton points at kernel.py::placeholder (with NO
    `placeholder` callable defined) must resolve to
    BLOCKED(no-upstream-entrypoint), NEVER UNRUN(needs-gpu). The old
    generator unconditionally labelled every non-U1-blocked task
    UNRUN(needs-gpu) -- the overclaim codex flagged."""
    # Resolvable entrypoint -> WOULD reach the CUDA gate -> UNRUN.
    td_ok = _mk_task(
        "import torch\ndef ent(x):\n    x.add_(1.0)\n",
        gt="kernel.py::ent",
    )
    ok_res, ok_ep, _why = pap._resolve_upstream_entrypoint(td_ok)
    assert ok_res is True and ok_ep == "ent"

    # meta points at ::placeholder, no such callable -> NOT resolvable.
    td_ph = _mk_task(
        "import torch\ndef ent(x):\n    x.add_(1.0)\n",
        gt="kernel.py::placeholder",
    )
    ph_res, ph_ep, ph_why = pap._resolve_upstream_entrypoint(td_ph)
    assert ph_res is False
    assert ph_ep == "placeholder"
    assert "no module-level callable" in ph_why

    # End-to-end through the real status renderer: the placeholder task
    # is BLOCKED(no-upstream-entrypoint); the resolvable one is
    # UNRUN(needs-gpu). NEITHER is mislabelled.
    tc_ok = pap.classify_task(
        td_ok.name, (td_ok / "kernel.py").read_text()
    )
    tc_ok.task_id = "T_OK"
    tc_ph = pap.classify_task(
        td_ph.name, (td_ph / "kernel.py").read_text()
    )
    tc_ph.task_id = "T_PH"
    # HOLE 1: the renderer now ACTUALLY runs s2_verify_task for the
    # resolvable task. We inject a verify_fn that returns the genuine
    # UNRUN verdict (smoke-passed, no-GPU) so this test deterministically
    # exercises the STATUS-MAPPING logic; the placeholder task is still
    # settled WITHOUT a verify call (its entrypoint never resolves).
    calls: list = []

    def _vf(td):
        calls.append(td.name)
        return pap.S2Result(
            task_id=td.name, status="UNRUN", smoke_ran=True,
            reason="S2-smoke passed (CPU); UNRUN(needs-gpu).",
        )

    # MUST-FIX 1 (codex r2 [confirmed]): the hardened resolver verdict
    # is the single source of truth for a meta-pre-set task. Inject a
    # PER-TASK resolver verdict (mirrors what the REAL resolver returns
    # for each): the resolvable task genuinely passes the hardened
    # anchor (UNRUN -> override falls through to the real `_vf`); the
    # ::placeholder task genuinely has no candidate callable
    # (no-host-wrapper -> honest no-upstream-entrypoint). This proves
    # the override does NOT over-broaden -- a genuine no-callable task
    # is STILL no-upstream-entrypoint, never a false UNRUN.
    def _rvf(task_dir):
        if task_dir.name == td_ph.name:
            return (
                "BLOCKED(no-verified-upstream-entrypoint: "
                "no-host-wrapper)",
                "no host wrapper launches a @triton.jit kernel "
                "(no-host-wrapper)",
            )
        return ("UNRUN(needs-gpu)", "hardened anchor 2/2 shapes ok")

    md = pap._render_status_md(
        [tc_ok, tc_ph], gpu=False, eligible_count=2,
        task_dirs_by_id={"T_OK": td_ok, "T_PH": td_ph},
        verify_fn=_vf, resolver_verdict_fn=_rvf,
    )
    # Only the RESOLVABLE task triggers a real verify; the ::placeholder
    # task is pre-filtered (no smoke run) -- codex r4 "keep that part".
    assert calls == [td_ok.name], calls
    assert "| T_PH | " in md
    ph_row = next(
        ln for ln in md.splitlines() if ln.startswith("| T_PH | ")
    )
    ok_row = next(
        ln for ln in md.splitlines() if ln.startswith("| T_OK | ")
    )
    assert "BLOCKED(no-upstream-entrypoint)" in ph_row, ph_row
    assert "UNRUN(needs-gpu)" not in ph_row, ph_row
    assert "UNRUN(needs-gpu)" in ok_row, ok_row
    assert "BLOCKED(no-upstream-entrypoint)" not in ok_row, ok_row
    # The benchmark-truth banner + honest summary line are present.
    assert "BENCHMARK TRUTH" in md
    assert "BLOCKED(no-upstream-entrypoint): 1" in md
    assert "UNRUN(needs-gpu): 1" in md
    assert "PASS: 0" in md


def test_render_status_runs_real_s2_smoke_not_static_guess(monkeypatch):
    """HOLE 1 (codex r4 [confirmed]): _render_status_md must DERIVE a
    resolvable task's status from the REAL s2_verify_task verdict, NOT
    a static UNRUN(needs-gpu) guess. A smoke MISMATCH must surface as
    BLOCKED(smoke-fail) (NOT an overclaimed UNRUN); a smoke PASS as
    UNRUN(needs-gpu); a (impossible-here) PASS as PASS. This FAILS if
    the renderer reverts to statically labelling every resolvable task
    UNRUN(needs-gpu)."""
    td = _mk_task("import torch\ndef ent(x):\n    x.add_(1.0)\n")
    tc = pap.classify_task(td.name, (td / "kernel.py").read_text())

    # MUST-FIX 1 (codex r2 [confirmed]): inject a resolver verdict that
    # AGREES the entrypoint genuinely passes the hardened anchor so the
    # single-source-of-truth override falls through to the real
    # `verify_fn` -- this test exercises the S2 status-MAPPING logic
    # (smoke MISMATCH -> smoke-fail, etc.), not the override itself.
    def _rvf_pass(_td):
        return ("UNRUN(needs-gpu)", "hardened anchor 2/2 shapes ok")

    def _mk(status, reason):
        tc.task_id = "T"
        return pap._render_status_md(
            [tc], gpu=False, eligible_count=1,
            task_dirs_by_id={"T": td},
            verify_fn=lambda _t: pap.S2Result(
                task_id="T", status=status, smoke_ran=True, reason=reason
            ),
            resolver_verdict_fn=_rvf_pass,
        )

    # 1. Real smoke MISMATCH -> BLOCKED(smoke-fail), NOT UNRUN.
    md_b = _mk("BLOCKED", "S2-smoke MISMATCH: original side did not run")
    b_row = next(ln for ln in md_b.splitlines() if ln.startswith("| T | "))
    assert "BLOCKED(smoke-fail)" in b_row, b_row
    assert "UNRUN(needs-gpu)" not in b_row, b_row
    assert "BLOCKED(smoke-fail): 1" in md_b
    assert "S2-smoke MISMATCH" in b_row  # the REAL reason is surfaced

    # 2. Real smoke PASS (no GPU) -> UNRUN(needs-gpu).
    md_u = _mk("UNRUN", "smoke passed; needs-gpu")
    u_row = next(ln for ln in md_u.splitlines() if ln.startswith("| T | "))
    assert "UNRUN(needs-gpu)" in u_row, u_row
    assert "BLOCKED(smoke-fail)" not in u_row, u_row

    # 3. PASS verdict (only ever from a real GPU pass) -> PASS, honest.
    md_p = _mk("PASS", "S2-gpu-authoritative PASS")
    p_row = next(ln for ln in md_p.splitlines() if ln.startswith("| T | "))
    assert "| PASS |" in p_row, p_row

    # 4. A verify CRASH is itself BLOCKED truth (never a silent UNRUN).
    tc.task_id = "T"
    md_c = pap._render_status_md(
        [tc], gpu=False, eligible_count=1, task_dirs_by_id={"T": td},
        verify_fn=lambda _t: (_ for _ in ()).throw(RuntimeError("boom")),
        resolver_verdict_fn=_rvf_pass,
    )
    c_row = next(ln for ln in md_c.splitlines() if ln.startswith("| T | "))
    assert "BLOCKED(smoke-fail)" in c_row and "boom" in c_row, c_row


def test_resolver_mirrors_runner_resolution(tmp_path):
    """HOLE 2 (codex r4 [confirmed]): _resolve_upstream_entrypoint must
    agree with what _pa_subproc_runner._resolve_entrypoint would
    actually do for the ORIGINAL side (real kernel.py, no synthesised
    run). We build a real module for each construct, run the runner's
    resolver, and assert the static resolver's verdict matches
    callable(runner_result) -- with the documented CONSERVATIVE rule
    for statically-ambiguous RHS."""
    import importlib.util as _ilu

    runner = _load_subproc_runner()

    def _runner_callable(src: str, ep_name: str) -> bool:
        mp = tmp_path / f"k_{abs(hash(src))}.py"
        mp.write_text(src)
        sp = _ilu.spec_from_file_location("_k_mod", mp)
        mod = _ilu.module_from_spec(sp)
        sp.loader.exec_module(mod)
        fn = runner._resolve_entrypoint(
            mod, {"entrypoint": f"kernel.py::{ep_name}"}
        )
        return callable(fn)

    # (label, src, ep, expect_static_resolvable, expect_runner_callable)
    cases = [
        ("def", "import torch\ndef ent(x):\n    return x\n", "ent",
         True, True),
        ("callable_assign",
         "import torch\nent = (lambda x: x)\n", "ent", True, True),
        ("noncallable_assign", "ent = 5\n", "ent", False, False),
        ("placeholder_missing",
         "import torch\ndef other():\n    return 1\n", "placeholder",
         False, False),
        ("run_fallback",
         "import torch\nent = 5\ndef run(inputs):\n    return {}\n",
         "ent", True, True),
    ]
    for label, src, ep, exp_static, exp_runner in cases:
        st_res, _ep, _why = pap._resolve_upstream_entrypoint(
            _mk_task(src, gt=f"kernel.py::{ep}")
        )
        rn = _runner_callable(src, ep)
        assert rn is exp_runner, (label, "runner", rn)
        assert st_res is exp_static, (label, "static", st_res, _why)
        # The CORE invariant: the static verdict must MATCH the runner's
        # actual callable() outcome (so status never diverges from what
        # s2_verify would do).
        assert st_res is rn, (label, "DIVERGED", st_res, rn, _why)


# ───────────────────────── HOLE C ─────────────────────────
def test_alias_key_is_storage_identity_not_data_ptr():
    """Same-storage views at DIFFERENT offsets have different data_ptr
    but the SAME storage -> they ARE aliased. The storage_id key groups
    them; the old data_ptr key (reverted bug) SPLIT them."""
    base = torch.arange(8, dtype=torch.float32)
    v0, v1 = base[0:4], base[4:8]  # one storage, two offsets
    assert v0.data_ptr() != v1.data_ptr()  # element ptrs differ
    meta = {
        "a": {"data_ptr": v0.data_ptr(),
              "storage_id": v0.untyped_storage().data_ptr(),
              "storage_offset": 0},
        "b": {"data_ptr": v1.data_ptr(),
              "storage_id": v1.untyped_storage().data_ptr(),
              "storage_offset": 4},
    }
    groups = pap._alias_groups(meta)
    assert any({"a", "b"} <= g for g in groups), groups  # grouped
    # the reverted (data_ptr) key would have split them:
    legacy: dict = {}
    for k, m in meta.items():
        legacy.setdefault(m["data_ptr"], set()).add(k)
    assert all(len(g) == 1 for g in legacy.values())  # bug behaviour


def test_output_aliasing_unmodified_input_is_flagged():
    """SAME VALUES on both sides, only the alias relation differs:
    original returns a FRESH clone; severed returns the UNMODIFIED
    input itself. Previously only mutated inputs entered the alias
    relation so this was INVISIBLE; now it MUST be flagged."""
    orig = Path(tempfile.mkdtemp()) / "kernel.py"
    orig.write_text("import torch\ndef ent(x):\n    return x.clone()\n")
    sev = Path(tempfile.mkdtemp()) / "kernel.py"
    sev.write_text(
        "import torch\ndef run(inputs):\n    return {'_out0': inputs['x']}\n"
    )
    eq, why, *_ = pap._run_s2_pair(
        orig, sev, "ent", _SHAPES, ["P1"], 0, False, None, True
    )
    assert eq is False
    assert "alias relationship differs" in why


def test_compare_ignores_process_local_pointers_for_layout():
    """Cross-side metadata equality must EXCLUDE process-local pointers
    (data_ptr/storage_id/storage_nbytes always differ between the two
    subprocesses) -- otherwise even an equivalent kernel spuriously
    fails. The alias RELATIONSHIP is still checked (position-independent
    grouping)."""
    m = {"dtype": "torch.float32", "shape": [4], "stride": [1],
         "storage_offset": 0, "data_ptr": 111, "storage_id": 222,
         "storage_nbytes": 16}
    lf = pap._layout_fields(m)
    assert "data_ptr" not in lf and "storage_id" not in lf
    assert "storage_nbytes" not in lf
    assert lf == {"dtype": "torch.float32", "shape": [4],
                  "stride": [1], "storage_offset": 0}


# ───────── codex r5 ISSUE 1: no silent skip+positional-shift ─────────
def test_signature_adapter_missing_required_param_is_explicit_blocked():
    """codex r5 ISSUE 1 (real S2-harness soundness bug). When
    shapes.yaml omits a NO-default entrypoint parameter, the adapter
    must NOT silently skip it and SHIFT the remaining positional args
    (which lands e.g. a bool in a tensor slot -> a misleading crash OR,
    far worse, a wrong-but-"successful" comparison). It must raise a
    hard, explicit ShapesIncompleteError naming the missing param.

    FAILS ON REVERT: the old body
    ``args = [inputs[p] for p in params if p in inputs]`` would call
    ``fn(b, c)`` (a SILENT shift) -- ``c`` (a bool) lands in the ``b``
    slot, raising AttributeError 'bool'... ('.shape'), NOT a
    shapes-incomplete error, and crucially proving the shift happened.
    This test asserts the explicit-by-name behaviour, so reverting it
    makes the assertions fail."""
    runner = _load_subproc_runner()

    captured: list = []

    def fn(a, b, c, sm_scale=0.5):
        # Faithful "upstream" -- records exactly what it was called
        # with so a silent shift is provable.
        captured.append((a, b, c, sm_scale))
        return a

    adapted = runner._signature_adapter(fn, "fn")

    # `b` (no default) is MISSING from inputs; `c` IS present. The old
    # skip+shift would do fn(inputs['a'], inputs['c']) -> c bound to b.
    import pytest

    with pytest.raises(runner.ShapesIncompleteError) as ei:
        adapted({"a": 1, "c": True})
    msg = str(ei.value)
    assert runner.SHAPES_INCOMPLETE_MARKER in msg
    assert "missing param 'b'" in msg
    assert "for entrypoint 'fn'" in msg
    # The entrypoint was NEVER called with a shifted arg list.
    assert captured == [], captured

    # Sanity: when every NO-default param IS present it binds BY NAME
    # (order-independent) and the optional default is left alone.
    out = adapted({"c": 3, "b": 2, "a": 1})
    assert out == {"_out0": 1}
    assert captured == [(1, 2, 3, 0.5)], captured


def test_synth_run_template_missing_required_param_is_explicit_blocked():
    """codex r5 ISSUE 1: the SYNTHESISED severed run() must mirror the
    adapter -- map BY NAME, hard-error on a missing no-default param,
    NEVER silently skip+shift. FAILS ON REVERT: the old template body
    ``_args = [inputs[p] for p in _params if p in inputs]`` would
    silently shift and call the entrypoint; this test asserts a
    BLOCKED(shapes-incomplete RuntimeError instead."""
    import importlib.util as _ilu

    src = pap.synthesise_standalone_run(
        "import torch\n"
        "def ent(q, k, o, sm_scale=0.125):\n"
        "    return q\n",
        "ent",
    )
    assert "def run(" in src
    mp = Path(tempfile.mkdtemp()) / "k.py"
    mp.write_text(src)
    sp = _ilu.spec_from_file_location("_synth_k", mp)
    mod = _ilu.module_from_spec(sp)
    sp.loader.exec_module(mod)

    import pytest

    # `o` (no default) absent -> hard BLOCKED(shapes-incomplete, NOT a
    # shifted fn(q, k) call (which would return q "successfully").
    with pytest.raises(RuntimeError) as ei:
        mod.run({"q": 1, "k": 2})
    m = str(ei.value)
    assert "BLOCKED(shapes-incomplete" in m
    assert "missing param 'o'" in m
    # All present (incl. by-name order shuffle) -> binds correctly,
    # optional default untouched.
    assert mod.run({"k": 2, "o": 3, "q": 1}) == {"_out0": 1}


# ───────── codex r5 ISSUE 2: L2_10 inputs are INT8 per contract ──────
def test_L2_10_w8a8_inputs_are_int8_per_contract():
    """codex r5 ISSUE 2 (fixture dtype bug). L2_10 is a W8A8 *INT8*
    block matmul -- contract.md requires ``A_int8``/``B_int8`` and the
    kernel does ``tl.dot(int8, int8) -> int32``. shapes.yaml previously
    produced int64 (-> 'Unsupported lhs dtype int64' smoke error, a
    FIXTURE defect, not a severing one). FAILS ON REVERT: with
    dtype:int64 these dtype asserts fail."""
    import yaml as _yaml

    sh = _yaml.safe_load(
        (REPO / "tasks" / "level2" / "L2_10_w8a8_block_int8_matmul"
         / "shapes.yaml").read_text()
    )
    assert sh["inputs"]["A"]["dtype"] == "int8", sh["inputs"]["A"]
    assert sh["inputs"]["B"]["dtype"] == "int8", sh["inputs"]["B"]

    # The contract really does mandate INT8 (anchor the expectation).
    contract = (
        REPO / "tasks" / "level2" / "L2_10_w8a8_block_int8_matmul"
        / "contract.md"
    ).read_text()
    assert "INT8 GEMM" in contract
    assert "A_int8" in contract and "B_int8" in contract

    # build_inputs now actually materialises signed int8 A/B.
    from triton_ascend_bench.inputs import build_inputs

    inp = build_inputs(sh, sh["public_shapes"][0]["id"], seed=0)
    assert inp["A"].dtype is torch.int8, inp["A"].dtype
    assert inp["B"].dtype is torch.int8, inp["B"].dtype
    # randint:0:8 values fit signed int8.
    assert int(inp["A"].min()) >= -128 and int(inp["A"].max()) <= 127


# ───── codex r5 ISSUE 3: shapes-incomplete != smoke-fail/severing ────
def test_status_labels_shapes_incomplete_not_smoke_fail_or_severing():
    """codex r5 ISSUE 3 (status over-description = overclaim). A task
    whose entrypoint RESOLVES but whose shapes.yaml omits a required
    no-default param (the honest L3_04 category after ISSUE 1) MUST be
    labelled ``BLOCKED(shapes-incomplete)`` and described as a benchmark
    GAP -- NEVER ``BLOCKED(smoke-fail)`` / a 'genuine severing finding'.
    A genuinely severing smoke mismatch is STILL smoke-fail. FAILS ON
    REVERT: the old renderer routed every BLOCKED into smoke-fail with
    'genuine finding, NOT an overclaimed UNRUN'."""
    td = _mk_task("import torch\ndef ent(x):\n    x.add_(1.0)\n")

    def _row(md, tid):
        return next(
            ln for ln in md.splitlines() if ln.startswith(f"| {tid} | ")
        )

    # MUST-FIX 1 (codex r2 [confirmed]): inject a resolver verdict that
    # AGREES the entrypoint passes the hardened anchor so the
    # single-source-of-truth override falls through to the real
    # `verify_fn` -- this test exercises the S2 status-MAPPING
    # (shapes-incomplete vs smoke-fail), not the override itself.
    def _rvf_pass(_td):
        return ("UNRUN(needs-gpu)", "hardened anchor 2/2 shapes ok")

    # 1. shapes-incomplete reason -> BLOCKED(shapes-incomplete), GAP
    #    language, NOT smoke-fail / NOT 'severing'.
    tc1 = pap.classify_task(td.name, (td / "kernel.py").read_text())
    tc1.task_id = "T_SI"
    md_si = pap._render_status_md(
        [tc1], gpu=False, eligible_count=1,
        task_dirs_by_id={"T_SI": td},
        verify_fn=lambda _t: pap.S2Result(
            task_id="T_SI", status="BLOCKED", smoke_ran=True,
            reason="original side did not run (rc=2): "
            "ShapesIncompleteError: BLOCKED(shapes-incomplete: "
            "missing param 'o' for entrypoint 'context_attention_fwd'",
        ),
        resolver_verdict_fn=_rvf_pass,
    )
    si_row = _row(md_si, "T_SI")
    assert "BLOCKED(shapes-incomplete)" in si_row, si_row
    assert "BLOCKED(smoke-fail)" not in si_row, si_row
    assert "benchmark GAP" in si_row, si_row
    assert "NOT a severing defect" in si_row, si_row
    assert "BLOCKED(shapes-incomplete): 1" in md_si
    # The summary line for shapes-incomplete must NOT call it a genuine
    # severing finding, and smoke-fail count stays 0.
    assert "BLOCKED(smoke-fail): 0" in md_si
    assert "NOT a severing defect / NOT a smoke-fail" in md_si

    # 2. A genuinely severing smoke mismatch is STILL smoke-fail (we did
    #    not over-broaden shapes-incomplete).
    tc2 = pap.classify_task(td.name, (td / "kernel.py").read_text())
    tc2.task_id = "T_SF"
    md_sf = pap._render_status_md(
        [tc2], gpu=False, eligible_count=1,
        task_dirs_by_id={"T_SF": td},
        verify_fn=lambda _t: pap.S2Result(
            task_id="T_SF", status="BLOCKED", smoke_ran=True,
            reason="S2-smoke MISMATCH (phase=compare): leaf values "
            "differ at bit level -- severed reorders the reduction",
        ),
        resolver_verdict_fn=_rvf_pass,
    )
    sf_row = _row(md_sf, "T_SF")
    assert "BLOCKED(smoke-fail)" in sf_row, sf_row
    assert "BLOCKED(shapes-incomplete)" not in sf_row, sf_row
    assert "BLOCKED(smoke-fail): 1" in md_sf
    assert "BLOCKED(shapes-incomplete): 0" in md_sf
    # BENCHMARK TRUTH banner states the dominant blocker honestly.
    assert "dominant blocker" in md_sf
    assert "S2 PASS requires a GPU regardless" in md_sf


# ── codex r6 residual #2 [confirmed]: write-back on live re-resolve ──
def _rue():
    _s = importlib.util.spec_from_file_location(
        "resolve_upstream_entrypoints",
        REPO / "scripts" / "resolve_upstream_entrypoints.py",
    )
    _m = importlib.util.module_from_spec(_s)
    sys.modules["resolve_upstream_entrypoints"] = _m
    _s.loader.exec_module(_m)
    return _m


def test_writeback_fresh_verdict_updates_only_that_task(
    tmp_path, monkeypatch,
):
    """codex r6 residual #2 [confirmed] -- FAIL ON REVERT.

    After a forced live per-task re-resolve, the canonical cache's
    THAT-task entry MUST be updated with fresh ``input_fp`` +
    ``upstream_fp`` under a re-stamped ``_provenance``, every OTHER
    verdict preserved byte-for-byte, written crash-safely (temp+rename).
    Reverting the write-back (return-only, no persist) makes the
    "entry updated" assertion FAIL (the same task would be re-resolved
    forever).
    """
    import json

    rue = _rue()
    cache = tmp_path / "resolve_all.json"
    monkeypatch.setattr(pap, "RESOLVE_CACHE", cache)
    monkeypatch.setattr(pap, "_RESOLVER_CACHE_MEM", None)

    # Seed a cache with two tasks; one ("KEEPME") must be preserved
    # untouched, the other ("L1_26_merge_state", a REAL task so
    # input_fp resolves) carries a STALE upstream_fp/input_fp.
    keep = {
        "task_id": "KEEPME", "resolved": False, "status": "S",
        "reason": "R", "input_fp": "keep-ip", "upstream_fp": "keep-up",
    }
    stale = {
        "task_id": "L1_26_merge_state", "resolved": False,
        "status": "OLD", "reason": "stale-reason",
        "input_fp": "STALE", "upstream_fp": "STALE-UP",
    }
    cache.write_text(json.dumps({
        "_provenance": rue.build_provenance_stamp(),
        "verdicts": [keep, stale],
    }, indent=2))

    class _Fresh:
        def __init__(self):
            self.__dict__ = {
                "task_id": "L1_26_merge_state",
                "resolved": True, "entrypoint": "merge_state",
                "status": "UNRUN(needs-gpu)", "reason": "fresh!",
                "tried": ["merge_state"],
                "upstream_fp": "sha256:" + ("c" * 64),
            }

    fresh = _Fresh()
    pap._writeback_fresh_verdict(rue, fresh)

    out = json.loads(cache.read_text())
    by = {v["task_id"]: v for v in out["verdicts"]}
    # The stale task's entry is REPLACED with the fresh verdict +
    # freshly re-derived input_fp + the fresh upstream_fp.
    assert by["L1_26_merge_state"]["status"] == (
        "UNRUN(needs-gpu)"
    ), "write-back did NOT persist the fresh verdict (reverted -- "
    "the task would be re-resolved on every run forever)"
    td = rue._task_dir_for("L1_26_merge_state")
    assert by["L1_26_merge_state"]["input_fp"] == (
        rue.task_input_fingerprint(td)
    ), "fresh input_fp not re-derived on write-back"
    assert by["L1_26_merge_state"]["upstream_fp"] == (
        "sha256:" + ("c" * 64)
    ), "fresh upstream_fp not persisted on write-back"
    # The OTHER task is preserved byte-for-byte.
    assert by["KEEPME"] == keep, (
        "write-back clobbered an unrelated verdict -- it must update "
        "ONLY that task's entry"
    )
    # The whole cache is re-validatable (provenance re-stamped
    # consistently: code_fp + infer_state both current).
    _v, st = rue.load_resolver_cache(cache)
    assert st is None, f"re-stamped cache not fresh: {st}"
    # No leftover temp file (atomic rename cleaned up).
    assert not (
        cache.with_suffix(cache.suffix + ".tmp")
    ).exists()


def test_writeback_honors_pa_no_cache_regen(tmp_path, monkeypatch):
    """codex r6 residual #2 [confirmed] -- FAIL ON REVERT.

    With ``PA_NO_CACHE_REGEN=1`` the write-back MUST NOT persist (cache
    bytes unchanged) -- the caller still gets the correct fresh verdict
    in-memory, it just is not written. If the env honor is reverted the
    cache bytes change and this FAILS.
    """
    import json

    rue = _rue()
    cache = tmp_path / "resolve_all.json"
    monkeypatch.setattr(pap, "RESOLVE_CACHE", cache)
    monkeypatch.setattr(pap, "_RESOLVER_CACHE_MEM", None)
    seed = {
        "_provenance": rue.build_provenance_stamp(),
        "verdicts": [{
            "task_id": "L1_26_merge_state", "resolved": False,
            "status": "OLD", "reason": "r", "input_fp": "x",
            "upstream_fp": "y",
        }],
    }
    cache.write_text(json.dumps(seed, indent=2))
    before = cache.read_text()

    class _Fresh:
        def __init__(self):
            self.__dict__ = {
                "task_id": "L1_26_merge_state", "resolved": True,
                "status": "UNRUN(needs-gpu)", "reason": "fresh",
                "upstream_fp": "sha256:" + ("d" * 64),
            }

    monkeypatch.setenv("PA_NO_CACHE_REGEN", "1")
    pap._writeback_fresh_verdict(rue, _Fresh())
    assert cache.read_text() == before, (
        "write-back persisted despite PA_NO_CACHE_REGEN=1 -- the "
        "regen-disabled contract is reverted"
    )


def test_param_name_map_baked_into_severed_run_and_binder():
    """meta.interface.param_name_map maps a wrapper param to the SAME task
    tensor under a different key (e.g. wrapper `weight` <- shapes `w`), used
    IDENTICALLY by the severed synthesised run() (baked) and the original
    side's binder. bound records the TASK key so in-place output recon
    indexes inputs correctly (codex finding 3). No map => unchanged."""
    import ast as _ast
    # severed side: baked into synthesise_standalone_run
    out = pap.synthesise_standalone_run(
        "def foo(x, weight, bias):\n    return x\n", "foo",
        {"weight": "w", "bias": "b"},
    )
    _ast.parse(out)  # valid Python
    assert "_PNM = {'weight': 'w', 'bias': 'b'}" in out
    assert "_PNM[_n] in inputs" in out
    # original side: the shared binder
    import importlib.util as _ilu
    sp = _ilu.spec_from_file_location(
        "_pa_subproc_runner", REPO / "scripts" / "_pa_subproc_runner.py"
    )
    sub = _ilu.module_from_spec(sp)
    sys.modules["_pa_subproc_runner"] = sub
    sp.loader.exec_module(sub)

    def foo(x, weight, bias):
        return None

    _, bound, kwargs = sub._map_inputs_by_name(
        foo, "foo", {"x": 1, "w": 2, "b": 3}, {"weight": "w", "bias": "b"}
    )
    assert kwargs == {"x": 1, "weight": 2, "bias": 3}
    assert set(bound) == {"x", "w", "b"}  # TASK keys, not wrapper names
    # backward-compat: no map -> the strict missing-param error
    import pytest
    with pytest.raises(sub.ShapesIncompleteError):
        sub._map_inputs_by_name(foo, "foo", {"x": 1, "w": 2, "b": 3}, {})
