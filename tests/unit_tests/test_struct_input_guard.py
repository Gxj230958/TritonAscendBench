"""U4: the struct-input importer fail-closed guard (R7).

``_materialise_struct`` is on the runtime fixture path (eval -> build_inputs_for
-> build_inputs) and imports an arbitrary dotted path taken from a task's
shapes.yaml ``{struct: '<dotted.path.Class>'}``. On the runtime path the guard
allows only ascendbench / stdlib / torch top-level packages (an allowlist, so an
arbitrary YAML-driven import cannot smuggle in vllm/sglang or a third-party dep
that imports upstream indirectly); the provenance/oracle path opts in via
``AB_ALLOW_UPSTREAM_STRUCT=1``.

The tests stub ``importlib.import_module`` so they assert the GUARD's behavior
(reached vs. not reached) deterministically, independent of whether the named
package happens to be importable / stubbed in ``sys.modules`` by an unrelated
test (a dev box may have a real upstream tree on ``sys.path``).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch  # noqa: F401  -- ensures torch is importable for the allowlist test
import yaml

from ascendbench.input_spec.inputs import _materialise_struct

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_ALLOWED = {"ascendbench", "torch"} | sys.stdlib_module_names


def _call(struct_path: str):
    return _materialise_struct(
        "x", {"struct": struct_path, "fields": {}}, {}, torch.Generator()
    )


def _spy_import_module(monkeypatch):
    """Replace importlib.import_module with a spy that records the modpath it was
    asked for and raises, so the test never depends on a real import."""
    called: dict[str, str] = {}

    def _spy(modpath, *a, **k):
        called["modpath"] = modpath
        raise ModuleNotFoundError(f"stub: {modpath}")

    monkeypatch.setattr(importlib, "import_module", _spy)
    return called


@pytest.mark.parametrize(
    "path",
    [
        "sglang.srt.lora.utils.LoRABatchInfo",
        "vllm.something.Cls",
        "numpy.ndarray",  # any non-runtime third-party is blocked by the allowlist
    ],
)
def test_struct_non_runtime_import_blocked(path, monkeypatch):
    """Without the opt-in, a struct whose top package is not ascendbench/stdlib/
    torch is rejected by the guard BEFORE import_module is reached."""
    monkeypatch.delenv("AB_ALLOW_UPSTREAM_STRUCT", raising=False)
    called = _spy_import_module(monkeypatch)
    with pytest.raises(ImportError) as exc:
        _call(path)
    assert "release-boundary violation" in str(exc.value)
    assert "modpath" not in called, "guard must fire before import_module"


def test_struct_optin_bypasses_guard(monkeypatch):
    """With the provenance opt-in the guard does NOT fire: execution reaches
    import_module with the upstream path, not the boundary error."""
    monkeypatch.setenv("AB_ALLOW_UPSTREAM_STRUCT", "1")
    called = _spy_import_module(monkeypatch)
    with pytest.raises(ImportError) as exc:
        _call("sglang.srt.lora.utils.LoRABatchInfo")
    assert called.get("modpath") == "sglang.srt.lora.utils"
    assert "release-boundary violation" not in str(exc.value)


@pytest.mark.parametrize(
    "path",
    [
        "ascendbench.does_not_exist.Cls",  # ascendbench allowed
        "torch.nn.modules.linear.Linear",  # torch allowed
        "collections.OrderedDict",         # stdlib allowed
    ],
)
def test_struct_runtime_allowed_passes_guard(path, monkeypatch):
    """A runtime-allowed top package (ascendbench/stdlib/torch) is not blocked:
    import_module is reached normally."""
    monkeypatch.delenv("AB_ALLOW_UPSTREAM_STRUCT", raising=False)
    called = _spy_import_module(monkeypatch)
    with pytest.raises(ImportError) as exc:
        _call(path)
    assert "release-boundary violation" not in str(exc.value)
    assert called.get("modpath") == path.rpartition(".")[0]


def _iter_struct_paths(node):
    """Yield every value bound to a ``struct`` key anywhere in a parsed doc."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "struct" and isinstance(v, str):
                yield v
            else:
                yield from _iter_struct_paths(v)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_struct_paths(item)


def test_no_task_declares_non_runtime_struct_input():
    """Corpus-lint (R7): no task's shapes.yaml may declare a `struct:` input whose
    top-level package is outside the runtime allowlist. Parses the YAML and
    traverses for `struct` keys (robust to quoting / spacing forms)."""
    offenders = []
    for sy in (_REPO_ROOT / "tasks").rglob("shapes.yaml"):
        try:
            docs = list(yaml.safe_load_all(sy.read_text(encoding="utf-8")))
        except yaml.YAMLError:
            continue
        for doc in docs:
            for struct_path in _iter_struct_paths(doc):
                top = struct_path.split(".", 1)[0]
                if top not in _RUNTIME_ALLOWED:
                    offenders.append(f"{sy.relative_to(_REPO_ROOT)}: struct={struct_path}")
    assert not offenders, f"tasks declare non-runtime struct inputs: {offenders}"
