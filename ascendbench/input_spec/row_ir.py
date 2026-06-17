"""Phase-2 U5: a typed ``InputSpec`` / ``RowSpec`` IR with registered handlers.

This is the **single extension boundary** for the explicit-shape format (codex's highest-leverage
scale fix). Instead of every consumer (manifest / build_inputs / IPC / sandbox / cost / constraints)
re-parsing raw YAML and hard-coding the dtype + dist + input-kind tables, they consume this typed IR,
and a new dtype / dist kind / input kind is added by **registering a handler** + bumping
``SCHEMA_VERSION`` -- not by editing every consumer.

U5 defines the IR + the registry INTERFACE (bridged to the existing `inputs.py` tables so the known
set is populated). Wiring the consumers to consume the IR is U8. The IR hosts all six current input
forms (tensor / scalar / scalar_knob / derive / scalar_seq / optional) and the named ROW ENVIRONMENT
(tensor axes + named ``params`` for the non-axis symbols U4 classified).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1   # bump when a handler set or the row shape changes; readers gate on this


# ── handler registries: the extension boundary ────────────────────────────────────────────────────

# dtype name -> (torch dtype factory, byte size). Populated from inputs._DTYPES so the v1 set matches
# build_inputs exactly; a new dtype (fp4/int4/mxfp/NPU-native) registers here instead of editing the
# manifest / cost-model / IPC dtype tables one by one.
DTYPE_REGISTRY: dict[str, dict[str, Any]] = {}
# dist kind -> a materialiser handler (Callable). The v1 set is bootstrapped from inputs.DIST_KINDS
# (the single authority -- never hand-copied, no drift); a new fill rule registers here. NOTE: importing
# DIST_KINDS pulls in inputs.py, which imports torch at module top, so the bootstrap is torch-coupled
# in practice -- a torch-less env leaves this empty and validate() then fails loud (not a silent pass).
DIST_REGISTRY: dict[str, Callable | None] = {}
# input kind -> validator/describer. A new input form registers via register_input_kind().
INPUT_KINDS: set[str] = {"tensor", "scalar", "scalar_knob", "derive", "scalar_seq", "optional", "struct"}

VALID_ROLES = {"public", "hidden"}
VALID_PROVENANCE = {"synthetic", "npukb-borrowed", "model-sourced", "edge", "baseline"}
# the param kinds row_from_materialized emits (the named ROW ENVIRONMENT) + the row-only class (U4).
VALID_PARAM_KINDS = {"scalar", "scalar_knob", "derive", "scalar_seq", "optional", "row-only"}


def register_dtype(name: str, torch_dtype: Any, byte_size: int) -> None:
    DTYPE_REGISTRY[name] = {"dtype": torch_dtype, "bytes": byte_size}


def register_dist(kind: str, handler: Callable | None = None) -> None:
    DIST_REGISTRY[kind] = handler


def register_input_kind(kind: str, *, is_param: bool = False) -> None:
    """Register a new input FORM. A non-tensor/non-struct form lands in the row environment as a
    ``ParamSpec``, so pass ``is_param=True`` to also admit it in ``RowSpec.validate`` -- otherwise
    INPUT_KINDS and VALID_PARAM_KINDS drift and the new kind is rejected at validation."""
    INPUT_KINDS.add(kind)
    if is_param:
        VALID_PARAM_KINDS.add(kind)


def _bootstrap_registries() -> None:
    """Populate the v1 registries from the existing inputs.py authorities so the IR's known set == what
    build_inputs already supports (no divergence, no drift). Both imports route through inputs.py, which
    imports torch at module top -- so a torch-less env leaves BOTH registries empty and ``validate``
    fails loud (RuntimeError) rather than silently skipping the dtype/dist checks."""
    if not DIST_REGISTRY:
        try:
            from ascendbench.input_spec.inputs import DIST_KINDS  # the single authority
            for kind in DIST_KINDS:
                DIST_REGISTRY.setdefault(kind, None)
        except Exception:
            pass  # torch-less env: leave empty -> TensorSpec.validate() fails loud, not a silent pass
    if not DTYPE_REGISTRY:
        try:
            import torch

            from ascendbench.input_spec.inputs import _DTYPES
        except Exception:
            return
        _byte = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2, torch.int8: 1,
                 torch.int32: 4, torch.int64: 8, torch.bool: 1, torch.float8_e4m3fn: 1}
        for name, dt in _DTYPES.items():
            register_dtype(name, dt, _byte.get(dt, 0))


# ── the typed IR ──────────────────────────────────────────────────────────────────────────────────

@dataclass
class TensorSpec:
    """An explicit tensor input: literal shape + dtype + named axes + fill rule."""
    shape: list[int]
    dtype: str
    axes: list[str]                 # named axes (dim expressions; the glossary maps symbols to names)
    dist: dict[str, Any]            # {kind, args} -- the fill rule

    def validate(self) -> None:
        _bootstrap_registries()
        # fail LOUD if the dtype registry could not bootstrap (torch/inputs unavailable) rather than
        # silently skipping the dtype/dist checks -- a vacuous pass would violate the fail-loud invariant.
        if not DTYPE_REGISTRY:
            raise RuntimeError("dtype registry not bootstrapped (torch/inputs unavailable) -- "
                               "cannot validate; do not treat as valid")
        if self.dtype not in DTYPE_REGISTRY:
            raise ValueError(f"unknown dtype {self.dtype!r} -- register it via register_dtype()")
        kind = self.dist.get("kind")
        if kind not in DIST_REGISTRY:
            raise ValueError(f"unknown dist kind {kind!r} -- register it via register_dist()")
        if len(self.axes) != len(self.shape):
            raise ValueError(f"axes/shape rank mismatch: {self.axes} vs {self.shape}")


@dataclass
class ParamSpec:
    """A non-tensor row symbol: a scalar/flag/block/group value (the named ROW ENVIRONMENT)."""
    kind: str                       # scalar | scalar_knob | derive | scalar_seq | optional | row-only
    value: Any = None
    expr: str | None = None         # the source expression for derive / scalar_knob


@dataclass
class RowSpec:
    """One explicit shape row."""
    id: str
    role: str                       # public | hidden
    provenance: str                 # synthetic | npukb-borrowed | model-sourced | edge | baseline
    tensors: dict[str, TensorSpec] = field(default_factory=dict)
    params: dict[str, ParamSpec] = field(default_factory=dict)
    structs: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"row {self.id}: bad role {self.role!r} (want {sorted(VALID_ROLES)})")
        if self.provenance not in VALID_PROVENANCE:
            raise ValueError(f"row {self.id}: bad provenance {self.provenance!r}")
        for t in self.tensors.values():
            t.validate()
        for nm, p in self.params.items():
            if p.kind not in VALID_PARAM_KINDS:
                raise ValueError(f"row {self.id}: param {nm!r} bad kind {p.kind!r}")


@dataclass
class TaskShapeSpec:
    """A task's full explicit shape set + the schema version readers gate on."""
    schema_version: int
    rows: list[RowSpec] = field(default_factory=list)

    def validate(self) -> None:
        for r in self.rows:
            r.validate()


# ── bridge from the U2 materializer (grounds the IR in real corpus data) ───────────────────────────

def row_from_materialized(sid: str, role: str, provenance: str, desc: dict[str, dict]) -> RowSpec:
    """Build a RowSpec from `materialize.materialize_row(...)` output.

    NOTE (struct limitation): a ``struct`` input is stored as its raw spec only; the IR does not yet
    model struct fields / refs, so U8 cannot rebuild full build_inputs behaviour from a struct RowSpec
    alone. Harmless today (no ``struct`` inputs in the corpus); revisit when a struct task appears."""
    row = RowSpec(id=sid, role=role, provenance=provenance)
    for name, d in desc.items():
        kind = d["kind"]
        if kind == "tensor":
            row.tensors[name] = TensorSpec(shape=d["shape"], dtype=d["dtype"],
                                           axes=d["axes"], dist=d["dist"])
        elif kind == "struct":
            row.structs[name] = d.get("struct")
        elif kind == "optional":
            row.params[name] = ParamSpec(kind="optional", value=d.get("value"))
        else:   # scalar | scalar_knob | derive | scalar_seq
            row.params[name] = ParamSpec(kind=kind, value=d.get("value"), expr=d.get("expr"))
    return row
