"""inputs.py — build oracle/solution input dicts from shapes.yaml.

The Phase C shapes.yaml schema (consumed here):

    size_knobs: {B: 2, L: 8, D: 16}          # default knob values
    public_shapes:
      - {id: P1, name: tiny, knobs: {B: 1, L: 4, D: 8}}
      ...
    hidden_shapes:
      - {id: H1, name: h_odd, knobs: {B: 3, L: 13, D: 40}}
    inputs:
      x:      {dims: [B, L, D], dtype: float32, dist: randn}
      weight: {dims: [D],       dtype: float32, dist: ones}
      eps:    {scalar: 1.0e-5}
      idx:    {dims: [B], dtype: int64, dist: "randint:0:B"}

`build_inputs(shapes_doc, shape_id, seed)` returns the `dict` passed to
`oracle_pytorch.run(inputs)` and to the candidate solution. Tensor specs
are size-knob formulas so one inputs block covers every shape row.

This is the bridge eval.py was missing (was a Phase 2.B TODO): with it,
shapes.yaml is actually executable and the oracle can be run per shape.
"""
from __future__ import annotations

from typing import Any

import torch

_DTYPES = {
    "float32": torch.float32, "float16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "fp16": torch.float16, "fp32": torch.float32,
    "int8": torch.int8, "int32": torch.int32, "int64": torch.int64,
    "long": torch.int64, "bool": torch.bool,
    "float8_e4m3fn": torch.float8_e4m3fn,
}


def _resolve_dims(dims: list[Any], knobs: dict[str, int]) -> list[int]:
    out: list[int] = []
    for d in dims:
        if isinstance(d, int):
            out.append(d)
        elif isinstance(d, str):
            # Simple arithmetic over knob names, e.g. "2*D" / "L+1".
            out.append(int(eval(d, {"__builtins__": {}}, knobs)))  # noqa: S307
        else:
            raise ValueError(f"bad dim {d!r}")
    return out


def _make_tensor(spec: dict[str, Any], knobs: dict[str, int],
                 gen: torch.Generator) -> torch.Tensor:
    dims = _resolve_dims(spec.get("dims", []), knobs)
    dtype = _DTYPES.get(str(spec.get("dtype", "float32")), torch.float32)
    dist = str(spec.get("dist", "randn"))
    shape = tuple(dims) if dims else ()

    if dist == "randn":
        t = torch.randn(shape, generator=gen, dtype=torch.float32)
    elif dist == "rand":
        t = torch.rand(shape, generator=gen, dtype=torch.float32)
    elif dist == "ones":
        t = torch.ones(shape, dtype=torch.float32)
    elif dist == "zeros":
        t = torch.zeros(shape, dtype=torch.float32)
    elif dist == "eye":
        n = shape[-1] if shape else 1
        t = torch.eye(n, dtype=torch.float32)
        if len(shape) > 2:
            t = t.expand(shape).contiguous()
    elif dist == "tril":
        n = shape[-1] if shape else 1
        base = torch.tril(torch.randn(shape, generator=gen, dtype=torch.float32))
        t = base + torch.eye(n, dtype=torch.float32) * (n + 1)  # well-conditioned
    elif dist.startswith("randint:"):
        _, lo, hi = dist.split(":")
        lo_i = int(eval(lo, {"__builtins__": {}}, knobs))  # noqa: S307
        hi_i = max(int(eval(hi, {"__builtins__": {}}, knobs)), lo_i + 1)  # noqa: S307
        t = torch.randint(lo_i, hi_i, shape, generator=gen, dtype=torch.int64)
    elif dist == "softmax_rows":
        t = torch.softmax(torch.randn(shape, generator=gen, dtype=torch.float32), dim=-1)
    elif dist.startswith("offsets"):
        # Monotonic [0, c1, ..., total] int64 of length = shape[-1].
        # `offsets:N`        random increments ≤ N.
        # `offsets:to=<expr>`  final value == eval(expr, knobs) — used
        #   when the offset total must equal another tensor's length
        #   (cu_seqlens[-1] == n_tokens, etc.).
        n = (shape[-1] if shape else 2)
        segs = max(n - 1, 1)
        t = torch.zeros(n, dtype=torch.int64)
        opt = dist.split(":", 1)[1] if ":" in dist else "3"
        if opt.startswith("to="):
            total = int(eval(opt[3:], {"__builtins__": {}}, knobs))  # noqa: S307
            # Distribute exactly `total` over `segs` non-negative
            # increments (empty segments are valid — oracles skip s==e).
            inc = torch.zeros(segs, dtype=torch.int64)
            if total > 0:
                base = total // segs
                inc += base
                inc[: total - base * segs] += 1
            t[1:] = torch.cumsum(inc, 0)
        else:
            step = int(opt)
            inc = torch.randint(1, max(step, 2) + 1, (segs,), generator=gen)
            t[1:] = torch.cumsum(inc, 0)
        return t
    elif dist == "arange":
        n = shape[-1] if shape else 1
        return torch.arange(n, dtype=torch.int64).reshape(shape)
    elif dist.startswith("full:"):
        # Constant-filled int64 tensor; value = eval(expr, knobs). Used
        # when a length vector must match an offsets cumsum (e.g.
        # b_seq_len == per-segment length of b_start_loc).
        val = int(eval(dist.split(":", 1)[1], {"__builtins__": {}}, knobs))  # noqa: S307
        return torch.full(shape, val, dtype=torch.int64)
    else:
        t = torch.randn(shape, generator=gen, dtype=torch.float32)

    if dtype in (torch.int8, torch.int32, torch.int64, torch.bool):
        if dist == "randn" or dist == "rand":
            t = (t.abs() * 8).to(dtype)
        else:
            t = t.to(dtype)
    elif dtype == torch.float8_e4m3fn:
        t = t.clamp(-448, 448).to(dtype)
    else:
        t = t.to(dtype)
    return t


def build_inputs(
    shapes_doc: dict[str, Any], shape_id: str, seed: int = 0,
) -> dict[str, Any]:
    """Construct the input dict for a given shape row.

    Scalars (``{scalar: <value>}``) are passed through verbatim. Tensor
    specs are materialised from size-knob formulas with a fixed seed so
    oracle and candidate see identical inputs.
    """
    rows = (shapes_doc.get("public_shapes", [])
            + shapes_doc.get("hidden_shapes", []))
    row = next((r for r in rows if r.get("id") == shape_id), None)
    if row is None:
        raise KeyError(f"shape id {shape_id!r} not in shapes.yaml")

    knobs = dict(shapes_doc.get("size_knobs", {}))
    knobs.update(row.get("knobs", {}))

    gen = torch.Generator()
    gen.manual_seed(seed)

    out: dict[str, Any] = {}
    for name, spec in (shapes_doc.get("inputs", {}) or {}).items():
        if not isinstance(spec, dict):
            out[name] = spec
            continue
        if "scalar" in spec:
            out[name] = spec["scalar"]
        elif "scalar_knob" in spec:
            out[name] = int(eval(  # noqa: S307
                str(spec["scalar_knob"]), {"__builtins__": {}}, knobs))
        else:
            out[name] = _make_tensor(spec, knobs, gen)
    return out


def shape_ids(shapes_doc: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(public_ids, hidden_ids)."""
    pub = [r["id"] for r in shapes_doc.get("public_shapes", [])]
    hid = [r["id"] for r in shapes_doc.get("hidden_shapes", [])]
    return pub, hid
