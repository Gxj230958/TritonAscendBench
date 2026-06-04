"""Apply production-dtype change specs (from the apply-spec workflow).

Deterministic: applies shape_flips (key->dtype) to shapes.yaml, then groups by
oracle_change so the non-trivial oracle edits are surfaced (not auto-applied).
Run: python3 scripts/_apply_dtype_specs.py <specs.json>
"""
import json
import re
import sys
from pathlib import Path


def _task_dir(task: str) -> Path | None:
    import glob
    ds = glob.glob(f"tasks/level*/{task}")
    return Path(ds[0]) if ds else None


def apply_shape_flips(task: str, flips: list[dict]) -> str:
    d = _task_dir(task)
    if d is None:
        return "NODIR"
    p = d / "shapes.yaml"
    lines = p.read_text().split("\n")
    want = {f["key"]: f["dtype"] for f in flips}
    out, in_in, cur, changed = [], False, None, []
    for ln in lines:
        if re.match(r"^inputs:\s*$", ln):
            in_in = True
            out.append(ln)
            continue
        if in_in and re.match(r"^[a-z_]+:", ln):
            in_in = False
        if in_in:
            mk = re.match(r"^  ([A-Za-z_]\w*):\s*$", ln)
            if mk:
                cur = mk.group(1)
            md = re.match(r"^(\s+dtype:\s*)\w+\s*$", ln)
            if md and cur in want:
                out.append(md.group(1) + want[cur])
                changed.append(f"{cur}->{want[cur]}")
                continue
        out.append(ln)
    p.write_text("\n".join(out))
    return f"flipped {changed}" if changed else "no matching key"


def main() -> int:
    specs = json.load(open(sys.argv[1]))
    groups: dict[str, list] = {}
    for s in specs:
        t = s["task"]
        if s["production_dtype"] == "keep_float32":
            groups.setdefault("keep_fp32", []).append(t)
            continue
        msg = apply_shape_flips(t, s.get("shape_flips") or [])
        oc = s.get("oracle_change", "none")
        groups.setdefault(oc, []).append((t, msg, s.get("oracle_detail", "")))
    for k, v in groups.items():
        print(f"\n### oracle_change={k}: {len(v)}")
        for item in v:
            if isinstance(item, tuple):
                print(f"  {item[0]:46s} {item[1]}"
                      + (f"  :: {item[2][:90]}" if item[2] else ""))
            else:
                print(f"  {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
