#!/usr/bin/env python3
"""verify_version_matrix.py — sanity-check infra/version_matrix.toml.

Per specs/process_isolation.md. Catches:
- Missing required recorded-stack fields
- Unparseable TOML
- Unresolved placeholder values (`<TBD>`, `<frozen-...>`, etc.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:                              # 3.10 fallback
    import tomli as tomllib                              # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO_ROOT / "infra" / "version_matrix.toml"

REQUIRED_KEYS = [
    ("npu", "cann"),
    ("npu", "torch_npu"),
    ("npu", "triton_ascend"),
    ("gpu", "cuda"),
    ("gpu", "python"),
    ("gpu", "torch"),
    ("gpu", "triton"),
    ("gpu_capture", "cuda"),
    ("gpu_capture", "torch"),
    ("gpu_capture", "triton"),
    ("gpu_capture", "python"),
    ("gpu_capture", "device_identity"),
    ("gpu_capture", "compute_capability"),
    ("gpu_capture", "fp8_semantics"),
    ("gpu_capture", "extra_pins"),
    ("python_deps", "numpy"),
    ("python_deps", "pyyaml"),
    ("python_deps", "safetensors"),
    ("python_deps", "click"),
    ("python_deps", "rich"),
    ("hardware", "910b"),
    ("protocol",),
]

PLACEHOLDER_PATTERNS = ("<TBD>", "<frozen-", "<set-by-", "<fill-after-", "<no-git")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=str(DEFAULT_PATH))
    args = parser.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f"ERROR: not found: {p}", file=sys.stderr)
        return 1

    try:
        data = tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError as e:
        print(f"ERROR: cannot parse TOML: {e}", file=sys.stderr)
        return 1

    missing: list[str] = []
    placeholders: list[tuple[str, str]] = []

    def _walk(d: dict, path: tuple[str, ...] = ()) -> None:
        for k, v in d.items():
            here = path + (k,)
            if isinstance(v, dict):
                _walk(v, here)
            elif isinstance(v, str):
                for pat in PLACEHOLDER_PATTERNS:
                    if pat in v:
                        placeholders.append((".".join(here), v))
                        break

    _walk(data)

    for keypath in REQUIRED_KEYS:
        d: dict = data
        for seg in keypath:
            if not isinstance(d, dict) or seg not in d:
                missing.append(".".join(keypath))
                break
            d = d[seg]

    ok = True
    if missing:
        ok = False
        for k in missing:
            print(f"ERROR: missing required key {k}", file=sys.stderr)
    if placeholders:
        ok = False
        for path, val in placeholders:
            print(f"ERROR: placeholder in {path}: {val!r}", file=sys.stderr)

    if ok:
        print(f"version_matrix.toml OK: {p}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
