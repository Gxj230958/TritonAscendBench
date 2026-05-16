"""dataset.py — TaskDef loader.

Reads `tasks/level{1,2,3}/<task_id>/` per `specs/task_api_contract.md`:
- meta.yaml      (mandatory metadata)
- shapes.yaml    (public shapes)
- contract.md    (human-readable contract)
- kernel.py      (frozen GPU reference)
- oracle_pytorch.py
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = REPO_ROOT / "tasks"


@dataclass
class ShapeSpec:
    id: str
    name: str
    params: dict[str, Any]


@dataclass
class TaskDef:
    task_id: str
    level: int
    task_dir: Path
    meta: dict[str, Any]
    public_shapes: list[ShapeSpec]
    contract_md: str
    kernel_py: str
    kernel_py_sha256: str
    oracle_module_name: str = "oracle_pytorch"

    @property
    def tolerance_class(self) -> str:
        # spec schema: top-level `tolerance_class`
        return self.meta.get("tolerance_class") or self.meta["tolerance"]["class"]

    @property
    def oracle_tier(self) -> str:
        # spec schema: top-level `oracle_simplified` (string, not nested .tier)
        v = self.meta.get("oracle_simplified")
        if isinstance(v, str):
            return v
        return v["tier"]                                  # back-compat fallback

    @property
    def upstream_sha(self) -> str:
        return self.meta["upstream"]["commit_sha"]


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_task(task_id: str) -> TaskDef:
    """Locate `tasks/level{1,2,3}/<task_id>/` and parse it."""
    task_dir: Path | None = None
    for level in ("level1", "level2", "level3"):
        candidate = TASKS_ROOT / level / task_id
        if candidate.is_dir():
            task_dir = candidate
            break
    if task_dir is None:
        raise FileNotFoundError(f"Task {task_id!r} not found under {TASKS_ROOT}/")

    meta = yaml.safe_load((task_dir / "meta.yaml").read_text())
    shapes_doc = yaml.safe_load((task_dir / "shapes.yaml").read_text())
    contract_md = (task_dir / "contract.md").read_text()
    kernel_py = (task_dir / "kernel.py").read_text()

    public_shapes = [
        ShapeSpec(
            id=s["id"],
            name=s["name"],
            params={k: v for k, v in s.items() if k not in {"id", "name"}},
        )
        for s in shapes_doc["public_shapes"]
    ]

    raw_level = meta["level"]
    if isinstance(raw_level, str):
        level_str = raw_level.upper().lstrip("L")
        level_int = int(level_str) if level_str.isdigit() else 0
    else:
        level_int = int(raw_level)

    return TaskDef(
        task_id=task_id,
        level=level_int,
        task_dir=task_dir,
        meta=meta,
        public_shapes=public_shapes,
        contract_md=contract_md,
        kernel_py=kernel_py,
        kernel_py_sha256=_sha256_text(kernel_py),
    )


def iter_all_tasks() -> list[TaskDef]:
    out: list[TaskDef] = []
    for level in ("level1", "level2", "level3"):
        level_dir = TASKS_ROOT / level
        if not level_dir.is_dir():
            continue
        for d in sorted(level_dir.iterdir()):
            if d.is_dir() and not d.name.startswith("_"):
                out.append(load_task(d.name))
    return out
