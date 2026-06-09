"""Compat shim — implementation moved to ascendbench._util.atomic."""
from ascendbench._util.atomic import (  # noqa: F401
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    rollback_task,
    snapshot_task_baseline,
)
