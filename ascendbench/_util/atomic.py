"""Atomic write + baseline snapshot + rsync rollback helpers (Unit 1.1).

INTERNAL bench-team automation lib module. Imported by orchestrator
entry-points after KTD-11 ``sys.path`` bootstrap. Pure stdlib + PyYAML.

Public API
----------
* ``atomic_write_yaml(path, data)``          -- POSIX-atomic YAML writer.
* ``atomic_write_json(path, obj)``           -- POSIX-atomic JSON writer.
* ``atomic_write_text(path, text)``          -- POSIX-atomic text writer.
* ``snapshot_task_baseline(task_dir, dest_tar_gz, manifest_json)``
* ``rollback_task(task_dir, baseline_tar_gz, manifest_json)``
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import socket
import subprocess
import tarfile
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("automation.lib.atomic")


# ----------------------------------------------------------------------------
# Atomic writers
# ----------------------------------------------------------------------------

def _tmp_suffix() -> str:
    """Return a host/pid/uuid8 tmp suffix that is safe across NFS-style
    filesystems where dotfiles inherit ACLs from parent dir."""
    return f".tmp.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex[:8]}"


def _atomic_replace(path: Path, payload: str) -> None:
    """Write ``payload`` (str -- UTF-8 encoded internally) atomically.

    Thin wrapper around :func:`atomic_write_bytes`; preserved for the
    historical str-payload call sites. New binary writers should use
    :func:`atomic_write_bytes` directly.
    """
    atomic_write_bytes(path, payload.encode("utf-8"))


def atomic_write_bytes(
    path: os.PathLike[str] | str,
    payload: bytes,
    *,
    mode: int = 0o644,
) -> None:
    """Public atomic *bytes* writer.

    Same durability-correct sequence as the historical
    :func:`_atomic_replace` (tmp sibling -> fsync -> os.replace ->
    parent dir fsync), now accepting a raw ``bytes`` payload so
    callers writing safetensors / signature blobs don't need to
    round-trip through str.

    POSIX-atomic sequence:
      1. open tmp + write bytes
      2. ``os.fsync(tmp_fd)`` to push file data to disk
      3. close tmp
      4. ``os.replace(tmp, target)`` (atomic rename)
      5. ``os.fsync(parent_dir_fd)`` so the directory entry is durable

    Parent dir is created if missing. ``mode`` is the POSIX permission
    bits applied at creation -- the M3.2 sealed_golden writer passes
    ``0o600`` so the sealed artifact is unreadable by submitter uid.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(
            "atomic_write_bytes payload must be bytes-like (got "
            f"{type(payload).__name__})"
        )
    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + _tmp_suffix())
    try:
        # Step 1-3: write tmp + fsync data
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        try:
            os.write(fd, bytes(payload))
            os.fsync(fd)
        finally:
            os.close(fd)
        # Step 5: atomic rename
        os.replace(tmp, path)
        # Step 6: durable directory entry. Best-effort on platforms
        # where opening a directory is not supported (Windows); on
        # POSIX (the only target) this MUST succeed.
        try:
            dir_fd = os.open(parent, os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except Exception:
        # On error, best-effort clean up the tmp file (do not let it
        # accumulate across retries).
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def atomic_write_yaml(path: os.PathLike[str] | str, data: Any) -> None:
    """Atomically write ``data`` as YAML to ``path``.

    Always uses ``sort_keys=False, allow_unicode=True`` (bench convention,
    matches ``scripts/phase_a_prime.py:1071-1073``)."""
    path = Path(path)
    payload = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    _atomic_replace(path, payload)


def atomic_write_json(path: os.PathLike[str] | str, obj: Any) -> None:
    """Atomically write ``obj`` as pretty JSON to ``path``.

    Uses ``indent=2, sort_keys=True`` so diffs across runs stay stable."""
    path = Path(path)
    payload = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    _atomic_replace(path, payload)


def atomic_write_text(path: os.PathLike[str] | str, text: str) -> None:
    """Atomically write a plain text payload (e.g. ``oracle_pytorch.py``)."""
    path = Path(path)
    _atomic_replace(path, text)


# ----------------------------------------------------------------------------
# Baseline snapshot + rollback
# ----------------------------------------------------------------------------

def _iso_utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _list_files(task_dir: Path) -> list[str]:
    """Return sorted list of regular files under ``task_dir`` as paths
    relative to ``task_dir``."""
    out: list[str] = []
    for root, _dirs, files in os.walk(task_dir):
        for name in files:
            p = Path(root) / name
            out.append(str(p.relative_to(task_dir)))
    out.sort()
    return out


def snapshot_task_baseline(
    task_dir: os.PathLike[str] | str,
    dest_tar_gz: os.PathLike[str] | str,
    manifest_json: os.PathLike[str] | str,
) -> dict[str, Any]:
    """Tar+gzip ``task_dir`` to ``dest_tar_gz``, write file manifest.

    Manifest schema::

        {
          "task_id": "<task_dir.name>",
          "baseline_sha256": "<sha256(dest_tar_gz)>",
          "file_list": ["<relpath>", ...]  # sorted, regular files only
          "snapshot_at": "<ISO UTC>",
        }
    """
    task_dir = Path(task_dir).resolve()
    dest_tar_gz = Path(dest_tar_gz)
    manifest_json = Path(manifest_json)

    if not task_dir.is_dir():
        raise FileNotFoundError(f"task_dir does not exist: {task_dir}")

    dest_tar_gz.parent.mkdir(parents=True, exist_ok=True)

    # tar -czf <tar> -C <task_dir.parent> <task_dir.name>
    cmd = [
        "tar",
        "-czf",
        str(dest_tar_gz),
        "-C",
        str(task_dir.parent),
        task_dir.name,
    ]
    rc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if rc.returncode != 0:
        raise RuntimeError(
            f"tar failed (rc={rc.returncode}): {rc.stderr.strip()}"
        )

    manifest = {
        "task_id": task_dir.name,
        "baseline_sha256": _sha256_file(dest_tar_gz),
        "file_list": _list_files(task_dir),
        "snapshot_at": _iso_utc_now(),
    }
    atomic_write_json(manifest_json, manifest)
    logger.debug(
        "snapshot_task_baseline: task=%s tar=%s files=%d",
        task_dir.name, dest_tar_gz, len(manifest["file_list"]),
    )
    return manifest


def rollback_task(
    task_dir: os.PathLike[str] | str,
    baseline_tar_gz: os.PathLike[str] | str,
    manifest_json: os.PathLike[str] | str,
) -> None:
    """Restore ``task_dir`` byte-for-byte from ``baseline_tar_gz`` using
    ``rsync -a --delete``. Verifies file_list after.

    Steps:
      1. Extract tarball to a fresh tmp dir.
      2. ``rsync -a --delete <tmp>/<task_name>/ <task_dir>/``
      3. Verify ``find <task_dir>`` matches manifest ``file_list``.
      4. Remove tmp dir on success. On verify mismatch, leave tmp dir
         around for forensics + raise.
    """
    task_dir = Path(task_dir).resolve()
    baseline_tar_gz = Path(baseline_tar_gz)
    manifest_json = Path(manifest_json)

    if not baseline_tar_gz.is_file():
        raise FileNotFoundError(
            f"baseline tarball missing: {baseline_tar_gz}"
        )
    if not manifest_json.is_file():
        raise FileNotFoundError(f"manifest missing: {manifest_json}")

    # tarfile.open will raise ReadError on corrupt archive; we let it
    # propagate (caller logs + handles).
    with tarfile.open(baseline_tar_gz, "r:gz") as tf:
        tmp_baseline = Path(tempfile.mkdtemp(prefix="tab-rollback-"))
        try:
            # ``filter="data"`` rejects absolute paths / device files / world-writable bits.
            # Required by Python 3.14+ default; safe for our trusted tar archives.
            tf.extractall(tmp_baseline, filter="data")
        except Exception:
            shutil.rmtree(tmp_baseline, ignore_errors=True)
            raise

    extracted_root = tmp_baseline / task_dir.name
    if not extracted_root.is_dir():
        shutil.rmtree(tmp_baseline, ignore_errors=True)
        raise RuntimeError(
            f"baseline tar missing inner dir {task_dir.name!r} "
            f"(extracted under {tmp_baseline})"
        )

    # rsync -a --delete will mirror src into dst, removing any extra
    # files in dst that aren't in src (this is what restores us to the
    # captured baseline byte-for-byte).
    task_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync",
        "-a",
        "--delete",
        f"{extracted_root}/",
        f"{task_dir}/",
    ]
    rc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if rc.returncode != 0:
        # Leave tmp_baseline behind for forensics.
        raise RuntimeError(
            f"rsync failed (rc={rc.returncode}): {rc.stderr.strip()}; "
            f"tmp_baseline preserved at {tmp_baseline}"
        )

    # Verify file_list
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    expected = sorted(manifest.get("file_list", []))
    actual = _list_files(task_dir)
    if actual != expected:
        # Forensics: preserve tmp_baseline + raise.
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(
            f"rollback file_list mismatch (missing={missing[:5]}, "
            f"extra={extra[:5]}); tmp_baseline preserved at {tmp_baseline}"
        )

    shutil.rmtree(tmp_baseline, ignore_errors=True)
    logger.debug(
        "rollback_task: task=%s tar=%s restored %d files",
        task_dir.name, baseline_tar_gz, len(expected),
    )


__all__ = [
    "atomic_write_json",
    "atomic_write_text",
    "atomic_write_yaml",
    "rollback_task",
    "snapshot_task_baseline",
]
