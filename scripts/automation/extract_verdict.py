"""Extract a verdict + write strict-9-field result.json + sidecar provenance.

CLI entry-point (Unit 1.7). Parses ``pa_resolve.log`` (cmd=resolve) or
reads ``meta.yaml.phase_a_prime.*`` (cmd=phase_a_verify) and writes:

  <RESULTS_DIR>/<RUN_ID>/result.json       -- 9-field schema
  <RESULTS_DIR>/<RUN_ID>/_provenance.json  -- sidecar audit
  <RESULTS_DIR>/<RUN_ID>/DONE              -- read-barrier touch marker

The reader of these files must wait for ``DONE`` -- it is the barrier
guaranteeing both JSON files are fully written.

result.json schema (strict; ``additionalProperties: false``)::

    {
      "schema_version": "1.0",
      "run_id": "<string>",
      "task_id": "<TASK_ID>",
      "cmd": "resolve" | "phase_a_verify",
      "verdict": "PASS" | "BLOCKED" | "SKIPPED_BY_DESIGN",
      "verdict_class": "shapes-incomplete" | "oracle-mismatch" |
                       "upstream-not-invocable-here" | "no-host-wrapper" |
                       "smoke-fail" | "unexpected-blocked" | null,
      "reason": "<free-form>",
      "produced_at": "<ISO-8601 UTC>",
      "producer": "tier1_reverify" | "daemon"
    }
"""
from __future__ import annotations

# ---- KTD-11 sys.path bootstrap (CLI entry-point) -------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
# --------------------------------------------------------------------------

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import socket  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Any  # noqa: E402

import yaml  # noqa: E402

from automation.lib import atomic  # noqa: E402

logger = logging.getLogger("automation.extract_verdict")

SCHEMA_VERSION = "1.0"

_VALID_VERDICTS = {"PASS", "BLOCKED", "SKIPPED_BY_DESIGN"}
_KNOWN_VERDICT_CLASSES = {
    "shapes-incomplete",
    "oracle-mismatch",
    "upstream-not-invocable-here",
    "no-host-wrapper",
    "smoke-fail",
    "unexpected-blocked",
}
_KNOWN_PRODUCERS = {"tier1_reverify", "daemon"}
_RESULT_SCHEMA_KEYS = {
    "schema_version", "run_id", "task_id", "cmd", "verdict",
    "verdict_class", "reason", "produced_at", "producer",
}


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# resolve_log parser
# ---------------------------------------------------------------------------

_RESOLVE_PASS_RE = re.compile(
    r"^\[(?P<tid>[^\]]+)\]\s+RESOLVED\s+\S+\s+::\s+PASS",
)
_RESOLVE_BLOCKED_RE = re.compile(
    r"^\[(?P<tid>[^\]]+)\]\s+BLOCKED\s+.*?::\s+BLOCKED\((?P<reason>[^)]+)\)",
)


def parse_resolve_log(
    log_path: Path, task_id: str,
) -> tuple[str, str | None, str, str]:
    """Return (verdict, verdict_class, reason, raw_subclass).

    ``raw_subclass`` is the resolver's verbatim subclass string when the
    line is BLOCKED -- the sidecar logs it even if we collapse to the
    fallback ``unexpected-blocked`` verdict_class.
    """
    if not log_path.is_file():
        return ("BLOCKED", "unexpected-blocked",
                f"resolve log not found at {log_path}", "")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if not line.startswith(f"[{task_id}]"):
            continue
        m = _RESOLVE_PASS_RE.match(line)
        if m:
            return ("PASS", None, "resolver PASS", "")
        m = _RESOLVE_BLOCKED_RE.match(line)
        if m:
            raw = m.group("reason").strip()
            # Subclass is the prefix before ':' if present, e.g.
            # "no-verified-upstream-entrypoint: shapes-incomplete"
            if ":" in raw:
                subclass = raw.split(":", 1)[1].strip()
            else:
                subclass = raw
            if subclass in _KNOWN_VERDICT_CLASSES:
                vclass = subclass
            else:
                vclass = "unexpected-blocked"
            return ("BLOCKED", vclass, f"resolver BLOCKED ({raw})", subclass)
        # Other [task] lines -- skip
    return ("BLOCKED", "unexpected-blocked",
            f"no verdict line for {task_id} in resolve log", "")


# ---------------------------------------------------------------------------
# phase_a_verify (meta.yaml) parser
# ---------------------------------------------------------------------------

def parse_phase_a_verify(
    repo_root: Path,
    task_id: str,
    stdout_log: Path | None = None,
) -> tuple[str, str | None, str, str]:
    """Return (verdict, verdict_class, reason, raw_subclass).

    Authoritative source = ``tasks/<level>/<task_id>/meta.yaml`` ->
    ``phase_a_prime.status / reason``. Falls back to ``stdout_log`` if
    meta is missing/invalid.
    """
    meta_path = _find_task_meta(repo_root, task_id)
    if meta_path is not None and meta_path.is_file():
        try:
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            meta = None
        if isinstance(meta, dict):
            pap = meta.get("phase_a_prime") or {}
            status = pap.get("status")
            reason = pap.get("reason", "") or ""
            if status == "PASS":
                return ("PASS", None, "phase_a_prime.status=PASS", "")
            if status == "SKIPPED_BY_DESIGN":
                return ("SKIPPED_BY_DESIGN", None,
                        f"phase_a_prime.status=SKIPPED_BY_DESIGN ({reason})", "")
            if status == "BLOCKED":
                # Try to extract a subclass from the reason
                subclass = _extract_subclass_from_reason(reason)
                vclass = subclass if subclass in _KNOWN_VERDICT_CLASSES else "unexpected-blocked"
                return ("BLOCKED", vclass,
                        f"phase_a_prime.status=BLOCKED ({reason})", subclass)
            # status is None or unknown -- fall through to stdout fallback
    if stdout_log is not None and stdout_log.is_file():
        text = stdout_log.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"S2Result\(status=(\w+)[^,]*,\s*reason=([^)]+)\)", text)
        if m:
            status = m.group(1)
            reason = m.group(2).strip().strip("'\"")
            if status == "PASS":
                return ("PASS", None, "stdout S2Result=PASS", "")
            if status == "SKIPPED_BY_DESIGN":
                return ("SKIPPED_BY_DESIGN", None,
                        f"stdout S2Result=SKIPPED_BY_DESIGN ({reason})", "")
            subclass = _extract_subclass_from_reason(reason)
            vclass = subclass if subclass in _KNOWN_VERDICT_CLASSES else "unexpected-blocked"
            return ("BLOCKED", vclass, f"stdout S2Result=BLOCKED ({reason})", subclass)
    return ("BLOCKED", "unexpected-blocked",
            "no verdict source (no meta.yaml.phase_a_prime, no stdout fallback)",
            "")


def _find_task_meta(repo_root: Path, task_id: str) -> Path | None:
    """Best-effort lookup of ``tasks/<level>/<task_id>/meta.yaml``."""
    for level_dir in (repo_root / "tasks").glob("*"):
        if not level_dir.is_dir():
            continue
        candidate = level_dir / task_id / "meta.yaml"
        if candidate.is_file():
            return candidate
    return None


def _extract_subclass_from_reason(reason: str) -> str:
    for known in _KNOWN_VERDICT_CLASSES:
        if known in reason:
            return known
    return ""


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_result_schema(result: dict[str, Any]) -> None:
    keys = set(result.keys())
    extra = keys - _RESULT_SCHEMA_KEYS
    missing = _RESULT_SCHEMA_KEYS - keys
    if extra or missing:
        raise ValueError(
            f"result.json schema violation: extra={sorted(extra)}, "
            f"missing={sorted(missing)}"
        )
    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: expected {SCHEMA_VERSION!r}, "
            f"got {result['schema_version']!r}"
        )
    if result["verdict"] not in _VALID_VERDICTS:
        raise ValueError(f"verdict not in {sorted(_VALID_VERDICTS)}: "
                         f"{result['verdict']!r}")
    vc = result["verdict_class"]
    if vc is not None and vc not in _KNOWN_VERDICT_CLASSES:
        raise ValueError(f"verdict_class not in known set: {vc!r}")
    if result["cmd"] not in ("resolve", "phase_a_verify"):
        raise ValueError(f"cmd not in known set: {result['cmd']!r}")
    if result["producer"] not in _KNOWN_PRODUCERS:
        raise ValueError(f"producer not in {sorted(_KNOWN_PRODUCERS)}: "
                         f"{result['producer']!r}")


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_verdict(
    *,
    task_id: str,
    cmd: str,
    run_id: str,
    producer: str,
    out_dir: Path,
    verdict: str,
    verdict_class: str | None,
    reason: str,
    raw_subclass: str = "",
    started_at: str = "",
    extra_provenance: dict[str, Any] | None = None,
) -> Path:
    """Write result.json + _provenance.json + DONE atomically."""
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "task_id": task_id,
        "cmd": cmd,
        "verdict": verdict,
        "verdict_class": verdict_class,
        "reason": reason,
        "produced_at": _iso_utc_now(),
        "producer": producer,
    }
    validate_result_schema(result)

    code_fp = _sha256_file(Path(__file__))
    provenance = {
        "code_fingerprint": code_fp,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "started_at": started_at or _iso_utc_now(),
        "finished_at": _iso_utc_now(),
        "raw_verdict_class": raw_subclass,
        "extra": extra_provenance or {},
    }

    # Order: provenance first, then result, then DONE (read barrier).
    try:
        atomic.atomic_write_json(run_dir / "_provenance.json", provenance)
        atomic.atomic_write_json(run_dir / "result.json", result)
    except Exception:
        # rollback any partial files
        for f in ("_provenance.json", "result.json", "DONE"):
            try:
                (run_dir / f).unlink(missing_ok=True)
            except OSError:
                pass
        raise

    done = run_dir / "DONE"
    done.touch()
    return run_dir / "result.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, dest="task_id")
    parser.add_argument("--cmd", required=True, choices=["resolve", "phase_a_verify"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--producer", required=True, choices=sorted(_KNOWN_PRODUCERS),
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--resolve-log", type=Path, default=None)
    parser.add_argument("--stdout-log", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    args = parser.parse_args(argv)

    started_at = _iso_utc_now()
    if args.cmd == "resolve":
        log = args.resolve_log
        if log is None:
            log = args.repo_root / "results" / "gpu_pipeline" / "pa_resolve.log"
        verdict, vclass, reason, raw_sub = parse_resolve_log(log, args.task_id)
    else:
        verdict, vclass, reason, raw_sub = parse_phase_a_verify(
            args.repo_root, args.task_id, stdout_log=args.stdout_log,
        )

    write_verdict(
        task_id=args.task_id,
        cmd=args.cmd,
        run_id=args.run_id,
        producer=args.producer,
        out_dir=args.out_dir,
        verdict=verdict,
        verdict_class=vclass,
        reason=reason,
        raw_subclass=raw_sub,
        started_at=started_at,
    )
    print(json.dumps({
        "task": args.task_id, "verdict": verdict, "verdict_class": vclass,
        "run_id": args.run_id,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
