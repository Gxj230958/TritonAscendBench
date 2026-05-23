"""GPFS POSIX behaviour spike (M0 Unit 0.3).

Two-host coordinated spike that empirically measures the actual atomicity
and cross-host visibility of POSIX primitives we plan to rely on for the
queue / lease / lock libraries:

    1. GPFS inode consistency           -- same st_dev / st_ino / mount on both hosts
    2. ``os.rename`` cross-host atomicity -- dst-appears vs src-disappears timing
    3. ``os.mkdir`` cross-host race      -- 100x concurrent mkdir, expect 1 success + 99 EEXIST
    4. ``os.symlink`` cross-host visibility -- read-back delay distribution
    5. heartbeat ``mtime`` jitter        -- A writes mtime, B reads, P50/P95/P99 lag
    6. multi-host tmp file trample       -- ``<file>.tmp.<host>.<pid>.<uuid8>`` collision-freeness

Output: a Markdown file with one statistic block per probe, plus the
recommended threshold values that feed `automation/lib/queue.py` and
`automation/lib/lock.py` defaults.

Status: SKELETON. The six probes below are NOT yet implemented; each
raises ``NotImplementedError`` when invoked. Running this script with
``--peer-host`` exits non-zero until the probes are filled in. Running
without ``--peer-host`` also exits non-zero (the spike requires a
cross-host peer by construction). This is intentional -- the M0 plan
forbids fabricated statistics; an unimplemented spike MUST fail loud
rather than emit a markdown report with ``anomaly_count = 0`` /
``P50 = P95 = P99 = 0`` placeholders that could be mistaken for real
empirical evidence.

See plan §Unit 0.3 for required probe semantics.

Usage::

    # On host A:
    python3 scripts/automation/spike/gpfs_posix_spike.py \\
        --N 100 --peer-host B --out docs/release/internal-gpfs-spike-results.md

    # On host B (mirror, swap roles):
    python3 scripts/automation/spike/gpfs_posix_spike.py \\
        --N 100 --peer-host A --out docs/release/internal-gpfs-spike-results-B.md

This file is INTERNAL (export-ignored). It is intentionally a one-shot
script -- once the spike data lands in the results markdown the script
remains available for re-running whenever the GPFS volume changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

# --- KTD-11 sys.path bootstrap (CLI entry-point) ---------------------------
# spike/ lives under scripts/automation/spike/, so parents[3] == <repo_root>.
# The bootstrap MUST run before any ``from automation...`` import below,
# hence the ``E402`` noqa on the subsequent stdlib imports.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
# Subsequent imports (e.g. ``from automation.lib import ...``) resolve once
# the bootstrap above runs. This spike itself does not depend on lib/ yet.
# ---------------------------------------------------------------------------

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Any  # noqa: E402


@dataclass
class ProbeStat:
    """One probe's measurement summary -- the schema the spike results
    markdown is keyed off."""

    name: str
    N: int
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0
    anomaly_count: int = 0
    decision_threshold: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_markdown_block(self) -> str:
        lines = [
            f"### {self.name}",
            "",
            "| Field               | Value             |",
            "|---------------------|-------------------|",
            f"| N                   | {self.N}          |",
            f"| p50 (ms)            | {self.p50_ms:.3f} |",
            f"| p95 (ms)            | {self.p95_ms:.3f} |",
            f"| p99 (ms)            | {self.p99_ms:.3f} |",
            f"| max (ms)            | {self.max_ms:.3f} |",
            f"| anomaly_count       | {self.anomaly_count} |",
            f"| decision_threshold  | {self.decision_threshold} |",
        ]
        if self.extra:
            lines.append("")
            lines.append("Extra:")
            lines.append("```json")
            lines.append(json.dumps(self.extra, indent=2, sort_keys=True))
            lines.append("```")
        lines.append("")
        return "\n".join(lines)


def _percentiles(samples_ms: list[float]) -> tuple[float, float, float, float]:
    if not samples_ms:
        return (0.0, 0.0, 0.0, 0.0)
    s = sorted(samples_ms)

    def pct(p: float) -> float:
        if len(s) == 1:
            return s[0]
        # Linear interpolation (statistics.quantiles needs >=2 datapoints).
        idx = (len(s) - 1) * (p / 100.0)
        lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
        frac = idx - lo
        return s[lo] + (s[hi] - s[lo]) * frac

    return (pct(50), pct(95), pct(99), max(s))


# --- Probe stubs -----------------------------------------------------------
# Real spike execution requires two hosts coordinated over the same GPFS
# mount. The probes below sketch the measurement protocol. They are NOT
# implemented yet -- each raises ``NotImplementedError`` so that an
# unimplemented run fails loud rather than emitting a fake-zero markdown.

_PROBE_NOT_IMPLEMENTED = (
    "Probe {name} is not implemented yet -- see plan §Unit 0.3 for the "
    "required cross-host probe semantics. This script is a skeleton; "
    "running it MUST fail loud until the probes are filled in."
)


def probe_inode_consistency(workdir: Path, peer_host: str | None, n: int) -> ProbeStat:
    """Probe 1: device + inode + mount agreement across hosts.

    Local side stats the path; remote side (over SSH) stats the same path
    and the spike records ``st_dev``, ``st_ino``, and the mount table
    entry for that path. They must match. P95/P99 lag here is symbolic
    (the probe is a one-shot consistency check, not a latency
    measurement).
    """
    raise NotImplementedError(_PROBE_NOT_IMPLEMENTED.format(name="1 (inode consistency)"))


def probe_rename_atomicity(workdir: Path, peer_host: str | None, n: int) -> ProbeStat:
    """Probe 2: ``os.rename(tmp, dst)`` cross-host atomicity.

    Local writes ``tmp`` and renames to ``dst``. Peer polls every 1ms
    recording the timestamp at which ``dst`` first appears and at which
    ``tmp`` last existed. POSIX requires the two timestamps to coincide;
    GPFS client caches can introduce delay.

    Output threshold candidate: ``HEARTBEAT_GRACE_S = max(60, P99 * 3)``.
    """
    raise NotImplementedError(_PROBE_NOT_IMPLEMENTED.format(name="2 (rename atomicity)"))


def probe_mkdir_race(workdir: Path, peer_host: str | None, n: int) -> ProbeStat:
    """Probe 3: ``os.mkdir(same_path)`` race -- expect 1 win + (N-1) ``FileExistsError``.

    multiprocessing barrier synchronises start across both hosts via a
    pre-staged barrier file in GPFS itself.
    """
    raise NotImplementedError(_PROBE_NOT_IMPLEMENTED.format(name="3 (mkdir race)"))


def probe_symlink_visibility(workdir: Path, peer_host: str | None, n: int) -> ProbeStat:
    """Probe 4: ``os.symlink(target, link)`` local; peer ``os.readlink(link)``.

    Measures read-back delay distribution.
    """
    raise NotImplementedError(_PROBE_NOT_IMPLEMENTED.format(name="4 (symlink visibility)"))


def probe_heartbeat_mtime(workdir: Path, peer_host: str | None, n: int) -> ProbeStat:
    """Probe 5: heartbeat ``mtime`` jitter cross-host.

    Local touches a heartbeat file every 5min (N iterations); peer reads
    ``stat.st_mtime`` and records (local_wall_time_when_touch_returned -
    peer_observed_mtime). Output feeds ``HEARTBEAT_INTERVAL`` /
    ``HEARTBEAT_GRACE_S`` defaults.
    """
    raise NotImplementedError(_PROBE_NOT_IMPLEMENTED.format(name="5 (heartbeat mtime)"))


def probe_tmp_file_trample(workdir: Path, peer_host: str | None, n: int) -> ProbeStat:
    """Probe 6: ``<file>.tmp.<host>.<pid>.<uuid8>`` collision-freeness.

    Both hosts spawn N writers in parallel, each producing a unique tmp
    name prefix, then rename to the common dst. The probe asserts no
    writer's tmp file is observed by the other host until rename
    succeeds, and the dst's final content matches one writer's payload.

    The expected prefix shape (host + pid + uuid8) is exercised here so
    the imports stay honest; a real implementation would build N of
    these and stage cross-host writers.
    """
    _ = f".tmp.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    raise NotImplementedError(_PROBE_NOT_IMPLEMENTED.format(name="6 (tmp file trample)"))


# --- Aggregation + markdown --------------------------------------------------

_PROBE_FUNCS = (
    probe_inode_consistency,
    probe_rename_atomicity,
    probe_mkdir_race,
    probe_symlink_visibility,
    probe_heartbeat_mtime,
    probe_tmp_file_trample,
)


def run_all_probes(workdir: Path, peer_host: str, n: int) -> list[ProbeStat]:
    """Invoke each probe in order. The skeleton implementation lets the
    first ``NotImplementedError`` propagate so the caller surfaces the
    exact unimplemented probe rather than papering over it with a
    zero-filled ProbeStat."""
    return [probe(workdir, peer_host, n) for probe in _PROBE_FUNCS]


def render_results_markdown(
    stats: list[ProbeStat], peer_host: str | None, n: int
) -> str:
    header = [
        "# INTERNAL: GPFS POSIX spike results",
        "",
        "> M0 Unit 0.3 -- empirical atomicity / visibility data feeding the",
        "> `automation/lib/queue.py` and `automation/lib/lock.py` default",
        "> parameter choices (`HEARTBEAT_INTERVAL`, `LEASE_TTL`,",
        "> `HEARTBEAT_GRACE_S`, reaper miss count `N`).",
        "",
        f"- host: `{socket.gethostname()}`",
        f"- peer host: `{peer_host or '<not provided -- dry-run only>'}`",
        f"- N: {n}",
        f"- timestamp (epoch): {int(time.time())}",
        "",
        "Each probe block lists P50 / P95 / P99 / max sample lag and the",
        "decision threshold the probe is gating. `anomaly_count > 0` blocks",
        "promotion to M1.",
        "",
    ]
    blocks = [s.to_markdown_block() for s in stats]
    return "\n".join(header + blocks)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPFS POSIX behaviour spike (M0 Unit 0.3)"
    )
    parser.add_argument(
        "--N",
        dest="n",
        type=int,
        default=100,
        help="Sample count per probe (default 100).",
    )
    parser.add_argument(
        "--peer-host",
        type=str,
        default=None,
        help="SSH-reachable peer hostname for the cross-host probes. "
        "REQUIRED -- omitting it exits non-zero (the spike has no "
        "single-host fallback; cross-host atomicity / visibility is the "
        "whole point of the spike).",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="GPFS-mounted working directory shared between the two hosts. "
        "Defaults to a temp dir under the current GPFS mount.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output Markdown path. Defaults to "
        "docs/release/internal-gpfs-spike-results.md under the repo root.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Fail loud #1: cross-host probes by definition need a peer host. The
    # M0 plan forbids fabricated statistics, so a dry-run path that emits
    # a markdown with ``anomaly_count = 0`` placeholders is NOT allowed.
    if not args.peer_host:
        print(
            "ERROR: gpfs_posix_spike requires --peer-host (cross-host probes). "
            "This script is a skeleton -- the 6 probes are not yet implemented. "
            "See plan §Unit 0.3 for required probe semantics.",
            file=sys.stderr,
        )
        return 2

    workdir = args.workdir or (_REPO_ROOT / ".cache" / "gpfs_spike")
    workdir.mkdir(parents=True, exist_ok=True)
    out = args.out or (_REPO_ROOT / "docs" / "release" / "internal-gpfs-spike-results.md")

    # Fail loud #2: probes are unimplemented. Surface the exact probe
    # that failed; do NOT emit a fake-zero markdown.
    try:
        stats = run_all_probes(workdir, args.peer_host, args.n)
    except NotImplementedError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    md = render_results_markdown(stats, args.peer_host, args.n)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")

    print(f"spike complete -- results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
