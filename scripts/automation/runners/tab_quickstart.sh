#!/usr/bin/env bash
# Operator wrapper for bench acceptance gate.
# Machine targets: 'setup'/'resolve' run on A; 'verify-c'/'smoke'/'full' run on B;
# 'status'/'clean'/'help' run on either. In single-machine mode (A=B), all on one host.
#
# Wraps the five lower-level runner scripts (setup_machine_a_gpu.sh,
# run_acceptance_gate.sh, ...) and the resolver entrypoint into a small
# set of one-shot subcommands so an operator only configures three
# variables (TAB_GPFS_ROOT / TAB_C_HOST / TAB_C_CLI_CMD) in a single
# conf file and then drives the whole gate with:
#
#   bash scripts/automation/runners/tab_quickstart.sh <subcommand>
#
# The wrapper itself is INTERNAL (matches the scripts/automation/
# export-ignore pattern in .gitattributes / .dockerignore).
#
# Variable precedence: env > conf file > script defaults.
# Conf file resolution (high to low):
#   1. $TAB_CONFIG env var (explicit override)
#   2. <repo-root>/tab_quickstart.conf  (recommended; gitignored)
#   3. ~/.tab-quickstart.conf            (legacy fallback)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNERS_DIR="$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Resolve config file (priority: $TAB_CONFIG env > repo-local > home > none).
if [[ -n "${TAB_CONFIG:-}" ]]; then
  CONFIG_FILE="$TAB_CONFIG"
  CONFIG_SOURCE="TAB_CONFIG env"
elif [[ -f "$REPO_ROOT/tab_quickstart.conf" ]]; then
  CONFIG_FILE="$REPO_ROOT/tab_quickstart.conf"
  CONFIG_SOURCE="repo-local"
elif [[ -f "$HOME/.tab-quickstart.conf" ]]; then
  CONFIG_FILE="$HOME/.tab-quickstart.conf"
  CONFIG_SOURCE="home (legacy fallback)"
else
  CONFIG_FILE=""
  CONFIG_SOURCE="(none -- required vars must come from env)"
fi
EXAMPLE_CONF_PATH="scripts/automation/runners/tab_quickstart.conf.example"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
step() { echo "[step] $*"; }
ok()   { echo "[ok] $*"; }
warn() { echo "[warn] $*"; }
fail() { echo "[fail] $*" >&2; exit 1; }
fatal() { echo "[fatal] $*" >&2; exit 2; }

# Confirm-or-abort prompt with non-TTY guard: when stdin is not a TTY
# (e.g. piped or backgrounded) and --force was not passed, fail fast
# with a clear hint instead of hanging on `read`. Returns 0 to proceed,
# returns 1 (caller should propagate) when the user answers no, and
# `exit 2` (fatal) when stdin is non-interactive and --force is absent.
_confirm_or_abort() {
  local message="$1"
  local force="${2:-}"
  if [[ "$force" == "--force" ]]; then
    return 0
  fi
  if [[ ! -t 0 ]]; then
    fatal "$message -- stdin is not a TTY; pass --force to skip prompt or run interactively"
  fi
  warn "$message"
  local ans
  read -r -p "confirm? [y/N] " ans
  case "$ans" in
    y|Y|yes|YES)  return 0 ;;
    *)            echo "[abort]"; return 1 ;;
  esac
}

usage() {
  cat <<EOF
Usage: bash $(basename "${BASH_SOURCE[0]}") <subcommand> [--force]

Subcommands (Runs on = which machine to execute from in three-machine deploy;
                       single-machine mode -- A=B same host -- run all on one host):

  setup        Runs on: A      Verify A-machine env (GPU / torch / triton / GPFS / pip).
  verify-c     Runs on: B      Probe C-machine SSH reachability + CLI version.
  resolve      Runs on: A      Produce resolver classification log (uses GPU).
                               Pass --force to overwrite an existing log without prompt.
  smoke        Runs on: B      Phase 1 smoke (orchestrator on B, bench on A via daemon, LLM on C).
  full         Runs on: B      Phase all (acceptance gate Phase 1+2+3+4, ~2-4 hr).
  status       Runs on: either Show config + state + report + logs.
  clean        Runs on: either Remove state / queue / results / .phase4-worktree.
                               Pass --force to skip confirmation prompt.
  help         Runs on: either Show this message.

Config sources (priority: env > conf file > defaults):
  CONFIG_FILE: ${CONFIG_FILE:-<none>}  [${CONFIG_SOURCE}]
  EXAMPLE:     ${EXAMPLE_CONF_PATH}

Conf file resolution (high to low):
  1. \$TAB_CONFIG env var (explicit override)
  2. <repo-root>/tab_quickstart.conf  (recommended; gitignored)
  3. ~/.tab-quickstart.conf            (legacy fallback)

Required variables (set in conf file or env):
  TAB_GPFS_ROOT    Working / shared directory for state, queue, results, logs.
  TAB_C_HOST       SSH target for the LLM CLI host (user@host or ~/.ssh/config alias).
  TAB_C_CLI_CMD    CLI command template with literal '{prompt}', e.g.
                     'my-cli --print {prompt}'  (no surrounding quotes around {prompt}).

Optional variables:
  TAB_BENCH_REPO              default: \$(git rev-parse --show-toplevel)
  TAB_RESOLVE_LOG             default: \$TAB_GPFS_ROOT/pa_resolve.log
  TAB_RESOLVE_CHECKPOINT_DIR  default: \$TAB_GPFS_ROOT/resolve_ckpt
                              Per-task verdict JSON for crash-resume; auto-passed
                              to resolver as --checkpoint-dir + --resume.
                              Cleared by 'resolve --force' and by 'clean'.
  TAB_RESOLVE_WORKERS         default: 1
                              Parallel resolver threads (1 = sequential).
                              4-8 is a sweet spot on a 32+ GB machine.
  TAB_STATE_PATH              default: \$TAB_GPFS_ROOT/state/triage_state.json
  TAB_LLM_TIMEOUT_SECONDS     default: 600
  TAB_SETUP_SKIP_INSTALL      set to '1' to skip pip install in 'setup'
  TAB_VERBOSE                 set to '1' to enable bash 'set -x'
EOF
}

# Subcommand dispatch happens *after* config is loaded; the only case
# we want to handle pre-load is "help", so it works even with no conf.
case "${1:-}" in
  help|-h|--help|"")
    usage
    exit 0
    ;;
esac

# ---------------------------------------------------------------------------
# Load conf file (optional; missing file is fine -- env can supply all vars).
#
# Precedence is env > conf file > defaults. Bash `source` would overwrite
# pre-set env vars, so we snapshot which TAB_* names were *present* in the
# pre-conf environment (using bash's ${VAR+x} parameter-presence operator),
# source the conf, then restore env values on top of any conf-supplied
# values. Presence-based snapshot (not emptiness-based) is required so an
# explicit empty env value -- e.g. `TAB_SETUP_SKIP_INSTALL=''` -- still
# wins over a conf file that sets the same variable to a non-empty value.
# `printf -v` (rather than `eval`) is used for the restore step to avoid
# shell injection from snapshotted values.
# ---------------------------------------------------------------------------
declare -A _ENV_PRESENCE _ENV_VALUE
for _var in TAB_GPFS_ROOT TAB_C_HOST TAB_C_CLI_CMD TAB_BENCH_REPO \
            TAB_RESOLVE_LOG TAB_RESOLVE_CHECKPOINT_DIR TAB_RESOLVE_WORKERS \
            TAB_STATE_PATH TAB_LLM_TIMEOUT_SECONDS \
            TAB_SETUP_SKIP_INSTALL TAB_VERBOSE; do
  if [[ -n "${!_var+x}" ]]; then
    _ENV_PRESENCE[$_var]=1
    _ENV_VALUE[$_var]="${!_var}"
  fi
done
unset _var

if [[ -n "$CONFIG_FILE" && -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi

# Restore env-supplied values on top of conf (presence-based, includes "").
for _var in "${!_ENV_PRESENCE[@]}"; do
  printf -v "$_var" '%s' "${_ENV_VALUE[$_var]}"
done
unset _var

# ---------------------------------------------------------------------------
# Required variables (fail-loud with conf-file hint).
# ---------------------------------------------------------------------------
_require() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    local hint="${CONFIG_FILE:-$REPO_ROOT/tab_quickstart.conf (create from example)}"
    echo "[fatal] $name not set; configure in $hint or export env" >&2
    echo "[fatal] example conf: $EXAMPLE_CONF_PATH" >&2
    exit 2
  fi
}
_require TAB_GPFS_ROOT
_require TAB_C_HOST
_require TAB_C_CLI_CMD

# Validate CLI template contains the literal {prompt} placeholder so we
# fail at config time, not when the orchestrator tries to invoke the CLI.
# Mirrors scripts/automation/llm_client.py:_is_inside_shell_quotes so that
# wrapper-level and orchestrator-level validation reject the exact same
# set of bad templates.
_validate_cli_cmd_template() {
  local tmpl="$1"
  local count
  # `grep -o ... | wc -l` exits 1 under set -e + pipefail when there are
  # zero matches; `|| true` keeps the pipeline alive so we can produce a
  # diagnostic instead of bailing silently with rc=1.
  count=$(grep -o '{prompt}' <<<"$tmpl" | wc -l | tr -d ' ' || true)
  count="${count:-0}"
  if [[ "$count" != "1" ]]; then
    fatal "TAB_C_CLI_CMD must contain literal '{prompt}' placeholder exactly once; got $count occurrence(s) in: $tmpl"
  fi
  # Quote-depth check: track shell quote state up to the placeholder so a
  # template like 'cli "{prompt}"' or "cli '{prompt}'" -- where the bench
  # would double-quote the already-shlex.quote'd prompt -- fails up front.
  local before c in_single=0 in_double=0 i
  before="${tmpl%%\{prompt\}*}"
  for (( i=0; i<${#before}; i++ )); do
    c="${before:i:1}"
    if [[ "$c" == "'" && $in_double -eq 0 ]]; then
      in_single=$((1 - in_single))
    elif [[ "$c" == '"' && $in_single -eq 0 ]]; then
      in_double=$((1 - in_double))
    fi
  done
  if [[ $in_single -ne 0 || $in_double -ne 0 ]]; then
    fatal "TAB_C_CLI_CMD: do NOT wrap {prompt} in shell quotes (even outer ones); bench wraps via shlex automatically. Got: $tmpl"
  fi
}
_validate_cli_cmd_template "$TAB_C_CLI_CMD"

# ---------------------------------------------------------------------------
# Optional variables (defaults).
# ---------------------------------------------------------------------------
TAB_BENCH_REPO="${TAB_BENCH_REPO:-$REPO_ROOT}"
TAB_RESOLVE_LOG="${TAB_RESOLVE_LOG:-$TAB_GPFS_ROOT/pa_resolve.log}"
TAB_RESOLVE_CHECKPOINT_DIR="${TAB_RESOLVE_CHECKPOINT_DIR:-$TAB_GPFS_ROOT/resolve_ckpt}"
TAB_RESOLVE_WORKERS="${TAB_RESOLVE_WORKERS:-1}"
TAB_STATE_PATH="${TAB_STATE_PATH:-$TAB_GPFS_ROOT/state/triage_state.json}"
TAB_LLM_TIMEOUT_SECONDS="${TAB_LLM_TIMEOUT_SECONDS:-600}"
TAB_SETUP_SKIP_INSTALL="${TAB_SETUP_SKIP_INSTALL:-}"
TAB_VERBOSE="${TAB_VERBOSE:-}"

if [[ -n "$TAB_VERBOSE" ]]; then
  set -x
fi

# ---------------------------------------------------------------------------
# Subcommand: setup
# ---------------------------------------------------------------------------
cmd_setup() {
  step "running setup_machine_a_gpu.sh (repo: $TAB_BENCH_REPO)"
  local setup_args=(--gpfs-root "$TAB_GPFS_ROOT")
  if [[ -n "$TAB_SETUP_SKIP_INSTALL" ]]; then
    setup_args+=(--skip-install)
  fi
  bash "$RUNNERS_DIR/setup_machine_a_gpu.sh" "${setup_args[@]}"
}

# ---------------------------------------------------------------------------
# Subcommand: verify-c
# ---------------------------------------------------------------------------
cmd_verify_c() {
  step "ssh probe: 'echo OK' on $TAB_C_HOST"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$TAB_C_HOST" 'echo OK' 2>/dev/null \
        | grep -q '^OK$'; then
    fail "ssh probe failed: $TAB_C_HOST (BatchMode key auth required)"
  fi
  ok "ssh reachable"

  # Extract CLI name from the template (first whitespace-separated token,
  # ignoring leading env var assignments like `env LANG=C my-cli ...`).
  local cli_name=""
  local tok
  for tok in $TAB_C_CLI_CMD; do
    case "$tok" in
      *=*) continue ;;
      env) continue ;;
      *)   cli_name="$tok"; break ;;
    esac
  done
  if [[ -z "$cli_name" ]]; then
    fail "could not derive CLI binary name from TAB_C_CLI_CMD: $TAB_C_CLI_CMD"
  fi

  step "C-machine CLI version: $cli_name --version on $TAB_C_HOST"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$TAB_C_HOST" \
          "command -v $cli_name >/dev/null && $cli_name --version" 2>&1; then
    fail "$cli_name --version on $TAB_C_HOST"
  fi
  ok "CLI reachable on $TAB_C_HOST"
}

# ---------------------------------------------------------------------------
# Subcommand: resolve
# ---------------------------------------------------------------------------
cmd_resolve() {
  local force_flag="${1:-}"
  if [[ -s "$TAB_RESOLVE_LOG" && "$force_flag" != "--force" ]]; then
    local lines
    lines="$(wc -l < "$TAB_RESOLVE_LOG")"
    _confirm_or_abort \
      "$TAB_RESOLVE_LOG already exists ($lines lines); overwrite?" \
      "$force_flag" \
      || return 1
  fi
  if [[ "$force_flag" == "--force" ]]; then
    # --force = full re-decide from scratch: clear the per-task ckpt
    # dir so --resume below does not silently skip tasks the user
    # specifically wanted re-run. The default (no --force) preserves
    # the ckpt so an interrupted sweep picks up where it stopped.
    rm -rf "$TAB_RESOLVE_CHECKPOINT_DIR"
  fi
  mkdir -p "$(dirname "$TAB_RESOLVE_LOG")"
  mkdir -p "$TAB_RESOLVE_CHECKPOINT_DIR"
  step "running resolver -> $TAB_RESOLVE_LOG (workers=$TAB_RESOLVE_WORKERS, ckpt=$TAB_RESOLVE_CHECKPOINT_DIR)"
  # --refresh-cache: also write the structured per-task verdict cache
  # to results/resolve_all.json (provenance + reason + reject_notes +
  # input_fp + upstream_fp per task) so fix_orchestrator and any
  # debugging have real evidence, not just the bare status header.
  # --checkpoint-dir + --resume: persist one JSON per task as it
  # finishes -- an OOM/SIGKILL/network glitch mid-sweep no longer
  # discards every completed verdict; the next invocation resumes
  # exactly where it stopped. --workers parallelises across N threads
  # (peak memory scales ~linearly with N because each in-flight task
  # holds at least one vllm-importing oracle subprocess).
  (cd "$TAB_BENCH_REPO" && python3 scripts/resolve_upstream_entrypoints.py \
        --refresh-cache \
        --checkpoint-dir "$TAB_RESOLVE_CHECKPOINT_DIR" \
        --resume \
        --workers "$TAB_RESOLVE_WORKERS" \
        2>&1) \
    | tee "$TAB_RESOLVE_LOG"
  ok "resolve log produced"
  # --refresh-cache also writes the structured per-task verdict cache
  # to $TAB_BENCH_REPO/resolve_all.json. Mirror it to $TAB_GPFS_ROOT so
  # downstream subcommands (smoke / full / verify-c) can load either
  # the plain-text log or the structured JSON via the shared
  # automation.lib.resolve_loader -- the loader auto-detects format.
  local cache_src="$TAB_BENCH_REPO/resolve_all.json"
  local cache_dst="$TAB_GPFS_ROOT/resolve_all.json"
  if [[ -s "$cache_src" ]]; then
    cp -f "$cache_src" "$cache_dst"
    ok "structured cache mirrored to $cache_dst"
  fi
  step "verdict distribution:"
  # Prefer the structured cache when available (resolve_all.json) so the
  # distribution numbers come from the canonical JSON, not a grep over a
  # noisy plain-text tail.
  local dist_src="$TAB_RESOLVE_LOG"
  if [[ -s "$cache_dst" ]]; then
    dist_src="$cache_dst"
  fi
  python3 - "$dist_src" <<'PY' || true
import collections, pathlib, sys
sys.path.insert(0, "scripts")
from automation.lib.resolve_loader import load_resolve_artifact
p = pathlib.Path(sys.argv[1])
try:
    records = load_resolve_artifact(p)
except Exception as exc:
    print(f"  (could not parse {p}: {exc})")
    raise SystemExit(0)
c = collections.Counter(r["verdict_class"] for r in records)
for cls in sorted(c):
    print(f"  {cls:32s} {c[cls]}")
print(f"  total                            {len(records)}")
PY
}

# ---------------------------------------------------------------------------
# Internal: invoke run_acceptance_gate.sh for a given phase.
# ---------------------------------------------------------------------------
_run_phase() {
  local phase="$1"
  step "running acceptance gate (phase=$phase) via $TAB_C_HOST"
  bash "$RUNNERS_DIR/run_acceptance_gate.sh" \
    --gpfs-root "$TAB_GPFS_ROOT" \
    --llm-transport ssh-cli \
    --llm-ssh-host "$TAB_C_HOST" \
    --llm-cli-cmd "$TAB_C_CLI_CMD" \
    --llm-timeout-seconds "$TAB_LLM_TIMEOUT_SECONDS" \
    --resolve-log "$TAB_RESOLVE_LOG" \
    --state-path "$TAB_STATE_PATH" \
    --phase "$phase"
}

# ---------------------------------------------------------------------------
# Subcommand: smoke (Phase 1).
# ---------------------------------------------------------------------------
cmd_smoke() {
  if [[ ! -s "$TAB_RESOLVE_LOG" ]]; then
    fail "$TAB_RESOLVE_LOG missing or empty; run 'resolve' first"
  fi
  _run_phase 1
}

# ---------------------------------------------------------------------------
# Subcommand: full (Phase all).
# ---------------------------------------------------------------------------
cmd_full() {
  if [[ ! -s "$TAB_RESOLVE_LOG" ]]; then
    fail "$TAB_RESOLVE_LOG missing or empty; run 'resolve' first"
  fi
  _run_phase all
}

# ---------------------------------------------------------------------------
# Subcommand: status -- read-only summary, never invokes SSH or runners.
# ---------------------------------------------------------------------------
cmd_status() {
  echo "[runs from: $(hostname)]"
  echo "== config =="
  if [[ -n "$CONFIG_FILE" && -f "$CONFIG_FILE" ]]; then
    echo "  CONFIG_FILE:       $CONFIG_FILE (loaded) [${CONFIG_SOURCE}]"
  elif [[ -n "$CONFIG_FILE" ]]; then
    echo "  CONFIG_FILE:       $CONFIG_FILE (missing) [${CONFIG_SOURCE}]"
  else
    echo "  CONFIG_FILE:       <none> [${CONFIG_SOURCE}]"
  fi
  echo "  TAB_GPFS_ROOT:     $TAB_GPFS_ROOT"
  echo "  TAB_C_HOST:        $TAB_C_HOST"
  echo "  TAB_C_CLI_CMD:     $TAB_C_CLI_CMD"
  if [[ -s "$TAB_RESOLVE_LOG" ]]; then
    local rlines
    rlines="$(wc -l < "$TAB_RESOLVE_LOG")"
    echo "  TAB_RESOLVE_LOG:   $TAB_RESOLVE_LOG ($rlines lines)"
  else
    echo "  TAB_RESOLVE_LOG:   $TAB_RESOLVE_LOG (missing/empty)"
  fi
  if [[ -f "$TAB_STATE_PATH" ]]; then
    echo "  TAB_STATE_PATH:    $TAB_STATE_PATH (exists)"
  else
    echo "  TAB_STATE_PATH:    $TAB_STATE_PATH (missing)"
  fi
  echo
  echo "== state =="
  if [[ -f "$TAB_STATE_PATH" ]]; then
    python3 - "$TAB_STATE_PATH" <<'PY' || echo "  warning: state file unreadable"
import json, pathlib, sys
from collections import Counter
p = pathlib.Path(sys.argv[1])
try:
    data = json.loads(p.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"  warning: could not parse state file: {exc}")
    raise SystemExit(0)
if "schema_version" not in data:
    print("  warning: state file missing schema_version")
tasks = data.get("tasks", {}) or {}
counter = Counter((t or {}).get("state", "?") for t in tasks.values())
print(f"  total tasks: {sum(counter.values())}")
for k, v in sorted(counter.items()):
    print(f"    {k}: {v}")
PY
  else
    echo "  (no state file yet -- run 'smoke' or 'full' first)"
  fi
  echo
  echo "== reports =="
  local release_dir="$TAB_BENCH_REPO/docs/release"
  if [[ -d "$release_dir" ]]; then
    local latest_report=""
    # shellcheck disable=SC2012
    latest_report="$(ls -t "$release_dir"/internal-acceptance-report-*.md 2>/dev/null | head -1 || true)"
    if [[ -n "$latest_report" ]]; then
      echo "  latest acceptance report: $latest_report"
      echo "  (last 10 lines:)"
      tail -10 "$latest_report" | sed 's/^/    /'
    else
      echo "  (no acceptance report yet)"
    fi
  else
    echo "  (no docs/release/ directory at $release_dir)"
  fi
  echo
  echo "== logs =="
  if [[ -d "$TAB_GPFS_ROOT/logs" ]]; then
    # shellcheck disable=SC2012
    ls -lt "$TAB_GPFS_ROOT/logs/" 2>/dev/null | head -5 || echo "  (no logs)"
  else
    echo "  (no logs directory at $TAB_GPFS_ROOT/logs)"
  fi
}

# ---------------------------------------------------------------------------
# Subcommand: clean.
# ---------------------------------------------------------------------------
cmd_clean() {
  local force_flag="${1:-}"
  echo "will delete:"
  echo "  $TAB_GPFS_ROOT/state"
  echo "  $TAB_GPFS_ROOT/queue"
  echo "  $TAB_GPFS_ROOT/results"
  echo "  $TAB_GPFS_ROOT/.phase4-worktree"
  echo "  $TAB_RESOLVE_CHECKPOINT_DIR"
  _confirm_or_abort \
    "will delete state/queue/results/.phase4-worktree/resolve_ckpt under $TAB_GPFS_ROOT" \
    "$force_flag" \
    || return 1
  rm -rf \
    "$TAB_GPFS_ROOT/state" \
    "$TAB_GPFS_ROOT/queue" \
    "$TAB_GPFS_ROOT/results" \
    "$TAB_GPFS_ROOT/.phase4-worktree" \
    "$TAB_RESOLVE_CHECKPOINT_DIR"
  ok "cleaned (logs/ + pa_resolve.log preserved)"
}

# ---------------------------------------------------------------------------
# Dispatch.
# ---------------------------------------------------------------------------
cmd="${1:-help}"
shift || true
case "$cmd" in
  setup)     cmd_setup "$@" ;;
  verify-c)  cmd_verify_c "$@" ;;
  resolve)   cmd_resolve "$@" ;;
  smoke)     cmd_smoke "$@" ;;
  full)      cmd_full "$@" ;;
  status)    cmd_status "$@" ;;
  clean)     cmd_clean "$@" ;;
  help|-h|--help)  usage ;;
  *)         fatal "unknown subcommand: $cmd (try 'help')" ;;
esac
