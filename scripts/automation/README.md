# scripts/automation/ -- INTERNAL bench-team automation

This directory holds the bench-team internal automation orchestrator
(LLM-driven fix loop, evaluator daemon, queue / lease / lock libraries,
release-boundary helpers).

It is **NOT** part of the bench release artifact:

* `.gitattributes` marks this tree `export-ignore`, so `git archive`
  output (release tarball) does not include it.
* `.dockerignore` excludes it from the Docker build context.
* `pyproject.toml [tool.setuptools.packages.find]` already restricts
  the wheel to `ascendbench*`, so `pip install` artefacts do
  not ship this directory either.
* CI (`.github/workflows/ci.yml`) runs a release-tarball smoke check
  plus a `grep` step ensuring bench core code never `import`s from
  here -- the dependency edge is strictly one-way (`automation` may
  call into bench core via `importlib.util.spec_from_file_location`,
  bench core never references `automation`).

## Layout

| Path                | Purpose                                              |
|---------------------|------------------------------------------------------|
| `__init__.py`       | Package marker with INTERNAL docstring               |
| `lib/`              | Shared libraries (atomic write, queue, lock, ...)    |
| `prompts/`          | LLM prompt templates                                 |
| `spike/`            | One-off M0 spike scripts (GPFS POSIX, LLM A/B)       |

## Conventions

* Every CLI entry-point file (anything intended to be run via
  `python3 scripts/automation/<sub>/<file>.py`) carries a fixed
  `sys.path` bootstrap header that inserts `<repo_root>/scripts` so
  `from automation.lib import ...` resolves regardless of cwd.
* `lib/*.py` and `prompts/*` are import-only modules and do not carry
  the bootstrap header -- their caller bootstraps once.
* Runtime extras live under `[project.optional-dependencies].automation`
  in `pyproject.toml`. Install with `pip install -e ".[automation]"`.
* All code, comments, log strings, and docstrings are English-only.
