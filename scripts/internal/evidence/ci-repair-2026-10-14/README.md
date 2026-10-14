# CPU CI Repair Evidence

The JUnit files retain the diagnosis runs and subsequent focused retries.
`receipt-audit.json` matches the latest outcome for every test in the current
pytest collection: 2,559 passing cases, two existing integration skips, no
missing cases and no unresolved failures. An obsolete renamed test is not part
of the current collection. Earlier failed attempts remain in the raw receipts.

The audit source is preserved as `receipt-audit.py.txt`. It ran from the CI
repair worktree with Python 3.11.14 and the CPU-only test environment. Local
socket integration ran outside the restricted agent sandbox. Rollback tests
used rsync 3.2.7 and libpopt extracted under `/tmp`, without a system install.
The remote CI job separately runs the complete suite on Python 3.12 with
coverage, followed by schema, version matrix and the hard S6 release gate.

These are CPU code-validation receipts. They do not update the archived NPU
campaign outcomes or establish additional verified device coverage.
