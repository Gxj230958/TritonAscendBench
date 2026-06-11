# Submission Throttling + Sybil Detection

> 中文版：[../zh/reference/submission_throttling.md](../zh/reference/submission_throttling.md)

## Rate Limits (Per Task Per Org Identity)

| Mode | Limit |
|------|-------|
| `self-reported` (v0.1) | **3 submissions per week per (org_identity, task_id)** |
| `verified` (v1.0) | **2 submissions per week per (org_identity, task_id)** (verified runs are expensive) |
| Initial onboarding (first org submission) | 1 free submission, then standard limit |

## Org Identity Derivation

Each submission's `org_identity` = hash of:
- GPG signing key fingerprint (preferred — strongest) OR
- Email domain (fallback — e.g. `@example-university.edu`) OR
- IP block (last fallback — `/24` for IPv4, `/64` for IPv6)

## Sybil Detection

Multiple "different" org_identities exhibiting coordinated patterns:

| Signal | Threshold | Action |
|--------|----------|--------|
| Same Docker digest across N+ "different" submitters | N ≥ 3 | Flag for human audit |
| Same source-code hash across "different" submitters | N ≥ 2 | Flag |
| Submission timing burst (< 60s gap from different IPs) | 3+ | Throttle + log |
| Same `runtime_env/Dockerfile` SHA across orgs | N ≥ 5 | Audit |
| Suspicious progressive iteration on hidden shapes after rejection | Heuristic | Block + manual review |

Sybil flag → submission stays on `self-reported` board only, not verified ranking. Org subject to slower throttling.

## Probe-Through-Failure Defense

If submitter exhausts week budget but resubmissions cluster around hidden shape boundaries (statistical signature of probing), block submitter for 30 days.

Hidden adversarial set rotation per epoch (~6 months) limits cumulative information leakage.

## Hidden Result Log Sanitization

When publishing verified result JSON:
- Public log: `{task_id, aggregate_pass_rate, public_shape_timings, container_digest}`
- Withheld: `hidden_shape_dimensions`, `hidden_shape_failure_inputs`, `expected_vs_actual_tensor_data`

Hidden failure reasons are aggregated:
```
"L3_04 hidden cases: 5 passed, 2 numerical_diff_exceeded, 1 nan_inf_inconsistency"
```

NOT:
```
"L3_04 hidden case [seq=12345, hidden=4096]: out_c[0,5] = 1.234 vs golden = 1.245"  # never published
```

## Quota Reset

- Weekly window: Monday 00:00 UTC (sliding 7-day window)
- Manual reset: org-admin via ops API (audited)

## Storage

a submitter-quota database (SQLite or Postgres) managed by the verified-runner leaderboard / broker infrastructure — future work, see `docs/DESIGN.md §6`.
