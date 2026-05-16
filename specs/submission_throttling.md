# Submission Throttling + Sybil Detection

> v0 — per v4.13 §5b.9 + R11-P1-2 + R18-S5

## Rate Limits (Per Task Per Org Identity)

| Mode | Limit |
|------|-------|
| `self-reported` (v0.1) | **3 submissions per week per (org_identity, task_id)** |
| `verified` (v1.0) | **2 submissions per week per (org_identity, task_id)** (verified runs are expensive) |
| Initial onboarding (first org submission) | 1 free submission, then standard limit |

## Org Identity Derivation

Each submission's `org_identity` = hash of:
- GPG signing key fingerprint (preferred — strongest) OR
- Email domain (fallback — `@huawei.com` / `@bytedance.com` / `@stanford.edu`) OR
- IP block (last fallback — `/24` for IPv4, `/64` for IPv6)

## Sybil Detection (R18-S5 + R11-P1-2)

Multiple "different" org_identities exhibiting coordinated patterns:

| Signal | Threshold | Action |
|--------|----------|--------|
| Same Docker digest across N+ "different" submitters | N ≥ 3 | Flag for human audit |
| Same source-code hash across "different" submitters | N ≥ 2 | Flag |
| Submission timing burst (< 60s gap from different IPs) | 3+ | Throttle + log |
| Same `runtime_env/Dockerfile` SHA across orgs | N ≥ 5 | Audit |
| Suspicious progressive iteration on hidden shapes after rejection | Heuristic | Block + manual review |

Sybil flag → submission stays on `self-reported` board only, not verified ranking. Org subject to slower throttling.

## Probe-Through-Failure Defense (R18-S5)

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

`infra/submitter_quotas.db` (SQLite or Postgres) — schema in `triton_ascend_bench/leaderboard_validator.py`.
