# Phase 21 — Authorization-Aware IDOR/BOLA Engine Redesign

## Scope

This phase is local-only. It changes the decision boundary of the deterministic IDOR/BOLA engine; it does not add a live HTTP scanner, external connector, browser automation, target code execution, or production-accuracy claim.

## Root cause of the Phase 20 false positives

The previous `IDORBOLAEngine` selected `resource.owner_id` or the parent owner as the baseline and treated every other non-admin identity as an unauthorized comparison identity. If that identity received protected content, the engine could report IDOR without knowing the application's actual authorization model.

That assumption was wrong for:

1. `app-vault`: a different member in the same tenant is legitimately authorized even though the resource owner is another member.
2. `app-projects`: a project member is legitimately authorized to read a nested task even though the member is not the task owner or project owner.

The old behavior was measured before this redesign as two held-out false positives, with held-out precision `0.5` and F1 `0.6667`.

## Design

`AuthorizationObservation` and `PolicyDecision` were added as a narrow local adapter contract:

- `ALLOW`: the explicitly authorized local policy adapter says the identity may perform this operation.
- `DENY`: the adapter says the identity should be denied.
- `UNKNOWN`: the policy semantics are unavailable or malformed.

The engine now:

1. Searches for an authorized baseline identity using the observable policy adapter instead of assuming direct ownership is sufficient.
2. Confirms an IDOR/BOLA finding only when the policy observation is `DENY` and protected content is actually returned.
3. Treats an explicit `ALLOW` for an authorized non-owner as `SAFE`, even when protected content is returned.
4. Produces a `SUSPECTED`/`HYPOTHESIS` result when protected content is observed but policy semantics are unknown.
5. Keeps `INCONCLUSIVE` for unavailable or insufficient observations and `ERROR` for invalid observation failures.
6. Never reads benchmark oracle fields or expected outcomes.

The local adapters expose policy behavior as an explicitly authorized test seam, not as benchmark labels. Vulnerable application implementations deliberately return data contrary to the adapter's policy rule; this creates a controlled differential test without passing expected labels into the engine.

Malformed or missing adapters fail closed to `UNKNOWN`; they cannot promote an observation to `CONFIRMED`.

## Supported local semantics

Covered by the new tests and corpus:

- Direct resource ownership.
- Same-tenant membership.
- Project/group membership for nested resources.
- Operation-level role authorization.
- Read-versus-write operation differences in the local policy adapter.
- Missing and expired credentials.
- Deny-by-default behavior.
- Unavailable/ambiguous observations.
- Protected content with misleading status differences.

The engine still does not model arbitrary production policy languages, sessions, capability graphs, delegated sharing, cache interactions, or framework-specific authorization middleware without a trusted explicit adapter.

## Differential cases

The Phase 20 independent applications remain the held-out evaluation corpus. The two original false positives are now explicit regressions:

- `V-01`: valid same-tenant member access is `SAFE`.
- `P-01`: valid project-member nested access is `SAFE`.

Genuine unauthorized implementations remain positive:

- `L-02`: resource-owner bypass.
- `C-02`: operation-role bypass.
- `V-02`: cross-tenant access.
- `P-02`: non-member nested access.

A policy adapter removed from the vulnerable ledger application produces `SUSPECTED`, not `VULNERABLE`, and no `REPORT_READY` finding.

## Re-evaluation

The current Phase 21 run reports:

| Split | Total | TP | FP | TN | FN | SUSPECTED | INCONCLUSIVE | ERROR | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 2 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | 1.0 |
| Validation | 3 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1.0 | 1.0 | 1.0 |
| Held-out | 4 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | 1.0 |
| Overall | 9 | 4 | 0 | 4 | 0 | 0 | 1 | 0 | 1.0 | 1.0 | 1.0 |

`INCONCLUSIVE`, `SUSPECTED`, and `ERROR` are not counted as true negatives and are excluded from precision/recall/F1 denominators as appropriate.

The original Phase 20 held-out baseline remains preserved in the evaluator output and documentation:

- TP=2, FP=2, TN=0, FN=0.
- Precision=0.5, Recall=1.0, F1=0.6667.
- The two false positives were `app-vault` same-tenant access and `app-projects` valid membership access.

## Security regression status

The change does not modify approval integrity, policy authority, knowledge provenance, scope enforcement, task authentication, `EXECUTION_UNKNOWN`, secret redaction, quarantine, or host-execution behavior. The full backend suite is rerun after the engine change.

## Limitations and verdict

The result supports a stronger claim than Phase 20: the engine no longer confirms the two observed false positives when an explicit policy adapter establishes valid tenant or project membership, while the controlled unauthorized cases remain detectable.

It does **not** prove production-grade accuracy, arbitrary-policy generalization, zero-day discovery, or readiness for external targets. External live scanning remains disabled and out of scope.
