# Phase 20 — Cross-Application IDOR/BOLA Generalization Audit

## Scope and safety

This phase evaluates the existing `IDORBOLAEngine` against synthetic, deterministic, in-memory applications only. It does not add a socket client, live target connector, browser automation, arbitrary code execution, or external bug-bounty testing.

The implementation is in `src/private_agent/bugbounty/cross_application.py` and the regression/evaluation tests are in `tests/test_phase20_cross_application.py`.

## Phase 19 review

Phase 19 used 18 cases with a single `IndependentAuthorizationFixture` model. The cases varied policies, endpoint shapes, identifiers, and response outcomes, but they were not independent application implementations. Its oracle was separate from the engine and the engine did not read the expected labels, but the benchmark still shared one fixture contract and one authorization implementation family.

Phase 20 therefore treats Phase 19 as within-model validation rather than cross-application generalization.

## Independent applications

The new corpus has five application implementations:

| Application | Authorization architecture | Split | Cases |
|---|---|---|---:|
| `app-ledger` | Resource-layer ownership with 404 denial | train | 2 |
| `app-console` | Operation-layer role authorization with 403/401 responses | validation | 2 |
| `app-vault` | Tenant-layer membership and cross-tenant isolation | held-out | 2 |
| `app-projects` | Nested-resource membership authorization | held-out | 2 |
| `app-reports` | Unavailable/ambiguous service response | validation | 1 |

The applications use independently implemented data models and request logic. They share only the narrow observable adapter contract required by the existing engine: identities, resources, endpoints, URL construction, and a local `request()` method.

The evaluation oracle is stored in `CrossApplicationOracle` and is not attached to application objects or passed to `IDORBOLAEngine`. The engine receives only an application and a scope. All case identifiers, expected outcomes, and splits are harness metadata.

The split is by application: every case belonging to one application remains in the same split. Held-out applications were not used to tune the engine.

## Results

`evaluate_cross_application_corpus()` produced this reproducible result:

| Split | Total | TP | FP | TN | FN | Inconclusive | Errors | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 2 | 1 | 0 | 1 | 0 | 0 | 0 | 1.0 | 1.0 | 1.0 |
| Validation | 3 | 1 | 0 | 1 | 0 | 1 | 0 | 1.0 | 1.0 | 1.0 |
| Held-out | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 0.5 | 1.0 | 0.6667 |
| Overall | 9 | 4 | 2 | 2 | 0 | 1 | 0 | 0.6667 | 1.0 | 0.8 |

Inconclusive cases and errors are excluded from precision, recall, and F1 and are never counted as true negatives. `external_targets_contacted` is `0`.

## Per-application results

- `app-ledger`: TP=1, TN=1, FP=0, FN=0. The resource-owner model was handled correctly.
- `app-console`: TP=1, TN=1, FP=0, FN=0. The role-operation model was handled correctly in this small sample.
- `app-vault`: TP=1, FP=1. The safe tenant policy allows a different same-tenant member to read the document; the engine assumes the resource owner is the only valid baseline authority and reports a false positive.
- `app-projects`: TP=1, FP=1. The safe nested membership policy allows a project member who is not the resource owner; the engine's owner/parent heuristic reports a false positive.
- `app-reports`: one `INCONCLUSIVE` result because the service never produced a usable authorization decision.

Every confirmed finding had observed authorization-comparison evidence with a requesting identity, resource binding, expected/observed status and body structures, and protected content. Status differences alone did not create confirmed findings.

## Findings and limitations

The strongest confirmed limitation is semantic: `IDORBOLAEngine` currently models “unauthorized” primarily as “not the resource owner” or “not the parent owner.” That is too narrow for legitimate tenant membership and nested membership policies. The two held-out false positives are therefore evidence of a real generalization gap, not benchmark noise.

Other limitations remain:

- The generic adapter does not model arbitrary policy engines, sessions, capabilities, or operation-specific authorization graphs.
- The sample is small and synthetic; it does not estimate production accuracy.
- The applications are deterministic in-memory models and do not test framework-specific routing, serialization, caching, or identity propagation.
- No external network target was contacted.
- No zero-day or production-grade discovery claim is made.

## Verdict

**PARTIAL GENERALIZATION**

The evidence supports that the current engine can detect controlled owner, role, tenant-bypass, and nested-resource violations across several local application implementations, including held-out applications. It also demonstrates a concrete false-positive risk for valid tenant/member access policies. Generalization beyond this bounded set is not established, and the engine is not ready for external-target testing based on this phase alone.

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/test_phase20_cross_application.py

PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import json
from private_agent.bugbounty.cross_application import evaluate_cross_application_corpus
print(json.dumps(evaluate_cross_application_corpus(), indent=2, sort_keys=True))
PY
```
