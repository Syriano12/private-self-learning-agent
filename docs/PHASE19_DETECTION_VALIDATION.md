# Phase 19 — Bug Bounty Detection Validation

## Scope

Phase 19 is a bounded local research harness. It does not add a live HTTP client, browser automation, exploit delivery, target code execution, or external-target capability.

The new validator is implemented in `src/private_agent/bugbounty/validation.py` and is exported from `private_agent.bugbounty`.

## Independent corpus

`build_independent_corpus()` creates 18 deterministic local samples:

- 8 train cases.
- 4 validation cases.
- 6 held-out cases.

The corpus covers horizontal IDOR, vertical authorization, nested-resource authorization, cross-tenant ownership, changed resource identifiers, changed paths, same-status protected content, same-status public summaries, status-only negative controls, object-level authorization, role authorization, missing authentication, expired credentials, insufficient evidence, and an observation error.

Each sample has a separate `CorpusOracle` containing the expected outcome. The engine receives only the sample's `fixture` and `scope`; it does not receive the oracle, case ID, expected outcome, split, or vulnerability class. The fixtures expose only ordinary application behavior: identities, resources, endpoints, and responses. Ground truth is not stored on fixture objects.

The current engine no longer reads `fixture.variant`. Evidence provenance uses the neutral value `fixture_source=deterministic_local_boundary` instead of a label-bearing fixture variant.

## Outcome and metric definitions

Case-level predictions are:

- `VULNERABLE`: at least one confirmed finding.
- `SAFE`: no finding and no inconclusive/error result.
- `INCONCLUSIVE`: the fixture produced an unavailable/ambiguous result.
- `ERROR`: the observation boundary failed before a defensible result.

Only determinate `VULNERABLE` and `SAFE` cases participate in TP, FP, TN, FN, precision, recall, and F1. Inconclusive and error cases are reported separately and are never counted as true negatives.

The held-out split is fixed before execution and is not used to tune the current engine. The split is a validation protocol, not proof of production generalization.

## Current measured result

The reproducible local result from `evaluate_independent_corpus()` is:

| Split | Total | TP | FP | TN | FN | Inconclusive | Errors | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 8 | 4 | 0 | 4 | 0 | 0 | 0 | 1.0 | 1.0 | 1.0 |
| Validation | 4 | 1 | 0 | 2 | 0 | 1 | 0 | 1.0 | 1.0 | 1.0 |
| Held-out | 6 | 2 | 0 | 3 | 0 | 0 | 1 | 1.0 | 1.0 | 1.0 |
| Overall | 18 | 7 | 0 | 9 | 0 | 1 | 1 | 1.0 | 1.0 | 1.0 |

`external_targets_contacted` is `0`.

The result is intentionally not described as production accuracy. The fixtures are synthetic, deterministic, small, and designed around a narrow object-level authorization model. The current held-out result demonstrates reproducibility across the supplied local variations, not real-world recall or zero-day discovery.

## Evidence integrity

A confirmed IDOR/BOLA finding must include:

- A controlled authorization comparison.
- Owner identity and test identity.
- Resource and endpoint binding.
- A request URL containing the bound resource identifier.
- Expected and observed status/body structures.
- Reproduction steps.
- Protected content observed by a different controlled identity.

`validate_confirmed_evidence()` rejects missing identity/resource bindings, same-identity comparisons, missing comparison fields, denied observed responses, missing evidence, and non-confirmed finding states. Status codes alone are not sufficient.

Existing evidence redaction remains active. Public reports and public run serialization expose an opaque authorization-reference digest rather than the raw reference.

## Validation command

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import json
from private_agent.bugbounty import evaluate_independent_corpus
print(json.dumps(evaluate_independent_corpus(), indent=2, sort_keys=True))
PY
```

The Phase 19 tests are in `tests/test_phase19_detection_validation.py`. They verify oracle isolation, reproducibility, split reporting, inconclusive/error treatment, evidence quality, and robustness to case-order, resource-ID, path, and response-format changes.
