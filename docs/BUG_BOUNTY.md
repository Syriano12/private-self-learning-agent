# Authorized Bug Bounty workflow

The repository includes a bounded, local-first Bug Bounty workflow under `private_agent.bugbounty`. It is deliberately conservative: it parses supplied artifacts, enforces an explicit scope object, and runs active authorization checks against deterministic fixtures. Phase 25 also provides opt-in adapters for installed ProjectDiscovery `httpx` and `nuclei` binaries, but only through the normal SecurityController approval boundary and only for an explicit, unexpired scope. The adapters do not follow out-of-scope redirects, expand scope from discovered hosts, run target code, or grant permissions.

## Local deterministic run

From the repository root:

```bash
python3 -m private_agent.cli bug-bounty \
  --scope examples/bugbounty_local_scope.json \
  --fixture idor_horizontal \
  --report /tmp/bugbounty-report.md \
  --database /tmp/bugbounty.sqlite3
```

The command prints a bounded JSON run result and writes a Markdown report. Replace the fixture with `secure`, `idor_vertical`, `idor_nested`, `false_positive`, or `inconclusive` to exercise the local evaluation corpus. The `secure` fixture must produce no confirmed finding; the IDOR variants produce controlled findings; `false_positive` must not be promoted; and `inconclusive` must remain inconclusive.

Passive-only analysis of local OpenAPI/source artifacts can be run without `active_testing`:

```bash
python3 -m private_agent.cli bug-bounty \
  --scope examples/bugbounty_local_scope.json \
  --source-file ./authorized-local-source.py \
  --openapi ./authorized-openapi.json \
  --no-persist
```

Use `--url` only for passive parsing of supplied strings. The CLI does not fetch them.

## Scope contract

A scope requires a program identifier, authorization reference, timezone-aware validity interval, explicit assets and paths, permitted methods, permitted controlled identities, and permitted controlled resource identifiers. Active fixture requests must provide both an explicitly permitted identity and resource whose identifier is present in the URL path; passive analysis does not need either. Missing, expired, ambiguous, excluded, or conflicting scope decisions fail closed. Path prefixes are segment-bounded, so `/api/v1` cannot authorize `/api/v10`. `POST`, `PUT`, `PATCH`, and `DELETE` are prohibited by the example scope. A redirect, discovered subdomain, third-party URL, or model suggestion never adds an asset.

## Findings and evidence

Findings use the lifecycle `OBSERVED → HYPOTHESIS → VALIDATION_PENDING → CONFIRMED → REPORT_READY → HUMAN_REVIEWED`. Static source patterns remain `HYPOTHESIS` and are not confirmed. A confirmed fixture finding requires a controlled baseline, a controlled unauthorized identity, an explicit scope decision, a semantically meaningful protected-content comparison, bounded request/response evidence, and reproducible steps. Dynamic fields such as timestamps, request IDs, and CSRF values are normalized; credentials are redacted.

The IDOR/BOLA evaluator covers controlled horizontal access, role/vertical access, nested-resource access, secure behavior, harmless dynamic differences, and unavailable/inconclusive responses. It does not infer vulnerability from a status code alone. Passive input is bounded to 100 supplied URLs, 50 source documents, 2 MiB of source text, 500 OpenAPI paths, and 100 retained inventory entries.

## Persistence

When a `Store` is supplied, run metadata and bounded finding JSON are stored in additive `bugbounty_runs` and `bugbounty_findings` tables on the existing SQLite connection. Confirmed findings also create bounded `bugbounty_patterns` records with preconditions, provenance, and non-executable regression templates. Owner feedback can disable a pattern as a false positive or rejected hypothesis. Reopening the database preserves the run and pattern state. This persistence is evidence storage only; it is not a SecurityController policy source, an approval source, or a skill activation mechanism.

## Current capability boundary

Implemented and tested: deterministic scope enforcement, passive inventory from supplied URLs/OpenAPI/source documents, bounded deterministic planning, controlled IDOR/BOLA matrix evaluation, static advisory patterns for command/SQL/SSRF/XSS/path-traversal/deserialization/JWT/open-redirect/CORS sinks, fail-closed ProjectDiscovery `httpx`/`nuclei` adapters with bounded subprocess execution, trusted Nuclei manifest checks, evidence redaction, finding lifecycle, SQLite run persistence, Markdown/JSON reports, and benchmark metrics where ground truth is defined by the fixture corpus. Static matches and scanner matches remain hypotheses and never authorize active testing.

Advisory or not implemented: browser automation, automatic disclosure/submission, active destructive methods, credential testing, broad scanning, hostname DNS-pinned transport, generic open-ended HTTP clients, real zero-day discovery, model-weight training, and a production-grade sandbox runner. Scanner adapters fail closed when the binary, version, required flags, trusted templates, IPv4-literal scope binding, resource limits, or output contract cannot be verified. The committed local benchmark reports true positives, false positives, false negatives, true negatives, precision, and recall against its six-case fixture ground truth only; it does not measure `httpx`/`nuclei` detection quality or precision/recall against arbitrary real targets.
