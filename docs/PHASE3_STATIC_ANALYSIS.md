# Phase 3 — Static Call Graph, Data Flow, and AI Contract

This phase adds bounded Python-only static analysis to the existing repository inspection tools. It builds on the immutable `RepositorySnapshot`, `CodeEvidenceRef`, and Python AST records from Phase 1/2. It does not execute repository code and does not contact an AI provider.

## Repository inspection output

`RepositoryInspector.inspect("inspect_repository_structure", ...)` now adds:

- `call_graph`: Python symbol/call-site nodes and candidate edges. `OBSERVED_SYNTAX` means the syntax is present; `POTENTIAL_UNIQUE`, `POTENTIAL_AMBIGUOUS`, `DECLARED_IMPORT_NOT_IN_ANALYZED_TREE`, and `UNRESOLVED_*` describe bounded static resolution only. A missing import is only a potential external-or-unanalysed target, never a proven external dependency; `runtime_target_confirmed` is always false.
- `data_flow`: bounded may-flow nodes/edges for assignments, call arguments/parameters, and return/call-result relationships. Source, sanitizer, and sink labels are lexical candidates only; `runtime_value_verified` and `vulnerability_confirmed` are always false. Unresolved mappings and path-insensitive limits are reported explicitly.
- `analysis_evidence_registry`: schema-versioned compact registry. Phase 2 AST records already carry their own `CodeEvidenceRef`; this section contains only additional Phase 3 references and identifies the AST base section. `expand_analysis_evidence_registry` validates and reconstructs the full catalog for internal use.
- `ai_analysis_contract`: a digest-bound, provider-neutral descriptor over the repository summary, graph, flows, and evidence catalog. It contains no source snippets and grants neither execution nor finding-validation authority.

Graph/data-flow records reference evidence by stable `evidence_ids`; the corresponding references bind file path, SHA-256, line/column range, snapshot identity, parser/rule versions, truncation, and uncertainty. The repository inspector revalidates the snapshot before returning structure results. `BugInvestigator` merges its operation snapshots, rejects source drift, rebinds references to the final snapshot, and regenerates the descriptor.

## AI request and response

Use `private_agent.tools.python_ai_contract.materialize_ai_analysis_request(bundle, snapshot)` to validate the descriptor and produce the bounded provider-neutral request. It contains summary counts, graph/data-flow records, and a compact evidence catalog, but no raw source, excerpts, excluded paths, or provider credentials. `request_id` is a content digest of that exact materialized body; bundled requests also retain the descriptor digest and descriptor request ID separately. The local validator enforces request authority fields, evidence membership for every graph/flow record, response byte/node/depth limits, and provider-response schema. This function is local and makes no network call.

A future adapter may submit that request to a provider only under a separate approved integration. Model output must be passed to `private_agent.tools.python_analysis_bundle.validate_ai_hypothesis_response(response, request, bundle, snapshot, inspector)`. The validator rejects request-content changes, compact citation metadata that differs from the canonical EvidenceRefs, unknown graph/flow IDs, stale snapshots, schema extras, and execution/approval/permission fields. Accepted hypotheses are always marked `UNVERIFIED`, `ADVISORY_ONLY`, and `finding_promotion_allowed: false`; citations are not semantic proof.

## Explicit limitations

- Python syntax only; this is not whole-program or runtime dispatch resolution. Reflection, monkey-patching, dynamic imports, plugin registries, and runtime-generated calls remain unresolved.
- Data flow is a lexical/AST may-flow approximation. It is not path-sensitive and does not model all aliases, containers, exceptions, concurrency, descriptors, framework semantics, or dynamic dispatch.
- Source/sanitizer/sink names are heuristic candidates, not security classifications. No edge, path, or AI hypothesis confirms a vulnerability.
- Existing `SecurityController`, ToolContract, owner approval, scope, and Evidence Gate remain the authority boundaries. Phase 3 adds no permission, execution, or finding state transition.
- Analysis remains bounded by repository file/read limits, a 128 KiB default/hard response ceiling, a 32 KiB provider-neutral AI request ceiling, AST/data-flow limits (10,000 AST nodes, depth 128), and bounded graph/data-flow records (96 call-graph nodes, 128 edges, 16 data-flow nodes, 64 edges; evidence catalog 160 references). Excluded, malformed, redacted, stale, oversized, syntactically invalid, or truncated files are never represented as complete analysis.
