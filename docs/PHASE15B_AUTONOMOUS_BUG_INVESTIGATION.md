# Phase 15B — Autonomous Bug Investigation

## Scope

Phase 15B adds a read-only `investigate_repository_bug` tool to the existing ToolRegistry and Agent Core. The tool accepts a bug report, a repository identifier, and an optional revision. It reuses the Phase 15A repository access tools for metadata, bounded structure inspection, dependency inspection, literal search, and candidate-file reads.

The investigator never imports, executes, builds, tests, shells out to, or writes to the inspected repository. Repository content is treated as untrusted data and cannot grant capabilities, change policy, or authorize a tool call.

## Report contract

The successful output is a structured diagnostic report containing:

- `bug_report`, canonical `repository`, `revision`, and source type;
- `investigation_steps` showing the bounded operations actually attempted;
- `relevant_files` and `code_paths` with source locations;
- `findings` that distinguish direct observations from static inferences;
- `hypotheses` with supporting evidence, missing evidence, confidence, and a safe next step;
- `evidence` references containing repository, revision, path, line range, SHA-256, and a bounded excerpt;
- `root_cause.status`, which is `ESTABLISHED` only for the implemented deterministic static signature-mismatch finding, otherwise `UNRESOLVED`;
- `missing_evidence`, `recommended_next_verification_steps`, and explicit limitations.

A local fixture has an empty revision because it represents the working tree. For a GitHub source, the investigator first reads metadata and pins later operations to the returned commit before assembling evidence.

## Static analysis boundary

The current static analyzer parses bounded, redacted Python source with `ast.parse` and detects local call/signature keyword mismatches. It also reports lexical matches and declared code paths from the existing Phase 15A analyzers. It does not claim runtime causality from lexical matches, dependency declarations, comments, or repository instructions.

The implementation is intentionally conservative: runtime traces, failing inputs, resolved dependency versions, deployment state, and external-service behavior are reported as missing evidence rather than invented.

## Integration path

The tool is registered by `register_repository_tools()` and is available to the existing planner and executor. When a goal contains an explicit diagnostic marker and exactly one repository reference, the planner deterministically routes it to this tool before consulting an LLM provider; the same registered-tool proposal is used by the no-provider fallback. Architecture-only goals continue to select `inspect_repository_structure`.

The end-to-end path remains the existing authoritative path:

```text
POST /api/v1/tasks
  -> durable QUEUED task
  -> BackgroundWorker claim and lease
  -> Orchestrator planning
  -> SecurityController authorization
  -> GenericExecutor operation boundary
  -> investigate_repository_bug
  -> observation and verification
  -> persisted COMPLETED state
  -> GET /api/v1/tasks/{task_id}
```

Owner `DENY` and `REQUIRE_APPROVAL` rules are evaluated by the existing SecurityController before the investigator calls any repository operation. No alternate execution or approval path was added.

## Validation evidence

The Phase 15B test file contains 18 focused tests covering:

- cited static root-cause detection without executing the fixture;
- repository instructions remaining untrusted data;
- unresolved reports and missing evidence;
- planner selection versus architecture selection;
- owner `DENY` before any repository read;
- `REQUIRE_APPROVAL` and exact approval binding;
- Orchestrator persistence;
- missing repositories and unsupported local revisions;
- input and operation budgets;
- evidence hashes, uniqueness, and revision propagation;
- a real API/storage/worker end-to-end flow from POST through GET;
- search matches that exceed the bounded read set are unverified rather than hashless evidence;
- short unquoted credential literals in YAML/ENV-style content are redacted;
- shadowed or multiply-defined Python symbols remain unresolved rather than being promoted to a root cause;
- explicit bug-report goals are deterministically routed to the investigator even when an LLM provider is configured.

Validation performed for this implementation:

```text
python3 -m pytest -q tests/test_phase15b_bug_investigation.py
18 passed

python3 -m pytest -q tests/test_phase15a_repository.py tests/test_phase15a_github_access.py tests/test_phase15b_bug_investigation.py
62 passed

python3 -m pytest -q
364 passed

python3 -m compileall -q src tests
passed

git diff --check
passed
```

## Remaining limitations

This phase does not execute target-repository tests or runtime reproductions, does not implement automatic patch generation, and does not provide a distributed investigation coordinator. It also does not establish exactly-once behavior for external effects; the investigator itself is read-only, while the existing queue and worker safety controls remain responsible for task execution semantics.
