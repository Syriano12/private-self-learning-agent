# Phase 15A — Read-Only Repository Intelligence

## Implemented architecture

The existing `ToolRegistry` registers six repository tools. `api.build_orchestrator()` adds them to the default application without a new executor, API, worker, database, or agent framework. `Planner` can select them using their descriptions and schemas. Its existing no-provider fallback now accepts a bounded proposal from a trusted registered tool when the natural-language goal names exactly one GitHub repository. That proposal is validated normally and grants no permission.

`Orchestrator` still owns planning, preflight, execution, checkpoints, observations, verification, recovery, reflection, and learning. `GenericExecutor` still performs authorization and lease checks. GitHub tools declare `NETWORK_ACCESS`, which retains the existing HIGH-risk approval requirement. A configured GitHub credential is not an execution approval. No security invariant, policy precedence, approval binding, receipt, lease, watchdog, or EXECUTION_UNKNOWN implementation was changed.

The `inspect_repository_structure` operation performs bounded incremental exploration inside the authorized tool invocation. It lists the tree, parses supported manifests, derives candidate entry modules from manifest declarations, and reads selected source files. File choices depend on actual repository data, not a sequence tied to this project. The other five operations can also be planned independently through the same registry and executor.

Static Python syntax is parsed with `ast.parse`; JSON and TOML are parsed as data. Repository modules are never imported, dependency installers are never invoked, and scripts declared in manifests are never executed. No subprocess, clone, checkout, code edit, dependency installation, commit, push, or pull-request capability was added.

## Tools and schemas

Every input is a JSON object with required `repository: string`, optional `ref: string`, and `additionalProperties: false`. `repository` accepts `owner/name` or exactly `https://github.com/owner/name`, optionally ending in `.git`. GitHub Enterprise, SSH, `file:`, arbitrary hosts, credentials in URLs, query strings, fragments, encoded traversal, and repository subpage URLs are unsupported and rejected.

- `list_repository_files` adds optional `limit: integer`, from 1 to 200. It returns file paths, sizes, object SHAs where available, and an explicit truncation flag.
- `read_repository_file` requires `path: string`. Optional `start_line` is 1–1,000,000, `line_count` is 1–400, and `max_chars` is 1–16,000. It returns a UTF-8 excerpt, path, byte size, SHA-256, redaction status, and truncation status.
- `search_repository_code` requires `query: string`, limited to 128 characters. It accepts optional `path_prefix: string` and `limit: integer`, from 1 to 50. Search is literal, not executable expressions or regex. Matches cite paths and line numbers. Sampling and skipped files are disclosed.
- `inspect_repository_structure` uses only the shared repository/ref inputs. Its data includes main directories, manifest-declared entry points, dependencies, verified static declarations/imports with path/line references, lexical candidates, hypotheses, and evidence fingerprints.
- `inspect_repository_dependencies` uses only the shared inputs. Supported manifests are `pyproject.toml`, `requirements.txt`, `package.json`, `Cargo.toml`, `go.mod`, and `composer.json`. Constraints are reported as declared, not as installed or resolved versions. Unsupported/dynamic dependency systems remain unknown.
- `inspect_repository_metadata` uses only the shared inputs. It returns the canonical repository, selected ref, pinned commit where available, private flag, and source type.

All six use the same output envelope, including the content-bound repository snapshot:

```json
{
  "repository": "example/demo",
  "ref": "main",
  "commit": "<pinned GitHub commit SHA or empty for local fixture>",
  "source": "github_api",
  "untrusted": true,
  "operation": "inspect_repository_structure",
  "data": {},
  "limitations": [],
  "snapshot": {
    "schema_version": 1,
    "repository_identity": "github_api:example/demo",
    "snapshot_id": "sha256:<canonical snapshot digest>",
    "content_digest": "<SHA-256 of analyzed file records>",
    "tree_digest": "<SHA-256 of bounded tree manifest>"
  }
}
```

The envelope schema validates the existing fields and the added snapshot, and rejects additional top-level fields. Operation-specific evidence is in `data`. Successful results still pass through the existing observer and verifier; generic verification confirms the requested observable output fields, not runtime architectural correctness. The static analysis explicitly distinguishes source declarations from hypotheses and unavailable information.

## Immutable repository snapshot and CodeEvidenceRef

`RepositorySnapshot` identifies a bounded source view using canonical repository identity, source kind, pinned Git commit when available, source revision, SHA-256 content digest for files actually read, tree-manifest digest, exclusions, and explicit truncation state. Its sorted records are immutable after construction. For mutable local worktrees without a readable Git commit, `source_revision` is `working-tree`; the `snapshot_id` still changes when the observed tree/content or exclusion set changes. It is an integrity identifier, not a signature and not proof that the analysis is correct or complete.

`CodeEvidenceRef` binds each cited file/line range to `repository_identity`, `snapshot_id`, `commit`/`source_revision`, canonical path, full-file SHA-256, optional columns and symbol identity, parser/rule IDs and versions, evidence type, truncation, and uncertainty markers. A local result is accepted for report use only after `RepositoryInspector.verify_snapshot` rechecks the current tree, exclusions, Git commit when available, and every analyzed file digest. `validate_evidence_ref` performs that source verification before decoding the serialized ref. GitHub refs must correspond to an immutable commit and snapshots issued by the same inspector instance; after process restart they fail closed until the source is re-read by an explicit refresh flow. A raw digest or `CodeEvidenceRef.from_dict` by itself is never analysis validity or authorization.

`BugInvestigator` merges operation snapshots only when repository, revision, tree, and overlapping file digests agree. It revalidates the merged source and rewrites evidence refs against the unified snapshot before returning its diagnostic report. This does not change candidate/finding verification status, `SecurityController`, `ToolContract`, or Human Approval; a reference alone cannot promote a candidate. Repository content remains untrusted data and grants no execution or policy authority.

## Safe GitHub authentication

`GitHubAccess.from_environment()` reads only `GITHUB_READ_TOKEN` and comma-separated `GITHUB_READ_REPOSITORIES`. Private access requires both a securely supplied token and an exact repository allowlist entry. A token alone never enables private access to an arbitrary repository. Use a fine-grained read-only token restricted to the intended repositories and Contents: read permission.

Do not write the token into source, `.env` artifacts, logs, tool inputs, or a model prompt. Populate it through the deployment/operator secret configuration. `GH_TOKEN`, `GITHUB_TOKEN`, the host's logged-in `gh` CLI, and other ambient credentials are not borrowed by this capability. The existing task's GitHub integration is therefore not automatically transferred into the agent runtime.

Metadata is requested anonymously first. Subsequent commit/tree/blob requests use credentials only when the repository is private and allowlisted; public repositories remain anonymous even when listed. Requests use fixed `https://api.github.com` GET endpoints with redirects disabled. Content URLs returned by GitHub are not followed. Branch/ref resolution yields a commit SHA; subsequent tree/blob reads stay bound to that commit within each operation. Across separate operations, callers can use the returned commit as `ref` for consistency.

Errors expose fixed codes and safe messages, not GitHub response bodies, request headers, tokens, or stack traces. The access object has token-free representation and literal credential redaction. The tool layer also removes known sensitive environment values and common committed-secret patterns, and blocks credential files before reading them.

## Bounds and local fixtures

Default tool bounds are 64 KiB per file, 32 KiB per serialized result, 200 tree files, 12 source-read attempts, and 128 KiB cumulative content-read budget per operation. Known tree sizes are checked before reading. Each backend receives the remaining file allowance, and failed reads retain a conservative byte reservation rather than reopening the budget. Cached successful reads do not trigger another source request. Trusted construction configuration can lower limits; hard upper bounds are enforced. Structure analysis reads at most eight manifests and eight selected source candidates, subject to the stricter combined budget. Dependency output is capped at 100 declarations. GitHub response bodies are streamed with a separate default 2 MiB cap; oversized responses fail closed.

The local fixture adapter is constructor-injected using canonical repository identifiers mapped to operator-selected root directories. Model/task inputs cannot choose local roots. It uses directory file descriptors, `O_NOFOLLOW`, regular-file checks, and bounded reads to prevent traversal, symlink escapes, FIFOs, and unrestricted reads. Local tree scanning is bounded to 4,000 entries and eight levels. It excludes generated directories, credentials, SQLite/database files, and logs; oversized files, symlinks, unsafe entries, and unlisted tree remainders carry bounded exclusion reasons. This POSIX adapter is intended for controlled test fixtures, not as a GitHub clone manager.

Local and GitHub sources cannot be combined in one inspector. Unsupported mixed configuration fails explicitly instead of declaring network authority for a local operation. Register an inspector configured for the intended source. Local symlink/special-file omissions mark the tree incomplete; such entries are never followed.

Credential paths include `.env*`, `.git`, `.ssh`, `.aws`, `.azure`, `.kube`, `.netrc`, `.npmrc`, `.pypirc`, `.git-credentials`, singular/plural `secret`/`secrets` files (including hidden variants), credential files, and private key extensions. The same filtering is used for remote tool results. Redaction is defense in depth, not a proof that every possible arbitrary secret string can be recognized.

## Demonstrated end-to-end behavior

`test_real_agent_natural_language_task_returns_persisted_evidence_summary` invokes the actual `Orchestrator` with:

> Analyze this GitHub repository and explain its architecture: https://github.com/example/demo

Its controlled repository contains a Python dependency manifest, a CLI importing a service, and a `Service` class. The real registry, planner fallback, SecurityController, GenericExecutor, observations, generic verification, and SQLite checkpoints/results are used. The task reaches COMPLETED and its persisted output includes the `Service` declaration at the real source line, `httpx>=0.27` as a manifest declaration, and evidence for `pyproject.toml` and `src/demo/service.py`. There is exactly one authorized ACTION_EXECUTED audit event for the inspection operation.

A separate default-application test confirms that selecting a remote GitHub tool reaches WAITING_APPROVAL before repository access rather than silently permitting network activity. DENY and missing-controller tests fail if repository reading is attempted.

Core tests use controlled local fixtures and `httpx.MockTransport`; they do not depend on live GitHub or real personal tokens. Live public or private GitHub access was not tested in this phase.

## Remaining limitations

This is read-only repository intelligence, not autonomous bug fixing, code generation, execution, test running in an inspected repository, or PR preparation. This snapshot/evidence increment does not add new AST, call-graph, taint-analysis, or other Phase 15B capabilities.

Tree/search truncation, file-size failures, missing/binary files, malformed manifests, unsupported dependency formats, rate limits, timeouts, and unavailable repositories are reported safely. Results do not establish runtime behavior, service connectivity, installed versions, complete code coverage, or freedom from bugs. Python declarations are static facts; other supported source languages use labeled lexical candidates. Dynamic imports and uninspected modules remain unknown.

GitHub symlink/submodule and `.git` entries are recorded as excluded and never fetched; remote tree truncation includes an explicit remainder marker. Very large recursive trees can exceed the conservative response cap. There is no automatic cloning or unlimited subtree traversal. Local fixture reads are mutable working-tree reads; without a readable Git commit they are verified by a bounded re-read at report time, not pinned to a filesystem snapshot primitive. External request timeouts are per request; existing worker watchdog and EXECUTION_UNKNOWN rules continue to govern interrupted task execution.

## Official API references

[Git trees — read permissions, recursive truncation and limits](https://docs.github.com/en/rest/git/trees)

[Git blobs — read permissions and Base64 content](https://docs.github.com/en/rest/git/blobs)

## Actual validation and change manifest

Executed against the Phase 15A working tree:

```text
python3 -m pytest -q tests/test_phase15a_repository.py tests/test_phase15a_github_access.py
44 passed in 1.97s

python3 -m pytest -q
346 passed in 21.64s

python3 -m pytest --collect-only -q
346 tests collected in 1.66s

python3 -m compileall -q src tests
exit 0

git diff --check
exit 0
```

Collection and execution counts match; there were no skipped, deselected, or failed tests in the full run. The focused end-to-end natural-language repository task passed as part of both runs. Four new credential-path tests initially reproduced successful reads of `.netrc`, `.npmrc`, `.pypirc`, and `.git-credentials`; the shared path filter was corrected and all four now reject those files. No previous security tests were removed or weakened.

An independent read-only implementation review found further budget-accounting, singular-secret-file, omitted-symlink reporting, and prose-URL selection defects. Targeted regressions reproduced the budget leak, secret-file read, and fallback failures before correction; the final tests now enforce preflighted budgets, charged failed attempts, blocked secret files, explicit unsafe omissions, and natural-language/Markdown URL selection. A separate regression reproduced needless credential transmission to a public allowlisted repository; public access now stays anonymous. Unsupported mixed local/GitHub configuration is rejected explicitly. The end-to-end test also asserts that every static finding/dependency references an evidence path with a valid-length SHA-256. This is not an independent verification of runtime architecture; generic verification remains field-level as documented above.

Existing files changed: `src/private_agent/api.py`, `src/private_agent/core/planner.py`, and `src/private_agent/core/replanning.py`.

New implementation files: `src/private_agent/tools/github_access.py`, `src/private_agent/tools/local_repository.py`, `src/private_agent/tools/repository.py`, and `src/private_agent/tools/repository_analysis.py`.

New test files: `tests/test_phase15a_github_access.py` and `tests/test_phase15a_repository.py`. New documentation: this file.

Git remained on `main`; HEAD and the local `origin/main` reference stayed at `fe31f8dca909a80e4f150176bd60dfb53c3746b4`. Phase 15A changes are local and uncommitted. No staging, commit, push, destructive Git operation, or Phase 15B work was performed. No Git-visible generated artifacts or environment files were added.


## Phase 1 — Immutable Repository Snapshot + CodeEvidenceRef (2026-11-02 local working tree)

This increment adds `src/private_agent/tools/repository_snapshot.py` and `src/private_agent/tools/repository_paths.py`, and updates `src/private_agent/tools/repository.py`, `src/private_agent/tools/local_repository.py`, `src/private_agent/tools/github_access.py`, and `src/private_agent/tools/bug_investigation.py`. Regressions were added/updated in `tests/test_phase15a_repository.py`, `tests/test_phase15a_github_access.py`, and `tests/test_phase15b_bug_investigation.py`. These files may also contain unrelated local changes that predated this increment; this subsection records only the Phase 1 behavior and does not claim that every line in those files belongs to Phase 1.

Snapshot identity is built from canonical repository identity, source kind, commit/source revision, sorted bounded tree and analyzed-file manifests, content digests, exclusions, and truncation flags. Snapshot records are immutable mappings. Local snapshots use a readable commit when available; otherwise they are identified as `working-tree` and revalidated with a bounded tree/file reread. Evidence refs carry the snapshot/repository identity, source revision and commit, canonical file path, SHA-256, line range, optional columns/symbol, parser/rule IDs and versions, evidence type, truncation, and uncertainty. Excerpts now cover their returned multi-line range, and truncation is propagated from top-level and nested tool output into the snapshot and BugInvestigator evidence refs.

Source protections in this increment include hard bounds on serialized snapshot records, strict schemas, `O_NOFOLLOW` local reads, explicit remainder/exclusion reasons, rejection of snapshot drift (including deletion during revalidation), and one shared sensitive-path policy applied inside both local and GitHub adapters. The policy excludes common credential paths, SQLite files and rollback/WAL/SHM sidecars, and logs. GitHub requests explicitly set `trust_env=False` so ambient proxy/CA environment settings cannot silently intercept a private-repository bearer token. GitHub tests use `httpx.MockTransport`; no live GitHub requests or target scans were made for this increment.

BugInvestigator now merges and validates evidence against a consistent snapshot and preserves reference metadata. It does not change `SecurityController`, `ToolContract`, Human Approval, or finding-verification semantics. A digest/ref proves only bounded source linkage and integrity; it is not a signature, proof of parser execution, a correct analysis, a validated vulnerability, authorization, or permission to execute repository content. The existing static AST observations were not expanded into a new AST/call-graph/taint phase. Truncated evidence is now visibly marked, but consumers must still avoid treating static `ESTABLISHED` output as runtime reproduction or external vulnerability confirmation.

Validation performed on this local working tree:

```text
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider \
  tests/test_phase15a_repository.py \
  tests/test_phase15a_github_access.py \
  tests/test_phase15b_bug_investigation.py
76 passed in 9.30s

PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
730 passed in 76.84s

PYTHONDONTWRITEBYTECODE=1 python3 -m compileall -q src tests
exit 0

git diff --check
exit 0

Frontend: pnpm test — 37 passed; pnpm test:integration — 3 passed;
pnpm typecheck — exit 0; VITE_AGENT_API_BASE_URL=/api/v1 pnpm build — exit 0;
frontend git diff --check — exit 0.

Static AST guard: process_execution_static_scan=clean.
```

The Phase 1 work remains local and uncommitted on `main` at the existing HEAD `56831fd7119ccfae775893e9e187b60d5ec67dfa`. No staging, commit, push, install, external target request, or execution of inspected repository source was performed. This is local test evidence only; it does not establish behavior on Android/Termux or a deployed environment.
