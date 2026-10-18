# Phase 15C — Safe Code Modification

## Scope and integration

Phase 15C adds a bounded patch lifecycle to the existing `ToolRegistry`, `GenericExecutor`, `SecurityController`, `Planner`, `Orchestrator`, task persistence, and observability paths. It does not create a parallel agent framework, execute target-repository code, run target tests/builds/hooks, create commits, push, or modify an upstream repository.

The registered tools are:

- `prepare_repository_patch`: read-only preparation. It validates a caller-supplied unified diff against a configured local repository fixture, checks the pinned `local-tree:<sha256>` revision, enforces permitted files and configured limits, performs Python AST/JSON structural validation, stores durable session metadata, and returns a reviewable diff. It never writes the configured source tree.
- `apply_repository_patch`: high-risk `isolated_write` operation. It is authorized by the normal execution boundary and requires the existing approval policy for its HIGH risk. It rechecks the repository identity and base revision, copies the original content into an agent-owned workspace, applies the exact prepared diff there, validates the resulting files, and stores application metadata. It never writes the configured source tree.
- `rollback_repository_patch`: high-risk `isolated_write` operation. It requires the normal approval path, verifies that every workspace file still has the recorded post-application digest, and restores only the isolated workspace from the recorded backup. It refuses rollback on any concurrent workspace change.

The `isolated_write` permission is added to the default API Orchestrator permission set. Existing `private_write` permissions remain excluded. Owner `DENY`, unknown capabilities, approval binding, policy version checks, and the fail-closed execution boundary remain authoritative.

## Safety contract

A patch session is local-only and must resolve through the configured `RepositoryInspector.local_repositories` mapping. The agent-owned workspace root must be a separate, non-overlapping filesystem tree from every configured source repository. Remote repositories are rejected for modification; the existing Phase 15A/15B read-only GitHub path is not reused as a write authority.

The patch parser accepts a bounded unified diff only. It rejects path traversal, absolute paths, sensitive or excluded paths, renames, deletions, malformed headers/hunks, duplicate files, symlinks, binary/oversized content, secret-like literals, excessive files, excessive changed lines, and unexpected base revisions. Patch paths must remain inside the explicitly supplied `permitted_files` set. Preparation bounds the permitted-file list and serialized diagnosis size before persisting metadata.

The source tree is scanned with a content-addressed local revision. The revision includes safe regular files and their SHA-256 digests, while excluded/sensitive directories are ignored according to the existing repository policy. Symlinks and unsupported filesystem entries fail closed.

Preparation stores a `metadata.json` session under the configured agent-owned workspace root. The record includes the patch identifier, repository, base revision, permitted files, bug report, exact diff, digest, changed-file original/proposed digests, Phase 15B diagnosis/evidence linkage when supplied, validation result, application state, rollback state, and the explicit fact that target code was not executed. A later process can reload the session by `patch_id`; application revalidates the source revision before creating the isolated workspace.

Approval is intentionally separate from preparation. The tool's HIGH-risk `apply` and `rollback` actions go through `SecurityController.authorize()` and the existing `ApprovalGate` binding to task, action, tool, goal, inputs, capabilities, risk, and current policy version. The public `execute()` entry point for those mutating tools fails closed; only the `GenericExecutor` trusted-operation hook can reach the manager, and that hook revalidates the controller-issued binding, active authorization evidence, permission, current policy, and approval before writing. A pending or denied approval produces no workspace write. Owner policy `DENY` is evaluated before approval/allow and also produces no workspace write.

No operation imports, executes, compiles, installs, or invokes code from the target repository. Structural checks are limited to parsing changed Python and JSON text. Runtime behavior remains explicitly unverified until a future phase provides an independently sandboxed execution design.

## Rollback model

Application writes only to `<workspace_root>/<patch_id>/workspace`. Original bytes are copied to a sibling rollback directory before the proposed bytes are written, including zero-byte existing files. The configured repository remains unchanged. Rollback takes a per-session lock, verifies the complete workspace file/directory inventory against the recorded post-application state, and uses compare-before-write/delete checks; a changed or extra entry yields `rollback_conflict` without updating the session as rolled back. Patch-generated files are deleted only when they still match the recorded post-application state. A process interruption during a multi-file rollback remains conservative and must not be interpreted as a completed rollback.

This is a bounded reversible workspace operation, not an upstream Git operation and not an exactly-once guarantee for external systems. A process interruption during a filesystem transaction is reported conservatively; callers must not infer that a patch was applied unless the persisted application result says `APPLIED`.

## Validation

Focused validation:

```text
python3 -m pytest -q tests/test_phase15c_patch_lifecycle.py
# 21 tests (including direct-call fail-closed regression, workspace isolation, rollback inventory/empty-file/interleaving regressions, structurally different JSON fixture, limits, approvals, and Orchestrator persistence)
```

Compatibility validation:

```text
python3 -m pytest -q tests/test_phase15a_repository.py tests/test_phase15a_github_access.py tests/test_phase15b_bug_investigation.py tests/test_phase15c_patch_lifecycle.py
# 83 passed
```

Full validation:

```text
python3 -m pytest -q
# 385 passed

python3 -m compileall -q src tests
# passed

git diff --check
# passed
```

The Phase 15C tests cover valid preparation, evidence linkage and stale evidence, unauthorized scope, traversal/sensitive content, symlinks, base revision mismatch, malformed hunks, size limits, structural validation, original-content preservation, approval-required application, owner DENY, isolated workspace writes, target-code non-execution, concurrent-change rollback refusal, remote modification rejection, task persistence, and the existing Orchestrator execution path.

## Limitations

Patch session metadata is file-backed under the agent-owned workspace root; the task result and approval events are also persisted through the existing task/audit mechanisms when invoked through Orchestrator. There is no remote repository write path, commit/PR automation, runtime test execution, sandboxed target execution, or automatic retry for uncertain external behavior in this phase.
