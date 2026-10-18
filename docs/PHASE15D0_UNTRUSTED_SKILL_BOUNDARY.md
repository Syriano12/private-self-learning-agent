# Phase 15D.0 — Untrusted Skill Execution Boundary Remediation

## Scope

This remediation closes the immediate unsafe skill-execution paths identified by the Phase 15D audit. It is not an implementation of the full external sandbox and it does not claim strong isolation.

## Enforced behavior

The former `multiprocessing`/`Pipe` path has been removed. The agent no longer receives worker results through `Connection.recv()`, which could deserialize arbitrary Python objects with pickle semantics. A future external runner may use `decode_sandbox_result()`, a bounded UTF-8 JSON protocol that accepts only a small result envelope and JSON primitive values. It rejects binary payloads, malformed JSON, unknown fields, invalid statuses, missing `output` on success, missing `error` on failure, non-primitive values, excessive nesting, and oversized messages.

`SkillSandbox.execute()` now fails closed with `external_sandbox_unavailable`. It never forks, invokes `candidate.implementation`, reads a worker pipe, or treats a local process boundary as a security sandbox. Direct calls therefore produce quarantine without candidate-code side effects.

Activation and runtime execution also reject replacement sandbox providers at the execution seam. A provider assigned after construction is quarantined before its `execute()` method is reached, and the execution state is populated consistently for the trusted result validator.

`MutationTestingEngine` still generates bounded static mutation cases, but it does not invoke either the candidate-provided `mutation_test_runner` or a callable supplied to `run()`. When a runner would be required, the result is not passed and carries `external_sandbox_required`. `SkillLifecycleManager.submit()` preserves static contract/AST analysis and serializable metadata persistence, but candidates are quarantined and are not activated until an independently isolated runner exists. Learned callables are not retained as an execution authority in the manager's in-memory candidate map.

Candidate metadata serialization is explicit and excludes callable fields before sanitization; it does not use `dataclasses.asdict()` on the complete candidate, avoiding arbitrary `__deepcopy__` behavior from a supplied callable.

Existing tool authorization, SecurityController, approval binding, TrustedOperationBinding, execution receipts, lease checks, and Phase 15C patch controls are unchanged. The new quarantine is an additional prerequisite; it is not an approval bypass.

## Compatibility impact

Phase 8 skill activation and runtime execution are intentionally unavailable in this phase. Existing built-in/test callable workflows that depended on in-process `fork` execution now return `QUARANTINED`. Static analysis, contract validation, mutation generation, candidate persistence, redaction, and the normal tool execution boundary remain available. This behavior is intentional: keeping the old path would allow untrusted implementations or mutation runners to execute inside the agent host.

The current repository does not provide an external OCI/runtime sandbox, seccomp policy, process-group cleanup boundary, or locked validation image. No fallback to `fork`, temporary-directory execution, pickle IPC, repository tests, repository builds, hooks, or arbitrary target commands is permitted.

## Verification

Focused Phase 15D.0 tests cover:

- rejection of binary pickle-shaped payloads without Python object deserialization;
- malformed and oversized result rejection;
- acceptance of bounded primitive JSON results;
- direct `SkillSandbox` fail-closed behavior with zero implementation side effects;
- candidate mutation runner non-execution;
- `submit()` retaining static analysis while quarantining runtime execution;
- static analysis and contract validation remaining usable.

Further external sandbox work remains out of scope for this phase.
