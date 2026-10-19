# Phase 15D.1B — External Sandbox Runtime Adapter

## Architecture assessment

Phase 15D.1A defines bounded `SandboxSpec`, `SandboxResult`, `SandboxUnavailable`, and a read-only Podman capability probe. Phase 15D.0 keeps skill activation fail-closed because candidate callables and mutation runners must never execute in the agent host. The missing runtime components are therefore not a second Python worker or a fork-based shim; they are a separately reviewed external-runtime adapter and its proof boundary.

A real implementation still needs all of the following before it may execute anything:

1. A trusted command-manifest registry or equivalent signature/provenance root. A digest supplied by a caller is not provenance.
2. An adapter-owned invocation builder that accepts only a trusted manifest reference, never arbitrary `argv`, shell text, Python callables, repository hooks, or candidate-provided commands.
3. Independently verified immutable image identity and runtime identity.
4. Enforced and observable rootless execution, CPU/memory/pids/output limits, timeout handling, process-group cleanup, and a bounded result channel.
5. A restricted filesystem mount policy with no host-root or unrestricted repository access.
6. An independently verified disabled network policy.
7. A result attestation path binding the result to the exact spec, image, manifest, and controls, followed by the existing runtime verification, approval, and SecurityController boundaries.
8. Integration into `SkillSandbox` only after the complete proof exists, with conservative handling of timeout, cleanup failure, malformed result, and uncertain external state.

## Implemented safely in this subphase

`private_agent.skills.sandbox_runtime` adds:

- `SandboxRuntimeAdapter`, a strict abstract interface with `inspect`, `attest`, and `execute` methods. It does not accept arbitrary commands, host paths, or callables.
- `SandboxAttestation`, an immutable evidence envelope covering runtime, image, command-manifest provenance, rootless mode, resource limits, process cleanup, filesystem restrictions, network policy, and output limits.
- `validate_attestation()` and `validate_sandbox_result()` gates that require exact spec identity and all controls.
- A private adapter attestation token. A normal caller can construct data-shaped `VALID` fields, but cannot make them trusted through the public constructor; the internal factory rejects an incorrect token. This is an in-process trust gate, not a cryptographic boundary against already-executing host code.
- `PodmanSandboxAdapter` as a read-only, quarantine-only adapter. It delegates inspection to the existing Phase 15D.1A probe, refuses to fabricate missing controls, and has no execution implementation.
- `ExternalSandboxRunner`, which returns a `QUARANTINED` `SandboxResult` for unavailable or invalid state. Its public constructor rejects all adapter injection, including subclasses and mutated adapter instances, and the runner keeps no mutable adapter state, preventing an injected host-execution fallback.

The existing `SkillSandbox` remains unchanged and continues to quarantine without a verified external runtime. No SecurityController, human-approval, or Phase 15A–15D.0 behavior was weakened.

## What is enforced versus specified

Enforced now: immutable and bounded contract inputs; exact identity matching; required-control coverage; private attestation provenance; strict runtime-report availability; malformed/unknown result rejection; no arbitrary command or callable parameter in the adapter interface; no custom adapter injection; and fail-closed quarantine.

Specified but not operational: actual container creation, command-manifest provenance, rootless enforcement, cgroups/resource enforcement, process cleanup, filesystem isolation, network isolation, and successful result attestation. The current environment has no Podman executable, and this phase deliberately does not install dependencies, start containers, or execute target code. Consequently no claim of real sandbox isolation is made.

## Verification boundary

All tests use deterministic data-only seams. They do not start Podman, create a container, execute target code, invoke a candidate callable, run target tests/builds/hooks, or use multiprocessing, fork, pickle, or host-execution fallback. A real runtime integration test belongs only after the adapter implementation and its control attestations are independently available.
