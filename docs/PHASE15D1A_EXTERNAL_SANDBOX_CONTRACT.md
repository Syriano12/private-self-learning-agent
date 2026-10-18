# Phase 15D.1A — External Sandbox Contract and Runtime Capability Detection

## Scope

This subphase defines the trusted contract and read-only capability probe for a future external sandbox runner. It does not create a container, execute a target repository, execute a candidate callable, run tests/builds/hooks, install dependencies, or change host configuration.

The existing skill lifecycle remains fail-closed and continues to return `QUARANTINED` with `external_sandbox_unavailable` when no verified external sandbox is available.

## Contract

`private_agent.skills.sandbox_contract` defines three primary types:

- `SandboxSpec`: a frozen, bounded specification containing the immutable Podman runtime name/version, a pinned `sha256:` image identity, a pinned trusted-command-manifest digest, timeout/CPU/memory/process/output limits, a disabled network policy, and required security-control names.
- `SandboxResult`: a frozen result envelope compatible with the existing lifecycle fields while adding exit code, signal, timeout, runtime/image/manifest identities, bounded output digests, and an explicit `ValidationOutcome`.
- `SandboxUnavailable`: a structured fail-closed exception with the `SANDBOX_UNAVAILABLE` code and machine-readable reasons.

Validation rejects mutable image tags, malformed identities, unsupported host/network fallback values, missing or duplicate controls, non-finite or out-of-range resource limits, invalid exit metadata, and malformed output digests. `VALIDATION_UNKNOWN` is the conservative default for a result that has not been validated; `SANDBOX_UNAVAILABLE` is used when the external trust boundary cannot be established. A public `SandboxResult` constructor cannot mark a result `VALID`; a future runner must add the trusted attestation path rather than treating caller-supplied identity strings as proof.

The existing `SandboxResult` import from `private_agent.skills.lifecycle` is preserved by re-exporting the contract type through that module. No policy or approval semantics were changed.

## Read-only Podman detection

`detect_podman()` probes only:

1. `podman version --format {{.Client.Version}}` to verify the executable response and version shape.
2. `podman info --format json` to verify reported rootless mode and a non-empty seccomp profile.
3. `podman image inspect --format {{.Id}} <pinned digest>` when a `SandboxSpec` is supplied, to compare the local image identity without starting a container.

The detector invokes commands directly with `shell=False`, uses a two-second timeout per probe, rejects non-text or oversized stdout/stderr before parsing, does not pass repository paths or target commands, and never starts a container. JSON info parsing rejects duplicate keys and non-standard constants. A binary being present is not sufficient: the report distinguishes `detected`, `version_verified`, security verification, image identity verification, structural manifest pinning, manifest provenance verification, network-policy verification, and the final `ValidationOutcome`.

The current Sandbox environment has no `podman`, `docker`, or `nerdctl` executable on `PATH`. Therefore the actual result is `SANDBOX_UNAVAILABLE` with `runtime_missing`; no runtime probe command, image operation, or container was executed. The kernel/user facts were read-only observations only.

## Security boundary and limitations

This is a contract and detection foundation, not a runner. It does not claim that Podman is installed, rootless, seccomp-enforced, image-available, or safe for execution in this environment. The command-manifest digest is only structurally pinned here; a future trusted manifest registry or equivalent trust root must establish its provenance before it can be considered verified. This phase also does not prove that a future runner will enforce `network_policy=disabled`. Capability reports created by callers are not accepted as available without detector attestation. It does not verify a remote registry, create a container, apply limits, or execute `SandboxSpec`. These missing proofs produce `SANDBOX_UNAVAILABLE` rather than an execution fallback.

There is no fallback to `fork`, `multiprocessing`, a temporary directory, host Python execution, repository tests, builds, hooks, or arbitrary commands. The next subphase must separately design and review the actual runner before any execution is enabled.

## Verification

Focused contract tests cover missing/malformed runtime metadata, missing and unpinned image identities, unsupported security controls, missing security facts, bounded resources, immutable identities, host/network fallback rejection, structured unavailable errors, result metadata, and preservation of the existing zero-side-effect quarantine behavior.
