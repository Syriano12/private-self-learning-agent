# Phase 27 — Isolated Runtime Provisioning and Verification

## Verdict

**BLOCKED** for real ProjectDiscovery `httpx`/Nuclei execution in the current Sandbox environment.

This phase did not enable active external scanning, did not install an untrusted binary, and did not use `SafeSubprocessRunner` as a substitute for an OS-enforced sandbox.

## Environment evidence

- OS: Ubuntu 24.04.5 LTS.
- Kernel: `6.18.38+`.
- Process user: `ubuntu` (`uid=1000`), not root.
- `podman`, `docker`, `buildah`, `crun`, `runc`, `bwrap`, and `firejail`: unavailable.
- `newuidmap`, `newgidmap`, `slirp4netns`, `pasta`, and `fuse-overlayfs`: unavailable.
- `/etc/subuid` and `/etc/subgid` contain ranges for `ubuntu`, but no OCI runtime or rootless networking/UID helpers are installed.
- A harmless `unshare --user --map-root-user id` test succeeded. This proves only that a user namespace can be created; it does **not** prove filesystem, network, process, resource, or secret isolation.
- cgroup v2 is mounted and exposes CPU, memory, and pids controllers at the host level, but no scanner-specific cgroup is created or controlled by the project.
- The current process is not under a seccomp filter (`Seccomp: 0`) and no project-controlled AppArmor/SELinux profile was found.
- Root filesystem is writable (`ext4`, `rw`); no read-only scanner root or restricted writable mount is configured.

## Implementation outcome

The existing Phase 15D contract remains fail-closed:

- `PodmanSandboxAdapter.attest()` raises `SandboxUnavailable` because full controls are not independently verified.
- `ExternalSandboxRunner` cannot accept custom adapter injection and cannot execute a host fallback.
- `build_orchestrator()` passes `external_sandbox_available=False` to the registered scanner tools.
- `HttpxReconTool` and `NucleiScanTool` return `scanner_sandbox_unavailable` before parsing target inputs or invoking subprocess execution.
- Phase 26 signed provenance remains required. A version string, arbitrary SHA-256, environment flag, or caller-provided keyring is not treated as publisher authenticity.
- DNS-pinned research transport keeps rejecting mixed allowed/disallowed DNS answers and revalidates addresses before opening a socket, but this transport is not silently substituted for scanner execution.

## Real-tool validation

No actual ProjectDiscovery binary was executed. No Nuclei template was executed. No container was started. No external production target was contacted.

The existing fake-executable tests remain unit/integration tests for parsing, approval, timeout, output bounds, scope, and fail-closed behavior only; they are not evidence of real-tool execution or isolation.

## Minimum prerequisite to continue

A future execution environment must provide, through an administrator-approved and auditable setup:

1. A supported rootless OCI runtime, preferably Podman, with a verified version and actual rootless operation.
2. `newuidmap`/`newgidmap` and an approved rootless network mechanism, or a documented network-disabled policy for local-only tests.
3. A verified image digest and command-manifest provenance.
4. Enforced read-only base filesystem, narrowly scoped writable directory, no host secret mounts, CPU/memory/pids/time/output limits, and process-tree cleanup.
5. Runtime-enforced network allowlisting and tests proving denial outside the authorized destination.
6. Independently verified ProjectDiscovery release signatures/checksums and trusted Nuclei template provenance.
7. A controlled local test application for real-tool validation.

Installing a runtime alone is insufficient. The project must first implement and test a built-in adapter that produces trustworthy attestation for all required controls. Until then, changing `external_sandbox_available` would be a security bypass and is prohibited.
