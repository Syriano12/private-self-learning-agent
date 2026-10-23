# Phase 28 — Portable Scanner Runtime Deployment Kit

## Scope and verdict

This package prepares a reproducible, administrator-controlled deployment for a dedicated Ubuntu 24.04 host. It does not provision the current Sandbox, does not install packages, and does not activate external scanner execution here.

The current environment remains **BLOCKED**. The preflight utility reports the blockers instead of treating a binary lookup or an environment flag as proof of isolation.

## Files

- `src/private_agent/skills/runtime_preflight.py` — bounded read-only host checks with `PASS`, `FAIL`, and `INCONCLUSIVE`.
- `scripts/phase28_runtime_acceptance.py` — opt-in host acceptance harness. It uses only a local digest-pinned image, `--pull=never`, fixed argv, and controlled no-production probes.
- `tests/test_phase28_runtime_kit.py` — deterministic contract tests for the preflight and activation boundary.
- `docs/PHASE28_PORTABLE_RUNTIME_KIT.md` — this deployment and acceptance procedure.

## Read-only preflight

From the backend repository:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m private_agent.cli runtime-preflight --json
```

Exit codes:

- `0`: every preflight check is `PASS`.
- `1`: at least one required check is `FAIL`.
- `2`: no check failed, but at least one required property is `INCONCLUSIVE`.

The command never installs packages, changes sysctls, edits subordinate-ID files, starts a container, pulls an image, or executes a scanner.

## Administrator-controlled Ubuntu 24.04 setup

Run the following only on a dedicated host by an authorized administrator after reviewing the organization's change and rollback process. Do not paste credentials or tokens into these commands or deployment files.

```bash
sudo apt update
sudo apt install --yes podman uidmap slirp4netns fuse-overlayfs crun gpgv jq

# Replace vulnsandbox with the dedicated non-root service account.
sudo useradd --create-home --shell /bin/bash vulnsandbox || true
sudo loginctl enable-linger vulnsandbox
sudo loginctl terminate-user vulnsandbox || true

# The ranges must be dedicated to the service account and must not overlap.
grep '^vulnsandbox:' /etc/subuid /etc/subgid

# Run as vulnsandbox, not as root.
su - vulnsandbox -c 'podman info --format json'
```

The exact package source, repository pinning, update cadence, and host firewall policy belong to the administrator's change-management system. The project must not silently install or upgrade them.

The host must provide:

- A verified rootless Podman installation and actual rootless operation.
- `newuidmap`/`newgidmap`, subordinate UID/GID ranges, and approved rootless networking helpers.
- cgroup v2 with CPU, memory, and pids controllers.
- a kernel capable of seccomp and user namespaces.
- a host policy allowing read-only root filesystems, narrow tmpfs/work volumes, dropped capabilities, and `no-new-privileges`.
- process-group and container cancellation that can be independently tested.
- enough disk and memory for the configured bounded workload.

## Runtime acceptance harness

Create or import an administrator-approved local acceptance image. The image reference must include an immutable digest and must already exist locally; the harness never pulls it. The image must contain only the controlled test utilities required by the fixed probes (`/bin/sh`, `cat`, `awk`, and `wc`). It must not contain production credentials, scanner templates, or customer data.

Run on the dedicated host as the non-root service account:

```bash
VULNORA_ENABLE_RUNTIME_ACCEPTANCE=1 \
VULNORA_ACCEPTANCE_IMAGE='localhost/vulnora-acceptance@sha256:<64-lowercase-hex>' \
PYTHONDONTWRITEBYTECODE=1 \
python3 scripts/phase28_runtime_acceptance.py --json
```

The harness refuses to run unless explicit opt-in is set, preflight is `PASS`, and the image digest is locally verified. It uses `--pull=never`, `--network=none`, `--read-only`, `--cap-drop=all`, `no-new-privileges`, bounded pids/memory/CPU, and a small `tmpfs` for its fixed probes. It checks:

- read-only root filesystem and a narrowly writable temporary volume;
- memory and pids cgroup values visible inside the container;
- no default network route in the network-disabled probe;
- no expected host secret path mounted;
- task-local temporary storage is not shared between independent containers;
- local image identity matches the requested digest.

A `PASS` from this harness is evidence only for the listed probes and image. It is not permission to scan external targets and is not a substitute for the scanner-specific acceptance suite below.

## Scanner-specific acceptance still required

Before enabling `httpx` or Nuclei, add and run a dedicated host suite using only a controlled local test application and safe, administrator-approved templates. The suite must record raw evidence and fail rather than downgrade a missing assertion. It must verify:

1. scanner process and PID namespace isolation;
2. no read access to unrelated host files or service-account secrets;
3. read-only base filesystem, bounded writable volume, and no host bind mounts;
4. CPU, memory, pids, output, and wall-clock limits under stress;
5. network denial outside the explicitly permitted local test network;
6. cancellation, process-tree cleanup, and no orphan descendants;
7. task-to-task isolation with concurrent containers;
8. signed binary provenance, exact executable digest, and safe-template manifest/content digests;
9. continued ScopeDefinition enforcement and Phase 24 Evidence Gate behavior;
10. `EXECUTION_UNKNOWN` handling when a runtime result or cleanup acknowledgment is lost.

Until that suite exists and passes on the dedicated host, the scanner tools must remain unavailable.

## Activation contract

`HttpxReconTool` and `NucleiScanTool` no longer accept `external_sandbox_available=True`. Passing that legacy flag raises an error. The API passes no activation flag and no environment variable can activate a scanner.

A future built-in runtime adapter must issue an opaque `RuntimeActivationProof` only after it has validated the exact runtime identity, image digest, command-manifest provenance, and enforced execution boundary. The public constructor is disabled. Metadata inspection alone cannot issue the proof. The current `PodmanSandboxAdapter` deliberately cannot issue it, so the current default remains `scanner_sandbox_unavailable`.

## Tool provenance

For each binary and template set, retain administrator-controlled records for:

- publisher identity and trusted signer fingerprint;
- release version and exact artifact identity;
- signature verification result and checksum;
- local executable digest after installation;
- template manifest digest and every template content digest;
- revocation status and update decision;
- rollback artifact and previous known-good identity.

A caller-supplied checksum, version string, keyring path, or environment variable is not publisher authentication.

## Rollback and cleanup

If any acceptance assertion fails:

1. keep scanner activation disabled;
2. stop and remove only the dedicated acceptance containers and temporary volumes;
3. preserve the preflight JSON, acceptance JSON, runtime logs, and image/template identities;
4. revoke the failed binary/template identity from the administrator's allowlist;
5. restore the previous known-good runtime/tool artifacts through the administrator's package/change process;
6. rerun preflight and the complete acceptance suite before any reactivation.

Do not use `git reset`, `git clean`, `git restore`, `git checkout`, `git stash`, or `git revert` as part of runtime rollback.

## Current Sandbox result

In the Manus Sandbox used for Phase 28, the preflight is expected to remain `FAIL`: no Podman/Docker OCI runtime, no UID/GID mapping helpers, and no rootless networking helper are installed. No scanner or container is started here.
