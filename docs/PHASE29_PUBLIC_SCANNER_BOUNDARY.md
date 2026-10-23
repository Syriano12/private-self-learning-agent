# Phase 29 — Public Scanner Adapter Boundary

## Decision

`HttpxAdapter` and `NucleiAdapter` are now non-executing adapters. They may validate the authorized target and, for Nuclei, validate the trusted template manifest, but they never invoke a host process and never parse output from a host process. If a valid target reaches either adapter while no verified external executor is wired, the result is `UNAVAILABLE` with `external_sandbox_executor_unavailable`.

`SafeSubprocessRunner` remains only as a compatibility facade for old imports. Its `run()` method is inert and always returns `ScanStatus.UNAVAILABLE`; it does not inspect or execute caller-supplied argv. It is not an execution boundary.

`RuntimeActivationProof` alone is not an execution capability. The registered tools continue to require a proof and then fail closed because no bound OCI executor exists.

## Call-graph review

| Path | External-process behavior | Decision |
|---|---|---|
| `HttpxAdapter.scan()` | Scope validation only; no `availability()`, argv construction, or runner call | Blocked with `external_sandbox_executor_unavailable` |
| `NucleiAdapter.scan()` | Scope and template-manifest validation only; no scanner process or output parsing | Blocked with `external_sandbox_executor_unavailable` |
| `SafeSubprocessRunner.run()` | Compatibility method returns `UNAVAILABLE` without starting a process | Blocked/inert |
| `ProjectDiscoveryProbe.probe()` | No executable lookup, version probe, help probe, or caller runner injection | Blocked with provenance/sandbox reason |
| `HttpxReconTool.execute()` / `NucleiScanTool.execute()` | Requires exact `RuntimeActivationProof`; still refuses without a bound external executor | Blocked with `scanner_sandbox_unavailable` or `scanner_sandbox_executor_unavailable` |
| API → queue → Worker → `GenericExecutor` | Uses the registered tools; approval and policy checks remain active; tool fails before adapter execution | Blocked; fake executable is not called |
| `trusted_tools._signed_checksum()` | Fixed `/usr/bin/gpgv` may be invoked only to verify signed local provenance files; it never executes a scanner binary or target code | Provenance verification only, not scanner execution |
| `runtime_preflight.py` / `sandbox_contract.py` | Read-only probes for runtime capabilities (`podman`/`unshare`) through injected/preconfigured subprocess runner | Host capability detection only; no target code or scanner execution |
| `scripts/phase28_runtime_acceptance.py` | Explicit acceptance harness for a provisioned Podman runtime; uses fixed acceptance commands and `shell=False` | Not part of API/Worker public scanner path; remains opt-in and unrun in Sandbox |
| `skills/lifecycle.py` | Subprocess strings are static forbidden-pattern detection | No process execution |

A source scan after the change found no `subprocess.run`, `Popen`, `runner.run`, or equivalent call remaining in `src/private_agent/bugbounty/scanners.py` or `src/private_agent/tools/scanners.py`.

## Tests

The new regression file `tests/test_phase29_public_scanner_boundary.py` proves:

- direct `HttpxAdapter.scan()` does not call a fake executable;
- direct `NucleiAdapter.scan()` does not call a fake executable;
- direct `SafeSubprocessRunner.run()` does not call a fake executable;
- caller-supplied Probe runner injection is rejected;
- all fail closed with an explicit unavailable result.

Existing Phase 25/26 tests were updated to assert the new quarantine contract rather than treating host fixture execution as a valid scanner result. The API/Worker regression also asserts that an approved queued scanner task does not create the fake executable marker.

## Limitations

This phase does not create an external sandbox and does not make live scanning available. A future implementation must introduce a genuinely verified OCI execution boundary with immutable runtime/image identity, signed tool provenance, resource and filesystem controls, network policy, process-tree cleanup, and an opaque task-bound execution capability. Until then all active scanners remain unavailable by design.
