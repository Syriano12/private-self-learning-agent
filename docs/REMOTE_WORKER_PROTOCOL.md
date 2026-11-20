# Remote scanner-worker protocol

This milestone adds a durable control-plane protocol for a future isolated scanner worker. It does **not** enable `httpx`, `nuclei`, shell execution, subprocess execution, or host fallback.

## Contract

The control plane stores worker identity metadata, credential digests and versions, jobs, leases, results, and replay nonces in schema 12. A worker receives a credential once through an out-of-band registration/rotation operation. The raw credential is never persisted or returned in a job envelope.

Each worker request is authenticated with `Authorization: Worker <credential>` plus `X-Worker-ID`, `X-Worker-Timestamp`, `X-Worker-Nonce`, and `X-Worker-Signature`. The signature covers the HTTP method, path, timestamp, nonce, and canonical request-body digest. The control plane enforces a bounded clock window and durable nonce uniqueness, so replayed requests are rejected.

The HTTP control-plane endpoints are:

- `POST /api/v1/worker/jobs/claim` with `{}`
- `POST /api/v1/worker/jobs/{job_id}/heartbeat` with `{job_id, lease_id}`
- `POST /api/v1/worker/jobs/{job_id}/result` with `{job_id, lease_id, result}`

The worker must retain the lease fence. A result with a stale lease, expired lease, wrong worker, or `EXECUTION_UNKNOWN` state is rejected. An expired leased job is converted to `EXECUTION_UNKNOWN`; it is never blindly retried because the scan may have happened remotely even if its response was lost. Repeated identical result submissions are idempotent; conflicting submissions are rejected.

## Dispatch gate

Active jobs require all of the following before an envelope can be created:

- a worker record whose runtime acceptance status is `VERIFIED`;
- pinned runtime, image, and execution-manifest identities;
- a valid, non-ambiguous, non-expired `ScopeDefinition` and an allowed active target, identity, resource, and method;
- an `APPROVED` control-plane approval for the exact task and an approval digest matching the request;
- a scope digest matching the supplied scope snapshot.

The current implementation has no verified worker acceptance record in this environment, so active dispatch remains blocked with `runtime_acceptance_required`. The protocol does not manufacture or self-assert a `RuntimeActivationProof`.

## Result boundary

Results are bounded and sanitized before persistence. Candidate findings are forced to `HYPOTHESIS`/`UNVERIFIED`; the result includes worker provenance and `finding_confirmation_allowed: false`. Owner credentials, cookies, API keys, tokens, private keys, and control-plane database access are not sent to the worker envelope or stored in results.

The only end-to-end execution path covered in this milestone is a deterministic local fixture job. It validates the protocol mechanics without contacting an external target. Passing these tests is not proof that an isolated scanner runtime exists or that live HackerOne testing is ready.
