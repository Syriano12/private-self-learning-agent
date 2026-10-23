# Phase 18 — Vulnora local integration

## Supported local startup

Run the backend from the repository root with a local-only listener, an interactive owner bootstrap, and a required worker:

```bash
export AGENT_REQUIRE_OWNER_AUTH=1
export AGENT_OWNER_EMAIL=ashbm1995@gmail.com
export AGENT_DB_PATH="/tmp/vulnora-agent.sqlite3"
export AGENT_ALLOWED_ORIGINS="http://localhost:3000"
PYTHONPATH=src python3 -m private_agent.cli auth init-owner
PYTHONPATH=src python3 -m private_agent.cli serve --with-worker --host 127.0.0.1 --port 8123
```

`GET /health` is a liveness check. `GET /ready` is operational readiness: when the application requires a worker it reports `worker=running` only after the worker lifecycle has started. A worker-less operational app reports `503 not_ready` and rejects new task creation with `worker_not_ready`; it does not accept tasks that would silently remain queued.

Run Vulnora separately:

```bash
cd /home/ubuntu/agentctrl
VITE_AGENT_API_BASE_URL=http://127.0.0.1:8123/api/v1 pnpm dev
```

Vulnora signs in with the configured owner email and an HttpOnly session cookie; mutating requests also send the readable CSRF cookie as `X-CSRF-Token`. No password or session token is placed in the bundle, URL, browser storage, service-worker cache, or logs. Remote API origins must be HTTPS and must match the configured origin; HTTP is accepted only for loopback development. Backend cross-origin access is disabled unless `AGENT_ALLOWED_ORIGINS` contains explicit HTTP(S) origins; wildcard CORS is rejected.

## Real request path

A normal task follows:

```text
Vulnora ApiClient
  -> authenticated POST /api/v1/tasks
  -> SQLite task_states + task_queue transaction
  -> BackgroundWorker lease and heartbeat
  -> Orchestrator / SecurityController / GenericExecutor
  -> observation and verification
  -> persisted task state and GET /api/v1/tasks/{task_id}
  -> GET /api/v1/tasks history in Vulnora
```

The frontend task list reads the server-owned paginated history. Session-created rows are not the authorization boundary. Approval decisions remain empty-body server actions and cannot change bindings. `EXECUTION_UNKNOWN` is treated as a stop condition; neither backend worker recovery nor Vulnora polling automatically replays it.

## Bug Bounty integration

`POST /api/v1/bug-bounty/tasks` accepts a bounded, explicit `ScopeDefinition`, an optional deterministic fixture, and bounded local artifacts. The request is persisted as a normal queued task with a typed `task_kind`. The worker dispatches a deterministic plan to the registered `bugbounty_analysis` tool; the plan still passes normal plan validation, policy evaluation, approval rules, execution-boundary checks, observation, and verification.

Available read endpoints:

- `GET /api/v1/bug-bounty/tasks/{task_id}` — status, sanitized run, limitations, and Markdown report when complete.
- `GET /api/v1/bug-bounty/tasks/{task_id}/inventory` — persisted asset/endpoint inventory.
- `GET /api/v1/bug-bounty/tasks/{task_id}/findings` — finding lifecycle records.
- `GET /api/v1/bug-bounty/tasks/{task_id}/evidence` — sanitized finding/test evidence.
- `GET /api/v1/bug-bounty/tasks/{task_id}/report` — structured report text and run ID.

The first release is local-only: deterministic fixtures, passive local URL/OpenAPI parsing, and static source-artifact analysis. It has no live target HTTP connector, browser automation, exploit delivery, arbitrary code execution, or zero-day guarantee. Active fixture testing requires an explicit active scope and test identities/resources. The UI labels these limits and never turns a hypothesis into a confirmed external finding.

## E2E coverage

`agentctrl/src/integration/backend.e2e.test.ts` starts `tests/vulnora_e2e_server.py`, imports the actual TypeScript `ApiClient`, and uses the real FastAPI application, SQLite storage, worker, orchestrator, execution boundary, and verification path with a deterministic local tool. It verifies readiness, queue acceptance, completion, `VERIFIED` state, persisted task history, and authentication rejection. Backend `tests/test_phase18_integration.py` additionally verifies worker-less readiness/task rejection, Bug Bounty queue-to-report behavior, section endpoints, CORS allowlisting, and unauthorized access.

## Remaining limitations

This is a locally testable integration, not production readiness. The backend remains single-owner rather than multi-tenant, the LLM provider is not required for the deterministic integration harness, live external target testing is intentionally disabled, and a real externally verified OCI sandbox is still unavailable. Use only written authorization for any future live connector work.
