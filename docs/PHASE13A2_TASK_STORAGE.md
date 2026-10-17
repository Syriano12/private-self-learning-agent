# Phase 13A.2 — Durable Task Storage and Task History API

## Implemented scope

Phase 13A.2 adds the persistence foundation for background task execution without starting a worker in this phase.

Implemented:

- SQLite schema migration 4.
- Durable `task_queue` table and indexes.
- Minimal `TaskState.QUEUED` status.
- Atomic queued-task creation.
- Atomic idempotency handling for queued creation.
- Authenticated `GET /api/v1/tasks`.
- Bounded keyset pagination.
- Typed, allowlisted task summaries.
- Migration, restart, idempotency, authentication, pagination, and sensitive-field regression tests.

Not implemented in this phase:

- Background worker.
- `run_forever()`.
- Queue claiming or lease renewal.
- Cloud deployment.
- Redis or PostgreSQL.
- Orchestrator execution refactoring.

## Schema migration 4

`Store.CURRENT_SCHEMA_VERSION` is now `4`. Migration 4 creates:

- `task_queue`
- `idx_task_queue_claim`
- `idx_task_queue_lease`

The queue row includes operational fields for a later worker: queue status, attempt count, maximum attempts, availability time, lease metadata, and the last error. Lease identifiers and worker metadata are never returned by the task-history API.

Existing `task_states` rows are backfilled into `task_queue` during migration. States that indicate an interrupted or potentially side-effecting action are represented conservatively as `EXECUTION_UNKNOWN`; they are not automatically retried.

The existing migration validation remains fail-closed. If marker 4 exists while the queue table or required indexes are missing, opening the database raises `UnsupportedSchemaVersion` rather than silently repairing an inconsistent schema.

## Atomic queued creation

`Store.create_queued_task()` uses `BEGIN IMMEDIATE` and persists the following in one transaction:

1. `TaskState` with status `QUEUED`.
2. Initial `TASK_QUEUED` checkpoint.
3. Legacy task summary row.
4. `task_queue` row.
5. Optional idempotency record.

The API does not call `Orchestrator.run()` during `POST /api/v1/tasks`. It returns a durable `QUEUED` response after the transaction commits. The existing `201 Created` status is retained for compatibility; execution is intentionally deferred until the worker phase.

A repeated idempotency key with the same request fingerprint returns the same task ID and does not create another queue row. Reusing the key with a different goal returns `409 idempotency_conflict`.

## Task history API

### `GET /api/v1/tasks`

The endpoint requires the existing Bearer authentication and returns a typed response:

```json
{
  "items": [
    {
      "task_id": "...",
      "goal": "...",
      "status": "QUEUED",
      "queue_status": "QUEUED",
      "current_phase": "QUEUED",
      "current_step": "",
      "plan_version": 0,
      "verification_status": "NOT_STARTED",
      "attempt_count": 0,
      "created_at": "...",
      "updated_at": "...",
      "last_checkpoint_at": "...",
      "last_successful_checkpoint": "TASK_QUEUED",
      "available_at": "...",
      "last_error_code": "",
      "waiting_for_approval": false,
      "execution_unknown": false
    }
  ],
  "next_cursor": null,
  "api_version": "v1",
  "schema_version": 1,
  "request_id": "..."
}
```

Query parameters:

- `limit`: 1–100, default 20.
- `cursor`: opaque URL-safe keyset cursor.
- `status`: validated `TaskState` status.

Pagination is ordered by `created_at DESC, task_id DESC`. The cursor contains only the last ordering pair and does not contain credentials or task payloads.

The response does not expose raw `state_json`, plans, action inputs, tool outputs, authorization headers, secrets, lease IDs, worker IDs, or unrestricted task payloads.

Direct Orchestrator-created tasks remain visible through a safe fallback projection even if they do not have a queue row. This preserves the existing direct CLI/core execution path while queue-created tasks use the durable queue record.

## Operational boundary

A queued task may be inspected through the existing task detail endpoint. The existing resume endpoint rejects an unstarted `QUEUED` task with `409 task_queued`; it does not run the task inline. Safe resume and `EXECUTION_UNKNOWN` behavior remain unchanged for tasks that already have an execution plan.

The next phase can add a worker that claims queue rows transactionally, revalidates security before every side effect, and never retries `EXECUTION_UNKNOWN` without explicit resolution.
