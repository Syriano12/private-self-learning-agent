# Phase 22 — Worker/SQLite Concurrency and Task-Queue Reliability

## Scope

This phase fixes the demonstrated same-process SQLite writer race behind the intermittent Worker/lifespan failures. It does not change authorization, approvals, lease semantics, `EXECUTION_UNKNOWN`, external-target behavior, or the Vulnora frontend.

## Baseline evidence

The previously observed failure occurred in the Worker/lifespan test with:

- task state remaining `READY`;
- queue row remaining `LEASED`;
- worker still `running` with no fatal error;
- the worker thread stopped in `SecurityController -> Observability -> Store._lease_write() -> sqlite3.Connection.commit()`.

The historical measurement from the preceding phase was 2 failures in 20 isolated repetitions. A fresh Phase 22 pre-fix repetition of the exact test produced 0 failures in 20 runs, confirming that the race was intermittent rather than continuously reproducible.

The root cause was nevertheless reproduced deterministically at the storage seam: two `Store` instances opened separate SQLite connections to the same file, each with its own instance-local `RLock`. With `PRAGMA busy_timeout = 0`, concurrent observability writes and concurrent queue claims produced `sqlite3.OperationalError: database is locked` before the fix.

The Worker heartbeat and watchdog intentionally use separate connections, so per-`Store` locking did not coordinate those writers with the orchestrator/API connection.

## Fix

`Store` now obtains one process-wide re-entrant lock keyed by the canonical database path. The lock is used by every `Store` instance for:

- schema initialization;
- existing transaction-scoped writes;
- lease claim/renew/finish/recovery writes;
- observability and audit writes;
- connection close.

The audit also found direct writes in the legacy `SkillRegistry`; those writes now use the same shared lock in `src/private_agent/skills/registry.py`. Connections remain separate per `Store` and are still created with `check_same_thread=False` only because the existing API/worker lifecycle uses the connection through controlled locking. The fix does not merge connections and does not add an unbounded retry or arbitrary sleep.

SQLite's own `busy_timeout` remains relevant for independent processes. The process-wide lock only coordinates connections inside the current Python process; it is not presented as a distributed lock.

## Regression tests

`tests/test_phase22_sqlite_concurrency.py` adds five deterministic tests:

1. Concurrent observability writes from two same-path `Store` instances succeed with `busy_timeout=0` and preserve all 120 unique events.
2. An injected SQLite trigger failure rolls back an observability transaction.
3. Concurrent queue claims have exactly one winner and leave the queue in `LEASED` state.
4. Concurrent direct `SkillRegistry` writes from four separate `Store` connections succeed with `busy_timeout=0`.
5. A failure after task-state mutation but before queue finalization rolls back both state and queue changes together.

The existing Worker tests continue to cover startup/shutdown during work, approval transitions, lease fencing, `EXECUTION_UNKNOWN`, and no automatic retry of uncertain effects.

## Validation

- Exact flaky lifespan test after the fix: 50/50 passed, 0 failures.
- Phase 22 focused suite: 5 passed; repeated 20/20 with 0 failures.
- Worker/API/lease regression suite after the audit fix: 59 passed.
- Security and Bug Bounty regression suite: 188 passed.
- Full backend suite: 567 passed in 32.63s.
- `python3 -m compileall -q src tests`: passed.
- `git diff --check`: passed.

## Remaining limits

This establishes same-process serialization for the worker-critical transaction paths and the audited direct `SkillRegistry` writes. It does not provide a distributed lock across independent processes or hosts, and it does not make SQLite a multi-worker production database. Some legacy/read-only Store methods still access the connection through helper paths without an explicit lexical lock; they were not part of the Worker failure path and remain a separate storage-wide hardening item. External side effects remain subject to the existing lease fencing and conservative `EXECUTION_UNKNOWN` handling.
