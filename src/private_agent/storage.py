from __future__ import annotations

import json
import hashlib
import logging
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from private_agent.core.knowledge import (
    KnowledgeAttestation,
    KnowledgeAttestor,
    KnowledgeClaim,
    KnowledgeConflict,
    KnowledgeEvidence,
    KnowledgeSource,
    KnowledgeValidationError,
    MAX_EXCERPT_LENGTH,
    canonicalize_source_uri,
)
from private_agent.bugbounty.models import safe_json
from private_agent.core.experience import sanitize_value
from private_agent.core.planning_provenance import planning_mode_from_plan, plan_version_from_state
from private_agent.core.task_state import MAX_TASK_GOAL_LENGTH

LOGGER = logging.getLogger(__name__)

_DATABASE_LOCKS_GUARD = threading.Lock()
_DATABASE_LOCKS: dict[str, threading.RLock] = {}


def _database_lock(path: Path) -> threading.RLock:
    """Return one process-wide lock for every SQLite database path.

    ``Store`` instances used by the worker heartbeat, API lifecycle, and
    recovery code may own separate SQLite connections to the same file. SQLite
    still serializes their writers, so a per-instance lock is insufficient and
    can produce a ``database is locked`` wait while another in-process writer
    is active. The process-wide lock coordinates those connections without
    changing the database's transaction or lease semantics. SQLite's own
    busy timeout remains the boundary for independent processes.
    """
    key = ":memory:" if str(path) == ":memory:" else str(path.resolve())
    with _DATABASE_LOCKS_GUARD:
        lock = _DATABASE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _DATABASE_LOCKS[key] = lock
        return lock


class StoreClosingError(RuntimeError):
    """Raised when a new Store operation starts after shutdown has begun."""


class StoreClosedError(StoreClosingError):
    """Raised when a Store operation is attempted after its connection closed."""


class StoreTransactionActiveError(RuntimeError):
    """Raised when closing would implicitly roll back an open transaction."""


class _GuardedSQLiteCursor(sqlite3.Cursor):
    def execute(self, sql: str, parameters: Any = ()):
        with self.connection._store_transaction_operation():
            return super().execute(sql, parameters)

    def executemany(self, sql: str, parameters: Any):
        with self.connection._store_transaction_operation():
            return super().executemany(sql, parameters)

    def executescript(self, sql_script: str):
        with self.connection._store_transaction_operation(reject_if_active=True):
            script = self.connection._store_prepare_executescript(sql_script)
            return super().executescript(script)

    def fetchone(self):
        with self.connection._store_db_operation():
            return super().fetchone()

    def fetchmany(self, size: int | None = None):
        with self.connection._store_db_operation():
            return super().fetchmany() if size is None else super().fetchmany(size)

    def fetchall(self):
        with self.connection._store_db_operation():
            return super().fetchall()

    def __next__(self):
        with self.connection._store_db_operation():
            return super().__next__()


class _GuardedSQLiteConnection(sqlite3.Connection):
    def _store_db_operation(self):
        operation = getattr(self, "_store_access_guard", None)
        return operation() if operation is not None else nullcontext()

    @staticmethod
    def _store_first_sql_words(statement: str, *, limit: int = 3) -> tuple[str, ...]:
        index = 0
        words: list[str] = []
        while index < len(statement) and len(words) < limit:
            if statement[index].isspace() or statement[index] == "\ufeff":
                index += 1
                continue
            if statement.startswith("--", index):
                newline = statement.find("\n", index + 2)
                if newline < 0:
                    return ()
                index = newline + 1
                continue
            if statement.startswith("/*", index):
                end_comment = statement.find("*/", index + 2)
                if end_comment < 0:
                    return ()
                index = end_comment + 2
                continue
            start = index
            while index < len(statement) and (statement[index].isalnum() or statement[index] == "_"):
                index += 1
            if start == index:
                if statement[index] in ".=(),":
                    index += 1
                    continue
                break
            words.append(statement[start:index].upper())
        return tuple(words)

    @classmethod
    def _store_prepare_executescript(cls, sql_script: str) -> str:
        """Wrap transactional scripts and preserve standalone autocommit statements.

        Scripts mixing PRAGMA/VACUUM with transactional statements are refused
        before execution; these commands cannot participate in the same atomic
        SQLite transaction.
        """
        if not isinstance(sql_script, str):
            return sql_script
        heads: list[tuple[str, ...]] = []
        statement_buffer: list[str] = []
        for character in sql_script:
            statement_buffer.append(character)
            if character == ";":
                statement = "".join(statement_buffer)
                if sqlite3.complete_statement(statement):
                    words = cls._store_first_sql_words(statement)
                    if words:
                        heads.append(words)
                    statement_buffer.clear()
        if statement_buffer:
            words = cls._store_first_sql_words("".join(statement_buffer))
            if words:
                heads.append(words)

        transaction_controls = {"BEGIN", "COMMIT", "END", "ROLLBACK", "SAVEPOINT", "RELEASE"}
        controls = [words[0] for words in heads if words[0] in transaction_controls]
        autocommit_commands = {"PRAGMA", "VACUUM"}
        if any(words[0] in autocommit_commands for words in heads):
            if all(words[0] in autocommit_commands for words in heads):
                return sql_script
            raise sqlite3.ProgrammingError("executescript_mixed_autocommit_and_transactional_unsupported")

        def ends_transaction(words: tuple[str, ...]) -> bool:
            return words[0] in {"COMMIT", "END"} or (
                words[0] == "ROLLBACK" and "TO" not in words[1:3]
            )

        if controls and heads[0][0] == "SAVEPOINT":
            if any(ends_transaction(words) for words in heads[1:-1]):
                raise sqlite3.ProgrammingError("executescript_transaction_boundary_must_be_final")
            prepared = f"BEGIN IMMEDIATE;\n{sql_script}"
            return prepared if heads and ends_transaction(heads[-1]) else f"{prepared}\nCOMMIT;"
        if controls and heads[0][0] != "BEGIN":
            raise sqlite3.ProgrammingError("executescript_transaction_control_must_start_with_begin")
        if controls and any(ends_transaction(words) for words in heads[1:-1]):
            raise sqlite3.ProgrammingError("executescript_transaction_boundary_must_be_final")
        if controls:
            return sql_script
        return f"BEGIN IMMEDIATE;\n{sql_script}\nCOMMIT;"

    @contextmanager
    def _store_transaction_operation(self, *, reject_if_active: bool = False):
        """Keep the connection transaction state and the shared lock consistent."""
        with self._store_db_operation():
            was_in_transaction = self.in_transaction
            if reject_if_active and was_in_transaction:
                raise sqlite3.ProgrammingError("executescript_not_allowed_in_transaction")
            try:
                yield
            except BaseException as operation_error:
                # An implicit transaction created by this call belongs to this
                # failed call; roll it back before permitting another thread in.
                if not was_in_transaction and self.in_transaction:
                    try:
                        sqlite3.Connection.rollback(self)
                    except BaseException as rollback_error:
                        raise operation_error from rollback_error
                raise
            finally:
                self._store_sync_transaction_lock()

    def cursor(self, factory=None):
        if factory is not None and factory is not _GuardedSQLiteCursor:
            raise sqlite3.ProgrammingError("custom_cursor_factory_not_supported")
        with self._store_db_operation():
            return super().cursor(_GuardedSQLiteCursor)

    def _store_sync_transaction_lock(self) -> None:
        """Match transaction ownership and the process-wide RLock to SQLite."""
        in_transaction = self.in_transaction
        lock = getattr(self, "_store_transaction_lock", None)
        if lock is None:
            return
        owner = getattr(self, "_store_transaction_owner", None)
        thread_id = threading.get_ident()
        if in_transaction:
            if owner == thread_id:
                return
            if owner is not None:
                raise RuntimeError("sqlite_transaction_thread_changed")
            lock.acquire()
            self._store_transaction_owner = thread_id
            return
        if owner == thread_id:
            self._store_release_transaction_lock()
        elif owner is not None:
            raise RuntimeError("sqlite_transaction_owner_mismatch")

    def _store_release_transaction_lock(self) -> None:
        owner = getattr(self, "_store_transaction_owner", None)
        if owner != threading.get_ident():
            return
        self._store_transaction_owner = None
        lock = getattr(self, "_store_transaction_lock", None)
        if lock is not None:
            lock.release()

    def execute(self, sql: str, parameters: Any = ()):
        cursor = self.cursor()
        try:
            return cursor.execute(sql, parameters)
        except BaseException:
            cursor.close()
            raise

    def executemany(self, sql: str, parameters: Any):
        cursor = self.cursor()
        try:
            return cursor.executemany(sql, parameters)
        except BaseException:
            cursor.close()
            raise

    def executescript(self, sql_script: str):
        cursor = self.cursor()
        try:
            return cursor.executescript(sql_script)
        except BaseException:
            cursor.close()
            raise

    def commit(self) -> None:
        with self._store_transaction_operation():
            super().commit()

    def rollback(self) -> None:
        with self._store_transaction_operation():
            super().rollback()

    def __enter__(self):
        with self._store_db_operation():
            return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        with self._store_transaction_operation():
            return super().__exit__(exc_type, exc_value, traceback)

    def close(self) -> None:
        handler = getattr(self, "_store_close_handler", None)
        if handler is None:
            super().close()
        else:
            handler()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_queue_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise QueueLeaseLost("invalid_queue_lease_timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass
class KnowledgeItem:
    id: str
    domain: str
    concept: str
    knowledge_type: str
    content: str
    source: str
    evidence: str
    confidence: float = 0.5
    verification_status: str = "unverified"
    source_reliability: float = 0.5
    version: int = 1
    usage_count: int = 0
    success_rate: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    last_verified_at: str = ""
    related_skills: str = "[]"
    contradictions: str = "[]"
    dependencies: str = "[]"


CURRENT_SCHEMA_VERSION = 18
MAX_TASK_MESSAGE_LENGTH = MAX_TASK_GOAL_LENGTH
MAX_TASK_CONVERSATION_MESSAGES = 32
MAX_TASK_CONVERSATION_CHARS = 16_000

_MIGRATION_TABLE_COLUMNS: dict[int, dict[str, dict[str, str]]] = {
    2: {
        "owner_permission_rules": {
            "rule_id": "TEXT", "policy_version": "INTEGER", "actor": "TEXT", "effect": "TEXT",
            "scope_type": "TEXT", "scope_key": "TEXT", "conditions_json": "TEXT", "expires_at": "TEXT",
            "reason": "TEXT", "status": "TEXT", "created_at": "TEXT", "disabled_at": "TEXT",
        },
        "owner_policy_audit": {
            "event_id": "TEXT", "rule_id": "TEXT", "policy_version": "INTEGER", "event_type": "TEXT",
            "actor": "TEXT", "previous_json": "TEXT", "new_json": "TEXT", "reason": "TEXT", "created_at": "TEXT",
        },
    },
    3: {
        "owner_policy_state": {"singleton": "INTEGER", "current_version": "INTEGER", "updated_at": "TEXT"},
        "owner_permission_rule_history": {
            "history_id": "TEXT", "rule_id": "TEXT", "policy_version": "INTEGER", "event_type": "TEXT",
            "snapshot_json": "TEXT", "actor": "TEXT", "reason": "TEXT", "created_at": "TEXT",
        },
    },
    4: {
        "task_queue": {
            "task_id": "TEXT", "queue_status": "TEXT", "attempt_count": "INTEGER", "max_attempts": "INTEGER",
            "available_at": "TEXT", "lease_id": "TEXT", "worker_id": "TEXT", "lease_acquired_at": "TEXT",
            "lease_expires_at": "TEXT", "last_started_at": "TEXT", "last_finished_at": "TEXT",
            "last_error_code": "TEXT", "last_error_message": "TEXT", "created_at": "TEXT", "updated_at": "TEXT",
        },
    },
    5: {
        "knowledge_sources": {
            "source_id": "TEXT", "canonical_uri": "TEXT", "source_type": "TEXT", "source_class": "TEXT",
            "retrieved_at": "TEXT", "published_at": "TEXT", "content_sha256": "TEXT", "revision": "TEXT",
            "status": "TEXT", "metadata_json": "TEXT", "created_at": "TEXT",
        },
        "knowledge_claims": {
            "claim_id": "TEXT", "claim_fingerprint": "TEXT", "domain": "TEXT", "subject": "TEXT",
            "predicate": "TEXT", "object_text": "TEXT", "knowledge_type": "TEXT", "verification_status": "TEXT",
            "status": "TEXT", "confidence": "REAL", "observed_at": "TEXT", "last_verified_at": "TEXT",
            "valid_from": "TEXT", "valid_until": "TEXT", "version": "INTEGER", "metadata_json": "TEXT",
            "created_at": "TEXT", "updated_at": "TEXT",
        },
        "knowledge_evidence": {
            "evidence_id": "TEXT", "evidence_fingerprint": "TEXT", "claim_id": "TEXT", "source_id": "TEXT",
            "task_id": "TEXT", "step_id": "TEXT", "locator": "TEXT", "excerpt": "TEXT", "excerpt_sha256": "TEXT",
            "verification_status": "TEXT", "observed_at": "TEXT", "metadata_json": "TEXT", "created_at": "TEXT",
        },
        "knowledge_conflicts": {
            "conflict_id": "TEXT", "conflict_fingerprint": "TEXT", "claim_id": "TEXT",
            "conflicting_claim_id": "TEXT", "conflict_type": "TEXT", "status": "TEXT", "detected_at": "TEXT",
            "evidence_ids_json": "TEXT", "resolution_note": "TEXT", "metadata_json": "TEXT", "created_at": "TEXT",
        },
    },
    6: {
        "auth_users": {
            "user_id": "TEXT", "email": "TEXT", "role": "TEXT", "password_hash": "TEXT",
            "status": "TEXT", "failed_login_count": "INTEGER", "locked_until": "TEXT",
            "last_login_at": "TEXT", "created_at": "TEXT", "updated_at": "TEXT",
        },
        "auth_sessions": {
            "session_hash": "TEXT", "user_id": "TEXT", "csrf_hash": "TEXT", "created_at": "TEXT",
            "expires_at": "TEXT", "last_seen_at": "TEXT", "revoked_at": "TEXT",
        },
        "auth_events": {
            "event_id": "TEXT", "event_type": "TEXT", "user_id": "TEXT", "email": "TEXT",
            "created_at": "TEXT", "remote_key": "TEXT", "metadata_json": "TEXT",
        },
    },
    7: {
        "task_owners": {
            "task_id": "TEXT", "user_id": "TEXT", "created_at": "TEXT",
        },
    },
    8: {
        "bugbounty_program_analyses": {
            "analysis_id": "TEXT", "owner_user_id": "TEXT", "analysis_json": "TEXT", "created_at": "TEXT",
        },
        "bugbounty_research_stages": {
            "stage_id": "TEXT", "analysis_id": "TEXT", "owner_user_id": "TEXT", "stage_type": "TEXT",
            "stage_status": "TEXT", "budget_json": "TEXT", "partial_result_json": "TEXT", "result_json": "TEXT",
            "error_code": "TEXT", "started_at": "TEXT", "finished_at": "TEXT", "created_at": "TEXT", "updated_at": "TEXT",
        },
    },
    9: {
        "bugbounty_finding_reviews": {
            "review_id": "TEXT", "task_id": "TEXT", "finding_id": "TEXT", "owner_user_id": "TEXT",
            "decision": "TEXT", "duplicate_of": "TEXT", "note": "TEXT", "evidence_sha256": "TEXT",
            "evidence_validated": "INTEGER", "actor_user_id": "TEXT", "created_at": "TEXT",
        },
    },
    10: {
        "bugbounty_research_stages": {
            "attempt_count": "INTEGER", "max_attempts": "INTEGER", "lease_id": "TEXT", "worker_id": "TEXT",
            "lease_acquired_at": "TEXT", "lease_expires_at": "TEXT", "last_error_code": "TEXT",
            "last_error_message": "TEXT",
        },
    },
    11: {
        "bugbounty_research_stages": {
            "input_json": "TEXT",
        },
    },
    12: {
        "remote_workers": {
            "worker_id": "TEXT", "credential_digest": "TEXT", "credential_version": "INTEGER",
            "previous_credential_digest": "TEXT", "previous_credential_version": "INTEGER",
            "previous_credential_expires_at": "TEXT", "status": "TEXT", "runtime_identity": "TEXT",
            "image_digest": "TEXT", "manifest_digest": "TEXT", "acceptance_status": "TEXT",
            "created_at": "TEXT", "updated_at": "TEXT", "last_seen_at": "TEXT",
        },
        "remote_worker_jobs": {
            "job_id": "TEXT", "task_id": "TEXT", "worker_id": "TEXT", "idempotency_key": "TEXT",
            "envelope_json": "TEXT", "lease_id": "TEXT", "status": "TEXT", "lease_expires_at": "TEXT",
            "heartbeat_at": "TEXT", "attempt_count": "INTEGER", "result_json": "TEXT",
            "result_digest": "TEXT", "last_error_code": "TEXT", "created_at": "TEXT", "updated_at": "TEXT",
        },
        "remote_worker_nonces": {
            "worker_id": "TEXT", "nonce": "TEXT", "seen_at": "TEXT", "request_digest": "TEXT",
        },
    },
    13: {
        "knowledge_attestations": {
            "attestation_id": "TEXT", "claim_id": "TEXT", "source_id": "TEXT",
            "claim_version": "INTEGER", "key_id": "TEXT", "signature": "TEXT",
            "attestation_json": "TEXT", "created_at": "TEXT",
        },
    },
    14: {
        "knowledge_attestations": {
            "attestation_id": "TEXT", "claim_id": "TEXT", "source_id": "TEXT",
            "claim_version": "INTEGER", "key_id": "TEXT", "signature": "TEXT",
            "attestation_json": "TEXT", "created_at": "TEXT",
        },
        "task_messages": {
            "message_id": "TEXT", "task_id": "TEXT", "sequence": "INTEGER", "role": "TEXT",
            "message_kind": "TEXT", "content": "TEXT", "status": "TEXT", "idempotency_key": "TEXT",
            "request_fingerprint": "TEXT", "created_at": "TEXT", "consumed_at": "TEXT",
            "consumed_checkpoint": "TEXT",
        },
    },
    15: {
        "task_result_events": {
            "event_id": "TEXT", "task_id": "TEXT", "owner_user_id": "TEXT", "sequence": "INTEGER", "role": "TEXT",
            "message_kind": "TEXT", "task_status": "TEXT", "verification_status": "TEXT",
            "result_available": "INTEGER", "verified_action_count": "INTEGER", "failed_action_count": "INTEGER",
            "blocked_action_count": "INTEGER", "pending_action_count": "INTEGER",
            "approval_pending_count": "INTEGER", "approval_approved_count": "INTEGER",
            "approval_denied_count": "INTEGER", "approval_expired_count": "INTEGER",
            "approval_cancelled_count": "INTEGER", "result_digest": "TEXT",
            "transition_fingerprint": "TEXT", "created_at": "TEXT",
        },
    },
    16: {
        "task_assistant_replies": {
            "reply_id": "TEXT", "reply_key": "TEXT", "event_id": "TEXT", "task_id": "TEXT",
            "owner_user_id": "TEXT", "source_sequence": "INTEGER", "source_digest": "TEXT",
            "role": "TEXT", "message_kind": "TEXT", "reply_code": "TEXT", "content": "TEXT",
            "generation_mode": "TEXT", "created_at": "TEXT",
        },
        "task_assistant_reply_claims": {
            "event_id": "TEXT", "task_id": "TEXT", "owner_user_id": "TEXT", "lease_owner": "TEXT",
            "lease_expires_at": "TEXT", "attempt_count": "INTEGER", "created_at": "TEXT", "updated_at": "TEXT",
        },
    },
    17: {
        "task_conversation_sequences": {
            "task_id": "TEXT", "last_sequence": "INTEGER",
        },
    },
    18: {
        "task_result_events": {
            "planning_mode": "TEXT", "plan_version": "INTEGER",
        },
    },
}

_MIGRATION_INDEX_COLUMNS: dict[str, tuple[str, ...]] = {
    "idx_owner_rules_active_scope": ("status", "scope_type", "scope_key"),
    "idx_owner_rules_expiry": ("status", "expires_at"),
    "idx_task_queue_claim": ("queue_status", "available_at", "created_at", "task_id"),
    "idx_task_queue_lease": ("queue_status", "lease_expires_at"),
    "idx_knowledge_sources_identity": ("canonical_uri", "revision", "content_sha256"),
    "idx_knowledge_claims_fingerprint": ("claim_fingerprint",),
    "idx_knowledge_claims_freshness": ("status", "verification_status", "valid_until", "last_verified_at"),
    "idx_knowledge_evidence_claim": ("claim_id", "source_id", "observed_at"),
    "idx_knowledge_conflicts_claim": ("claim_id", "conflicting_claim_id", "status"),
    "idx_auth_users_email": ("email",),
    "idx_auth_sessions_user": ("user_id",),
    "idx_auth_sessions_expiry": ("expires_at", "revoked_at"),
    "idx_auth_events_created": ("created_at",),
    "idx_task_owners_user": ("user_id", "created_at", "task_id"),
    "idx_program_analysis_owner": ("owner_user_id", "created_at"),
    "idx_research_stages_analysis": ("analysis_id", "created_at", "stage_id"),
    "idx_research_stages_owner": ("owner_user_id", "updated_at", "stage_id"),
    "idx_finding_reviews_task": ("task_id", "finding_id", "created_at", "review_id"),
    "idx_finding_reviews_owner": ("owner_user_id", "created_at", "review_id"),
    "idx_research_stages_claim": ("stage_status", "lease_expires_at", "created_at", "stage_id"),
    "idx_remote_workers_status": ("status", "acceptance_status", "updated_at", "worker_id"),
    "idx_remote_worker_jobs_claim": ("status", "worker_id", "created_at", "job_id"),
    "idx_remote_worker_jobs_lease": ("status", "lease_expires_at", "job_id"),
    "idx_remote_worker_jobs_idempotency": ("worker_id", "idempotency_key"),
    "idx_knowledge_attestations_claim": ("claim_id", "claim_version", "created_at", "attestation_id"),
    "idx_task_messages_task_sequence": ("task_id", "sequence"),
    "idx_task_messages_pending": ("task_id", "status", "sequence"),
    "idx_task_result_events_task_sequence": ("task_id", "sequence"),
    "idx_task_assistant_replies_task_sequence": ("task_id", "source_sequence"),
    "idx_task_assistant_reply_claims_lease": ("lease_expires_at", "event_id"),
}


class UnsupportedSchemaVersion(RuntimeError):
    """Raised when a database is newer than this application understands."""


class QueueLeaseLost(RuntimeError):
    """Raised when a worker tries to write after losing its queue lease."""


class Store:
    def __init__(
        self,
        path: str | Path = "data/agent.sqlite3",
        *,
        knowledge_attestation_key: str | bytes | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db_lock = _database_lock(self.path)
        self._lifecycle = threading.Condition(self._db_lock)
        self._closing = False
        self._worker_shutdown_requested = False
        self._closed = False
        self._background_workers: set[Any] = set()
        self._worker_thread_ids: dict[int, int] = {}
        self.db = sqlite3.connect(self.path, check_same_thread=False, factory=_GuardedSQLiteConnection)
        self.db._store_access_guard = self._db_operation
        self.db._store_close_handler = self.close
        self.db._store_transaction_lock = self._db_lock
        self.db._store_transaction_owner = None
        self.db.row_factory = sqlite3.Row
        self._lease_local = threading.local()
        self._approval_gate: Any | None = None
        if knowledge_attestation_key is None:
            knowledge_attestation_key = os.environ.get("AGENT_KNOWLEDGE_ATTESTATION_KEY")
        self._knowledge_attestor = KnowledgeAttestor(knowledge_attestation_key) if knowledge_attestation_key else None
        with self._db_lock:
            self.init_schema()

    @property
    def closing(self) -> bool:
        with self._lifecycle:
            return self._closing or self._worker_shutdown_requested

    @property
    def closed(self) -> bool:
        with self._lifecycle:
            return self._closed

    def request_worker_shutdown(self) -> None:
        """Fence new worker claims without closing the Store connection.

        This fence is terminal for the Store's worker lifecycle; create a new
        Store before starting a new worker after application shutdown.
        """
        with self._lifecycle:
            if self._closed:
                return
            self._worker_shutdown_requested = True
            self._lifecycle.notify_all()

    @contextmanager
    def _db_operation(self):
        """Serialize a SQLite call with shutdown and reject post-close callers.

        During draining, only an already-registered worker thread may finish its
        in-flight claim. Other callers and new worker registrations are fenced.
        """
        with self._lifecycle:
            if self._closed:
                raise StoreClosedError("store_closed")
            if self._closing and threading.get_ident() not in self._worker_thread_ids:
                raise StoreClosingError("store_closing")
            yield

    def register_background_worker(self, worker: Any) -> None:
        with self._lifecycle:
            if self._closing or self._worker_shutdown_requested or self._closed:
                raise StoreClosingError("store_closing")
            self._background_workers.add(worker)

    def enter_background_worker(self, worker: Any) -> None:
        with self._lifecycle:
            if worker not in self._background_workers:
                raise StoreClosingError("worker_not_registered_with_store")
            thread_id = threading.get_ident()
            self._worker_thread_ids[thread_id] = self._worker_thread_ids.get(thread_id, 0) + 1

    def leave_background_worker(self, worker: Any) -> None:
        with self._lifecycle:
            self._leave_worker_thread_locked(threading.get_ident())
            self._background_workers.discard(worker)
            self._lifecycle.notify_all()

    def _leave_worker_thread_locked(self, thread_id: int) -> None:
        depth = self._worker_thread_ids.get(thread_id, 0)
        if depth <= 1:
            self._worker_thread_ids.pop(thread_id, None)
        else:
            self._worker_thread_ids[thread_id] = depth - 1

    @contextmanager
    def background_worker_operation(self, worker: Any):
        """Track one run_once call, including direct calls outside start()/run_forever()."""
        thread_id = threading.get_ident()
        with self._lifecycle:
            registered = worker in self._background_workers
            if self._closed or (
                (self._closing or self._worker_shutdown_requested)
                and (not registered or thread_id not in self._worker_thread_ids)
            ):
                raise StoreClosingError("store_closing")
            temporary_registration = not registered
            if temporary_registration:
                self._background_workers.add(worker)
            self._worker_thread_ids[thread_id] = self._worker_thread_ids.get(thread_id, 0) + 1
        try:
            yield
        finally:
            with self._lifecycle:
                self._leave_worker_thread_locked(thread_id)
                if temporary_registration:
                    self._background_workers.discard(worker)
                self._lifecycle.notify_all()

    def unregister_background_worker(self, worker: Any) -> None:
        with self._lifecycle:
            self._background_workers.discard(worker)
            self._lifecycle.notify_all()

    def assert_worker_claim_allowed(self) -> None:
        """Fence acquisition of a new task/stage lease once shutdown starts."""
        with self._lifecycle:
            if self._closed:
                raise StoreClosedError("store_closed")
            if self._closing or self._worker_shutdown_requested:
                raise StoreClosingError("store_closing_no_new_claims")

    def _bind_approval_gate(self, gate: Any, *, owner: Any | None = None) -> None:
        """Bind the in-process approval writer through a SecurityController only."""
        if gate is None:
            raise ValueError("approval_gate_required")
        try:
            from private_agent.security import SecurityController
        except ImportError as exc:
            raise ValueError("approval_gate_owner_unavailable") from exc
        if not isinstance(owner, SecurityController) or getattr(owner, "store", None) is not self:
            raise ValueError("approval_gate_owner_required")
        if getattr(owner, "approval_gate", None) is not gate:
            raise ValueError("approval_gate_owner_mismatch")
        self._approval_gate = gate

    @contextmanager
    def lease_context(self, task_id: str, lease_id: str, worker_id: str):
        """Bind a claim to this thread while Orchestrator persists its checkpoints."""
        previous = getattr(self._lease_local, "claim", None)
        self._lease_local.claim = (str(task_id), str(lease_id), str(worker_id))
        try:
            yield
        finally:
            if previous is None:
                try:
                    del self._lease_local.claim
                except AttributeError:
                    pass
            else:
                self._lease_local.claim = previous

    def _active_lease(self) -> tuple[str, str, str] | None:
        claim = getattr(self._lease_local, "claim", None)
        return claim if isinstance(claim, tuple) and len(claim) == 3 else None

    def has_active_lease_context(self, task_id: str) -> bool:
        """Return whether this thread is executing under the task's lease context."""
        claim = self._active_lease()
        return bool(claim is not None and claim[0] == str(task_id))

    def _assert_active_lease(self, task_id: str) -> None:
        """Fence all state/result writes for queue-managed tasks."""
        row = self.db.execute(
            "SELECT queue_status, lease_id, worker_id, lease_expires_at FROM task_queue WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return
        claim = self._active_lease()
        if claim is None:
            raise QueueLeaseLost(f"queue_lease_required:{task_id}")
        claimed_task, lease_id, worker_id = claim
        if claimed_task != task_id or row["queue_status"] != "LEASED" or row["lease_id"] != lease_id or row["worker_id"] != worker_id:
            raise QueueLeaseLost(f"queue_lease_lost:{task_id}")
        if not row["lease_expires_at"] or _parse_queue_time(row["lease_expires_at"]) <= datetime.now(timezone.utc):
            raise QueueLeaseLost(f"queue_lease_expired:{task_id}")

    def assert_active_lease(self, task_id: str) -> None:
        """Check the current lease immediately before an external side effect."""
        with self._db_lock:
            self._assert_active_lease(str(task_id))

    @contextmanager
    def _lease_write(self, task_id: str | None = None):
        """Run one sensitive write under a transaction-scoped lease check.

        ``BEGIN IMMEDIATE`` makes the lease check and the following write one
        serialization point.  A stale worker therefore cannot check a lease,
        lose it to recovery, and then commit a checkpoint or event afterward.
        """
        with self._db_lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                if task_id:
                    self._assert_active_lease(str(task_id))
                yield
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise

    @staticmethod
    def _required_objects(version: int) -> tuple[tuple[str, str], ...]:
        if version == 2:
            return (
                ("table", "owner_permission_rules"),
                ("table", "owner_policy_audit"),
                ("index", "idx_owner_rules_active_scope"),
                ("index", "idx_owner_rules_expiry"),
            )
        if version == 3:
            return (
                ("table", "owner_policy_state"),
                ("table", "owner_permission_rule_history"),
            )
        if version == 4:
            return (
                ("table", "task_queue"),
                ("index", "idx_task_queue_claim"),
                ("index", "idx_task_queue_lease"),
            )
        if version == 5:
            return (
                ("table", "knowledge_sources"),
                ("table", "knowledge_claims"),
                ("table", "knowledge_evidence"),
                ("table", "knowledge_conflicts"),
                ("index", "idx_knowledge_sources_identity"),
                ("index", "idx_knowledge_claims_fingerprint"),
                ("index", "idx_knowledge_claims_freshness"),
                ("index", "idx_knowledge_evidence_claim"),
                ("index", "idx_knowledge_conflicts_claim"),
            )
        if version == 6:
            return (
                ("table", "auth_users"),
                ("table", "auth_sessions"),
                ("table", "auth_events"),
                ("index", "idx_auth_users_email"),
                ("index", "idx_auth_sessions_user"),
                ("index", "idx_auth_sessions_expiry"),
                ("index", "idx_auth_events_created"),
            )
        if version == 7:
            return (
                ("table", "task_owners"),
                ("index", "idx_task_owners_user"),
            )
        if version == 8:
            return (
                ("table", "bugbounty_program_analyses"),
                ("table", "bugbounty_research_stages"),
                ("index", "idx_program_analysis_owner"),
                ("index", "idx_research_stages_analysis"),
                ("index", "idx_research_stages_owner"),
            )
        if version == 9:
            return (
                ("table", "bugbounty_finding_reviews"),
                ("index", "idx_finding_reviews_task"),
                ("index", "idx_finding_reviews_owner"),
            )
        if version == 10:
            return (
                ("table", "bugbounty_research_stages"),
                ("index", "idx_research_stages_claim"),
            )
        if version == 11:
            return (("table", "bugbounty_research_stages"),)
        if version == 12:
            return (
                ("table", "remote_workers"),
                ("table", "remote_worker_jobs"),
                ("table", "remote_worker_nonces"),
                ("index", "idx_remote_workers_status"),
                ("index", "idx_remote_worker_jobs_claim"),
                ("index", "idx_remote_worker_jobs_lease"),
                ("index", "idx_remote_worker_jobs_idempotency"),
            )
        if version == 13:
            return (
                ("table", "knowledge_attestations"),
                ("index", "idx_knowledge_attestations_claim"),
            )
        if version == 14:
            return (
                ("table", "knowledge_attestations"),
                ("index", "idx_knowledge_attestations_claim"),
                ("table", "task_messages"),
                ("index", "idx_task_messages_task_sequence"),
                ("index", "idx_task_messages_pending"),
            )
        if version == 15:
            return (
                ("table", "task_result_events"),
                ("index", "idx_task_result_events_task_sequence"),
                ("trigger", "task_result_events_owner_matches_task"),
                ("trigger", "task_result_events_no_update"),
                ("trigger", "task_result_events_no_delete"),
            )
        if version == 16:
            return (
                ("table", "task_assistant_replies"),
                ("table", "task_assistant_reply_claims"),
                ("index", "idx_task_assistant_replies_task_sequence"),
                ("index", "idx_task_assistant_reply_claims_lease"),
                ("trigger", "task_assistant_replies_owner_matches_event"),
                ("trigger", "task_assistant_replies_no_update"),
                ("trigger", "task_assistant_replies_no_delete"),
                ("trigger", "task_assistant_reply_claims_owner_matches_event"),
            )
        if version == 17:
            return (
                ("table", "task_conversation_sequences"),
                ("trigger", "task_conversation_sequences_monotonic_guard"),
                ("trigger", "task_conversation_sequences_no_delete"),
                ("trigger", "task_messages_sequence_counter_guard"),
                ("trigger", "task_messages_sequence_cross_type_guard"),
                ("trigger", "task_messages_sequence_immutable"),
                ("trigger", "task_result_events_sequence_counter_guard"),
                ("trigger", "task_result_events_sequence_cross_type_guard"),
            )
        if version == 18:
            return (("table", "task_result_events"),)
        return ()

    def _assert_migration_objects(
        self, version: int, *, legacy_conversation_schema13: bool = False
    ) -> None:
        if version == 13 and not legacy_conversation_schema13:
            attestations = self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='knowledge_attestations'"
            ).fetchone()
            conversations = self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='task_messages'"
            ).fetchone()
            if attestations is None and conversations is not None:
                # Backend PR1 used schema marker 13 for task_messages. Accept
                # that legacy layout only after validating its complete table
                # and indexes; migration 14 adds the other v13 feature.
                self._assert_migration_objects(13, legacy_conversation_schema13=True)
                return
        if legacy_conversation_schema13:
            required_objects = (
                ("table", "task_messages"),
                ("index", "idx_task_messages_task_sequence"),
                ("index", "idx_task_messages_pending"),
            )
            schema_version_for_columns = 14
        else:
            required_objects = self._required_objects(version)
            schema_version_for_columns = version
        for object_type, name in required_objects:
            row = self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?",
                (object_type, name),
            ).fetchone()
            if row is None:
                raise UnsupportedSchemaVersion(f"inconsistent_schema_version:{version}:missing_{object_type}:{name}")
            if object_type == "table":
                columns = {
                    str(item["name"]): str(item["type"]).upper()
                    for item in self.db.execute(f"PRAGMA table_info({name})").fetchall()
                }
                expected = _MIGRATION_TABLE_COLUMNS.get(schema_version_for_columns, {}).get(name, {})
                for column, expected_type in expected.items():
                    if column not in columns:
                        raise UnsupportedSchemaVersion(
                            f"inconsistent_schema_version:{version}:missing_column:{name}.{column}"
                        )
                    if columns[column] != expected_type:
                        raise UnsupportedSchemaVersion(
                            f"inconsistent_schema_version:{version}:invalid_column_type:{name}.{column}"
                        )
                table_sql_row = self.db.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
                ).fetchone()
                table_sql = " ".join(str(table_sql_row["sql"] or "").upper().split()) if table_sql_row else ""
                required_fragments = {
                    "owner_policy_state": ("CHECK (SINGLETON = 1)",),
                    "task_queue": ("CHECK (QUEUE_STATUS IN", "FOREIGN KEY (TASK_ID)"),
                "knowledge_claims": ("CHECK (CONFIDENCE >= 0 AND CONFIDENCE <= 1)", "CHECK (VERSION >= 1)"),
                "knowledge_evidence": ("FOREIGN KEY (CLAIM_ID)", "FOREIGN KEY (SOURCE_ID)"),
                "knowledge_attestations": (
                    "CHECK (CLAIM_VERSION >= 1)",
                    "FOREIGN KEY (CLAIM_ID)",
                    "FOREIGN KEY (SOURCE_ID)",
                ),
                "knowledge_conflicts": ("CHECK (CLAIM_ID <> CONFLICTING_CLAIM_ID)", "FOREIGN KEY (CLAIM_ID)", "FOREIGN KEY (CONFLICTING_CLAIM_ID)"),
                "auth_users": ("CHECK (ROLE IN ('OWNER', 'OPERATOR', 'VIEWER'))", "CHECK (STATUS IN ('ACTIVE', 'DISABLED'))"),
                "auth_sessions": ("FOREIGN KEY (USER_ID)",),
                "task_owners": ("FOREIGN KEY (TASK_ID)", "FOREIGN KEY (USER_ID)"),
                "bugbounty_finding_reviews": ("CHECK (EVIDENCE_VALIDATED IN (0, 1))",),
                "remote_workers": ("CHECK (STATUS IN ('ACTIVE', 'ROTATED', 'REVOKED'))", "CHECK (ACCEPTANCE_STATUS IN ('UNVERIFIED', 'VERIFIED'))"),
                    "remote_worker_jobs": ("CHECK (STATUS IN ('QUEUED', 'LEASED', 'COMPLETED', 'FAILED', 'EXECUTION_UNKNOWN', 'BLOCKED'))",),
                    "remote_worker_nonces": ("PRIMARY KEY (WORKER_ID, NONCE)",),
                    "task_messages": (
                        "CHECK (ROLE = 'USER')",
                        "CHECK (MESSAGE_KIND IN ('INITIAL', 'FOLLOW_UP'))",
                        "CHECK (STATUS IN ('PENDING', 'CONSUMED'))",
                        "UNIQUE (TASK_ID, SEQUENCE)",
                        "FOREIGN KEY (TASK_ID) REFERENCES TASK_STATES(TASK_ID)",
                    ),
                    "task_result_events": (
                        "CHECK (ROLE = 'SYSTEM')",
                        "CHECK (MESSAGE_KIND = 'RESULT')",
                        "CHECK (TASK_STATUS IN",
                        "UNIQUE (TASK_ID, SEQUENCE)",
                        "UNIQUE (TASK_ID, TRANSITION_FINGERPRINT)",
                        "FOREIGN KEY (TASK_ID) REFERENCES TASK_STATES(TASK_ID)",
                        "FOREIGN KEY (OWNER_USER_ID) REFERENCES AUTH_USERS(USER_ID)",
                    ),
                    "task_assistant_replies": (
                        "CHECK (ROLE = 'ASSISTANT')",
                        "CHECK (MESSAGE_KIND = 'RESULT_REPLY')",
                        "CHECK (REPLY_CODE IN",
                        "EVENT_ID TEXT NOT NULL UNIQUE",
                        "UNIQUE (TASK_ID, REPLY_KEY)",
                        "FOREIGN KEY (EVENT_ID) REFERENCES TASK_RESULT_EVENTS(EVENT_ID)",
                        "FOREIGN KEY (TASK_ID) REFERENCES TASK_STATES(TASK_ID)",
                        "FOREIGN KEY (OWNER_USER_ID) REFERENCES AUTH_USERS(USER_ID)",
                    ),
                    "task_assistant_reply_claims": (
                        "EVENT_ID TEXT PRIMARY KEY",
                        "FOREIGN KEY (EVENT_ID) REFERENCES TASK_RESULT_EVENTS(EVENT_ID)",
                        "FOREIGN KEY (TASK_ID) REFERENCES TASK_STATES(TASK_ID)",
                        "FOREIGN KEY (OWNER_USER_ID) REFERENCES AUTH_USERS(USER_ID)",
                    ),
                    "task_conversation_sequences": (
                        "LAST_SEQUENCE INTEGER NOT NULL CHECK (LAST_SEQUENCE >= 0)",
                        "FOREIGN KEY (TASK_ID) REFERENCES TASK_STATES(TASK_ID)",
                    ),
                }.get(name, ())
                if version >= 18 and name == "task_result_events":
                    required_fragments += (
                        "CHECK (PLANNING_MODE IN ('UNKNOWN', 'PROVIDER_VALIDATED', 'DETERMINISTIC_FALLBACK', 'DETERMINISTIC_ROUTE', 'DETERMINISTIC_REPLAN'))",
                        "CHECK (PLAN_VERSION >= 0)",
                    )
                for fragment in required_fragments:
                    if fragment not in table_sql:
                        raise UnsupportedSchemaVersion(
                            f"inconsistent_schema_version:{version}:missing_constraint:{name}"
                        )
            elif object_type == "index" and name in _MIGRATION_INDEX_COLUMNS:
                actual = tuple(
                    str(item["name"])
                    for item in self.db.execute(f"PRAGMA index_info({name})").fetchall()
                )
                if actual != _MIGRATION_INDEX_COLUMNS[name]:
                    raise UnsupportedSchemaVersion(
                        f"inconsistent_schema_version:{version}:invalid_index:{name}"
                    )
            elif object_type == "trigger" and name.startswith(("task_messages_sequence_", "task_result_events_", "task_assistant_replies_", "task_assistant_reply_claims_")):
                trigger_row = self.db.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?", (name,)
                ).fetchone()
                trigger_sql = " ".join(str(trigger_row["sql"] or "").upper().split()) if trigger_row else ""
                required_trigger_fragments = {
                    "task_result_events_owner_matches_task": (
                        "BEFORE INSERT ON TASK_RESULT_EVENTS",
                        "FROM TASK_OWNERS",
                        "NEW.TASK_ID",
                        "NEW.OWNER_USER_ID",
                        "TASK_RESULT_EVENTS_OWNER_MISMATCH",
                    ),
                    "task_result_events_no_update": (
                        "BEFORE UPDATE ON TASK_RESULT_EVENTS",
                        "TASK_RESULT_EVENTS_IMMUTABLE",
                    ),
                    "task_result_events_no_delete": (
                        "BEFORE DELETE ON TASK_RESULT_EVENTS",
                        "TASK_RESULT_EVENTS_IMMUTABLE",
                    ),
                    "task_assistant_replies_owner_matches_event": (
                        "BEFORE INSERT ON TASK_ASSISTANT_REPLIES",
                        "FROM TASK_RESULT_EVENTS",
                        "NEW.EVENT_ID",
                        "NEW.OWNER_USER_ID",
                        "TASK_ASSISTANT_REPLIES_OWNER_MISMATCH",
                    ),
                    "task_assistant_replies_no_update": (
                        "BEFORE UPDATE ON TASK_ASSISTANT_REPLIES",
                        "TASK_ASSISTANT_REPLIES_IMMUTABLE",
                    ),
                    "task_assistant_replies_no_delete": (
                        "BEFORE DELETE ON TASK_ASSISTANT_REPLIES",
                        "TASK_ASSISTANT_REPLIES_IMMUTABLE",
                    ),
                    "task_assistant_reply_claims_owner_matches_event": (
                        "BEFORE INSERT ON TASK_ASSISTANT_REPLY_CLAIMS",
                        "FROM TASK_RESULT_EVENTS",
                        "NEW.EVENT_ID",
                        "NEW.OWNER_USER_ID",
                        "TASK_ASSISTANT_REPLY_CLAIM_OWNER_MISMATCH",
                    ),
                    "task_messages_sequence_counter_guard": (
                        "BEFORE INSERT ON TASK_MESSAGES",
                        "TASK_CONVERSATION_SEQUENCES",
                        "LAST_SEQUENCE = NEW.SEQUENCE",
                        "TASK_MESSAGES_SEQUENCE_NOT_ALLOCATED",
                    ),
                    "task_messages_sequence_cross_type_guard": (
                        "BEFORE INSERT ON TASK_MESSAGES",
                        "FROM TASK_RESULT_EVENTS",
                        "NEW.TASK_ID",
                        "NEW.SEQUENCE",
                        "TASK_CONVERSATION_SEQUENCE_DUPLICATE",
                    ),
                    "task_messages_sequence_immutable": (
                        "BEFORE UPDATE OF SEQUENCE ON TASK_MESSAGES",
                        "TASK_MESSAGES_SEQUENCE_IMMUTABLE",
                    ),
                    "task_result_events_sequence_counter_guard": (
                        "BEFORE INSERT ON TASK_RESULT_EVENTS",
                        "TASK_CONVERSATION_SEQUENCES",
                        "LAST_SEQUENCE = NEW.SEQUENCE",
                        "TASK_RESULT_EVENTS_SEQUENCE_NOT_ALLOCATED",
                    ),
                    "task_result_events_sequence_cross_type_guard": (
                        "BEFORE INSERT ON TASK_RESULT_EVENTS",
                        "FROM TASK_MESSAGES",
                        "NEW.TASK_ID",
                        "NEW.SEQUENCE",
                        "TASK_CONVERSATION_SEQUENCE_DUPLICATE",
                    ),
                    "task_conversation_sequences_monotonic_guard": (
                        "BEFORE UPDATE OF LAST_SEQUENCE ON TASK_CONVERSATION_SEQUENCES",
                        "NEW.LAST_SEQUENCE != OLD.LAST_SEQUENCE + 1",
                        "TASK_CONVERSATION_SEQUENCE_NOT_MONOTONIC",
                    ),
                    "task_conversation_sequences_no_delete": (
                        "BEFORE DELETE ON TASK_CONVERSATION_SEQUENCES",
                        "TASK_CONVERSATION_SEQUENCE_IMMUTABLE",
                    ),
                }.get(name, ())
                if any(fragment not in trigger_sql for fragment in required_trigger_fragments):
                    raise UnsupportedSchemaVersion(
                        f"inconsistent_schema_version:{version}:invalid_trigger:{name}"
                    )
        if version == 3:
            state_rows = self.db.execute(
                "SELECT singleton, current_version FROM owner_policy_state"
            ).fetchall()
            if len(state_rows) != 1 or int(state_rows[0]["singleton"]) != 1:
                raise UnsupportedSchemaVersion("inconsistent_schema_version:3:owner_policy_state_singleton")
            maximum = self.db.execute(
                """SELECT MAX(policy_version) AS version FROM (
                       SELECT policy_version FROM security_policies
                       UNION ALL
                       SELECT policy_version FROM owner_permission_rules
                   )"""
            ).fetchone()
            if int(state_rows[0]["current_version"]) < int(maximum["version"] or 0):
                raise UnsupportedSchemaVersion("inconsistent_schema_version:3:owner_policy_version_regressed")

    def init_schema(self) -> None:
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA busy_timeout = 5000")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
          id TEXT PRIMARY KEY, goal TEXT NOT NULL, status TEXT NOT NULL,
          plan_json TEXT NOT NULL, result_json TEXT NOT NULL, attempts INTEGER NOT NULL,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS task_state_schema (
          schema_version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS task_states (
          task_id TEXT PRIMARY KEY, state_version INTEGER NOT NULL,
          status TEXT NOT NULL, current_phase TEXT NOT NULL, current_step TEXT NOT NULL,
          plan_version INTEGER NOT NULL, plan_fingerprint TEXT NOT NULL,
          state_fingerprint TEXT NOT NULL, state_json TEXT NOT NULL,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
          last_checkpoint_at TEXT NOT NULL, last_successful_checkpoint TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS task_checkpoints (
          checkpoint_id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
          sequence INTEGER NOT NULL, checkpoint_type TEXT NOT NULL,
          state_version INTEGER NOT NULL, state_fingerprint TEXT NOT NULL,
          state_json TEXT NOT NULL, created_at TEXT NOT NULL,
          UNIQUE(task_id, sequence)
        );
        CREATE TABLE IF NOT EXISTS task_action_records (
          action_key TEXT PRIMARY KEY, task_id TEXT NOT NULL, step_id TEXT NOT NULL,
          tool_name TEXT NOT NULL, status TEXT NOT NULL, action_json TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS episodes (
          id TEXT PRIMARY KEY, task_id TEXT NOT NULL, goal TEXT NOT NULL,
          outcome TEXT NOT NULL, lessons_json TEXT NOT NULL, observations_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS knowledge_items (
          id TEXT PRIMARY KEY, domain TEXT NOT NULL, concept TEXT NOT NULL,
          knowledge_type TEXT NOT NULL, content TEXT NOT NULL, source TEXT NOT NULL,
          evidence TEXT NOT NULL, confidence REAL NOT NULL, verification_status TEXT NOT NULL,
          source_reliability REAL NOT NULL, version INTEGER NOT NULL, usage_count INTEGER NOT NULL,
          success_rate REAL NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
          last_verified_at TEXT NOT NULL, related_skills TEXT NOT NULL,
          contradictions TEXT NOT NULL, dependencies TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS skills (
          name TEXT NOT NULL, version INTEGER NOT NULL, description TEXT NOT NULL,
          implementation TEXT NOT NULL, confidence REAL NOT NULL, success_rate REAL NOT NULL,
          enabled INTEGER NOT NULL, source_knowledge TEXT NOT NULL, tests_json TEXT NOT NULL,
          updated_at TEXT NOT NULL, PRIMARY KEY(name, version)
        );
        CREATE TABLE IF NOT EXISTS observations (
          id TEXT PRIMARY KEY, task_id TEXT NOT NULL, step_id TEXT NOT NULL,
          tool_name TEXT NOT NULL, execution_status TEXT NOT NULL,
          observation_json TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS verifications (
          id TEXT PRIMARY KEY, task_id TEXT NOT NULL, step_id TEXT NOT NULL,
          tool_name TEXT NOT NULL, status TEXT NOT NULL,
          verification_json TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS execution_events (
          id TEXT PRIMARY KEY, task_id TEXT NOT NULL, step_id TEXT,
          event_type TEXT NOT NULL, event_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS experiences (
          task_id TEXT PRIMARY KEY, goal TEXT NOT NULL,
          experience_json TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reflection_insights (
          insight_id TEXT PRIMARY KEY, pattern TEXT NOT NULL,
          confidence REAL NOT NULL, insight_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS learned_strategies (
          strategy_id TEXT PRIMARY KEY, status TEXT NOT NULL,
          confidence REAL NOT NULL, strategy_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS skill_candidates (
          skill_id TEXT PRIMARY KEY, name TEXT NOT NULL, version INTEGER NOT NULL,
          parent_version INTEGER, status TEXT NOT NULL, candidate_json TEXT NOT NULL,
          contract_json TEXT NOT NULL, analysis_json TEXT NOT NULL,
          mutation_json TEXT NOT NULL, runtime_json TEXT NOT NULL,
          cross_verification_json TEXT NOT NULL, approval_json TEXT NOT NULL,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
          UNIQUE(name, version)
        );
        CREATE TABLE IF NOT EXISTS skill_runtime_metrics (
          skill_id TEXT PRIMARY KEY, name TEXT NOT NULL, version INTEGER NOT NULL,
          executions INTEGER NOT NULL, successes INTEGER NOT NULL,
          failures INTEGER NOT NULL, verification_failures INTEGER NOT NULL,
          timeouts INTEGER NOT NULL, contract_violations INTEGER NOT NULL,
          rollback_count INTEGER NOT NULL, success_rate REAL NOT NULL,
          updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS skill_lifecycle_events (
          id TEXT PRIMARY KEY, skill_id TEXT NOT NULL, event_type TEXT NOT NULL,
          event_json TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS security_policies (
          policy_version INTEGER PRIMARY KEY, policy_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS approval_requests (
          approval_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, action_id TEXT NOT NULL,
          tool_or_skill TEXT NOT NULL, requested_capabilities_json TEXT NOT NULL,
          risk_level TEXT NOT NULL, reason TEXT NOT NULL, planned_effect TEXT NOT NULL,
          input_fingerprint TEXT NOT NULL, policy_version INTEGER NOT NULL,
          requested_at TEXT NOT NULL, expires_at TEXT NOT NULL, status TEXT NOT NULL,
          decided_at TEXT NOT NULL, actor TEXT NOT NULL, approval_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS security_decisions (
          decision_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, action_id TEXT NOT NULL,
          decision_json TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit_events (
          event_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, event_type TEXT NOT NULL,
          task_id TEXT NOT NULL, action_id TEXT NOT NULL, actor TEXT NOT NULL,
          tool_or_skill TEXT NOT NULL, capabilities_json TEXT NOT NULL,
          risk_level TEXT NOT NULL, policy_version INTEGER NOT NULL,
          approval_id TEXT NOT NULL, decision TEXT NOT NULL, reason TEXT NOT NULL,
          event_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS observability_events (
          sequence INTEGER PRIMARY KEY AUTOINCREMENT,
          event_id TEXT NOT NULL UNIQUE,
          event_type TEXT NOT NULL,
          timestamp TEXT NOT NULL,
          task_id TEXT NOT NULL,
          action_id TEXT NOT NULL,
          step_id TEXT NOT NULL,
          correlation_id TEXT NOT NULL,
          component TEXT NOT NULL,
          severity TEXT NOT NULL,
          schema_version INTEGER NOT NULL,
          metadata_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS api_idempotency_keys (
          idempotency_key TEXT PRIMARY KEY,
          request_fingerprint TEXT NOT NULL,
          task_id TEXT NOT NULL,
          response_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        """)
        # The historical schema is version 1.  Keep its creation intact, then
        # apply only monotonic migrations for databases opened by newer code.
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO task_state_schema(schema_version, applied_at) VALUES (?,?)",
                (1, now()),
            )
            rows = self.db.execute("SELECT schema_version FROM task_state_schema ORDER BY schema_version").fetchall()
            versions = {int(row["schema_version"]) for row in rows}
            unsupported = sorted(version for version in versions if version > CURRENT_SCHEMA_VERSION)
            if unsupported:
                raise UnsupportedSchemaVersion(f"unsupported_schema_version:{unsupported[-1]}")
            for version in range(2, CURRENT_SCHEMA_VERSION + 1):
                if version in versions:
                    self._assert_migration_objects(version)
                    continue
                existing_objects = self.db.execute(
                    "SELECT 1 FROM sqlite_master WHERE (type = 'table' OR type = 'index') "
                    "AND name IN ({}) LIMIT 1".format(
                        ",".join("?" for _ in self._required_objects(version))
                    ),
                    tuple(name for _, name in self._required_objects(version)),
                ).fetchone()
                # Migration 10 alters the table created by migration 8, so the
                # table's pre-existence is expected and must not short-circuit
                # the ALTER statements below.
                # Migration 14 reconciles both known schema-13 layouts; schema
                # 13 has already been validated, and strict combined validation
                # runs immediately after the additive migration.
                reconcile_schema13 = version == 14 and 13 in versions
                if existing_objects is not None and version not in {10, 11, 18} and not reconcile_schema13:
                    self._assert_migration_objects(version)
                self._apply_migration(version)
                self._assert_migration_objects(version)
                versions.add(version)

    def _apply_migration(self, version: int) -> None:
        """Apply one migration inside the caller's transaction, monotonically."""
        if version == 2:
            self.db.execute("""CREATE TABLE IF NOT EXISTS owner_permission_rules (
              rule_id TEXT PRIMARY KEY, policy_version INTEGER NOT NULL,
              actor TEXT NOT NULL, effect TEXT NOT NULL, scope_type TEXT NOT NULL,
              scope_key TEXT NOT NULL, conditions_json TEXT NOT NULL,
              expires_at TEXT NOT NULL, reason TEXT NOT NULL, status TEXT NOT NULL,
              created_at TEXT NOT NULL, disabled_at TEXT NOT NULL
            )""")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_owner_rules_active_scope ON owner_permission_rules(status, scope_type, scope_key)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_owner_rules_expiry ON owner_permission_rules(status, expires_at)")
            self.db.execute("""CREATE TABLE IF NOT EXISTS owner_policy_audit (
              event_id TEXT PRIMARY KEY, rule_id TEXT NOT NULL,
              policy_version INTEGER NOT NULL, event_type TEXT NOT NULL,
              actor TEXT NOT NULL, previous_json TEXT NOT NULL,
              new_json TEXT NOT NULL, reason TEXT NOT NULL,
              created_at TEXT NOT NULL
            )""")
            self.db.execute("INSERT OR IGNORE INTO task_state_schema(schema_version, applied_at) VALUES (?, ?)", (2, now()))
            return
        if version == 3:
            self.db.execute("""CREATE TABLE IF NOT EXISTS owner_policy_state (
              singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
              current_version INTEGER NOT NULL CHECK (current_version >= 0),
              updated_at TEXT NOT NULL
            )""")
            self.db.execute("""CREATE TABLE IF NOT EXISTS owner_permission_rule_history (
              history_id TEXT PRIMARY KEY, rule_id TEXT NOT NULL,
              policy_version INTEGER NOT NULL, event_type TEXT NOT NULL,
              snapshot_json TEXT NOT NULL, actor TEXT NOT NULL,
              reason TEXT NOT NULL, created_at TEXT NOT NULL
            )""")
            security = self.db.execute("SELECT COALESCE(MAX(policy_version), 0) AS version FROM security_policies").fetchone()
            rules = self.db.execute("SELECT COALESCE(MAX(policy_version), 0) AS version FROM owner_permission_rules").fetchone()
            initial = max(int(security["version"] or 0), int(rules["version"] or 0))
            self.db.execute(
                "INSERT OR IGNORE INTO owner_policy_state(singleton, current_version, updated_at) VALUES (1, ?, ?)",
                (initial, now()),
            )
            self.db.execute(
                "UPDATE owner_policy_state SET current_version = MAX(current_version, ?), updated_at = ? WHERE singleton = 1",
                (initial, now()),
            )
            self.db.execute("INSERT OR IGNORE INTO task_state_schema(schema_version, applied_at) VALUES (?, ?)", (3, now()))
            return
        if version == 4:
            self.db.execute("""CREATE TABLE IF NOT EXISTS task_queue (
              task_id TEXT PRIMARY KEY,
              queue_status TEXT NOT NULL CHECK (queue_status IN (
                'QUEUED', 'LEASED', 'WAITING_APPROVAL', 'RETRY_WAIT',
                'EXECUTION_UNKNOWN', 'COMPLETED', 'FAILED', 'BLOCKED'
              )),
              attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
              max_attempts INTEGER NOT NULL DEFAULT 3 CHECK (max_attempts >= 1),
              available_at TEXT NOT NULL,
              lease_id TEXT NOT NULL DEFAULT '',
              worker_id TEXT NOT NULL DEFAULT '',
              lease_acquired_at TEXT NOT NULL DEFAULT '',
              lease_expires_at TEXT NOT NULL DEFAULT '',
              last_started_at TEXT NOT NULL DEFAULT '',
              last_finished_at TEXT NOT NULL DEFAULT '',
              last_error_code TEXT NOT NULL DEFAULT '',
              last_error_message TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY (task_id) REFERENCES task_states(task_id)
            )""")
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_task_queue_claim "
                "ON task_queue(queue_status, available_at, created_at, task_id)"
            )
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_task_queue_lease "
                "ON task_queue(queue_status, lease_expires_at)"
            )
            self.db.execute("""INSERT OR IGNORE INTO task_queue
              (task_id, queue_status, attempt_count, max_attempts, available_at,
               lease_id, worker_id, lease_acquired_at, lease_expires_at,
               last_started_at, last_finished_at, last_error_code, last_error_message,
               created_at, updated_at)
              SELECT task_id,
                     CASE status
                       WHEN 'COMPLETED' THEN 'COMPLETED'
                       WHEN 'BLOCKED' THEN 'BLOCKED'
                       WHEN 'FAILED' THEN 'FAILED'
                       WHEN 'WAITING_APPROVAL' THEN 'WAITING_APPROVAL'
                       WHEN 'EXECUTION_UNKNOWN' THEN 'EXECUTION_UNKNOWN'
                       WHEN 'EXECUTING' THEN 'EXECUTION_UNKNOWN'
                       WHEN 'OBSERVING' THEN 'EXECUTION_UNKNOWN'
                       WHEN 'VERIFYING' THEN 'EXECUTION_UNKNOWN'
                       ELSE 'QUEUED'
                     END,
                     0, 3, created_at, '', '', '', '', '', '', '', '', created_at, updated_at
                FROM task_states""")
            self.db.execute("INSERT OR IGNORE INTO task_state_schema(schema_version, applied_at) VALUES (?, ?)", (4, now()))
            return
        if version == 5:
            self.db.execute("""CREATE TABLE IF NOT EXISTS knowledge_sources (
              source_id TEXT PRIMARY KEY,
              canonical_uri TEXT NOT NULL,
              source_type TEXT NOT NULL,
              source_class TEXT NOT NULL,
              retrieved_at TEXT NOT NULL,
              published_at TEXT NOT NULL DEFAULT '',
              content_sha256 TEXT NOT NULL,
              revision TEXT NOT NULL DEFAULT '',
              status TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE(canonical_uri, revision, content_sha256)
            )""")
            self.db.execute("""CREATE TABLE IF NOT EXISTS knowledge_claims (
              claim_id TEXT PRIMARY KEY,
              claim_fingerprint TEXT NOT NULL UNIQUE,
              domain TEXT NOT NULL,
              subject TEXT NOT NULL,
              predicate TEXT NOT NULL,
              object_text TEXT NOT NULL,
              knowledge_type TEXT NOT NULL,
              verification_status TEXT NOT NULL,
              status TEXT NOT NULL,
              confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
              observed_at TEXT NOT NULL,
              last_verified_at TEXT NOT NULL DEFAULT '',
              valid_from TEXT NOT NULL DEFAULT '',
              valid_until TEXT NOT NULL DEFAULT '',
              version INTEGER NOT NULL CHECK (version >= 1),
              metadata_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )""")
            self.db.execute("""CREATE TABLE IF NOT EXISTS knowledge_evidence (
              evidence_id TEXT PRIMARY KEY,
              evidence_fingerprint TEXT NOT NULL UNIQUE,
              claim_id TEXT NOT NULL,
              source_id TEXT NOT NULL,
              task_id TEXT NOT NULL DEFAULT '',
              step_id TEXT NOT NULL DEFAULT '',
              locator TEXT NOT NULL DEFAULT '',
              excerpt TEXT NOT NULL,
              excerpt_sha256 TEXT NOT NULL,
              verification_status TEXT NOT NULL,
              observed_at TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE(claim_id, source_id, task_id, step_id, locator, excerpt_sha256),
              FOREIGN KEY (claim_id) REFERENCES knowledge_claims(claim_id),
              FOREIGN KEY (source_id) REFERENCES knowledge_sources(source_id)
            )""")
            self.db.execute("""CREATE TABLE IF NOT EXISTS knowledge_conflicts (
              conflict_id TEXT PRIMARY KEY,
              conflict_fingerprint TEXT NOT NULL UNIQUE,
              claim_id TEXT NOT NULL,
              conflicting_claim_id TEXT NOT NULL,
              conflict_type TEXT NOT NULL,
              status TEXT NOT NULL,
              detected_at TEXT NOT NULL,
              evidence_ids_json TEXT NOT NULL,
              resolution_note TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              CHECK (claim_id <> conflicting_claim_id),
              UNIQUE(claim_id, conflicting_claim_id, conflict_type),
              FOREIGN KEY (claim_id) REFERENCES knowledge_claims(claim_id),
              FOREIGN KEY (conflicting_claim_id) REFERENCES knowledge_claims(claim_id)
            )""")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_sources_identity ON knowledge_sources(canonical_uri, revision, content_sha256)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_claims_fingerprint ON knowledge_claims(claim_fingerprint)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_claims_freshness ON knowledge_claims(status, verification_status, valid_until, last_verified_at)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_evidence_claim ON knowledge_evidence(claim_id, source_id, observed_at)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_conflicts_claim ON knowledge_conflicts(claim_id, conflicting_claim_id, status)")
            self.db.execute("INSERT OR IGNORE INTO task_state_schema(schema_version, applied_at) VALUES (?, ?)", (5, now()))
            return
        if version == 6:
            self.db.execute("""CREATE TABLE IF NOT EXISTS auth_users (
              user_id TEXT PRIMARY KEY,
              email TEXT NOT NULL UNIQUE,
              role TEXT NOT NULL CHECK (role IN ('OWNER', 'OPERATOR', 'VIEWER')),
              password_hash TEXT NOT NULL,
              status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'DISABLED')),
              failed_login_count INTEGER NOT NULL DEFAULT 0 CHECK (failed_login_count >= 0),
              locked_until TEXT NOT NULL DEFAULT '',
              last_login_at TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )""")
            self.db.execute("""CREATE TABLE IF NOT EXISTS auth_sessions (
              session_hash TEXT PRIMARY KEY,
              user_id TEXT NOT NULL,
              csrf_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              expires_at TEXT NOT NULL,
              last_seen_at TEXT NOT NULL,
              revoked_at TEXT NOT NULL DEFAULT '',
              FOREIGN KEY (user_id) REFERENCES auth_users(user_id)
            )""")
            self.db.execute("""CREATE TABLE IF NOT EXISTS auth_events (
              event_id TEXT PRIMARY KEY,
              event_type TEXT NOT NULL,
              user_id TEXT NOT NULL,
              email TEXT NOT NULL,
              created_at TEXT NOT NULL,
              remote_key TEXT NOT NULL,
              metadata_json TEXT NOT NULL
            )""")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_auth_users_email ON auth_users(email)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions(user_id)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_expiry ON auth_sessions(expires_at, revoked_at)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_auth_events_created ON auth_events(created_at)")
            self.db.execute("INSERT OR IGNORE INTO task_state_schema(schema_version, applied_at) VALUES (?, ?)", (6, now()))
            return
        if version == 7:
            self.db.execute("""CREATE TABLE IF NOT EXISTS task_owners (
              task_id TEXT PRIMARY KEY,
              user_id TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY (task_id) REFERENCES task_states(task_id),
              FOREIGN KEY (user_id) REFERENCES auth_users(user_id)
            )""")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_task_owners_user ON task_owners(user_id, created_at, task_id)")
            self.db.execute("INSERT OR IGNORE INTO task_state_schema(schema_version, applied_at) VALUES (?, ?)", (7, now()))
            return
        if version == 8:
            self.db.execute("""CREATE TABLE IF NOT EXISTS bugbounty_program_analyses (
              analysis_id TEXT PRIMARY KEY,
              owner_user_id TEXT NOT NULL,
              analysis_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            )""")
            self.db.execute("""CREATE TABLE IF NOT EXISTS bugbounty_research_stages (
              stage_id TEXT PRIMARY KEY,
              analysis_id TEXT NOT NULL,
              owner_user_id TEXT NOT NULL,
              stage_type TEXT NOT NULL,
              stage_status TEXT NOT NULL,
              budget_json TEXT NOT NULL,
              partial_result_json TEXT NOT NULL,
              result_json TEXT NOT NULL,
              error_code TEXT NOT NULL,
              started_at TEXT NOT NULL,
              finished_at TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY (analysis_id) REFERENCES bugbounty_program_analyses(analysis_id)
            )""")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_program_analysis_owner ON bugbounty_program_analyses(owner_user_id, created_at)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_research_stages_analysis ON bugbounty_research_stages(analysis_id, created_at, stage_id)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_research_stages_owner ON bugbounty_research_stages(owner_user_id, updated_at, stage_id)")
            self.db.execute("INSERT OR IGNORE INTO task_state_schema(schema_version, applied_at) VALUES (?, ?)", (8, now()))
            return
        if version == 9:
            self.db.execute("""CREATE TABLE IF NOT EXISTS bugbounty_finding_reviews (
              review_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL,
              finding_id TEXT NOT NULL,
              owner_user_id TEXT NOT NULL,
              decision TEXT NOT NULL,
              duplicate_of TEXT NOT NULL DEFAULT '',
              note TEXT NOT NULL DEFAULT '',
              evidence_sha256 TEXT NOT NULL DEFAULT '',
              evidence_validated INTEGER NOT NULL DEFAULT 0 CHECK (evidence_validated IN (0, 1)),
              actor_user_id TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY (task_id) REFERENCES task_states(task_id)
            )""")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_finding_reviews_task ON bugbounty_finding_reviews(task_id, finding_id, created_at, review_id)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_finding_reviews_owner ON bugbounty_finding_reviews(owner_user_id, created_at, review_id)")
            self.db.execute("INSERT OR IGNORE INTO task_state_schema(schema_version, applied_at) VALUES (?, ?)", (9, now()))
            return
        if version == 10:
            for column, definition in (
                ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
                ("max_attempts", "INTEGER NOT NULL DEFAULT 1"),
                ("lease_id", "TEXT NOT NULL DEFAULT ''"),
                ("worker_id", "TEXT NOT NULL DEFAULT ''"),
                ("lease_acquired_at", "TEXT NOT NULL DEFAULT ''"),
                ("lease_expires_at", "TEXT NOT NULL DEFAULT ''"),
                ("last_error_code", "TEXT NOT NULL DEFAULT ''"),
                ("last_error_message", "TEXT NOT NULL DEFAULT ''"),
            ):
                self.db.execute(f"ALTER TABLE bugbounty_research_stages ADD COLUMN {column} {definition}")
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_research_stages_claim "
                "ON bugbounty_research_stages(stage_status, lease_expires_at, created_at, stage_id)"
            )
            self.db.execute("INSERT OR IGNORE INTO task_state_schema(schema_version, applied_at) VALUES (?, ?)", (10, now()))
            return
        if version == 11:
            self.db.execute(
                "ALTER TABLE bugbounty_research_stages ADD COLUMN input_json TEXT NOT NULL DEFAULT '{}'"
            )
            self.db.execute("INSERT OR IGNORE INTO task_state_schema(schema_version, applied_at) VALUES (?, ?)", (11, now()))
            return
        if version == 12:
            self.db.execute("""CREATE TABLE IF NOT EXISTS remote_workers (
              worker_id TEXT PRIMARY KEY,
              credential_digest TEXT NOT NULL,
              credential_version INTEGER NOT NULL CHECK (credential_version >= 1),
              previous_credential_digest TEXT NOT NULL DEFAULT '',
              previous_credential_version INTEGER NOT NULL DEFAULT 0 CHECK (previous_credential_version >= 0),
              previous_credential_expires_at TEXT NOT NULL DEFAULT '',
              status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'ROTATED', 'REVOKED')),
              runtime_identity TEXT NOT NULL,
              image_digest TEXT NOT NULL,
              manifest_digest TEXT NOT NULL,
              acceptance_status TEXT NOT NULL CHECK (acceptance_status IN ('UNVERIFIED', 'VERIFIED')),
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              last_seen_at TEXT NOT NULL DEFAULT ''
            )""")
            self.db.execute("""CREATE TABLE IF NOT EXISTS remote_worker_jobs (
              job_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL DEFAULT '',
              worker_id TEXT NOT NULL,
              idempotency_key TEXT NOT NULL,
              envelope_json TEXT NOT NULL,
              lease_id TEXT NOT NULL,
              status TEXT NOT NULL CHECK (status IN ('QUEUED', 'LEASED', 'COMPLETED', 'FAILED', 'EXECUTION_UNKNOWN', 'BLOCKED')),
              lease_expires_at TEXT NOT NULL DEFAULT '',
              heartbeat_at TEXT NOT NULL DEFAULT '',
              attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
              result_json TEXT NOT NULL DEFAULT '{}',
              result_digest TEXT NOT NULL DEFAULT '',
              last_error_code TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              UNIQUE(worker_id, idempotency_key),
              FOREIGN KEY (worker_id) REFERENCES remote_workers(worker_id)
            )""")
            self.db.execute("""CREATE TABLE IF NOT EXISTS remote_worker_nonces (
              worker_id TEXT NOT NULL,
              nonce TEXT NOT NULL,
              seen_at TEXT NOT NULL,
              request_digest TEXT NOT NULL,
              PRIMARY KEY (worker_id, nonce),
              FOREIGN KEY (worker_id) REFERENCES remote_workers(worker_id)
            )""")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_remote_workers_status ON remote_workers(status, acceptance_status, updated_at, worker_id)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_remote_worker_jobs_claim ON remote_worker_jobs(status, worker_id, created_at, job_id)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_remote_worker_jobs_lease ON remote_worker_jobs(status, lease_expires_at, job_id)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_remote_worker_jobs_idempotency ON remote_worker_jobs(worker_id, idempotency_key)")
            self.db.execute("INSERT OR IGNORE INTO task_state_schema(schema_version, applied_at) VALUES (?, ?)", (12, now()))
            return
        if version == 13:
            self.db.execute("""CREATE TABLE IF NOT EXISTS knowledge_attestations (
              attestation_id TEXT PRIMARY KEY,
              claim_id TEXT NOT NULL,
              source_id TEXT NOT NULL,
              claim_version INTEGER NOT NULL CHECK (claim_version >= 1),
              key_id TEXT NOT NULL,
              signature TEXT NOT NULL,
              attestation_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY (claim_id) REFERENCES knowledge_claims(claim_id),
              FOREIGN KEY (source_id) REFERENCES knowledge_sources(source_id)
            )""")
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_attestations_claim "
                "ON knowledge_attestations(claim_id, claim_version, created_at, attestation_id)"
            )
            self.db.execute("INSERT OR IGNORE INTO task_state_schema(schema_version, applied_at) VALUES (?, ?)", (13, now()))
            return
        if version == 14:
            # Idempotently carry both branches' schema-13 features forward.
            # This preserves a PR1-era task_messages table and fills in the
            # knowledge_attestations table for that legacy layout.
            self.db.execute("""CREATE TABLE IF NOT EXISTS knowledge_attestations (
              attestation_id TEXT PRIMARY KEY,
              claim_id TEXT NOT NULL,
              source_id TEXT NOT NULL,
              claim_version INTEGER NOT NULL CHECK (claim_version >= 1),
              key_id TEXT NOT NULL,
              signature TEXT NOT NULL,
              attestation_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY (claim_id) REFERENCES knowledge_claims(claim_id),
              FOREIGN KEY (source_id) REFERENCES knowledge_sources(source_id)
            )""")
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_attestations_claim "
                "ON knowledge_attestations(claim_id, claim_version, created_at, attestation_id)"
            )
            self.db.execute("""CREATE TABLE IF NOT EXISTS task_messages (
              message_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL,
              sequence INTEGER NOT NULL CHECK (sequence >= 1),
              role TEXT NOT NULL CHECK (role = 'USER'),
              message_kind TEXT NOT NULL CHECK (message_kind IN ('INITIAL', 'FOLLOW_UP')),
              content TEXT NOT NULL CHECK (length(trim(content)) > 0),
              status TEXT NOT NULL CHECK (status IN ('PENDING', 'CONSUMED')),
              idempotency_key TEXT,
              request_fingerprint TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL,
              consumed_at TEXT NOT NULL DEFAULT '',
              consumed_checkpoint TEXT NOT NULL DEFAULT '',
              UNIQUE (task_id, sequence),
              UNIQUE (task_id, idempotency_key),
              FOREIGN KEY (task_id) REFERENCES task_states(task_id)
            )""")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_task_messages_task_sequence ON task_messages(task_id, sequence)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_task_messages_pending ON task_messages(task_id, status, sequence)")
            self.db.execute("INSERT OR IGNORE INTO task_state_schema(schema_version, applied_at) VALUES (?, ?)", (14, now()))
            return
        if version == 15:
            self.db.execute("""CREATE TABLE IF NOT EXISTS task_result_events (
              event_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL,
              owner_user_id TEXT NOT NULL,
              sequence INTEGER NOT NULL CHECK (sequence >= 1),
              role TEXT NOT NULL CHECK (role = 'SYSTEM'),
              message_kind TEXT NOT NULL CHECK (message_kind = 'RESULT'),
              task_status TEXT NOT NULL CHECK (task_status IN (
                'CREATED', 'QUEUED', 'PLANNING', 'READY', 'EXECUTING', 'EXECUTION_UNKNOWN',
                'OBSERVING', 'VERIFYING', 'WAITING_APPROVAL', 'BLOCKED', 'FAILED', 'COMPLETED'
              )),
              verification_status TEXT NOT NULL CHECK (verification_status IN (
                'VERIFIED', 'UNVERIFIED', 'EXECUTION_UNKNOWN', 'FAILED',
                'WAITING_APPROVAL', 'BLOCKED', 'IN_PROGRESS'
              )),
              result_available INTEGER NOT NULL CHECK (result_available IN (0, 1)),
              verified_action_count INTEGER NOT NULL CHECK (verified_action_count >= 0),
              failed_action_count INTEGER NOT NULL CHECK (failed_action_count >= 0),
              blocked_action_count INTEGER NOT NULL CHECK (blocked_action_count >= 0),
              pending_action_count INTEGER NOT NULL CHECK (pending_action_count >= 0),
              approval_pending_count INTEGER NOT NULL CHECK (approval_pending_count >= 0),
              approval_approved_count INTEGER NOT NULL CHECK (approval_approved_count >= 0),
              approval_denied_count INTEGER NOT NULL CHECK (approval_denied_count >= 0),
              approval_expired_count INTEGER NOT NULL CHECK (approval_expired_count >= 0),
              approval_cancelled_count INTEGER NOT NULL CHECK (approval_cancelled_count >= 0),
              result_digest TEXT NOT NULL,
              transition_fingerprint TEXT NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE (task_id, sequence),
              UNIQUE (task_id, transition_fingerprint),
              FOREIGN KEY (task_id) REFERENCES task_states(task_id),
              FOREIGN KEY (owner_user_id) REFERENCES auth_users(user_id)
            )""")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_task_result_events_task_sequence ON task_result_events(task_id, sequence)")
            self.db.execute("""CREATE TRIGGER IF NOT EXISTS task_result_events_owner_matches_task
              BEFORE INSERT ON task_result_events
              WHEN NOT EXISTS (
                SELECT 1 FROM task_owners WHERE task_id = NEW.task_id AND user_id = NEW.owner_user_id
              ) BEGIN
                SELECT RAISE(ABORT, 'task_result_events_owner_mismatch');
              END""")
            self.db.execute("""CREATE TRIGGER IF NOT EXISTS task_result_events_no_update
              BEFORE UPDATE ON task_result_events BEGIN
                SELECT RAISE(ABORT, 'task_result_events_immutable');
              END""")
            self.db.execute("""CREATE TRIGGER IF NOT EXISTS task_result_events_no_delete
              BEFORE DELETE ON task_result_events BEGIN
                SELECT RAISE(ABORT, 'task_result_events_immutable');
              END""")
            self.db.execute("INSERT OR IGNORE INTO task_state_schema(schema_version, applied_at) VALUES (?, ?)", (15, now()))
            return
        if version == 16:
            self.db.execute("""CREATE TABLE IF NOT EXISTS task_assistant_replies (
              reply_id TEXT PRIMARY KEY,
              reply_key TEXT NOT NULL UNIQUE,
              event_id TEXT NOT NULL UNIQUE,
              task_id TEXT NOT NULL,
              owner_user_id TEXT NOT NULL,
              source_sequence INTEGER NOT NULL CHECK (source_sequence >= 1),
              source_digest TEXT NOT NULL,
              role TEXT NOT NULL CHECK (role = 'ASSISTANT'),
              message_kind TEXT NOT NULL CHECK (message_kind = 'RESULT_REPLY'),
              reply_code TEXT NOT NULL CHECK (reply_code IN (
                'VERIFIED_A', 'VERIFIED_B', 'UNVERIFIED_A', 'UNVERIFIED_B',
                'FAILED_A', 'FAILED_B', 'BLOCKED_A', 'BLOCKED_B', 'UNKNOWN_A', 'UNKNOWN_B',
                'APPROVAL_PENDING_A', 'APPROVAL_PENDING_B', 'APPROVAL_DECISION_A', 'APPROVAL_DECISION_B',
                'IN_PROGRESS_A', 'IN_PROGRESS_B'
              )),
              content TEXT NOT NULL CHECK (length(content) BETWEEN 1 AND 500),
              generation_mode TEXT NOT NULL CHECK (generation_mode IN ('MODEL_VARIANT', 'DETERMINISTIC_FALLBACK')),
              created_at TEXT NOT NULL,
              UNIQUE (task_id, reply_key),
              FOREIGN KEY (event_id) REFERENCES task_result_events(event_id),
              FOREIGN KEY (task_id) REFERENCES task_states(task_id),
              FOREIGN KEY (owner_user_id) REFERENCES auth_users(user_id)
            )""")
            self.db.execute("""CREATE TABLE IF NOT EXISTS task_assistant_reply_claims (
              event_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL,
              owner_user_id TEXT NOT NULL,
              lease_owner TEXT NOT NULL,
              lease_expires_at TEXT NOT NULL,
              attempt_count INTEGER NOT NULL CHECK (attempt_count >= 1),
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY (event_id) REFERENCES task_result_events(event_id),
              FOREIGN KEY (task_id) REFERENCES task_states(task_id),
              FOREIGN KEY (owner_user_id) REFERENCES auth_users(user_id)
            )""")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_task_assistant_replies_task_sequence ON task_assistant_replies(task_id, source_sequence)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_task_assistant_reply_claims_lease ON task_assistant_reply_claims(lease_expires_at, event_id)")
            self.db.execute("""CREATE TRIGGER IF NOT EXISTS task_assistant_replies_owner_matches_event
              BEFORE INSERT ON task_assistant_replies
              WHEN NOT EXISTS (
                SELECT 1 FROM task_result_events AS e
                JOIN task_owners AS o ON o.task_id = e.task_id AND o.user_id = e.owner_user_id
                WHERE e.event_id = NEW.event_id AND e.task_id = NEW.task_id
                  AND e.owner_user_id = NEW.owner_user_id AND e.sequence = NEW.source_sequence
                  AND e.result_digest = NEW.source_digest
              ) BEGIN
                SELECT RAISE(ABORT, 'task_assistant_replies_owner_mismatch');
              END""")
            self.db.execute("""CREATE TRIGGER IF NOT EXISTS task_assistant_replies_no_update
              BEFORE UPDATE ON task_assistant_replies BEGIN
                SELECT RAISE(ABORT, 'task_assistant_replies_immutable');
              END""")
            self.db.execute("""CREATE TRIGGER IF NOT EXISTS task_assistant_replies_no_delete
              BEFORE DELETE ON task_assistant_replies BEGIN
                SELECT RAISE(ABORT, 'task_assistant_replies_immutable');
              END""")
            self.db.execute("""CREATE TRIGGER IF NOT EXISTS task_assistant_reply_claims_owner_matches_event
              BEFORE INSERT ON task_assistant_reply_claims
              WHEN NOT EXISTS (
                SELECT 1 FROM task_result_events AS e
                JOIN task_owners AS o ON o.task_id = e.task_id AND o.user_id = e.owner_user_id
                WHERE e.event_id = NEW.event_id AND e.task_id = NEW.task_id
                  AND e.owner_user_id = NEW.owner_user_id
              ) BEGIN
                SELECT RAISE(ABORT, 'task_assistant_reply_claim_owner_mismatch');
              END""")
            self.db.execute("INSERT OR IGNORE INTO task_state_schema(schema_version, applied_at) VALUES (?, ?)", (16, now()))
            return
        if version == 17:
            self.db.execute("""CREATE TABLE IF NOT EXISTS task_conversation_sequences (
              task_id TEXT PRIMARY KEY,
              last_sequence INTEGER NOT NULL CHECK (last_sequence >= 0),
              FOREIGN KEY (task_id) REFERENCES task_states(task_id)
            )""")

            # Result events and their assistant replies were immutable in v16.
            # This one-time migration rewrites only their sequence bindings,
            # then reinstalls the immutability guards in the same transaction.
            self.db.execute("DROP TRIGGER IF EXISTS task_result_events_no_update")
            self.db.execute("DROP TRIGGER IF EXISTS task_assistant_replies_no_update")

            maximum = self.db.execute(
                "SELECT MAX(sequence) AS sequence FROM ("
                "SELECT sequence FROM task_messages UNION ALL "
                "SELECT sequence FROM task_result_events)"
            ).fetchone()
            item_count = int(self.db.execute(
                "SELECT (SELECT COUNT(*) FROM task_messages) + "
                "(SELECT COUNT(*) FROM task_result_events) AS count"
            ).fetchone()["count"])
            offset = int(maximum["sequence"] or 0) + item_count + 1
            if offset + int(maximum["sequence"] or 0) > (2**63 - 1):
                raise UnsupportedSchemaVersion("conversation_sequence_migration_overflow")
            self.db.execute("UPDATE task_messages SET sequence = sequence + ?", (offset,))
            self.db.execute("UPDATE task_result_events SET sequence = sequence + ?", (offset,))
            self.db.execute(
                "UPDATE task_assistant_replies SET source_sequence = "
                "(SELECT sequence FROM task_result_events WHERE event_id = task_assistant_replies.event_id)"
            )

            task_ids = [str(row["task_id"]) for row in self.db.execute(
                "SELECT task_id FROM task_states ORDER BY task_id"
            ).fetchall()]
            migrated_last_sequences: dict[str, int] = {}
            for task_id in task_ids:
                messages = [dict(row) for row in self.db.execute(
                    "SELECT message_id, sequence, created_at FROM task_messages "
                    "WHERE task_id = ? ORDER BY sequence", (task_id,)
                ).fetchall()]
                events = [dict(row) for row in self.db.execute(
                    "SELECT event_id, sequence, created_at FROM task_result_events "
                    "WHERE task_id = ? ORDER BY sequence", (task_id,)
                ).fetchall()]

                # Preserve each legacy per-type sequence. Timestamps are used
                # only to make a deterministic best-effort historical merge;
                # ties put USER messages before SYSTEM results. Exact cross-type
                # order for legacy timestamp ties was never stored.
                merged: list[tuple[str, dict[str, Any]]] = []
                message_index = event_index = 0
                while message_index < len(messages) or event_index < len(events):
                    if event_index >= len(events):
                        item_type, item = "message", messages[message_index]
                        message_index += 1
                    elif message_index >= len(messages):
                        item_type, item = "event", events[event_index]
                        event_index += 1
                    else:
                        message = messages[message_index]
                        result_event = events[event_index]
                        message_key = (str(message["created_at"]), 0, int(message["sequence"]))
                        event_key = (str(result_event["created_at"]), 1, int(result_event["sequence"]))
                        if message_key <= event_key:
                            item_type, item = "message", message
                            message_index += 1
                        else:
                            item_type, item = "event", result_event
                            event_index += 1
                    merged.append((item_type, item))

                for sequence, (item_type, item) in enumerate(merged, start=1):
                    if item_type == "message":
                        self.db.execute(
                            "UPDATE task_messages SET sequence = ? WHERE message_id = ?",
                            (sequence, str(item["message_id"])),
                        )
                    else:
                        event_id = str(item["event_id"])
                        self.db.execute(
                            "UPDATE task_result_events SET sequence = ? WHERE event_id = ?",
                            (sequence, event_id),
                        )
                        self.db.execute(
                            "UPDATE task_assistant_replies SET source_sequence = ? WHERE event_id = ?",
                            (sequence, event_id),
                        )
                migrated_last_sequences[task_id] = len(merged)

            for task_id, last_sequence in migrated_last_sequences.items():
                self.db.execute(
                    "INSERT INTO task_conversation_sequences(task_id, last_sequence) VALUES (?, ?)",
                    (task_id, last_sequence),
                )

            self.db.execute("""CREATE TRIGGER IF NOT EXISTS task_messages_sequence_counter_guard
              BEFORE INSERT ON task_messages
              WHEN NOT EXISTS (
                SELECT 1 FROM task_conversation_sequences
                WHERE task_id = NEW.task_id AND last_sequence = NEW.sequence
              ) BEGIN
                SELECT RAISE(ABORT, 'task_messages_sequence_not_allocated');
              END""")
            self.db.execute("""CREATE TRIGGER IF NOT EXISTS task_messages_sequence_cross_type_guard
              BEFORE INSERT ON task_messages
              WHEN EXISTS (
                SELECT 1 FROM task_result_events
                WHERE task_id = NEW.task_id AND sequence = NEW.sequence
              ) BEGIN
                SELECT RAISE(ABORT, 'task_conversation_sequence_duplicate');
              END""")
            self.db.execute("""CREATE TRIGGER IF NOT EXISTS task_messages_sequence_immutable
              BEFORE UPDATE OF sequence ON task_messages BEGIN
                SELECT RAISE(ABORT, 'task_messages_sequence_immutable');
              END""")
            self.db.execute("""CREATE TRIGGER IF NOT EXISTS task_result_events_sequence_counter_guard
              BEFORE INSERT ON task_result_events
              WHEN NOT EXISTS (
                SELECT 1 FROM task_conversation_sequences
                WHERE task_id = NEW.task_id AND last_sequence = NEW.sequence
              ) BEGIN
                SELECT RAISE(ABORT, 'task_result_events_sequence_not_allocated');
              END""")
            self.db.execute("""CREATE TRIGGER IF NOT EXISTS task_result_events_sequence_cross_type_guard
              BEFORE INSERT ON task_result_events
              WHEN EXISTS (
                SELECT 1 FROM task_messages
                WHERE task_id = NEW.task_id AND sequence = NEW.sequence
              ) BEGIN
                SELECT RAISE(ABORT, 'task_conversation_sequence_duplicate');
              END""")
            self.db.execute("""CREATE TRIGGER IF NOT EXISTS task_conversation_sequences_monotonic_guard
              BEFORE UPDATE OF last_sequence ON task_conversation_sequences
              WHEN NEW.last_sequence != OLD.last_sequence + 1 BEGIN
                SELECT RAISE(ABORT, 'task_conversation_sequence_not_monotonic');
              END""")
            self.db.execute("""CREATE TRIGGER IF NOT EXISTS task_conversation_sequences_no_delete
              BEFORE DELETE ON task_conversation_sequences BEGIN
                SELECT RAISE(ABORT, 'task_conversation_sequence_immutable');
              END""")
            self.db.execute("""CREATE TRIGGER IF NOT EXISTS task_result_events_no_update
              BEFORE UPDATE ON task_result_events BEGIN
                SELECT RAISE(ABORT, 'task_result_events_immutable');
              END""")
            self.db.execute("""CREATE TRIGGER IF NOT EXISTS task_assistant_replies_no_update
              BEFORE UPDATE ON task_assistant_replies BEGIN
                SELECT RAISE(ABORT, 'task_assistant_replies_immutable');
              END""")
            self.db.execute("INSERT OR IGNORE INTO task_state_schema(schema_version, applied_at) VALUES (?, ?)", (17, now()))
            return
        if version == 18:
            columns = {
                str(item["name"])
                for item in self.db.execute("PRAGMA table_info(task_result_events)").fetchall()
            }
            if "planning_mode" not in columns:
                self.db.execute(
                    "ALTER TABLE task_result_events ADD COLUMN planning_mode TEXT NOT NULL "
                    "DEFAULT 'UNKNOWN' CHECK (planning_mode IN "
                    "('UNKNOWN', 'PROVIDER_VALIDATED', 'DETERMINISTIC_FALLBACK', "
                    "'DETERMINISTIC_ROUTE', 'DETERMINISTIC_REPLAN'))"
                )
            if "plan_version" not in columns:
                self.db.execute(
                    "ALTER TABLE task_result_events ADD COLUMN plan_version INTEGER NOT NULL "
                    "DEFAULT 0 CHECK (plan_version >= 0)"
                )
            self.db.execute("INSERT OR IGNORE INTO task_state_schema(schema_version, applied_at) VALUES (?, ?)", (18, now()))
            return
        raise UnsupportedSchemaVersion(f"unknown_migration:{version}")

    def latest_owner_policy_version(self) -> int:
        with self._db_lock:
            row = self.db.execute("SELECT current_version FROM owner_policy_state WHERE singleton = 1").fetchone()
        if row is None:
            raise UnsupportedSchemaVersion("missing_owner_policy_state")
        return int(row["current_version"])

    def _allocate_owner_policy_version(self) -> int:
        try:
            self.db.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                raise RuntimeError("owner_policy_version_busy") from exc
            raise
        try:
            current = self.latest_owner_policy_version()
            version = current + 1
            self.db.execute(
                "UPDATE owner_policy_state SET current_version = ?, updated_at = ? WHERE singleton = 1",
                (version, now()),
            )
            return version
        except Exception:
            self.db.rollback()
            raise

    def owner_permission_rules(self, *, active_only: bool = True) -> list[dict[str, Any]]:
        query = "SELECT * FROM owner_permission_rules"
        with self._db_lock:
            if active_only:
                query += " WHERE status = 'ACTIVE' AND (expires_at = '' OR expires_at > ?)"
                rows = self.db.execute(query + " ORDER BY policy_version DESC, created_at DESC", (now(),)).fetchall()
            else:
                rows = self.db.execute(query + " ORDER BY policy_version DESC, created_at DESC").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["conditions"] = json.loads(item.pop("conditions_json"))
            result.append(item)
        return result

    def create_owner_permission_rule(self, rule: Any, *, actor: str, reason: str) -> dict[str, Any]:
        if not isinstance(actor, str) or not actor.strip() or len(actor) > 256:
            raise ValueError("owner_actor_required")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
            raise ValueError("owner_rule_reason_required")
        payload = rule.to_dict() if hasattr(rule, "to_dict") else dict(rule)
        from private_agent.security import OwnerPermissionRule

        validated = OwnerPermissionRule(
            rule_id=str(payload.get("rule_id", "")),
            effect=str(payload.get("effect", "")),
            scope_type=str(payload.get("scope_type", "")),
            scope_key=str(payload.get("scope_key", "")),
            conditions=dict(payload.get("conditions", {})),
            expires_at=str(payload.get("expires_at", "")),
            policy_version=int(payload.get("policy_version", 0)),
            actor=actor,
            reason=reason,
            created_at=str(payload.get("created_at") or now()),
        )
        payload = validated.to_dict()
        stamp = now()
        with self._db_lock, self.db:
            version = self._allocate_owner_policy_version()
            payload["policy_version"] = version
            payload["actor"] = actor
            payload["reason"] = reason
            payload["created_at"] = payload.get("created_at") or stamp
            payload["status"] = "ACTIVE"
            payload.setdefault("disabled_at", "")
            self.db.execute(
                """INSERT INTO owner_permission_rules
                (rule_id, policy_version, actor, effect, scope_type, scope_key, conditions_json,
                 expires_at, reason, status, created_at, disabled_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (payload["rule_id"], version, actor, payload["effect"], payload["scope_type"],
                 payload["scope_key"], json.dumps(payload.get("conditions", {}), sort_keys=True),
                 payload.get("expires_at", ""), reason, "ACTIVE", payload["created_at"], ""),
            )
            snapshot = json.dumps(payload, sort_keys=True)
            self.db.execute(
                """INSERT INTO owner_policy_audit
                (event_id, rule_id, policy_version, event_type, actor, previous_json, new_json, reason, created_at)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (f"owner-policy:{payload['rule_id']}:{version}", payload["rule_id"], version,
                 "RULE_CREATED", actor, "{}", snapshot, reason, stamp),
            )
            self.db.execute(
                """INSERT INTO owner_permission_rule_history
                (history_id, rule_id, policy_version, event_type, snapshot_json, actor, reason, created_at)
                VALUES (?,?,?,?,?,?,?,?)""",
                (f"owner-rule-history:{payload['rule_id']}:{version}", payload["rule_id"], version,
                 "RULE_CREATED", snapshot, actor, reason, stamp),
            )
        return payload

    def disable_owner_permission_rule(self, rule_id: str, *, actor: str, reason: str) -> dict[str, Any]:
        if not actor or len(actor) > 256:
            raise ValueError("owner_actor_required")
        if not reason or len(reason) > 1000:
            raise ValueError("owner_rule_reason_required")
        with self._db_lock, self.db:
            row = self.db.execute("SELECT * FROM owner_permission_rules WHERE rule_id = ?", (rule_id,)).fetchone()
            if row is None:
                raise KeyError("owner_rule_not_found")
            previous = dict(row)
            version = self._allocate_owner_policy_version()
            stamp = now()
            self.db.execute("UPDATE owner_permission_rules SET status='DISABLED', disabled_at=?, policy_version=? WHERE rule_id=?", (stamp, version, rule_id))
            updated = dict(self.db.execute("SELECT * FROM owner_permission_rules WHERE rule_id = ?", (rule_id,)).fetchone())
            snapshot = json.dumps(updated, sort_keys=True)
            self.db.execute(
                """INSERT INTO owner_policy_audit
                (event_id, rule_id, policy_version, event_type, actor, previous_json, new_json, reason, created_at)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (f"owner-policy:{rule_id}:{version}", rule_id, version, "RULE_DISABLED", actor,
                 json.dumps(previous, sort_keys=True), snapshot, reason, stamp),
            )
            self.db.execute(
                """INSERT INTO owner_permission_rule_history
                (history_id, rule_id, policy_version, event_type, snapshot_json, actor, reason, created_at)
                VALUES (?,?,?,?,?,?,?,?)""",
                (f"owner-rule-history:{rule_id}:{version}", rule_id, version,
                 "RULE_DISABLED", snapshot, actor, reason, stamp),
            )
        return updated

    def owner_permission_rule_history(self, rule_id: str | None = None) -> list[dict[str, Any]]:
        with self._db_lock:
            if rule_id is None:
                rows = self.db.execute("SELECT * FROM owner_permission_rule_history ORDER BY policy_version, created_at").fetchall()
            else:
                rows = self.db.execute("SELECT * FROM owner_permission_rule_history WHERE rule_id = ? ORDER BY policy_version, created_at", (rule_id,)).fetchall()
        return [dict(row) for row in rows]

    def owner_policy_audit(self) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = self.db.execute("SELECT * FROM owner_policy_audit ORDER BY policy_version, created_at").fetchall()
        return [dict(row) for row in rows]

    def save_task(self, task_id: str, goal: str, status: str, plan: Any, result: Any, attempts: int) -> None:
        stamp = now()
        with self._lease_write(task_id):
            self.db.execute("INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?,?,?)",
                (task_id, goal, status, json.dumps(plan), json.dumps(result), attempts, stamp, stamp))

    def get_api_idempotency(self, idempotency_key: str) -> dict[str, Any] | None:
        with self._db_lock:
            row = self.db.execute(
                "SELECT idempotency_key, request_fingerprint, task_id, response_json, created_at FROM api_idempotency_keys WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        return dict(row) if row else None

    def save_api_idempotency(
        self,
        idempotency_key: str,
        request_fingerprint: str,
        task_id: str,
        response_json: str,
    ) -> None:
        with self._db_lock, self.db:
            self.db.execute(
                "INSERT INTO api_idempotency_keys (idempotency_key, request_fingerprint, task_id, response_json, created_at) VALUES (?,?,?,?,?)",
                (idempotency_key, request_fingerprint, task_id, response_json, now()),
            )

    def _persist_task_state_in_transaction(
        self,
        payload: dict[str, Any],
        checkpoint_type: str,
        *,
        task_status: str | None = None,
        enforce_lease: bool = True,
    ) -> None:
        """Persist one state snapshot while the caller owns the transaction."""
        task_id = str(payload["task_id"])
        if enforce_lease:
            self._assert_active_lease(task_id)
        status = str(task_status or payload["status"])
        created_at = str(payload["created_at"])
        updated_at = str(payload["updated_at"])
        state_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        checkpoint_at = str(payload.get("last_checkpoint_at") or updated_at or now())
        row = self.db.execute(
            "SELECT COALESCE(MAX(sequence), 0) AS sequence FROM task_checkpoints WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        sequence = int(row["sequence"] or 0) + 1
        self.db.execute(
            """INSERT OR REPLACE INTO task_states
            (task_id, state_version, status, current_phase, current_step,
             plan_version, plan_fingerprint, state_fingerprint, state_json,
             created_at, updated_at, last_checkpoint_at, last_successful_checkpoint)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                task_id,
                int(payload["state_version"]),
                status,
                payload["current_phase"],
                payload["current_step"],
                int(payload["plan_version"]),
                payload["plan_fingerprint"],
                payload["state_fingerprint"],
                state_json,
                created_at,
                updated_at,
                payload["last_checkpoint_at"],
                payload["last_successful_checkpoint"],
            ),
        )
        self.db.execute(
            """INSERT INTO task_checkpoints
            (checkpoint_id, task_id, sequence, checkpoint_type, state_version,
             state_fingerprint, state_json, created_at)
            VALUES (?,?,?,?,?,?,?,?)""",
            (
                f"{task_id}:{sequence}",
                task_id,
                sequence,
                checkpoint_type,
                int(payload["state_version"]),
                payload["state_fingerprint"],
                state_json,
                checkpoint_at,
            ),
        )
        for action_key, action_payload in payload.get("action_records", {}).items():
            self.db.execute(
                """INSERT OR REPLACE INTO task_action_records
                (action_key, task_id, step_id, tool_name, status, action_json, updated_at)
                VALUES (?,?,?,?,?,?,?)""",
                (
                    action_key,
                    task_id,
                    action_payload.get("step_id", ""),
                    action_payload.get("tool_name", ""),
                    action_payload.get("status", "NOT_STARTED"),
                    json.dumps(action_payload, ensure_ascii=False, sort_keys=True),
                    action_payload.get("updated_at", updated_at),
                ),
            )
        self.db.execute(
            """INSERT OR REPLACE INTO tasks
            (id, goal, status, plan_json, result_json, attempts, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?)""",
            (
                task_id,
                payload["goal"],
                status.lower(),
                json.dumps(payload.get("plan", {}), ensure_ascii=False, sort_keys=True),
                json.dumps(payload.get("result", {}), ensure_ascii=False, sort_keys=True),
                int(payload.get("context", {}).get("attempts", 0)),
                created_at,
                updated_at,
            ),
        )
        self._persist_task_result_event_in_transaction(payload, status)

    def _allocate_task_conversation_sequence_in_transaction(self, task_id: str) -> int:
        """Allocate the next task-wide conversation sequence inside a write transaction."""
        if not self.db.in_transaction:
            raise RuntimeError("conversation_sequence_transaction_required")
        self.db.execute(
            "INSERT INTO task_conversation_sequences(task_id, last_sequence) VALUES (?, 1) "
            "ON CONFLICT(task_id) DO UPDATE SET last_sequence = last_sequence + 1",
            (str(task_id),),
        )
        row = self.db.execute(
            "SELECT last_sequence FROM task_conversation_sequences WHERE task_id = ?",
            (str(task_id),),
        ).fetchone()
        if row is None:
            raise RuntimeError("conversation_sequence_allocation_failed")
        return int(row["last_sequence"])

    def _persist_task_result_event_in_transaction(self, payload: dict[str, Any], status: str) -> None:
        """Append a compact immutable result snapshot when its trusted summary changes."""
        task_id = str(payload["task_id"])
        owner = self.db.execute(
            "SELECT user_id FROM task_owners WHERE task_id = ?", (task_id,)
        ).fetchone()
        if owner is None:
            return
        owner_user_id = str(owner["user_id"])
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        result_json = json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))
        result_digest = hashlib.sha256(result_json.encode("utf-8")).hexdigest()
        planning_mode = planning_mode_from_plan(payload.get("plan"))
        plan_version = plan_version_from_state(payload)
        verification = result.get("verification") if isinstance(result.get("verification"), dict) else {}
        if status == "COMPLETED":
            verification_status = "VERIFIED" if verification.get("verified") is True else "UNVERIFIED"
        elif status == "EXECUTION_UNKNOWN":
            verification_status = "EXECUTION_UNKNOWN"
        elif status == "FAILED":
            verification_status = "FAILED"
        elif status == "WAITING_APPROVAL":
            verification_status = "WAITING_APPROVAL"
        elif status == "BLOCKED":
            verification_status = "BLOCKED"
        else:
            verification_status = "IN_PROGRESS"
        action_records = payload.get("action_records")
        action_statuses = [
            str(record.get("status", ""))
            for record in action_records.values()
            if isinstance(action_records, dict) and isinstance(record, dict)
        ] if isinstance(action_records, dict) else []
        approval_counts = {
            "PENDING": 0,
            "APPROVED": 0,
            "DENIED": 0,
            "EXPIRED": 0,
            "CANCELLED": 0,
        }
        for row in self.db.execute(
            "SELECT status, COUNT(*) AS count FROM approval_requests WHERE task_id = ? GROUP BY status",
            (task_id,),
        ).fetchall():
            key = str(row["status"])
            if key in approval_counts:
                approval_counts[key] = int(row["count"])
        summary = {
            "owner_user_id": owner_user_id,
            "task_status": str(status),
            "verification_status": verification_status,
            "result_available": int(bool(result)),
            "verified_action_count": action_statuses.count("VERIFIED"),
            "failed_action_count": action_statuses.count("FAILED"),
            "blocked_action_count": action_statuses.count("BLOCKED"),
            "pending_action_count": len(payload.get("pending_actions", [])) if isinstance(payload.get("pending_actions"), list) else 0,
            "approval_pending_count": approval_counts["PENDING"],
            "approval_approved_count": approval_counts["APPROVED"],
            "approval_denied_count": approval_counts["DENIED"],
            "approval_expired_count": approval_counts["EXPIRED"],
            "approval_cancelled_count": approval_counts["CANCELLED"],
            "result_digest": result_digest,
            "planning_mode": planning_mode,
            "plan_version": plan_version,
        }
        latest = self.db.execute(
            "SELECT owner_user_id, task_status, verification_status, result_available, "
            "verified_action_count, failed_action_count, blocked_action_count, pending_action_count, "
            "approval_pending_count, approval_approved_count, approval_denied_count, "
            "approval_expired_count, approval_cancelled_count, result_digest, planning_mode, plan_version "
            "FROM task_result_events WHERE task_id = ? ORDER BY sequence DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        if latest is not None and all(latest[key] == value for key, value in summary.items()):
            return
        transition = {
            "task_id": task_id,
            "state_version": int(payload["state_version"]),
            "state_fingerprint": str(payload["state_fingerprint"]),
            **summary,
        }
        transition_fingerprint = hashlib.sha256(
            json.dumps(transition, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        duplicate = self.db.execute(
            "SELECT 1 FROM task_result_events WHERE task_id = ? AND transition_fingerprint = ?",
            (task_id, transition_fingerprint),
        ).fetchone()
        if duplicate is not None:
            return
        sequence = self._allocate_task_conversation_sequence_in_transaction(task_id)
        event_id = hashlib.sha256(f"task-result:{task_id}:{transition_fingerprint}".encode("utf-8")).hexdigest()
        self.db.execute(
            """INSERT OR IGNORE INTO task_result_events
            (event_id, task_id, owner_user_id, sequence, role, message_kind, task_status,
             verification_status, result_available, verified_action_count, failed_action_count,
             blocked_action_count, pending_action_count, approval_pending_count, approval_approved_count,
             approval_denied_count, approval_expired_count, approval_cancelled_count, result_digest,
             transition_fingerprint, created_at, planning_mode, plan_version)
            VALUES (?, ?, ?, ?, 'SYSTEM', 'RESULT', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id, task_id, owner_user_id, sequence, summary["task_status"],
                summary["verification_status"], summary["result_available"], summary["verified_action_count"],
                summary["failed_action_count"], summary["blocked_action_count"], summary["pending_action_count"],
                summary["approval_pending_count"], summary["approval_approved_count"],
                summary["approval_denied_count"], summary["approval_expired_count"],
                summary["approval_cancelled_count"], summary["result_digest"], transition_fingerprint,
                str(payload.get("updated_at") or now()), summary["planning_mode"], summary["plan_version"],
            ),
        )

    def _persist_task_result_event_for_current_state_in_transaction(self, task_id: str) -> None:
        """Append an event using the validated persisted state while the caller owns the transaction."""
        from private_agent.core.task_state import StateIntegrityError, TaskState

        owner = self.db.execute(
            "SELECT 1 FROM task_owners WHERE task_id = ?", (str(task_id),)
        ).fetchone()
        if owner is None:
            return
        row = self.db.execute(
            "SELECT state_json FROM task_states WHERE task_id = ?", (str(task_id),)
        ).fetchone()
        if row is None:
            raise ValueError("approval_task_state_missing")
        try:
            state = TaskState.from_dict(json.loads(str(row["state_json"])))
        except (TypeError, ValueError, KeyError, StateIntegrityError) as exc:
            raise ValueError("approval_task_state_untrusted") from exc
        if state.task_id != str(task_id):
            raise ValueError("approval_task_state_binding_mismatch")
        self._persist_task_result_event_in_transaction(state.to_dict(), state.status)

    def save_task_state(
        self,
        state: Any,
        checkpoint_type: str,
        *,
        task_status: str | None = None,
        consume_message_ids: list[str] | tuple[str, ...] = (),
    ) -> None:
        """Persist state, checkpoint and action records in one SQLite transaction."""
        payload = state.to_dict() if hasattr(state, "to_dict") else dict(state)
        with self._lease_write(str(payload["task_id"])):
            message_ids = tuple(dict.fromkeys(str(item) for item in consume_message_ids))
            if message_ids or checkpoint_type == "BUG_BOUNTY_FOLLOWUP_ACKNOWLEDGED":
                task_id = str(payload["task_id"])
                self._assert_verified_conversation_checkpoint_in_transaction(task_id, checkpoint_type=checkpoint_type)
            if message_ids:
                task_id = str(payload["task_id"])
                placeholders = ",".join("?" for _ in message_ids)
                rows = self.db.execute(
                    f"SELECT message_id FROM task_messages WHERE task_id = ? AND status = 'PENDING' AND message_id IN ({placeholders})",
                    (task_id, *message_ids),
                ).fetchall()
                if len(rows) != len(message_ids):
                    raise ValueError("conversation_message_not_pending")
            self._persist_task_state_in_transaction(payload, checkpoint_type, task_status=task_status)
            if message_ids:
                stamp = now()
                placeholders = ",".join("?" for _ in message_ids)
                cursor = self.db.execute(
                    f"UPDATE task_messages SET status = 'CONSUMED', consumed_at = ?, consumed_checkpoint = ? "
                    f"WHERE task_id = ? AND status = 'PENDING' AND message_id IN ({placeholders})",
                    (stamp, checkpoint_type, str(payload["task_id"]), *message_ids),
                )
                if cursor.rowcount != len(message_ids):
                    raise ValueError("conversation_message_not_pending")

    def _assert_verified_conversation_checkpoint_in_transaction(
        self, task_id: str, *, checkpoint_type: str | None = None
    ) -> None:
        """Require the persisted latest checkpoint to prove one completed action."""
        from private_agent.core.task_state import StateIntegrityError, TaskState

        checkpoint = self.db.execute(
            "SELECT checkpoint_type FROM task_checkpoints WHERE task_id = ? ORDER BY sequence DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        row = self.db.execute("SELECT state_json FROM task_states WHERE task_id = ?", (task_id,)).fetchone()
        if checkpoint is None or str(checkpoint["checkpoint_type"]) not in {"AFTER_VERIFICATION", "APPROVAL_REVALIDATED"} or row is None:
            raise ValueError("conversation_safe_checkpoint_required")
        try:
            state = TaskState.from_dict(json.loads(str(row["state_json"])))
        except (StateIntegrityError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("conversation_task_state_untrusted") from exc
        if checkpoint_type == "BUG_BOUNTY_FOLLOWUP_ACKNOWLEDGED" and state.context.get("task_kind") != "bug_bounty":
            raise ValueError("conversation_safe_checkpoint_required")
        if state.status != "READY" or not state.current_step:
            raise ValueError("conversation_safe_checkpoint_required")
        verified = False
        for key, record in state.action_records.items():
            if not isinstance(record, dict) or record.get("step_id") != state.current_step:
                continue
            observation = state.observations.get(key)
            verification = state.verification_results.get(key)
            output = record.get("output")
            if (
                key in state.completed_actions
                and record.get("action_key") == key
                and record.get("input_fingerprint") == key
                and record.get("status") == "VERIFIED"
                and record.get("execution_status") == "success"
                and isinstance(output, dict)
                and output.get("status") == "success"
                and isinstance(observation, dict)
                and record.get("observation") == observation
                and observation.get("execution_status") == "success"
                and observation.get("observed_error") is None
                and isinstance(verification, dict)
                and record.get("verification") == verification
                and verification.get("status") == "VERIFIED"
                and verification.get("verified") is True
            ):
                verified = True
                break
        if not verified:
            raise ValueError("conversation_verified_action_required")

    def create_queued_task(
        self,
        goal: str,
        *,
        initial_message: str | None = None,
        task_id: str | None = None,
        idempotency_key: str | None = None,
        request_fingerprint: str | None = None,
        context: dict[str, Any] | None = None,
        max_attempts: int = 3,
        owner_user_id: str | None = None,
    ) -> dict[str, Any]:
        """Atomically create a QUEUED state, owner, queue, first message and idempotency record."""
        from private_agent.core.task_state import TaskState

        if not isinstance(goal, str):
            raise ValueError("goal_must_be_string")
        goal = goal.strip()
        if not goal:
            raise ValueError("goal_required")
        if len(goal) > MAX_TASK_GOAL_LENGTH:
            raise ValueError("goal_too_long")
        if initial_message is not None:
            if not isinstance(initial_message, str):
                raise ValueError("task_message_must_be_string")
            initial_message = initial_message.strip()
            if not initial_message:
                raise ValueError("task_message_required")
            if len(initial_message) > MAX_TASK_MESSAGE_LENGTH:
                raise ValueError("task_message_too_long")
        if max_attempts < 1:
            raise ValueError("max_attempts_invalid")
        if idempotency_key is not None and not request_fingerprint:
            raise ValueError("request_fingerprint_required")
        if context is not None and not isinstance(context, dict):
            raise ValueError("task_context_must_be_object")
        if owner_user_id is not None and (not isinstance(owner_user_id, str) or not owner_user_id.strip() or len(owner_user_id) > 256):
            raise ValueError("task_owner_invalid")
        raw_context = dict(context or {})
        context_payload = safe_json(raw_context) if raw_context.get("task_kind") == "bug_bounty" else sanitize_value(raw_context)
        try:
            context_json = json.dumps(context_payload, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("task_context_not_json_safe") from exc
        if len(context_json.encode("utf-8")) > 128 * 1024:
            raise ValueError("task_context_too_large")

        self._db_lock.acquire()
        try:
            self.db.execute("BEGIN IMMEDIATE")
            if idempotency_key:
                existing = self.db.execute(
                    "SELECT request_fingerprint, task_id FROM api_idempotency_keys WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if existing is not None:
                    if existing["request_fingerprint"] != request_fingerprint:
                        raise ValueError("idempotency_conflict")
                    task = self.db.execute(
                        "SELECT 1 FROM task_states WHERE task_id = ?",
                        (existing["task_id"],),
                    ).fetchone()
                    queue = self.db.execute(
                        "SELECT 1 FROM task_queue WHERE task_id = ?",
                        (existing["task_id"],),
                    ).fetchone()
                    if task is None or queue is None:
                        raise UnsupportedSchemaVersion("idempotency_task_queue_missing")
                    if owner_user_id is not None:
                        owner = self.db.execute(
                            "SELECT user_id FROM task_owners WHERE task_id = ?", (existing["task_id"],)
                        ).fetchone()
                        if owner is None or str(owner["user_id"]) != owner_user_id:
                            raise ValueError("idempotency_conflict")
                    self.db.commit()
                    return {"task_id": str(existing["task_id"]), "replayed": True}

            state = TaskState.create_queued(task_id or str(uuid.uuid4()), goal, context=context_payload)
            payload = state.to_dict()
            stamp = now()
            self._persist_task_state_in_transaction(payload, "TASK_QUEUED")
            self.db.execute(
                """INSERT INTO task_queue
                (task_id, queue_status, attempt_count, max_attempts, available_at,
                 lease_id, worker_id, lease_acquired_at, lease_expires_at,
                 last_started_at, last_finished_at, last_error_code, last_error_message,
                 created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    state.task_id,
                    "QUEUED",
                    0,
                    int(max_attempts),
                    stamp,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    state.created_at,
                    stamp,
                ),
            )
            if owner_user_id is not None:
                self.db.execute(
                    "INSERT INTO task_owners (task_id, user_id, created_at) VALUES (?, ?, ?)",
                    (state.task_id, owner_user_id, stamp),
                )
            if initial_message is not None:
                initial_sequence = self._allocate_task_conversation_sequence_in_transaction(state.task_id)
                self.db.execute(
                    """INSERT INTO task_messages
                    (message_id, task_id, sequence, role, message_kind, content, status,
                     idempotency_key, request_fingerprint, created_at, consumed_at, consumed_checkpoint)
                    VALUES (?, ?, ?, 'USER', 'INITIAL', ?, 'CONSUMED', NULL, '', ?, ?, 'TASK_QUEUED')""",
                    (str(uuid.uuid4()), state.task_id, initial_sequence, initial_message, stamp, stamp),
                )
            if owner_user_id is not None:
                self._persist_task_result_event_in_transaction(payload, "QUEUED")
            if idempotency_key:
                self.db.execute(
                    """INSERT INTO api_idempotency_keys
                    (idempotency_key, request_fingerprint, task_id, response_json, created_at)
                    VALUES (?,?,?,?,?)""",
                    (
                        idempotency_key,
                        str(request_fingerprint),
                        state.task_id,
                        json.dumps({"task_id": state.task_id, "status": "QUEUED"}, sort_keys=True),
                        stamp,
                    ),
                )
            self.db.commit()
            return {"task_id": state.task_id, "replayed": False}
        except Exception:
            self.db.rollback()
            raise
        finally:
            self._db_lock.release()

    def task_messages(
        self,
        task_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return only user-authored messages, optionally enforcing task ownership."""
        with self._db_lock:
            if owner_user_id is None:
                rows = self.db.execute(
                    "SELECT message_id, task_id, sequence, role, message_kind, content, status, "
                    "created_at, consumed_at, consumed_checkpoint FROM task_messages "
                    "WHERE task_id = ? ORDER BY sequence",
                    (str(task_id),),
                ).fetchall()
            else:
                rows = self.db.execute(
                    "SELECT m.message_id, m.task_id, m.sequence, m.role, m.message_kind, m.content, "
                    "m.status, m.created_at, m.consumed_at, m.consumed_checkpoint "
                    "FROM task_messages AS m JOIN task_owners AS o ON o.task_id = m.task_id "
                    "WHERE m.task_id = ? AND o.user_id = ? ORDER BY m.sequence",
                    (str(task_id), str(owner_user_id)),
                ).fetchall()
        return [dict(row) for row in rows]

    def task_result_events(
        self,
        task_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return immutable result summaries only while the task-owner binding matches."""
        query = (
            "SELECT e.event_id, e.task_id, e.owner_user_id, e.sequence, e.role, e.message_kind, "
            "e.task_status, e.verification_status, e.result_available, e.verified_action_count, "
            "e.failed_action_count, e.blocked_action_count, e.pending_action_count, "
            "e.approval_pending_count, e.approval_approved_count, e.approval_denied_count, "
            "e.approval_expired_count, e.approval_cancelled_count, e.result_digest, "
            "e.planning_mode, e.plan_version, e.created_at "
            "FROM task_result_events AS e JOIN task_owners AS o "
            "ON o.task_id = e.task_id AND o.user_id = e.owner_user_id "
            "WHERE e.task_id = ?"
        )
        params: tuple[Any, ...] = (str(task_id),)
        if owner_user_id is not None:
            query += " AND e.owner_user_id = ?"
            params = (str(task_id), str(owner_user_id))
        query += " ORDER BY e.sequence"
        with self._db_lock:
            rows = self.db.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def task_conversation_items(
        self,
        task_id: str,
        *,
        owner_user_id: str | None = None,
        after_sequence: int = 0,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Read one server-ordered page; the cursor is exclusive and append-only."""
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int) or after_sequence < 0:
            raise ValueError("conversation_cursor_invalid")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500
        ):
            raise ValueError("conversation_limit_invalid")
        task_id = str(task_id)
        message_owner = ""
        event_owner = ""
        params: list[Any] = [task_id]
        if owner_user_id is not None:
            message_owner = " AND EXISTS (SELECT 1 FROM task_owners AS mo " \
                "WHERE mo.task_id = m.task_id AND mo.user_id = ?)"
            event_owner = " AND e.owner_user_id = ?"
        if owner_user_id is not None:
            params.append(str(owner_user_id))
        params.append(task_id)
        if owner_user_id is not None:
            params.append(str(owner_user_id))
        params.append(after_sequence)
        query = (
            "SELECT * FROM ("
            "SELECT 'USER_MESSAGE' AS item_type, m.message_id AS item_id, m.sequence, m.task_id, "
            "m.role, m.message_kind, m.content, m.status, m.created_at, m.consumed_at, "
            "m.consumed_checkpoint, NULL AS task_status, NULL AS verification_status, "
            "NULL AS result_available, NULL AS verified_action_count, NULL AS failed_action_count, "
            "NULL AS blocked_action_count, NULL AS pending_action_count, NULL AS approval_pending_count, "
            "NULL AS approval_approved_count, NULL AS approval_denied_count, NULL AS approval_expired_count, "
            "NULL AS approval_cancelled_count, NULL AS result_digest, "
            "NULL AS planning_mode, NULL AS plan_version "
            "FROM task_messages AS m WHERE m.task_id = ?" + message_owner + " UNION ALL "
            "SELECT 'SYSTEM_RESULT' AS item_type, e.event_id AS item_id, e.sequence, e.task_id, "
            "e.role, e.message_kind, NULL AS content, NULL AS status, e.created_at, NULL AS consumed_at, "
            "NULL AS consumed_checkpoint, e.task_status, e.verification_status, e.result_available, "
            "e.verified_action_count, e.failed_action_count, e.blocked_action_count, e.pending_action_count, "
            "e.approval_pending_count, e.approval_approved_count, e.approval_denied_count, "
            "e.approval_expired_count, e.approval_cancelled_count, e.result_digest, "
            "e.planning_mode, e.plan_version "
            "FROM task_result_events AS e JOIN task_owners AS eo "
            "ON eo.task_id = e.task_id AND eo.user_id = e.owner_user_id "
            "WHERE e.task_id = ?" + event_owner + ") AS conversation_items "
            "WHERE sequence > ? ORDER BY sequence"
        )
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit + 1)
        with self._db_lock:
            rows = [dict(row) for row in self.db.execute(query, tuple(params)).fetchall()]
        has_more = limit is not None and len(rows) > limit
        if has_more:
            rows = rows[:limit]
        items: list[dict[str, Any]] = []
        for row in rows:
            sequence = int(row["sequence"])
            if row["item_type"] == "USER_MESSAGE":
                message = {
                    "message_id": str(row["item_id"]), "task_id": task_id, "sequence": sequence,
                    "role": str(row["role"]), "message_kind": str(row["message_kind"]),
                    "content": str(row["content"]), "status": str(row["status"]),
                    "created_at": str(row["created_at"]), "consumed_at": str(row["consumed_at"]),
                    "consumed_checkpoint": str(row["consumed_checkpoint"]),
                }
                items.append({"item_type": "USER_MESSAGE", "sequence": sequence, "message": message})
            else:
                result_event = {
                    "event_id": str(row["item_id"]), "task_id": task_id, "sequence": sequence,
                    "role": str(row["role"]), "message_kind": str(row["message_kind"]),
                    "task_status": str(row["task_status"]),
                    "verification_status": str(row["verification_status"]),
                    "result_available": bool(row["result_available"]),
                    "verified_action_count": int(row["verified_action_count"]),
                    "failed_action_count": int(row["failed_action_count"]),
                    "blocked_action_count": int(row["blocked_action_count"]),
                    "pending_action_count": int(row["pending_action_count"]),
                    "approval_pending_count": int(row["approval_pending_count"]),
                    "approval_approved_count": int(row["approval_approved_count"]),
                    "approval_denied_count": int(row["approval_denied_count"]),
                    "approval_expired_count": int(row["approval_expired_count"]),
                    "approval_cancelled_count": int(row["approval_cancelled_count"]),
                    "result_digest": str(row["result_digest"]),
                    "planning_mode": str(row["planning_mode"]),
                    "plan_version": int(row["plan_version"]),
                    "created_at": str(row["created_at"]),
                }
                items.append({"item_type": "SYSTEM_RESULT", "sequence": sequence, "result_event": result_event})
        return {
            "items": items,
            "has_more": bool(has_more),
            "next_cursor": int(items[-1]["sequence"]) if has_more and items else None,
        }

    def claim_next_task_assistant_reply_event(
        self,
        worker_id: str,
        *,
        lease_ttl_seconds: int = 300,
        task_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Lease the latest unanswered user-facing result event across restarts."""
        worker_id = str(worker_id).strip()
        if not worker_id or len(worker_id) > 200:
            raise ValueError("assistant_reply_worker_id_invalid")
        if isinstance(lease_ttl_seconds, bool) or int(lease_ttl_seconds) < 1:
            raise ValueError("assistant_reply_lease_invalid")
        current = now()
        expires = (datetime.now(timezone.utc) + timedelta(seconds=int(lease_ttl_seconds))).isoformat()
        conditions = [
            "(e.task_status IN ('COMPLETED', 'FAILED', 'BLOCKED', 'EXECUTION_UNKNOWN', 'WAITING_APPROVAL') "
            "OR e.verification_status IN ('FAILED', 'BLOCKED', 'EXECUTION_UNKNOWN', 'WAITING_APPROVAL'))",
            "NOT EXISTS (SELECT 1 FROM task_assistant_replies AS r WHERE r.event_id = e.event_id)",
            "NOT EXISTS (SELECT 1 FROM task_result_events AS newer WHERE newer.task_id = e.task_id AND newer.sequence > e.sequence)",
            "NOT EXISTS (SELECT 1 FROM task_assistant_reply_claims AS c WHERE c.event_id = e.event_id AND c.lease_expires_at > ?)",
        ]
        params: list[Any] = [current]
        if task_id is not None:
            conditions.append("e.task_id = ?")
            params.append(str(task_id))
        with self._db_lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                row = self.db.execute(
                    "SELECT e.* FROM task_result_events AS e WHERE " + " AND ".join(conditions)
                    + " ORDER BY e.created_at, e.sequence LIMIT 1",
                    tuple(params),
                ).fetchone()
                if row is None:
                    self.db.commit()
                    return None
                event = dict(row)
                existing = self.db.execute(
                    "SELECT attempt_count, created_at FROM task_assistant_reply_claims WHERE event_id = ?",
                    (str(event["event_id"]),),
                ).fetchone()
                attempt_count = int(existing["attempt_count"]) + 1 if existing is not None else 1
                created_at = str(existing["created_at"]) if existing is not None else current
                self.db.execute(
                    """INSERT INTO task_assistant_reply_claims
                    (event_id, task_id, owner_user_id, lease_owner, lease_expires_at, attempt_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(event_id) DO UPDATE SET task_id = excluded.task_id,
                      owner_user_id = excluded.owner_user_id, lease_owner = excluded.lease_owner,
                      lease_expires_at = excluded.lease_expires_at, attempt_count = excluded.attempt_count,
                      updated_at = excluded.updated_at""",
                    (
                        str(event["event_id"]), str(event["task_id"]), str(event["owner_user_id"]),
                        worker_id, expires, attempt_count, created_at, current,
                    ),
                )
                self.db.commit()
                event["result_available"] = bool(event["result_available"])
                return event
            except Exception:
                self.db.rollback()
                raise

    def save_task_assistant_reply(
        self,
        event_id: str,
        *,
        worker_id: str,
        reply_code: str,
        content: str,
        generation_mode: str,
    ) -> dict[str, Any]:
        """Persist only a validated fixed phrase for the currently leased result."""
        from private_agent.core.assistant_reply import render_reply_for_event

        with self._db_lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                event_row = self.db.execute(
                    "SELECT * FROM task_result_events WHERE event_id = ?", (str(event_id),)
                ).fetchone()
                if event_row is None:
                    raise ValueError("assistant_reply_event_missing")
                event = dict(event_row)
                event["result_available"] = bool(event["result_available"])
                expected_content = render_reply_for_event(event, str(reply_code))
                if content != expected_content:
                    raise ValueError("assistant_reply_content_not_allowlisted")
                if generation_mode not in {"MODEL_VARIANT", "DETERMINISTIC_FALLBACK"}:
                    raise ValueError("assistant_reply_generation_mode_invalid")
                claim = self.db.execute(
                    "SELECT lease_owner, lease_expires_at FROM task_assistant_reply_claims WHERE event_id = ?",
                    (str(event_id),),
                ).fetchone()
                if claim is None or str(claim["lease_owner"]) != str(worker_id) or str(claim["lease_expires_at"]) <= now():
                    raise QueueLeaseLost("assistant_reply_claim_lost")
                reply_key = hashlib.sha256(
                    f"assistant-reply:{event['event_id']}:{event['result_digest']}".encode("utf-8")
                ).hexdigest()
                reply_id = hashlib.sha256(f"assistant-reply-id:{reply_key}".encode("utf-8")).hexdigest()
                created_at = now()
                self.db.execute(
                    """INSERT OR IGNORE INTO task_assistant_replies
                    (reply_id, reply_key, event_id, task_id, owner_user_id, source_sequence, source_digest,
                     role, message_kind, reply_code, content, generation_mode, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'ASSISTANT', 'RESULT_REPLY', ?, ?, ?, ?)""",
                    (
                        reply_id, reply_key, str(event["event_id"]), str(event["task_id"]),
                        str(event["owner_user_id"]), int(event["sequence"]), str(event["result_digest"]),
                        str(reply_code), expected_content, str(generation_mode), created_at,
                    ),
                )
                self.db.execute(
                    "DELETE FROM task_assistant_reply_claims WHERE event_id = ? AND lease_owner = ?",
                    (str(event_id), str(worker_id)),
                )
                saved = self.db.execute(
                    "SELECT * FROM task_assistant_replies WHERE event_id = ?", (str(event_id),)
                ).fetchone()
                if saved is None:
                    raise RuntimeError("assistant_reply_persistence_failed")
                self.db.commit()
                return dict(saved)
            except Exception:
                self.db.rollback()
                raise

    def task_assistant_replies(
        self,
        task_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read server-created replies only when both task and source-event owners match."""
        query = (
            "SELECT r.reply_id, r.task_id, r.owner_user_id, r.event_id, r.source_sequence, r.source_digest, "
            "r.role, r.message_kind, r.reply_code, r.content, r.generation_mode, r.created_at "
            "FROM task_assistant_replies AS r JOIN task_owners AS o "
            "ON o.task_id = r.task_id AND o.user_id = r.owner_user_id "
            "JOIN task_result_events AS e ON e.event_id = r.event_id AND e.task_id = r.task_id "
            "AND e.owner_user_id = r.owner_user_id AND e.sequence = r.source_sequence "
            "AND e.result_digest = r.source_digest WHERE r.task_id = ?"
        )
        params: tuple[Any, ...] = (str(task_id),)
        if owner_user_id is not None:
            query += " AND r.owner_user_id = ?"
            params = (str(task_id), str(owner_user_id))
        query += " ORDER BY r.source_sequence, r.created_at"
        with self._db_lock:
            rows = self.db.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def append_task_message(
        self,
        task_id: str,
        content: str,
        *,
        owner_user_id: str | None = None,
        idempotency_key: str | None = None,
        request_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        """Persist a follow-up as pending without changing task state or plan."""
        from private_agent.core.task_state import StateIntegrityError, TERMINAL_STATUSES, TaskState

        task_id = str(task_id)
        if not isinstance(content, str):
            raise ValueError("task_message_must_be_string")
        content = content.strip()
        if not content:
            raise ValueError("task_message_required")
        if len(content) > MAX_TASK_MESSAGE_LENGTH:
            raise ValueError("task_message_too_long")
        if owner_user_id is not None and (not isinstance(owner_user_id, str) or not owner_user_id.strip() or len(owner_user_id) > 256):
            raise ValueError("task_owner_invalid")
        if idempotency_key is not None and not request_fingerprint:
            raise ValueError("message_request_fingerprint_required")
        self._db_lock.acquire()
        try:
            self.db.execute("BEGIN IMMEDIATE")
            owner = self.db.execute("SELECT user_id FROM task_owners WHERE task_id = ?", (task_id,)).fetchone()
            if owner_user_id is not None and (owner is None or str(owner["user_id"]) != owner_user_id):
                raise KeyError("task_not_found")
            task_row = self.db.execute("SELECT state_json FROM task_states WHERE task_id = ?", (task_id,)).fetchone()
            queue_row = self.db.execute("SELECT queue_status FROM task_queue WHERE task_id = ?", (task_id,)).fetchone()
            if task_row is None or queue_row is None:
                raise KeyError("task_not_found")
            try:
                state = TaskState.from_dict(json.loads(str(task_row["state_json"])))
            except (StateIntegrityError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("task_state_untrusted") from exc
            queue_status = str(queue_row["queue_status"])
            if idempotency_key is not None:
                existing = self.db.execute(
                    "SELECT * FROM task_messages WHERE task_id = ? AND idempotency_key = ?",
                    (task_id, idempotency_key),
                ).fetchone()
                if existing is not None:
                    if str(existing["request_fingerprint"]) != str(request_fingerprint):
                        raise ValueError("message_idempotency_conflict")
                    if (
                        str(existing["status"]) == "PENDING"
                        and (state.status in TERMINAL_STATUSES or queue_status in {"COMPLETED", "FAILED", "BLOCKED"})
                    ):
                        raise ValueError("conversation_terminal")
                    result = dict(existing)
                    result["replayed"] = True
                    self.db.commit()
                    return result
            if state.status == "EXECUTION_UNKNOWN" or queue_status == "EXECUTION_UNKNOWN":
                raise ValueError("conversation_execution_unknown")
            if state.status in TERMINAL_STATUSES or queue_status in {"COMPLETED", "FAILED", "BLOCKED"}:
                raise ValueError("conversation_terminal")
            eligible = (
                state.status == "QUEUED" and queue_status == "QUEUED"
            ) or (
                state.status == "WAITING_APPROVAL" and queue_status in {"WAITING_APPROVAL", "QUEUED", "LEASED"}
            ) or (
                queue_status == "LEASED" and state.status in {"PLANNING", "EXECUTING", "OBSERVING", "VERIFYING"}
            )
            if state.status == "READY" and queue_status == "LEASED" and state.pending_actions:
                eligible = any(
                    isinstance(record, dict)
                    and record.get("step_id") == state.current_step
                    and record.get("status") == "VERIFIED"
                    and key in state.completed_actions
                    for key, record in state.action_records.items()
                )
            if not eligible:
                raise ValueError("conversation_not_accepting_messages")
            totals = self.db.execute(
                "SELECT COUNT(*) AS message_count, COALESCE(SUM(length(content)), 0) AS total_chars "
                "FROM task_messages WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if int(totals["message_count"] or 0) >= MAX_TASK_CONVERSATION_MESSAGES:
                raise ValueError("conversation_message_limit")
            if int(totals["total_chars"] or 0) + len(content) > MAX_TASK_CONVERSATION_CHARS:
                raise ValueError("conversation_size_limit")
            stamp = now()
            message_id = str(uuid.uuid4())
            sequence = self._allocate_task_conversation_sequence_in_transaction(task_id)
            self.db.execute(
                """INSERT INTO task_messages
                (message_id, task_id, sequence, role, message_kind, content, status,
                 idempotency_key, request_fingerprint, created_at, consumed_at, consumed_checkpoint)
                VALUES (?, ?, ?, 'USER', 'FOLLOW_UP', ?, 'PENDING', ?, ?, ?, '', '')""",
                (message_id, task_id, sequence, content, idempotency_key, str(request_fingerprint or ""), stamp),
            )
            result = dict(self.db.execute("SELECT * FROM task_messages WHERE message_id = ?", (message_id,)).fetchone())
            result["replayed"] = False
            self.db.commit()
            return result
        except Exception:
            self.db.rollback()
            raise
        finally:
            self._db_lock.release()

    def pending_task_messages_for_worker(self, task_id: str) -> list[dict[str, Any]]:
        """Read pending follow-ups only under the active worker lease at a verified checkpoint."""
        if not self.has_active_lease_context(task_id):
            raise QueueLeaseLost("conversation_worker_lease_required")
        with self._db_lock:
            self._assert_active_lease(str(task_id))
            self._assert_verified_conversation_checkpoint_in_transaction(str(task_id))
            rows = self.db.execute(
                "SELECT message_id, task_id, sequence, content, created_at FROM task_messages "
                "WHERE task_id = ? AND message_kind = 'FOLLOW_UP' AND status = 'PENDING' ORDER BY sequence",
                (str(task_id),),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _queue_status_for_task_status(status: str) -> str:
        return {
            "COMPLETED": "COMPLETED",
            "FAILED": "FAILED",
            "BLOCKED": "BLOCKED",
            "WAITING_APPROVAL": "WAITING_APPROVAL",
            "EXECUTION_UNKNOWN": "EXECUTION_UNKNOWN",
        }.get(status, "FAILED")

    def _mark_expired_lease_in_transaction(self, row: sqlite3.Row, stamp: str) -> dict[str, Any]:
        """Fence an expired worker and conservatively persist an unknown outcome."""
        from private_agent.core.task_state import StateIntegrityError, TERMINAL_STATUSES, TaskState, TaskStopReasonCode

        task_id = str(row["task_id"])
        state_row = self.db.execute(
            "SELECT state_json FROM task_states WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        queue_status = "EXECUTION_UNKNOWN"
        error_code = "lease_expired_execution_unknown"
        error_message = "Worker lease expired; automatic retry is disabled for safety"
        if state_row is not None:
            try:
                state = TaskState.from_dict(json.loads(state_row["state_json"]))
            except (TypeError, ValueError, KeyError, StateIntegrityError) as exc:
                state = None
                queue_status = "BLOCKED"
                error_code = "malformed_task_state"
                error_message = f"Persisted task state failed integrity validation: {type(exc).__name__}"
            if state is None:
                queue_status = "BLOCKED"
                error_code = "malformed_task_state"
            elif state.status in TERMINAL_STATUSES:
                queue_status = self._queue_status_for_task_status(state.status)
                error_code = "lease_expired_after_terminal_state"
            elif state.status == "WAITING_APPROVAL":
                # The Orchestrator can finish planning with approval pending
                # before the worker manages to release its lease.
                queue_status = "WAITING_APPROVAL"
                error_code = "lease_expired_waiting_approval"
            else:
                state.recovery_state = {
                    **state.recovery_state,
                    "reason": "worker_lease_expired",
                    "external_side_effect_may_have_occurred": True,
                }
                if state.status != "EXECUTION_UNKNOWN":
                    state.checkpoint(
                        "EXECUTION_UNKNOWN",
                        status="EXECUTION_UNKNOWN",
                        phase="EXECUTION_UNKNOWN",
                        successful=False,
                    )
                else:
                    state.touch()
                state.set_stop_reason(
                    TaskStopReasonCode.EXECUTION_UNKNOWN,
                    step_id=state.current_step or None,
                    action_id=next(
                        (key for key, record in state.action_records.items() if record.get("step_id") == state.current_step),
                        None,
                    ),
                )
                self._persist_task_state_in_transaction(
                    state.to_dict(),
                    "EXECUTION_UNKNOWN",
                    task_status="EXECUTION_UNKNOWN",
                    enforce_lease=False,
                )

        updated = self.db.execute(
            """UPDATE task_queue
               SET queue_status = ?, lease_id = '', worker_id = '',
                   lease_acquired_at = '', lease_expires_at = '',
                   last_finished_at = ?, last_error_code = ?,
                   last_error_message = ?, updated_at = ?
             WHERE task_id = ? AND queue_status = 'LEASED' AND lease_id = ?""",
            (
                queue_status,
                stamp,
                error_code,
                error_message,
                stamp,
                task_id,
                str(row["lease_id"]),
            ),
        )
        if updated.rowcount != 1:
            raise QueueLeaseLost(f"expired_lease_fencing_failed:{task_id}")
        return {
            "kind": "EXECUTION_UNKNOWN" if queue_status == "EXECUTION_UNKNOWN" else queue_status,
            "task_id": task_id,
            "queue_status": queue_status,
            "error_code": error_code,
        }

    def claim_next_task(self, worker_id: str, *, lease_ttl_seconds: int = 60) -> dict[str, Any] | None:
        """Atomically claim one runnable task or recover one expired lease."""
        worker_id = str(worker_id).strip()
        if not worker_id:
            raise ValueError("worker_id_required")
        if lease_ttl_seconds < 1:
            raise ValueError("lease_ttl_invalid")
        stamp = now()
        expires = (datetime.now(timezone.utc) + timedelta(seconds=int(lease_ttl_seconds))).isoformat()
        self._db_lock.acquire()
        try:
            self.assert_worker_claim_allowed()
            self.db.execute("BEGIN IMMEDIATE")
            expired = self.db.execute(
                """SELECT * FROM task_queue
                   WHERE queue_status = 'LEASED'
                     AND lease_expires_at <> ''
                     AND lease_expires_at <= ?
                   ORDER BY lease_expires_at, task_id
                   LIMIT 1""",
                (stamp,),
            ).fetchone()
            if expired is not None:
                recovered = self._mark_expired_lease_in_transaction(expired, stamp)
                self.db.commit()
                return recovered

            candidate = self.db.execute(
                """SELECT * FROM task_queue
                   WHERE queue_status IN ('QUEUED', 'RETRY_WAIT')
                     AND available_at <= ?
                     AND attempt_count < max_attempts
                   ORDER BY available_at, created_at, task_id
                   LIMIT 1""",
                (stamp,),
            ).fetchone()
            if candidate is None:
                self.db.commit()
                return None

            lease_id = str(uuid.uuid4())
            updated = self.db.execute(
                """UPDATE task_queue
                   SET queue_status = 'LEASED', attempt_count = attempt_count + 1,
                       lease_id = ?, worker_id = ?, lease_acquired_at = ?,
                       lease_expires_at = ?, last_started_at = ?, updated_at = ?
                 WHERE task_id = ? AND queue_status IN ('QUEUED', 'RETRY_WAIT')
                   AND available_at <= ? AND attempt_count < max_attempts""",
                (lease_id, worker_id, stamp, expires, stamp, stamp, candidate["task_id"], stamp),
            )
            if updated.rowcount != 1:
                raise QueueLeaseLost(f"queue_claim_lost:{candidate['task_id']}")
            task_row = self.db.execute(
                "SELECT state_json FROM task_states WHERE task_id = ?",
                (candidate["task_id"],),
            ).fetchone()
            if task_row is None:
                raise UnsupportedSchemaVersion(f"queue_task_state_missing:{candidate['task_id']}")
            try:
                from private_agent.core.task_state import StateIntegrityError, TaskState

                task_payload = json.loads(task_row["state_json"])
                state = TaskState.from_dict(task_payload)
                goal = state.goal
                if state.status in {"EXECUTION_UNKNOWN", "BLOCKED", "FAILED", "COMPLETED"}:
                    queue_status = self._queue_status_for_task_status(state.status)
                    self.db.execute(
                        "UPDATE task_queue SET queue_status = ?, lease_id = '', worker_id = '', "
                        "lease_acquired_at = '', lease_expires_at = '', last_error_code = ?, updated_at = ? "
                        "WHERE task_id = ?",
                        (queue_status, "queue_state_not_executable", now(), candidate["task_id"]),
                    )
                    self.db.commit()
                    return {
                        "kind": queue_status,
                        "task_id": str(candidate["task_id"]),
                        "queue_status": queue_status,
                        "error_code": "queue_state_not_executable",
                    }
                if state.status not in {"CREATED", "QUEUED", "READY", "WAITING_APPROVAL"}:
                    raise StateIntegrityError(f"queue_state_not_executable:{state.status}")
            except (TypeError, ValueError, KeyError, StateIntegrityError) as exc:
                stamp = now()
                self.db.execute(
                    "UPDATE task_queue SET queue_status = 'BLOCKED', lease_id = '', worker_id = '', "
                    "lease_acquired_at = '', lease_expires_at = '', last_finished_at = ?, "
                    "last_error_code = ?, last_error_message = ?, updated_at = ? WHERE task_id = ?",
                    (
                        stamp,
                        "malformed_task_state",
                        "Persisted task state failed integrity validation; operator review required",
                        stamp,
                        candidate["task_id"],
                    ),
                )
                self.db.execute("UPDATE tasks SET status = 'blocked', updated_at = ? WHERE id = ?", (stamp, candidate["task_id"]))
                self.db.commit()
                return {
                    "kind": "BLOCKED",
                    "task_id": str(candidate["task_id"]),
                    "queue_status": "BLOCKED",
                    "error_code": "malformed_task_state",
                    "error": str(exc),
                }
            self.db.commit()
            return {
                "kind": "CLAIMED",
                "task_id": str(candidate["task_id"]),
                "goal": goal,
                "lease_id": lease_id,
                "worker_id": worker_id,
                "attempt_count": int(candidate["attempt_count"]) + 1,
                "max_attempts": int(candidate["max_attempts"]),
                "lease_expires_at": expires,
            }
        except Exception:
            self.db.rollback()
            raise
        finally:
            self._db_lock.release()

    def renew_task_lease(
        self,
        task_id: str,
        lease_id: str,
        worker_id: str,
        *,
        lease_ttl_seconds: int = 60,
    ) -> dict[str, Any]:
        """Renew only the currently owned, non-expired lease."""
        if lease_ttl_seconds < 1:
            raise ValueError("lease_ttl_invalid")
        stamp = now()
        expires = (datetime.now(timezone.utc) + timedelta(seconds=int(lease_ttl_seconds))).isoformat()
        self._db_lock.acquire()
        try:
            self.db.execute("BEGIN IMMEDIATE")
            updated = self.db.execute(
                """UPDATE task_queue
                   SET lease_expires_at = ?, updated_at = ?
                 WHERE task_id = ? AND queue_status = 'LEASED'
                   AND lease_id = ? AND worker_id = ?
                   AND lease_expires_at <> '' AND lease_expires_at > ?""",
                (expires, stamp, task_id, lease_id, worker_id, stamp),
            )
            if updated.rowcount != 1:
                raise QueueLeaseLost(f"queue_lease_lost:{task_id}")
            self.db.commit()
            return {"task_id": str(task_id), "lease_id": str(lease_id), "worker_id": str(worker_id), "lease_expires_at": expires}
        except Exception:
            self.db.rollback()
            raise
        finally:
            self._db_lock.release()

    def finish_task_claim(
        self,
        claim: dict[str, Any],
        queue_status: str,
        *,
        error_code: str = "",
        error_message: str = "",
        state_status: str | None = None,
    ) -> dict[str, Any]:
        """Fence and close a claim, optionally persisting a worker failure state."""
        allowed = {"WAITING_APPROVAL", "EXECUTION_UNKNOWN", "COMPLETED", "FAILED", "BLOCKED"}
        if queue_status not in allowed:
            raise ValueError(f"invalid_queue_finish_status:{queue_status}")
        task_id = str(claim["task_id"])
        lease_id = str(claim["lease_id"])
        worker_id = str(claim["worker_id"])
        stamp = now()
        self._db_lock.acquire()
        try:
            self.db.execute("BEGIN IMMEDIATE")
            current = self.db.execute(
                "SELECT * FROM task_queue WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if current is None or current["queue_status"] != "LEASED" or current["lease_id"] != lease_id or current["worker_id"] != worker_id:
                raise QueueLeaseLost(f"queue_lease_lost:{task_id}")
            if not current["lease_expires_at"] or _parse_queue_time(current["lease_expires_at"]) <= datetime.now(timezone.utc):
                raise QueueLeaseLost(f"queue_lease_expired:{task_id}")

            if state_status is not None:
                from private_agent.core.task_state import TaskState, TaskStopReasonCode, TERMINAL_STATUSES

                state_row = self.db.execute(
                    "SELECT state_json FROM task_states WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if state_row is not None:
                    state = TaskState.from_dict(json.loads(state_row["state_json"]))
                    if state_status == "EXECUTION_UNKNOWN":
                        if state.status != "EXECUTION_UNKNOWN":
                            state.recovery_state = {
                                **state.recovery_state,
                                "reason": "worker_execution_interrupted",
                                "external_side_effect_may_have_occurred": True,
                            }
                            state.checkpoint(
                                "EXECUTION_UNKNOWN",
                                status="EXECUTION_UNKNOWN",
                                phase="EXECUTION_UNKNOWN",
                                successful=False,
                            )
                        state.set_stop_reason(
                            TaskStopReasonCode.EXECUTION_UNKNOWN,
                            step_id=state.current_step or None,
                            action_id=next(
                                (key for key, record in state.action_records.items() if record.get("step_id") == state.current_step),
                                None,
                            ),
                        )
                    elif state_status == "FAILED" and state.status not in TERMINAL_STATUSES:
                        state.result = {
                            **state.result,
                            "final_status": "FAILED",
                            "errors": list(state.result.get("errors", [])) + [error_message or error_code or "worker_failed"],
                        }
                        state.checkpoint("WORKER_FAILED", status="FAILED", phase="WORKER_FAILED", successful=False)
                        state.set_stop_reason(
                            TaskStopReasonCode.WORKER_FAILED,
                            step_id=state.current_step or None,
                            action_id=next(
                                (key for key, record in state.action_records.items() if record.get("step_id") == state.current_step),
                                None,
                            ),
                        )
                    self._persist_task_state_in_transaction(
                        state.to_dict(),
                        "EXECUTION_UNKNOWN" if state_status == "EXECUTION_UNKNOWN" else "WORKER_FAILED",
                        task_status=state.status,
                        enforce_lease=False,
                    )

            updated = self.db.execute(
                """UPDATE task_queue
                   SET queue_status = ?, lease_id = '', worker_id = '',
                       lease_acquired_at = '', lease_expires_at = '',
                       last_finished_at = ?, last_error_code = ?,
                       last_error_message = ?, updated_at = ?
                 WHERE task_id = ? AND queue_status = 'LEASED'
                   AND lease_id = ? AND worker_id = ?""",
                (
                    queue_status,
                    stamp,
                    str(error_code)[:200],
                    str(error_message)[:1000],
                    stamp,
                    task_id,
                    lease_id,
                    worker_id,
                ),
            )
            if updated.rowcount != 1:
                raise QueueLeaseLost(f"queue_finish_lost:{task_id}")
            self.db.commit()
            return {"task_id": task_id, "queue_status": queue_status, "attempt_count": int(current["attempt_count"])}
        except Exception:
            self.db.rollback()
            raise

        finally:
            self._db_lock.release()

    def mark_task_execution_unknown_if_owned(
        self,
        claim: dict[str, Any],
        *,
        reason: str = "max_execution_age_exceeded",
    ) -> dict[str, Any]:
        """Fence an owned long-running claim without retrying its external action."""
        task_id = str(claim["task_id"])
        lease_id = str(claim["lease_id"])
        worker_id = str(claim["worker_id"])
        stamp = now()
        self._db_lock.acquire()
        try:
            self.db.execute("BEGIN IMMEDIATE")
            current = self.db.execute(
                "SELECT * FROM task_queue WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if current is None or current["queue_status"] != "LEASED" or current["lease_id"] != lease_id or current["worker_id"] != worker_id:
                raise QueueLeaseLost(f"queue_lease_lost:{task_id}")

            from private_agent.core.task_state import TaskState, TaskStopReasonCode, TERMINAL_STATUSES

            state_row = self.db.execute(
                "SELECT state_json FROM task_states WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            queue_status = "EXECUTION_UNKNOWN"
            if state_row is not None:
                state = TaskState.from_dict(json.loads(state_row["state_json"]))
                if state.status in TERMINAL_STATUSES:
                    queue_status = state.status
                elif state.status == "WAITING_APPROVAL":
                    queue_status = "WAITING_APPROVAL"
                else:
                    state.recovery_state = {
                        **state.recovery_state,
                        "reason": reason,
                        "external_side_effect_may_have_occurred": True,
                    }
                    if state.status != "EXECUTION_UNKNOWN":
                        state.checkpoint(
                            "EXECUTION_UNKNOWN",
                            status="EXECUTION_UNKNOWN",
                            phase="EXECUTION_UNKNOWN",
                            successful=False,
                        )
                    else:
                        state.touch()
                    state.set_stop_reason(
                        TaskStopReasonCode.EXECUTION_UNKNOWN,
                        step_id=state.current_step or None,
                        action_id=next(
                            (key for key, record in state.action_records.items() if record.get("step_id") == state.current_step),
                            None,
                        ),
                    )
                    self._persist_task_state_in_transaction(
                        state.to_dict(),
                        "EXECUTION_UNKNOWN",
                        task_status=state.status,
                        enforce_lease=False,
                    )

            updated = self.db.execute(
                """UPDATE task_queue
                   SET queue_status = ?, lease_id = '', worker_id = '',
                       lease_acquired_at = '', lease_expires_at = '',
                       last_finished_at = ?, last_error_code = ?,
                       last_error_message = ?, updated_at = ?
                 WHERE task_id = ? AND queue_status = 'LEASED'
                   AND lease_id = ? AND worker_id = ?""",
                (
                    queue_status,
                    stamp,
                    "execution_unknown" if queue_status == "EXECUTION_UNKNOWN" else "",
                    reason[:1000],
                    stamp,
                    task_id,
                    lease_id,
                    worker_id,
                ),
            )
            if updated.rowcount != 1:
                raise QueueLeaseLost(f"queue_unknown_lost:{task_id}")
            self.db.commit()
            return {"task_id": task_id, "queue_status": queue_status}
        except Exception:
            self.db.rollback()
            raise
        finally:
            self._db_lock.release()

    def requeue_waiting_approval_task(self, task_id: str) -> dict[str, Any]:
        """Explicitly make an approval-waiting task claimable again.

        This method does not approve anything.  The worker must still pass the
        task through Orchestrator resume and current SecurityController checks.
        """
        stamp = now()
        gate = self._approval_gate
        policy_lock = getattr(gate, "policy_version_lock", None)
        current_policy_version = getattr(gate, "current_policy_version", None)
        if (
            policy_lock is None
            or not callable(getattr(policy_lock, "acquire", None))
            or not callable(getattr(policy_lock, "release", None))
            or not callable(current_policy_version)
        ):
            raise ValueError("approval_policy_unavailable")
        policy_lock.acquire()
        try:
            with self._db_lock:
                self.db.execute("BEGIN IMMEDIATE")
                try:
                    queue = self.db.execute(
                        "SELECT queue_status FROM task_queue WHERE task_id = ?",
                        (task_id,),
                    ).fetchone()
                    state = self.db.execute(
                        "SELECT state_json FROM task_states WHERE task_id = ?",
                        (task_id,),
                    ).fetchone()
                    if queue is None or state is None:
                        raise ValueError(f"task_not_found:{task_id}")
                    from private_agent.core.task_state import TaskState

                    payload = json.loads(state["state_json"])
                    task_state = TaskState.from_dict(payload)
                    if queue["queue_status"] != "WAITING_APPROVAL" or task_state.status != "WAITING_APPROVAL":
                        raise ValueError("task_not_waiting_approval")
                    if not task_state.approval_ids:
                        raise ValueError("approval_not_approved")
                    current_version = current_policy_version()
                    for action_id, approval_id in task_state.approval_ids.items():
                        approval = self.get_approval_request(str(approval_id))
                        if (
                            not approval
                            or approval.get("status") != "APPROVED"
                            or str(approval.get("task_id", "")) != str(task_id)
                            or str(approval.get("action_id", "")) != str(action_id)
                        ):
                            raise ValueError("approval_not_approved")
                        try:
                            approval_version = int(approval.get("policy_version", -1))
                        except (TypeError, ValueError):
                            raise ValueError("approval_not_approved") from None
                        if approval_version != current_version:
                            raise ValueError("approval_policy_stale")
                    updated = self.db.execute(
                        """UPDATE task_queue
                           SET queue_status = 'QUEUED', available_at = ?,
                               last_error_code = '', last_error_message = '', updated_at = ?
                         WHERE task_id = ? AND queue_status = 'WAITING_APPROVAL'""",
                        (stamp, stamp, task_id),
                    )
                    if updated.rowcount != 1:
                        raise QueueLeaseLost(f"queue_requeue_lost:{task_id}")
                    self.db.commit()
                    return {"task_id": str(task_id), "queue_status": "QUEUED"}
                except Exception:
                    self.db.rollback()
                    raise
        finally:
            policy_lock.release()

    def waiting_approval_task_ids(self, *, limit: int = 100) -> list[str]:
        """Return a bounded snapshot for worker approval reconciliation."""
        if limit < 1 or limit > 1000:
            raise ValueError("waiting_approval_limit_invalid")
        with self._db_lock:
            rows = self.db.execute(
                """SELECT task_id FROM task_queue
                   WHERE queue_status = 'WAITING_APPROVAL'
                   ORDER BY updated_at, task_id
                   LIMIT ?""",
                (int(limit),),
            ).fetchall()
        return [str(row["task_id"]) for row in rows]

    def task_queue_record(self, task_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            row = self.db.execute("SELECT * FROM task_queue WHERE task_id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    def task_owner_user_id(self, task_id: str) -> str | None:
        with self._db_lock:
            row = self.db.execute("SELECT user_id FROM task_owners WHERE task_id = ?", (task_id,)).fetchone()
        return str(row["user_id"]) if row is not None else None

    def list_task_summaries(
        self,
        *,
        limit: int,
        cursor_created_at: str = "",
        cursor_task_id: str = "",
        status: str | None = None,
        owner_user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if status:
            clauses.append("s.status = ?")
            params.append(status)
        if owner_user_id is not None:
            clauses.append("o.user_id = ?")
            params.append(owner_user_id)
        if cursor_created_at and cursor_task_id:
            clauses.append("(s.created_at < ? OR (s.created_at = ? AND s.task_id < ?))")
            params.extend([cursor_created_at, cursor_created_at, cursor_task_id])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._db_lock:
            rows = self.db.execute(
                f"""SELECT s.task_id, s.status, s.current_phase, s.current_step,
                           s.plan_version, s.state_json, s.created_at, s.updated_at,
                           s.last_checkpoint_at, s.last_successful_checkpoint,
                           CASE
                             WHEN q.queue_status IS NOT NULL THEN q.queue_status
                             WHEN s.status IN ('EXECUTING', 'OBSERVING', 'VERIFYING') THEN 'EXECUTION_UNKNOWN'
                             WHEN s.status IN ('CREATED', 'QUEUED') THEN 'QUEUED'
                             WHEN s.status = 'WAITING_APPROVAL' THEN 'WAITING_APPROVAL'
                             ELSE s.status
                           END AS queue_status,
                           COALESCE(q.attempt_count, 0) AS attempt_count,
                           COALESCE(q.available_at, s.created_at) AS available_at,
                           COALESCE(q.last_error_code, '') AS last_error_code,
                           q.updated_at AS queue_updated_at
                    FROM task_states AS s
                    LEFT JOIN task_queue AS q ON q.task_id = s.task_id
                    LEFT JOIN task_owners AS o ON o.task_id = s.task_id
                    {where}
                    ORDER BY s.created_at DESC, s.task_id DESC
                    LIMIT ?""",
                (*params, int(limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_task_state(self, task_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            row = self.db.execute("SELECT state_json FROM task_states WHERE task_id = ?", (task_id,)).fetchone()
        if not row:
            return None
        try:
            payload = json.loads(row["state_json"])
        except (TypeError, ValueError) as exc:
            raise ValueError("corrupted_task_state_json") from exc
        return payload if isinstance(payload, dict) else None

    def get_task_state_row(self, task_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            row = self.db.execute("SELECT * FROM task_states WHERE task_id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    def task_checkpoints(self, task_id: str) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = self.db.execute(
                "SELECT * FROM task_checkpoints WHERE task_id = ? ORDER BY sequence",
                (task_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_task_checkpoint(self, task_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            row = self.db.execute(
                "SELECT * FROM task_checkpoints WHERE task_id = ? ORDER BY sequence DESC LIMIT 1",
                (task_id,),
            ).fetchone()
        return dict(row) if row else None

    def task_action_records(self, task_id: str) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = self.db.execute(
                "SELECT * FROM task_action_records WHERE task_id = ? ORDER BY updated_at",
                (task_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_episode(self, episode_id: str, task_id: str, goal: str, outcome: str, lessons: Any, observations: Any) -> None:
        with self._lease_write(task_id):
            self.db.execute("INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?)",
                (episode_id, task_id, goal, outcome, json.dumps(lessons), json.dumps(observations), now()))

    def add_knowledge(self, item: KnowledgeItem) -> None:
        if not item.created_at: item.created_at = now()
        item.updated_at = now()
        active = self._active_lease()
        active_task_id = active[0] if active is not None else None
        def bounded(value: Any, limit: int) -> str:
            safe = sanitize_value(str(value))
            return safe[:limit]

        source_value = bounded(item.source, 2048)
        if source_value.lower().startswith(("http://", "https://")):
            try:
                source_value = canonicalize_source_uri(source_value)
            except KnowledgeValidationError:
                source_value = "[REDACTED_SOURCE]"
        values = (
            bounded(item.id, 256),
            bounded(item.domain, 256),
            bounded(item.concept, 4096),
            bounded(item.knowledge_type, 128),
            bounded(item.content, 8192),
            source_value,
            bounded(item.evidence, MAX_EXCERPT_LENGTH),
            float(item.confidence),
            "needs_review",
            0.0,
            int(item.version),
            int(item.usage_count),
            float(item.success_rate),
            bounded(item.created_at, 64),
            bounded(item.updated_at, 64),
            "",
            bounded(item.related_skills, 4096),
            bounded(item.contradictions, 4096),
            bounded(item.dependencies, 4096),
        )
        with self._lease_write(active_task_id):
            self.db.execute("INSERT OR REPLACE INTO knowledge_items VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", values)

    @staticmethod
    def _knowledge_source_from_row(row: sqlite3.Row | dict[str, Any]) -> KnowledgeSource:
        source = KnowledgeSource(
            source_id=str(row["source_id"]),
            canonical_uri=str(row["canonical_uri"]),
            source_type=str(row["source_type"]),
            source_class=str(row["source_class"]),
            retrieved_at=str(row["retrieved_at"]),
            published_at=str(row["published_at"]),
            content_sha256=str(row["content_sha256"]),
            revision=str(row["revision"]),
            status=str(row["status"]),
            metadata=json.loads(str(row["metadata_json"])),
            created_at=str(row["created_at"]),
        )
        # A persisted row is not a provenance attestation. Reopening a
        # database must not turn caller-supplied source fields into trust.
        object.__setattr__(source, "content_bound", False)
        return source

    @staticmethod
    def _knowledge_claim_from_row(row: sqlite3.Row | dict[str, Any]) -> KnowledgeClaim:
        return KnowledgeClaim(
            claim_id=str(row["claim_id"]),
            claim_fingerprint=str(row["claim_fingerprint"]),
            domain=str(row["domain"]),
            subject=str(row["subject"]),
            predicate=str(row["predicate"]),
            object_text=str(row["object_text"]),
            knowledge_type=str(row["knowledge_type"]),
            verification_status=str(row["verification_status"]),
            status=str(row["status"]),
            confidence=float(row["confidence"]),
            observed_at=str(row["observed_at"]),
            last_verified_at=str(row["last_verified_at"]),
            valid_from=str(row["valid_from"]),
            valid_until=str(row["valid_until"]),
            version=int(row["version"]),
            metadata=json.loads(str(row["metadata_json"])),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _knowledge_evidence_from_row(row: sqlite3.Row | dict[str, Any]) -> KnowledgeEvidence:
        return KnowledgeEvidence(
            evidence_id=str(row["evidence_id"]),
            evidence_fingerprint=str(row["evidence_fingerprint"]),
            claim_id=str(row["claim_id"]),
            source_id=str(row["source_id"]),
            task_id=str(row["task_id"]),
            step_id=str(row["step_id"]),
            locator=str(row["locator"]),
            excerpt=str(row["excerpt"]),
            excerpt_digest=str(row["excerpt_sha256"]),
            verification_status=str(row["verification_status"]),
            observed_at=str(row["observed_at"]),
            metadata=json.loads(str(row["metadata_json"])),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _knowledge_conflict_from_row(row: sqlite3.Row | dict[str, Any]) -> KnowledgeConflict:
        return KnowledgeConflict(
            conflict_id=str(row["conflict_id"]),
            conflict_fingerprint=str(row["conflict_fingerprint"]),
            claim_id=str(row["claim_id"]),
            conflicting_claim_id=str(row["conflicting_claim_id"]),
            conflict_type=str(row["conflict_type"]),
            status=str(row["status"]),
            detected_at=str(row["detected_at"]),
            evidence_ids=list(json.loads(str(row["evidence_ids_json"]))),
            resolution_note=str(row["resolution_note"]),
            metadata=json.loads(str(row["metadata_json"])),
            created_at=str(row["created_at"]),
        )

    def _knowledge_lease_task_id(self) -> str | None:
        active = self._active_lease()
        return active[0] if active is not None else None

    def _assert_knowledge_provenance(self, task_id: str) -> None:
        if not task_id:
            return
        active = self._active_lease()
        if active is None:
            raise KnowledgeValidationError("knowledge_task_lease_required")
        if task_id != active[0]:
            raise KnowledgeValidationError("knowledge_task_provenance_mismatch")

    def _insert_knowledge_source(self, source: KnowledgeSource) -> KnowledgeSource:
        persisted_payload = source.to_dict()
        persisted_source = KnowledgeSource(**persisted_payload)
        # The public store API has no trusted fetcher/attestation input yet.
        # Keep the declared class as metadata, but never expose it as proof.
        object.__setattr__(persisted_source, "content_bound", False)
        by_id = self.db.execute("SELECT * FROM knowledge_sources WHERE source_id = ?", (source.source_id,)).fetchone()
        if by_id is not None:
            existing = self._knowledge_source_from_row(by_id)
            if existing.identity_key != persisted_source.identity_key:
                raise KnowledgeValidationError("source_id_conflict")
            if existing.status != persisted_source.status or existing.source_class != persisted_source.source_class:
                raise KnowledgeValidationError("source_transition_required")
            return existing
        duplicate = self.db.execute(
            "SELECT * FROM knowledge_sources WHERE canonical_uri = ? AND revision = ? AND content_sha256 = ?",
            persisted_source.identity_key,
        ).fetchone()
        if duplicate is not None:
            existing = self._knowledge_source_from_row(duplicate)
            if existing.status != persisted_source.status or existing.source_class != persisted_source.source_class:
                raise KnowledgeValidationError("source_transition_required")
            return existing
        self.db.execute(
            """INSERT INTO knowledge_sources
               (source_id, canonical_uri, source_type, source_class, retrieved_at,
                published_at, content_sha256, revision, status, metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                persisted_source.source_id,
                persisted_source.canonical_uri,
                persisted_source.source_type,
                persisted_source.source_class,
                persisted_source.retrieved_at,
                persisted_source.published_at,
                persisted_source.content_sha256,
                persisted_source.revision,
                persisted_source.status,
                json.dumps(persisted_source.metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                persisted_source.created_at,
            ),
        )
        return persisted_source

    def save_knowledge_source(self, source: KnowledgeSource) -> KnowledgeSource:
        if not isinstance(source, KnowledgeSource):
            raise KnowledgeValidationError("source_type_required")
        with self._lease_write(self._knowledge_lease_task_id()):
            return self._insert_knowledge_source(source)

    def get_knowledge_source(self, source_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM knowledge_sources WHERE source_id = ?", (str(source_id),)).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["provenance_attested"] = False
        payload["effective_source_class"] = "attested" if payload["provenance_attested"] else "unverified"
        return payload

    @staticmethod
    def _untrusted_claim_projection(claim: KnowledgeClaim) -> KnowledgeClaim:
        if claim.verification_status == "unverified" and claim.status == "PROPOSED" and not claim.last_verified_at:
            return claim
        payload = claim.to_dict()
        payload.update({"verification_status": "unverified", "status": "PROPOSED", "last_verified_at": "", "claim_fingerprint": ""})
        return KnowledgeClaim(**payload)

    def _insert_knowledge_claim(self, claim: KnowledgeClaim) -> KnowledgeClaim:
        # No trusted ingestion/attestation boundary exists in this release.
        # Every caller-provided claim is therefore stored as advisory data.
        claim = self._untrusted_claim_projection(claim)
        by_id = self.db.execute("SELECT * FROM knowledge_claims WHERE claim_id = ?", (claim.claim_id,)).fetchone()
        if by_id is not None:
            existing = self._knowledge_claim_from_row(by_id)
            if existing.claim_fingerprint != claim.claim_fingerprint:
                raise KnowledgeValidationError("claim_id_conflict")
            if existing.version != claim.version or existing.status != claim.status or existing.verification_status != claim.verification_status:
                raise KnowledgeValidationError("claim_transition_required")
            return existing
        duplicate = self.db.execute(
            "SELECT * FROM knowledge_claims WHERE claim_fingerprint = ?", (claim.claim_fingerprint,)
        ).fetchone()
        if duplicate is not None:
            existing = self._knowledge_claim_from_row(duplicate)
            if existing.version != claim.version or existing.status != claim.status or existing.verification_status != claim.verification_status:
                raise KnowledgeValidationError("claim_transition_required")
            return existing
        self.db.execute(
            """INSERT INTO knowledge_claims
               (claim_id, claim_fingerprint, domain, subject, predicate, object_text,
                knowledge_type, verification_status, status, confidence, observed_at,
                last_verified_at, valid_from, valid_until, version, metadata_json,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                claim.claim_id,
                claim.claim_fingerprint,
                claim.domain,
                claim.subject,
                claim.predicate,
                claim.object_text,
                claim.knowledge_type,
                claim.verification_status,
                claim.status,
                claim.confidence,
                claim.observed_at,
                claim.last_verified_at,
                claim.valid_from,
                claim.valid_until,
                claim.version,
                json.dumps(claim.metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                claim.created_at,
                claim.updated_at,
            ),
        )
        return claim

    def save_knowledge_claim(self, claim: KnowledgeClaim) -> KnowledgeClaim:
        if not isinstance(claim, KnowledgeClaim):
            raise KnowledgeValidationError("claim_type_required")
        with self._lease_write(self._knowledge_lease_task_id()):
            return self._insert_knowledge_claim(claim)

    def refresh_knowledge_claim(
        self,
        claim: KnowledgeClaim,
        evidence: list[KnowledgeEvidence] | None = None,
    ) -> KnowledgeClaim:
        if not isinstance(claim, KnowledgeClaim):
            raise KnowledgeValidationError("claim_type_required")
        evidence_items = evidence or []
        if not isinstance(evidence_items, list) or len(evidence_items) > 32:
            raise KnowledgeValidationError("refresh_evidence_invalid")
        if any(not isinstance(item, KnowledgeEvidence) for item in evidence_items):
            raise KnowledgeValidationError("refresh_evidence_types_required")
        if any(item.claim_id != claim.claim_id for item in evidence_items):
            raise KnowledgeValidationError("refresh_evidence_provenance_mismatch")
        with self._lease_write(self._knowledge_lease_task_id()):
            existing_row = self.db.execute(
                "SELECT * FROM knowledge_claims WHERE claim_fingerprint = ?", (claim.claim_fingerprint,)
            ).fetchone()
            if existing_row is None:
                return self._insert_knowledge_claim(claim)
            existing = self._knowledge_claim_from_row(existing_row)
            if claim.version <= existing.version:
                raise KnowledgeValidationError("claim_version_not_advanced")
            # A trusted ingestion/attestation boundary is not implemented in
            # this phase. Never promote caller-provided refresh data.
            claim = self._untrusted_claim_projection(claim)
            self.db.execute(
                """UPDATE knowledge_claims
                   SET verification_status = ?, status = ?, confidence = ?, observed_at = ?,
                       last_verified_at = ?, valid_from = ?, valid_until = ?, version = ?,
                       metadata_json = ?, updated_at = ?
                   WHERE claim_id = ? AND version = ?""",
                (
                    claim.verification_status,
                    claim.status,
                    claim.confidence,
                    claim.observed_at,
                    claim.last_verified_at,
                    claim.valid_from,
                    claim.valid_until,
                    claim.version,
                    json.dumps(claim.metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    claim.updated_at,
                    existing.claim_id,
                    existing.version,
                ),
            )
            for item in evidence_items:
                self._insert_knowledge_evidence(item)
            updated = self.db.execute("SELECT * FROM knowledge_claims WHERE claim_id = ?", (existing.claim_id,)).fetchone()
            if updated is None:
                raise KnowledgeValidationError("claim_refresh_failed")
            return self._knowledge_claim_from_row(updated)

    def get_knowledge_claim(self, claim_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM knowledge_claims WHERE claim_id = ?", (str(claim_id),)).fetchone()
        return dict(row) if row else None

    def _insert_knowledge_evidence(self, evidence: KnowledgeEvidence) -> KnowledgeEvidence:
        self._assert_knowledge_provenance(evidence.task_id)
        if self.db.execute("SELECT 1 FROM knowledge_claims WHERE claim_id = ?", (evidence.claim_id,)).fetchone() is None:
            raise KnowledgeValidationError("evidence_claim_missing")
        if self.db.execute("SELECT 1 FROM knowledge_sources WHERE source_id = ?", (evidence.source_id,)).fetchone() is None:
            raise KnowledgeValidationError("evidence_source_missing")
        by_id = self.db.execute("SELECT * FROM knowledge_evidence WHERE evidence_id = ?", (evidence.evidence_id,)).fetchone()
        if by_id is not None:
            existing = self._knowledge_evidence_from_row(by_id)
            if existing.evidence_fingerprint != evidence.evidence_fingerprint:
                raise KnowledgeValidationError("evidence_id_conflict")
            if existing.verification_status != evidence.verification_status:
                raise KnowledgeValidationError("evidence_transition_required")
            return existing
        duplicate = self.db.execute(
            "SELECT * FROM knowledge_evidence WHERE evidence_fingerprint = ?", (evidence.evidence_fingerprint,)
        ).fetchone()
        if duplicate is not None:
            existing = self._knowledge_evidence_from_row(duplicate)
            if existing.verification_status != evidence.verification_status:
                raise KnowledgeValidationError("evidence_transition_required")
            return existing
        self.db.execute(
            """INSERT INTO knowledge_evidence
               (evidence_id, evidence_fingerprint, claim_id, source_id, task_id, step_id,
                locator, excerpt, excerpt_sha256, verification_status, observed_at,
                metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                evidence.evidence_id,
                evidence.evidence_fingerprint,
                evidence.claim_id,
                evidence.source_id,
                evidence.task_id,
                evidence.step_id,
                evidence.locator,
                evidence.excerpt,
                evidence.excerpt_digest,
                evidence.verification_status,
                evidence.observed_at,
                json.dumps(evidence.metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                evidence.created_at,
            ),
        )
        return evidence

    def save_knowledge_evidence(self, evidence: KnowledgeEvidence) -> KnowledgeEvidence:
        if not isinstance(evidence, KnowledgeEvidence):
            raise KnowledgeValidationError("evidence_type_required")
        with self._lease_write(self._knowledge_lease_task_id()):
            return self._insert_knowledge_evidence(evidence)

    def list_knowledge_evidence(self, claim_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM knowledge_evidence WHERE claim_id = ? ORDER BY observed_at, evidence_id",
            (str(claim_id),),
        ).fetchall()
        return [dict(row) for row in rows]

    def _insert_knowledge_conflict(self, conflict: KnowledgeConflict) -> KnowledgeConflict:
        if conflict.status == "OPEN" and not conflict.evidence_ids:
            raise KnowledgeValidationError("open_conflict_evidence_required")
        for claim_id in (conflict.claim_id, conflict.conflicting_claim_id):
            if self.db.execute("SELECT 1 FROM knowledge_claims WHERE claim_id = ?", (claim_id,)).fetchone() is None:
                raise KnowledgeValidationError("conflict_claim_missing")
        if conflict.evidence_ids:
            evidence_rows = self.db.execute(
                "SELECT evidence_id, claim_id FROM knowledge_evidence WHERE evidence_id IN ({})".format(",".join("?" for _ in conflict.evidence_ids)),
                tuple(conflict.evidence_ids),
            ).fetchall()
            evidence_by_id = {str(row["evidence_id"]): str(row["claim_id"]) for row in evidence_rows}
            if any(
                evidence_id not in evidence_by_id
                or evidence_by_id[evidence_id] not in {conflict.claim_id, conflict.conflicting_claim_id}
                for evidence_id in conflict.evidence_ids
            ):
                raise KnowledgeValidationError("conflict_evidence_invalid")
            if conflict.status == "OPEN":
                verified_rows = self.db.execute(
                    """SELECT e.evidence_id
                       FROM knowledge_evidence AS e
                       JOIN knowledge_sources AS s ON s.source_id = e.source_id
                       WHERE e.evidence_id IN ({})
                         AND e.verification_status = 'verified'
                         AND s.status = 'ACTIVE'
                         AND s.source_class <> 'unverified'""".format(",".join("?" for _ in conflict.evidence_ids)),
                    tuple(conflict.evidence_ids),
                ).fetchall()
                if len(verified_rows) != len(conflict.evidence_ids):
                    raise KnowledgeValidationError("open_conflict_evidence_unverified")
        by_id = self.db.execute("SELECT * FROM knowledge_conflicts WHERE conflict_id = ?", (conflict.conflict_id,)).fetchone()
        if by_id is not None:
            existing = self._knowledge_conflict_from_row(by_id)
            if existing.conflict_fingerprint != conflict.conflict_fingerprint:
                raise KnowledgeValidationError("conflict_id_conflict")
            if existing.status != conflict.status:
                raise KnowledgeValidationError("conflict_transition_required")
            return existing
        duplicate = self.db.execute(
            "SELECT * FROM knowledge_conflicts WHERE conflict_fingerprint = ?", (conflict.conflict_fingerprint,)
        ).fetchone()
        if duplicate is not None:
            existing = self._knowledge_conflict_from_row(duplicate)
            if existing.status != conflict.status:
                raise KnowledgeValidationError("conflict_transition_required")
            return existing
        self.db.execute(
            """INSERT INTO knowledge_conflicts
               (conflict_id, conflict_fingerprint, claim_id, conflicting_claim_id,
                conflict_type, status, detected_at, evidence_ids_json, resolution_note,
                metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                conflict.conflict_id,
                conflict.conflict_fingerprint,
                conflict.claim_id,
                conflict.conflicting_claim_id,
                conflict.conflict_type,
                conflict.status,
                conflict.detected_at,
                json.dumps(conflict.evidence_ids, ensure_ascii=False, separators=(",", ":")),
                conflict.resolution_note,
                json.dumps(conflict.metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                conflict.created_at,
            ),
        )
        return conflict

    def _suppress_knowledge_attestations_for_conflict(self, conflict: KnowledgeConflict) -> None:
        if conflict.status != "OPEN":
            return
        claim_ids = sorted(
            {item for item in (conflict.claim_id, conflict.conflicting_claim_id) if item}
        )
        if not claim_ids:
            return
        rows = self.db.execute(
            "SELECT * FROM knowledge_attestations ORDER BY created_at, attestation_id"
        ).fetchall()
        if self._knowledge_attestor is None:
            if rows:
                raise KnowledgeValidationError("knowledge_attestation_key_missing")
            return
        for row in rows:
            try:
                attestation = KnowledgeAttestation.from_dict(
                    json.loads(str(row["attestation_json"]))
                )
            except (KnowledgeValidationError, TypeError, ValueError):
                if str(row["claim_id"]) in claim_ids:
                    raise KnowledgeValidationError("knowledge_attestation_invalid")
                continue
            signed_claim_id = str(attestation.claim_snapshot["claim_id"])
            if signed_claim_id not in claim_ids:
                if str(row["claim_id"]) in claim_ids:
                    raise KnowledgeValidationError("knowledge_attestation_invalid")
                continue
            if not self._knowledge_attestor.verify(attestation):
                raise KnowledgeValidationError("knowledge_attestation_invalid")
            updated = self._knowledge_attestor.suppress_for_conflict(
                attestation, conflict.conflict_id
            )
            attestation_json = json.dumps(
                updated.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            cursor = self.db.execute(
                """UPDATE knowledge_attestations
                   SET attestation_id = ?, claim_id = ?, source_id = ?, claim_version = ?,
                       key_id = ?, signature = ?, attestation_json = ?, created_at = ?
                   WHERE attestation_id = ?""",
                (
                    updated.attestation_id,
                    updated.claim_snapshot["claim_id"],
                    updated.source_snapshot["source_id"],
                    updated.claim_snapshot["version"],
                    updated.key_id,
                    updated.signature,
                    attestation_json,
                    updated.verified_at,
                    row["attestation_id"],
                ),
            )
            if cursor.rowcount != 1:
                raise KnowledgeValidationError("knowledge_attestation_conflict_update_failed")

    def save_knowledge_conflict(self, conflict: KnowledgeConflict) -> KnowledgeConflict:
        if not isinstance(conflict, KnowledgeConflict):
            raise KnowledgeValidationError("conflict_type_required")
        with self._lease_write(self._knowledge_lease_task_id()):
            saved = self._insert_knowledge_conflict(conflict)
            self._suppress_knowledge_attestations_for_conflict(saved)
            return saved

    def list_knowledge_conflicts(self, claim_id: str | None = None) -> list[dict[str, Any]]:
        if claim_id is None:
            rows = self.db.execute("SELECT * FROM knowledge_conflicts ORDER BY detected_at, conflict_id").fetchall()
        else:
            rows = self.db.execute(
                """SELECT * FROM knowledge_conflicts
                   WHERE claim_id = ? OR conflicting_claim_id = ?
                   ORDER BY detected_at, conflict_id""",
                (str(claim_id), str(claim_id)),
            ).fetchall()
        return [dict(row) for row in rows]

    def _project_knowledge_claim(self, claim: KnowledgeClaim, source: KnowledgeSource, evidence: KnowledgeEvidence) -> None:
        raise KnowledgeValidationError("knowledge_ingestion_attestation_required")

    def save_knowledge_bundle(
        self,
        source: KnowledgeSource,
        claim: KnowledgeClaim,
        evidence: list[KnowledgeEvidence],
    ) -> dict[str, Any]:
        if not isinstance(source, KnowledgeSource) or not isinstance(claim, KnowledgeClaim):
            raise KnowledgeValidationError("bundle_types_required")
        if not isinstance(evidence, list) or not evidence:
            raise KnowledgeValidationError("bundle_evidence_required")
        if len(evidence) > 32:
            raise KnowledgeValidationError("bundle_evidence_too_large")
        if any(not isinstance(item, KnowledgeEvidence) for item in evidence):
            raise KnowledgeValidationError("bundle_evidence_types_required")
        if any(item.claim_id != claim.claim_id or item.source_id != source.source_id for item in evidence):
            raise KnowledgeValidationError("bundle_provenance_mismatch")
        try:
            bundle_bytes = len(json.dumps({
                "source": source.to_dict(),
                "claim": claim.to_dict(),
                "evidence": [item.to_dict() for item in evidence],
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise KnowledgeValidationError("bundle_not_json_safe") from exc
        if bundle_bytes > 256 * 1024:
            raise KnowledgeValidationError("bundle_too_large")
        with self._lease_write(self._knowledge_lease_task_id()):
            saved_source = self._insert_knowledge_source(source)
            saved_claim = self._insert_knowledge_claim(claim)
            rebound_evidence: list[KnowledgeEvidence] = []
            for item in evidence:
                if item.claim_id == saved_claim.claim_id and item.source_id == saved_source.source_id:
                    rebound_evidence.append(item)
                    continue
                payload = item.to_dict()
                payload.update(
                    {
                        "claim_id": saved_claim.claim_id,
                        "source_id": saved_source.source_id,
                        "evidence_fingerprint": "",
                    }
                )
                rebound_evidence.append(KnowledgeEvidence(**payload))
            saved_evidence = [self._insert_knowledge_evidence(item) for item in rebound_evidence]
            return {
                "source": saved_source.to_dict(),
                "claim": saved_claim.to_dict(),
                "evidence": [item.to_dict() for item in saved_evidence],
            }

    def issue_attested_knowledge_bundle(
        self,
        source: KnowledgeSource,
        claim: KnowledgeClaim,
        evidence: list[KnowledgeEvidence],
        *,
        issuer_authorizer: Callable[
            [KnowledgeSource, KnowledgeClaim, list[KnowledgeEvidence]], str | None
        ],
        verified_at: str | None = None,
        attestation_id: str | None = None,
    ) -> dict[str, Any]:
        """Issue and persist a bundle using a server-supplied issuer decision.

        The callback must derive its identity from authenticated server context
        and validate the associated persisted review; it must not accept a
        reviewer identity supplied by an API caller.
        """
        if self._knowledge_attestor is None:
            raise KnowledgeValidationError("knowledge_attestation_key_missing")
        issuer = self._knowledge_attestor.with_issuer_authorizer(issuer_authorizer)
        attestation = issuer.issue(
            source,
            claim,
            evidence,
            verified_at=verified_at,
            attestation_id=attestation_id,
        )
        return self.save_attested_knowledge_bundle(source, claim, evidence, attestation)

    def save_attested_knowledge_bundle(
        self,
        source: KnowledgeSource,
        claim: KnowledgeClaim,
        evidence: list[KnowledgeEvidence],
        attestation: KnowledgeAttestation,
    ) -> dict[str, Any]:
        """Persist a reviewed bundle while keeping its ordinary claim row advisory."""
        if not isinstance(source, KnowledgeSource) or not isinstance(claim, KnowledgeClaim):
            raise KnowledgeValidationError("bundle_types_required")
        if not isinstance(evidence, list) or not evidence or len(evidence) > 32:
            raise KnowledgeValidationError("bundle_evidence_required")
        if any(not isinstance(item, KnowledgeEvidence) for item in evidence):
            raise KnowledgeValidationError("bundle_evidence_types_required")
        if not isinstance(attestation, KnowledgeAttestation):
            raise KnowledgeValidationError("knowledge_attestation_required")
        if self._knowledge_attestor is None:
            raise KnowledgeValidationError("knowledge_attestation_key_missing")
        if not source.content_bound or not self._knowledge_attestor.verify(attestation):
            raise KnowledgeValidationError("knowledge_attestation_invalid")
        source_snapshot = source.to_dict()
        claim_snapshot = claim.to_dict()
        evidence_snapshots = sorted(
            (item.to_dict() for item in evidence),
            key=lambda item: (item["evidence_id"], item["evidence_fingerprint"]),
        )
        if (
            source_snapshot != attestation.source_snapshot
            or claim_snapshot != attestation.claim_snapshot
            or evidence_snapshots != attestation.evidence_snapshots
        ):
            raise KnowledgeValidationError("knowledge_attestation_bundle_mismatch")
        if any(item.claim_id != claim.claim_id or item.source_id != source.source_id for item in evidence):
            raise KnowledgeValidationError("bundle_provenance_mismatch")
        encoded_bundle = json.dumps(
            {"source": source_snapshot, "claim": claim_snapshot, "evidence": evidence_snapshots},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(encoded_bundle.encode("utf-8")) > 256 * 1024:
            raise KnowledgeValidationError("bundle_too_large")
        attestation_json = json.dumps(
            attestation.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if len(attestation_json.encode("utf-8")) > 256 * 1024:
            raise KnowledgeValidationError("knowledge_attestation_too_large")

        with self._lease_write(self._knowledge_lease_task_id()):
            saved_source = self._insert_knowledge_source(source)
            saved_claim = self._insert_knowledge_claim(claim)
            saved_evidence: list[KnowledgeEvidence] = []
            for item in evidence:
                saved_evidence.append(self._insert_knowledge_evidence(item))
            if (
                saved_source.to_dict() != attestation.source_snapshot
                or saved_claim.claim_id != claim.claim_id
                or saved_claim.claim_fingerprint != claim.claim_fingerprint
            ):
                raise KnowledgeValidationError("knowledge_attestation_persisted_identity_mismatch")
            saved_evidence_snapshots = sorted(
                (item.to_dict() for item in saved_evidence),
                key=lambda item: (item["evidence_id"], item["evidence_fingerprint"]),
            )
            if saved_evidence_snapshots != attestation.evidence_snapshots:
                raise KnowledgeValidationError("knowledge_attestation_persisted_evidence_mismatch")
            open_conflicts = self.db.execute(
                """SELECT conflict_id FROM knowledge_conflicts
                   WHERE status = 'OPEN' AND (claim_id = ? OR conflicting_claim_id = ?)
                   ORDER BY conflict_id""",
                (claim.claim_id, claim.claim_id),
            ).fetchall()
            for conflict_row in open_conflicts:
                attestation = self._knowledge_attestor.suppress_for_conflict(
                    attestation, str(conflict_row["conflict_id"])
                )
            attestation_json = json.dumps(
                attestation.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            if len(attestation_json.encode("utf-8")) > 256 * 1024:
                raise KnowledgeValidationError("knowledge_attestation_too_large")
            existing = self.db.execute(
                "SELECT attestation_json, signature FROM knowledge_attestations WHERE attestation_id = ?",
                (attestation.attestation_id,),
            ).fetchone()
            if existing is not None:
                if existing["attestation_json"] != attestation_json or existing["signature"] != attestation.signature:
                    raise KnowledgeValidationError("knowledge_attestation_id_conflict")
            else:
                self.db.execute(
                    """INSERT INTO knowledge_attestations
                       (attestation_id, claim_id, source_id, claim_version, key_id, signature,
                        attestation_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        attestation.attestation_id,
                        claim.claim_id,
                        source.source_id,
                        claim.version,
                        attestation.key_id,
                        attestation.signature,
                        attestation_json,
                        attestation.verified_at,
                    ),
                )
            if not self._attestation_matches_persisted_rows(attestation):
                raise KnowledgeValidationError("knowledge_attestation_persistence_check_failed")
            return {
                "source": saved_source.to_dict(),
                "claim": saved_claim.to_dict(),
                "evidence": [item.to_dict() for item in saved_evidence],
                "attestation": attestation.to_dict(),
            }

    def _attestation_matches_persisted_rows(self, attestation: KnowledgeAttestation) -> bool:
        if self._knowledge_attestor is None or not self._knowledge_attestor.verify(attestation):
            return False
        source_row = self.db.execute(
            "SELECT * FROM knowledge_sources WHERE source_id = ?",
            (attestation.source_snapshot["source_id"],),
        ).fetchone()
        claim_row = self.db.execute(
            "SELECT * FROM knowledge_claims WHERE claim_id = ?",
            (attestation.claim_snapshot["claim_id"],),
        ).fetchone()
        if source_row is None or claim_row is None:
            return False
        try:
            stored_source = self._knowledge_source_from_row(source_row).to_dict()
            stored_claim = self._knowledge_claim_from_row(claim_row).to_dict()
        except (KnowledgeValidationError, TypeError, ValueError):
            return False
        if stored_source != attestation.source_snapshot:
            return False
        if (
            stored_claim["status"] != "PROPOSED"
            or stored_claim["verification_status"] != "unverified"
            or stored_claim["last_verified_at"] != ""
        ):
            return False
        for field_name in ("status", "verification_status", "last_verified_at"):
            stored_claim[field_name] = attestation.claim_snapshot[field_name]
        if stored_claim != attestation.claim_snapshot:
            return False
        evidence_rows = self.db.execute(
            """SELECT * FROM knowledge_evidence
               WHERE claim_id = ? AND source_id = ? ORDER BY evidence_id""",
            (attestation.claim_snapshot["claim_id"], attestation.source_snapshot["source_id"]),
        ).fetchall()
        try:
            stored_evidence = sorted(
                (self._knowledge_evidence_from_row(row).to_dict() for row in evidence_rows),
                key=lambda item: (item["evidence_id"], item["evidence_fingerprint"]),
            )
        except (KnowledgeValidationError, TypeError, ValueError):
            return False
        return stored_evidence == attestation.evidence_snapshots

    @staticmethod
    def _knowledge_as_of(value: str | None) -> datetime:
        raw = value or now()
        try:
            parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError as exc:
            raise KnowledgeValidationError("as_of_invalid") from exc
        if parsed.tzinfo is None:
            raise KnowledgeValidationError("as_of_timezone_required")
        return parsed.astimezone(timezone.utc)

    def _verified_knowledge_attestations(
        self,
    ) -> tuple[dict[str, KnowledgeAttestation], dict[str, int]]:
        if self._knowledge_attestor is None:
            return {}, {}
        result: dict[str, KnowledgeAttestation] = {}
        source_ids: dict[str, set[str]] = {}
        rows = self.db.execute(
            "SELECT * FROM knowledge_attestations ORDER BY created_at, attestation_id"
        ).fetchall()
        for row in rows:
            try:
                payload = json.loads(str(row["attestation_json"]))
                attestation = KnowledgeAttestation.from_dict(payload)
            except (KnowledgeValidationError, TypeError, ValueError):
                continue
            try:
                row_claim_version = int(row["claim_version"])
            except (TypeError, ValueError):
                continue
            if (
                row["attestation_id"] != attestation.attestation_id
                or row["claim_id"] != attestation.claim_snapshot["claim_id"]
                or row["source_id"] != attestation.source_snapshot["source_id"]
                or row_claim_version != int(attestation.claim_snapshot["version"])
                or row["key_id"] != attestation.key_id
                or row["signature"] != attestation.signature
                or not self._attestation_matches_persisted_rows(attestation)
            ):
                continue
            claim_id = attestation.claim_snapshot["claim_id"]
            source_ids.setdefault(claim_id, set()).add(attestation.source_snapshot["source_id"])
            previous = result.get(claim_id)
            if previous is None or (attestation.claim_snapshot["version"], attestation.verified_at) > (
                previous.claim_snapshot["version"], previous.verified_at
            ):
                result[claim_id] = attestation
        return result, {claim_id: len(items) for claim_id, items in source_ids.items()}

    def search_knowledge_claims(
        self,
        query: str,
        *,
        limit: int = 8,
        as_of: str | None = None,
        max_age_seconds: int = 30 * 24 * 60 * 60,
        include_unverified: bool = False,
        task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise KnowledgeValidationError("knowledge_query_required")
        if task_id is not None and (not isinstance(task_id, str) or not task_id.strip()):
            raise KnowledgeValidationError("knowledge_task_scope_invalid")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > 100:
            raise KnowledgeValidationError("knowledge_query_limit_invalid")
        if not isinstance(max_age_seconds, int) or isinstance(max_age_seconds, bool) or max_age_seconds < 1 or max_age_seconds > 10 * 365 * 24 * 60 * 60:
            raise KnowledgeValidationError("knowledge_freshness_invalid")
        reference = self._knowledge_as_of(as_of)
        terms = [term.lower() for term in query.split() if len(term) > 2]
        if not terms:
            return []
        rows = self.db.execute("SELECT * FROM knowledge_claims").fetchall()
        attested_claims, attested_source_counts = self._verified_knowledge_attestations()
        open_conflicts: dict[str, int] = {}
        for row in self.db.execute("SELECT claim_id, conflicting_claim_id FROM knowledge_conflicts WHERE status = 'OPEN'").fetchall():
            open_conflicts[str(row["claim_id"])] = open_conflicts.get(str(row["claim_id"]), 0) + 1
            open_conflicts[str(row["conflicting_claim_id"])] = open_conflicts.get(str(row["conflicting_claim_id"]), 0) + 1
        results: list[dict[str, Any]] = []
        current_owner_user_id = self.task_owner_user_id(task_id) if task_id is not None else None
        for row in rows:
            attestation = attested_claims.get(str(row["claim_id"]))
            if task_id is not None and attestation is not None:
                source_snapshot = attestation.source_snapshot
                claim_snapshot = attestation.claim_snapshot
                source_metadata = source_snapshot.get("metadata")
                claim_metadata = claim_snapshot.get("metadata")
                source_metadata = source_metadata if isinstance(source_metadata, dict) else {}
                claim_metadata = claim_metadata if isinstance(claim_metadata, dict) else {}
                bound_task_ids: set[str] = set()
                for metadata in [source_metadata, claim_metadata, *(item.get("metadata", {}) for item in attestation.evidence_snapshots)]:
                    if not isinstance(metadata, dict):
                        continue
                    bound_task_id = metadata.get("task_id")
                    if isinstance(bound_task_id, str) and bound_task_id:
                        bound_task_ids.add(bound_task_id)
                owner_review_bound = bool(
                    source_metadata.get("review_id")
                    or claim_metadata.get("review_id")
                    or str(source_snapshot.get("source_id", "")).startswith("source-review-")
                    or str(source_snapshot.get("canonical_uri", "")).startswith("local://bug-bounty/reviews/")
                )
                if bound_task_ids or owner_review_bound:
                    if len(bound_task_ids) != 1 or current_owner_user_id is None:
                        continue
                    bound_task_id = next(iter(bound_task_ids))
                    if self.task_owner_user_id(bound_task_id) != current_owner_user_id:
                        continue
            claim_snapshot = attestation.claim_snapshot if attestation is not None else None
            status = str(claim_snapshot["status"] if claim_snapshot is not None else "PROPOSED")
            verification_status = str(
                claim_snapshot["verification_status"] if claim_snapshot is not None else "unverified"
            )
            if status in {"RETIRED", "QUARANTINED"}:
                continue
            text = f"{row['subject']} {row['predicate']} {row['object_text']} {row['domain']}".lower()
            score = sum(term in text for term in terms)
            if not score:
                continue
            valid_from_raw = claim_snapshot["valid_from"] if claim_snapshot is not None else ""
            valid_until_raw = claim_snapshot["valid_until"] if claim_snapshot is not None else ""
            valid_from = self._knowledge_as_of(str(valid_from_raw)) if valid_from_raw else None
            valid_until = self._knowledge_as_of(str(valid_until_raw)) if valid_until_raw else None
            if valid_from is not None and reference < valid_from:
                continue
            if valid_until is not None and reference >= valid_until:
                continue
            attested_at = self._knowledge_as_of(attestation.verified_at) if attestation is not None else None
            age_seconds = (reference - attested_at).total_seconds() if attested_at is not None else None
            is_fresh = bool(age_seconds is not None and 0 <= age_seconds <= max_age_seconds)
            is_verified = bool(attestation is not None and verification_status == "verified")
            is_active = (
                status == "SUPPORTED"
                and is_verified
                and is_fresh
                and bool(attestation.evidence_snapshots if attestation is not None else [])
                and not open_conflicts.get(str(row["claim_id"]), 0)
                and not (attestation.conflict_suppression_ids if attestation is not None else [])
            )
            if not include_unverified and not is_active:
                continue
            item = dict(row)
            if claim_snapshot is not None:
                item.update(claim_snapshot)
                item["last_verified_at"] = attestation.verified_at
            else:
                item.update(
                    {
                        "status": "PROPOSED",
                        "verification_status": "unverified",
                        "last_verified_at": "",
                    }
                )
            if "metadata" not in item:
                item["metadata"] = json.loads(str(row["metadata_json"]))
            item.pop("metadata_json", None)
            item.update(
                {
                    "source_count": attested_source_counts.get(str(row["claim_id"]), 0),
                    "conflict_count": open_conflicts.get(str(row["claim_id"]), 0),
                    "is_verified": is_verified,
                    "is_fresh": is_fresh,
                    "is_active": is_active,
                    "is_attested": attestation is not None,
                    "attestation_id": attestation.attestation_id if attestation is not None else "",
                    "conflict_suppression_ids": (
                        list(attestation.conflict_suppression_ids) if attestation is not None else []
                    ),
                    "match_score": score,
                }
            )
            results.append(item)
        results.sort(key=lambda item: (-int(item["match_score"]), -float(item["confidence"]), str(item["claim_id"])))
        return results[:limit]

    def search_knowledge(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_TASK_GOAL_LENGTH:
            raise KnowledgeValidationError("knowledge_query_invalid")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > 100:
            raise KnowledgeValidationError("knowledge_query_limit_invalid")
        # Legacy knowledge_items do not bind their status, reliability, source,
        # evidence, or verification time to a trusted provenance attestation.
        # Fail closed for planner retrieval without changing or deleting rows.
        result: list[dict[str, Any]] = []
        self._emit_observability(
            "MEMORY_RETRIEVED",
            metadata={"memory_type": "knowledge", "result_count": len(result)},
        )
        return result

    def recent_episodes(self, limit: int = 10) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute("SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()]

    def save_observation(self, observation: Any) -> None:
        payload = observation.to_dict() if hasattr(observation, "to_dict") else observation
        with self._lease_write(str(payload["task_id"])):
            self.db.execute(
                "INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?,?)",
                (
                    f"{payload['task_id']}:{payload['step_id']}:{payload['timestamp']}",
                    payload["task_id"],
                    payload["step_id"],
                    payload["tool_name"],
                    payload["execution_status"],
                    json.dumps(payload, ensure_ascii=False),
                    payload["timestamp"],
                ),
            )

    def save_verification(self, task_id: str, step_id: str, tool_name: str, verification: Any) -> None:
        payload = verification.to_dict() if hasattr(verification, "to_dict") else verification
        with self._lease_write(task_id):
            self.db.execute(
                "INSERT OR REPLACE INTO verifications VALUES (?,?,?,?,?,?,?)",
                (
                    f"{task_id}:{step_id}:{payload.get('verifier_type', 'unknown')}:{payload.get('status')}",
                    task_id,
                    step_id,
                    tool_name,
                    payload["status"],
                    json.dumps(payload, ensure_ascii=False),
                    now(),
                ),
            )

    def save_event(self, task_id: str, event_type: str, event: dict[str, Any], *, step_id: str | None = None) -> None:
        stamp = now()
        with self._lease_write(task_id):
            self.db.execute(
                "INSERT OR REPLACE INTO execution_events VALUES (?,?,?,?,?,?)",
                (f"{task_id}:{event_type}:{stamp}", task_id, step_id, event_type, json.dumps(event, ensure_ascii=False), stamp),
            )

    def observations_for_task(self, task_id: str) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = self.db.execute("SELECT * FROM observations WHERE task_id = ? ORDER BY created_at", (task_id,)).fetchall()
        return [dict(row) for row in rows]

    def verifications_for_task(self, task_id: str) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = self.db.execute("SELECT * FROM verifications WHERE task_id = ? ORDER BY created_at", (task_id,)).fetchall()
        return [dict(row) for row in rows]

    def events_for_task(self, task_id: str) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = self.db.execute("SELECT * FROM execution_events WHERE task_id = ? ORDER BY created_at", (task_id,)).fetchall()
        return [dict(row) for row in rows]

    def save_observability_event(self, payload: dict[str, Any]) -> int:
        """Persist one validated event and return its deterministic sequence number.

        ``event_id`` is unique, so retrying a failed caller cannot duplicate an
        already persisted event. The insert and sequence allocation are atomic.
        """
        metadata_json = json.dumps(payload.get("metadata", {}), ensure_ascii=False, sort_keys=True)
        task_id = str(payload.get("task_id", ""))
        event_type = str(payload.get("event_type", ""))
        # Owner approval events are written outside a worker lease.  Execution,
        # tool, task, observation, verification, and security events tied to a
        # queued task must remain fenced to the active worker.
        requires_lease = bool(task_id) and not event_type.startswith(("APPROVAL_", "API_"))
        sequence: int
        with self._lease_write(task_id if requires_lease else None):
            try:
                self.db.execute(
                    """INSERT INTO observability_events
                    (event_id, event_type, timestamp, task_id, action_id, step_id,
                     correlation_id, component, severity, schema_version, metadata_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        payload["event_id"],
                        payload["event_type"],
                        payload["timestamp"],
                        payload.get("task_id", ""),
                        payload.get("action_id", ""),
                        payload.get("step_id", ""),
                        payload["correlation_id"],
                        payload["component"],
                        payload["severity"],
                        int(payload["schema_version"]),
                        metadata_json,
                    ),
                )
            except sqlite3.IntegrityError:
                row = self.db.execute(
                    "SELECT sequence, metadata_json, event_type FROM observability_events WHERE event_id = ?",
                    (payload["event_id"],),
                ).fetchone()
                if row is None or row["metadata_json"] != metadata_json or row["event_type"] != payload["event_type"]:
                    raise
                sequence = int(row["sequence"])
            else:
                row = self.db.execute(
                    "SELECT sequence FROM observability_events WHERE event_id = ?",
                    (payload["event_id"],),
                ).fetchone()
                if row is None:
                    raise sqlite3.OperationalError("observability_sequence_missing")
                sequence = int(row["sequence"])
        return sequence

    @staticmethod
    def _decode_observability_row(row: sqlite3.Row) -> dict[str, Any]:
        payload = {
            "sequence": int(row["sequence"]),
            "event_id": row["event_id"],
            "event_type": row["event_type"],
            "timestamp": row["timestamp"],
            "task_id": row["task_id"],
            "action_id": row["action_id"],
            "step_id": row["step_id"],
            "correlation_id": row["correlation_id"],
            "component": row["component"],
            "severity": row["severity"],
            "schema_version": int(row["schema_version"]),
        }
        try:
            metadata = json.loads(row["metadata_json"])
        except (TypeError, ValueError):
            metadata = {"observability_error": "malformed_metadata"}
        payload["metadata"] = metadata if isinstance(metadata, dict) else {}
        return payload

    def all_observability_events(self) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = self.db.execute("SELECT * FROM observability_events ORDER BY sequence ASC").fetchall()
        return [self._decode_observability_row(row) for row in rows]

    def observability_events_for_task(self, task_id: str) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = self.db.execute(
                "SELECT * FROM observability_events WHERE task_id = ? ORDER BY sequence ASC",
                (task_id,),
            ).fetchall()
        return [self._decode_observability_row(row) for row in rows]

    def recent_observability_events(self, limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 10000))
        with self._db_lock:
            rows = self.db.execute(
                "SELECT * FROM observability_events ORDER BY sequence DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [self._decode_observability_row(row) for row in reversed(rows)]

    def save_experience(self, experience: Any) -> None:
        payload = experience.to_dict() if hasattr(experience, "to_dict") else experience
        task_id = str(payload.get("task_id", ""))
        with self._lease_write(task_id):
            self.db.execute(
                "INSERT OR REPLACE INTO experiences VALUES (?,?,?,?)",
                (payload["task_id"], payload["goal"], json.dumps(payload, ensure_ascii=False), payload.get("created_at", now())),
            )
        self._emit_observability(
            "MEMORY_STORED",
            task_id=str(payload.get("task_id", "")),
            metadata={"memory_type": "experience", "action_count": len(payload.get("actions", []))},
        )

    def get_experience(self, task_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            row = self.db.execute("SELECT * FROM experiences WHERE task_id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    def all_experiences(self, *, owner_user_id: str | None = None) -> list[dict[str, Any]]:
        with self._db_lock:
            if owner_user_id is None:
                rows = self.db.execute("SELECT * FROM experiences ORDER BY created_at DESC").fetchall()
            elif not isinstance(owner_user_id, str) or not owner_user_id.strip():
                return []
            else:
                rows = self.db.execute(
                    """SELECT e.* FROM experiences AS e
                       JOIN task_owners AS o ON o.task_id = e.task_id
                       WHERE o.user_id = ?
                       ORDER BY e.created_at DESC""",
                    (owner_user_id,),
                ).fetchall()
        return [dict(row) for row in rows]

    def experience_task_ids_for_owner(self, owner_user_id: str) -> set[str]:
        if not isinstance(owner_user_id, str) or not owner_user_id.strip():
            return set()
        with self._db_lock:
            rows = self.db.execute(
                """SELECT e.task_id FROM experiences AS e
                   JOIN task_owners AS o ON o.task_id = e.task_id
                   WHERE o.user_id = ?""",
                (owner_user_id,),
            ).fetchall()
        return {str(row["task_id"]) for row in rows}

    def save_reflection(self, insight: Any) -> None:
        payload = insight.to_dict() if hasattr(insight, "to_dict") else insight
        active = self._active_lease()
        active_task_id = active[0] if active is not None else None
        with self._lease_write(active_task_id):
            self.db.execute(
                "INSERT OR REPLACE INTO reflection_insights VALUES (?,?,?,?,?)",
                (
                    payload["insight_id"],
                    payload.get("pattern", ""),
                    float(payload.get("confidence", 0.0)),
                    json.dumps(payload, ensure_ascii=False),
                    payload.get("created_at", now()),
                ),
            )
        self._emit_observability(
            "REFLECTION_COMPLETED",
            metadata={"insight_id": payload.get("insight_id", ""), "confidence": payload.get("confidence", 0.0)},
        )

    def get_reflection(self, insight_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            row = self.db.execute("SELECT * FROM reflection_insights WHERE insight_id = ?", (insight_id,)).fetchone()
        return dict(row) if row else None

    def all_reflections(self) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = self.db.execute("SELECT * FROM reflection_insights ORDER BY confidence DESC, insight_id ASC").fetchall()
        return [dict(row) for row in rows]

    def save_learned_strategy(self, strategy: Any) -> None:
        payload = strategy.to_dict() if hasattr(strategy, "to_dict") else strategy
        active = self._active_lease()
        active_task_id = active[0] if active is not None else None
        with self._lease_write(active_task_id):
            self.db.execute(
                "INSERT OR REPLACE INTO learned_strategies VALUES (?,?,?,?,?)",
                (
                    payload["strategy_id"],
                    payload.get("status", "uncertain"),
                    float(payload.get("confidence", 0.0)),
                    json.dumps(payload, ensure_ascii=False),
                    payload.get("created_at", now()),
                ),
            )
        self._emit_observability(
            "LEARNING_COMPLETED",
            metadata={"strategy_id": payload.get("strategy_id", ""), "status": payload.get("status", "")},
        )

    def get_learned_strategy(self, strategy_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            row = self.db.execute("SELECT * FROM learned_strategies WHERE strategy_id = ?", (strategy_id,)).fetchone()
        return dict(row) if row else None

    def all_learned_strategies(self) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = self.db.execute(
                "SELECT * FROM learned_strategies ORDER BY confidence DESC, strategy_id ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def save_skill_candidate(self, payload: dict[str, Any]) -> None:
        stamp = now()
        with self._db_lock, self.db:
            self.db.execute(
                """INSERT OR REPLACE INTO skill_candidates
                (skill_id, name, version, parent_version, status, candidate_json,
                 contract_json, analysis_json, mutation_json, runtime_json,
                 cross_verification_json, approval_json, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    payload["skill_id"],
                    payload["name"],
                    int(payload["version"]),
                    payload.get("parent_version"),
                    payload.get("status", "CANDIDATE"),
                    json.dumps(payload.get("candidate", {}), ensure_ascii=False),
                    json.dumps(payload.get("contract", {}), ensure_ascii=False),
                    json.dumps(payload.get("analysis", {}), ensure_ascii=False),
                    json.dumps(payload.get("mutation", {}), ensure_ascii=False),
                    json.dumps(payload.get("runtime", {}), ensure_ascii=False),
                    json.dumps(payload.get("cross_verification", {}), ensure_ascii=False),
                    json.dumps(payload.get("approval", {}), ensure_ascii=False),
                    payload.get("created_at", stamp),
                    stamp,
                ),
            )
        self._emit_observability(
            "SKILL_CANDIDATE",
            action_id=str(payload.get("skill_id", "")),
            metadata={
                "name": payload.get("name", ""),
                "version": payload.get("version", 0),
                "status": payload.get("status", "CANDIDATE"),
            },
        )

    def get_skill_candidate(self, skill_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            row = self.db.execute("SELECT * FROM skill_candidates WHERE skill_id = ?", (skill_id,)).fetchone()
        return dict(row) if row else None

    def get_skill_version(self, name: str, version: int) -> dict[str, Any] | None:
        with self._db_lock:
            row = self.db.execute(
                "SELECT * FROM skill_candidates WHERE name = ? AND version = ?",
                (name, version),
            ).fetchone()
        return dict(row) if row else None

    def all_skill_candidates(self, name: str | None = None) -> list[dict[str, Any]]:
        with self._db_lock:
            if name is None:
                rows = self.db.execute("SELECT * FROM skill_candidates ORDER BY name, version").fetchall()
            else:
                rows = self.db.execute(
                    "SELECT * FROM skill_candidates WHERE name = ? ORDER BY version",
                    (name,),
                ).fetchall()
        return [dict(row) for row in rows]

    def update_skill_status(self, skill_id: str, status: str) -> None:
        with self._db_lock, self.db:
            row = self.db.execute(
                "SELECT candidate_json FROM skill_candidates WHERE skill_id = ?",
                (skill_id,),
            ).fetchone()
            candidate_json = row["candidate_json"] if row else "{}"
            try:
                candidate_payload = json.loads(candidate_json)
            except (TypeError, ValueError):
                candidate_payload = {}
            candidate_payload["status"] = status
            self.db.execute(
                "UPDATE skill_candidates SET status = ?, candidate_json = ?, updated_at = ? WHERE skill_id = ?",
                (status, json.dumps(candidate_payload, ensure_ascii=False), now(), skill_id),
            )

    def save_skill_metrics(self, payload: dict[str, Any]) -> None:
        with self._db_lock, self.db:
            self.db.execute(
                """INSERT OR REPLACE INTO skill_runtime_metrics
                (skill_id, name, version, executions, successes, failures,
                 verification_failures, timeouts, contract_violations, rollback_count,
                 success_rate, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    payload["skill_id"],
                    payload["name"],
                    int(payload["version"]),
                    int(payload.get("executions", 0)),
                    int(payload.get("successes", 0)),
                    int(payload.get("failures", 0)),
                    int(payload.get("verification_failures", 0)),
                    int(payload.get("timeouts", 0)),
                    int(payload.get("contract_violations", 0)),
                    int(payload.get("rollback_count", 0)),
                    float(payload.get("success_rate", 0.0)),
                    payload.get("updated_at", now()),
                ),
            )

    def get_skill_metrics(self, skill_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            row = self.db.execute(
                "SELECT * FROM skill_runtime_metrics WHERE skill_id = ?",
                (skill_id,),
            ).fetchone()
        return dict(row) if row else None

    def save_skill_event(self, skill_id: str, event_type: str, event: dict[str, Any]) -> None:
        stamp = now()
        event_id = f"{skill_id}:{event_type}:{stamp}"
        with self._db_lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO skill_lifecycle_events VALUES (?,?,?,?,?)",
                (event_id, skill_id, event_type, json.dumps(event, ensure_ascii=False), stamp),
            )
        mapping = {
            "activated": "SKILL_ACTIVATED",
            "rollback": "SKILL_ROLLED_BACK",
            "status_changed": "SKILL_REGISTRY_UPDATED",
            "candidate_rejected_without_overwrite": "SKILL_BLOCKED",
        }
        status = str(event.get("status", "")).upper()
        observability_type = mapping.get(event_type)
        if status in {"QUARANTINED", "REJECTED", "BLOCKED", "DEGRADED"}:
            observability_type = "SKILL_BLOCKED"
        if observability_type:
            self._emit_observability(
                observability_type,
                action_id=skill_id,
                metadata={"lifecycle_event": event_type, **event},
            )

    def skill_events(self, skill_id: str) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = self.db.execute(
                "SELECT * FROM skill_lifecycle_events WHERE skill_id = ? ORDER BY created_at",
                (skill_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_security_policy(self, payload: dict[str, Any]) -> None:
        with self._db_lock, self.db:
            version = int(payload["policy_version"])
            self.db.execute(
                "INSERT OR REPLACE INTO security_policies VALUES (?,?,?)",
                (version, json.dumps(payload, ensure_ascii=False), now()),
            )
            state = self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='owner_policy_state'"
            ).fetchone()
            if state:
                self.db.execute(
                    "UPDATE owner_policy_state SET current_version = MAX(current_version, ?), updated_at = ? WHERE singleton = 1",
                    (version, now()),
                )

    def get_security_policy(self, policy_version: int) -> dict[str, Any] | None:
        with self._db_lock:
            row = self.db.execute(
                "SELECT * FROM security_policies WHERE policy_version = ?",
                (int(policy_version),),
            ).fetchone()
        return dict(row) if row else None

    def get_latest_security_policy(self) -> dict[str, Any] | None:
        with self._db_lock:
            row = self.db.execute(
                "SELECT * FROM security_policies ORDER BY policy_version DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def save_approval_request(
        self,
        payload: dict[str, Any],
        *,
        enforce_lease: bool = False,
    ) -> None:
        if not isinstance(payload, dict):
            raise ValueError("approval_payload_required")
        if str(payload.get("status", "")) != "PENDING":
            raise ValueError("approval_transition_requires_gate")
        task_id = str(payload["task_id"])
        with self._lease_write(task_id if enforce_lease else None):
            self._write_approval_payload(payload, task_id)

    def _write_approval_payload(self, payload: dict[str, Any], task_id: str) -> None:
        self.db.execute(
            """INSERT OR REPLACE INTO approval_requests
            (approval_id, task_id, action_id, tool_or_skill, requested_capabilities_json,
             risk_level, reason, planned_effect, input_fingerprint, policy_version,
             requested_at, expires_at, status, decided_at, actor, approval_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                payload["approval_id"],
                task_id,
                payload["action_id"],
                payload["tool_or_skill"],
                json.dumps(payload.get("requested_capabilities", []), ensure_ascii=False),
                payload["risk_level"],
                payload["reason"],
                payload["planned_effect"],
                payload["input_fingerprint"],
                int(payload["policy_version"]),
                payload["requested_at"],
                payload["expires_at"],
                payload["status"],
                payload.get("decided_at", ""),
                payload.get("actor", "system"),
                json.dumps(payload, ensure_ascii=False),
            ),
        )

    def _save_approval_transition(
        self,
        payload: dict[str, Any],
        *,
        gate: Any,
        enforce_lease: bool = False,
    ) -> None:
        """Persist a final approval transition only for the bound gate.

        Public storage callers can create pending requests, but cannot write a
        final decision. The gate-bound transition rechecks the persisted binding
        and the current policy version in one SQLite write transaction.
        """
        if self._approval_gate is not gate:
            raise ValueError("approval_transition_requires_gate")
        if not isinstance(payload, dict):
            raise ValueError("approval_payload_required")
        if str(payload.get("status", "")) not in {"APPROVED", "DENIED", "CANCELLED", "EXPIRED"}:
            raise ValueError("approval_transition_invalid_status")
        task_id = str(payload["task_id"])
        approving = payload["status"] == "APPROVED"
        policy_lock = getattr(gate, "policy_version_lock", None)
        current_policy_version = getattr(gate, "current_policy_version", None)
        if approving and (
            policy_lock is None
            or not callable(getattr(policy_lock, "acquire", None))
            or not callable(getattr(policy_lock, "release", None))
            or not callable(current_policy_version)
        ):
            raise ValueError("approval_policy_unavailable")
        if approving:
            policy_lock.acquire()
        binding_fields = (
            "approval_id", "task_id", "action_id", "tool_or_skill",
            "requested_capabilities", "risk_level", "reason", "planned_effect",
            "input_fingerprint", "policy_version", "requested_at", "expires_at", "goal",
        )
        try:
            with self._lease_write(task_id if enforce_lease else None):
                row = self.db.execute(
                    "SELECT approval_json FROM approval_requests WHERE approval_id = ?",
                    (payload["approval_id"],),
                ).fetchone()
                if row is None:
                    raise ValueError("approval_not_found")
                try:
                    current = json.loads(row["approval_json"])
                except (TypeError, ValueError) as exc:
                    raise ValueError("approval_payload_corrupt") from exc
                if not isinstance(current, dict) or current.get("status") != "PENDING":
                    raise ValueError("approval_transition_requires_pending")
                if any(current.get(field) != payload.get(field) for field in binding_fields):
                    raise ValueError("approval_binding_mismatch")
                if approving:
                    try:
                        current_version = current_policy_version()
                        approved_version = int(payload.get("policy_version", -1))
                    except Exception as exc:
                        raise ValueError("approval_policy_unavailable") from exc
                    if approved_version != current_version:
                        raise ValueError("approval_policy_stale")
                self._write_approval_payload(payload, task_id)
                self._persist_task_result_event_for_current_state_in_transaction(task_id)
        finally:
            if approving:
                policy_lock.release()

    def get_approval_request(self, approval_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            row = self.db.execute(
                "SELECT approval_json FROM approval_requests WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
        if not row:
            return None
        try:
            payload = json.loads(row["approval_json"])
        except (TypeError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def all_approval_requests(self) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = self.db.execute(
                "SELECT approval_json FROM approval_requests ORDER BY requested_at"
            ).fetchall()
        requests: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["approval_json"])
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict):
                requests.append(payload)
        return requests

    def save_security_decision(self, payload: dict[str, Any]) -> None:
        decision_id = f"{payload['task_id']}:{payload['action_id']}:{payload['evaluated_at']}"
        with self._lease_write(str(payload["task_id"])):
            self.db.execute(
                "INSERT OR REPLACE INTO security_decisions VALUES (?,?,?,?,?)",
                (
                    decision_id,
                    payload["task_id"],
                    payload["action_id"],
                    json.dumps(payload, ensure_ascii=False),
                    payload.get("evaluated_at", now()),
                ),
            )

    def security_decisions_for_task(self, task_id: str) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = self.db.execute(
                "SELECT decision_json FROM security_decisions WHERE task_id = ? ORDER BY created_at",
                (task_id,),
            ).fetchall()
        decisions: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["decision_json"])
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict):
                decisions.append(payload)
        return decisions

    def save_audit_event(self, payload: dict[str, Any], *, requires_lease: bool | None = None) -> None:
        event_type = str(payload.get("event_type", ""))
        task_id = str(payload.get("task_id", ""))
        if requires_lease is None:
            requires_lease = event_type in {"ACTION_EXECUTED", "ACTION_FAILED", "ACTION_BLOCKED"}
        with self._lease_write(task_id if requires_lease else None):
            self.db.execute(
                """INSERT OR REPLACE INTO audit_events
                (event_id, timestamp, event_type, task_id, action_id, actor,
                 tool_or_skill, capabilities_json, risk_level, policy_version,
                 approval_id, decision, reason, event_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    payload["event_id"],
                    payload["timestamp"],
                    event_type,
                    task_id,
                    payload["action_id"],
                    payload["actor"],
                    payload["tool_or_skill"],
                    json.dumps(payload.get("capabilities", []), ensure_ascii=False),
                    payload["risk_level"],
                    int(payload["policy_version"]),
                    payload.get("approval_id", ""),
                    payload["decision"],
                    payload["reason"],
                    json.dumps(payload, ensure_ascii=False),
                ),
            )

    def all_audit_events(self, *, task_id: str | None = None) -> list[dict[str, Any]]:
        with self._db_lock:
            if task_id is None:
                rows = self.db.execute("SELECT event_json FROM audit_events ORDER BY timestamp").fetchall()
            else:
                rows = self.db.execute(
                    "SELECT event_json FROM audit_events WHERE task_id = ? ORDER BY timestamp",
                    (task_id,),
                ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["event_json"])
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return events

    def _emit_observability(
        self,
        event_type: str,
        *,
        task_id: str = "",
        action_id: str = "",
        step_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        observer = getattr(self, "observability", None)
        if observer is None:
            return
        try:
            observer.emit(
                event_type,
                task_id=task_id,
                action_id=action_id,
                step_id=step_id,
                metadata=metadata or {},
                component="storage",
            )
        except Exception:
            # ObservabilityMonitor's default is degraded mode. This guard also
            # protects legacy/custom observers from changing storage semantics.
            return

    def close(self) -> None:
        current_thread_id = threading.get_ident()
        with self._lifecycle:
            if self._closed:
                return
            if current_thread_id in self._worker_thread_ids:
                raise RuntimeError("store_close_from_worker_thread")
            if self.db.in_transaction:
                raise StoreTransactionActiveError("store_transaction_active")
            if self._closing:
                while self._closing and not self._closed:
                    self._lifecycle.wait()
                if self._closed:
                    return
            self._worker_shutdown_requested = True
            self._closing = True
            workers = tuple(self._background_workers)
            self._lifecycle.notify_all()

        for worker in workers:
            try:
                worker.request_stop()
            except BaseException as exc:
                # A broken shutdown hook must not cause an early SQLite close.
                LOGGER.error("store_worker_request_stop_failed", extra={"error_type": type(exc).__name__})
                stop_event = getattr(worker, "_stop_event", None)
                set_stop = getattr(stop_event, "set", None)
                if callable(set_stop):
                    try:
                        set_stop()
                    except BaseException as fallback_exc:
                        LOGGER.error(
                            "store_worker_stop_signal_fallback_failed",
                            extra={"error_type": type(fallback_exc).__name__},
                        )

        for worker in workers:
            thread = getattr(worker, "_loop_thread", None)
            if thread is threading.current_thread():
                raise RuntimeError("store_close_from_worker_thread")
            try:
                worker.join(timeout=None)
            except BaseException:
                # Retain the hard safety invariant even if the helper itself fails.
                if thread is not None and thread.ident is not None:
                    try:
                        thread.join()
                    except BaseException as join_exc:
                        LOGGER.error("store_worker_thread_join_failed", extra={"error_type": type(join_exc).__name__})
            if thread is not None and thread.is_alive():
                thread.join()

        with self._lifecycle:
            while self._background_workers or self._worker_thread_ids:
                self._lifecycle.wait()

        try:
            with self._db_lock:
                sqlite3.Connection.close(self.db)
                self.db._store_release_transaction_lock()
        except BaseException:
            with self._lifecycle:
                self._closing = False
                self._lifecycle.notify_all()
            raise
        with self._lifecycle:
            self._closed = True
            self._lifecycle.notify_all()
