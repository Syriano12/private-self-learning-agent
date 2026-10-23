from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from private_agent.core.knowledge import (
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


CURRENT_SCHEMA_VERSION = 7

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
}


class UnsupportedSchemaVersion(RuntimeError):
    """Raised when a database is newer than this application understands."""


class QueueLeaseLost(RuntimeError):
    """Raised when a worker tries to write after losing its queue lease."""


class Store:
    def __init__(self, path: str | Path = "data/agent.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db_lock = _database_lock(self.path)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._lease_local = threading.local()
        self._approval_gate: Any | None = None
        # No trusted knowledge-ingestion boundary exists yet. Caller-supplied
        # source class, content binding, and verification fields remain
        # advisory and cannot create an in-process trust authority.
        with self._db_lock:
            self.init_schema()

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
        return ()

    def _assert_migration_objects(self, version: int) -> None:
        for object_type, name in self._required_objects(version):
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
                expected = _MIGRATION_TABLE_COLUMNS.get(version, {}).get(name, {})
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
                "knowledge_conflicts": ("CHECK (CLAIM_ID <> CONFLICTING_CLAIM_ID)", "FOREIGN KEY (CLAIM_ID)", "FOREIGN KEY (CONFLICTING_CLAIM_ID)"),
                "auth_users": ("CHECK (ROLE IN ('OWNER', 'OPERATOR', 'VIEWER'))", "CHECK (STATUS IN ('ACTIVE', 'DISABLED'))"),
                "auth_sessions": ("FOREIGN KEY (USER_ID)",),
                "task_owners": ("FOREIGN KEY (TASK_ID)", "FOREIGN KEY (USER_ID)"),
            }.get(name, ())
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
                if existing_objects is not None:
                    self._assert_migration_objects(version)
                self._apply_migration(version)
                self._assert_migration_objects(version)

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

    def save_task_state(self, state: Any, checkpoint_type: str, *, task_status: str | None = None) -> None:
        """Persist state, checkpoint and action records in one SQLite transaction."""
        payload = state.to_dict() if hasattr(state, "to_dict") else dict(state)
        with self._lease_write(str(payload["task_id"])):
            self._persist_task_state_in_transaction(payload, checkpoint_type, task_status=task_status)

    def create_queued_task(
        self,
        goal: str,
        *,
        task_id: str | None = None,
        idempotency_key: str | None = None,
        request_fingerprint: str | None = None,
        context: dict[str, Any] | None = None,
        max_attempts: int = 3,
        owner_user_id: str | None = None,
    ) -> dict[str, Any]:
        """Atomically create a QUEUED state, checkpoint, queue row, and idempotency record."""
        from private_agent.core.task_state import TaskState

        if not goal or not goal.strip():
            raise ValueError("goal_required")
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

            state = TaskState.create_queued(task_id or str(uuid.uuid4()), goal.strip(), context=context_payload)
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
        from private_agent.core.task_state import StateIntegrityError, TERMINAL_STATUSES, TaskState

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
                from private_agent.core.task_state import TaskState, TERMINAL_STATUSES

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
                    elif state_status == "FAILED" and state.status not in TERMINAL_STATUSES:
                        state.result = {
                            **state.result,
                            "final_status": "FAILED",
                            "errors": list(state.result.get("errors", [])) + [error_message or error_code or "worker_failed"],
                        }
                        state.checkpoint("WORKER_FAILED", status="FAILED", phase="WORKER_FAILED", successful=False)
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

            from private_agent.core.task_state import TaskState, TERMINAL_STATUSES

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
        self._db_lock.acquire()
        try:
            self.db.execute("BEGIN IMMEDIATE")
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
            for action_id, approval_id in task_state.approval_ids.items():
                approval = self.get_approval_request(str(approval_id))
                if (
                    not approval
                    or approval.get("status") != "APPROVED"
                    or str(approval.get("task_id", "")) != str(task_id)
                    or str(approval.get("action_id", "")) != str(action_id)
                ):
                    raise ValueError("approval_not_approved")
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
            self._db_lock.release()

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

    def save_knowledge_conflict(self, conflict: KnowledgeConflict) -> KnowledgeConflict:
        if not isinstance(conflict, KnowledgeConflict):
            raise KnowledgeValidationError("conflict_type_required")
        with self._lease_write(self._knowledge_lease_task_id()):
            return self._insert_knowledge_conflict(conflict)

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

    def search_knowledge_claims(
        self,
        query: str,
        *,
        limit: int = 8,
        as_of: str | None = None,
        max_age_seconds: int = 30 * 24 * 60 * 60,
        include_unverified: bool = False,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise KnowledgeValidationError("knowledge_query_required")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > 100:
            raise KnowledgeValidationError("knowledge_query_limit_invalid")
        if not isinstance(max_age_seconds, int) or isinstance(max_age_seconds, bool) or max_age_seconds < 1 or max_age_seconds > 10 * 365 * 24 * 60 * 60:
            raise KnowledgeValidationError("knowledge_freshness_invalid")
        reference = self._knowledge_as_of(as_of)
        terms = [term.lower() for term in query.split() if len(term) > 2]
        if not terms:
            return []
        rows = self.db.execute("SELECT * FROM knowledge_claims").fetchall()
        open_conflicts: dict[str, int] = {}
        for row in self.db.execute("SELECT claim_id, conflicting_claim_id FROM knowledge_conflicts WHERE status = 'OPEN'").fetchall():
            open_conflicts[str(row["claim_id"])] = open_conflicts.get(str(row["claim_id"]), 0) + 1
            open_conflicts[str(row["conflicting_claim_id"])] = open_conflicts.get(str(row["conflicting_claim_id"]), 0) + 1
        source_counts = {
            str(row["claim_id"]): int(row["count"])
            for row in self.db.execute(
                "SELECT claim_id, COUNT(DISTINCT source_id) AS count FROM knowledge_evidence GROUP BY claim_id"
            ).fetchall()
        }
        # Without a trusted attestation boundary, no persisted evidence is
        # eligible to make a claim active or verified.
        evidence_state: dict[str, int] = {}
        results: list[dict[str, Any]] = []
        for row in rows:
            status = str(row["status"])
            verification_status = str(row["verification_status"])
            if status in {"RETIRED", "QUARANTINED"}:
                continue
            text = f"{row['subject']} {row['predicate']} {row['object_text']} {row['domain']}".lower()
            score = sum(term in text for term in terms)
            if not score:
                continue
            valid_from = self._knowledge_as_of(str(row["valid_from"])) if row["valid_from"] else None
            valid_until = self._knowledge_as_of(str(row["valid_until"])) if row["valid_until"] else None
            if valid_from is not None and reference < valid_from:
                continue
            if valid_until is not None and reference >= valid_until:
                continue
            last_verified = self._knowledge_as_of(str(row["last_verified_at"])) if row["last_verified_at"] else None
            is_fresh = bool(last_verified is not None and (reference - last_verified).total_seconds() <= max_age_seconds)
            is_verified = verification_status == "verified"
            is_active = (
                status == "SUPPORTED"
                and is_verified
                and is_fresh
                and evidence_state.get(str(row["claim_id"]), 0) > 0
                and not open_conflicts.get(str(row["claim_id"]), 0)
            )
            if not include_unverified and not is_active:
                continue
            item = dict(row)
            item["metadata"] = json.loads(str(row["metadata_json"]))
            item.pop("metadata_json", None)
            item.update(
                {
                    "source_count": source_counts.get(str(row["claim_id"]), 0),
                    "conflict_count": open_conflicts.get(str(row["claim_id"]), 0),
                    "is_verified": is_verified,
                    "is_fresh": is_fresh,
                    "is_active": is_active,
                    "match_score": score,
                }
            )
            results.append(item)
        results.sort(key=lambda item: (-int(item["match_score"]), -float(item["confidence"]), str(item["claim_id"])))
        return results[:limit]

    def search_knowledge(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip() or len(query) > 1024:
            raise KnowledgeValidationError("knowledge_query_invalid")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > 100:
            raise KnowledgeValidationError("knowledge_query_limit_invalid")
        terms = [t.lower() for t in query.split() if len(t) > 2]
        rows = self.db.execute("SELECT * FROM knowledge_items ORDER BY confidence DESC, updated_at DESC").fetchall()
        scored = []
        for row in rows:
            text = f"{row['concept']} {row['content']} {row['domain']}".lower()
            score = sum(term in text for term in terms)
            if score: scored.append((score, dict(row)))
        result = [item for _, item in sorted(scored, key=lambda x: (x[0], x[1]['confidence']), reverse=True)[:limit]]
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

    def all_experiences(self) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = self.db.execute("SELECT * FROM experiences ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

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
        final decision. The gate-bound transition also rechecks the persisted
        binding inside the same transaction before replacing the row.
        """
        if self._approval_gate is not gate:
            raise ValueError("approval_transition_requires_gate")
        if not isinstance(payload, dict):
            raise ValueError("approval_payload_required")
        if str(payload.get("status", "")) not in {"APPROVED", "DENIED", "CANCELLED", "EXPIRED"}:
            raise ValueError("approval_transition_invalid_status")
        task_id = str(payload["task_id"])
        binding_fields = (
            "approval_id", "task_id", "action_id", "tool_or_skill",
            "requested_capabilities", "risk_level", "reason", "planned_effect",
            "input_fingerprint", "policy_version", "requested_at", "expires_at", "goal",
        )
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
            self._write_approval_payload(payload, task_id)

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
        with self._db_lock:
            self.db.close()
