from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


CURRENT_SCHEMA_VERSION = 4


class UnsupportedSchemaVersion(RuntimeError):
    """Raised when a database is newer than this application understands."""


class Store:
    def __init__(self, path: str | Path = "data/agent.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.init_schema()

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
        return ()

    def _assert_migration_objects(self, version: int) -> None:
        for object_type, name in self._required_objects(version):
            row = self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?",
                (object_type, name),
            ).fetchone()
            if row is None:
                raise UnsupportedSchemaVersion(f"inconsistent_schema_version:{version}:missing_{object_type}:{name}")

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
            for version in sorted(versions):
                if version >= 2:
                    self._assert_migration_objects(version)
            for version in range(2, CURRENT_SCHEMA_VERSION + 1):
                if version not in versions:
                    self._apply_migration(version)

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
            self.db.execute("INSERT INTO task_state_schema(schema_version, applied_at) VALUES (?, ?)", (2, now()))
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
                "INSERT INTO owner_policy_state(singleton, current_version, updated_at) VALUES (1, ?, ?)",
                (initial, now()),
            )
            self.db.execute("INSERT INTO task_state_schema(schema_version, applied_at) VALUES (?, ?)", (3, now()))
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
            self.db.execute("INSERT INTO task_state_schema(schema_version, applied_at) VALUES (?, ?)", (4, now()))
            return
        raise UnsupportedSchemaVersion(f"unknown_migration:{version}")

    def latest_owner_policy_version(self) -> int:
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
        with self.db:
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
        with self.db:
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
        if rule_id is None:
            rows = self.db.execute("SELECT * FROM owner_permission_rule_history ORDER BY policy_version, created_at").fetchall()
        else:
            rows = self.db.execute("SELECT * FROM owner_permission_rule_history WHERE rule_id = ? ORDER BY policy_version, created_at", (rule_id,)).fetchall()
        return [dict(row) for row in rows]

    def owner_policy_audit(self) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM owner_policy_audit ORDER BY policy_version, created_at").fetchall()
        return [dict(row) for row in rows]

    def save_task(self, task_id: str, goal: str, status: str, plan: Any, result: Any, attempts: int) -> None:
        stamp = now()
        self.db.execute("INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?,?,?)",
            (task_id, goal, status, json.dumps(plan), json.dumps(result), attempts, stamp, stamp))
        self.db.commit()

    def get_api_idempotency(self, idempotency_key: str) -> dict[str, Any] | None:
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
        self.db.execute(
            "INSERT INTO api_idempotency_keys (idempotency_key, request_fingerprint, task_id, response_json, created_at) VALUES (?,?,?,?,?)",
            (idempotency_key, request_fingerprint, task_id, response_json, now()),
        )
        self.db.commit()

    def _persist_task_state_in_transaction(
        self,
        payload: dict[str, Any],
        checkpoint_type: str,
        *,
        task_status: str | None = None,
    ) -> None:
        """Persist one state snapshot while the caller owns the transaction."""
        task_id = str(payload["task_id"])
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
        with self.db:
            self._persist_task_state_in_transaction(payload, checkpoint_type, task_status=task_status)

    def create_queued_task(
        self,
        goal: str,
        *,
        task_id: str | None = None,
        idempotency_key: str | None = None,
        request_fingerprint: str | None = None,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Atomically create a QUEUED state, checkpoint, queue row, and idempotency record."""
        from private_agent.core.task_state import TaskState

        if not goal or not goal.strip():
            raise ValueError("goal_required")
        if max_attempts < 1:
            raise ValueError("max_attempts_invalid")
        if idempotency_key is not None and not request_fingerprint:
            raise ValueError("request_fingerprint_required")

        self.db.execute("BEGIN IMMEDIATE")
        try:
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
                    self.db.commit()
                    return {"task_id": str(existing["task_id"]), "replayed": True}

            state = TaskState.create_queued(task_id or str(uuid.uuid4()), goal.strip())
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

    def task_queue_record(self, task_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM task_queue WHERE task_id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    def list_task_summaries(
        self,
        *,
        limit: int,
        cursor_created_at: str = "",
        cursor_task_id: str = "",
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if status:
            clauses.append("s.status = ?")
            params.append(status)
        if cursor_created_at and cursor_task_id:
            clauses.append("(s.created_at < ? OR (s.created_at = ? AND s.task_id < ?))")
            params.extend([cursor_created_at, cursor_created_at, cursor_task_id])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
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
                {where}
                ORDER BY s.created_at DESC, s.task_id DESC
                LIMIT ?""",
            (*params, int(limit)),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_task_state(self, task_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT state_json FROM task_states WHERE task_id = ?", (task_id,)).fetchone()
        if not row:
            return None
        try:
            payload = json.loads(row["state_json"])
        except (TypeError, ValueError) as exc:
            raise ValueError("corrupted_task_state_json") from exc
        return payload if isinstance(payload, dict) else None

    def get_task_state_row(self, task_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM task_states WHERE task_id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    def task_checkpoints(self, task_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM task_checkpoints WHERE task_id = ? ORDER BY sequence",
            (task_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def latest_task_checkpoint(self, task_id: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM task_checkpoints WHERE task_id = ? ORDER BY sequence DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        return dict(row) if row else None

    def task_action_records(self, task_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM task_action_records WHERE task_id = ? ORDER BY updated_at",
            (task_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def save_episode(self, episode_id: str, task_id: str, goal: str, outcome: str, lessons: Any, observations: Any) -> None:
        self.db.execute("INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?)",
            (episode_id, task_id, goal, outcome, json.dumps(lessons), json.dumps(observations), now()))
        self.db.commit()

    def add_knowledge(self, item: KnowledgeItem) -> None:
        if not item.created_at: item.created_at = now()
        item.updated_at = now()
        self.db.execute("INSERT OR REPLACE INTO knowledge_items VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", tuple(asdict(item).values()))
        self.db.commit()

    def search_knowledge(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
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
        self.db.commit()

    def save_verification(self, task_id: str, step_id: str, tool_name: str, verification: Any) -> None:
        payload = verification.to_dict() if hasattr(verification, "to_dict") else verification
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
        self.db.commit()

    def save_event(self, task_id: str, event_type: str, event: dict[str, Any], *, step_id: str | None = None) -> None:
        stamp = now()
        self.db.execute(
            "INSERT OR REPLACE INTO execution_events VALUES (?,?,?,?,?,?)",
            (f"{task_id}:{event_type}:{stamp}", task_id, step_id, event_type, json.dumps(event, ensure_ascii=False), stamp),
        )
        self.db.commit()

    def observations_for_task(self, task_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM observations WHERE task_id = ? ORDER BY created_at", (task_id,)).fetchall()
        return [dict(row) for row in rows]

    def verifications_for_task(self, task_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM verifications WHERE task_id = ? ORDER BY created_at", (task_id,)).fetchall()
        return [dict(row) for row in rows]

    def events_for_task(self, task_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM execution_events WHERE task_id = ? ORDER BY created_at", (task_id,)).fetchall()
        return [dict(row) for row in rows]

    def save_observability_event(self, payload: dict[str, Any]) -> int:
        """Persist one validated event and return its deterministic sequence number.

        ``event_id`` is unique, so retrying a failed caller cannot duplicate an
        already persisted event. The insert and sequence allocation are atomic.
        """
        metadata_json = json.dumps(payload.get("metadata", {}), ensure_ascii=False, sort_keys=True)
        with self.db:
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
                return int(row["sequence"])
            row = self.db.execute(
                "SELECT sequence FROM observability_events WHERE event_id = ?",
                (payload["event_id"],),
            ).fetchone()
            if row is None:
                raise sqlite3.OperationalError("observability_sequence_missing")
            return int(row["sequence"])

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
        rows = self.db.execute("SELECT * FROM observability_events ORDER BY sequence ASC").fetchall()
        return [self._decode_observability_row(row) for row in rows]

    def observability_events_for_task(self, task_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM observability_events WHERE task_id = ? ORDER BY sequence ASC",
            (task_id,),
        ).fetchall()
        return [self._decode_observability_row(row) for row in rows]

    def recent_observability_events(self, limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 10000))
        rows = self.db.execute(
            "SELECT * FROM observability_events ORDER BY sequence DESC LIMIT ?",
            (safe_limit,),
        ).fetchall()
        return [self._decode_observability_row(row) for row in reversed(rows)]

    def save_experience(self, experience: Any) -> None:
        payload = experience.to_dict() if hasattr(experience, "to_dict") else experience
        self.db.execute(
            "INSERT OR REPLACE INTO experiences VALUES (?,?,?,?)",
            (payload["task_id"], payload["goal"], json.dumps(payload, ensure_ascii=False), payload.get("created_at", now())),
        )
        self.db.commit()
        self._emit_observability(
            "MEMORY_STORED",
            task_id=str(payload.get("task_id", "")),
            metadata={"memory_type": "experience", "action_count": len(payload.get("actions", []))},
        )

    def get_experience(self, task_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM experiences WHERE task_id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    def all_experiences(self) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM experiences ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def save_reflection(self, insight: Any) -> None:
        payload = insight.to_dict() if hasattr(insight, "to_dict") else insight
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
        self.db.commit()
        self._emit_observability(
            "REFLECTION_COMPLETED",
            metadata={"insight_id": payload.get("insight_id", ""), "confidence": payload.get("confidence", 0.0)},
        )

    def get_reflection(self, insight_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM reflection_insights WHERE insight_id = ?", (insight_id,)).fetchone()
        return dict(row) if row else None

    def all_reflections(self) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM reflection_insights ORDER BY confidence DESC, insight_id ASC").fetchall()
        return [dict(row) for row in rows]

    def save_learned_strategy(self, strategy: Any) -> None:
        payload = strategy.to_dict() if hasattr(strategy, "to_dict") else strategy
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
        self.db.commit()
        self._emit_observability(
            "LEARNING_COMPLETED",
            metadata={"strategy_id": payload.get("strategy_id", ""), "status": payload.get("status", "")},
        )

    def get_learned_strategy(self, strategy_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM learned_strategies WHERE strategy_id = ?", (strategy_id,)).fetchone()
        return dict(row) if row else None

    def all_learned_strategies(self) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM learned_strategies ORDER BY confidence DESC, strategy_id ASC"
        ).fetchall()
        return [dict(row) for row in rows]

    def save_skill_candidate(self, payload: dict[str, Any]) -> None:
        stamp = now()
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
        self.db.commit()
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
        row = self.db.execute("SELECT * FROM skill_candidates WHERE skill_id = ?", (skill_id,)).fetchone()
        return dict(row) if row else None

    def get_skill_version(self, name: str, version: int) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM skill_candidates WHERE name = ? AND version = ?",
            (name, version),
        ).fetchone()
        return dict(row) if row else None

    def all_skill_candidates(self, name: str | None = None) -> list[dict[str, Any]]:
        if name is None:
            rows = self.db.execute("SELECT * FROM skill_candidates ORDER BY name, version").fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM skill_candidates WHERE name = ? ORDER BY version",
                (name,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_skill_status(self, skill_id: str, status: str) -> None:
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
        self.db.commit()

    def save_skill_metrics(self, payload: dict[str, Any]) -> None:
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
        self.db.commit()

    def get_skill_metrics(self, skill_id: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM skill_runtime_metrics WHERE skill_id = ?",
            (skill_id,),
        ).fetchone()
        return dict(row) if row else None

    def save_skill_event(self, skill_id: str, event_type: str, event: dict[str, Any]) -> None:
        stamp = now()
        event_id = f"{skill_id}:{event_type}:{stamp}"
        self.db.execute(
            "INSERT OR REPLACE INTO skill_lifecycle_events VALUES (?,?,?,?,?)",
            (event_id, skill_id, event_type, json.dumps(event, ensure_ascii=False), stamp),
        )
        self.db.commit()
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
        rows = self.db.execute(
            "SELECT * FROM skill_lifecycle_events WHERE skill_id = ? ORDER BY created_at",
            (skill_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def save_security_policy(self, payload: dict[str, Any]) -> None:
        with self.db:
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
        row = self.db.execute(
            "SELECT * FROM security_policies WHERE policy_version = ?",
            (int(policy_version),),
        ).fetchone()
        return dict(row) if row else None

    def get_latest_security_policy(self) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM security_policies ORDER BY policy_version DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def save_approval_request(self, payload: dict[str, Any]) -> None:
        self.db.execute(
            """INSERT OR REPLACE INTO approval_requests
            (approval_id, task_id, action_id, tool_or_skill, requested_capabilities_json,
             risk_level, reason, planned_effect, input_fingerprint, policy_version,
             requested_at, expires_at, status, decided_at, actor, approval_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                payload["approval_id"],
                payload["task_id"],
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
        self.db.commit()

    def get_approval_request(self, approval_id: str) -> dict[str, Any] | None:
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
        self.db.commit()

    def security_decisions_for_task(self, task_id: str) -> list[dict[str, Any]]:
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

    def save_audit_event(self, payload: dict[str, Any]) -> None:
        self.db.execute(
            """INSERT OR REPLACE INTO audit_events
            (event_id, timestamp, event_type, task_id, action_id, actor,
             tool_or_skill, capabilities_json, risk_level, policy_version,
             approval_id, decision, reason, event_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                payload["event_id"],
                payload["timestamp"],
                payload["event_type"],
                payload["task_id"],
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
        self.db.commit()

    def all_audit_events(self, *, task_id: str | None = None) -> list[dict[str, Any]]:
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
        self.db.close()
