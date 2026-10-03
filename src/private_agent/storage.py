from __future__ import annotations

import json
import sqlite3
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


class Store:
    def __init__(self, path: str | Path = "data/agent.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.init_schema()

    def init_schema(self) -> None:
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
        """)
        self.db.execute(
            "INSERT OR IGNORE INTO task_state_schema(schema_version, applied_at) VALUES (?,?)",
            (1, now()),
        )
        self.db.commit()

    def save_task(self, task_id: str, goal: str, status: str, plan: Any, result: Any, attempts: int) -> None:
        stamp = now()
        self.db.execute("INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?,?,?)",
            (task_id, goal, status, json.dumps(plan), json.dumps(result), attempts, stamp, stamp))
        self.db.commit()

    def save_task_state(self, state: Any, checkpoint_type: str, *, task_status: str | None = None) -> None:
        """Persist state, checkpoint and action records in one SQLite transaction."""
        payload = state.to_dict() if hasattr(state, "to_dict") else dict(state)
        task_id = str(payload["task_id"])
        status = str(task_status or payload["status"])
        created_at = str(payload["created_at"])
        updated_at = str(payload["updated_at"])
        state_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        checkpoint_at = str(payload.get("last_checkpoint_at") or updated_at or now())
        with self.db:
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
        return [item for _, item in sorted(scored, key=lambda x: (x[0], x[1]['confidence']), reverse=True)[:limit]]

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

    def save_experience(self, experience: Any) -> None:
        payload = experience.to_dict() if hasattr(experience, "to_dict") else experience
        self.db.execute(
            "INSERT OR REPLACE INTO experiences VALUES (?,?,?,?)",
            (payload["task_id"], payload["goal"], json.dumps(payload, ensure_ascii=False), payload.get("created_at", now())),
        )
        self.db.commit()

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

    def skill_events(self, skill_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM skill_lifecycle_events WHERE skill_id = ? ORDER BY created_at",
            (skill_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def save_security_policy(self, payload: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO security_policies VALUES (?,?,?)",
            (int(payload["policy_version"]), json.dumps(payload, ensure_ascii=False), now()),
        )
        self.db.commit()

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

    def close(self) -> None:
        self.db.close()
