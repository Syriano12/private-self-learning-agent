from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from private_agent.storage import Store, now


@dataclass
class Skill:
    name: str
    version: int
    description: str
    implementation: str
    confidence: float
    success_rate: float
    enabled: bool
    source_knowledge: list[str]
    tests: list[str]


class SkillRegistry:
    """Versioned registry retaining the legacy Skill API and the Phase 8 lifecycle."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def create(self, name: str, description: str, implementation: str, source_knowledge: list[str] | None = None) -> Skill:
        skill = Skill(name, 1, description, implementation, 0.5, 0.0, False, source_knowledge or [], [])
        self.store.db.execute(
            "INSERT OR REPLACE INTO skills VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                skill.name,
                skill.version,
                skill.description,
                skill.implementation,
                skill.confidence,
                skill.success_rate,
                int(skill.enabled),
                json.dumps(skill.source_knowledge),
                json.dumps(skill.tests),
                now(),
            ),
        )
        self.store.db.commit()
        return skill

    def enable(self, name: str, version: int = 1) -> None:
        self.store.db.execute("UPDATE skills SET enabled=1 WHERE name=? AND version=?", (name, version))
        self.store.db.commit()

    def disable(self, name: str, version: int = 1) -> None:
        self.store.db.execute("UPDATE skills SET enabled=0 WHERE name=? AND version=?", (name, version))
        self.store.db.commit()

    def list_enabled(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.store.db.execute("SELECT * FROM skills WHERE enabled=1 ORDER BY name, version").fetchall()
        ]

    def persist_candidate(
        self,
        candidate: Any,
        *,
        contract: Any | None = None,
        analysis: Any | None = None,
        mutation: Any | None = None,
        runtime: Any | None = None,
        sandbox: Any | None = None,
        cross_verification: Any | None = None,
        approval: Any | None = None,
        status: str | None = None,
    ) -> None:
        candidate_payload = candidate.to_dict()
        candidate_payload["status"] = status or candidate_payload.get("status", "CANDIDATE")
        existing_version = self.store.get_skill_version(candidate_payload["name"], int(candidate_payload["version"]))
        if existing_version is not None and existing_version["skill_id"] != candidate_payload["skill_id"]:
            self.store.save_skill_event(
                candidate_payload["skill_id"],
                "candidate_rejected_without_overwrite",
                {
                    "name": candidate_payload["name"],
                    "version": candidate_payload["version"],
                    "existing_skill_id": existing_version["skill_id"],
                },
            )
            return
        runtime_payload = _payload(runtime)
        if sandbox is not None:
            runtime_payload = {"verification": runtime_payload, "sandbox": _payload(sandbox)}
        self.store.save_skill_candidate(
            {
                "skill_id": candidate_payload["skill_id"],
                "name": candidate_payload["name"],
                "version": candidate_payload["version"],
                "parent_version": candidate_payload.get("parent_version"),
                "status": candidate_payload["status"],
                "candidate": candidate_payload,
                "contract": _payload(contract),
                "analysis": _payload(analysis),
                "mutation": _payload(mutation),
                "runtime": runtime_payload,
                "cross_verification": _payload(cross_verification),
                "approval": _payload(approval),
                "created_at": candidate_payload.get("created_at", now()),
            }
        )
        self.store.save_skill_event(
            candidate_payload["skill_id"],
            "status_changed",
            {"status": candidate_payload["status"], "version": candidate_payload["version"]},
        )

    def get_candidate(self, skill_id: str) -> dict[str, Any] | None:
        row = self.store.get_skill_candidate(skill_id)
        return _decode_candidate_row(row) if row else None

    def get_version(self, name: str, version: int) -> dict[str, Any] | None:
        row = self.store.get_skill_version(name, version)
        return _decode_candidate_row(row) if row else None

    def active(self, name: str) -> dict[str, Any] | None:
        rows = self.store.all_skill_candidates(name)
        active = [row for row in rows if row["status"] == "ACTIVE"]
        if not active:
            return None
        return _decode_candidate_row(max(active, key=lambda row: int(row["version"])))

    def activate(self, candidate: Any, *, approval: Any | None = None) -> dict[str, Any]:
        existing = self.active(candidate.name)
        if existing and int(candidate.version) <= int(existing["version"]):
            raise ValueError("active_skill_version_must_increase")
        if self.store.get_skill_candidate(candidate.skill_id) is None:
            self.persist_candidate(candidate, approval=approval, status="ACTIVE")
        else:
            self.store.update_skill_status(candidate.skill_id, "ACTIVE")
        self.store.db.execute(
            "INSERT OR REPLACE INTO skills VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                candidate.name,
                int(candidate.version),
                candidate.description,
                candidate.entry_point,
                float(getattr(candidate, "confidence", 1.0)),
                0.0,
                1,
                json.dumps(candidate.provenance, ensure_ascii=False),
                json.dumps(candidate.tests, ensure_ascii=False),
                now(),
            ),
        )
        self.store.db.commit()
        self.store.save_skill_event(
            candidate.skill_id,
            "activated",
            {"name": candidate.name, "version": candidate.version},
        )
        return self.get_candidate(candidate.skill_id) or {}

    def mark_status(self, skill_id: str, status: str, *, event: dict[str, Any] | None = None) -> None:
        self.store.update_skill_status(skill_id, status)
        self.store.save_skill_event(skill_id, "status_changed", {"status": status, **(event or {})})

    def rollback(self, name: str, version: int | None = None, *, reason: str = "") -> dict[str, Any] | None:
        current = self.get_version(name, int(version)) if version is not None else self.active(name)
        if current is None:
            return None
        rows = self.store.all_skill_candidates(name)
        previous = [
            row
            for row in rows
            if row["status"] == "ACTIVE" and int(row["version"]) < int(current["version"])
        ]
        if not previous:
            self.mark_status(current["skill_id"], "QUARANTINED", event={"reason": "no_previous_known_good_version"})
            return None
        previous_row = max(previous, key=lambda row: int(row["version"]))
        self.store.update_skill_status(current["skill_id"], "ROLLED_BACK")
        self.store.update_skill_status(previous_row["skill_id"], "ACTIVE")
        self.store.db.execute("UPDATE skills SET enabled=0 WHERE name=? AND version=?", (name, int(current["version"])))
        self.store.db.execute("UPDATE skills SET enabled=1 WHERE name=? AND version=?", (name, int(previous_row["version"])))
        self.store.db.commit()
        self.store.save_skill_event(
            current["skill_id"],
            "rollback",
            {
                "from_version": int(current["version"]),
                "to_version": int(previous_row["version"]),
                "reason": reason,
            },
        )
        return _decode_candidate_row(previous_row)

    def record_runtime(
        self,
        candidate: Any,
        *,
        success: bool,
        verification_failed: bool = False,
        timeout: bool = False,
        contract_violation: bool = False,
        rolled_back: bool = False,
    ) -> dict[str, Any]:
        existing = self.store.get_skill_metrics(candidate.skill_id) or {
            "skill_id": candidate.skill_id,
            "name": candidate.name,
            "version": candidate.version,
            "executions": 0,
            "successes": 0,
            "failures": 0,
            "verification_failures": 0,
            "timeouts": 0,
            "contract_violations": 0,
            "rollback_count": 0,
        }
        existing["executions"] += 1
        existing["successes"] += int(success)
        existing["failures"] += int(not success)
        existing["verification_failures"] += int(verification_failed)
        existing["timeouts"] += int(timeout)
        existing["contract_violations"] += int(contract_violation)
        existing["rollback_count"] += int(rolled_back)
        existing["success_rate"] = round(existing["successes"] / existing["executions"], 4)
        existing["updated_at"] = now()
        self.store.save_skill_metrics(existing)
        return existing

    def record_rollback(self, candidate: Any) -> dict[str, Any] | None:
        metrics = self.store.get_skill_metrics(candidate.skill_id)
        if metrics is None:
            return None
        metrics["rollback_count"] = int(metrics.get("rollback_count", 0)) + 1
        metrics["updated_at"] = now()
        self.store.save_skill_metrics(metrics)
        return metrics

    def events(self, skill_id: str) -> list[dict[str, Any]]:
        return self.store.skill_events(skill_id)


def _payload(value: Any | None) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    return {"value": str(value)}


def _decode_candidate_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    try:
        candidate = json.loads(row["candidate_json"])
        contract = json.loads(row["contract_json"])
        analysis = json.loads(row["analysis_json"])
        mutation = json.loads(row["mutation_json"])
        runtime = json.loads(row["runtime_json"])
        cross = json.loads(row["cross_verification_json"])
        approval = json.loads(row["approval_json"])
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "candidate": candidate,
        "contract": contract,
        "analysis": analysis,
        "mutation": mutation,
        "runtime": runtime,
        "cross_verification": cross,
        "approval": approval,
        "status": row["status"],
        "skill_id": row["skill_id"],
        "name": row["name"],
        "version": int(row["version"]),
        "parent_version": row["parent_version"],
    }
