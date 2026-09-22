from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from private_agent.core.experience import ExperienceRecord, sanitize_value


@dataclass
class ExperienceQuery:
    goal: str
    task_type: str = "general"
    required_capabilities: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    failure_type: str = ""
    recovery_strategy: str = ""


@dataclass
class RetrievedExperience:
    experience_id: str
    relevance: float
    task_summary: str
    strategy_used: list[str]
    outcome: str
    failure_diagnosis: list[str]
    recovery_strategy: list[str]
    verification_evidence: list[dict[str, Any]]
    why_relevant: list[str]
    identity: dict[str, Any]
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return sanitize_value({
            "experience_id": self.experience_id,
            "relevance": round(self.relevance, 4),
            "task_summary": self.task_summary,
            "strategy_used": self.strategy_used,
            "outcome": self.outcome,
            "failure_diagnosis": self.failure_diagnosis,
            "recovery_strategy": self.recovery_strategy,
            "verification_evidence": self.verification_evidence,
            "why_relevant": self.why_relevant,
            "identity": self.identity,
            "created_at": self.created_at,
        })


class ExperienceMemory(ABC):
    """Storage-independent operational experience memory interface."""

    @abstractmethod
    def store(self, experience: ExperienceRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: ExperienceQuery) -> list[RetrievedExperience]:
        raise NotImplementedError


class SQLiteExperienceMemory(ExperienceMemory):
    """SQLite adapter; the agent only depends on the ExperienceMemory interface."""

    def __init__(
        self,
        store: Any,
        *,
        max_results: int = 5,
        max_chars: int = 6000,
        max_evidence_per_experience: int = 3,
    ) -> None:
        self.store_backend = store
        self.max_results = max(1, max_results)
        self.max_chars = max(500, max_chars)
        self.max_evidence_per_experience = max(1, max_evidence_per_experience)

    def store(self, experience: ExperienceRecord) -> None:
        experience.build_identity()
        self.store_backend.save_experience(experience)

    def retrieve(self, query: ExperienceQuery) -> list[RetrievedExperience]:
        rows = self.store_backend.all_experiences()
        scored: list[tuple[float, RetrievedExperience]] = []
        for row in rows:
            payload = self._decode_row(row)
            if not payload:
                continue
            candidate = self._to_candidate(payload)
            score, reasons = self._score(query, payload, candidate)
            if score <= 0:
                continue
            candidate.relevance = score
            candidate.why_relevant = reasons
            scored.append((score, candidate))
        scored.sort(key=lambda item: (-item[0], self._recency_key(item[1].created_at)))
        selected: list[RetrievedExperience] = []
        used_chars = 0
        for _, candidate in scored:
            item = candidate.to_dict()
            size = len(str(item))
            if selected and used_chars + size > self.max_chars:
                continue
            selected.append(candidate)
            used_chars += size
            if len(selected) >= self.max_results:
                break
        return selected

    def _decode_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        import json

        try:
            payload = json.loads(row["experience_json"])
        except (KeyError, TypeError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def _to_candidate(self, payload: dict[str, Any]) -> RetrievedExperience:
        identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
        actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
        strategy_used = [str(action.get("tool_name")) for action in actions if action.get("tool_name")]
        recovery = [str(action.get("recovery", {}).get("strategy")) for action in actions if isinstance(action.get("recovery"), dict) and action.get("recovery", {}).get("strategy")]
        failures = [str(action.get("diagnosis", {}).get("failure_type")) for action in actions if isinstance(action.get("diagnosis"), dict) and action.get("diagnosis", {}).get("failure_type")]
        evidence: list[dict[str, Any]] = []
        for action in actions:
            verification = action.get("verification")
            if isinstance(verification, dict):
                evidence.extend(verification.get("evidence", [])[: self.max_evidence_per_experience])
            if len(evidence) >= self.max_evidence_per_experience:
                break
        return RetrievedExperience(
            experience_id=str(payload.get("task_id", "")),
            relevance=0.0,
            task_summary=str(payload.get("goal", ""))[:500],
            strategy_used=list(dict.fromkeys(strategy_used)),
            outcome=str(payload.get("final_outcome", "")),
            failure_diagnosis=list(dict.fromkeys(failures)),
            recovery_strategy=list(dict.fromkeys(recovery)),
            verification_evidence=evidence[: self.max_evidence_per_experience],
            why_relevant=[],
            identity=identity,
            created_at=str(payload.get("created_at", "")),
        )

    @staticmethod
    def _score(query: ExperienceQuery, payload: dict[str, Any], candidate: RetrievedExperience) -> tuple[float, list[str]]:
        identity = candidate.identity
        score = 0.0
        reasons: list[str] = []
        query_tokens = _tokens(query.goal)
        candidate_tokens = _tokens(candidate.task_summary)
        goal_overlap = _overlap(query_tokens, candidate_tokens)
        if goal_overlap:
            score += min(0.45, goal_overlap * 0.15)
            reasons.append(f"goal_overlap:{goal_overlap}")
        if query.task_type and identity.get("task_type") == query.task_type:
            score += 0.15
            reasons.append("task_type_match")
        candidate_capabilities = set(identity.get("required_capabilities", []))
        capability_overlap = len(candidate_capabilities.intersection(query.required_capabilities))
        if capability_overlap:
            score += min(0.15, capability_overlap * 0.05)
            reasons.append(f"capability_match:{capability_overlap}")
        candidate_tools = set(identity.get("tools", []))
        tool_overlap = len(candidate_tools.intersection(query.tools))
        if tool_overlap:
            score += min(0.15, tool_overlap * 0.05)
            reasons.append(f"tool_match:{tool_overlap}")
        candidate_failures = set(identity.get("failure_types", []))
        if query.failure_type and query.failure_type in candidate_failures:
            score += 0.12
            reasons.append("failure_type_match")
        candidate_recovery = set(identity.get("recovery_strategies", []))
        if query.recovery_strategy and query.recovery_strategy in candidate_recovery:
            score += 0.08
            reasons.append("recovery_strategy_match")
        candidate_constraints = identity.get("constraints", {})
        if query.constraints and isinstance(candidate_constraints, dict):
            matches = sum(candidate_constraints.get(key) == value for key, value in query.constraints.items())
            if matches:
                score += min(0.1, matches * 0.05)
                reasons.append(f"constraint_match:{matches}")
        if candidate.outcome == "COMPLETED" and score > 0:
            score += 0.05
            reasons.append("verified_outcome")
        elif candidate.outcome in {"FAILED", "BLOCKED", "ABORTED"}:
            reasons.append("failure_warning")
        return min(1.0, score), reasons

    @staticmethod
    def _recency_key(timestamp: str) -> float:
        try:
            return -datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError):
            return 0.0


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in value.split() if len(token.strip()) > 2}


def _overlap(left: Iterable[str], right: Iterable[str]) -> int:
    return len(set(left).intersection(right))
