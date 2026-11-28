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
    task_id: str = ""


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
    failure_actions: list[dict[str, Any]] = field(default_factory=list)
    recovery_transitions: list[dict[str, Any]] = field(default_factory=list)

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
            "failure_actions": self.failure_actions,
            "recovery_transitions": self.recovery_transitions,
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
        if query.task_id:
            owner_user_id = self.store_backend.task_owner_user_id(query.task_id)
            if not owner_user_id:
                return []
            rows = self.store_backend.all_experiences(owner_user_id=owner_user_id)
        else:
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
        identity = dict(payload.get("identity")) if isinstance(payload.get("identity"), dict) else {}
        explicit_verification_status = payload.get("verification_status")
        final_outcome = payload.get("final_outcome")
        normalized_outcome = (
            "EXECUTION_UNKNOWN"
            if explicit_verification_status == "EXECUTION_UNKNOWN"
            else str(final_outcome or "")
        )
        valid_statuses = {"VERIFIED", "FAILED", "INSUFFICIENT", "BLOCKED"}
        consistent_verification = (
            isinstance(explicit_verification_status, str)
            and explicit_verification_status in valid_statuses
            and identity.get("verification_status") == explicit_verification_status
            and identity.get("outcome") == final_outcome
            and isinstance(payload.get("verification_scope"), str)
            and bool(payload.get("verification_scope"))
            and identity.get("verification_scope") == payload.get("verification_scope")
            and (payload.get("security_impact_verified") is None or type(payload.get("security_impact_verified")) is bool)
            and (identity.get("security_impact_verified") is None or type(identity.get("security_impact_verified")) is bool)
            and identity.get("security_impact_verified") == payload.get("security_impact_verified")
            and not (
                explicit_verification_status == "VERIFIED"
                and (final_outcome != "COMPLETED" or payload.get("security_impact_verified") is False)
            )
        )
        if consistent_verification:
            identity["verification_status"] = explicit_verification_status
            identity["verification_scope"] = payload["verification_scope"]
            identity["security_impact_verified"] = payload.get("security_impact_verified")
        elif explicit_verification_status == "EXECUTION_UNKNOWN":
            identity["outcome"] = "EXECUTION_UNKNOWN"
            identity["verification_status"] = "EXECUTION_UNKNOWN"
            identity["verification_scope"] = "UNKNOWN"
            identity["security_impact_verified"] = None
        elif isinstance(explicit_verification_status, str) and explicit_verification_status:
            identity["verification_status"] = "UNVERIFIED"
            identity["verification_scope"] = "CONFLICTED"
            identity["security_impact_verified"] = None
        else:
            identity["verification_status"] = (
                "UNVERIFIED" if final_outcome == "COMPLETED"
                else str(final_outcome or "UNVERIFIED")
            )
            identity["verification_scope"] = (
                "EXECUTION_ONLY" if final_outcome == "COMPLETED" else "UNKNOWN"
            )
            identity["security_impact_verified"] = None
        actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
        actions = [action for action in actions if isinstance(action, dict)]
        strategy_used = [str(action.get("tool_name")) for action in actions if action.get("tool_name")]
        recovery = [str(action.get("recovery", {}).get("strategy")) for action in actions if isinstance(action.get("recovery"), dict) and action.get("recovery", {}).get("strategy")]
        failures = [str(action.get("diagnosis", {}).get("failure_type")) for action in actions if isinstance(action.get("diagnosis"), dict) and action.get("diagnosis", {}).get("failure_type")]
        failure_actions: list[dict[str, Any]] = []
        recovery_transitions: list[dict[str, Any]] = []
        for index, action in enumerate(actions):
            tool_name = action.get("tool_name")
            step_id = action.get("step_id")
            diagnosis = action.get("diagnosis")
            if not isinstance(tool_name, str) or not tool_name or not isinstance(step_id, str) or not step_id:
                continue
            failure_type = diagnosis.get("failure_type") if isinstance(diagnosis, dict) else None
            if isinstance(failure_type, str) and failure_type:
                failure_actions.append({
                    "tool_name": tool_name,
                    "step_id": step_id,
                    "failure_type": failure_type,
                })
            recovery_decision = action.get("recovery")
            if not isinstance(recovery_decision, dict) or recovery_decision.get("strategy") != "CHANGE_TOOL":
                continue
            alternative_tool = recovery_decision.get("replacement_tool")
            if (
                recovery_decision.get("original_tool") != tool_name
                or recovery_decision.get("step_id") != step_id
                or not isinstance(alternative_tool, str)
                or not alternative_tool
                or not isinstance(failure_type, str)
                or not failure_type
            ):
                continue
            alternative_action = next(
                (
                    later for later in actions[index + 1:]
                    if later.get("step_id") == step_id and later.get("tool_name") == alternative_tool
                ),
                None,
            )
            alternative_verification = (
                alternative_action.get("verification")
                if isinstance(alternative_action, dict)
                else None
            )
            alternative_evidence = (
                alternative_verification.get("evidence")
                if isinstance(alternative_verification, dict)
                else None
            )
            recovery_transitions.append({
                "failed_tool": tool_name,
                "alternative_tool": alternative_tool,
                "failure_type": failure_type,
                "recovery_strategy": "CHANGE_TOOL",
                "step_id": step_id,
                "alternative_verified": bool(
                    isinstance(alternative_action, dict)
                    and alternative_action.get("execution_status") == "success"
                    and isinstance(alternative_verification, dict)
                    and alternative_verification.get("status") == "VERIFIED"
                    and alternative_verification.get("verified") is True
                    and isinstance(alternative_evidence, list)
                    and any(isinstance(item, dict) for item in alternative_evidence)
                ),
            })
        evidence: list[dict[str, Any]] = []
        verified_action_tools: list[str] = []
        verified_final_action_tool = ""
        learning_verifier_type = ""
        learning_verifier_trusted = False
        for action in actions:
            tool_name = action.get("tool_name")
            if isinstance(tool_name, str) and tool_name:
                verified_final_action_tool = ""
                learning_verifier_type = ""
                learning_verifier_trusted = False
            verification = action.get("verification")
            if not isinstance(verification, dict):
                continue
            raw_evidence = verification.get("evidence")
            action_evidence = [item for item in raw_evidence if isinstance(item, dict)] if isinstance(raw_evidence, list) else []
            if (
                isinstance(tool_name, str)
                and tool_name
                and action.get("execution_status") == "success"
                and verification.get("status") == "VERIFIED"
                and verification.get("verified") is not False
                and action_evidence
            ):
                verified_action_tools.append(tool_name)
                verified_final_action_tool = tool_name
                raw_verifier_type = verification.get("verifier_type")
                learning_verifier_type = raw_verifier_type if isinstance(raw_verifier_type, str) else ""
                learning_verifier_trusted = verification.get("learning_trusted") is True
                remaining = self.max_evidence_per_experience - len(evidence)
                evidence.extend(action_evidence[:remaining])
        identity["verified_action_tools"] = list(dict.fromkeys(verified_action_tools))
        identity["verified_final_action_tool"] = verified_final_action_tool
        identity["learning_verifier_type"] = learning_verifier_type
        identity["learning_verifier_trusted"] = learning_verifier_trusted
        if (
            identity.get("verification_status") == "VERIFIED"
            and (not strategy_used or verified_final_action_tool != strategy_used[-1])
        ):
            identity["verification_status"] = "UNVERIFIED"
            identity["verification_scope"] = "ACTION_UNVERIFIED"
            identity["security_impact_verified"] = None
            identity["learning_verifier_type"] = ""
            identity["learning_verifier_trusted"] = False
        return RetrievedExperience(
            experience_id=str(payload.get("task_id", "")),
            relevance=0.0,
            task_summary=str(payload.get("goal", ""))[:500],
            strategy_used=strategy_used,
            outcome=normalized_outcome,
            failure_diagnosis=list(dict.fromkeys(failures)),
            recovery_strategy=list(dict.fromkeys(recovery)),
            verification_evidence=evidence[: self.max_evidence_per_experience],
            why_relevant=[],
            identity=identity,
            created_at=str(payload.get("created_at", "")),
            failure_actions=failure_actions,
            recovery_transitions=recovery_transitions,
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
        if (
            candidate.outcome == "COMPLETED"
            and score > 0
            and identity.get("verification_status") == "VERIFIED"
            and identity.get("security_impact_verified") is not False
            and bool(candidate.strategy_used)
            and identity.get("verified_final_action_tool") == candidate.strategy_used[-1]
        ):
            score += 0.05
            reasons.append("verified_outcome")
        elif candidate.outcome == "COMPLETED" and score > 0:
            reasons.append("completed_outcome")
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
