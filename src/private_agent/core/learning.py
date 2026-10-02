from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from private_agent.core.experience import sanitize_value
from private_agent.core.reflection import ReflectionInsight


@dataclass
class LearnedStrategy:
    """Structured, non-executable guidance derived from verified reflection evidence."""

    strategy_id: str
    condition: dict[str, Any]
    preferred_action: dict[str, Any]
    avoided_action: dict[str, Any]
    rationale: str
    supporting_insight_ids: list[str]
    supporting_experience_ids: list[str]
    confidence: float
    applicability: dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "uncertain"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = round(float(self.confidence), 2)
        return sanitize_value(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LearnedStrategy":
        return cls(
            strategy_id=str(payload["strategy_id"]),
            condition=dict(payload.get("condition", {})),
            preferred_action=dict(payload.get("preferred_action", {})),
            avoided_action=dict(payload.get("avoided_action", {})),
            rationale=str(payload.get("rationale", "")),
            supporting_insight_ids=list(payload.get("supporting_insight_ids", [])),
            supporting_experience_ids=list(payload.get("supporting_experience_ids", [])),
            confidence=float(payload.get("confidence", 0.0)),
            applicability=dict(payload.get("applicability", {})),
            created_at=str(payload.get("created_at", "")),
            status=str(payload.get("status", "uncertain")),
            metadata=dict(payload.get("metadata", {})),
        )


class LearningMemory(ABC):
    """Storage-independent persistence for structured learned strategies."""

    @abstractmethod
    def save(self, strategy: LearnedStrategy) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, strategy_id: str) -> LearnedStrategy | None:
        raise NotImplementedError

    @abstractmethod
    def retrieve_relevant(self, context: dict[str, Any], *, limit: int = 5) -> list[LearnedStrategy]:
        raise NotImplementedError

    @abstractmethod
    def list(self) -> list[LearnedStrategy]:
        raise NotImplementedError


class SQLiteLearningMemory(LearningMemory):
    """SQLite adapter; LearningEngine depends only on LearningMemory."""

    def __init__(self, store: Any, *, min_confidence: float = 0.70) -> None:
        self.store_backend = store
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))

    def save(self, strategy: LearnedStrategy) -> None:
        self.store_backend.save_learned_strategy(strategy)

    def get(self, strategy_id: str) -> LearnedStrategy | None:
        row = self.store_backend.get_learned_strategy(strategy_id)
        if not row:
            return None
        try:
            payload = json.loads(row["strategy_json"])
        except (KeyError, TypeError, ValueError):
            return None
        return LearnedStrategy.from_dict(payload) if isinstance(payload, dict) else None

    def retrieve_relevant(self, context: dict[str, Any], *, limit: int = 5) -> list[LearnedStrategy]:
        candidates: list[tuple[float, LearnedStrategy]] = []
        for strategy in self.list():
            if strategy.status != "active" or strategy.confidence < self.min_confidence:
                continue
            score = self._score(context, strategy)
            if score > 0:
                candidates.append((score, strategy))
        candidates.sort(key=lambda item: (-item[0], -item[1].confidence, item[1].strategy_id))
        return [strategy for _, strategy in candidates[: max(0, limit)]]

    def list(self) -> list[LearnedStrategy]:
        strategies: list[LearnedStrategy] = []
        for row in self.store_backend.all_learned_strategies():
            try:
                payload = json.loads(row["strategy_json"])
                if isinstance(payload, dict):
                    strategies.append(LearnedStrategy.from_dict(payload))
            except (KeyError, TypeError, ValueError):
                continue
        return sorted(strategies, key=lambda item: (-item.confidence, item.strategy_id))

    @classmethod
    def _score(cls, context: dict[str, Any], strategy: LearnedStrategy) -> float:
        condition = strategy.condition
        score = 0.0
        goal_overlap = len(_tokens(context.get("goal", "")) & _tokens(condition.get("goal", "")))
        semantic_match = bool(goal_overlap)
        if goal_overlap:
            score += min(0.50, goal_overlap * 0.15)
        task_type_match = (
            context.get("task_type")
            and condition.get("task_type")
            and context.get("task_type") != "general"
            and condition.get("task_type") != "general"
            and context.get("task_type") == condition.get("task_type")
        )
        if task_type_match:
            score += 0.20
            semantic_match = True
        context_tools = set(context.get("tools", []))
        condition_tools = set(condition.get("tools", []))
        context_capabilities = set(context.get("required_capabilities", []))
        condition_capabilities = set(condition.get("required_capabilities", []))
        capability_match = context_capabilities.intersection(condition_capabilities)
        if capability_match:
            score += 0.10
            semantic_match = True
        context_failures = set(context.get("failure_types", []))
        condition_failures = set(condition.get("failure_types", []))
        failure_match = context_failures.intersection(condition_failures)
        if failure_match:
            score += 0.10
            semantic_match = True
        if context_tools.intersection(condition_tools):
            score += 0.20
        if not semantic_match:
            return 0.0
        return min(1.0, score)


class LearningEngine:
    """Convert ReflectionInsights into deterministic, non-executable learned policies."""

    def __init__(self, learning_memory: LearningMemory, *, min_confidence: float = 0.70) -> None:
        self.learning_memory = learning_memory
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))

    def learn(self, insights: Iterable[ReflectionInsight | dict[str, Any]]) -> list[LearnedStrategy]:
        learned: list[LearnedStrategy] = []
        for raw_insight in insights:
            insight = raw_insight if isinstance(raw_insight, ReflectionInsight) else ReflectionInsight.from_dict(raw_insight)
            candidate = self._from_insight(insight)
            if candidate is None:
                continue
            existing = self.learning_memory.get(candidate.strategy_id)
            strategy = self._merge(existing, candidate) if existing is not None else candidate
            self.learning_memory.save(strategy)
            learned.append(strategy)
        unique = {strategy.strategy_id: strategy for strategy in learned}
        return sorted(unique.values(), key=lambda item: item.strategy_id)

    def learn_from_insights(self, insights: Iterable[ReflectionInsight | dict[str, Any]]) -> list[LearnedStrategy]:
        return self.learn(insights)

    def _from_insight(self, insight: ReflectionInsight) -> LearnedStrategy | None:
        task_context = dict(insight.task_context or {})
        condition_base = {
            "task_type": str(task_context.get("task_type", "general")),
            "goal": str(task_context.get("goal", "")),
            "required_capabilities": sorted(task_context.get("required_capabilities", []) or []),
            "failure_types": sorted(set(insight.failure_types or [])),
        }
        pattern = insight.pattern
        if pattern == "successful_recovery_alternative":
            failed_tool = str(insight.condition.get("failed_tool", ""))
            alternative_tool = str(insight.condition.get("alternative_tool", ""))
            if not failed_tool or not alternative_tool:
                return None
            condition = {
                **condition_base,
                "failed_tool": failed_tool,
                "alternative_tool": alternative_tool,
                "tools": sorted({failed_tool, alternative_tool}),
            }
            preferred_action = {"type": "prefer_tool", "tool": alternative_tool}
            avoided_action = {"type": "avoid_unchanged_retry", "tool": failed_tool}
            positive = True
        elif pattern == "repeated_failure_warning":
            failed_tool = str(insight.condition.get("failed_tool", ""))
            if not failed_tool:
                return None
            condition = {
                **condition_base,
                "failed_tool": failed_tool,
                "tools": [failed_tool],
            }
            preferred_action = {"type": "require_alternative_or_verification"}
            avoided_action = {"type": "avoid_unverified_retry", "tool": failed_tool}
            positive = False
        else:
            return None

        strategy_identity = {
            "condition": condition,
            "preferred_action": preferred_action,
            "avoided_action": avoided_action,
        }
        strategy_id = "strategy-" + hashlib.sha256(
            json.dumps(strategy_identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        conflict_ids = sorted(set(insight.metadata.get("conflicting_experience_ids", []) or []))
        support_ids = sorted(set(insight.supporting_experience_ids or []))
        status = self._status(float(insight.confidence), conflict_ids)
        return LearnedStrategy(
            strategy_id=strategy_id,
            condition=condition,
            preferred_action=preferred_action,
            avoided_action=avoided_action,
            rationale=str(insight.derived_strategy),
            supporting_insight_ids=[insight.insight_id],
            supporting_experience_ids=support_ids,
            confidence=round(float(insight.confidence), 2),
            applicability={
                "task_type": condition.get("task_type", "general"),
                "failure_types": condition.get("failure_types", []),
                "positive_guidance": positive,
            },
            created_at=insight.created_at,
            status=status,
            metadata={
                "source_patterns": [pattern],
                "conflicting_experience_ids": conflict_ids,
                "evidence_count": len(support_ids),
                "positive_guidance": positive,
            },
        )

    def _merge(self, existing: LearnedStrategy, candidate: LearnedStrategy) -> LearnedStrategy:
        supporting_insights = sorted(set(existing.supporting_insight_ids) | set(candidate.supporting_insight_ids))
        supporting_experiences = sorted(set(existing.supporting_experience_ids) | set(candidate.supporting_experience_ids))
        existing_conflicts = set(existing.metadata.get("conflicting_experience_ids", []) or [])
        candidate_conflicts = set(candidate.metadata.get("conflicting_experience_ids", []) or [])
        conflicts = sorted(existing_conflicts | candidate_conflicts)
        confidence = max(float(existing.confidence), float(candidate.confidence))
        if conflicts:
            confidence -= min(0.45, len(conflicts) * 0.12)
        confidence = round(max(0.05, min(0.95, confidence)), 2)
        patterns = sorted(
            set(existing.metadata.get("source_patterns", []) or [])
            | set(candidate.metadata.get("source_patterns", []) or [])
        )
        metadata = {
            **existing.metadata,
            **candidate.metadata,
            "source_patterns": patterns,
            "conflicting_experience_ids": conflicts,
            "evidence_count": len(supporting_experiences),
            "positive_guidance": candidate.metadata.get("positive_guidance", existing.metadata.get("positive_guidance", False)),
        }
        status = self._status(confidence, conflicts)
        return LearnedStrategy(
            strategy_id=existing.strategy_id,
            condition=candidate.condition,
            preferred_action=candidate.preferred_action,
            avoided_action=candidate.avoided_action,
            rationale=candidate.rationale or existing.rationale,
            supporting_insight_ids=supporting_insights,
            supporting_experience_ids=supporting_experiences,
            confidence=confidence,
            applicability=candidate.applicability,
            created_at=min(filter(None, [existing.created_at, candidate.created_at]), default=candidate.created_at),
            status=status,
            metadata=metadata,
        )

    def _status(self, confidence: float, conflicts: list[str]) -> str:
        if conflicts or confidence < self.min_confidence:
            return "uncertain"
        return "active"


def _tokens(value: Any) -> set[str]:
    return {token for token in re.findall(r"\w+", str(value).lower()) if len(token) > 2}
