from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from private_agent.core.experience import ExperienceRecord, sanitize_value
from private_agent.core.memory import ExperienceMemory, ExperienceQuery, RetrievedExperience


@dataclass
class ReflectionInsight:
    insight_id: str
    task_context: dict[str, Any]
    pattern: str
    condition: dict[str, Any]
    observed_behavior: str
    derived_strategy: str
    evidence: list[dict[str, Any]]
    confidence: float
    supporting_experience_ids: list[str]
    failure_types: list[str] = field(default_factory=list)
    recovery_strategies: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = round(float(self.confidence), 2)
        return sanitize_value(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReflectionInsight":
        return cls(
            insight_id=str(payload["insight_id"]),
            task_context=dict(payload.get("task_context", {})),
            pattern=str(payload.get("pattern", "")),
            condition=dict(payload.get("condition", {})),
            observed_behavior=str(payload.get("observed_behavior", "")),
            derived_strategy=str(payload.get("derived_strategy", "")),
            evidence=list(payload.get("evidence", [])),
            confidence=float(payload.get("confidence", 0.0)),
            supporting_experience_ids=list(payload.get("supporting_experience_ids", [])),
            failure_types=list(payload.get("failure_types", [])),
            recovery_strategies=list(payload.get("recovery_strategies", [])),
            created_at=str(payload.get("created_at", "")),
            metadata=dict(payload.get("metadata", {})),
        )


class ReflectionMemory(ABC):
    """Storage-independent persistence for structured reflection insights."""

    @abstractmethod
    def save(self, insight: ReflectionInsight) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, insight_id: str) -> ReflectionInsight | None:
        raise NotImplementedError

    @abstractmethod
    def list_relevant(self, context: dict[str, Any], *, limit: int = 5) -> list[ReflectionInsight]:
        raise NotImplementedError

    def retrieve(self, insight_id: str) -> ReflectionInsight | None:
        return self.get(insight_id)


class SQLiteReflectionMemory(ReflectionMemory):
    """SQLite adapter; ReflectionEngine depends only on ReflectionMemory."""

    def __init__(self, store: Any) -> None:
        self.store_backend = store

    def save(self, insight: ReflectionInsight) -> None:
        self.store_backend.save_reflection(insight)

    def get(self, insight_id: str) -> ReflectionInsight | None:
        row = self.store_backend.get_reflection(insight_id)
        if not row:
            return None
        try:
            payload = json.loads(row["insight_json"])
        except (KeyError, TypeError, ValueError):
            return None
        return ReflectionInsight.from_dict(payload) if isinstance(payload, dict) else None

    def list_relevant(self, context: dict[str, Any], *, limit: int = 5) -> list[ReflectionInsight]:
        candidates: list[tuple[float, ReflectionInsight]] = []
        for row in self.store_backend.all_reflections():
            try:
                payload = json.loads(row["insight_json"])
                insight = ReflectionInsight.from_dict(payload)
            except (KeyError, TypeError, ValueError):
                continue
            score = self._score(context, insight)
            if score > 0:
                candidates.append((score, insight))
        candidates.sort(key=lambda item: (-item[0], -item[1].confidence, item[1].insight_id))
        return [insight for _, insight in candidates[: max(1, limit)]]

    @staticmethod
    def _score(context: dict[str, Any], insight: ReflectionInsight) -> float:
        task_context = insight.task_context
        condition = insight.condition
        score = 0.0
        goal_overlap = len(_tokens(context.get("goal", "")) & _tokens(task_context.get("goal", "")))
        if goal_overlap:
            score += min(0.55, goal_overlap * 0.15)
        if context.get("task_type") and context.get("task_type") == task_context.get("task_type"):
            score += 0.2
        query_tools = set(context.get("tools", []))
        insight_tools = set(condition.get("tools", [])) | {condition.get("failed_tool", ""), condition.get("alternative_tool", "")}
        if query_tools & insight_tools:
            score += 0.15
        query_failures = set(context.get("failure_types", []))
        if query_failures & set(insight.failure_types):
            score += 0.1
        query_recovery = set(context.get("recovery_strategies", []))
        if query_recovery & set(insight.recovery_strategies):
            score += 0.05
        return score


class ReflectionEngine:
    """Pure analysis from retrieved operational experiences into structured insights."""

    def __init__(
        self,
        experience_memory: ExperienceMemory,
        reflection_memory: ReflectionMemory | None = None,
        *,
        min_supporting_experiences: int = 2,
    ) -> None:
        self.experience_memory = experience_memory
        self.reflection_memory = reflection_memory
        self.min_supporting_experiences = max(1, min_supporting_experiences)

    def reflect(self, query: ExperienceQuery) -> list[ReflectionInsight]:
        retrieved = self.experience_memory.retrieve(query)
        relevant = [item for item in retrieved if self._is_relevant(item, query)]
        insights = self._analyze(relevant, query)
        if self.reflection_memory is not None:
            for insight in insights:
                self.reflection_memory.save(insight)
        return insights

    def reflect_for_experience(self, experience: ExperienceRecord) -> list[ReflectionInsight]:
        identity = experience.build_identity()
        failure_types = list(identity.get("failure_types", []))
        recovery_strategies = list(identity.get("recovery_strategies", []))
        query = ExperienceQuery(
            goal=experience.goal,
            task_type=str(identity.get("task_type", "general")),
            required_capabilities=list(identity.get("required_capabilities", [])),
            tools=list(identity.get("tools", [])),
            constraints=dict(identity.get("constraints", {})),
            failure_type=failure_types[0] if failure_types else "",
            recovery_strategy=recovery_strategies[0] if recovery_strategies else "",
        )
        return self.reflect(query)

    def _analyze(self, candidates: list[RetrievedExperience], query: ExperienceQuery) -> list[ReflectionInsight]:
        insights: list[ReflectionInsight] = []
        insights.extend(self._recovery_patterns(candidates, query))
        insights.extend(self._repeated_failure_patterns(candidates, query))
        insights.extend(self._verification_patterns(candidates, query))
        insights.extend(self._input_patterns(candidates, query))
        unique = {insight.insight_id: insight for insight in insights}
        return sorted(unique.values(), key=lambda item: (-item.confidence, item.pattern, item.insight_id))

    def _recovery_patterns(self, candidates: list[RetrievedExperience], query: ExperienceQuery) -> list[ReflectionInsight]:
        groups: dict[tuple[Any, ...], list[RetrievedExperience]] = {}
        for candidate in candidates:
            if candidate.outcome != "COMPLETED" or len(candidate.strategy_used) < 2:
                continue
            failed_tool, alternative_tool = candidate.strategy_used[0], candidate.strategy_used[-1]
            if not failed_tool or not alternative_tool or failed_tool == alternative_tool:
                continue
            if not candidate.failure_diagnosis:
                continue
            key = (query.task_type, failed_tool, alternative_tool, tuple(sorted(candidate.failure_diagnosis)))
            groups.setdefault(key, []).append(candidate)
        insights = []
        for key, supporting in sorted(groups.items(), key=lambda item: str(item[0])):
            if len(supporting) < self.min_supporting_experiences:
                continue
            task_type, failed_tool, alternative_tool, failures = key
            conflicts = [
                candidate
                for candidate in candidates
                if candidate.outcome != "COMPLETED"
                and candidate.strategy_used
                and candidate.strategy_used[0] == failed_tool
                and set(failures).intersection(candidate.failure_diagnosis)
            ]
            ids = _ids(supporting)
            insights.append(
                self._make_insight(
                    query=query,
                    pattern="successful_recovery_alternative",
                    condition={"task_type": task_type, "failed_tool": failed_tool, "alternative_tool": alternative_tool, "failure_types": list(failures)},
                    observed_behavior=f"{failed_tool} failed under {', '.join(failures)}; {alternative_tool} later produced a verified outcome.",
                    derived_strategy=f"Consider {alternative_tool} as an alternative to {failed_tool} under the same condition.",
                    supporting=supporting,
                    conflicts=conflicts,
                    failure_types=list(failures),
                    recovery_strategies=_strategies(supporting),
                    metadata={"pattern_type": "recovery", "conflicting_experience_ids": _ids(conflicts), "alternative_success": True},
                )
            )
        return insights

    def _repeated_failure_patterns(self, candidates: list[RetrievedExperience], query: ExperienceQuery) -> list[ReflectionInsight]:
        groups: dict[tuple[Any, ...], list[RetrievedExperience]] = {}
        for candidate in candidates:
            if candidate.outcome == "COMPLETED" or not candidate.failure_diagnosis:
                continue
            tool = candidate.strategy_used[0] if candidate.strategy_used else "unknown_tool"
            key = (query.task_type, tool, tuple(sorted(candidate.failure_diagnosis)))
            groups.setdefault(key, []).append(candidate)
        insights = []
        for key, supporting in sorted(groups.items(), key=lambda item: str(item[0])):
            if len(supporting) < max(2, self.min_supporting_experiences):
                continue
            task_type, failed_tool, failures = key
            conflicts = [
                candidate
                for candidate in candidates
                if candidate.outcome == "COMPLETED"
                and candidate.strategy_used
                and candidate.strategy_used[0] == failed_tool
                and set(failures).intersection(candidate.failure_diagnosis)
            ]
            insights.append(
                self._make_insight(
                    query=query,
                    pattern="repeated_failure_warning",
                    condition={"task_type": task_type, "failed_tool": failed_tool, "failure_types": list(failures)},
                    observed_behavior=f"{failed_tool} failed {len(supporting)} times under {', '.join(failures)}.",
                    derived_strategy=f"Treat {failed_tool} as unreliable under this condition and require an alternative or additional verification.",
                    supporting=supporting,
                    conflicts=conflicts,
                    failure_types=list(failures),
                    recovery_strategies=_strategies(supporting),
                    metadata={"pattern_type": "repeated_failure", "conflicting_experience_ids": _ids(conflicts), "automatic_rule": False},
                )
            )
        return insights

    def _verification_patterns(self, candidates: list[RetrievedExperience], query: ExperienceQuery) -> list[ReflectionInsight]:
        supporting = [candidate for candidate in candidates if "VERIFICATION_FAILED" in candidate.failure_diagnosis]
        if len(supporting) < self.min_supporting_experiences:
            return []
        tools = sorted({candidate.strategy_used[-1] for candidate in supporting if candidate.strategy_used})
        return [
            self._make_insight(
                query=query,
                pattern="verification_failure_requires_evidence",
                condition={"task_type": query.task_type, "tools": tools, "failure_type": "VERIFICATION_FAILED"},
                observed_behavior="Tool execution was recorded, but verification evidence was insufficient or failed.",
                derived_strategy="Require independent verification evidence before treating this result as a successful task outcome.",
                supporting=supporting,
                conflicts=[],
                failure_types=["VERIFICATION_FAILED"],
                recovery_strategies=_strategies(supporting),
                metadata={"pattern_type": "verification_failure", "automatic_rule": False},
            )
        ]

    def _input_patterns(self, candidates: list[RetrievedExperience], query: ExperienceQuery) -> list[ReflectionInsight]:
        supporting = [
            candidate
            for candidate in candidates
            if "INVALID_INPUT" in candidate.failure_diagnosis
            and set(candidate.recovery_strategy).intersection({"RETRY_WITH_MODIFIED_INPUT", "CHANGE_PARAMETERS"})
        ]
        if len(supporting) < self.min_supporting_experiences:
            return []
        return [
            self._make_insight(
                query=query,
                pattern="invalid_input_adjustment",
                condition={"task_type": query.task_type, "failure_type": "INVALID_INPUT"},
                observed_behavior="The initial input was diagnosed as invalid and a modified-parameter recovery was recorded.",
                derived_strategy="Validate and adjust tool inputs before repeating the operation.",
                supporting=supporting,
                conflicts=[],
                failure_types=["INVALID_INPUT"],
                recovery_strategies=_strategies(supporting),
                metadata={"pattern_type": "input_failure", "automatic_rule": False},
            )
        ]

    @staticmethod
    def _is_relevant(candidate: RetrievedExperience, query: ExperienceQuery) -> bool:
        if not candidate.experience_id:
            return False
        if _tokens(query.goal) & _tokens(candidate.task_summary):
            return True
        if query.task_type and candidate.identity.get("task_type") == query.task_type:
            return True
        if set(query.tools) & set(candidate.identity.get("tools", [])):
            return True
        if query.failure_type and query.failure_type in candidate.failure_diagnosis:
            return True
        if query.recovery_strategy and query.recovery_strategy in candidate.recovery_strategy:
            return True
        return False

    def _make_insight(
        self,
        *,
        query: ExperienceQuery,
        pattern: str,
        condition: dict[str, Any],
        observed_behavior: str,
        derived_strategy: str,
        supporting: list[RetrievedExperience],
        conflicts: list[RetrievedExperience],
        failure_types: list[str],
        recovery_strategies: list[str],
        metadata: dict[str, Any],
    ) -> ReflectionInsight:
        support_ids = _ids(supporting)
        conflict_ids = _ids(conflicts)
        confidence = _confidence(len(support_ids), len(conflict_ids), alternative_success=metadata.get("alternative_success", False))
        created_at = max((candidate.created_at for candidate in supporting if candidate.created_at), default="1970-01-01T00:00:00+00:00")
        identity_payload = {
            "pattern": pattern,
            "task_context": _query_dict(query),
            "condition": condition,
            "supporting_experience_ids": support_ids,
        }
        insight_id = "insight-" + hashlib.sha256(json.dumps(identity_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        evidence = [
            {
                "experience_id": candidate.experience_id,
                "outcome": candidate.outcome,
                "verification_evidence": candidate.verification_evidence[:3],
            }
            for candidate in supporting
        ]
        full_metadata = dict(metadata)
        full_metadata["conflicting_experience_ids"] = conflict_ids
        return ReflectionInsight(
            insight_id=insight_id,
            task_context=_query_dict(query),
            pattern=pattern,
            condition=condition,
            observed_behavior=observed_behavior,
            derived_strategy=derived_strategy,
            evidence=evidence,
            confidence=confidence,
            supporting_experience_ids=support_ids,
            failure_types=sorted(set(failure_types)),
            recovery_strategies=sorted(set(recovery_strategies)),
            created_at=created_at,
            metadata=full_metadata,
        )


def _query_dict(query: ExperienceQuery) -> dict[str, Any]:
    return {
        "goal": query.goal,
        "task_type": query.task_type,
        "required_capabilities": sorted(query.required_capabilities),
        "tools": sorted(query.tools),
        "constraints": query.constraints,
        "failure_type": query.failure_type,
        "recovery_strategy": query.recovery_strategy,
    }


def _ids(candidates: Iterable[RetrievedExperience]) -> list[str]:
    return sorted({candidate.experience_id for candidate in candidates if candidate.experience_id})


def _strategies(candidates: Iterable[RetrievedExperience]) -> list[str]:
    return sorted({strategy for candidate in candidates for strategy in candidate.recovery_strategy if strategy})


def _confidence(supporting: int, conflicts: int, *, alternative_success: bool) -> float:
    score = 0.45 + min(0.30, supporting * 0.12)
    if alternative_success:
        score += 0.12
    score -= min(0.30, conflicts * 0.08)
    return round(max(0.10, min(0.95, score)), 2)


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in str(value).split() if len(token.strip()) > 2}
