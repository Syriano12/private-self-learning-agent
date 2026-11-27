from __future__ import annotations

import hashlib
import json
import re
import sqlite3
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

    def __init__(
        self,
        store: Any,
        *,
        min_confidence: float = 0.70,
        min_relevance: float = 0.45,
        max_support_age_seconds: int = 180 * 24 * 60 * 60,
    ) -> None:
        self.store_backend = store
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.min_relevance = max(0.0, min(1.0, float(min_relevance)))
        self.max_support_age_seconds = max(1, int(max_support_age_seconds))

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
        scoped, owner_user_id = self._owner_scope(context)
        if scoped and not owner_user_id:
            return []
        candidates: list[tuple[float, LearnedStrategy]] = []
        for strategy in self.list():
            if strategy.status != "active" or strategy.confidence < self.min_confidence:
                continue
            if not (
                self._has_revalidated_recovery_support(strategy, owner_user_id=owner_user_id)
                or self._has_revalidated_owner_review_support(strategy, owner_user_id=owner_user_id)
            ):
                continue
            if self._has_revalidated_owner_review_conflict(strategy, owner_user_id=owner_user_id):
                continue
            score = self._score(context, strategy)
            if score >= self.min_relevance:
                candidates.append((score, strategy))
        candidates.sort(key=lambda item: (-item[0], -item[1].confidence, item[1].strategy_id))
        return [strategy for _, strategy in candidates[: max(0, limit)]]

    def get_for_planning(self, strategy_id: str, context: dict[str, Any]) -> LearnedStrategy | None:
        """Return canonical stored advisory guidance only after current support checks."""
        if not isinstance(strategy_id, str) or not strategy_id:
            return None
        scoped, owner_user_id = self._owner_scope(context)
        if scoped and not owner_user_id:
            return None
        try:
            strategy = self.get(strategy_id)
        except (KeyError, TypeError, ValueError):
            return None
        if (
            strategy is None
            or strategy.status != "active"
            or strategy.confidence < self.min_confidence
            or not (
                self._has_revalidated_recovery_support(strategy, owner_user_id=owner_user_id)
                or self._has_revalidated_owner_review_support(strategy, owner_user_id=owner_user_id)
            )
            or self._has_revalidated_owner_review_conflict(strategy, owner_user_id=owner_user_id)
            or self._score(context, strategy) < self.min_relevance
        ):
            return None
        return strategy

    def _owner_scope(self, context: dict[str, Any]) -> tuple[bool, str | None]:
        if "task_id" not in context:
            return True, None
        task_id = context.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            return True, None
        try:
            owner_user_id = self.store_backend.task_owner_user_id(task_id)
        except (AttributeError, TypeError, ValueError):
            return True, None
        return True, owner_user_id if isinstance(owner_user_id, str) and owner_user_id else None

    def _has_revalidated_recovery_support(
        self, strategy: LearnedStrategy, *, owner_user_id: str | None = None,
    ) -> bool:
        """Re-read each claimed support record before allowing a plan-time hint."""
        metadata = strategy.metadata if isinstance(strategy.metadata, dict) else {}
        condition = strategy.condition if isinstance(strategy.condition, dict) else {}
        preferred = strategy.preferred_action if isinstance(strategy.preferred_action, dict) else {}
        avoided = strategy.avoided_action if isinstance(strategy.avoided_action, dict) else {}
        support_ids = strategy.supporting_experience_ids
        if (
            not isinstance(support_ids, list)
            or len(support_ids) < 2
            or any(not isinstance(item, str) or not item for item in support_ids)
            or len(set(support_ids)) != len(support_ids)
            or metadata.get("evidence_count") != len(support_ids)
            or metadata.get("source_patterns") != ["successful_recovery_alternative"]
            or metadata.get("conflicting_experience_ids") != []
            or preferred.get("type") != "prefer_tool"
            or avoided.get("type") != "avoid_unchanged_retry"
        ):
            return False

        failed_tool = str(condition.get("failed_tool", ""))
        alternative_tool = str(condition.get("alternative_tool", ""))
        raw_failure_types = condition.get("failure_types")
        raw_condition_tools = condition.get("tools")
        if (
            not isinstance(raw_failure_types, list)
            or any(not isinstance(item, str) or not item for item in raw_failure_types)
            or not isinstance(raw_condition_tools, list)
            or any(not isinstance(item, str) or not item for item in raw_condition_tools)
        ):
            return False
        failure_types = set(raw_failure_types)
        condition_tools = set(raw_condition_tools)
        if (
            not failed_tool
            or not alternative_tool
            or failed_tool == alternative_tool
            or preferred.get("tool") != alternative_tool
            or avoided.get("tool") != failed_tool
            or not failure_types
            or not {failed_tool, alternative_tool}.issubset(condition_tools)
        ):
            return False

        now = datetime.now(timezone.utc)
        for experience_id in support_ids:
            if owner_user_id is not None:
                try:
                    if self.store_backend.task_owner_user_id(experience_id) != owner_user_id:
                        return False
                except (AttributeError, TypeError, ValueError):
                    return False
            row = self.store_backend.get_experience(experience_id)
            if not isinstance(row, dict):
                return False
            try:
                payload = json.loads(row["experience_json"])
            except (KeyError, TypeError, ValueError):
                return False
            if not isinstance(payload, dict) or payload.get("task_id") != experience_id:
                return False
            created_at = payload.get("created_at")
            if not isinstance(created_at, str) or row.get("created_at") != created_at:
                return False
            try:
                timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                return False
            if timestamp.tzinfo is None:
                return False
            age = (now - timestamp.astimezone(timezone.utc)).total_seconds()
            if age < -300 or age > self.max_support_age_seconds:
                return False

            identity = payload.get("identity")
            if not isinstance(identity, dict):
                return False
            outcome = payload.get("final_outcome")
            verification_status = payload.get("verification_status")
            verification_scope = payload.get("verification_scope")
            impact_verified = payload.get("security_impact_verified")
            if (
                outcome != "COMPLETED"
                or verification_status != "VERIFIED"
                or not isinstance(verification_scope, str)
                or not verification_scope
                or (impact_verified is not None and impact_verified is not True)
                or identity.get("outcome") != outcome
                or identity.get("verification_status") != verification_status
                or identity.get("verification_scope") != verification_scope
                or identity.get("security_impact_verified") != impact_verified
            ):
                return False

            actions = payload.get("actions")
            if not isinstance(actions, list):
                return False
            named_actions = [
                action for action in actions
                if isinstance(action, dict) and isinstance(action.get("tool_name"), str) and action["tool_name"]
            ]
            if (
                len(named_actions) < 2
                or named_actions[-1].get("tool_name") != alternative_tool
            ):
                return False
            final_action = named_actions[-1]
            if not self._has_verified_change_tool_transition(
                named_actions,
                failed_tool=failed_tool,
                alternative_tool=alternative_tool,
                failure_types=failure_types,
                final_action=final_action,
            ):
                return False
            verification = final_action.get("verification")
            evidence = verification.get("evidence") if isinstance(verification, dict) else None
            if (
                final_action.get("execution_status") != "success"
                or not isinstance(verification, dict)
                or verification.get("status") != "VERIFIED"
                or verification.get("verified") is not True
                or not isinstance(evidence, list)
                or not any(isinstance(item, dict) for item in evidence)
            ):
                return False
        return True

    @staticmethod
    def _has_verified_change_tool_transition(
        actions: list[dict[str, Any]],
        *,
        failed_tool: str,
        alternative_tool: str,
        failure_types: set[Any],
        final_action: dict[str, Any],
    ) -> bool:
        for index, action in enumerate(actions):
            diagnosis = action.get("diagnosis")
            recovery = action.get("recovery")
            step_id = action.get("step_id")
            if (
                action.get("tool_name") != failed_tool
                or not isinstance(step_id, str)
                or not step_id
                or not isinstance(diagnosis, dict)
                or diagnosis.get("failure_type") not in failure_types
                or not isinstance(recovery, dict)
                or recovery.get("strategy") != "CHANGE_TOOL"
                or recovery.get("original_tool") != failed_tool
                or recovery.get("replacement_tool") != alternative_tool
                or recovery.get("step_id") != step_id
            ):
                continue
            subsequent = next(
                (
                    later for later in actions[index + 1:]
                    if later.get("step_id") == step_id
                    and later.get("tool_name") == alternative_tool
                ),
                None,
            )
            if subsequent is not final_action:
                continue
            verification = final_action.get("verification")
            evidence = verification.get("evidence") if isinstance(verification, dict) else None
            if (
                final_action.get("execution_status") == "success"
                and isinstance(verification, dict)
                and verification.get("status") == "VERIFIED"
                and verification.get("verified") is True
                and isinstance(evidence, list)
                and any(isinstance(item, dict) for item in evidence)
            ):
                return True
        return False

    def _has_revalidated_owner_review_support(
        self, strategy: LearnedStrategy, *, owner_user_id: str | None = None,
    ) -> bool:
        """Retain owner-reviewed context without making it tool-selection guidance."""
        metadata = strategy.metadata if isinstance(strategy.metadata, dict) else {}
        condition = strategy.condition if isinstance(strategy.condition, dict) else {}
        preferred = strategy.preferred_action if isinstance(strategy.preferred_action, dict) else {}
        avoided = strategy.avoided_action if isinstance(strategy.avoided_action, dict) else {}
        support_ids = strategy.supporting_experience_ids
        review_id = metadata.get("review_id")
        evidence_sha256 = metadata.get("evidence_sha256")
        finding_id = metadata.get("finding_id")
        if (
            metadata.get("source_patterns") != ["bugbounty_review_feedback"]
            or metadata.get("evidence_count") != 1
            or metadata.get("positive_guidance") is not True
            or metadata.get("conflicting_experience_ids") != []
            or metadata.get("verified_outcome") is not True
            or metadata.get("non_authoritative") is not True
            or condition.get("task_type") != "bug_bounty"
            or condition.get("review_decision") != "CONFIRM"
            or preferred.get("type") != "require_evidence_review"
            or preferred.get("vulnerability_class") != condition.get("vulnerability_class")
            or avoided.get("type") != "auto_confirm_finding"
            or not isinstance(review_id, str)
            or not review_id
            or not isinstance(finding_id, str)
            or not finding_id
            or not isinstance(evidence_sha256, str)
            or not evidence_sha256
            or support_ids != [f"finding-review:{review_id}"]
        ):
            return False
        try:
            from private_agent.bugbounty.reviews import FindingReviewRepository

            review = FindingReviewRepository(self.store_backend).get_by_id(review_id)
        except (AttributeError, TypeError, ValueError):
            return False
        if (
            review is None
            or review.decision != "CONFIRM"
            or review.finding_id != finding_id
            or not review.evidence_validated
            or review.evidence_sha256 != evidence_sha256
        ):
            return False
        if owner_user_id is not None:
            try:
                if (
                    review.owner_user_id != owner_user_id
                    or review.actor_user_id != owner_user_id
                    or self.store_backend.task_owner_user_id(review.task_id) != owner_user_id
                ):
                    return False
            except (AttributeError, TypeError, ValueError):
                return False
        try:
            timestamp = datetime.fromisoformat(review.created_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError):
            return False
        if timestamp.tzinfo is None:
            return False
        age = (datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds()
        return -300 <= age <= self.max_support_age_seconds

    def _has_revalidated_owner_review_conflict(
        self, strategy: LearnedStrategy, *, owner_user_id: str | None = None,
    ) -> bool:
        """Suppress positive Bug Bounty guidance when this owner has a fresh contrary review.

        A conflicting review only suppresses advisory reuse; it never creates a
        negative finding or changes verification, authorization, or scope.
        """
        condition = strategy.condition if isinstance(strategy.condition, dict) else {}
        vulnerability_class = condition.get("vulnerability_class")
        if (
            owner_user_id is None
            or condition.get("task_type") != "bug_bounty"
            or not isinstance(vulnerability_class, str)
            or not vulnerability_class
        ):
            return False

        try:
            from private_agent.bugbounty.reviews import FindingReviewRepository

            reviews = FindingReviewRepository(self.store_backend)
            other_strategies = self.list()
        except (AttributeError, TypeError, ValueError, sqlite3.Error):
            return True

        now = datetime.now(timezone.utc)
        for candidate in other_strategies:
            candidate_condition = candidate.condition if isinstance(candidate.condition, dict) else {}
            metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
            decision = candidate_condition.get("review_decision")
            review_id = metadata.get("review_id")
            if (
                candidate.strategy_id == strategy.strategy_id
                or metadata.get("source_patterns") != ["bugbounty_review_feedback"]
                or candidate_condition.get("task_type") != "bug_bounty"
                or candidate_condition.get("vulnerability_class") != vulnerability_class
                or decision not in {"REFUTE", "INCONCLUSIVE", "DUPLICATE"}
                or not isinstance(review_id, str)
                or not review_id
            ):
                continue
            try:
                review = reviews.get_by_id(review_id)
                if (
                    review is None
                    or review.decision != decision
                    or review.owner_user_id != owner_user_id
                    or review.actor_user_id != owner_user_id
                    or self.store_backend.task_owner_user_id(review.task_id) != owner_user_id
                ):
                    continue
                timestamp = datetime.fromisoformat(review.created_at.replace("Z", "+00:00"))
            except sqlite3.Error:
                return True
            except (AttributeError, TypeError, ValueError):
                continue
            if timestamp.tzinfo is None:
                continue
            age = (now - timestamp.astimezone(timezone.utc)).total_seconds()
            if -300 <= age <= self.max_support_age_seconds:
                return True
        return False

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
        # General task labels and shared tool names are not enough: require
        # at least two concrete goal tokens before experience can steer a plan.
        if not semantic_match or goal_overlap < 2:
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
        extra_metadata: dict[str, Any] = {}
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
        elif pattern == "bugbounty_review_feedback":
            vulnerability_class = str(insight.condition.get("vulnerability_class", "")).strip()
            decision = str(insight.condition.get("review_decision", "")).strip()
            if not vulnerability_class or decision not in {"CONFIRM", "REFUTE", "INCONCLUSIVE", "DUPLICATE"}:
                return None
            condition = {
                **condition_base,
                "vulnerability_class": vulnerability_class,
                "review_decision": decision,
                "tools": ["bugbounty_analysis"],
            }
            preferred_action = {"type": "require_evidence_review", "vulnerability_class": vulnerability_class}
            avoided_action = {"type": "auto_confirm_finding"}
            positive = decision == "CONFIRM"
            extra_metadata = {
                "review_id": str(insight.metadata.get("review_id", ""))[:128],
                "finding_id": str(insight.metadata.get("finding_id", ""))[:128],
                "evidence_sha256": str(insight.metadata.get("evidence_sha256", ""))[:128],
                "verified_outcome": bool(insight.metadata.get("verified_outcome", False)),
                "non_authoritative": True,
            }
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
                **extra_metadata,
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
