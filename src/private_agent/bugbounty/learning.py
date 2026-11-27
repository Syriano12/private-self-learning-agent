"""Evidence-gated advisory learning for Bug Bounty review outcomes.

This bridge deliberately updates only the existing non-executable LearningEngine.
It never changes scope, approval state, finding state, or tool permissions. Raw
finding content is not copied into the insight; only stable identifiers and
validated evidence digests are retained.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Any

from private_agent.core.learning import LearnedStrategy, LearningEngine, SQLiteLearningMemory
from private_agent.core.reflection import ReflectionInsight

from .models import Finding, FindingState, validate_confirmed_evidence
from .reviews import FindingReview, FindingReviewDecision


class BugBountyLearningService:
    """Persist owner-reviewed Bug Bounty outcomes as advisory strategies only."""

    def __init__(self, store: Any, *, learning_engine: LearningEngine | None = None) -> None:
        self.store = store
        self.learning_engine = learning_engine or LearningEngine(SQLiteLearningMemory(store))

    def planning_advisories(self, *, task_id: str, goal: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return small, non-authoritative hints backed by current owner-bound reviews."""
        if not isinstance(task_id, str) or not task_id or not isinstance(goal, str) or not goal.strip():
            return []
        memory = getattr(self.learning_engine, "learning_memory", None)
        retrieve = getattr(memory, "retrieve_relevant", None)
        if not callable(retrieve):
            return []
        try:
            raw_strategies = retrieve(
                {
                    "task_type": "bug_bounty",
                    "goal": goal,
                    "tools": ["bugbounty_analysis"],
                    "task_id": task_id,
                },
                limit=max(0, min(int(limit), 5)),
            )
        except (AttributeError, TypeError, ValueError, KeyError, sqlite3.Error):
            return []

        if not isinstance(raw_strategies, list):
            return []
        strategies = [item for item in raw_strategies if isinstance(item, LearnedStrategy)]
        advisories: list[dict[str, Any]] = []
        for strategy in strategies:
            condition = strategy.condition if isinstance(strategy.condition, dict) else {}
            preferred = strategy.preferred_action if isinstance(strategy.preferred_action, dict) else {}
            avoided = strategy.avoided_action if isinstance(strategy.avoided_action, dict) else {}
            metadata = strategy.metadata if isinstance(strategy.metadata, dict) else {}
            applicability = strategy.applicability if isinstance(strategy.applicability, dict) else {}
            vulnerability_class = condition.get("vulnerability_class")
            review_id = metadata.get("review_id")
            finding_id = metadata.get("finding_id")
            evidence_sha256 = metadata.get("evidence_sha256")
            if (
                strategy.status != "active"
                or strategy.confidence < 0.70
                or condition.get("task_type") != "bug_bounty"
                or condition.get("review_decision") != "CONFIRM"
                or not isinstance(vulnerability_class, str)
                or not vulnerability_class
                or preferred.get("type") != "require_evidence_review"
                or preferred.get("vulnerability_class") != vulnerability_class
                or avoided.get("type") != "auto_confirm_finding"
                or applicability.get("positive_guidance") is not True
                or metadata.get("source_patterns") != ["bugbounty_review_feedback"]
                or metadata.get("positive_guidance") is not True
                or metadata.get("verified_outcome") is not True
                or metadata.get("non_authoritative") is not True
                or metadata.get("conflicting_experience_ids") != []
                or not isinstance(review_id, str)
                or not review_id
                or not isinstance(finding_id, str)
                or not finding_id
                or not isinstance(evidence_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", evidence_sha256) is None
                or strategy.supporting_experience_ids != [f"finding-review:{review_id}"]
            ):
                continue
            advisories.append(
                {
                    "strategy_id": strategy.strategy_id,
                    "vulnerability_class": vulnerability_class,
                    "review_id": review_id,
                    "finding_id": finding_id,
                    "evidence_sha256": evidence_sha256,
                    "confidence": float(strategy.confidence),
                    "requirement": "prioritize_owner_review_before_confirmation",
                    "non_authoritative": True,
                }
            )
        advisories.sort(key=lambda item: item["strategy_id"])
        return advisories[: max(0, min(int(limit), 5))]

    def apply_review(self, review: FindingReview, finding: Finding) -> list[LearnedStrategy]:
        if not isinstance(review, FindingReview) or not isinstance(finding, Finding):
            raise ValueError("bugbounty_learning_types_required")
        decision = FindingReviewDecision(review.decision)
        if decision == FindingReviewDecision.CONFIRM:
            self._require_confirmed_evidence(review, finding)

        confidence = 0.90 if decision == FindingReviewDecision.CONFIRM else 0.55
        verified_outcome = decision == FindingReviewDecision.CONFIRM
        evidence_sha256 = review.evidence_sha256 if review.evidence_validated else ""
        vulnerability_class = str(finding.vulnerability_class)[:256]
        insight = ReflectionInsight(
            insight_id=f"bugbounty-review-{review.review_id}",
            task_context={
                "task_type": "bug_bounty",
                "goal": f"Bug Bounty finding review: {vulnerability_class}",
                "required_capabilities": ["evidence_review"],
            },
            pattern="bugbounty_review_feedback",
            condition={
                "task_type": "bug_bounty",
                "vulnerability_class": vulnerability_class,
                "review_decision": decision.value,
                "tools": ["bugbounty_analysis"],
            },
            observed_behavior=(
                f"Owner review marked a {vulnerability_class} candidate as {decision.value}; "
                "the outcome remains advisory and does not authorize execution."
            ),
            derived_strategy=(
                "Require the backend Evidence Gate before report-ready and explicit owner review "
                "before treating a Bug Bounty finding as confirmed; never infer authorization or "
                "verification from learned guidance."
            ),
            evidence=[
                {
                    "review_id": review.review_id,
                    "finding_id": review.finding_id,
                    "decision": decision.value,
                    "evidence_sha256": evidence_sha256,
                    "verified_outcome": verified_outcome,
                }
            ],
            confidence=confidence,
            supporting_experience_ids=[f"finding-review:{review.review_id}"],
            metadata={
                "source": "owner_finding_review",
                "review_id": review.review_id,
                "finding_id": review.finding_id,
                "evidence_sha256": evidence_sha256,
                "verified_outcome": verified_outcome,
                "non_authoritative": True,
            },
        )
        return self.learning_engine.learn([insight])

    @staticmethod
    def _require_confirmed_evidence(review: FindingReview, finding: Finding) -> None:
        if finding.state not in {FindingState.REPORT_READY, FindingState.HUMAN_REVIEWED}:
            raise ValueError("bugbounty_learning_finding_not_report_ready")
        errors = validate_confirmed_evidence(finding, target_state=finding.state)
        current_digest = finding._evidence_digest()
        if errors or finding.validation_status != "VALIDATED":
            raise ValueError("bugbounty_learning_evidence_not_validated")
        if not review.evidence_validated or not review.evidence_sha256:
            raise ValueError("bugbounty_learning_review_evidence_required")
        if review.evidence_sha256 != current_digest or review.evidence_sha256 != finding.validated_evidence_sha256:
            raise ValueError("bugbounty_learning_evidence_digest_mismatch")
