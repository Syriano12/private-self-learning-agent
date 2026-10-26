"""Evidence-gated advisory learning for Bug Bounty review outcomes.

This bridge deliberately updates only the existing non-executable LearningEngine.
It never changes scope, approval state, finding state, or tool permissions. Raw
finding content is not copied into the insight; only stable identifiers and
validated evidence digests are retained.
"""
from __future__ import annotations

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
                "Require the backend Evidence Gate and explicit owner review before treating a "
                "Bug Bounty candidate as report-ready; never infer authorization from learned guidance."
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
