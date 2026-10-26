"""Owner review records for Bug Bounty findings.

Review records are control-plane decisions. They never authorize a target, start a
scanner, or replace the semantic Evidence Gate.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .models import _bounded_text, safe_json, utc_now


class FindingReviewDecision(str, Enum):
    CONFIRM = "CONFIRM"
    REFUTE = "REFUTE"
    INCONCLUSIVE = "INCONCLUSIVE"
    DUPLICATE = "DUPLICATE"


@dataclass(frozen=True)
class FindingReview:
    review_id: str
    task_id: str
    finding_id: str
    owner_user_id: str
    decision: str
    duplicate_of: str = ""
    note: str = ""
    evidence_sha256: str = ""
    evidence_validated: bool = False
    actor_user_id: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        for name, value, limit in (
            ("review_id", self.review_id, 128), ("task_id", self.task_id, 128),
            ("finding_id", self.finding_id, 128), ("owner_user_id", self.owner_user_id, 128),
        ):
            _bounded_text(value, limit=limit, field_name=name)
        try:
            object.__setattr__(self, "decision", FindingReviewDecision(self.decision).value)
        except ValueError as exc:
            raise ValueError("finding_review_decision_invalid") from exc
        if self.duplicate_of and self.duplicate_of == self.finding_id:
            raise ValueError("finding_review_duplicate_self")
        if self.decision == FindingReviewDecision.DUPLICATE.value and not self.duplicate_of:
            raise ValueError("finding_review_duplicate_target_required")
        if not isinstance(self.note, str) or len(self.note) > 2000:
            raise ValueError("finding_review_note_invalid")
        if not isinstance(self.evidence_sha256, str) or len(self.evidence_sha256) > 128:
            raise ValueError("finding_review_evidence_digest_invalid")
        if type(self.evidence_validated) is not bool:
            raise ValueError("finding_review_validation_flag_invalid")
        if self.decision == FindingReviewDecision.CONFIRM.value and not self.evidence_validated:
            raise ValueError("finding_review_confirm_requires_validated_evidence")
        if not isinstance(self.actor_user_id, str) or len(self.actor_user_id) > 128:
            raise ValueError("finding_review_actor_invalid")
        object.__setattr__(self, "created_at", self.created_at or utc_now())

    @classmethod
    def create(
        cls, task_id: str, finding_id: str, owner_user_id: str, decision: FindingReviewDecision | str,
        *, duplicate_of: str = "", note: str = "", evidence_sha256: str = "", evidence_validated: bool = False,
        actor_user_id: str = "",
    ) -> "FindingReview":
        return cls(
            review_id="REVIEW-" + uuid.uuid4().hex,
            task_id=task_id, finding_id=finding_id, owner_user_id=owner_user_id,
            decision=FindingReviewDecision(decision).value, duplicate_of=duplicate_of,
            note=note, evidence_sha256=evidence_sha256, evidence_validated=evidence_validated,
            actor_user_id=actor_user_id or owner_user_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id, "task_id": self.task_id, "finding_id": self.finding_id,
            "decision": self.decision, "duplicate_of": self.duplicate_of, "note": self.note,
            "evidence_sha256": self.evidence_sha256, "evidence_validated": self.evidence_validated,
            "actor_user_id": self.actor_user_id, "created_at": self.created_at,
        }


class FindingReviewRepository:
    def __init__(self, store: Any) -> None:
        if store is None or not hasattr(store, "db") or not hasattr(store, "_db_lock"):
            raise ValueError("finding_review_store_required")
        self.store = store

    @staticmethod
    def _owner(owner_user_id: str | None) -> str:
        return owner_user_id or "legacy-owner"

    @staticmethod
    def _row(row: Any) -> FindingReview:
        return FindingReview(
            review_id=str(row["review_id"]), task_id=str(row["task_id"]), finding_id=str(row["finding_id"]),
            owner_user_id=str(row["owner_user_id"]), decision=str(row["decision"]), duplicate_of=str(row["duplicate_of"]),
            note=str(row["note"]), evidence_sha256=str(row["evidence_sha256"]),
            evidence_validated=bool(row["evidence_validated"]), actor_user_id=str(row["actor_user_id"]),
            created_at=str(row["created_at"]),
        )

    def save(self, review: FindingReview, *, is_owner: bool = False) -> FindingReview:
        owner = self._owner(review.owner_user_id)
        with self.store._db_lock, self.store.db:
            self.store.db.execute(
                "INSERT INTO bugbounty_finding_reviews(review_id, task_id, finding_id, owner_user_id, decision, duplicate_of, note, evidence_sha256, evidence_validated, actor_user_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (review.review_id, review.task_id, review.finding_id, owner, review.decision, review.duplicate_of,
                 review.note, review.evidence_sha256, int(review.evidence_validated), review.actor_user_id, review.created_at),
            )
        policy_version = 0
        try:
            policy_version = self.store.latest_owner_policy_version()
        except Exception:
            pass
        self.store.save_audit_event({
            "event_id": review.review_id, "timestamp": review.created_at,
            "event_type": "BUGBOUNTY_FINDING_REVIEW", "task_id": review.task_id,
            "action_id": review.finding_id, "actor": review.actor_user_id,
            "tool_or_skill": "bugbounty.finding_review", "capabilities": [],
            "risk_level": "MEDIUM", "policy_version": policy_version,
            "approval_id": "", "decision": review.decision,
            "reason": review.note or "owner_finding_review", "review": review.to_dict(),
        }, requires_lease=False)
        return review

    def list_for_task(self, task_id: str, *, owner_user_id: str | None, is_owner: bool = False) -> list[FindingReview]:
        owner = self._owner(owner_user_id)
        query = "SELECT * FROM bugbounty_finding_reviews WHERE task_id = ?" + ("" if is_owner else " AND owner_user_id = ?") + " ORDER BY created_at, review_id"
        params = (task_id,) if is_owner else (task_id, owner)
        with self.store._db_lock:
            rows = self.store.db.execute(query, params).fetchall()
        return [self._row(row) for row in rows]
