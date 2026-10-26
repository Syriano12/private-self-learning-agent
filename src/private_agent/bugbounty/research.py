"""Durable, bounded research-stage records.

Stages are control-plane records only. This module never fetches targets, launches
scanners, executes tools, or turns a proposed scope into authorization.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping

from .local_evidence import normalize_research_input
from .models import safe_json, utc_now


MAX_STAGE_RESULT_BYTES = 32_000
MIN_STAGE_OUTPUT_BYTES = 256
MAX_STAGE_BUDGET_ITEMS = 10_000
MAX_STAGE_TIMEOUT_SECONDS = 86_400
MAX_STAGE_COUNT_PER_ANALYSIS = 32


class ResearchStageType(str, Enum):
    PROGRAM_ANALYSIS = "PROGRAM_ANALYSIS"
    INVENTORY = "INVENTORY"
    PASSIVE_RESEARCH = "PASSIVE_RESEARCH"
    HYPOTHESIS_GENERATION = "HYPOTHESIS_GENERATION"
    VERIFICATION = "VERIFICATION"
    REPORT_PREPARATION = "REPORT_PREPARATION"


class ResearchStageStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"
    UNAVAILABLE = "UNAVAILABLE"


_TERMINAL = {
    ResearchStageStatus.COMPLETED.value,
    ResearchStageStatus.PARTIAL.value,
    ResearchStageStatus.FAILED.value,
    ResearchStageStatus.CANCELLED.value,
    ResearchStageStatus.UNKNOWN.value,
    ResearchStageStatus.UNAVAILABLE.value,
}


@dataclass(frozen=True)
class StageBudget:
    max_items: int = 100
    timeout_seconds: int = 300
    max_output_bytes: int = MAX_STAGE_RESULT_BYTES

    def __post_init__(self) -> None:
        if type(self.max_items) is not int or not 1 <= self.max_items <= MAX_STAGE_BUDGET_ITEMS:
            raise ValueError("stage_max_items_invalid")
        if type(self.timeout_seconds) is not int or not 1 <= self.timeout_seconds <= MAX_STAGE_TIMEOUT_SECONDS:
            raise ValueError("stage_timeout_invalid")
        if type(self.max_output_bytes) is not int or not MIN_STAGE_OUTPUT_BYTES <= self.max_output_bytes <= MAX_STAGE_RESULT_BYTES:
            raise ValueError("stage_output_limit_invalid")

    def to_dict(self) -> dict[str, int]:
        return {"max_items": self.max_items, "timeout_seconds": self.timeout_seconds, "max_output_bytes": self.max_output_bytes}

    @classmethod
    def from_value(cls, value: Any) -> "StageBudget":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("stage_budget_object_required")
        return cls(
            max_items=value.get("max_items", 100),
            timeout_seconds=value.get("timeout_seconds", 300),
            max_output_bytes=value.get("max_output_bytes", MAX_STAGE_RESULT_BYTES),
        )


@dataclass(frozen=True)
class ResearchStage:
    stage_id: str
    analysis_id: str
    owner_user_id: str
    stage_type: str
    stage_status: str = ResearchStageStatus.QUEUED.value
    budget: StageBudget = field(default_factory=StageBudget)
    input_payload: dict[str, Any] = field(default_factory=dict)
    partial_result: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    error_code: str = ""
    started_at: str = ""
    finished_at: str = ""
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        for name, value, limit in (
            ("stage_id", self.stage_id, 128), ("analysis_id", self.analysis_id, 128),
            ("owner_user_id", self.owner_user_id, 128), ("stage_type", self.stage_type, 64),
        ):
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise ValueError(f"{name}_invalid")
        try:
            stage_type = ResearchStageType(self.stage_type).value
        except ValueError as exc:
            raise ValueError("stage_type_invalid") from exc
        object.__setattr__(self, "stage_type", stage_type)
        try:
            status = ResearchStageStatus(self.stage_status).value
        except ValueError as exc:
            raise ValueError("stage_status_invalid") from exc
        object.__setattr__(self, "stage_status", status)
        object.__setattr__(self, "input_payload", normalize_research_input(self.stage_type, self.input_payload))
        object.__setattr__(self, "partial_result", safe_json(self.partial_result))
        object.__setattr__(self, "result", safe_json(self.result))
        if not isinstance(self.error_code, str) or len(self.error_code) > 128:
            raise ValueError("stage_error_code_invalid")
        if not self.created_at:
            object.__setattr__(self, "created_at", utc_now())
        if not self.updated_at:
            object.__setattr__(self, "updated_at", self.created_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id, "analysis_id": self.analysis_id, "stage_type": self.stage_type,
            "stage_status": self.stage_status, "budget": self.budget.to_dict(),
            "partial_result": safe_json(self.partial_result), "result": safe_json(self.result), "error_code": self.error_code,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }


class ResearchStageRepository:
    """Owner-scoped persistence with a closed transition vocabulary."""

    def __init__(self, store: Any) -> None:
        if store is None or not hasattr(store, "db") or not hasattr(store, "_db_lock"):
            raise ValueError("research_stage_store_required")
        self.store = store

    @staticmethod
    def _owner(owner_user_id: str | None) -> str:
        return owner_user_id or "legacy-owner"

    @staticmethod
    def _row_to_stage(row: Any) -> ResearchStage:
        input_payload = json.loads(row["input_json"]) if "input_json" in row.keys() else {}
        return ResearchStage(
            stage_id=str(row["stage_id"]), analysis_id=str(row["analysis_id"]), owner_user_id=str(row["owner_user_id"]),
            stage_type=str(row["stage_type"]), stage_status=str(row["stage_status"]),
            budget=StageBudget.from_value(json.loads(row["budget_json"])),
            input_payload=input_payload,
            partial_result=json.loads(row["partial_result_json"]), result=json.loads(row["result_json"]),
            error_code=str(row["error_code"]), started_at=str(row["started_at"]), finished_at=str(row["finished_at"]),
            created_at=str(row["created_at"]), updated_at=str(row["updated_at"]),
        )

    def create(
        self,
        analysis_id: str,
        stage_type: ResearchStageType | str,
        *,
        owner_user_id: str | None,
        budget: StageBudget | None = None,
        input_payload: Mapping[str, Any] | None = None,
    ) -> ResearchStage:
        owner = self._owner(owner_user_id)
        stage = ResearchStage(
            stage_id="STAGE-" + uuid.uuid4().hex,
            analysis_id=analysis_id, owner_user_id=owner,
            stage_type=ResearchStageType(stage_type).value,
            budget=budget or StageBudget(), input_payload=dict(input_payload or {}), created_at=utc_now(), updated_at=utc_now(),
        )
        with self.store._db_lock, self.store.db:
            count = self.store.db.execute("SELECT COUNT(*) AS count FROM bugbounty_research_stages WHERE analysis_id = ? AND owner_user_id = ?", (analysis_id, owner)).fetchone()
            if int(count["count"]) >= MAX_STAGE_COUNT_PER_ANALYSIS:
                raise ValueError("research_stage_limit_reached")
            self.store.db.execute(
                "INSERT INTO bugbounty_research_stages(stage_id, analysis_id, owner_user_id, stage_type, stage_status, budget_json, input_json, partial_result_json, result_json, error_code, started_at, finished_at, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (stage.stage_id, stage.analysis_id, stage.owner_user_id, stage.stage_type, stage.stage_status, json.dumps(stage.budget.to_dict(), sort_keys=True), json.dumps(stage.input_payload, ensure_ascii=False, sort_keys=True), json.dumps(stage.partial_result, sort_keys=True), json.dumps(stage.result, sort_keys=True), "", "", "", stage.created_at, stage.updated_at),
            )
        return stage

    @staticmethod
    def _lease_deadline(ttl_seconds: int) -> str:
        if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= MAX_STAGE_TIMEOUT_SECONDS:
            raise ValueError("stage_lease_ttl_invalid")
        return (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()

    def claim_next(self, worker_id: str, *, lease_ttl_seconds: int = 60) -> dict[str, Any] | None:
        """Atomically claim one queued stage.

        Expired RUNNING stages are made UNKNOWN, never requeued. This keeps an
        interrupted stage from being executed twice when its work could have
        produced an uncertain result.
        """
        if not isinstance(worker_id, str) or not worker_id.strip() or len(worker_id) > 128:
            raise ValueError("stage_worker_id_invalid")
        worker_id = worker_id.strip()
        acquired_at = utc_now()
        expires_at = self._lease_deadline(lease_ttl_seconds)
        lease_id = "STAGELEASE-" + uuid.uuid4().hex
        with self.store._db_lock:
            self.store.db.execute("BEGIN IMMEDIATE")
            try:
                self.store.db.execute(
                    """UPDATE bugbounty_research_stages
                       SET stage_status = ?, error_code = ?, last_error_code = ?,
                           last_error_message = ?, finished_at = ?, updated_at = ?,
                           lease_id = '', worker_id = '', lease_expires_at = ''
                       WHERE stage_status = ? AND (lease_expires_at = '' OR lease_expires_at <= ?)""",
                    (
                        ResearchStageStatus.UNKNOWN.value,
                        "stage_lease_expired",
                        "stage_lease_expired",
                        "Stage lease expired; automatic retry is disabled",
                        acquired_at,
                        acquired_at,
                        ResearchStageStatus.RUNNING.value,
                        acquired_at,
                    ),
                )
                row = self.store.db.execute(
                    """SELECT stage_id, analysis_id, owner_user_id, attempt_count
                       FROM bugbounty_research_stages
                       WHERE stage_status = ? AND attempt_count < max_attempts
                       ORDER BY created_at, stage_id LIMIT 1""",
                    (ResearchStageStatus.QUEUED.value,),
                ).fetchone()
                if row is None:
                    self.store.db.commit()
                    return None
                cursor = self.store.db.execute(
                    """UPDATE bugbounty_research_stages
                       SET stage_status = ?, started_at = ?, updated_at = ?,
                           attempt_count = attempt_count + 1, lease_id = ?, worker_id = ?,
                           lease_acquired_at = ?, lease_expires_at = ?, last_error_code = '', last_error_message = ''
                       WHERE stage_id = ? AND stage_status = ? AND attempt_count < max_attempts""",
                    (
                        ResearchStageStatus.RUNNING.value,
                        acquired_at,
                        acquired_at,
                        lease_id,
                        worker_id,
                        acquired_at,
                        expires_at,
                        str(row["stage_id"]),
                        ResearchStageStatus.QUEUED.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ValueError("stage_claim_lost")
                self.store.db.commit()
            except Exception:
                self.store.db.rollback()
                raise
        return {
            "kind": "CLAIMED",
            "stage_id": str(row["stage_id"]),
            "analysis_id": str(row["analysis_id"]),
            "owner_user_id": str(row["owner_user_id"]),
            "lease_id": lease_id,
            "worker_id": worker_id,
            "attempt_count": int(row["attempt_count"]),
            "lease_acquired_at": acquired_at,
            "lease_expires_at": expires_at,
        }

    @staticmethod
    def _bounded_payload(value: Mapping[str, Any] | None, *, max_bytes: int, name: str) -> dict[str, Any]:
        payload = safe_json(dict(value or {}))
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        if len(encoded) > max_bytes:
            raise ValueError(f"{name}_too_large")
        return payload

    def finish_claim(
        self,
        claim: Mapping[str, Any],
        status: ResearchStageStatus | str,
        *,
        result: Mapping[str, Any] | None = None,
        partial_result: Mapping[str, Any] | None = None,
        error_code: str = "",
        error_message: str = "",
    ) -> ResearchStage:
        status_value = ResearchStageStatus(status).value
        if status_value not in _TERMINAL:
            raise ValueError("research_stage_terminal_status_required")
        stage_id = str(claim.get("stage_id", ""))
        owner = str(claim.get("owner_user_id", ""))
        lease_id = str(claim.get("lease_id", ""))
        worker_id = str(claim.get("worker_id", ""))
        if not stage_id or not owner or not lease_id or not worker_id:
            raise ValueError("stage_claim_invalid")
        current = self.get(stage_id, owner_user_id=owner)
        if current is None:
            raise KeyError("research_stage_not_found")
        safe_result = self._bounded_payload(result, max_bytes=current.budget.max_output_bytes, name="stage_result")
        safe_partial = self._bounded_payload(partial_result, max_bytes=current.budget.max_output_bytes, name="stage_partial_result")
        if not isinstance(error_code, str) or len(error_code) > 128:
            raise ValueError("stage_error_code_invalid")
        if not isinstance(error_message, str) or len(error_message) > 512:
            raise ValueError("stage_error_message_invalid")
        stamp = utc_now()
        with self.store._db_lock:
            self.store.db.execute("BEGIN IMMEDIATE")
            try:
                cursor = self.store.db.execute(
                    """UPDATE bugbounty_research_stages
                       SET stage_status = ?, result_json = ?, partial_result_json = ?, error_code = ?,
                           last_error_code = ?, last_error_message = ?, finished_at = ?, updated_at = ?,
                           lease_id = '', worker_id = '', lease_expires_at = ''
                       WHERE stage_id = ? AND owner_user_id = ? AND stage_status = ?
                         AND lease_id = ? AND worker_id = ? AND lease_expires_at > ?""",
                    (
                        status_value,
                        json.dumps(safe_result, ensure_ascii=False, sort_keys=True),
                        json.dumps(safe_partial, ensure_ascii=False, sort_keys=True),
                        error_code,
                        error_code,
                        error_message,
                        stamp,
                        stamp,
                        stage_id,
                        owner,
                        ResearchStageStatus.RUNNING.value,
                        lease_id,
                        worker_id,
                        stamp,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ValueError("stage_lease_lost")
                self.store.db.commit()
            except Exception:
                self.store.db.rollback()
                raise
        result_stage = self.get(stage_id, owner_user_id=owner)
        if result_stage is None:
            raise KeyError("research_stage_not_found")
        return result_stage

    def mark_unknown_if_owned(self, claim: Mapping[str, Any], *, reason: str) -> bool:
        stage_id = str(claim.get("stage_id", ""))
        owner = str(claim.get("owner_user_id", ""))
        lease_id = str(claim.get("lease_id", ""))
        worker_id = str(claim.get("worker_id", ""))
        if not stage_id or not owner or not lease_id or not worker_id:
            return False
        reason = str(reason)[:128]
        stamp = utc_now()
        with self.store._db_lock, self.store.db:
            cursor = self.store.db.execute(
                """UPDATE bugbounty_research_stages
                   SET stage_status = ?, error_code = ?, last_error_code = ?, last_error_message = ?,
                       finished_at = ?, updated_at = ?, lease_id = '', worker_id = '', lease_expires_at = ''
                   WHERE stage_id = ? AND owner_user_id = ? AND stage_status = ? AND lease_id = ? AND worker_id = ?""",
                (
                    ResearchStageStatus.UNKNOWN.value,
                    reason,
                    reason,
                    "Stage execution became uncertain; automatic retry is disabled",
                    stamp,
                    stamp,
                    stage_id,
                    owner,
                    ResearchStageStatus.RUNNING.value,
                    lease_id,
                    worker_id,
                ),
            )
            return cursor.rowcount == 1

    def _get_owned(self, stage_id: str, owner_user_id: str | None, *, is_owner: bool = False) -> ResearchStage | None:
        owner = self._owner(owner_user_id)
        with self.store._db_lock:
            row = self.store.db.execute("SELECT * FROM bugbounty_research_stages WHERE stage_id = ?" + ("" if is_owner else " AND owner_user_id = ?"), (stage_id,) if is_owner else (stage_id, owner)).fetchone()
        return self._row_to_stage(row) if row is not None else None

    def get(self, stage_id: str, *, owner_user_id: str | None, is_owner: bool = False) -> ResearchStage | None:
        return self._get_owned(stage_id, owner_user_id, is_owner=is_owner)

    def list(self, analysis_id: str, *, owner_user_id: str | None, is_owner: bool = False) -> list[ResearchStage]:
        owner = self._owner(owner_user_id)
        query = "SELECT * FROM bugbounty_research_stages WHERE analysis_id = ?" + ("" if is_owner else " AND owner_user_id = ?") + " ORDER BY created_at, stage_id"
        params = (analysis_id,) if is_owner else (analysis_id, owner)
        with self.store._db_lock:
            rows = self.store.db.execute(query, params).fetchall()
        return [self._row_to_stage(row) for row in rows]

    def start(self, stage_id: str, *, owner_user_id: str | None, is_owner: bool = False) -> ResearchStage:
        stage = self._get_owned(stage_id, owner_user_id, is_owner=is_owner)
        if stage is None:
            raise KeyError("research_stage_not_found")
        if stage.stage_status != ResearchStageStatus.QUEUED.value:
            raise ValueError("research_stage_not_startable")
        stamp = utc_now()
        with self.store._db_lock, self.store.db:
            predicate = "stage_id = ? AND stage_status = 'QUEUED'" + ("" if is_owner else " AND owner_user_id = ?")
            params = (ResearchStageStatus.RUNNING.value, stamp, stamp, stage_id) if is_owner else (ResearchStageStatus.RUNNING.value, stamp, stamp, stage_id, self._owner(owner_user_id))
            cursor = self.store.db.execute(f"UPDATE bugbounty_research_stages SET stage_status = ?, started_at = ?, updated_at = ? WHERE {predicate}", params)
            if cursor.rowcount != 1:
                raise ValueError("research_stage_not_startable")
        return self._get_owned(stage_id, owner_user_id, is_owner=is_owner)  # type: ignore[return-value]

    def finish(self, stage_id: str, status: ResearchStageStatus | str, *, owner_user_id: str | None, result: Mapping[str, Any] | None = None, partial_result: Mapping[str, Any] | None = None, error_code: str = "", is_owner: bool = False) -> ResearchStage:
        status_value = ResearchStageStatus(status).value
        if status_value not in _TERMINAL:
            raise ValueError("research_stage_terminal_status_required")
        stage = self._get_owned(stage_id, owner_user_id, is_owner=is_owner)
        if stage is None:
            raise KeyError("research_stage_not_found")
        if stage.stage_status != ResearchStageStatus.RUNNING.value:
            raise ValueError("research_stage_not_running")
        stamp = utc_now()
        safe_result = self._bounded_payload(result, max_bytes=stage.budget.max_output_bytes, name="stage_result")
        safe_partial = self._bounded_payload(partial_result, max_bytes=stage.budget.max_output_bytes, name="stage_partial_result")
        if not isinstance(error_code, str) or len(error_code) > 128:
            raise ValueError("stage_error_code_invalid")
        with self.store._db_lock, self.store.db:
            predicate = "stage_id = ? AND stage_status = 'RUNNING' AND lease_id = '' AND worker_id = '' AND lease_expires_at = ''" + ("" if is_owner else " AND owner_user_id = ?")
            params = (status_value, json.dumps(safe_result, sort_keys=True), json.dumps(safe_partial, sort_keys=True), error_code, stamp, stamp, stage_id) if is_owner else (status_value, json.dumps(safe_result, sort_keys=True), json.dumps(safe_partial, sort_keys=True), error_code, stamp, stamp, stage_id, self._owner(owner_user_id))
            cursor = self.store.db.execute(f"UPDATE bugbounty_research_stages SET stage_status = ?, result_json = ?, partial_result_json = ?, error_code = ?, finished_at = ?, updated_at = ?, lease_id = '', worker_id = '', lease_expires_at = '' WHERE {predicate}", params)
            if cursor.rowcount != 1:
                raise ValueError("research_stage_not_running")
        return self._get_owned(stage_id, owner_user_id, is_owner=is_owner)  # type: ignore[return-value]

    def cancel(self, stage_id: str, *, owner_user_id: str | None, is_owner: bool = False) -> ResearchStage:
        stage = self._get_owned(stage_id, owner_user_id, is_owner=is_owner)
        if stage is None:
            raise KeyError("research_stage_not_found")
        if stage.stage_status in _TERMINAL:
            raise ValueError("research_stage_not_cancellable")
        stamp = utc_now()
        with self.store._db_lock, self.store.db:
            terminal = ",".join(f"'{value}'" for value in sorted(_TERMINAL))
            predicate = f"stage_id = ? AND stage_status NOT IN ({terminal})" + ("" if is_owner else " AND owner_user_id = ?")
            params = (ResearchStageStatus.CANCELLED.value, stamp, stamp, stage_id) if is_owner else (ResearchStageStatus.CANCELLED.value, stamp, stamp, stage_id, self._owner(owner_user_id))
            cursor = self.store.db.execute(f"UPDATE bugbounty_research_stages SET stage_status = ?, finished_at = ?, updated_at = ?, lease_id = '', worker_id = '', lease_expires_at = '' WHERE {predicate}", params)
            if cursor.rowcount != 1:
                raise ValueError("research_stage_not_cancellable")
        return self._get_owned(stage_id, owner_user_id, is_owner=is_owner)  # type: ignore[return-value]
