from __future__ import annotations

import base64
import binascii
import hashlib
import ipaddress
import json
import logging
import os
import re
import secrets
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Literal, Mapping
from urllib.parse import urlparse

from fastapi import APIRouter, Body, Depends, FastAPI, Header, HTTPException, Path as APIPath, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from private_agent.core.llm import build_provider
from private_agent.core.knowledge import (
    KnowledgeClaim,
    KnowledgeEvidence,
    KnowledgeSource,
    KnowledgeValidationError,
    sha256_hex,
)
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.planning_provenance import (
    UNKNOWN,
    normalize_plan_version,
    normalize_planning_mode,
    planning_mode_from_plan,
)
from private_agent.core.task_state import (
    MAX_TASK_GOAL_LENGTH,
    TASK_STATUSES,
    StateIntegrityError,
    TaskState,
    TaskStopReason,
    TaskStopReasonCode,
    action_fingerprint,
)
from private_agent.auth import (
    ACTIVE,
    CSRF_COOKIE_NAME,
    AuthError,
    AuthManager,
    AuthPrincipal,
    AuthUser,
    ROLE_OPERATOR,
    ROLE_OWNER,
    ROLE_VIEWER,
    SESSION_COOKIE_NAME,
)
from private_agent.bugbounty.models import Finding, FindingState, ScopeDefinition, public_reference, reconcile_bug_bounty_metrics, safe_json, validate_confirmed_evidence
from private_agent.bugbounty.learning import BugBountyLearningService
from private_agent.bugbounty.local_evidence import normalize_execution_stage_ids, normalize_execution_stage_refs, normalize_research_input
from private_agent.bugbounty.programs import ProgramAnalysisRepository, PublicPolicyRetriever, analyze_program, scope_proposal_digest
from private_agent.bugbounty.research import ResearchStageRepository, ResearchStageStatus, ResearchStageType, StageBudget
from private_agent.bugbounty.reviews import FindingReview, FindingReviewDecision, FindingReviewRepository
from private_agent.bugbounty.remote_worker import RemoteWorkerProtocol, WorkerAuthentication, WorkerProtocolError
from private_agent.observability import ObservabilityMonitor, safe_metadata
from private_agent.security import ApprovalRequest, SecurityController
from private_agent.skills.lifecycle import CapabilityCandidateAssessor, CapabilityCandidateAssessmentError, SkillCandidate
from private_agent.skills.registry import SkillRegistry
from private_agent.skills.runtime_preflight import run_host_preflight
from private_agent.storage import MAX_TASK_MESSAGE_LENGTH, QueueLeaseLost, Store
from private_agent.bugbounty.trusted_tools import load_tool_spec_from_environment
from private_agent.tools.research import ToolRegistry
from private_agent.tools.scanners import HttpxReconTool, NucleiScanTool, build_scanner_limits
from private_agent.worker import BackgroundWorker


API_VERSION = "v1"
API_SCHEMA_VERSION = 3
SERVICE_NAME = "private-agent"
LOGGER = logging.getLogger(__name__)
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_IDEMPOTENCY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_REDACTED_KEYS = {
    "input",
    "inputs",
    "output",
    "outputs",
    "payload",
    "raw_input",
    "raw_output",
    "raw_payload",
    "headers",
    "authorization",
    "credentials",
}
_PUBLIC_RECOVERY_REASONS = frozenset({
    "approval_required", "policy_blocked", "capability_unavailable",
    "execution_interrupted_before_verified_result", "worker_lease_expired",
    "worker_execution_interrupted", "ambiguous_execution_state_invalid_action_record",
    "invalid_persisted_action_record", "ambiguous_execution_state_missing_action_record",
    "ambiguous_execution_state_requires_observation", "unknown_execution_resolution_not_safe",
    "resume_security_revalidation_blocked", "persisted_action_completion_conflict",
    "persisted_verified_action_untrusted", "persisted_execution_result_corrupted",
    "conversation_security_revalidation_error", "max_execution_age_exceeded",
})
_PUBLIC_RECOVERY_REASON_PREFIXES = frozenset({"persisted_plan_invalid", "resume_security_revalidation_error"})
_PUBLIC_RECOVERY_STRATEGIES = frozenset({
    "RETRY", "RETRY_WITH_MODIFIED_INPUT", "CHANGE_TOOL", "CHANGE_PARAMETERS",
    "SKIP_OPTIONAL_STEP", "REPLAN", "ABORT",
})
_PUBLIC_FAILURE_TYPES = frozenset({
    "TOOL_ERROR", "INVALID_INPUT", "MISSING_DEPENDENCY", "PERMISSION_BLOCKED",
    "TIMEOUT", "RATE_LIMITED", "NETWORK_ERROR", "VERIFICATION_FAILED",
    "INSUFFICIENT_RESULT", "INVALID_OUTPUT", "UNKNOWN_FAILURE",
})
_BEARER = HTTPBearer(auto_error=False)


class APIError(Exception):
    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        *,
        task_id: str = "",
        approval_id: str = "",
        checks: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.task_id = task_id
        self.approval_id = approval_id
        self.checks = dict(checks or {})


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskCreateRequest(APIModel):
    goal: str = Field(min_length=1, max_length=MAX_TASK_GOAL_LENGTH)

    @field_validator("goal")
    @classmethod
    def non_blank_goal(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("goal_required")
        return value


class CapabilityCandidateAssessmentRequest(APIModel):
    target_tool: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=1, le=1_000_000)
    description: str = Field(default="", max_length=1024)
    source: str = Field(min_length=1, max_length=65_536)
    entry_point: str = Field(min_length=1, max_length=128)
    declared_capabilities: list[str] = Field(default_factory=list, max_length=64)
    allowed_imports: list[str] = Field(default_factory=list, max_length=64)
    allowed_network_domains: list[str] = Field(default_factory=list, max_length=64)
    expected_inputs: dict[str, Any] = Field(default_factory=dict)
    expected_outputs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("target_tool")
    @classmethod
    def valid_target_tool(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
            raise ValueError("candidate_target_tool_invalid")
        return value

    @field_validator("entry_point")
    @classmethod
    def valid_entry_point(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,127}", value):
            raise ValueError("candidate_entry_point_invalid")
        return value

    @field_validator("source")
    @classmethod
    def bounded_source_bytes(cls, value: str) -> str:
        if len(value.encode("utf-8")) > CapabilityCandidateAssessor.MAX_SOURCE_BYTES:
            raise ValueError("candidate_source_too_large")
        return value

    @field_validator("declared_capabilities", "allowed_imports", "allowed_network_domains")
    @classmethod
    def bounded_manifest_strings(cls, value: list[str]) -> list[str]:
        if any(not item.strip() or len(item) > 128 or any(ord(char) < 32 for char in item) for item in value):
            raise ValueError("candidate_manifest_value_invalid")
        return value

    @field_validator("expected_inputs", "expected_outputs")
    @classmethod
    def bounded_contract_objects(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError, RecursionError):
            raise ValueError("candidate_contract_invalid") from None
        if len(encoded) > 8_192:
            raise ValueError("candidate_contract_too_large")
        return value


class CapabilityCandidateAssessmentResponse(APIModel):
    candidate_id: str
    capability_gap_task_id: str
    target_tool: str
    version: int
    status: str
    review_status: str
    promotion_status: str = "BLOCKED"
    can_promote: bool = False
    blocking_gates: list[str] = Field(default_factory=list)
    gate_statuses: dict[str, str] = Field(default_factory=dict)
    source_sha256: str
    contract: dict[str, Any]
    analysis: dict[str, Any]
    execution_attempted: bool
    installation_attempted: bool
    registration_attempted: bool
    permissions_granted: bool
    created_at: str
    request_id: str = ""


class TaskMessageRequest(APIModel):
    content: str = Field(min_length=1, max_length=MAX_TASK_MESSAGE_LENGTH)

    @field_validator("content")
    @classmethod
    def non_blank_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("task_message_required")
        return value


class BugBountyCreateRequest(APIModel):
    scope: dict[str, Any]
    scanner: str = "deterministic"
    target: str | None = Field(default=None, max_length=2048)
    identity: str = Field(default="", max_length=256)
    resource_id: str = Field(default="", max_length=256)
    fixture: str | None = None
    urls: list[str] = Field(default_factory=list, max_length=100)
    openapi: dict[str, Any] | None = None
    source_documents: dict[str, str] = Field(default_factory=dict)

    @field_validator("scope")
    @classmethod
    def valid_scope(cls, value: dict[str, Any]) -> dict[str, Any]:
        ScopeDefinition.from_dict(value)
        return safe_json(value)

    @field_validator("scanner")
    @classmethod
    def valid_scanner(cls, value: str) -> str:
        if value not in {"deterministic", "httpx", "nuclei"}:
            raise ValueError("scanner_unsupported")
        return value

    @field_validator("target")
    @classmethod
    def valid_target(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or any(ord(char) < 32 for char in value)):
            raise ValueError("target_invalid")
        return value

    @model_validator(mode="after")
    def validate_scanner_binding(self) -> "BugBountyCreateRequest":
        if self.scanner in {"httpx", "nuclei"} and (
            not self.target or not self.identity or not self.resource_id
        ):
            raise ValueError("scanner_target_binding_required")
        if self.scanner == "deterministic" and self.target is not None:
            raise ValueError("deterministic_target_not_supported")
        return self

    @field_validator("fixture")
    @classmethod
    def valid_fixture(cls, value: str | None) -> str | None:
        allowed = {"secure", "idor_horizontal", "idor_vertical", "idor_nested", "false_positive", "inconclusive"}
        if value is not None and value not in allowed:
            raise ValueError("fixture_unsupported")
        return value

    @field_validator("urls")
    @classmethod
    def valid_urls(cls, value: list[str]) -> list[str]:
        if any(not isinstance(item, str) or not item.strip() or len(item) > 2048 for item in value):
            raise ValueError("urls_invalid")
        return list(value)

    @field_validator("openapi", "source_documents")
    @classmethod
    def bounded_json(cls, value: Any) -> Any:
        return safe_json(value or {})


class ResumeRequest(APIModel):
    approval_ids: dict[str, str] = Field(default_factory=dict)

    @field_validator("approval_ids")
    @classmethod
    def valid_approval_ids(cls, value: dict[str, str]) -> dict[str, str]:
        for key, approval_id in value.items():
            if not key or not approval_id or len(key) > 256 or len(approval_id) > 256:
                raise ValueError("approval_ids_invalid")
        return value


class ProgramAnalysisRequest(APIModel):
    program_url: str = Field(min_length=8, max_length=2048)
    policy_text: str = Field(default="", max_length=128_000)
    source_type: str = Field(default="user_provided_policy", min_length=1, max_length=64)
    fetch_public_policy: bool = False

    @model_validator(mode="after")
    def validate_public_policy_request(self) -> "ProgramAnalysisRequest":
        if self.fetch_public_policy:
            if self.policy_text.strip():
                raise ValueError("public_policy_fetch_cannot_include_policy_text")
            if self.source_type != "hackerone_program":
                raise ValueError("public_policy_source_not_supported")
        return self


class ProgramAnalysisTaskRequest(APIModel):
    proposal_digest: str = Field(min_length=64, max_length=64)
    authorization_reference: str = Field(min_length=1, max_length=512)
    authorization_attested: bool = False
    source_stage_ids: list[str] = Field(default_factory=list, max_length=4)

    @field_validator("proposal_digest")
    @classmethod
    def valid_proposal_digest(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            raise ValueError("proposal_digest_invalid")
        return value.lower()

    @field_validator("authorization_reference")
    @classmethod
    def valid_authorization_reference(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ord(char) < 32 for char in value):
            raise ValueError("authorization_reference_invalid")
        return value

    @field_validator("authorization_attested", mode="before")
    @classmethod
    def strict_authorization_attestation(cls, value: Any) -> bool:
        if type(value) is not bool:
            raise ValueError("authorization_attestation_invalid")
        return value

    @field_validator("source_stage_ids")
    @classmethod
    def valid_source_stage_ids(cls, value: list[str]) -> list[str]:
        return normalize_execution_stage_ids(value)


class ProgramAnalysisResponse(APIModel):
    analysis_id: str
    program_url: str
    source: dict[str, Any]
    status: str
    scope_proposal: dict[str, Any] | None = None
    proposal_digest: str = ""
    exclusions: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    observations: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    api_version: str = API_VERSION
    schema_version: int = API_SCHEMA_VERSION
    request_id: str = ""


class ResearchStageRequest(APIModel):
    stage_type: str
    budget: dict[str, Any] = Field(default_factory=dict)
    input_payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("stage_type")
    @classmethod
    def valid_stage_type(cls, value: str) -> str:
        try:
            return ResearchStageType(value).value
        except ValueError as exc:
            raise ValueError("stage_type_invalid") from exc

    @field_validator("budget")
    @classmethod
    def valid_budget(cls, value: dict[str, Any]) -> dict[str, Any]:
        StageBudget.from_value(value)
        return safe_json(value)

    @field_validator("input_payload")
    @classmethod
    def valid_input_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("stage_input_payload_object_required")
        return value

    @model_validator(mode="after")
    def validate_stage_input(self) -> "ResearchStageRequest":
        self.input_payload = normalize_research_input(self.stage_type, self.input_payload)
        return self


class ResearchStageResponse(APIModel):
    stage_id: str
    analysis_id: str
    stage_type: str
    stage_status: str
    budget: dict[str, int]
    source_stage_refs: list[dict[str, str]] = Field(default_factory=list)
    partial_result: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    error_code: str = ""
    started_at: str = ""
    finished_at: str = ""
    created_at: str
    updated_at: str
    api_version: str = API_VERSION
    schema_version: int = API_SCHEMA_VERSION
    request_id: str = ""


class FindingReviewRequest(APIModel):
    decision: str
    duplicate_of: str = Field(default="", max_length=128)
    note: str = Field(default="", max_length=2000)

    @field_validator("decision")
    @classmethod
    def valid_decision(cls, value: str) -> str:
        try:
            return FindingReviewDecision(value).value
        except ValueError as exc:
            raise ValueError("finding_review_decision_invalid") from exc


class FindingReviewResponse(APIModel):
    review_id: str
    task_id: str
    finding_id: str
    decision: str
    duplicate_of: str = ""
    note: str = ""
    evidence_sha256: str = ""
    evidence_validated: bool = False
    actor_user_id: str = ""
    created_at: str
    api_version: str = API_VERSION
    schema_version: int = API_SCHEMA_VERSION
    request_id: str = ""


class FindingKnowledgeAttestationResponse(APIModel):
    review_id: str
    attestation: dict[str, Any]
    api_version: str = API_VERSION
    schema_version: int = API_SCHEMA_VERSION
    request_id: str = ""


class ApprovalActionRequest(APIModel):
    """No client-controlled binding fields are accepted for approval actions."""

    pass


class ErrorResponse(APIModel):
    error_code: str
    message: str
    request_id: str
    task_id: str = ""
    approval_id: str = ""
    checks: dict[str, str] = Field(default_factory=dict)
    api_version: str = API_VERSION
    schema_version: int = API_SCHEMA_VERSION


class TaskCreateResponse(APIModel):
    task_id: str
    conversation_id: str
    status: str
    created_at: str
    current_phase: str
    api_version: str = API_VERSION
    schema_version: int = API_SCHEMA_VERSION
    request_id: str = ""
    idempotent_replay: bool = False


class TaskMessageResponse(APIModel):
    message_id: str
    conversation_id: str
    task_id: str
    sequence: int
    role: str
    message_kind: str
    content: str
    status: str
    created_at: str
    consumed_at: str = ""
    consumed_checkpoint: str = ""
    idempotent_replay: bool = False
    request_id: str = ""


class TaskResultEventResponse(APIModel):
    event_id: str
    conversation_id: str
    task_id: str
    sequence: int
    role: str
    message_kind: str
    task_status: str
    verification_status: str
    planning_mode: str
    plan_version: int
    result_available: bool
    verified_action_count: int
    failed_action_count: int
    blocked_action_count: int
    pending_action_count: int
    approval_pending_count: int
    approval_approved_count: int
    approval_denied_count: int
    approval_expired_count: int
    approval_cancelled_count: int
    result_digest: str
    created_at: str


class TaskConversationItemResponse(APIModel):
    item_type: str
    sequence: int
    message: TaskMessageResponse | None = None
    result_event: TaskResultEventResponse | None = None

    @model_validator(mode="after")
    def validate_item(self) -> "TaskConversationItemResponse":
        if self.item_type == "USER_MESSAGE":
            if self.message is None or self.result_event is not None or self.message.sequence != self.sequence:
                raise ValueError("task_conversation_item_invalid")
        elif self.item_type == "SYSTEM_RESULT":
            if self.result_event is None or self.message is not None or self.result_event.sequence != self.sequence:
                raise ValueError("task_conversation_item_invalid")
        else:
            raise ValueError("task_conversation_item_invalid")
        return self


class TaskAssistantReplyResponse(APIModel):
    reply_id: str
    conversation_id: str
    task_id: str
    source_event_id: str
    source_sequence: int
    role: str
    message_kind: str
    reply_code: str
    content: str
    created_at: str


class TaskConversationResponse(APIModel):
    conversation_id: str
    task_id: str
    task_status: str
    queue_status: str
    items: list[TaskConversationItemResponse] = Field(default_factory=list)
    messages: list[TaskMessageResponse]
    result_events: list[TaskResultEventResponse] = Field(default_factory=list)
    assistant_replies: list[TaskAssistantReplyResponse] = Field(default_factory=list)
    assistant_reply_supported: bool = False
    next_cursor: int | None = None
    has_more: bool = False
    api_version: str = API_VERSION
    schema_version: int = API_SCHEMA_VERSION
    request_id: str = ""

    @model_validator(mode="after")
    def validate_ordered_items(self) -> "TaskConversationResponse":
        sequences = [item.sequence for item in self.items]
        if any(sequence < 1 for sequence in sequences) or any(
            left >= right for left, right in zip(sequences, sequences[1:])
        ):
            raise ValueError("task_conversation_sequence_invalid")
        message_ids = [item.message.message_id for item in self.items if item.message is not None]
        result_ids = [item.result_event.event_id for item in self.items if item.result_event is not None]
        if message_ids != [message.message_id for message in self.messages] or result_ids != [
            event.event_id for event in self.result_events
        ]:
            raise ValueError("task_conversation_items_mismatch")
        event_bindings = {(event.event_id, event.sequence) for event in self.result_events}
        if any((reply.source_event_id, reply.source_sequence) not in event_bindings for reply in self.assistant_replies):
            raise ValueError("task_conversation_assistant_reply_mismatch")
        if self.has_more:
            if not sequences or self.next_cursor != sequences[-1]:
                raise ValueError("task_conversation_cursor_invalid")
        elif self.next_cursor is not None:
            raise ValueError("task_conversation_cursor_invalid")
        return self


class TaskStatusFilter(str, Enum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    PLANNING = "PLANNING"
    READY = "READY"
    EXECUTING = "EXECUTING"
    EXECUTION_UNKNOWN = "EXECUTION_UNKNOWN"
    OBSERVING = "OBSERVING"
    VERIFYING = "VERIFYING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class TaskStopReasonResponse(APIModel):
    """Closed public reason code and safe identifiers; no free-form detail."""

    code: TaskStopReasonCode
    step_id: str | None = None
    action_id: str | None = None


class TaskSummaryResponse(APIModel):
    task_id: str
    goal: str
    status: str
    queue_status: str
    current_phase: str
    current_step: str
    plan_version: int
    verification_status: str
    attempt_count: int
    created_at: str
    updated_at: str
    last_checkpoint_at: str
    last_successful_checkpoint: str
    available_at: str
    last_error_code: str = ""
    waiting_for_approval: bool = False
    execution_unknown: bool = False
    stop_reason: TaskStopReasonResponse | None = None


class TaskListResponse(APIModel):
    items: list[TaskSummaryResponse]
    next_cursor: str | None = None
    api_version: str = API_VERSION
    schema_version: int = API_SCHEMA_VERSION
    request_id: str = ""


class BugBountyTaskResponse(APIModel):
    task_id: str
    status: str
    verification_status: str
    recovery_state: dict[str, Any] = Field(default_factory=dict)
    run: dict[str, Any] | None = None
    scan: dict[str, Any] | None = None
    analysis_provenance: dict[str, Any] | None = None
    report: str = ""
    limitations: list[str] = Field(default_factory=list)
    api_version: str = API_VERSION
    schema_version: int = API_SCHEMA_VERSION
    request_id: str = ""


class BugBountySectionResponse(APIModel):
    task_id: str
    status: str
    data: Any = None
    api_version: str = API_VERSION
    schema_version: int = API_SCHEMA_VERSION
    request_id: str = ""


class CheckpointResponse(APIModel):
    last_checkpoint_at: str
    last_successful_checkpoint: str
    checkpoint_type: str = ""
    sequence: int = 0


class ActionSummary(APIModel):
    action_id: str
    status: str
    step_id: str = ""
    tool_name: str = ""
    execution_status: str = ""
    planning_mode: str = UNKNOWN
    plan_version: int = 0
    input_fingerprint: str = ""
    updated_at: str = ""


class ApprovalResponse(APIModel):
    approval_id: str
    task_id: str
    action_id: str
    tool_or_skill: str
    requested_capabilities: list[str]
    risk_level: str
    reason: str
    planned_effect: str
    input_fingerprint: str
    policy_version: int
    requested_at: str
    expires_at: str
    status: str
    decided_at: str = ""
    actor: str = ""
    api_version: str = API_VERSION
    schema_version: int = API_SCHEMA_VERSION


class TaskStateResponse(APIModel):
    task_id: str
    goal: str
    task_kind: str
    status: str
    current_phase: str
    current_step: str
    plan_version: int
    planning_mode: str
    completed_actions: list[str]
    pending_actions: list[str]
    failed_actions: list[str]
    actions: list[ActionSummary]
    recovery_state: dict[str, Any]
    verification_status: str
    checkpoint: CheckpointResponse
    approvals: list[ApprovalResponse]
    specialist_assignment_snapshot: "SpecialistAssignmentSnapshotResponse"
    created_at: str
    updated_at: str
    queue_status: str = ""
    stop_reason: TaskStopReasonResponse | None = None
    api_version: str = API_VERSION
    schema_version: int = API_SCHEMA_VERSION
    request_id: str = ""


class SpecialistAssignmentResponse(APIModel):
    step_id: str = Field(pattern=r"^[A-Za-z0-9._:-]{1,128}$")
    specialist_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    profile_version: int = Field(ge=1)
    tool_name: str = Field(pattern=r"^[A-Za-z0-9_.:-]{1,128}$")
    capabilities: list[Literal["READ_WORKSPACE"]]
    status: Literal["ASSIGNED", "EXECUTING", "OBSERVING", "VERIFIED", "FAILED", "BLOCKED", "EXECUTION_UNKNOWN", "UNKNOWN"]
    reason: TaskStopReasonCode | None = None


class SpecialistAssignmentSnapshotResponse(APIModel):
    task_id: str
    plan_version: int = Field(ge=0)
    status: Literal["AVAILABLE", "NONE", "UNAVAILABLE"]
    provenance: Literal["SERVER_CURRENT_PLAN"] = "SERVER_CURRENT_PLAN"
    reason: Literal["assignment_not_verifiable"] | None = None
    assignments: list[SpecialistAssignmentResponse] = Field(default_factory=list)


class EventResponse(APIModel):
    event_id: str
    event_type: str
    timestamp: str
    task_id: str
    action_id: str
    step_id: str
    correlation_id: str
    component: str
    severity: str
    schema_version: int
    sequence: int
    metadata: dict[str, Any]
    api_version: str = API_VERSION
    response_schema_version: int = API_SCHEMA_VERSION


class MetricsResponse(APIModel):
    model_config = ConfigDict(extra="allow")

    task_id: str | None = None
    task_count: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    blocked_tasks: int = 0
    action_count: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    blocked_actions: int = 0
    blocked_lifecycle_actions: int = 0
    verification_success: int = 0
    verification_failure: int = 0
    recovery_count: int = 0
    replan_count: int = 0
    approval_requests: int = 0
    approval_approvals: int = 0
    approval_denials: int = 0
    approval_expirations: int = 0
    tool_failures: int = 0
    execution_unknown: int = 0
    task_duration_ms: float = 0.0
    action_duration_ms: float = 0.0
    tool_duration_ms: float = 0.0
    planning_duration_ms: float = 0.0
    observation_duration_ms: float = 0.0
    verification_duration_ms: float = 0.0
    recovery_duration_ms: float = 0.0
    observability_event_count: int = 0
    api_version: str = API_VERSION
    schema_version: int = API_SCHEMA_VERSION


class HealthResponse(APIModel):
    status: str
    service: str = SERVICE_NAME
    api_version: str = API_VERSION
    schema_version: int = API_SCHEMA_VERSION


class ReadinessResponse(APIModel):
    status: str
    checks: dict[str, str]
    service: str = SERVICE_NAME
    api_version: str = API_VERSION
    schema_version: int = API_SCHEMA_VERSION
    request_id: str = ""


class RuntimePreflightCheckResponse(APIModel):
    name: str
    status: str
    detail: str


class RuntimePreflightResponse(APIModel):
    verdict: str
    host_ready: bool
    can_run_acceptance: bool
    acceptance_required: bool
    runtime_identity: str | None = None
    checks: list[RuntimePreflightCheckResponse]
    request_id: str = ""


class ToolResponse(APIModel):
    name: str
    description: str = ""
    permission_level: str = ""
    risk_level: str = ""
    required_capabilities: list[str] = Field(default_factory=list)
    network_required: bool = False


class AuthenticatedPrincipal(AuthPrincipal):
    """Compatibility name for the API's authenticated principal contract."""


class LoginRequest(APIModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=12, max_length=1024)


class AuthUserResponse(APIModel):
    user_id: str
    email: str
    role: str
    status: str
    created_at: str
    updated_at: str
    last_login_at: str = ""


class LoginResponse(APIModel):
    authenticated: bool = True
    user: AuthUserResponse
    expires_at: str
    api_version: str = API_VERSION
    schema_version: int = API_SCHEMA_VERSION


class UserCreateRequest(APIModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=12, max_length=1024)
    role: str = Field(pattern="^(OPERATOR|VIEWER)$")


class UserRoleRequest(APIModel):
    role: str = Field(pattern="^(OPERATOR|VIEWER)$")


class WorkerClaimRequest(APIModel):
    pass


class WorkerHeartbeatRequest(APIModel):
    job_id: str = Field(min_length=1, max_length=256)
    lease_id: str = Field(min_length=1, max_length=256)


class WorkerResultRequest(APIModel):
    job_id: str = Field(min_length=1, max_length=256)
    lease_id: str = Field(min_length=1, max_length=256)
    result: dict[str, Any]


class TokenAuthenticator:
    """Bearer authentication for the single-owner deployment.

    There is intentionally no development fallback. A missing configuration is
    a readiness failure, not an authentication bypass.
    """

    _MIN_TOKEN_LENGTH = 16
    _PLACEHOLDER_MARKERS = ("change_me", "replace_me", "replace-with", "your_token", "example_token", "placeholder")

    def __init__(self, token: str | None = None) -> None:
        self.token = token if token is not None else os.getenv("AGENT_API_TOKEN")

    @property
    def configured(self) -> bool:
        if not isinstance(self.token, str):
            return False
        candidate = self.token.strip()
        if len(candidate) < self._MIN_TOKEN_LENGTH or any(
            marker in candidate.lower() for marker in self._PLACEHOLDER_MARKERS
        ):
            return False
        return candidate == self.token and " " not in candidate

    def authenticate(self, authorization: str | None) -> AuthenticatedPrincipal:
        if not self.configured:
            raise APIError(503, "api_auth_not_configured", "API authentication is not configured")
        if not authorization or not authorization.startswith("Bearer "):
            raise APIError(401, "unauthenticated", "Bearer authentication is required")
        candidate = authorization[len("Bearer ") :]
        if not candidate or " " in candidate or not secrets.compare_digest(candidate, str(self.token)):
            raise APIError(401, "invalid_credentials", "Invalid API credentials")
        return AuthenticatedPrincipal(
            subject="single-owner",
            user_id="",
            email="",
            role=ROLE_OWNER,
            auth_type="legacy_bearer",
        )


def _public_value(value: Any, *, key: str = "") -> Any:
    """Return a response-safe projection, never a raw internal object."""
    lowered = key.lower()
    if lowered in _REDACTED_KEYS:
        return {"redacted": True}
    if isinstance(value, dict):
        return {str(name): _public_value(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_public_value(item, key=key) for item in value[:100]]
    if isinstance(value, tuple):
        return [_public_value(item, key=key) for item in value[:100]]
    return safe_metadata({"value": value}).get("value")


def _recovery_identifier(value: Any) -> str:
    if isinstance(value, str) and _REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return ""


def _recovery_identifier_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(
        item for item in (_recovery_identifier(candidate) for candidate in value[:limit]) if item
    ))


def _public_recovery_state(value: Any) -> dict[str, Any]:
    """Project task recovery data through a closed schema; omit raw diagnostics and inputs."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    reason = value.get("reason")
    if isinstance(reason, str):
        reason_code = reason.partition(":")[0]
        if reason in _PUBLIC_RECOVERY_REASONS:
            result["reason"] = reason
        elif reason_code in _PUBLIC_RECOVERY_REASON_PREFIXES:
            result["reason"] = reason_code
        else:
            result["reason"] = "recovery_required"

    decision_value = value.get("decision")
    if isinstance(decision_value, dict):
        decision: dict[str, Any] = {}
        strategy = decision_value.get("strategy")
        if isinstance(strategy, str) and strategy in _PUBLIC_RECOVERY_STRATEGIES:
            decision["strategy"] = strategy
        for field in ("step_id", "original_tool", "replacement_tool"):
            identifier = _recovery_identifier(decision_value.get(field))
            if identifier:
                decision[field] = identifier
        terminal_status = decision_value.get("terminal_status")
        if isinstance(terminal_status, str) and terminal_status in {"FAILED", "BLOCKED", "CANCELLED", "COMPLETED"}:
            decision["terminal_status"] = terminal_status
        if decision:
            result["decision"] = decision

    diagnosis_value = value.get("diagnosis")
    if isinstance(diagnosis_value, dict):
        diagnosis: dict[str, Any] = {}
        failure_type = diagnosis_value.get("failure_type")
        if isinstance(failure_type, str) and failure_type in _PUBLIC_FAILURE_TYPES:
            diagnosis["failure_type"] = failure_type
        for field in ("failed_step", "tool_name"):
            identifier = _recovery_identifier(diagnosis_value.get(field))
            if identifier:
                diagnosis[field] = identifier
        recoverable = diagnosis_value.get("recoverable")
        if isinstance(recoverable, bool):
            diagnosis["recoverable"] = recoverable
        suggested_recovery = diagnosis_value.get("suggested_recovery")
        if isinstance(suggested_recovery, str) and suggested_recovery in _PUBLIC_RECOVERY_STRATEGIES:
            diagnosis["suggested_recovery"] = suggested_recovery
        if diagnosis:
            result["diagnosis"] = diagnosis

    round_number = value.get("round")
    if isinstance(round_number, int) and not isinstance(round_number, bool) and 0 <= round_number <= 1_000_000:
        result["round"] = round_number

    if value.get("external_side_effect_may_have_occurred") is True:
        result["external_side_effect_may_have_occurred"] = True
    for field in ("action_key", "step_id"):
        identifier = _recovery_identifier(value.get(field))
        if identifier:
            result[field] = identifier
    invalid_keys = value.get("invalid_action_record_keys")
    if isinstance(invalid_keys, list):
        result["invalid_action_record_count"] = min(len(invalid_keys), 100_000)

    gap_value = value.get("capability_gap")
    if isinstance(gap_value, dict):
        gap: dict[str, Any] = {}
        code = gap_value.get("code")
        if isinstance(code, str) and code in {"UNREGISTERED_TOOL_REQUEST", "NO_REGISTERED_TOOLS"}:
            gap["code"] = code
        if gap_value.get("source") == "planner_registry_validation":
            gap["source"] = "planner_registry_validation"
        summary_by_code = {
            "UNREGISTERED_TOOL_REQUEST": "The planner requested a tool that is not registered; no tool was executed or installed.",
            "NO_REGISTERED_TOOLS": "No tools are registered for planning; no tool was executed or installed.",
        }
        if isinstance(code, str) and code in summary_by_code and gap_value.get("summary") == summary_by_code[code]:
            gap["summary"] = summary_by_code[code]
        for field, limit in (("requested_tools", 20), ("registered_tools", 50)):
            identifiers = _recovery_identifier_list(gap_value.get(field), limit=limit)
            if field in gap_value:
                gap[field] = identifiers
        unavailable_count = gap_value.get("unavailable_tool_count")
        if isinstance(unavailable_count, int) and not isinstance(unavailable_count, bool) and 0 <= unavailable_count <= 1_000_000:
            gap["unavailable_tool_count"] = unavailable_count
        for field in ("execution_attempted", "acquisition_attempted", "permissions_granted"):
            flag = gap_value.get(field)
            if isinstance(flag, bool):
                gap[field] = flag
        if gap:
            result["capability_gap"] = gap
    return result


def _safe_id(value: str, field: str) -> str:
    if not value or len(value) > 256 or not _REQUEST_ID_PATTERN.fullmatch(value):
        raise APIError(400, f"invalid_{field}", f"Invalid {field}")
    return value


def _configured_cors_origins() -> list[str]:
    raw = os.getenv("AGENT_ALLOWED_ORIGINS", "")
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    if "*" in origins:
        raise ValueError("wildcard_cors_not_allowed")
    for origin in origins:
        parsed = urlparse(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("cors_origin_invalid")
        if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("cors_origin_insecure")
    return origins


def _encode_task_cursor(created_at: str, task_id: str) -> str:
    payload = json.dumps(
        {"created_at": created_at, "task_id": task_id},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_task_cursor(cursor: str) -> tuple[str, str]:
    if not cursor or len(cursor) > 512 or not re.fullmatch(r"[A-Za-z0-9_-]+", cursor):
        raise APIError(400, "invalid_cursor", "Invalid task list cursor")
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except (binascii.Error, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise APIError(400, "invalid_cursor", "Invalid task list cursor") from exc
    if not isinstance(payload, dict) or set(payload) != {"created_at", "task_id"}:
        raise APIError(400, "invalid_cursor", "Invalid task list cursor")
    created_at = payload.get("created_at")
    task_id = payload.get("task_id")
    if not isinstance(created_at, str) or not created_at or len(created_at) > 128:
        raise APIError(400, "invalid_cursor", "Invalid task list cursor")
    if not isinstance(task_id, str):
        raise APIError(400, "invalid_cursor", "Invalid task list cursor")
    try:
        task_id = _safe_id(task_id, "cursor_task_id")
    except APIError as exc:
        raise APIError(400, "invalid_cursor", "Invalid task list cursor") from exc
    return created_at, task_id


def _verification_status(state: TaskState) -> str:
    verification_status = "NOT_STARTED"
    statuses = [str(item.get("status", "")) for item in state.verification_results.values() if isinstance(item, dict)]
    if "BLOCKED" in statuses:
        verification_status = "BLOCKED"
    elif "FAILED" in statuses:
        verification_status = "FAILED"
    elif "INSUFFICIENT" in statuses:
        verification_status = "INSUFFICIENT"
    elif statuses and all(item == "VERIFIED" for item in statuses):
        verification_status = "VERIFIED"
    elif statuses:
        verification_status = statuses[-1]
    if isinstance(state.result.get("verification"), dict):
        result_status = state.result["verification"].get("status")
        if result_status:
            verification_status = str(result_status)
    return verification_status


def _task_stop_reason_response(reason: TaskStopReason | None) -> TaskStopReasonResponse | None:
    if reason is None:
        return None
    return TaskStopReasonResponse(
        code=reason.code,
        step_id=reason.step_id,
        action_id=reason.action_id,
    )


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", "")
    return value if isinstance(value, str) and value else uuid.uuid4().hex


def _event_response(event: dict[str, Any]) -> EventResponse:
    return EventResponse(
        event_id=str(event.get("event_id", "")),
        event_type=str(event.get("event_type", "")),
        timestamp=str(event.get("timestamp", "")),
        task_id=str(event.get("task_id", "")),
        action_id=str(event.get("action_id", "")),
        step_id=str(event.get("step_id", "")),
        correlation_id=str(event.get("correlation_id", "")),
        component=str(event.get("component", "")),
        severity=str(event.get("severity", "INFO")),
        schema_version=int(event.get("schema_version", 1)),
        sequence=int(event.get("sequence", 0)),
        metadata=_public_value(dict(event.get("metadata") or {})),
    )


def _approval_response(request: ApprovalRequest | dict[str, Any]) -> ApprovalResponse:
    payload = request.to_dict() if hasattr(request, "to_dict") else dict(request)
    return ApprovalResponse(
        approval_id=str(payload.get("approval_id", "")),
        task_id=str(payload.get("task_id", "")),
        action_id=str(payload.get("action_id", "")),
        tool_or_skill=str(payload.get("tool_or_skill", "")),
        requested_capabilities=[str(item) for item in payload.get("requested_capabilities", [])],
        risk_level=str(payload.get("risk_level", "")),
        reason=str(_public_value(payload.get("reason", ""))),
        planned_effect=str(_public_value(payload.get("planned_effect", ""))),
        input_fingerprint=str(payload.get("input_fingerprint", "")),
        policy_version=int(payload.get("policy_version", 0)),
        requested_at=str(payload.get("requested_at", "")),
        expires_at=str(payload.get("expires_at", "")),
        status=str(payload.get("status", "")),
        decided_at=str(payload.get("decided_at", "")),
        actor=str(payload.get("actor", "")),
    )


class AgentAPIService:
    """Thin, serialized adapter around the already-authoritative Agent Core."""

    def __init__(self, orchestrator: Orchestrator, *, token: str | None = None) -> None:
        self.orchestrator = orchestrator
        self.store: Store = orchestrator.store
        self.security: SecurityController | None = orchestrator.security
        self.observability: ObservabilityMonitor = orchestrator.observability
        self.authenticator = TokenAuthenticator(token)
        self.auth_manager = AuthManager(self.store)
        self.public_policy_retriever = PublicPolicyRetriever()
        self.remote_worker_protocol = RemoteWorkerProtocol(self.store)
        self.skill_registry = SkillRegistry(self.store)
        self.capability_candidate_assessor = CapabilityCandidateAssessor(self.skill_registry, orchestrator.tools)
        self.require_owner_auth = os.getenv("AGENT_REQUIRE_OWNER_AUTH", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.auto_requeue_approvals = False
        self._lock = threading.RLock()

    @property
    def assistant_reply_supported(self) -> bool:
        planner = getattr(self.orchestrator, "planner", None)
        provider = getattr(planner, "provider", None)
        if not callable(getattr(provider, "generate_json", None)):
            return False
        return getattr(provider, "is_configured", True) is not False

    @staticmethod
    def auth_user_response(user: AuthUser) -> AuthUserResponse:
        return AuthUserResponse(
            user_id=user.user_id,
            email=user.email,
            role=user.role,
            status=user.status,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login_at=user.last_login_at,
        )

    def owner_auth_enforced(self) -> bool:
        """A populated owner database must never silently retain bearer fallback."""
        return self.require_owner_auth or self.auth_manager.owner_status() != "missing"

    def authenticate_request(self, request: Request, authorization: str | None) -> AuthenticatedPrincipal:
        if authorization:
            if self.owner_auth_enforced():
                raise APIError(401, "legacy_bearer_disabled", "Bearer token authentication is disabled for this deployment")
            legacy = self.authenticator.authenticate(authorization)
            return legacy
        try:
            return self.auth_manager.authenticate_session(request.cookies.get(SESSION_COOKIE_NAME))
        except AuthError as exc:
            raise APIError(401, exc.code, exc.message) from exc

    def login(self, email: str, password: str, *, remote: str | None = None) -> Any:
        try:
            return self.auth_manager.login(email, password, remote=remote)
        except AuthError as exc:
            status_code = 503 if exc.code == "auth_state_invalid" else 429 if exc.code in {"login_locked", "login_rate_limited"} else 401
            raise APIError(status_code, exc.code, exc.message) from exc

    def verify_csrf(self, principal: AuthenticatedPrincipal, token: str | None) -> None:
        try:
            self.auth_manager.verify_csrf(principal, token)
        except AuthError as exc:
            raise APIError(403, exc.code, exc.message) from exc

    def logout(self, principal: AuthenticatedPrincipal) -> None:
        self.auth_manager.revoke_session(principal)

    def session_response(self, principal: AuthenticatedPrincipal) -> LoginResponse:
        if principal.is_legacy:
            user = AuthUser(
                user_id="legacy-owner",
                email=self.auth_manager.owner_email,
                role=ROLE_OWNER,
                status=ACTIVE,
                created_at="",
                updated_at="",
                last_login_at="",
            )
            return LoginResponse(user=self.auth_user_response(user), expires_at="")
        user = next((item for item in self.auth_manager.list_users() if item.user_id == principal.user_id), None)
        if user is None:
            raise APIError(401, "unauthenticated", "Authentication is required")
        return LoginResponse(user=self.auth_user_response(user), expires_at=self.auth_manager.session_expiry(principal))

    def analyze_program(
        self,
        payload: ProgramAnalysisRequest,
        *,
        request_id: str = "",
        principal: AuthenticatedPrincipal | None = None,
    ) -> ProgramAnalysisResponse:
        try:
            policy_text = payload.policy_text
            source_type = payload.source_type
            retrieval_observations: dict[str, Any] = {}
            if payload.fetch_public_policy:
                try:
                    retrieved = self.public_policy_retriever.retrieve(payload.program_url)
                except (TypeError, ValueError) as exc:
                    raise APIError(422, str(exc), "The allowlisted public policy could not be retrieved") from exc
                policy_text = retrieved.content
                source_type = retrieved.source_type
                retrieval_observations = retrieved.observations
            analysis = analyze_program(
                payload.program_url,
                policy_text,
                source_type=source_type,
                fetched_automatically=payload.fetch_public_policy,
                observations=retrieval_observations,
            )
            stored = ProgramAnalysisRepository(self.store).save(
                analysis,
                owner_user_id=self._owner_user_id(principal),
            )
        except (TypeError, ValueError) as exc:
            raise APIError(422, str(exc), "Program policy could not be analyzed") from exc
        self.observability.emit(
            "OBSERVATION_COMPLETED",
            correlation_id=request_id,
            component="bugbounty",
            metadata={"analysis_id": stored.analysis_id, "status": stored.status},
        )
        return ProgramAnalysisResponse(**stored.to_dict(), request_id=request_id)

    def get_program_analysis(
        self,
        analysis_id: str,
        *,
        principal: AuthenticatedPrincipal | None = None,
        request_id: str = "",
    ) -> ProgramAnalysisResponse:
        analysis_id = _safe_id(analysis_id, "analysis_id")
        payload = ProgramAnalysisRepository(self.store).get(
            analysis_id,
            owner_user_id=self._owner_user_id(principal),
            is_owner=principal is not None and principal.role == ROLE_OWNER,
        )
        if payload is None:
            raise APIError(404, "program_analysis_not_found", "Program analysis was not found")
        payload = dict(payload)
        payload["proposal_digest"] = scope_proposal_digest(payload.get("scope_proposal"))
        return ProgramAnalysisResponse(**payload, request_id=request_id)

    def list_program_analyses(
        self,
        *,
        principal: AuthenticatedPrincipal | None = None,
        limit: int = 50,
    ) -> list[ProgramAnalysisResponse]:
        try:
            payloads = ProgramAnalysisRepository(self.store).list(
                owner_user_id=self._owner_user_id(principal),
                is_owner=principal is not None and principal.role == ROLE_OWNER,
                limit=limit,
            )
        except ValueError as exc:
            raise APIError(422, str(exc), "Program analysis limit is invalid") from exc
        responses: list[ProgramAnalysisResponse] = []
        for payload in payloads:
            value = dict(payload)
            value["proposal_digest"] = scope_proposal_digest(value.get("scope_proposal"))
            responses.append(ProgramAnalysisResponse(**value))
        return responses

    def create_research_stage(
        self,
        analysis_id: str,
        payload: ResearchStageRequest,
        *,
        principal: AuthenticatedPrincipal | None = None,
        request_id: str = "",
    ) -> ResearchStageResponse:
        analysis_id = _safe_id(analysis_id, "analysis_id")
        owner = principal is not None and principal.role == ROLE_OWNER
        analysis = ProgramAnalysisRepository(self.store).get(analysis_id, owner_user_id=self._owner_user_id(principal), is_owner=owner)
        if analysis is None:
            raise APIError(404, "program_analysis_not_found", "Program analysis was not found")
        try:
            stage = ResearchStageRepository(self.store).create(
                analysis_id,
                payload.stage_type,
                owner_user_id=self._owner_user_id(principal),
                budget=StageBudget.from_value(payload.budget),
                input_payload=payload.input_payload,
            )
        except (TypeError, ValueError) as exc:
            raise APIError(422, str(exc), "Research stage could not be created") from exc
        return ResearchStageResponse(**stage.to_dict(), request_id=request_id)

    def list_research_stages(
        self,
        analysis_id: str,
        *,
        principal: AuthenticatedPrincipal | None = None,
        request_id: str = "",
    ) -> list[ResearchStageResponse]:
        analysis_id = _safe_id(analysis_id, "analysis_id")
        owner = principal is not None and principal.role == ROLE_OWNER
        analysis = ProgramAnalysisRepository(self.store).get(analysis_id, owner_user_id=self._owner_user_id(principal), is_owner=owner)
        if analysis is None:
            raise APIError(404, "program_analysis_not_found", "Program analysis was not found")
        stages = ResearchStageRepository(self.store).list(analysis_id, owner_user_id=self._owner_user_id(principal), is_owner=owner)
        return [ResearchStageResponse(**stage.to_dict(), request_id=request_id) for stage in stages]

    def get_research_stage(
        self,
        stage_id: str,
        *,
        principal: AuthenticatedPrincipal | None = None,
        request_id: str = "",
    ) -> ResearchStageResponse:
        stage_id = _safe_id(stage_id, "stage_id")
        owner = principal is not None and principal.role == ROLE_OWNER
        stage = ResearchStageRepository(self.store).get(stage_id, owner_user_id=self._owner_user_id(principal), is_owner=owner)
        if stage is None:
            raise APIError(404, "research_stage_not_found", "Research stage was not found")
        return ResearchStageResponse(**stage.to_dict(), request_id=request_id)

    def cancel_research_stage(
        self,
        stage_id: str,
        *,
        principal: AuthenticatedPrincipal | None = None,
        request_id: str = "",
    ) -> ResearchStageResponse:
        stage_id = _safe_id(stage_id, "stage_id")
        owner = principal is not None and principal.role == ROLE_OWNER
        try:
            stage = ResearchStageRepository(self.store).cancel(stage_id, owner_user_id=self._owner_user_id(principal), is_owner=owner)
        except KeyError as exc:
            raise APIError(404, "research_stage_not_found", "Research stage was not found") from exc
        except ValueError as exc:
            raise APIError(409, str(exc), "Research stage cannot be cancelled") from exc
        return ResearchStageResponse(**stage.to_dict(), request_id=request_id)

    def _finding_for_review(self, task_id: str, finding_id: str, principal: AuthenticatedPrincipal | None) -> tuple[TaskState, dict[str, Any], Finding]:
        self._assert_task_access(task_id, principal)
        state = self._state(task_id)
        if state.context.get("task_kind") != "bug_bounty":
            raise APIError(404, "bug_bounty_task_not_found", "Bug Bounty task not found", task_id=task_id)
        output = self._bug_bounty_output(state) or {}
        run = output.get("run") if isinstance(output.get("run"), dict) else {}
        scan = output.get("scan") if isinstance(output.get("scan"), dict) else {}
        candidates = run.get("findings", []) if isinstance(run.get("findings", []), list) else scan.get("candidate_findings", [])
        raw = next((item for item in candidates if isinstance(item, dict) and str(item.get("finding_id", "")) == finding_id), None)
        if raw is None:
            raise APIError(404, "finding_not_found", "Finding was not found", task_id=task_id)
        try:
            finding = Finding.from_dict(raw)
        except (TypeError, ValueError) as exc:
            raise APIError(409, "finding_untrusted", "Finding failed integrity validation", task_id=task_id) from exc
        return state, raw, finding

    def review_bug_bounty_finding(
        self, task_id: str, finding_id: str, payload: FindingReviewRequest,
        *, principal: AuthenticatedPrincipal | None = None, request_id: str = "",
    ) -> FindingReviewResponse:
        task_id = _safe_id(task_id, "task_id")
        finding_id = _safe_id(finding_id, "finding_id")
        if principal is None or principal.role != ROLE_OWNER:
            raise APIError(403, "owner_review_required", "Only the owner may review findings", task_id=task_id)
        _state, raw, finding = self._finding_for_review(task_id, finding_id, principal)
        actor_user_id = principal.user_id or "legacy-owner"
        reviews = FindingReviewRepository(self.store).list_for_task(task_id, owner_user_id=principal.user_id, is_owner=True)
        if payload.decision == FindingReviewDecision.DUPLICATE.value:
            target = payload.duplicate_of.strip()
            if not target or target == finding_id:
                raise APIError(422, "duplicate_target_required", "A different finding is required as the duplicate target", task_id=task_id)
            candidates = (_state.result or {}).get("step_results", []) if isinstance(_state.result, dict) else []
            target_exists = any(
                isinstance(step, dict) and isinstance(step.get("output"), dict)
                and any(isinstance(item, dict) and str(item.get("finding_id", "")) == target for item in (step["output"].get("run", {}) or {}).get("findings", []) if isinstance((step["output"].get("run", {}) or {}).get("findings", []), list))
                for step in candidates
            )
            if not target_exists:
                raise APIError(422, "duplicate_target_not_found", "Duplicate target must be another finding in this task", task_id=task_id)
        evidence_validated = False
        evidence_digest = ""
        if payload.decision == FindingReviewDecision.CONFIRM.value:
            if finding.state not in {FindingState.REPORT_READY, FindingState.HUMAN_REVIEWED}:
                raise APIError(409, "finding_not_report_ready", "Only an Evidence Gate validated finding may be confirmed", task_id=task_id)
            errors = validate_confirmed_evidence(finding, target_state=finding.state)
            evidence_digest = finding.validated_evidence_sha256
            evidence_validated = not errors and finding.validation_status == "VALIDATED" and evidence_digest == finding._evidence_digest()
            if not evidence_validated:
                raise APIError(409, "finding_evidence_not_validated", "Finding evidence has not passed the Evidence Gate", task_id=task_id)
        try:
            review = FindingReview.create(
                task_id, finding_id, actor_user_id, payload.decision,
                duplicate_of=payload.duplicate_of.strip(), note=payload.note,
                evidence_sha256=evidence_digest, evidence_validated=evidence_validated,
                actor_user_id=actor_user_id,
            )
            review = FindingReviewRepository(self.store).save(review, is_owner=True)
        except (TypeError, ValueError) as exc:
            raise APIError(422, str(exc), "Finding review could not be saved", task_id=task_id) from exc
        try:
            learned = BugBountyLearningService(self.store).apply_review(review, finding)
            self.store.save_audit_event({
                "event_id": f"LEARNING-{review.review_id}", "timestamp": review.created_at,
                "event_type": "BUGBOUNTY_LEARNING_UPDATED", "task_id": review.task_id,
                "action_id": review.finding_id, "actor": review.actor_user_id,
                "tool_or_skill": "bugbounty.learning", "capabilities": [],
                "risk_level": "LOW", "policy_version": 0, "approval_id": "",
                "decision": review.decision, "reason": "owner_review_advisory_learning",
                "learning": {"strategy_ids": [item.strategy_id for item in learned], "non_authoritative": True},
            }, requires_lease=False)
        except (TypeError, ValueError, KeyError):
            self.store.save_audit_event({
                "event_id": f"LEARNING-REJECTED-{review.review_id}", "timestamp": review.created_at,
                "event_type": "BUGBOUNTY_LEARNING_REJECTED", "task_id": review.task_id,
                "action_id": review.finding_id, "actor": review.actor_user_id,
                "tool_or_skill": "bugbounty.learning", "capabilities": [],
                "risk_level": "NOTICE", "policy_version": 0, "approval_id": "",
                "decision": review.decision, "reason": "learning_contract_rejected",
                "learning": {"non_authoritative": True},
            }, requires_lease=False)
        return FindingReviewResponse(**review.to_dict(), request_id=request_id)

    def issue_finding_knowledge_attestation(
        self,
        task_id: str,
        review_id: str,
        *,
        principal: AuthenticatedPrincipal | None = None,
        request_id: str = "",
    ) -> FindingKnowledgeAttestationResponse:
        task_id = _safe_id(task_id, "task_id")
        review_id = _safe_id(review_id, "review_id")
        if (
            principal is None
            or principal.role != ROLE_OWNER
            or principal.is_legacy
            or principal.auth_type != "session"
            or not principal.user_id
            or not principal.session_id
        ):
            raise APIError(
                403,
                "authenticated_reviewer_session_required",
                "A verified owner session is required to issue knowledge attestations",
                task_id=task_id,
            )
        active_reviewer = next(
            (user for user in self.auth_manager.list_users() if user.user_id == principal.user_id),
            None,
        )
        if active_reviewer is None or active_reviewer.role != ROLE_OWNER or active_reviewer.status != ACTIVE:
            raise APIError(403, "owner_review_required", "Only the active owner may issue knowledge attestations", task_id=task_id)

        review_repository = FindingReviewRepository(self.store)
        review = review_repository.get_by_id(review_id)
        if review is None or review.task_id != task_id:
            raise APIError(404, "finding_review_not_found", "Finding review was not found", task_id=task_id)
        if review.owner_user_id != principal.user_id or review.actor_user_id != principal.user_id:
            raise APIError(403, "reviewer_identity_mismatch", "The authenticated owner did not create this review", task_id=task_id)
        if review.decision != FindingReviewDecision.CONFIRM.value or not review.evidence_validated:
            raise APIError(409, "review_not_confirmed", "Only a confirmed review with validated evidence can be attested", task_id=task_id)

        _state, _raw, finding = self._finding_for_review(task_id, review.finding_id, principal)
        evidence_digest = finding._evidence_digest()
        evidence_errors = validate_confirmed_evidence(finding, target_state=finding.state)
        if (
            finding.state not in {FindingState.REPORT_READY, FindingState.HUMAN_REVIEWED}
            or evidence_errors
            or finding.validation_status != "VALIDATED"
            or not evidence_digest
            or finding.validated_evidence_sha256 != evidence_digest
            or review.evidence_sha256 != evidence_digest
        ):
            raise APIError(409, "review_evidence_stale", "The reviewed evidence no longer matches the validated finding", task_id=task_id)

        source_id = f"source-review-{review.review_id}"
        claim_id = f"claim-review-{review.review_id}"
        evidence_id = f"evidence-review-{review.review_id}"
        evidence_binding = {
            "review_id": review.review_id,
            "decision": review.decision,
            "evidence_sha256": evidence_digest,
        }
        evidence_excerpt = json.dumps(
            evidence_binding, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        source_document = {
            "review": {
                "review_id": review.review_id,
                "task_id": review.task_id,
                "finding_id": review.finding_id,
                "owner_user_id": review.owner_user_id,
                "actor_user_id": review.actor_user_id,
                "decision": review.decision,
                "evidence_validated": review.evidence_validated,
                "evidence_sha256": evidence_digest,
                "created_at": review.created_at,
            },
            "finding": {
                "vulnerability_class": finding.vulnerability_class,
                "state": finding.state.value,
                "validation_status": finding.validation_status,
                "evidence_count": len(finding.evidence),
            },
            "evidence_binding": evidence_binding,
        }
        source_content = json.dumps(
            source_document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        source = KnowledgeSource(
            source_id=source_id,
            canonical_uri=f"local://bug-bounty/reviews/{review.review_id}",
            source_type="local",
            source_class="local_verified",
            retrieved_at=review.created_at,
            content_sha256=sha256_hex(source_content),
            revision=review.evidence_sha256,
            metadata={
                "review_id": review.review_id,
                "task_id": review.task_id,
                "finding_id": review.finding_id,
                "evidence_sha256": evidence_digest,
            },
            created_at=review.created_at,
            content=source_content,
        )
        claim = KnowledgeClaim(
            claim_id=claim_id,
            domain="cybersecurity.bugbounty",
            subject=f"finding:{finding.finding_id}",
            predicate="owner_confirmed_after_evidence_gate",
            object_text=(
                f"Owner-confirmed {finding.vulnerability_class} finding "
                f"{finding.finding_id} with validated evidence digest {evidence_digest}"
            ),
            knowledge_type="reviewed_security_finding",
            verification_status="verified",
            status="SUPPORTED",
            confidence=finding.confidence,
            observed_at=review.created_at,
            last_verified_at=review.created_at,
            version=1,
            metadata={
                "review_id": review.review_id,
                "finding_id": review.finding_id,
                "evidence_sha256": evidence_digest,
            },
            created_at=review.created_at,
            updated_at=review.created_at,
        )
        evidence = [
            KnowledgeEvidence(
                evidence_id=evidence_id,
                claim_id=claim_id,
                source_id=source_id,
                # Owner reviews are issued after the worker lease ends. Keep
                # task_id bound in the signed source/metadata, but do not claim
                # this control-plane write was made under the worker lease.
                locator="persisted owner review and Evidence Gate digest",
                excerpt=evidence_excerpt,
                verification_status="verified",
                observed_at=review.created_at,
                metadata={
                    "task_id": task_id,
                    "review_id": review.review_id,
                    "evidence_sha256": evidence_digest,
                },
                created_at=review.created_at,
            )
        ]

        def authorize_issuer(
            candidate_source: KnowledgeSource,
            candidate_claim: KnowledgeClaim,
            candidate_evidence: list[KnowledgeEvidence],
        ) -> str | None:
            current_review = review_repository.get_by_id(review_id)
            if (
                current_review != review
                or current_review is None
                or current_review.task_id != task_id
                or current_review.decision != FindingReviewDecision.CONFIRM.value
                or not current_review.evidence_validated
                or current_review.owner_user_id != principal.user_id
                or current_review.actor_user_id != principal.user_id
                or current_review.evidence_sha256 != evidence_digest
                or candidate_source.to_dict() != source.to_dict()
                or candidate_source.content != source_content
                or candidate_claim.to_dict() != claim.to_dict()
                or candidate_evidence != evidence
            ):
                return None
            return principal.user_id

        try:
            saved = self.store.issue_attested_knowledge_bundle(
                source,
                claim,
                evidence,
                issuer_authorizer=authorize_issuer,
                verified_at=review.created_at,
                attestation_id=f"attestation-{review.review_id}",
            )
        except KnowledgeValidationError as exc:
            error_code = str(exc)
            status_code = 503 if error_code == "knowledge_attestation_key_missing" else 409
            raise APIError(
                status_code,
                error_code,
                "Knowledge attestation could not be issued",
                task_id=task_id,
            ) from exc
        return FindingKnowledgeAttestationResponse(
            review_id=review.review_id,
            attestation=saved["attestation"],
            request_id=request_id,
        )

    def list_finding_reviews(
        self, task_id: str, *, principal: AuthenticatedPrincipal | None = None, request_id: str = "",
    ) -> list[FindingReviewResponse]:
        task_id = _safe_id(task_id, "task_id")
        self._assert_task_access(task_id, principal)
        reviews = FindingReviewRepository(self.store).list_for_task(
            task_id, owner_user_id=self._owner_user_id(principal), is_owner=principal is not None and principal.role == ROLE_OWNER,
        )
        return [FindingReviewResponse(**review.to_dict(), request_id=request_id) for review in reviews]

    def list_users_response(self) -> list[AuthUserResponse]:
        return [self.auth_user_response(user) for user in self.auth_manager.list_users()]

    def create_user_response(
        self, email: str, password: str, role: str, *, principal: AuthenticatedPrincipal | None = None
    ) -> AuthUserResponse:
        try:
            actor = "legacy-owner" if principal is None or principal.is_legacy else principal.user_id
            return self.auth_user_response(self.auth_manager.create_user(email, password, role, actor_user_id=actor))
        except AuthError as exc:
            status_code = 409 if exc.code in {"user_exists", "owner_already_initialized"} else 422
            raise APIError(status_code, exc.code, exc.message) from exc

    def set_user_role_response(
        self, user_id: str, role: str, *, principal: AuthenticatedPrincipal | None = None
    ) -> AuthUserResponse:
        try:
            actor = "legacy-owner" if principal is None or principal.is_legacy else principal.user_id
            return self.auth_user_response(self.auth_manager.set_role(user_id, role, actor_user_id=actor))
        except AuthError as exc:
            status_code = 404 if exc.code == "user_not_found" else 409 if exc.code == "owner_immutable" else 422
            raise APIError(status_code, exc.code, exc.message) from exc

    def disable_user_response(self, user_id: str, *, principal: AuthenticatedPrincipal | None = None) -> AuthUserResponse:
        try:
            actor = "legacy-owner" if principal is None or principal.is_legacy else principal.user_id
            return self.auth_user_response(self.auth_manager.disable_user(user_id, actor_user_id=actor))
        except AuthError as exc:
            status_code = 404 if exc.code == "user_not_found" else 409 if exc.code == "owner_immutable" else 422
            raise APIError(status_code, exc.code, exc.message) from exc

    def _state(self, task_id: str) -> TaskState:
        task_id = _safe_id(task_id, "task_id")
        raw = self.store.get_task_state(task_id)
        if raw is None:
            raise APIError(404, "task_not_found", "Task was not found", task_id=task_id)
        try:
            return TaskState.from_dict(raw)
        except (StateIntegrityError, TypeError, ValueError) as exc:
            raise APIError(409, "task_state_untrusted", "Task state failed integrity validation", task_id=task_id) from exc

    def _assert_task_access(self, task_id: str, principal: AuthenticatedPrincipal | None) -> None:
        """Require ownership for non-owner accounts without disclosing task existence."""
        if principal is None or principal.is_legacy or principal.role == ROLE_OWNER:
            return
        owner_user_id = self.store.task_owner_user_id(task_id)
        if owner_user_id != principal.user_id:
            raise APIError(404, "task_not_found", "Task was not found", task_id=task_id)

    @staticmethod
    def _owner_user_id(principal: AuthenticatedPrincipal | None) -> str | None:
        return principal.user_id if principal is not None and not principal.is_legacy else None

    def _approval(self, approval_id: str) -> ApprovalRequest:
        approval_id = _safe_id(approval_id, "approval_id")
        if self.security is None:
            raise APIError(503, "security_not_configured", "Approval operations are unavailable")
        request = self.security.approval_gate.get(approval_id)
        if request is None:
            raise APIError(404, "approval_not_found", "Approval was not found", approval_id=approval_id)
        return request

    def _task_response(self, state: TaskState, *, request_id: str = "", replay: bool = False) -> TaskCreateResponse:
        return TaskCreateResponse(
            task_id=state.task_id,
            conversation_id=state.task_id,
            status=state.status,
            created_at=state.created_at,
            current_phase=state.current_phase,
            request_id=request_id,
            idempotent_replay=replay,
        )

    def _specialist_assignment_snapshot(self, state: TaskState) -> SpecialistAssignmentSnapshotResponse:
        plan_version = state.plan_version if state.plan else 0
        base = {"task_id": state.task_id, "plan_version": plan_version}
        if not state.plan:
            return SpecialistAssignmentSnapshotResponse(**base, status="NONE")

        steps = state.plan.get("steps", [])
        if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
            return SpecialistAssignmentSnapshotResponse(
                **base, status="UNAVAILABLE", reason="assignment_not_verifiable"
            )
        if not steps:
            return SpecialistAssignmentSnapshotResponse(**base, status="NONE")

        candidates = [step for step in steps if "specialist_assignment" in step]
        if not candidates:
            return SpecialistAssignmentSnapshotResponse(**base, status="NONE")

        registry = getattr(self.orchestrator, "specialist_registry", None)
        if registry is None:
            return SpecialistAssignmentSnapshotResponse(
                **base, status="UNAVAILABLE", reason="assignment_not_verifiable"
            )

        assignments: list[SpecialistAssignmentResponse] = []
        public_capabilities = {"READ_WORKSPACE"}
        public_action_statuses = {
            "ASSIGNED", "EXECUTING", "OBSERVING", "VERIFIED", "FAILED",
            "BLOCKED", "EXECUTION_UNKNOWN",
        }
        for step in candidates:
            raw_assignment = step.get("specialist_assignment")
            step_id = step.get("id")
            tool_name = step.get("tool")
            objective = step.get("objective")
            inputs = step.get("input")
            bindings = step.get("input_bindings", {})
            if (
                not isinstance(raw_assignment, dict)
                or not isinstance(step_id, str)
                or not isinstance(tool_name, str)
                or not isinstance(objective, str)
                or not isinstance(inputs, dict)
                or not isinstance(bindings, dict)
            ):
                return SpecialistAssignmentSnapshotResponse(
                    **base, status="UNAVAILABLE", reason="assignment_not_verifiable"
                )
            try:
                tool = self.orchestrator.tools.get(tool_name)
                trusted = registry.validate_assignment(
                    raw_assignment,
                    tool=tool,
                    objective=objective,
                    inputs=inputs,
                    input_bindings=bindings,
                )
            except Exception:
                trusted = False
            capabilities = raw_assignment.get("effective_capabilities")
            if (
                not trusted
                or not isinstance(capabilities, list)
                or not capabilities
                or not set(capabilities).issubset(public_capabilities)
            ):
                return SpecialistAssignmentSnapshotResponse(
                    **base, status="UNAVAILABLE", reason="assignment_not_verifiable"
                )

            action_id = action_fingerprint(
                plan_version=state.plan_version,
                step_id=step_id,
                tool=tool_name,
                inputs=inputs,
                specialist_assignment=raw_assignment,
            )
            record = state.action_records.get(action_id, {})
            raw_status = record.get("status") if isinstance(record, dict) else None
            if raw_status is None or raw_status == "NOT_STARTED":
                assignment_status = "ASSIGNED"
            elif raw_status in public_action_statuses:
                assignment_status = raw_status
            else:
                assignment_status = "UNKNOWN"
            reason = None
            if (
                state.stop_reason is not None
                and state.stop_reason.step_id == step_id
                and (state.stop_reason.action_id is None or state.stop_reason.action_id == action_id)
            ):
                reason = state.stop_reason.code
            assignments.append(
                SpecialistAssignmentResponse(
                    step_id=step_id,
                    specialist_id=raw_assignment["specialist_id"],
                    profile_version=raw_assignment["profile_version"],
                    tool_name=tool_name,
                    capabilities=capabilities,
                    status=assignment_status,
                    reason=reason,
                )
            )

        return SpecialistAssignmentSnapshotResponse(
            **base, status="AVAILABLE", assignments=assignments
        )

    def task_state_response(self, state: TaskState, *, request_id: str = "") -> TaskStateResponse:
        checkpoint = self.store.latest_task_checkpoint(state.task_id) or {}
        queue = self.store.task_queue_record(state.task_id) or {}
        actions: list[ActionSummary] = []
        for action_id, raw in sorted(state.action_records.items()):
            actions.append(
                ActionSummary(
                    action_id=action_id,
                    status=str(raw.get("status", "NOT_STARTED")),
                    step_id=str(raw.get("step_id", "")),
                    tool_name=str(raw.get("tool_name", "")),
                    execution_status=str(raw.get("execution_status", "")),
                    planning_mode=normalize_planning_mode(raw.get("planning_mode", UNKNOWN)),
                    plan_version=normalize_plan_version(raw.get("plan_version", 0)),
                    input_fingerprint=str(raw.get("input_fingerprint", "")),
                    updated_at=str(raw.get("updated_at", "")),
                )
            )
        approvals: list[ApprovalResponse] = []
        if self.security is not None:
            for approval_id in sorted(set(state.approval_ids.values())):
                request = self.security.approval_gate.get(approval_id)
                if request is not None:
                    approvals.append(_approval_response(request))
        return TaskStateResponse(
            task_id=state.task_id,
            goal=str(_public_value(state.goal)),
            task_kind="BUG_BOUNTY" if state.context.get("task_kind") == "bug_bounty" else "GENERAL",
            status=state.status,
            current_phase=state.current_phase,
            current_step=state.current_step,
            plan_version=state.plan_version,
            planning_mode=planning_mode_from_plan(state.plan),
            completed_actions=list(state.completed_actions),
            pending_actions=list(state.pending_actions),
            failed_actions=list(state.failed_actions),
            actions=actions,
            recovery_state=self._capability_gap_recovery_state(state),
            verification_status=_verification_status(state),
            checkpoint=CheckpointResponse(
                last_checkpoint_at=state.last_checkpoint_at,
                last_successful_checkpoint=state.last_successful_checkpoint,
                checkpoint_type=str(checkpoint.get("checkpoint_type", "")),
                sequence=int(checkpoint.get("sequence", 0)),
            ),
            approvals=approvals,
            specialist_assignment_snapshot=self._specialist_assignment_snapshot(state),
            created_at=state.created_at,
            updated_at=state.updated_at,
            queue_status=str(queue.get("queue_status", "")),
            stop_reason=_task_stop_reason_response(state.stop_reason),
            request_id=request_id,
        )

    def create_task(
        self,
        goal: str,
        *,
        idempotency_key: str | None = None,
        request_id: str = "",
        context: dict[str, Any] | None = None,
        request_fingerprint: str | None = None,
        principal: AuthenticatedPrincipal | None = None,
    ) -> TaskCreateResponse:
        goal = goal.strip()
        if not goal:
            raise APIError(422, "goal_required", "Task goal is required")
        fingerprint = request_fingerprint or hashlib.sha256(json.dumps({"goal": goal}, sort_keys=True).encode()).hexdigest()
        key = None
        if idempotency_key is not None:
            if not _IDEMPOTENCY_PATTERN.fullmatch(idempotency_key):
                raise APIError(400, "invalid_idempotency_key", "Invalid Idempotency-Key")
            key = idempotency_key
        with self._lock:
            try:
                created = self.store.create_queued_task(
                    goal,
                    initial_message=goal,
                    idempotency_key=key,
                    request_fingerprint=fingerprint if key else None,
                    context=context,
                    owner_user_id=self._owner_user_id(principal),
                )
            except ValueError as exc:
                if str(exc) == "idempotency_conflict":
                    raise APIError(409, "idempotency_conflict", "Idempotency-Key was already used for another request") from exc
                raise APIError(422, str(exc), "Task could not be queued") from exc
            state = self._state(str(created["task_id"]))
            return self._task_response(state, request_id=request_id, replay=bool(created["replayed"]))

    def list_tasks(
        self,
        *,
        limit: int = 20,
        cursor: str | None = None,
        status_filter: str | None = None,
        request_id: str = "",
        principal: AuthenticatedPrincipal | None = None,
    ) -> TaskListResponse:
        if limit < 1 or limit > 100:
            raise APIError(422, "invalid_limit", "Task list limit must be between 1 and 100")
        if status_filter is not None and status_filter not in TASK_STATUSES:
            raise APIError(422, "invalid_status", "Invalid task status filter")
        cursor_created_at = ""
        cursor_task_id = ""
        if cursor:
            cursor_created_at, cursor_task_id = _decode_task_cursor(cursor)
        rows = self.store.list_task_summaries(
            limit=limit + 1,
            cursor_created_at=cursor_created_at,
            cursor_task_id=cursor_task_id,
            status=status_filter,
            owner_user_id=None if principal is None or principal.is_legacy or principal.role == ROLE_OWNER else principal.user_id,
        )
        has_more = len(rows) > limit
        page = rows[:limit]
        items: list[TaskSummaryResponse] = []
        for row in page:
            try:
                state_payload = json.loads(str(row["state_json"]))
                state = TaskState.from_dict(state_payload)
            except (StateIntegrityError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise APIError(409, "task_state_untrusted", "Task state failed integrity validation", task_id=str(row.get("task_id", ""))) from exc
            items.append(
                TaskSummaryResponse(
                    task_id=state.task_id,
                    goal=str(_public_value(state.goal)),
                    status=state.status,
                    queue_status=str(row["queue_status"]),
                    current_phase=state.current_phase,
                    current_step=state.current_step,
                    plan_version=state.plan_version,
                    verification_status=_verification_status(state),
                    attempt_count=int(row["attempt_count"]),
                    created_at=state.created_at,
                    updated_at=state.updated_at,
                    last_checkpoint_at=state.last_checkpoint_at,
                    last_successful_checkpoint=state.last_successful_checkpoint,
                    available_at=str(row["available_at"]),
                    last_error_code=str(row["last_error_code"]),
                    waiting_for_approval=state.status == "WAITING_APPROVAL",
                    execution_unknown=state.status == "EXECUTION_UNKNOWN",
                    stop_reason=_task_stop_reason_response(state.stop_reason),
                )
            )
        next_cursor = None
        if has_more and page:
            next_cursor = _encode_task_cursor(str(page[-1]["created_at"]), str(page[-1]["task_id"]))
        return TaskListResponse(items=items, next_cursor=next_cursor, request_id=request_id)

    def get_task(self, task_id: str, *, request_id: str = "", principal: AuthenticatedPrincipal | None = None) -> TaskStateResponse:
        self._assert_task_access(task_id, principal)
        return self.task_state_response(self._state(task_id), request_id=request_id)

    def get_task_conversation(
        self,
        task_id: str,
        *,
        request_id: str = "",
        principal: AuthenticatedPrincipal | None = None,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> TaskConversationResponse:
        task_id = _safe_id(task_id, "task_id")
        self._assert_task_access(task_id, principal)
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int) or after_sequence < 0:
            raise APIError(400, "invalid_conversation_cursor", "Invalid conversation cursor", task_id=task_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise APIError(400, "invalid_conversation_limit", "Invalid conversation page size", task_id=task_id)
        state = self._state(task_id)
        queue = self.store.task_queue_record(task_id) or {}
        owner_user_id = (
            principal.user_id
            if principal is not None and not principal.is_legacy and principal.role != ROLE_OWNER
            else None
        )
        page = self.store.task_conversation_items(
            task_id,
            owner_user_id=owner_user_id,
            after_sequence=after_sequence,
            limit=limit,
        )
        messages: list[TaskMessageResponse] = []
        result_events: list[TaskResultEventResponse] = []
        items: list[TaskConversationItemResponse] = []
        for item in page["items"]:
            sequence = int(item["sequence"])
            if item["item_type"] == "USER_MESSAGE":
                source = item["message"]
                message = TaskMessageResponse(
                    message_id=str(source["message_id"]),
                    conversation_id=task_id,
                    task_id=task_id,
                    sequence=sequence,
                    role=str(source["role"]),
                    message_kind=str(source["message_kind"]),
                    content=str(source["content"]),
                    status=str(source["status"]),
                    created_at=str(source["created_at"]),
                    consumed_at=str(source.get("consumed_at", "")),
                    consumed_checkpoint=str(source.get("consumed_checkpoint", "")),
                )
                messages.append(message)
                items.append(TaskConversationItemResponse(
                    item_type="USER_MESSAGE", sequence=sequence, message=message,
                ))
            else:
                source = item["result_event"]
                result_event = TaskResultEventResponse(
                    event_id=str(source["event_id"]),
                    conversation_id=task_id,
                    task_id=task_id,
                    sequence=sequence,
                    role=str(source["role"]),
                    message_kind=str(source["message_kind"]),
                    task_status=str(source["task_status"]),
                    verification_status=str(source["verification_status"]),
                    planning_mode=str(source.get("planning_mode", "UNKNOWN")),
                    plan_version=int(source.get("plan_version", 0)),
                    result_available=bool(source["result_available"]),
                    verified_action_count=int(source["verified_action_count"]),
                    failed_action_count=int(source["failed_action_count"]),
                    blocked_action_count=int(source["blocked_action_count"]),
                    pending_action_count=int(source["pending_action_count"]),
                    approval_pending_count=int(source["approval_pending_count"]),
                    approval_approved_count=int(source["approval_approved_count"]),
                    approval_denied_count=int(source["approval_denied_count"]),
                    approval_expired_count=int(source["approval_expired_count"]),
                    approval_cancelled_count=int(source["approval_cancelled_count"]),
                    result_digest=str(source["result_digest"]),
                    created_at=str(source["created_at"]),
                )
                result_events.append(result_event)
                items.append(TaskConversationItemResponse(
                    item_type="SYSTEM_RESULT", sequence=sequence, result_event=result_event,
                ))
        included_event_ids = {event.event_id for event in result_events}
        assistant_replies = self.store.task_assistant_replies(task_id, owner_user_id=owner_user_id)
        return TaskConversationResponse(
            conversation_id=task_id,
            task_id=task_id,
            task_status=state.status,
            queue_status=str(queue.get("queue_status", "")),
            items=items,
            messages=messages,
            result_events=result_events,
            assistant_replies=[
                TaskAssistantReplyResponse(
                    reply_id=str(item["reply_id"]),
                    conversation_id=task_id,
                    task_id=task_id,
                    source_event_id=str(item["event_id"]),
                    source_sequence=int(item["source_sequence"]),
                    role=str(item["role"]),
                    message_kind=str(item["message_kind"]),
                    reply_code=str(item["reply_code"]),
                    content=str(item["content"]),
                    created_at=str(item["created_at"]),
                )
                for item in assistant_replies
                if str(item["event_id"]) in included_event_ids
            ],
            assistant_reply_supported=self.assistant_reply_supported,
            next_cursor=page["next_cursor"],
            has_more=page["has_more"],
            request_id=request_id,
        )

    def append_task_message(
        self,
        task_id: str,
        content: str,
        *,
        idempotency_key: str | None = None,
        request_id: str = "",
        principal: AuthenticatedPrincipal | None = None,
    ) -> TaskMessageResponse:
        task_id = _safe_id(task_id, "task_id")
        if principal is not None and not principal.is_legacy and principal.role not in {ROLE_OWNER, ROLE_OPERATOR}:
            raise APIError(403, "role_forbidden", "This role cannot add task messages", task_id=task_id)
        self._assert_task_access(task_id, principal)
        key = None
        if idempotency_key is not None:
            if not _IDEMPOTENCY_PATTERN.fullmatch(idempotency_key):
                raise APIError(400, "invalid_idempotency_key", "Invalid Idempotency-Key", task_id=task_id)
            key = idempotency_key
        fingerprint = hashlib.sha256(
            json.dumps({"task_id": task_id, "content": content.strip()}, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        try:
            stored = self.store.append_task_message(
                task_id,
                content,
                owner_user_id=(
                    principal.user_id
                    if principal is not None and not principal.is_legacy and principal.role != ROLE_OWNER
                    else None
                ),
                idempotency_key=key,
                request_fingerprint=fingerprint if key else None,
            )
        except KeyError as exc:
            raise APIError(404, "task_not_found", "Task was not found", task_id=task_id) from exc
        except ValueError as exc:
            code = str(exc)
            if code in {"message_idempotency_conflict"}:
                raise APIError(409, code, "Idempotency-Key was already used for another message", task_id=task_id) from exc
            if code in {"conversation_execution_unknown", "conversation_terminal", "conversation_not_accepting_messages", "conversation_message_limit", "conversation_size_limit"}:
                raise APIError(409, code, "Task cannot accept a follow-up message in its current state", task_id=task_id) from exc
            status_code = 413 if code == "task_message_too_long" else 422
            raise APIError(status_code, code, "Task message could not be stored", task_id=task_id) from exc
        return TaskMessageResponse(
            message_id=str(stored["message_id"]),
            conversation_id=task_id,
            task_id=task_id,
            sequence=int(stored["sequence"]),
            role=str(stored["role"]),
            message_kind=str(stored["message_kind"]),
            content=str(stored["content"]),
            status=str(stored["status"]),
            created_at=str(stored["created_at"]),
            consumed_at=str(stored.get("consumed_at", "")),
            consumed_checkpoint=str(stored.get("consumed_checkpoint", "")),
            idempotent_replay=bool(stored.get("replayed", False)),
            request_id=request_id,
        )

    @staticmethod
    def _candidate_lifecycle_state(record: dict[str, Any]) -> dict[str, Any]:
        contract = record.get("contract") if isinstance(record.get("contract"), dict) else {}
        validation = contract.get("validation") if isinstance(contract.get("validation"), dict) else {}
        analysis = record.get("analysis") if isinstance(record.get("analysis"), dict) else {}
        mutation = record.get("mutation") if isinstance(record.get("mutation"), dict) else {}
        runtime = record.get("runtime") if isinstance(record.get("runtime"), dict) else {}
        sandbox = runtime.get("sandbox") if isinstance(runtime.get("sandbox"), dict) else {}
        runtime_verification = runtime.get("verification") if isinstance(runtime.get("verification"), dict) else {}
        cross = record.get("cross_verification") if isinstance(record.get("cross_verification"), dict) else {}
        approval = record.get("approval") if isinstance(record.get("approval"), dict) else {}
        checks = approval.get("required_checks") if isinstance(approval.get("required_checks"), dict) else {}
        candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else {}
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        assessment = metadata.get("capability_assessment") if isinstance(metadata.get("capability_assessment"), dict) else {}

        def result_gate(value: Any) -> str:
            return "PASSED" if value is True else "FAILED" if value is False else "NOT_RUN"

        assessment_passed = (
            assessment.get("mode") == "bounded_static_only"
            and record.get("status") in {"CANDIDATE", "APPROVED", "ACTIVE"}
            and validation.get("passed") is True
            and analysis.get("passed") is True
            and checks.get("candidate_assessment") is True
        )
        assessment_status = (
            "NOT_RUN" if not assessment
            else "PASSED" if assessment_passed
            else "FAILED" if checks.get("candidate_assessment") is False or record.get("status") == "REJECTED"
            else "NOT_REVIEWED"
        )
        is_capability_assessment = assessment.get("mode") == "bounded_static_only"
        registered = str(record.get("status", "")) == "ACTIVE"
        gate_statuses = {
            "candidate_assessment": assessment_status,
            "contract_validation": result_gate(validation.get("passed")),
            "static_analysis": result_gate(analysis.get("passed")),
            "independent_review": "PASSED" if checks.get("independent_review") is True else "NOT_REVIEWED",
            "isolation_validation": "PASSED" if checks.get("isolation_validation") is True else "BLOCKED" if is_capability_assessment else "NOT_RUN",
            "owner_approval": "PASSED" if checks.get("owner_approval") is True else "NOT_GRANTED",
            "mutation_threshold": result_gate(mutation.get("passed")),
            "sandbox_execution": "PASSED" if sandbox.get("status") == "PASSED" else "BLOCKED",
            "runtime_verification": result_gate(runtime_verification.get("passed")) if runtime_verification else "BLOCKED",
            "cross_verification": result_gate(cross.get("passed")),
            "no_blocking_findings": "PASSED" if checks.get("no_blocking_findings") is True else "NOT_REVIEWED",
            "security_authorization": "PASSED" if checks.get("security_authorization") is True else "NOT_GRANTED",
        }
        satisfied = {"PASSED"}
        blocking_gates = [name for name, gate_status in gate_statuses.items() if gate_status not in satisfied]
        status = str(record.get("status", ""))
        required_for_promotion = {
            "candidate_assessment", "contract_validation", "static_analysis", "independent_review",
            "isolation_validation", "owner_approval",
            "mutation_threshold", "sandbox_execution", "runtime_verification", "cross_verification",
            "no_blocking_findings", "security_authorization",
        }
        can_promote = (
            status == "APPROVED"
            and approval.get("status") == "APPROVED"
            and approval.get("approved") is True
            and all(gate_statuses[name] == "PASSED" for name in required_for_promotion)
        )
        promotion_status = (
            "PROMOTED" if registered
            else "REJECTED" if status == "REJECTED"
            else "READY" if can_promote
            else "BLOCKED"
        )
        return {
            "promotion_status": promotion_status,
            "can_promote": can_promote,
            "blocking_gates": blocking_gates,
            "gate_statuses": gate_statuses,
        }

    @staticmethod
    def _aggregate_capability_candidate_lifecycle(summaries: list[dict[str, Any]]) -> dict[str, Any]:
        active_count = sum(item.get("status") == "ACTIVE" for item in summaries)
        ready_count = sum(item.get("can_promote") is True for item in summaries)
        if summaries and active_count == len(summaries):
            promotion_status = "PROMOTED"
        elif active_count:
            promotion_status = "PARTIALLY_PROMOTED"
        elif ready_count == len(summaries) and ready_count:
            promotion_status = "READY"
        elif ready_count:
            promotion_status = "PARTIALLY_READY"
        elif summaries and all(item.get("status") == "REJECTED" for item in summaries):
            promotion_status = "REJECTED"
        else:
            promotion_status = "BLOCKED"

        isolation_states = [
            (item.get("gate_statuses") or {}).get("isolation_validation", "NOT_RUN")
            for item in summaries
        ]
        execution_states = [
            (item.get("gate_statuses") or {}).get("sandbox_execution", "NOT_RUN")
            for item in summaries
        ]
        if isolation_states and all(
            isolation == "PASSED" and execution == "PASSED"
            for isolation, execution in zip(isolation_states, execution_states)
        ):
            isolation_status = "PASSED"
            isolation_reason = ""
        elif "FAILED" in isolation_states or "FAILED" in execution_states:
            isolation_status = "FAILED"
            isolation_reason = (
                "isolation_validation_failed_and_runtime_unavailable"
                if "BLOCKED" in isolation_states or "BLOCKED" in execution_states
                else "isolation_validation_failed"
            )
        elif "BLOCKED" in isolation_states or "BLOCKED" in execution_states or not summaries:
            isolation_status = "UNAVAILABLE"
            isolation_reason = "external_sandbox_unavailable"
        else:
            isolation_status = "NOT_VALIDATED"
            isolation_reason = "isolation_validation_not_run"

        blocking_gates = sorted({
            gate
            for item in summaries
            if item.get("status") != "ACTIVE" and item.get("can_promote") is not True
            for gate in item.get("blocking_gates", [])
        })
        outstanding = [item for item in summaries if item.get("status") != "ACTIVE"]
        return {
            "promotion_status": promotion_status,
            "can_promote": bool(outstanding) and all(item.get("can_promote") is True for item in outstanding),
            "isolation_status": isolation_status,
            "isolation_reason": isolation_reason,
            "execution_attempted": any(item.get("execution_attempted", False) for item in summaries),
            "registration_attempted": any(
                item.get("registration_attempted", False) or item.get("status") == "ACTIVE"
                for item in summaries
            ),
            "permissions_granted": any(item.get("permissions_granted", False) for item in summaries),
            "blocking_gates": blocking_gates,
            "candidates": summaries,
        }

    def _capability_gap_recovery_state(self, state: TaskState) -> dict[str, Any]:
        recovery = _public_recovery_state(state.recovery_state)
        gap = recovery.get("capability_gap")
        if not isinstance(gap, dict):
            return recovery
        requested = gap.get("requested_tools") if isinstance(gap.get("requested_tools"), list) else []
        target_tools = [name for name in requested if isinstance(name, str)]
        records = self.skill_registry.candidates_for_capability_gap(state.task_id, target_tools)
        latest_by_tool: dict[str, dict[str, Any]] = {}
        for record in records:
            candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else {}
            name = str(candidate.get("name", ""))
            previous = latest_by_tool.get(name)
            if previous is None or int(candidate.get("version", 0)) > int(previous.get("candidate", {}).get("version", 0)):
                latest_by_tool[name] = record

        summaries: list[dict[str, Any]] = []
        for name in sorted(set(target_tools)):
            record = latest_by_tool.get(name)
            if record is None:
                summaries.append({
                    "target_tool": name,
                    "candidate_id": "",
                    "status": "NOT_PROPOSED",
                    "review_status": "NOT_STARTED",
                    "promotion_status": "BLOCKED",
                    "can_promote": False,
                    "blocking_gates": [
                        "candidate_assessment", "contract_validation", "static_analysis", "independent_review",
                        "isolation_validation", "owner_approval", "mutation_threshold", "sandbox_execution",
                        "runtime_verification", "cross_verification", "no_blocking_findings", "security_authorization",
                    ],
                    "gate_statuses": {
                        "candidate_assessment": "NOT_RUN",
                        "contract_validation": "NOT_RUN",
                        "static_analysis": "NOT_RUN",
                        "independent_review": "NOT_REVIEWED",
                        "isolation_validation": "BLOCKED",
                        "owner_approval": "NOT_GRANTED",
                        "mutation_threshold": "NOT_RUN",
                        "sandbox_execution": "BLOCKED",
                        "runtime_verification": "BLOCKED",
                        "cross_verification": "NOT_RUN",
                        "no_blocking_findings": "NOT_REVIEWED",
                        "security_authorization": "NOT_GRANTED",
                    },
                    "execution_attempted": False,
                    "registration_attempted": False,
                    "permissions_granted": False,
                })
                continue
            candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else {}
            metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
            assessment = metadata.get("capability_assessment") if isinstance(metadata.get("capability_assessment"), dict) else {}
            approval = record.get("approval") if isinstance(record.get("approval"), dict) else {}
            summaries.append({
                "target_tool": name,
                "candidate_id": str(record.get("skill_id", "")),
                "version": int(record.get("version", 0)),
                "status": str(record.get("status", "")),
                "review_status": str(approval.get("status", assessment.get("review_status", "NOT_REVIEWED"))),
                **self._candidate_lifecycle_state(record),
                "execution_attempted": bool(assessment.get("execution_attempted", False))
                or bool((record.get("runtime") or {}).get("sandbox")),
                "registration_attempted": bool(assessment.get("registration_attempted", False))
                or str(record.get("status", "")) == "ACTIVE",
                "permissions_granted": bool(assessment.get("permissions_granted", False)),
            })
        gap["candidate_lifecycle"] = self._aggregate_capability_candidate_lifecycle(summaries)
        return recovery

    @staticmethod
    def _capability_candidate_response(
        record: dict[str, Any], *, request_id: str = ""
    ) -> CapabilityCandidateAssessmentResponse:
        candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else {}
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        assessment = metadata.get("capability_assessment") if isinstance(metadata.get("capability_assessment"), dict) else {}
        if assessment.get("mode") != "bounded_static_only":
            raise APIError(404, "capability_candidate_not_found", "Capability candidate assessment was not found")
        return CapabilityCandidateAssessmentResponse(
            candidate_id=str(record.get("skill_id", "")),
            capability_gap_task_id=str(assessment.get("capability_gap_task_id", "")),
            target_tool=str(assessment.get("target_tool", "")),
            version=int(record.get("version", 0)),
            status=str(record.get("status", "")),
            review_status=str(record.get("approval", {}).get("status", assessment.get("review_status", "NOT_REVIEWED")))
            if isinstance(record.get("approval"), dict) else str(assessment.get("review_status", "NOT_REVIEWED")),
            **AgentAPIService._candidate_lifecycle_state(record),
            source_sha256=str(assessment.get("source_sha256", "")),
            contract=record.get("contract") if isinstance(record.get("contract"), dict) else {},
            analysis=record.get("analysis") if isinstance(record.get("analysis"), dict) else {},
            execution_attempted=bool(assessment.get("execution_attempted", False)),
            installation_attempted=bool(assessment.get("installation_attempted", False)),
            registration_attempted=bool(assessment.get("registration_attempted", False)),
            permissions_granted=bool(assessment.get("permissions_granted", False)),
            created_at=str(candidate.get("created_at", "")),
            request_id=request_id,
        )

    def assess_capability_candidate(
        self,
        task_id: str,
        payload: CapabilityCandidateAssessmentRequest,
        *,
        request_id: str = "",
        principal: AuthenticatedPrincipal | None = None,
    ) -> CapabilityCandidateAssessmentResponse:
        if principal is None or principal.role != ROLE_OWNER:
            raise APIError(403, "forbidden", "Only the owner can assess capability candidates")
        self._assert_task_access(task_id, principal)
        candidate = SkillCandidate(
            skill_id=str(uuid.uuid4()),
            name=payload.target_tool,
            version=payload.version,
            description=payload.description,
            source=payload.source,
            entry_point=payload.entry_point,
            declared_capabilities=list(payload.declared_capabilities),
            allowed_imports=list(payload.allowed_imports),
            allowed_network_domains=list(payload.allowed_network_domains),
            expected_inputs=dict(payload.expected_inputs),
            expected_outputs=dict(payload.expected_outputs),
            metadata={},
            provenance={},
            status="CANDIDATE",
        )
        with self._lock:
            try:
                record = self.capability_candidate_assessor.assess_for_gap(
                    candidate,
                    gap_task_id=task_id,
                    target_tool=payload.target_tool,
                )
            except CapabilityCandidateAssessmentError as exc:
                if exc.code == "capability_gap_task_not_found":
                    code, status_code = exc.code, 404
                elif exc.code in {
                    "candidate_version_conflict",
                    "candidate_persistence_conflict",
                    "candidate_not_bound_to_capability_gap",
                    "capability_gap_resolved",
                }:
                    code, status_code = exc.code, 409
                else:
                    code, status_code = exc.code, 422
                raise APIError(status_code, code, "Capability candidate assessment could not be completed") from exc
        response = self._capability_candidate_response(record, request_id=request_id)
        return response

    def get_capability_candidate_assessment(
        self,
        candidate_id: str,
        *,
        request_id: str = "",
        principal: AuthenticatedPrincipal | None = None,
    ) -> CapabilityCandidateAssessmentResponse:
        if principal is None or principal.role != ROLE_OWNER:
            raise APIError(403, "forbidden", "Only the owner can inspect capability candidates")
        record = self.skill_registry.get_candidate(candidate_id)
        if record is None:
            raise APIError(404, "capability_candidate_not_found", "Capability candidate assessment was not found")
        return self._capability_candidate_response(record, request_id=request_id)

    @staticmethod
    def _bug_bounty_output(state: TaskState) -> dict[str, Any] | None:
        result = state.result if isinstance(state.result, dict) else {}
        for step_result in result.get("step_results", []):
            if not isinstance(step_result, dict):
                continue
            output = step_result.get("output")
            if isinstance(output, dict) and (isinstance(output.get("run"), dict) or isinstance(output.get("scan"), dict)):
                return output
        return None

    @staticmethod
    def _sanitize_bug_bounty_scan(scan: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(scan)
        raw_findings = scan.get("candidate_findings", [])
        findings: list[dict[str, Any]] = []
        if isinstance(raw_findings, list):
            for raw in raw_findings[:100]:
                if not isinstance(raw, dict):
                    continue
                try:
                    findings.append(Finding.from_dict(raw).to_dict(public=True))
                except (TypeError, ValueError):
                    fallback = dict(raw)
                    fallback["state"] = FindingState.HYPOTHESIS.value
                    fallback["validation"] = {
                        "status": "REJECTED",
                        "errors": ["scanner_finding_validation_failed"],
                        "evidence_sha256": "",
                        "validator": "not_validated",
                    }
                    findings.append(fallback)
        sanitized["candidate_findings"] = findings
        sanitized["observations"] = safe_json(scan.get("observations", []))[:500] if isinstance(scan.get("observations", []), list) else []
        sanitized["errors"] = [str(item)[:256] for item in scan.get("errors", [])[:50]] if isinstance(scan.get("errors", []), list) else ["scan_errors_malformed"]
        sanitized["limitations"] = [str(item)[:512] for item in scan.get("limitations", [])[:50]] if isinstance(scan.get("limitations", []), list) else ["scan_limitations_malformed"]
        return sanitized

    @staticmethod
    def _sanitize_bug_bounty_run(run: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Revalidate persisted findings before exposing run/report aggregates."""
        sanitized = dict(run)
        raw_findings = run.get("findings", [])
        findings: list[dict[str, Any]] = []
        downgraded = not isinstance(raw_findings, list) and bool(run.get("report"))
        for raw in raw_findings if isinstance(raw_findings, list) else []:
            if not isinstance(raw, dict):
                downgraded = True
                continue
            try:
                finding = Finding.from_dict(raw)
                rendered = finding.to_dict(public=True)
                if raw.get("state") in {state.value for state in (FindingState.CONFIRMED, FindingState.REPORT_READY, FindingState.HUMAN_REVIEWED)} and rendered.get("state") != raw.get("state"):
                    downgraded = True
                findings.append(rendered)
            except (TypeError, ValueError):
                downgraded = True
                fallback = dict(raw)
                fallback["state"] = FindingState.HYPOTHESIS.value
                fallback["validation"] = {
                    "status": "REJECTED",
                    "errors": ["persisted_finding_validation_failed"],
                    "evidence_sha256": "",
                    "validator": "idor_bola_semantic_v1",
                }
                findings.append(fallback)
        tests = run.get("tests", []) if isinstance(run.get("tests", []), list) else []
        raw_stage_refs = run.get("research_stage_refs", [])
        try:
            sanitized["research_stage_refs"] = normalize_execution_stage_refs(raw_stage_refs)
        except ValueError:
            sanitized["research_stage_refs"] = []
            downgraded = True
        sanitized["findings"] = findings
        sanitized["metrics"] = reconcile_bug_bounty_metrics(run.get("metrics") if isinstance(run.get("metrics"), dict) else {}, findings, [item for item in tests if isinstance(item, dict)])
        limitations = [str(item) for item in run.get("limitations", [])] if isinstance(run.get("limitations", []), list) else []
        if downgraded and "report_suppressed_after_finding_validation" not in limitations:
            limitations.append("report_suppressed_after_finding_validation")
        sanitized["limitations"] = limitations[:50]
        return sanitized, downgraded

    def create_bug_bounty_task(
        self,
        payload: BugBountyCreateRequest,
        *,
        idempotency_key: str | None = None,
        request_id: str = "",
        principal: AuthenticatedPrincipal | None = None,
        program_analysis_provenance: Mapping[str, Any] | None = None,
    ) -> TaskCreateResponse:
        execution_scope = dict(payload.scope)
        execution_scope["scope_reference"] = public_reference(execution_scope.pop("authorization_reference"))
        request = {
            "scope": execution_scope,
            "scanner": payload.scanner,
            "target": payload.target or "",
            "identity": payload.identity,
            "resource_id": payload.resource_id,
            "fixture": payload.fixture or "",
            "urls": list(payload.urls),
            "openapi": payload.openapi or {},
            "source_documents": dict(payload.source_documents),
        }
        goal = f"Bug Bounty analysis for authorized program {payload.scope['program_id']}"
        context: dict[str, Any] = {"task_kind": "bug_bounty", "bugbounty_request": request}
        fingerprint_payload: dict[str, Any] = {"goal": goal, "request": request}
        if program_analysis_provenance is not None:
            provenance = safe_json(dict(program_analysis_provenance))
            context["program_analysis_provenance"] = provenance
            fingerprint_payload["program_analysis_provenance"] = {
                key: value for key, value in provenance.items() if key != "scope_attested_at"
            }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return self.create_task(
            goal,
            idempotency_key=idempotency_key,
            request_id=request_id,
            context=context,
            request_fingerprint=fingerprint,
            principal=principal,
        )

    def create_bug_bounty_task_from_analysis(
        self,
        analysis_id: str,
        payload: ProgramAnalysisTaskRequest,
        *,
        idempotency_key: str | None = None,
        request_id: str = "",
        principal: AuthenticatedPrincipal | None = None,
    ) -> TaskCreateResponse:
        analysis_id = _safe_id(analysis_id, "analysis_id")
        if principal is None or principal.is_legacy or principal.role != ROLE_OWNER:
            raise APIError(403, "analysis_handoff_forbidden", "An authenticated owner is required to review and attest a program-analysis proposal")
        if not payload.authorization_attested:
            raise APIError(422, "authorization_attestation_required", "Explicit written-authorization attestation is required")

        is_owner = principal.role == ROLE_OWNER
        analysis = ProgramAnalysisRepository(self.store).get(
            analysis_id,
            owner_user_id=self._owner_user_id(principal),
            is_owner=is_owner,
        )
        if analysis is None:
            raise APIError(404, "program_analysis_not_found", "Program analysis was not found")
        proposal = analysis.get("scope_proposal")
        if analysis.get("status") != "READY_FOR_SCOPE_REVIEW" or not isinstance(proposal, Mapping):
            raise APIError(409, "program_analysis_not_eligible", "Only a current server-returned scope proposal can be used")

        source = analysis.get("source")
        observations = analysis.get("observations")
        if (
            not isinstance(source, Mapping)
            or source.get("source_type") == "hackerone_program"
            or isinstance(observations, Mapping) and observations.get("authorization_derived") is False
        ):
            raise APIError(409, "public_policy_is_not_authorization", "Public-policy observations cannot authorize a Bug Bounty task")
        source_digest = source.get("content_sha256") if isinstance(source, Mapping) else None
        if not isinstance(source_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", source_digest):
            raise APIError(409, "program_analysis_provenance_invalid", "The stored analysis source provenance is invalid")

        expected_proposal_digest = scope_proposal_digest(proposal)
        if not expected_proposal_digest or payload.proposal_digest != expected_proposal_digest:
            raise APIError(409, "program_analysis_stale", "The reviewed proposal changed; reload and review the latest server result")
        stored_reference = proposal.get("authorization_reference")
        if not isinstance(stored_reference, str) or not stored_reference.startswith("sha256:"):
            raise APIError(409, "program_analysis_provenance_invalid", "The stored proposal does not contain a redacted authorization reference")

        scope_payload = dict(proposal)
        scope_payload["authorization_reference"] = payload.authorization_reference
        try:
            validated_scope = ScopeDefinition.from_dict(scope_payload)
            task_payload = BugBountyCreateRequest(scope=validated_scope.to_dict(public=False), scanner="deterministic")
        except (TypeError, ValueError) as exc:
            raise APIError(409, "scope_proposal_no_longer_valid", "The server proposal is no longer valid for task creation") from exc

        try:
            research_stage_refs = ResearchStageRepository(self.store).resolve_execution_sources(
                analysis_id,
                owner_user_id=self._owner_user_id(principal),
                stage_ids=payload.source_stage_ids,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise APIError(
                409,
                "program_analysis_research_stage_invalid",
                "Selected research stages must be completed, owned by this analysis, and have valid persisted results",
            ) from exc

        provenance = {
            "analysis_id": analysis_id,
            "analysis_status": "READY_FOR_SCOPE_REVIEW",
            "source_content_sha256": source_digest,
            "proposal_digest": expected_proposal_digest,
            "scope_reference_digest": public_reference(payload.authorization_reference),
            "scope_attestation": True,
            "scope_attested_by_user_id": principal.user_id,
            "scope_attested_at": datetime.now(timezone.utc).isoformat(),
            "execution_mode": "deterministic_local",
            "research_stage_refs": research_stage_refs,
        }
        return self.create_bug_bounty_task(
            task_payload,
            idempotency_key=idempotency_key,
            request_id=request_id,
            principal=principal,
            program_analysis_provenance=provenance,
        )

    def bug_bounty_task(self, task_id: str, *, request_id: str = "", principal: AuthenticatedPrincipal | None = None) -> BugBountyTaskResponse:
        self._assert_task_access(task_id, principal)
        state = self._state(task_id)
        if state.context.get("task_kind") != "bug_bounty":
            raise APIError(404, "bug_bounty_task_not_found", "Bug Bounty task not found", task_id=task_id)
        output = self._bug_bounty_output(state) or {}
        run = output.get("run") if isinstance(output.get("run"), dict) else None
        safe_run, report_suppressed = self._sanitize_bug_bounty_run(run) if run is not None else (None, False)
        if safe_run is not None:
            safe_run["finding_reviews"] = [
                review.to_dict() for review in FindingReviewRepository(self.store).list_for_task(
                    task_id, owner_user_id=self._owner_user_id(principal), is_owner=principal is not None and principal.role == ROLE_OWNER,
                )
            ]
        scan = output.get("scan") if isinstance(output.get("scan"), dict) else None
        safe_scan = self._sanitize_bug_bounty_scan(scan) if scan is not None else None
        raw_provenance = state.context.get("program_analysis_provenance")
        analysis_provenance = None
        if isinstance(raw_provenance, Mapping):
            expected_keys = (
                "analysis_id", "analysis_status", "source_content_sha256", "proposal_digest",
                "scope_reference_digest", "scope_attested_by_user_id", "scope_attested_at", "execution_mode",
            )
            analysis_provenance = {key: str(raw_provenance.get(key, ""))[:256] for key in expected_keys}
            analysis_provenance["scope_attestation"] = raw_provenance.get("scope_attestation") is True
            try:
                analysis_provenance["research_stage_refs"] = normalize_execution_stage_refs(
                    raw_provenance.get("research_stage_refs", [])
                )
            except ValueError as exc:
                raise APIError(409, "program_analysis_provenance_invalid", "The stored research-stage lineage is invalid", task_id=task_id) from exc
        limitations = list(safe_run.get("limitations", [])) if safe_run else []
        if safe_scan:
            limitations.extend(str(item) for item in safe_scan.get("limitations", [])[:50])
        return BugBountyTaskResponse(
            task_id=state.task_id,
            status=state.status,
            verification_status=_verification_status(state),
            recovery_state=self._capability_gap_recovery_state(state),
            run=safe_run,
            scan=safe_scan,
            analysis_provenance=analysis_provenance,
            report="" if report_suppressed else str(output.get("report", "")),
            limitations=[str(item) for item in limitations[:50]],
            request_id=request_id,
        )

    def bug_bounty_section(self, task_id: str, section: str, *, request_id: str = "", principal: AuthenticatedPrincipal | None = None) -> BugBountySectionResponse:
        response = self.bug_bounty_task(task_id, request_id=request_id, principal=principal)
        data: Any = None
        if response.run is not None:
            if section == "inventory":
                data = response.run.get("inventory", {})
            elif section == "findings":
                data = response.run.get("findings", [])
            elif section == "evidence":
                findings = response.run.get("findings", [])
                tests = response.run.get("tests", [])
                data = {
                    "finding_evidence": [item.get("evidence", []) for item in findings if isinstance(item, dict)],
                    "test_evidence": [item.get("evidence") for item in tests if isinstance(item, dict) and item.get("evidence")],
                }
            elif section == "report":
                data = {"report": response.report, "run_id": response.run.get("run_id", "")}
        elif response.scan is not None:
            if section == "inventory":
                data = {"observations": response.scan.get("observations", []), "execution": response.scan.get("execution", {})}
            elif section == "findings":
                data = response.scan.get("candidate_findings", [])
            elif section == "evidence":
                data = {"candidate_finding_evidence": [item.get("evidence", []) for item in response.scan.get("candidate_findings", []) if isinstance(item, dict)]}
            elif section == "report":
                data = {"scan": response.scan}
        return BugBountySectionResponse(task_id=response.task_id, status=response.status, data=data, request_id=request_id)

    def resume_task(
        self,
        task_id: str,
        approval_ids: dict[str, str] | None = None,
        *,
        request_id: str = "",
        principal: AuthenticatedPrincipal | None = None,
    ) -> TaskStateResponse:
        task_id = _safe_id(task_id, "task_id")
        self._assert_task_access(task_id, principal)
        with self._lock:
            state = self._state(task_id)
            if state.status == "QUEUED":
                raise APIError(409, "task_queued", "Task is queued and has not started yet", task_id=task_id)
            queue = self.store.task_queue_record(task_id)
            if queue is not None:
                queue_status = str(queue.get("queue_status", ""))
                if queue_status == "WAITING_APPROVAL" and state.status == "WAITING_APPROVAL":
                    try:
                        self.store.requeue_waiting_approval_task(task_id)
                    except ValueError as exc:
                        code = str(exc)
                        if code == "approval_not_approved":
                            raise APIError(409, "approval_not_approved", "Task approvals are not all approved", task_id=task_id) from exc
                        if code == "approval_policy_stale":
                            raise APIError(409, "approval_stale", "A task approval is no longer valid under the current security policy", task_id=task_id) from exc
                        if code == "approval_policy_unavailable":
                            raise APIError(503, "approval_policy_unavailable", "The current security policy could not be checked", task_id=task_id) from exc
                        raise APIError(409, "task_not_requeueable", "Task is not safely requeueable", task_id=task_id) from exc
                    return self.task_state_response(self._state(task_id), request_id=request_id)
                raise APIError(
                    409,
                    "task_queue_managed",
                    "Queue-managed tasks must be resumed by the background worker",
                    task_id=task_id,
                )
            self.orchestrator.resume_task(task_id, approval_ids=approval_ids or {})
            state = self._state(task_id)
            return self.task_state_response(state, request_id=request_id)

    def list_approvals(
        self, *, pending_only: bool = True, principal: AuthenticatedPrincipal | None = None
    ) -> list[ApprovalResponse]:
        if self.security is None:
            raise APIError(503, "security_not_configured", "Approval operations are unavailable")
        raw_requests = self.store.all_approval_requests()
        responses: list[ApprovalResponse] = []
        for raw in raw_requests:
            task_id = str(raw.get("task_id", ""))
            if (
                principal is not None
                and not principal.is_legacy
                and principal.role != ROLE_OWNER
                and self.store.task_owner_user_id(task_id) != principal.user_id
            ):
                continue
            approval_id = str(raw.get("approval_id", ""))
            current = self.security.approval_gate.get(approval_id)
            item = _approval_response(current or raw)
            if pending_only and item.status != "PENDING":
                continue
            responses.append(item)
        return responses

    def get_approval_response(
        self, approval_id: str, *, principal: AuthenticatedPrincipal | None = None
    ) -> ApprovalResponse:
        request = self._approval(approval_id)
        self._assert_task_access(request.task_id, principal)
        return _approval_response(request)

    def approve(self, approval_id: str, *, principal: AuthenticatedPrincipal | None = None) -> ApprovalResponse:
        with self._lock:
            request = self._approval(approval_id)
            self._assert_task_access(request.task_id, principal)
            try:
                actor = "api-owner" if principal is None or principal.is_legacy else principal.user_id
                approved = self.security.approval_gate.approve(request.approval_id, actor=actor)  # type: ignore[union-attr]
            except ValueError as exc:
                error_code = str(exc).split(":", 1)[0]
                if error_code == "approval_policy_stale":
                    raise APIError(
                        409,
                        "approval_stale",
                        "Approval was invalidated by a newer security policy",
                        approval_id=request.approval_id,
                        task_id=request.task_id,
                    ) from exc
                if error_code == "approval_policy_unavailable":
                    raise APIError(
                        503,
                        "approval_policy_unavailable",
                        "The current security policy could not be checked",
                        approval_id=request.approval_id,
                        task_id=request.task_id,
                    ) from exc
                raise APIError(409, "approval_not_pending", "Approval is no longer pending", approval_id=request.approval_id, task_id=request.task_id) from exc
            if self.auto_requeue_approvals:
                try:
                    self.store.requeue_waiting_approval_task(approved.task_id)
                except QueueLeaseLost:
                    # A worker or recovery loop won the queue transition. The
                    # approval remains authoritative and the task is still
                    # bound to the current execution checks.
                    pass
                except ValueError as exc:
                    error_code = str(exc).split(":", 1)[0]
                    if error_code == "approval_policy_stale":
                        raise APIError(
                            409,
                            "approval_stale",
                            "Approval was recorded but became stale before the task could be requeued",
                            approval_id=approved.approval_id,
                            task_id=approved.task_id,
                        ) from exc
                    if error_code == "approval_policy_unavailable":
                        raise APIError(
                            503,
                            "approval_policy_unavailable",
                            "Approval was recorded but the current policy could not be checked for requeue",
                            approval_id=approved.approval_id,
                            task_id=approved.task_id,
                        ) from exc
                    if str(exc) not in {"approval_not_approved", "task_not_waiting_approval"}:
                        raise APIError(
                            409,
                            "approval_requeue_failed",
                            "Approval was recorded but the task could not be requeued safely",
                            approval_id=approved.approval_id,
                            task_id=approved.task_id,
                        ) from exc
            return _approval_response(approved)

    def deny(self, approval_id: str, *, principal: AuthenticatedPrincipal | None = None) -> ApprovalResponse:
        with self._lock:
            request = self._approval(approval_id)
            self._assert_task_access(request.task_id, principal)
            try:
                actor = "api-owner" if principal is None or principal.is_legacy else principal.user_id
                return _approval_response(self.security.approval_gate.deny(request.approval_id, actor=actor))  # type: ignore[union-attr]
            except ValueError as exc:
                raise APIError(409, "approval_not_pending", "Approval is no longer pending", approval_id=request.approval_id, task_id=request.task_id) from exc

    def task_timeline(self, task_id: str, *, principal: AuthenticatedPrincipal | None = None) -> list[EventResponse]:
        self._assert_task_access(task_id, principal)
        self._state(task_id)
        return [_event_response(item) for item in self.observability.get_task_timeline(task_id)]

    def task_metrics(self, task_id: str, *, principal: AuthenticatedPrincipal | None = None) -> MetricsResponse:
        self._assert_task_access(task_id, principal)
        self._state(task_id)
        return MetricsResponse(**self.observability.get_task_metrics(task_id))

    def system_metrics(self) -> MetricsResponse:
        return MetricsResponse(**self.observability.get_system_metrics())

    def failures(self, task_id: str | None = None, *, principal: AuthenticatedPrincipal | None = None) -> list[EventResponse]:
        if task_id is not None:
            self._assert_task_access(task_id, principal)
            self._state(task_id)
        elif principal is not None and not principal.is_legacy and principal.role != ROLE_OWNER:
            raise APIError(403, "forbidden", "Global event feeds require OWNER role")
        return [_event_response(item) for item in self.observability.get_failures(task_id)]

    def security_events(self, task_id: str | None = None, *, principal: AuthenticatedPrincipal | None = None) -> list[EventResponse]:
        if task_id is not None:
            self._assert_task_access(task_id, principal)
            self._state(task_id)
        elif principal is not None and not principal.is_legacy and principal.role != ROLE_OWNER:
            raise APIError(403, "forbidden", "Global event feeds require OWNER role")
        return [_event_response(item) for item in self.observability.get_security_events(task_id)]

    def tool_events(self, task_id: str | None = None, *, principal: AuthenticatedPrincipal | None = None) -> list[EventResponse]:
        if task_id is not None:
            self._assert_task_access(task_id, principal)
            self._state(task_id)
        elif principal is not None and not principal.is_legacy and principal.role != ROLE_OWNER:
            raise APIError(403, "forbidden", "Global event feeds require OWNER role")
        return [_event_response(item) for item in self.observability.get_tool_events(task_id)]

    def readiness(self) -> tuple[bool, dict[str, str]]:
        owner_status = self.auth_manager.owner_status()
        legacy_status = self.authenticator.configured
        if owner_status == "configured":
            api_authentication = "configured"
            owner_authentication = "configured"
        elif owner_status == "invalid" or self.require_owner_auth:
            api_authentication = "invalid" if owner_status == "invalid" else "missing"
            owner_authentication = owner_status
        elif legacy_status:
            api_authentication = "configured"
            owner_authentication = "legacy_compatibility"
        else:
            api_authentication = "missing"
            owner_authentication = owner_status
        checks: dict[str, str] = {
            "api_authentication": api_authentication,
            "owner_authentication": owner_authentication,
        }
        try:
            with self.store._db_lock:
                self.store.db.execute("SELECT 1").fetchone()
                checks["sqlite"] = "ok"
                required = {
                    "task_states",
                    "task_checkpoints",
                    "approval_requests",
                    "observability_events",
                    "api_idempotency_keys",
                    "task_queue",
                    "auth_users",
                    "auth_sessions",
                    "task_owners",
                }
                rows = self.store.db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                tables = {str(row["name"]) for row in rows}
                checks["schema"] = "ok" if required.issubset(tables) else "missing"
        except Exception:
            checks["sqlite"] = "unavailable"
            checks["schema"] = "unknown"
        ready_values = {"configured", "ok", "legacy_compatibility"}
        if self.owner_auth_enforced():
            ready_values.discard("legacy_compatibility")
        return all(value in ready_values for value in checks.values()), checks


def build_orchestrator(*, database_path: str | None = None) -> Orchestrator:
    store = Store(database_path or os.getenv("AGENT_DB_PATH", "data/agent.sqlite3"))
    registry = ToolRegistry()
    # Local/Termux operation is fail-closed: approval or an environment flag
    # must not enable a network-capable tool. Registration belongs to a future
    # independently verified runtime boundary, not to this application profile.
    from private_agent.tools.bugbounty import BugBountyAnalysisTool

    registry.register(BugBountyAnalysisTool(store))
    scanner_limits = build_scanner_limits()
    httpx_binary = os.getenv("AGENT_HTTPX_BINARY", "httpx")
    nuclei_binary = os.getenv("AGENT_NUCLEI_BINARY", "nuclei")
    # Active scanner execution remains quarantined until a built-in external
    # OCI sandbox issues an opaque, execution-bound proof. Configuration flags
    # and environment variables are never treated as isolation evidence.
    registry.register(
        HttpxReconTool(
            binary_path=httpx_binary,
            limits=scanner_limits,
            trusted_installation=load_tool_spec_from_environment("httpx", httpx_binary),
        )
    )
    registry.register(
        NucleiScanTool(
            binary_path=nuclei_binary,
            templates_root=os.getenv("AGENT_NUCLEI_TEMPLATES_DIR") or None,
            expected_manifest_sha256=os.getenv("AGENT_NUCLEI_MANIFEST_SHA256", ""),
            limits=scanner_limits,
            trusted_installation=load_tool_spec_from_environment("nuclei", nuclei_binary),
        )
    )
    from private_agent.tools.repository import register_repository_tools

    register_repository_tools(registry)
    from private_agent.core.specialists import SpecialistRegistry

    specialist_registry = SpecialistRegistry.for_local_repository_tools(registry)
    planner = None
    if os.getenv("GEMINI_API_KEY", "").strip():
        from private_agent.core.planner import Planner

        planner = Planner(build_provider(), specialists=specialist_registry)
    else:
        from private_agent.core.planner import Planner

        planner = Planner(specialists=specialist_registry)
    return Orchestrator(
        store,
        registry,
        max_attempts=int(os.getenv("AGENT_MAX_ATTEMPTS", "2")),
        planner=planner,
        # Patch tools write only to agent-owned isolated workspaces; ordinary
        # private_write tools remain excluded from the default permission set.
        approved_permissions={"public_read", "isolated_write"},
        security_controller=SecurityController(store=store),
        specialist_registry=specialist_registry,
    )


def build_api_service(*, token: str | None = None, database_path: str | None = None) -> AgentAPIService:
    return AgentAPIService(build_orchestrator(database_path=database_path), token=token)


def _service(request: Request) -> AgentAPIService:
    service = getattr(request.app.state, "agent_api_service", None)
    if service is None:
        factory: Callable[[], AgentAPIService] = request.app.state.agent_api_service_factory
        service = factory()
        request.app.state.agent_api_service = service
    return service


def _principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_BEARER),
    service: AgentAPIService = Depends(_service),
) -> AuthenticatedPrincipal:
    authorization = None if credentials is None else f"{credentials.scheme} {credentials.credentials}"
    principal = service.authenticate_request(request, authorization)
    request.state.principal = principal.subject
    return principal


def _require_roles(*roles: str) -> Callable[..., AuthenticatedPrincipal]:
    allowed = frozenset(roles)

    def dependency(principal: AuthenticatedPrincipal = Depends(_principal)) -> AuthenticatedPrincipal:
        if principal.role not in allowed:
            raise APIError(403, "forbidden", "The authenticated role cannot perform this operation")
        return principal

    return dependency


def _csrf_protect(
    request: Request,
    principal: AuthenticatedPrincipal = Depends(_principal),
    service: AgentAPIService = Depends(_service),
) -> AuthenticatedPrincipal:
    service.verify_csrf(principal, request.headers.get("X-CSRF-Token"))
    return principal


def _secure_cookie(request: Request) -> bool:
    configured = os.getenv("AGENT_COOKIE_SECURE", "").strip().lower()
    if configured in {"1", "true", "yes", "on"}:
        return True
    if configured in {"0", "false", "no", "off"}:
        return False
    forwarded = request.headers.get("X-Forwarded-Proto", "").split(",", 1)[0].strip().lower()
    return request.url.scheme == "https" or forwarded == "https"


def _request_remote(request: Request) -> str | None:
    """Use the reverse proxy's normalized client address when present.

    The production Nginx config overwrites X-Real-IP from the connected client
    address before proxying to the loopback backend. Invalid values are ignored
    so a directly exposed backend cannot turn arbitrary headers into unlimited
    rate-limit buckets.
    """
    candidate = request.headers.get("X-Real-IP", "").strip()
    if candidate:
        try:
            return str(ipaddress.ip_address(candidate))
        except ValueError:
            pass
    return request.client.host if request.client is not None else None


def _cookie_samesite() -> str:
    value = os.getenv("AGENT_COOKIE_SAMESITE", "lax").strip().lower()
    if value not in {"lax", "strict", "none"}:
        raise APIError(500, "cookie_configuration_invalid", "Cookie SameSite configuration is invalid")
    return value


def _set_auth_cookies(request: Request, response: Response, login: Any, ttl_seconds: int) -> None:
    secure = _secure_cookie(request)
    samesite = _cookie_samesite()
    if samesite == "none" and not secure:
        raise APIError(500, "cookie_configuration_invalid", "SameSite=None requires Secure cookies")
    response.set_cookie(
        SESSION_COOKIE_NAME,
        login.session_token,
        max_age=ttl_seconds,
        httponly=True,
        secure=secure,
        samesite=samesite,
        path="/",
    )
    response.set_cookie(
        CSRF_COOKIE_NAME,
        login.csrf_token,
        max_age=ttl_seconds,
        httponly=False,
        secure=secure,
        samesite=samesite,
        path="/",
    )


def _error_payload(request: Request, error: APIError) -> dict[str, Any]:
    return ErrorResponse(
        error_code=error.error_code,
        message=error.message,
        request_id=_request_id(request),
        task_id=error.task_id,
        approval_id=error.approval_id,
        checks=error.checks,
    ).model_dump()


def create_app(
    *,
    service: AgentAPIService | None = None,
    token: str | None = None,
    worker: BackgroundWorker | None = None,
    worker_shutdown_timeout: float = 5.0,
    require_worker: bool = False,
) -> FastAPI:
    if worker is not None and service is None:
        raise ValueError("worker_requires_explicit_service")
    if worker_shutdown_timeout < 0:
        raise ValueError("worker_shutdown_timeout_invalid")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        worker_thread = None
        if worker is not None:
            worker_thread = worker.start()
            application.state.agent_worker_thread = worker_thread
        application.state.lifespan_started = True
        try:
            yield
        finally:
            application.state.lifespan_started = False
            if worker is not None:
                shutdown_error: str | None = None
                shutdown_timed_out = False
                shutdown_base_exception: BaseException | None = None

                def record_shutdown_error(event_name: str, exc: BaseException) -> None:
                    nonlocal shutdown_error, shutdown_base_exception
                    shutdown_error = shutdown_error or type(exc).__name__
                    if not isinstance(exc, Exception) and shutdown_base_exception is None:
                        shutdown_base_exception = exc
                    LOGGER.error(event_name, extra={"error_type": type(exc).__name__})

                try:
                    # This lifecycle fence is independent of the worker's stop
                    # Event, so a broken request_stop/set path cannot leave an
                    # idle loop running forever. It does not close SQLite.
                    worker.store.request_worker_shutdown()
                except BaseException as exc:
                    record_shutdown_error("worker_shutdown_store_request_failed", exc)

                def request_stop_safely() -> None:
                    try:
                        worker.request_stop()
                    except BaseException as exc:
                        record_shutdown_error("worker_shutdown_request_failed", exc)
                        stop_event = getattr(worker, "_stop_event", None)
                        set_stop = getattr(stop_event, "set", None)
                        if callable(set_stop):
                            try:
                                set_stop()
                            except BaseException as fallback_exc:
                                record_shutdown_error("worker_shutdown_signal_fallback_failed", fallback_exc)

                request_stop_safely()
                try:
                    stopped = worker.stop(timeout=worker_shutdown_timeout)
                    shutdown_timed_out = not stopped
                except BaseException as exc:
                    record_shutdown_error("worker_shutdown_wait_failed", exc)
                    request_stop_safely()
                finally:
                    # The configured timeout is an observability threshold, not
                    # permission to close resources under an active claim.
                    try:
                        worker.join(timeout=None)
                    except BaseException as exc:
                        record_shutdown_error("worker_shutdown_drain_failed", exc)
                        if worker_thread is not None:
                            try:
                                worker_thread.join()
                            except BaseException as join_exc:
                                record_shutdown_error("worker_shutdown_thread_join_failed", join_exc)
                    if worker_thread is not None and worker_thread.is_alive():
                        try:
                            worker_thread.join()
                        except BaseException as join_exc:
                            record_shutdown_error("worker_shutdown_thread_join_failed", join_exc)
                application.state.worker_shutdown_timed_out = shutdown_timed_out
                application.state.worker_shutdown_error = shutdown_error
                application.state.worker_shutdown_clean = bool(
                    worker_thread is None or not worker_thread.is_alive()
                ) and worker.fatal_error is None and shutdown_error is None
                if shutdown_base_exception is not None:
                    raise shutdown_base_exception

    application = FastAPI(
        title="Private Self-Learning Agent API",
        version=API_VERSION,
        description="Stable authenticated interface over the existing Agent Core.",
        lifespan=lifespan,
    )
    cors_origins = _configured_cors_origins()
    if cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "X-Request-ID", "X-CSRF-Token"],
            max_age=600,
        )
    application.state.agent_api_service = service
    application.state.agent_api_service_factory = lambda: build_api_service(token=token)
    application.state.agent_worker = worker
    application.state.worker_required = bool(require_worker or worker is not None)
    if service is not None:
        # The service can be reused by tests or by an embedding process. The
        # queue transition is enabled only for the application instance that
        # owns a worker lifecycle; never let a prior app configuration leak.
        service.auto_requeue_approvals = worker is not None

    @application.middleware("http")
    async def request_correlation(request: Request, call_next: Callable[..., Any]) -> Any:
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if _REQUEST_ID_PATTERN.fullmatch(supplied) else uuid.uuid4().hex
        timeline_read = request.method == "GET" and re.fullmatch(r"/(?:api/v1/)?tasks/[^/]+/timeline", request.url.path) is not None
        request.state.request_id = request_id
        current_service = getattr(application.state, "agent_api_service", None)
        if current_service is None and (request.url.path.startswith("/api/") or request.url.path == "/tasks/run"):
            current_service = _service(request)
        if current_service is not None:
            with current_service._lock:
                current_service.observability.emit(
                    "API_REQUEST_STARTED",
                    correlation_id=request_id,
                    component="api",
                    metadata={"method": request.method, "path": request.url.path},
                )
        try:
            response = await call_next(request)
        except Exception:
            if current_service is not None:
                with current_service._lock:
                    current_service.observability.emit(
                        "API_REQUEST_FAILED",
                        task_id="" if timeline_read else getattr(request.state, "task_id", ""),
                        correlation_id=request_id,
                        component="api",
                        severity="ERROR",
                        metadata={"method": request.method, "path": request.url.path},
                    )
            raise
        current_service = getattr(application.state, "agent_api_service", None)
        if current_service is not None:
            with current_service._lock:
                current_service.observability.emit(
                    "API_REQUEST_COMPLETED",
                    task_id="" if timeline_read else getattr(request.state, "task_id", ""),
                    correlation_id=request_id,
                    component="api",
                    severity="ERROR" if response.status_code >= 500 else "INFO",
                    metadata={"method": request.method, "path": request.url.path, "status_code": response.status_code},
                )
        response.headers["X-Request-ID"] = request_id
        return response

    @application.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=_error_payload(request, exc))

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        del exc
        error = APIError(422, "validation_error", "Request validation failed")
        return JSONResponse(status_code=422, content=_error_payload(request, error))

    @application.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        error = APIError(exc.status_code, "http_error", "Request could not be processed")
        return JSONResponse(status_code=exc.status_code, content=_error_payload(request, error))

    @application.exception_handler(Exception)
    async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
        del exc
        error = APIError(500, "internal_error", "Internal server error")
        return JSONResponse(status_code=500, content=_error_payload(request, error))

    def _require_worker_ready(request: Request) -> None:
        worker_instance = getattr(request.app.state, "agent_worker", None)
        if getattr(request.app.state, "worker_required", False) and (
            worker_instance is None or not worker_instance.running
        ):
            raise APIError(503, "worker_not_ready", "Task processing worker is not running")

    def _worker_authenticate(request: Request, service: AgentAPIService, body: Mapping[str, Any]) -> WorkerAuthentication:
        authorization = request.headers.get("Authorization", "")
        if not authorization.startswith("Worker "):
            raise HTTPException(status_code=401, detail="worker_authentication_required")
        try:
            return service.remote_worker_protocol.authenticate(
                worker_id=request.headers.get("X-Worker-ID", ""),
                credential=authorization[len("Worker "):],
                method=request.method,
                path=request.url.path,
                timestamp=int(request.headers.get("X-Worker-Timestamp", "")),
                nonce=request.headers.get("X-Worker-Nonce", ""),
                body=body,
                signature=request.headers.get("X-Worker-Signature", ""),
            )
        except (WorkerProtocolError, ValueError) as exc:
            code = getattr(exc, "code", "worker_authentication_failed")
            http_status = 409 if code in {"worker_request_replay", "worker_timestamp_out_of_window"} else 401
            raise HTTPException(status_code=http_status, detail=code) from exc

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.get("/ready", response_model=ReadinessResponse, responses={503: {"model": ErrorResponse}}, tags=["system"])
    def ready(request: Request, service: AgentAPIService = Depends(_service)) -> ReadinessResponse:
        ready_status, checks = service.readiness()
        worker_instance = getattr(request.app.state, "agent_worker", None)
        if getattr(request.app.state, "worker_required", False) and worker_instance is None:
            checks["worker"] = "missing"
            ready_status = False
        elif worker_instance is not None:
            if worker_instance.running:
                checks["worker"] = "running"
            elif worker_instance.fatal_error is not None:
                checks["worker"] = "fatal"
            else:
                checks["worker"] = "stopped"
            ready_status = ready_status and checks["worker"] == "running"
        else:
            checks["worker"] = "not_required"
        response = ReadinessResponse(status="ready" if ready_status else "not_ready", checks=checks, request_id=_request_id(request))
        if not ready_status:
            raise APIError(503, "not_ready", "Required API dependencies are not ready", checks=checks)
        return response

    @application.post("/api/v1/auth/login", response_model=LoginResponse, responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}})
    def login(request: Request, payload: LoginRequest, response: Response, service: AgentAPIService = Depends(_service)) -> LoginResponse:
        remote = _request_remote(request)
        login_result = service.login(payload.email, payload.password, remote=remote)
        _set_auth_cookies(request, response, login_result, service.auth_manager.session_ttl_seconds)
        return LoginResponse(
            user=service.auth_user_response(login_result.user),
            expires_at=login_result.expires_at,
        )

    @application.get("/api/v1/auth/session", response_model=LoginResponse, responses={401: {"model": ErrorResponse}})
    def session(request: Request, service: AgentAPIService = Depends(_service), principal: AuthenticatedPrincipal = Depends(_principal)) -> LoginResponse:
        del request
        return service.session_response(principal)

    @application.post("/api/v1/auth/logout", response_model=dict[str, bool], responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}})
    def logout(
        request: Request,
        response: Response,
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_csrf_protect),
    ) -> dict[str, bool]:
        del request
        service.logout(principal)
        response.delete_cookie(SESSION_COOKIE_NAME, path="/")
        response.delete_cookie(CSRF_COOKIE_NAME, path="/")
        return {"logged_out": True}

    router = APIRouter(
        prefix="/api/v1",
        tags=["agent-api"],
        dependencies=[Depends(_principal)],
        responses={
            400: {"model": ErrorResponse},
            401: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )

    @router.get("/runtime/preflight", response_model=RuntimePreflightResponse)
    def runtime_preflight(request: Request) -> RuntimePreflightResponse:
        report = run_host_preflight()
        return RuntimePreflightResponse(
            verdict=report.verdict.value,
            host_ready=report.host_ready,
            can_run_acceptance=report.can_run_acceptance,
            acceptance_required=report.acceptance_required,
            runtime_identity=report.runtime_identity,
            checks=[
                RuntimePreflightCheckResponse(name=check.name, status=check.status.value, detail=check.detail)
                for check in report.checks
            ],
            request_id=_request_id(request),
        )

    @router.post(
        "/tasks",
        response_model=TaskCreateResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(_require_roles(ROLE_OWNER, ROLE_OPERATOR)), Depends(_csrf_protect)],
    )
    def create_task(
        request: Request,
        payload: TaskCreateRequest,
        service: AgentAPIService = Depends(_service),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> TaskCreateResponse:
        _require_worker_ready(request)
        response = service.create_task(payload.goal, idempotency_key=idempotency_key, request_id=_request_id(request), principal=principal)
        request.state.task_id = response.task_id
        return response

    @router.post(
        "/bug-bounty/programs/analyze",
        response_model=ProgramAnalysisResponse,
        dependencies=[Depends(_require_roles(ROLE_OWNER, ROLE_OPERATOR)), Depends(_csrf_protect)],
    )
    def analyze_bug_bounty_program(
        request: Request,
        payload: ProgramAnalysisRequest,
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> ProgramAnalysisResponse:
        return service.analyze_program(payload, request_id=_request_id(request), principal=principal)

    @router.get("/bug-bounty/programs", response_model=list[ProgramAnalysisResponse])
    def list_bug_bounty_programs(
        limit: int = Query(default=50, ge=1, le=100),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> list[ProgramAnalysisResponse]:
        return service.list_program_analyses(principal=principal, limit=limit)

    @router.get("/bug-bounty/programs/{analysis_id}", response_model=ProgramAnalysisResponse)
    def get_bug_bounty_program(
        request: Request,
        analysis_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> ProgramAnalysisResponse:
        return service.get_program_analysis(analysis_id, principal=principal, request_id=_request_id(request))

    @router.post(
        "/bug-bounty/programs/{analysis_id}/task",
        response_model=TaskCreateResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(_require_roles(ROLE_OWNER)), Depends(_csrf_protect)],
    )
    def create_bug_bounty_task_from_analysis(
        request: Request,
        payload: ProgramAnalysisTaskRequest,
        analysis_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> TaskCreateResponse:
        _require_worker_ready(request)
        response = service.create_bug_bounty_task_from_analysis(
            analysis_id,
            payload,
            idempotency_key=idempotency_key,
            request_id=_request_id(request),
            principal=principal,
        )
        request.state.task_id = response.task_id
        return response

    @router.post(
        "/bug-bounty/programs/{analysis_id}/stages",
        response_model=ResearchStageResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(_require_roles(ROLE_OWNER, ROLE_OPERATOR)), Depends(_csrf_protect)],
    )
    def create_research_stage(
        request: Request,
        payload: ResearchStageRequest,
        analysis_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> ResearchStageResponse:
        return service.create_research_stage(analysis_id, payload, principal=principal, request_id=_request_id(request))

    @router.get("/bug-bounty/programs/{analysis_id}/stages", response_model=list[ResearchStageResponse])
    def list_research_stages(
        request: Request,
        analysis_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> list[ResearchStageResponse]:
        return service.list_research_stages(analysis_id, principal=principal, request_id=_request_id(request))

    @router.get("/bug-bounty/research-stages/{stage_id}", response_model=ResearchStageResponse)
    def get_research_stage(
        request: Request,
        stage_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> ResearchStageResponse:
        return service.get_research_stage(stage_id, principal=principal, request_id=_request_id(request))

    @router.post(
        "/bug-bounty/research-stages/{stage_id}/cancel",
        response_model=ResearchStageResponse,
        dependencies=[Depends(_require_roles(ROLE_OWNER, ROLE_OPERATOR)), Depends(_csrf_protect)],
    )
    def cancel_research_stage(
        request: Request,
        stage_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> ResearchStageResponse:
        return service.cancel_research_stage(stage_id, principal=principal, request_id=_request_id(request))

    @router.post(
        "/bug-bounty/tasks",
        response_model=TaskCreateResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(_require_roles(ROLE_OWNER, ROLE_OPERATOR)), Depends(_csrf_protect)],
    )
    def create_bug_bounty_task(
        request: Request,
        payload: BugBountyCreateRequest,
        service: AgentAPIService = Depends(_service),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> TaskCreateResponse:
        _require_worker_ready(request)
        response = service.create_bug_bounty_task(
            payload,
            idempotency_key=idempotency_key,
            request_id=_request_id(request),
            principal=principal,
        )
        request.state.task_id = response.task_id
        return response

    @router.get("/tasks", response_model=TaskListResponse)
    def list_tasks(
        request: Request,
        limit: int = Query(default=20, ge=1, le=100),
        cursor: str | None = Query(default=None, max_length=512),
        status_filter: TaskStatusFilter | None = Query(default=None, alias="status"),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> TaskListResponse:
        return service.list_tasks(
            limit=limit,
            cursor=cursor,
            status_filter=status_filter.value if status_filter is not None else None,
            request_id=_request_id(request),
            principal=principal,
        )

    @router.get("/tasks/{task_id}", response_model=TaskStateResponse)
    def get_task(
        request: Request,
        task_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> TaskStateResponse:
        request.state.task_id = task_id
        return service.get_task(task_id, request_id=_request_id(request), principal=principal)

    @router.get("/tasks/{task_id}/messages", response_model=TaskConversationResponse)
    def get_task_conversation(
        request: Request,
        task_id: str = APIPath(..., min_length=1, max_length=256),
        after_sequence: int = Query(default=0, ge=0, le=9223372036854775807),
        limit: int = Query(default=100, ge=1, le=100),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> TaskConversationResponse:
        request.state.task_id = task_id
        return service.get_task_conversation(
            task_id,
            request_id=_request_id(request),
            principal=principal,
            after_sequence=after_sequence,
            limit=limit,
        )

    @router.post(
        "/tasks/{task_id}/messages",
        response_model=TaskMessageResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(_require_roles(ROLE_OWNER, ROLE_OPERATOR)), Depends(_csrf_protect)],
    )
    def append_task_message(
        request: Request,
        payload: TaskMessageRequest,
        task_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> TaskMessageResponse:
        request.state.task_id = task_id
        return service.append_task_message(
            task_id,
            payload.content,
            idempotency_key=idempotency_key,
            request_id=_request_id(request),
            principal=principal,
        )

    @router.post(
        "/tasks/{task_id}/capability-candidates/assess",
        response_model=CapabilityCandidateAssessmentResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(_require_roles(ROLE_OWNER)), Depends(_csrf_protect)],
    )
    def assess_capability_candidate(
        request: Request,
        payload: CapabilityCandidateAssessmentRequest,
        task_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> CapabilityCandidateAssessmentResponse:
        request.state.task_id = task_id
        return service.assess_capability_candidate(
            task_id,
            payload,
            request_id=_request_id(request),
            principal=principal,
        )

    @router.get(
        "/capability-candidates/{candidate_id}",
        response_model=CapabilityCandidateAssessmentResponse,
        dependencies=[Depends(_require_roles(ROLE_OWNER))],
    )
    def get_capability_candidate_assessment(
        request: Request,
        candidate_id: str = APIPath(..., min_length=1, max_length=128),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> CapabilityCandidateAssessmentResponse:
        result = service.get_capability_candidate_assessment(
            candidate_id,
            request_id=_request_id(request),
            principal=principal,
        )
        request.state.task_id = result.capability_gap_task_id
        return result

    @router.post(
        "/tasks/{task_id}/resume",
        response_model=TaskStateResponse,
        dependencies=[Depends(_require_roles(ROLE_OWNER, ROLE_OPERATOR)), Depends(_csrf_protect)],
    )
    def resume_task(
        request: Request,
        payload: ResumeRequest | None = Body(default=None),
        task_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> TaskStateResponse:
        request.state.task_id = task_id
        return service.resume_task(task_id, (payload.approval_ids if payload else {}), request_id=_request_id(request), principal=principal)

    @router.get("/tasks/{task_id}/timeline", response_model=list[EventResponse])
    def task_timeline(
        request: Request,
        task_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> list[EventResponse]:
        request.state.task_id = task_id
        return service.task_timeline(task_id, principal=principal)

    @router.get("/tasks/{task_id}/metrics", response_model=MetricsResponse)
    def task_metrics(
        request: Request,
        task_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> MetricsResponse:
        request.state.task_id = task_id
        return service.task_metrics(task_id, principal=principal)

    @router.post(
        "/bug-bounty/tasks/{task_id}/findings/{finding_id}/review",
        response_model=FindingReviewResponse,
        dependencies=[Depends(_require_roles(ROLE_OWNER)), Depends(_csrf_protect)],
    )
    def review_bug_bounty_finding(
        request: Request,
        payload: FindingReviewRequest,
        task_id: str = APIPath(..., min_length=1, max_length=256),
        finding_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> FindingReviewResponse:
        request.state.task_id = task_id
        return service.review_bug_bounty_finding(task_id, finding_id, payload, principal=principal, request_id=_request_id(request))

    @router.post(
        "/bug-bounty/tasks/{task_id}/reviews/{review_id}/knowledge-attestation",
        response_model=FindingKnowledgeAttestationResponse,
        dependencies=[Depends(_require_roles(ROLE_OWNER)), Depends(_csrf_protect)],
    )
    def issue_finding_knowledge_attestation(
        request: Request,
        task_id: str = APIPath(..., min_length=1, max_length=256),
        review_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> FindingKnowledgeAttestationResponse:
        request.state.task_id = task_id
        return service.issue_finding_knowledge_attestation(
            task_id, review_id, principal=principal, request_id=_request_id(request)
        )

    @router.get("/bug-bounty/tasks/{task_id}/reviews", response_model=list[FindingReviewResponse])
    def list_finding_reviews(
        request: Request,
        task_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> list[FindingReviewResponse]:
        request.state.task_id = task_id
        return service.list_finding_reviews(task_id, principal=principal, request_id=_request_id(request))

    @router.get("/bug-bounty/tasks/{task_id}", response_model=BugBountyTaskResponse)
    def bug_bounty_task(
        request: Request,
        task_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> BugBountyTaskResponse:
        request.state.task_id = task_id
        return service.bug_bounty_task(task_id, request_id=_request_id(request), principal=principal)

    @router.get("/bug-bounty/tasks/{task_id}/{section}", response_model=BugBountySectionResponse)
    def bug_bounty_section(
        request: Request,
        task_id: str = APIPath(..., min_length=1, max_length=256),
        section: str = APIPath(..., pattern="^(inventory|findings|evidence|report)$"),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> BugBountySectionResponse:
        request.state.task_id = task_id
        return service.bug_bounty_section(task_id, section, request_id=_request_id(request), principal=principal)

    @router.get("/approvals", response_model=list[ApprovalResponse])
    def list_approvals(
        pending_only: bool = Query(default=True),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> list[ApprovalResponse]:
        return service.list_approvals(pending_only=pending_only, principal=principal)

    @router.get("/approvals/{approval_id}", response_model=ApprovalResponse)
    def get_approval(
        approval_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> ApprovalResponse:
        return service.get_approval_response(approval_id, principal=principal)

    @router.post(
        "/approvals/{approval_id}/approve",
        response_model=ApprovalResponse,
        dependencies=[Depends(_require_roles(ROLE_OWNER)), Depends(_csrf_protect)],
    )
    def approve_approval(
        approval_id: str = APIPath(..., min_length=1, max_length=256),
        payload: ApprovalActionRequest | None = Body(default=None),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> ApprovalResponse:
        del payload
        return service.approve(approval_id, principal=principal)

    @router.post(
        "/approvals/{approval_id}/deny",
        response_model=ApprovalResponse,
        dependencies=[Depends(_require_roles(ROLE_OWNER)), Depends(_csrf_protect)],
    )
    def deny_approval(
        approval_id: str = APIPath(..., min_length=1, max_length=256),
        payload: ApprovalActionRequest | None = Body(default=None),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> ApprovalResponse:
        del payload
        return service.deny(approval_id, principal=principal)

    @router.get("/metrics", response_model=MetricsResponse, dependencies=[Depends(_require_roles(ROLE_OWNER))])
    def system_metrics(service: AgentAPIService = Depends(_service)) -> MetricsResponse:
        return service.system_metrics()

    @router.get("/failures", response_model=list[EventResponse])
    def failures(
        request: Request,
        task_id: str | None = Query(default=None, max_length=256),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> list[EventResponse]:
        if request is not None and task_id:
            request.state.task_id = task_id
        return service.failures(task_id, principal=principal)

    @router.get("/security-events", response_model=list[EventResponse])
    def security_events(
        request: Request,
        task_id: str | None = Query(default=None, max_length=256),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> list[EventResponse]:
        if request is not None and task_id:
            request.state.task_id = task_id
        return service.security_events(task_id, principal=principal)

    @router.get("/tool-events", response_model=list[EventResponse])
    def tool_events(
        request: Request,
        task_id: str | None = Query(default=None, max_length=256),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> list[EventResponse]:
        if request is not None and task_id:
            request.state.task_id = task_id
        return service.tool_events(task_id, principal=principal)

    @router.get("/tools", response_model=list[ToolResponse])
    def tools(service: AgentAPIService = Depends(_service)) -> list[ToolResponse]:
        result: list[ToolResponse] = []
        for item in service.orchestrator.tools.available():
            result.append(
                ToolResponse(
                    name=str(item.get("name", "")),
                    description=str(item.get("description", "")),
                    permission_level=str(item.get("permission_level", "")),
                    risk_level=str(item.get("risk_level", "")),
                    required_capabilities=[str(value) for value in item.get("required_capabilities", [])],
                    network_required=bool(item.get("network_required", False)),
                )
            )
        return result

    @application.post("/api/v1/worker/jobs/claim", response_model=dict[str, Any] | None, tags=["remote-worker"])
    def worker_claim(
        request: Request,
        payload: WorkerClaimRequest,
        service: AgentAPIService = Depends(_service),
    ) -> dict[str, Any] | None:
        auth = _worker_authenticate(request, service, payload.model_dump())
        try:
            return service.remote_worker_protocol.claim(auth)
        except WorkerProtocolError as exc:
            raise HTTPException(status_code=409, detail=exc.code) from exc

    @application.post("/api/v1/worker/jobs/{job_id}/heartbeat", response_model=dict[str, Any], tags=["remote-worker"])
    def worker_heartbeat(
        request: Request,
        payload: WorkerHeartbeatRequest,
        service: AgentAPIService = Depends(_service),
    ) -> dict[str, Any]:
        auth = _worker_authenticate(request, service, payload.model_dump())
        try:
            return service.remote_worker_protocol.heartbeat(auth, job_id=payload.job_id, lease_id=payload.lease_id)
        except WorkerProtocolError as exc:
            raise HTTPException(status_code=409, detail=exc.code) from exc

    @application.post("/api/v1/worker/jobs/{job_id}/result", response_model=dict[str, Any], tags=["remote-worker"])
    def worker_result(
        request: Request,
        payload: WorkerResultRequest,
        service: AgentAPIService = Depends(_service),
    ) -> dict[str, Any]:
        auth = _worker_authenticate(request, service, payload.model_dump())
        try:
            return service.remote_worker_protocol.submit_result(
                auth, job_id=payload.job_id, lease_id=payload.lease_id, result=payload.result
            )
        except WorkerProtocolError as exc:
            status_code = 422 if exc.code.startswith("worker_result_") else 409
            raise HTTPException(status_code=status_code, detail=exc.code) from exc

    @router.get(
        "/users",
        response_model=list[AuthUserResponse],
        dependencies=[Depends(_require_roles(ROLE_OWNER))],
    )
    def list_users(service: AgentAPIService = Depends(_service)) -> list[AuthUserResponse]:
        return service.list_users_response()

    @router.post(
        "/users",
        response_model=AuthUserResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(_require_roles(ROLE_OWNER)), Depends(_csrf_protect)],
    )
    def create_user(
        payload: UserCreateRequest,
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> AuthUserResponse:
        return service.create_user_response(payload.email, payload.password, payload.role, principal=principal)

    @router.patch(
        "/users/{user_id}/role",
        response_model=AuthUserResponse,
        dependencies=[Depends(_require_roles(ROLE_OWNER)), Depends(_csrf_protect)],
    )
    def update_user_role(
        user_id: str = APIPath(..., min_length=1, max_length=256),
        payload: UserRoleRequest = Body(...),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> AuthUserResponse:
        return service.set_user_role_response(user_id, payload.role, principal=principal)

    @router.post(
        "/users/{user_id}/disable",
        response_model=AuthUserResponse,
        dependencies=[Depends(_require_roles(ROLE_OWNER)), Depends(_csrf_protect)],
    )
    def disable_user(
        user_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> AuthUserResponse:
        return service.disable_user_response(user_id, principal=principal)

    application.include_router(router)

    # Keep the historical route protected and typed; it is a compatibility alias,
    # not a second execution implementation. New clients should use /api/v1/tasks.
    @application.post(
        "/tasks/run",
        response_model=TaskCreateResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["legacy"],
        dependencies=[Depends(_require_roles(ROLE_OWNER, ROLE_OPERATOR)), Depends(_csrf_protect)],
    )
    def legacy_run(
        request: Request,
        payload: TaskCreateRequest,
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> TaskCreateResponse:
        _require_worker_ready(request)
        response = service.create_task(payload.goal, request_id=_request_id(request), principal=principal)
        request.state.task_id = response.task_id
        return response

    return application


app = create_app(require_worker=True)

__all__ = [
    "APIError",
    "AgentAPIService",
    "ApprovalActionRequest",
    "ApprovalResponse",
    "ErrorResponse",
    "EventResponse",
    "MetricsResponse",
    "ReadinessResponse",
    "ResumeRequest",
    "TaskCreateRequest",
    "TaskCreateResponse",
    "TaskStateResponse",
    "TokenAuthenticator",
    "app",
    "build_api_service",
    "build_orchestrator",
    "create_app",
]
