from __future__ import annotations

import base64
import binascii
import hashlib
import ipaddress
import json
import os
import re
import secrets
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable
from urllib.parse import urlparse

from fastapi import APIRouter, Body, Depends, FastAPI, Header, HTTPException, Path as APIPath, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from private_agent.core.llm import build_provider
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.task_state import MAX_TASK_GOAL_LENGTH, TASK_STATUSES, StateIntegrityError, TaskState
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
from private_agent.bugbounty.models import Finding, FindingState, ScopeDefinition, public_reference, reconcile_bug_bounty_metrics, safe_json
from private_agent.observability import ObservabilityMonitor, safe_metadata
from private_agent.security import ApprovalRequest, SecurityController
from private_agent.storage import QueueLeaseLost, Store
from private_agent.bugbounty.trusted_tools import load_tool_spec_from_environment
from private_agent.tools.research import ToolRegistry, WebResearchTool
from private_agent.tools.scanners import HttpxReconTool, NucleiScanTool, build_scanner_limits
from private_agent.worker import BackgroundWorker


API_VERSION = "v1"
API_SCHEMA_VERSION = 1
SERVICE_NAME = "private-agent"
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
    status: str
    created_at: str
    current_phase: str
    api_version: str = API_VERSION
    schema_version: int = API_SCHEMA_VERSION
    request_id: str = ""
    idempotent_replay: bool = False


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
    run: dict[str, Any] | None = None
    scan: dict[str, Any] | None = None
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
    status: str
    current_phase: str
    current_step: str
    plan_version: int
    completed_actions: list[str]
    pending_actions: list[str]
    failed_actions: list[str]
    actions: list[ActionSummary]
    recovery_state: dict[str, Any]
    verification_status: str
    checkpoint: CheckpointResponse
    approvals: list[ApprovalResponse]
    created_at: str
    updated_at: str
    queue_status: str = ""
    api_version: str = API_VERSION
    schema_version: int = API_SCHEMA_VERSION
    request_id: str = ""


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
    if any(item in {"FAILED", "BLOCKED", "INSUFFICIENT"} for item in statuses):
        verification_status = "FAILED"
    elif statuses and all(item == "VERIFIED" for item in statuses):
        verification_status = "VERIFIED"
    elif statuses:
        verification_status = statuses[-1]
    if isinstance(state.result.get("verification"), dict):
        result_status = state.result["verification"].get("status")
        if result_status:
            verification_status = str(result_status)
    return verification_status


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
        self.require_owner_auth = os.getenv("AGENT_REQUIRE_OWNER_AUTH", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.auto_requeue_approvals = False
        self._lock = threading.RLock()

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
            status=state.status,
            created_at=state.created_at,
            current_phase=state.current_phase,
            request_id=request_id,
            idempotent_replay=replay,
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
            status=state.status,
            current_phase=state.current_phase,
            current_step=state.current_step,
            plan_version=state.plan_version,
            completed_actions=list(state.completed_actions),
            pending_actions=list(state.pending_actions),
            failed_actions=list(state.failed_actions),
            actions=actions,
            recovery_state=_public_value(state.recovery_state),
            verification_status=_verification_status(state),
            checkpoint=CheckpointResponse(
                last_checkpoint_at=state.last_checkpoint_at,
                last_successful_checkpoint=state.last_successful_checkpoint,
                checkpoint_type=str(checkpoint.get("checkpoint_type", "")),
                sequence=int(checkpoint.get("sequence", 0)),
            ),
            approvals=approvals,
            created_at=state.created_at,
            updated_at=state.updated_at,
            queue_status=str(queue.get("queue_status", "")),
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
                )
            )
        next_cursor = None
        if has_more and page:
            next_cursor = _encode_task_cursor(str(page[-1]["created_at"]), str(page[-1]["task_id"]))
        return TaskListResponse(items=items, next_cursor=next_cursor, request_id=request_id)

    def get_task(self, task_id: str, *, request_id: str = "", principal: AuthenticatedPrincipal | None = None) -> TaskStateResponse:
        self._assert_task_access(task_id, principal)
        return self.task_state_response(self._state(task_id), request_id=request_id)

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
        fingerprint = hashlib.sha256(
            json.dumps({"goal": goal, "request": request}, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return self.create_task(
            goal,
            idempotency_key=idempotency_key,
            request_id=request_id,
            context={"task_kind": "bug_bounty", "bugbounty_request": request},
            request_fingerprint=fingerprint,
            principal=principal,
        )

    def bug_bounty_task(self, task_id: str, *, request_id: str = "", principal: AuthenticatedPrincipal | None = None) -> BugBountyTaskResponse:
        self._assert_task_access(task_id, principal)
        state = self._state(task_id)
        if state.context.get("task_kind") != "bug_bounty":
            raise APIError(404, "bug_bounty_task_not_found", "Bug Bounty task not found", task_id=task_id)
        output = self._bug_bounty_output(state) or {}
        run = output.get("run") if isinstance(output.get("run"), dict) else None
        safe_run, report_suppressed = self._sanitize_bug_bounty_run(run) if run is not None else (None, False)
        scan = output.get("scan") if isinstance(output.get("scan"), dict) else None
        safe_scan = self._sanitize_bug_bounty_scan(scan) if scan is not None else None
        limitations = list(safe_run.get("limitations", [])) if safe_run else []
        if safe_scan:
            limitations.extend(str(item) for item in safe_scan.get("limitations", [])[:50])
        return BugBountyTaskResponse(
            task_id=state.task_id,
            status=state.status,
            verification_status=_verification_status(state),
            run=safe_run,
            scan=safe_scan,
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


def build_orchestrator() -> Orchestrator:
    store = Store(os.getenv("AGENT_DB_PATH", "data/agent.sqlite3"))
    registry = ToolRegistry()
    registry.register(WebResearchTool(timeout=float(os.getenv("AGENT_HTTP_TIMEOUT", "15"))))
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
    planner = None
    if os.getenv("GEMINI_API_KEY"):
        from private_agent.core.planner import Planner

        planner = Planner(build_provider())
    else:
        from private_agent.core.planner import Planner

        planner = Planner()
    return Orchestrator(
        store,
        registry,
        max_attempts=int(os.getenv("AGENT_MAX_ATTEMPTS", "2")),
        planner=planner,
        # Patch tools write only to agent-owned isolated workspaces; ordinary
        # private_write tools remain excluded from the default permission set.
        approved_permissions={"public_read", "isolated_write"},
        security_controller=SecurityController(store=store),
    )


def build_api_service(*, token: str | None = None) -> AgentAPIService:
    return AgentAPIService(build_orchestrator(), token=token)


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
                stopped = worker.stop(timeout=worker_shutdown_timeout)
                if worker_thread is not None:
                    worker_thread.join(timeout=worker_shutdown_timeout)
                application.state.worker_shutdown_clean = bool(stopped and not worker_thread.is_alive()) if worker_thread is not None else bool(stopped)

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
                        task_id=getattr(request.state, "task_id", ""),
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
                    task_id=getattr(request.state, "task_id", ""),
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
