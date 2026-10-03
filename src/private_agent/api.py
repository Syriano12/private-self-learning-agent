from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import APIRouter, Body, Depends, FastAPI, Header, HTTPException, Path as APIPath, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator

from private_agent.core.llm import build_provider
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.task_state import StateIntegrityError, TaskState
from private_agent.observability import ObservabilityMonitor, safe_metadata
from private_agent.security import ApprovalRequest, SecurityController
from private_agent.storage import Store
from private_agent.tools.research import ToolRegistry, WebResearchTool


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
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.task_id = task_id
        self.approval_id = approval_id


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskCreateRequest(APIModel):
    goal: str = Field(min_length=1, max_length=4000)

    @field_validator("goal")
    @classmethod
    def non_blank_goal(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("goal_required")
        return value


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


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    subject: str = "single-owner"


class TokenAuthenticator:
    """Bearer authentication for the single-owner deployment.

    There is intentionally no development fallback. A missing configuration is
    a readiness failure, not an authentication bypass.
    """

    def __init__(self, token: str | None = None) -> None:
        self.token = token if token is not None else os.getenv("AGENT_API_TOKEN")

    @property
    def configured(self) -> bool:
        return bool(self.token)

    def authenticate(self, authorization: str | None) -> AuthenticatedPrincipal:
        if not self.configured:
            raise APIError(503, "api_auth_not_configured", "API authentication is not configured")
        if not authorization or not authorization.startswith("Bearer "):
            raise APIError(401, "unauthenticated", "Bearer authentication is required")
        candidate = authorization[len("Bearer ") :]
        if not candidate or " " in candidate or not secrets.compare_digest(candidate, str(self.token)):
            raise APIError(401, "invalid_credentials", "Invalid API credentials")
        return AuthenticatedPrincipal()


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
        self._lock = threading.RLock()

    def _state(self, task_id: str) -> TaskState:
        task_id = _safe_id(task_id, "task_id")
        raw = self.store.get_task_state(task_id)
        if raw is None:
            raise APIError(404, "task_not_found", "Task was not found", task_id=task_id)
        try:
            return TaskState.from_dict(raw)
        except (StateIntegrityError, TypeError, ValueError) as exc:
            raise APIError(409, "task_state_untrusted", "Task state failed integrity validation", task_id=task_id) from exc

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
            verification_status=verification_status,
            checkpoint=CheckpointResponse(
                last_checkpoint_at=state.last_checkpoint_at,
                last_successful_checkpoint=state.last_successful_checkpoint,
                checkpoint_type=str(checkpoint.get("checkpoint_type", "")),
                sequence=int(checkpoint.get("sequence", 0)),
            ),
            approvals=approvals,
            created_at=state.created_at,
            updated_at=state.updated_at,
            request_id=request_id,
        )

    def create_task(self, goal: str, *, idempotency_key: str | None = None, request_id: str = "") -> TaskCreateResponse:
        goal = goal.strip()
        if not goal:
            raise APIError(422, "goal_required", "Task goal is required")
        fingerprint = hashlib.sha256(json.dumps({"goal": goal}, sort_keys=True).encode()).hexdigest()
        key = None
        if idempotency_key is not None:
            if not _IDEMPOTENCY_PATTERN.fullmatch(idempotency_key):
                raise APIError(400, "invalid_idempotency_key", "Invalid Idempotency-Key")
            key = idempotency_key
        with self._lock:
            if key:
                existing = self.store.get_api_idempotency(key)
                if existing is not None:
                    if existing["request_fingerprint"] != fingerprint:
                        raise APIError(409, "idempotency_conflict", "Idempotency-Key was already used for another request")
                    state = self._state(existing["task_id"])
                    return self._task_response(state, request_id=request_id, replay=True)
            result = self.orchestrator.run(goal)
            task_id = str(result.get("task_id", ""))
            state = self._state(task_id)
            response = self._task_response(state, request_id=request_id)
            if key:
                self.store.save_api_idempotency(
                    key,
                    fingerprint,
                    task_id,
                    response.model_dump_json(),
                )
            return response

    def get_task(self, task_id: str, *, request_id: str = "") -> TaskStateResponse:
        return self.task_state_response(self._state(task_id), request_id=request_id)

    def resume_task(self, task_id: str, approval_ids: dict[str, str] | None = None, *, request_id: str = "") -> TaskStateResponse:
        task_id = _safe_id(task_id, "task_id")
        with self._lock:
            self._state(task_id)
            self.orchestrator.resume_task(task_id, approval_ids=approval_ids or {})
            state = self._state(task_id)
            return self.task_state_response(state, request_id=request_id)

    def list_approvals(self, *, pending_only: bool = True) -> list[ApprovalResponse]:
        if self.security is None:
            raise APIError(503, "security_not_configured", "Approval operations are unavailable")
        raw_requests = self.store.all_approval_requests()
        responses: list[ApprovalResponse] = []
        for raw in raw_requests:
            approval_id = str(raw.get("approval_id", ""))
            current = self.security.approval_gate.get(approval_id)
            item = _approval_response(current or raw)
            if pending_only and item.status != "PENDING":
                continue
            responses.append(item)
        return responses

    def approve(self, approval_id: str) -> ApprovalResponse:
        with self._lock:
            request = self._approval(approval_id)
            try:
                return _approval_response(self.security.approval_gate.approve(request.approval_id, actor="api-owner"))  # type: ignore[union-attr]
            except ValueError as exc:
                raise APIError(409, "approval_not_pending", "Approval is no longer pending", approval_id=request.approval_id, task_id=request.task_id) from exc

    def deny(self, approval_id: str) -> ApprovalResponse:
        with self._lock:
            request = self._approval(approval_id)
            try:
                return _approval_response(self.security.approval_gate.deny(request.approval_id, actor="api-owner"))  # type: ignore[union-attr]
            except ValueError as exc:
                raise APIError(409, "approval_not_pending", "Approval is no longer pending", approval_id=request.approval_id, task_id=request.task_id) from exc

    def task_timeline(self, task_id: str) -> list[EventResponse]:
        self._state(task_id)
        return [_event_response(item) for item in self.observability.get_task_timeline(task_id)]

    def task_metrics(self, task_id: str) -> MetricsResponse:
        self._state(task_id)
        return MetricsResponse(**self.observability.get_task_metrics(task_id))

    def system_metrics(self) -> MetricsResponse:
        return MetricsResponse(**self.observability.get_system_metrics())

    def failures(self, task_id: str | None = None) -> list[EventResponse]:
        if task_id is not None:
            self._state(task_id)
        return [_event_response(item) for item in self.observability.get_failures(task_id)]

    def security_events(self, task_id: str | None = None) -> list[EventResponse]:
        if task_id is not None:
            self._state(task_id)
        return [_event_response(item) for item in self.observability.get_security_events(task_id)]

    def tool_events(self, task_id: str | None = None) -> list[EventResponse]:
        if task_id is not None:
            self._state(task_id)
        return [_event_response(item) for item in self.observability.get_tool_events(task_id)]

    def readiness(self) -> tuple[bool, dict[str, str]]:
        checks: dict[str, str] = {"api_authentication": "configured" if self.authenticator.configured else "missing"}
        try:
            self.store.db.execute("SELECT 1").fetchone()
            checks["sqlite"] = "ok"
            required = {
                "task_states",
                "task_checkpoints",
                "approval_requests",
                "observability_events",
                "api_idempotency_keys",
            }
            rows = self.store.db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            tables = {str(row["name"]) for row in rows}
            checks["schema"] = "ok" if required.issubset(tables) else "missing"
        except Exception:
            checks["sqlite"] = "unavailable"
            checks["schema"] = "unknown"
        return all(value in {"configured", "ok"} for value in checks.values()), checks


def build_orchestrator() -> Orchestrator:
    store = Store(os.getenv("AGENT_DB_PATH", "data/agent.sqlite3"))
    registry = ToolRegistry()
    registry.register(WebResearchTool(timeout=float(os.getenv("AGENT_HTTP_TIMEOUT", "15"))))
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
    principal = service.authenticator.authenticate(authorization)
    request.state.principal = principal.subject
    return principal


def _error_payload(request: Request, error: APIError) -> dict[str, Any]:
    return ErrorResponse(
        error_code=error.error_code,
        message=error.message,
        request_id=_request_id(request),
        task_id=error.task_id,
        approval_id=error.approval_id,
    ).model_dump()


def create_app(*, service: AgentAPIService | None = None, token: str | None = None) -> FastAPI:
    application = FastAPI(
        title="Private Self-Learning Agent API",
        version=API_VERSION,
        description="Stable authenticated interface over the existing Agent Core.",
    )
    application.state.agent_api_service = service
    application.state.agent_api_service_factory = lambda: build_api_service(token=token)

    @application.middleware("http")
    async def request_correlation(request: Request, call_next: Callable[..., Any]) -> Any:
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if _REQUEST_ID_PATTERN.fullmatch(supplied) else uuid.uuid4().hex
        request.state.request_id = request_id
        current_service = getattr(application.state, "agent_api_service", None)
        if current_service is None and (request.url.path.startswith("/api/") or request.url.path == "/tasks/run"):
            current_service = _service(request)
        if current_service is not None:
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

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.get("/ready", response_model=ReadinessResponse, responses={503: {"model": ErrorResponse}}, tags=["system"])
    def ready(request: Request, service: AgentAPIService = Depends(_service)) -> ReadinessResponse:
        ready_status, checks = service.readiness()
        response = ReadinessResponse(status="ready" if ready_status else "not_ready", checks=checks, request_id=_request_id(request))
        if not ready_status:
            raise APIError(503, "not_ready", "Required API dependencies are not ready")
        return response

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

    @router.post("/tasks", response_model=TaskCreateResponse, status_code=status.HTTP_201_CREATED)
    def create_task(
        request: Request,
        payload: TaskCreateRequest,
        service: AgentAPIService = Depends(_service),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> TaskCreateResponse:
        response = service.create_task(payload.goal, idempotency_key=idempotency_key, request_id=_request_id(request))
        request.state.task_id = response.task_id
        return response

    @router.get("/tasks/{task_id}", response_model=TaskStateResponse)
    def get_task(
        request: Request,
        task_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
    ) -> TaskStateResponse:
        request.state.task_id = task_id
        return service.get_task(task_id, request_id=_request_id(request))

    @router.post("/tasks/{task_id}/resume", response_model=TaskStateResponse)
    def resume_task(
        request: Request,
        payload: ResumeRequest | None = Body(default=None),
        task_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
    ) -> TaskStateResponse:
        request.state.task_id = task_id
        return service.resume_task(task_id, (payload.approval_ids if payload else {}), request_id=_request_id(request))

    @router.get("/tasks/{task_id}/timeline", response_model=list[EventResponse])
    def task_timeline(
        request: Request,
        task_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
    ) -> list[EventResponse]:
        request.state.task_id = task_id
        return service.task_timeline(task_id)

    @router.get("/tasks/{task_id}/metrics", response_model=MetricsResponse)
    def task_metrics(
        request: Request,
        task_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
    ) -> MetricsResponse:
        request.state.task_id = task_id
        return service.task_metrics(task_id)

    @router.get("/approvals", response_model=list[ApprovalResponse])
    def list_approvals(
        pending_only: bool = Query(default=True),
        service: AgentAPIService = Depends(_service),
    ) -> list[ApprovalResponse]:
        return service.list_approvals(pending_only=pending_only)

    @router.get("/approvals/{approval_id}", response_model=ApprovalResponse)
    def get_approval(
        approval_id: str = APIPath(..., min_length=1, max_length=256),
        service: AgentAPIService = Depends(_service),
    ) -> ApprovalResponse:
        return _approval_response(service._approval(approval_id))

    @router.post("/approvals/{approval_id}/approve", response_model=ApprovalResponse)
    def approve_approval(
        approval_id: str = APIPath(..., min_length=1, max_length=256),
        payload: ApprovalActionRequest | None = Body(default=None),
        service: AgentAPIService = Depends(_service),
    ) -> ApprovalResponse:
        del payload
        return service.approve(approval_id)

    @router.post("/approvals/{approval_id}/deny", response_model=ApprovalResponse)
    def deny_approval(
        approval_id: str = APIPath(..., min_length=1, max_length=256),
        payload: ApprovalActionRequest | None = Body(default=None),
        service: AgentAPIService = Depends(_service),
    ) -> ApprovalResponse:
        del payload
        return service.deny(approval_id)

    @router.get("/metrics", response_model=MetricsResponse)
    def system_metrics(service: AgentAPIService = Depends(_service)) -> MetricsResponse:
        return service.system_metrics()

    @router.get("/failures", response_model=list[EventResponse])
    def failures(
        request: Request,
        task_id: str | None = Query(default=None, max_length=256),
        service: AgentAPIService = Depends(_service),
    ) -> list[EventResponse]:
        if request is not None and task_id:
            request.state.task_id = task_id
        return service.failures(task_id)

    @router.get("/security-events", response_model=list[EventResponse])
    def security_events(
        request: Request,
        task_id: str | None = Query(default=None, max_length=256),
        service: AgentAPIService = Depends(_service),
    ) -> list[EventResponse]:
        if request is not None and task_id:
            request.state.task_id = task_id
        return service.security_events(task_id)

    @router.get("/tool-events", response_model=list[EventResponse])
    def tool_events(
        request: Request,
        task_id: str | None = Query(default=None, max_length=256),
        service: AgentAPIService = Depends(_service),
    ) -> list[EventResponse]:
        if request is not None and task_id:
            request.state.task_id = task_id
        return service.tool_events(task_id)

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

    application.include_router(router)

    # Keep the historical route protected and typed; it is a compatibility alias,
    # not a second execution implementation. New clients should use /api/v1/tasks.
    @application.post("/tasks/run", response_model=TaskCreateResponse, status_code=status.HTTP_201_CREATED, tags=["legacy"])
    def legacy_run(
        request: Request,
        payload: TaskCreateRequest,
        service: AgentAPIService = Depends(_service),
        principal: AuthenticatedPrincipal = Depends(_principal),
    ) -> TaskCreateResponse:
        del principal
        response = service.create_task(payload.goal, request_id=_request_id(request))
        request.state.task_id = response.task_id
        return response

    return application


app = create_app()

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
