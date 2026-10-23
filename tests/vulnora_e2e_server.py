from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from private_agent.api import AgentAPIService, create_app
from private_agent.auth import AuthManager
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.planner import PlanStep, TaskPlan
from private_agent.observability import ObservabilityMonitor
from private_agent.security import SecurityController
from private_agent.storage import Store
from private_agent.tools.bugbounty import BugBountyAnalysisTool
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry
from private_agent.worker import BackgroundWorker


class VulnoraE2ETool:
    name = "vulnora_e2e_safe_tool"
    permission_level = "public_read"
    risk_level = "LOW"
    required_capabilities: list[str] = []
    declared_capabilities: list[str] = []
    network_required = False
    planned_effect = "Return a deterministic local verification value"

    def description(self) -> str:
        return "Deterministic local E2E tool"

    def input_schema(self) -> dict:
        return {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False}

    def output_schema(self) -> dict:
        return {"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"], "additionalProperties": False}

    def execute(self, inputs: dict, context) -> ToolResult:
        del context
        return ToolResult.success({"result": inputs["value"]})


class VulnoraE2EPlanner:
    provider = None
    max_steps = 12

    def create(self, goal: str, tools: ToolRegistry, prior_knowledge: list[dict]) -> TaskPlan:
        del prior_knowledge
        return TaskPlan(
            goal,
            [PlanStep("e2e-step-1", "Execute deterministic local verification", "vulnora_e2e_safe_tool", {"value": "verified-local-result"}, "E2E test plan", [], success_criteria={"required_fields": ["result"]})],
            "Deterministic E2E plan",
        )


def build_service(db_path: str, token: str | None) -> tuple[AgentAPIService, Store]:
    store = Store(Path(db_path))
    owner_password = os.getenv("VULNORA_E2E_OWNER_PASSWORD")
    if owner_password:
        owner_email = os.getenv("VULNORA_E2E_OWNER_EMAIL", "ashbm1995@gmail.com")
        AuthManager(store).initialize_owner(owner_email, owner_password)
    monitor = ObservabilityMonitor(store)
    security = SecurityController(store=store, observability=monitor)
    registry = ToolRegistry()
    registry.register(VulnoraE2ETool())
    registry.register(BugBountyAnalysisTool(store))
    core = Orchestrator(store, registry, planner=VulnoraE2EPlanner(), security_controller=security, observability=monitor, max_recovery_attempts=1)
    return AgentAPIService(core, token=token), store


if __name__ == "__main__":
    token = os.getenv("AGENT_API_TOKEN")
    service, store = build_service(os.environ["AGENT_DB_PATH"], token)
    worker = BackgroundWorker(service.orchestrator, worker_id="vulnora-e2e-worker", poll_interval_seconds=0.01, idle_poll_interval_seconds=0.02)
    application = create_app(service=service, worker=worker, require_worker=True)
    uvicorn.run(application, host="127.0.0.1", port=int(os.environ["VULNORA_E2E_PORT"]), log_level="error", access_log=False)
