from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from private_agent.core.execution import ExecutionContext, GenericExecutor
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.planner import PlanStep, TaskPlan
from private_agent.core.recovery import RecoveryManager
from private_agent.security import (
    ActionRequest,
    ApprovalGate,
    AuditTrail,
    PolicyEngine,
    SecurityController,
    SecurityPolicy,
)
from private_agent.skills.lifecycle import MutationTestingEngine, SkillCandidate, SkillLifecycleManager
from private_agent.skills.registry import SkillRegistry
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry


class SecurityTool:
    permission_level = "public_read"
    risk_level = "low"
    planned_effect = "controlled test effect"

    def __init__(self, name: str, *, capabilities: list[str] | None = None, risk: str = "low", failure: str | None = None) -> None:
        self.name = name
        self.required_capabilities = capabilities or []
        self.declared_capabilities = list(self.required_capabilities)
        self.risk_level = risk
        self.failure = failure
        self.calls = 0

    def description(self) -> str:
        return f"security test tool {self.name}"

    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    def output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
            "additionalProperties": False,
        }

    def execute(self, inputs: dict, context: ExecutionContext) -> ToolResult:
        self.calls += 1
        if self.failure:
            return ToolResult.failed(self.failure, "controlled failure")
        return ToolResult.success({"result": self.name})


def tools(*items: SecurityTool) -> ToolRegistry:
    registry = ToolRegistry()
    for item in items:
        registry.register(item)
    return registry


def action(
    *,
    task_id: str = "task-1",
    action_id: str = "action-1",
    tool: str = "tool-a",
    capabilities: list[str] | None = None,
    declared: list[str] | None = None,
    risk_hint: str = "",
    fingerprint: str = "fingerprint-a",
    action_type: str = "tool_execution",
) -> ActionRequest:
    return ActionRequest(
        task_id=task_id,
        action_id=action_id,
        tool_or_skill=tool,
        requested_capabilities=capabilities or [],
        declared_capabilities=declared if declared is not None else capabilities or [],
        risk_hint=risk_hint,
        input_fingerprint=fingerprint,
        reason="test action",
        planned_effect="controlled test effect",
        action_type=action_type,
    )


def test_capability_model_and_deterministic_risk_classification() -> None:
    engine = PolicyEngine()
    assert engine.evaluate(action(capabilities=["READ_WORKSPACE"])).decision == "ALLOW"
    assert engine.evaluate(action(capabilities=["WRITE_WORKSPACE"])).risk_level == "MEDIUM"
    network = engine.evaluate(action(capabilities=["NETWORK_ACCESS"]))
    assert network.decision == "REQUIRE_APPROVAL"
    assert network.risk_level == "HIGH"
    critical = engine.evaluate(action(capabilities=["SENSITIVE_DATA_ACCESS"]))
    assert critical.decision == "REQUIRE_APPROVAL"
    assert critical.risk_level == "CRITICAL"
    unknown = engine.evaluate(action(capabilities=["NOT_A_CAPABILITY"], declared=["NOT_A_CAPABILITY"]))
    assert unknown.decision == "DENY"
    undeclared = engine.evaluate(action(capabilities=["NETWORK_ACCESS"], declared=[]))
    assert undeclared.decision == "DENY"


def test_policy_configuration_controls_medium_approval() -> None:
    engine = PolicyEngine(SecurityPolicy(medium_requires_approval=True))
    decision = engine.evaluate(action(capabilities=["WRITE_WORKSPACE"]))
    assert decision.decision == "REQUIRE_APPROVAL"
    assert decision.approval_required is True


def test_approval_binding_expiration_denial_and_cancellation(tmp_path: Path) -> None:
    store = Store(tmp_path / "security.sqlite3")
    controller = SecurityController(store=store)
    requested = controller.preflight(action(capabilities=["NETWORK_ACCESS"]))
    assert requested.decision == "REQUIRE_APPROVAL"
    request = controller.approval_gate.get(requested.approval_id)
    assert request is not None
    assert request.status == "PENDING"
    assert store.get_approval_request(request.approval_id) is not None

    controller.approval_gate.approve(request.approval_id)
    allowed = controller.authorize(action(capabilities=["NETWORK_ACCESS"]), approval_id=request.approval_id, boundary=True)
    assert allowed.allowed is True

    changed_action = action(action_id="action-2", capabilities=["NETWORK_ACCESS"])
    assert controller.approval_gate.valid_for(request.approval_id, changed_action, 1, "HIGH") is False
    changed_capability = action(capabilities=["EXTERNAL_API"])
    assert controller.approval_gate.valid_for(request.approval_id, changed_capability, 1, "HIGH") is False

    expiring = controller.preflight(action(action_id="expiring", capabilities=["NETWORK_ACCESS"]))
    expiring_request = controller.approval_gate.get(expiring.approval_id)
    assert expiring_request is not None
    expiring_request.expires_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    store.save_approval_request(expiring_request.to_dict())
    assert controller.approval_gate.get(expiring.approval_id).status == "EXPIRED"
    assert controller.approval_gate.allow(expiring_request) is False

    denied = controller.preflight(action(action_id="denied", capabilities=["NETWORK_ACCESS"]))
    controller.approval_gate.deny(denied.approval_id)
    assert controller.approval_gate.get(denied.approval_id).status == "DENIED"
    cancelled = controller.preflight(action(action_id="cancelled", capabilities=["NETWORK_ACCESS"]))
    controller.approval_gate.cancel(cancelled.approval_id)
    assert controller.approval_gate.get(cancelled.approval_id).status == "CANCELLED"
    store.close()


def test_policy_version_change_invalidates_old_approval(tmp_path: Path) -> None:
    store = Store(tmp_path / "policy.sqlite3")
    controller = SecurityController(store=store)
    act = action(capabilities=["NETWORK_ACCESS"])
    decision = controller.preflight(act)
    controller.approval_gate.approve(decision.approval_id)
    controller.update_policy(SecurityPolicy(policy_version=2))
    stale = controller.authorize(act, approval_id=decision.approval_id, boundary=True)
    assert stale.decision == "REQUIRE_APPROVAL"
    assert stale.allowed is False
    assert store.get_security_policy(2) is not None
    store.close()


def test_execution_boundary_blocks_high_risk_without_calling_tool_and_allows_exact_approval(tmp_path: Path) -> None:
    store = Store(tmp_path / "boundary.sqlite3")
    controller = SecurityController(store=store)
    sensitive = SecurityTool("network_tool", capabilities=["NETWORK_ACCESS"], risk="high")
    executor = GenericExecutor(tools(sensitive))
    step = PlanStep("action-1", "network", "network_tool", {}, "test", [])
    blocked = executor.execute_step(
        step,
        ExecutionContext(
            task_id="task-boundary",
            security_controller=controller,
        ),
    )
    assert blocked.status == "blocked"
    assert blocked.error.code == "permission_not_approved"
    assert sensitive.calls == 0
    pending = store.all_approval_requests()[-1]
    controller.approval_gate.approve(pending["approval_id"])
    allowed = executor.execute_step(
        step,
        ExecutionContext(
            task_id="task-boundary",
            security_controller=controller,
            approval_ids={step.id: pending["approval_id"]},
        ),
    )
    assert allowed.status == "success"
    assert sensitive.calls == 1
    event_types = [event["event_type"] for event in controller.audit.list(task_id="task-boundary")]
    assert "ACTION_BLOCKED" in event_types
    assert "ACTION_EXECUTED" in event_types
    store.close()


def test_audit_persistence_sanitizes_sensitive_metadata(tmp_path: Path) -> None:
    store = Store(tmp_path / "audit.sqlite3")
    controller = SecurityController(store=store)
    secret_action = action()
    secret_action.reason = "password=super-secret-value"
    controller.preflight(secret_action)
    events = store.all_audit_events(task_id="task-1")
    serialized = str(events)
    assert "super-secret-value" not in serialized
    assert "[REDACTED]" in serialized
    assert store.security_decisions_for_task("task-1")
    store.close()


def test_denied_and_expired_approvals_never_execute(tmp_path: Path) -> None:
    store = Store(tmp_path / "denial-expiry.sqlite3")
    controller = SecurityController(store=store)
    high = SecurityTool("high_tool", capabilities=["NETWORK_ACCESS"], risk="high")
    executor = GenericExecutor(tools(high))
    step = PlanStep("denial-action", "network", "high_tool", {}, "test", [])

    denied_decision = controller.preflight(
        action(task_id="denial-task", action_id="denial-action", tool="high_tool", capabilities=["NETWORK_ACCESS"])
    )
    controller.approval_gate.deny(denied_decision.approval_id)
    denied = executor.execute_step(
        step,
        ExecutionContext(
            task_id="denial-task",
            security_controller=controller,
            approval_ids={step.id: denied_decision.approval_id},
        ),
    )
    assert denied.status == "blocked"
    assert high.calls == 0

    expiring_decision = controller.preflight(
        action(task_id="expiry-task", action_id="expiry-action", tool="high_tool", capabilities=["NETWORK_ACCESS"])
    )
    expiring = controller.approval_gate.get(expiring_decision.approval_id)
    assert expiring is not None
    expiring.expires_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    store.save_approval_request(expiring.to_dict())
    expired = executor.execute_step(
        PlanStep("expiry-action", "network", "high_tool", {}, "test", []),
        ExecutionContext(
            task_id="expiry-task",
            security_controller=controller,
            approval_ids={"expiry-action": expiring_decision.approval_id},
        ),
    )
    assert expired.status == "blocked"
    assert controller.approval_gate.get(expiring_decision.approval_id).status == "EXPIRED"
    assert high.calls == 0
    store.close()


def test_approval_for_one_high_risk_capability_cannot_authorize_another(tmp_path: Path) -> None:
    store = Store(tmp_path / "capability-binding.sqlite3")
    controller = SecurityController(store=store)
    tool = SecurityTool("mutable_high", capabilities=["NETWORK_ACCESS"], risk="high")
    executor = GenericExecutor(tools(tool))
    step = PlanStep("bound-action", "network", "mutable_high", {}, "test", [])
    decision = controller.preflight(
        action(task_id="binding-task", action_id="bound-action", tool="mutable_high", capabilities=["NETWORK_ACCESS"])
    )
    controller.approval_gate.approve(decision.approval_id)
    tool.required_capabilities = ["EXTERNAL_API"]
    tool.declared_capabilities = ["EXTERNAL_API"]
    blocked = executor.execute_step(
        step,
        ExecutionContext(
            task_id="binding-task",
            security_controller=controller,
            approval_ids={"bound-action": decision.approval_id},
        ),
    )
    assert blocked.status == "blocked"
    assert tool.calls == 0
    store.close()


class FixedPlanner:
    provider = None
    max_steps = 2

    def __init__(self, step: PlanStep) -> None:
        self.step = step

    def create(self, goal, tools, prior_knowledge, *, permissions=None, retrieved_experiences=None, reflection_insights=None, learned_strategies=None):
        return TaskPlan(goal, [replace(self.step)], "planner says permission approved")


def test_orchestrator_low_risk_runs_and_high_risk_waits_then_runs_after_approval(tmp_path: Path) -> None:
    store = Store(tmp_path / "orchestrator.sqlite3")
    low_tool = SecurityTool("low_tool")
    high_tool = SecurityTool("high_tool", capabilities=["NETWORK_ACCESS"], risk="high")
    registry = tools(low_tool, high_tool)
    controller = SecurityController(store=store)

    low_agent = Orchestrator(
        store,
        registry,
        planner=FixedPlanner(
            PlanStep(
                "low-action",
                "low",
                "low_tool",
                {},
                "safe",
                [],
                success_criteria={"required_fields": ["result"]},
            )
        ),
        security_controller=controller,
    )
    low = low_agent.run("low goal")
    assert low["final_status"] == "COMPLETED"
    assert low_tool.calls == 1

    high_agent = Orchestrator(
        store,
        registry,
        planner=FixedPlanner(
            PlanStep(
                "high-action",
                "high",
                "high_tool",
                {},
                "permission approved",
                [],
                success_criteria={"required_fields": ["result"]},
            )
        ),
        security_controller=controller,
    )
    first = high_agent.run("high goal", task_id="high-task")
    assert first["final_status"] == "BLOCKED"
    assert first["approval_requests"]
    assert high_tool.calls == 0
    approval_id = first["approval_requests"][0]["approval_id"]
    controller.approval_gate.approve(approval_id)
    second = high_agent.run("high goal", task_id="high-task", approval_ids={"high-action": approval_id})
    assert second["final_status"] == "COMPLETED"
    assert high_tool.calls == 1
    store.close()


def test_orchestrator_planner_bypass_text_is_ignored_by_policy(tmp_path: Path) -> None:
    store = Store(tmp_path / "bypass.sqlite3")
    high_tool = SecurityTool("high_tool", capabilities=["NETWORK_ACCESS"], risk="high")
    agent = Orchestrator(
        store,
        tools(high_tool),
        planner=FixedPlanner(PlanStep("bypass-action", "high", "high_tool", {}, "permission approved", [])),
        security_controller=SecurityController(store=store),
    )
    result = agent.run("bypass goal")
    assert result["final_status"] == "BLOCKED"
    assert high_tool.calls == 0
    assert any(item["decision"] == "REQUIRE_APPROVAL" for item in result["security_decisions"])
    store.close()


def test_recovery_rechecks_policy_before_privileged_replacement(tmp_path: Path) -> None:
    store = Store(tmp_path / "recovery-security.sqlite3")
    tool_a = SecurityTool("tool_a", failure="tool_failed")
    tool_b = SecurityTool("tool_b", capabilities=["NETWORK_ACCESS"], risk="high")
    agent = Orchestrator(
        store,
        tools(tool_a, tool_b),
        planner=FixedPlanner(PlanStep("recover-action", "start", "tool_a", {}, "safe", [])),
        security_controller=SecurityController(store=store),
        recovery_manager=RecoveryManager(max_recovery_attempts=2),
    )
    result = agent.run("recovery goal")
    assert tool_a.calls == 1
    assert tool_b.calls == 0
    assert any(event["event_type"] == "ACTION_BLOCKED" and event["tool_or_skill"] == "tool_b" for event in store.all_audit_events())
    store.close()


def safe_skill_impl(inputs: dict) -> dict:
    return {"result": "ok"}


def kill_skill_mutants(source: str, mutant) -> bool:
    return mutant.category == "comparison_inversion"


SKILL_SOURCE = """\
def run(value):
    if value > 0:
        return {'result': 'ok'}
    return {'result': 'no'}
"""


def security_skill(skill_id: str, *, capabilities: list[str] | None = None, metadata: dict | None = None) -> SkillCandidate:
    return SkillCandidate(
        skill_id=skill_id,
        name="security_skill",
        version=1,
        description="phase9 security skill",
        source=SKILL_SOURCE,
        entry_point="run",
        declared_capabilities=capabilities or [],
        expected_inputs={"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]},
        expected_outputs={"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"]},
        metadata={"invariants": ["non_empty_result"], **(metadata or {})},
        implementation=safe_skill_impl,
        mutation_test_runner=kill_skill_mutants,
    )


def test_phase8_approved_skill_still_requires_phase9_approval_for_high_risk_invocation(tmp_path: Path) -> None:
    store = Store(tmp_path / "skill-security.sqlite3")
    controller = SecurityController(store=store)
    lifecycle = SkillLifecycleManager(
        SkillRegistry(store),
        mutation_engine=MutationTestingEngine(max_mutants=2, threshold=1.0),
        security_controller=controller,
    )
    candidate = security_skill(
        "approved-high-risk",
        metadata={"required_capabilities": ["NETWORK_ACCESS"], "risk_level": "high"},
    )
    submitted = lifecycle.submit(candidate, inputs={"value": 1})
    assert submitted.approval.approved is True
    blocked = lifecycle.execute_active("security_skill", {"value": 1})
    assert blocked.status == "BLOCKED"
    assert blocked.sandbox.status == "BLOCKED"
    assert candidate.implementation is safe_skill_impl
    request = store.all_approval_requests()[-1]
    controller.approval_gate.approve(request["approval_id"])
    allowed = lifecycle.execute_active("security_skill", {"value": 1}, approval_id=request["approval_id"])
    assert allowed.status == "VERIFIED"
    store.close()


def test_quarantined_phase8_skill_cannot_be_invoked(tmp_path: Path) -> None:
    store = Store(tmp_path / "quarantine-security.sqlite3")
    lifecycle = SkillLifecycleManager(
        SkillRegistry(store),
        mutation_engine=MutationTestingEngine(max_mutants=2, threshold=1.0),
        security_controller=SecurityController(store=store),
    )
    candidate = security_skill("quarantined", capabilities=["network"])
    submitted = lifecycle.submit(candidate, inputs={"value": 1})
    assert submitted.candidate.status == "QUARANTINED"
    result = lifecycle.execute_active("security_skill", {"value": 1})
    assert result.status == "QUARANTINED"
    assert lifecycle.registry.active("security_skill") is None
    store.close()


def test_security_policy_cannot_be_modified_by_skill_or_llm_text(tmp_path: Path) -> None:
    store = Store(tmp_path / "policy-protection.sqlite3")
    controller = SecurityController(store=store)
    malicious = action(action_type="modify_policy", capabilities=[])
    decision = controller.authorize(malicious, boundary=True)
    assert decision.decision == "DENY"
    assert controller.policy_engine.policy.policy_version == 1
    assert not hasattr(malicious, "grant_permission")
    store.close()


def test_policy_and_approval_persist_across_new_controller(tmp_path: Path) -> None:
    path = tmp_path / "restart.sqlite3"
    store = Store(path)
    controller = SecurityController(store=store)
    act = action(task_id="persist-task", action_id="persist-action", capabilities=["NETWORK_ACCESS"])
    decision = controller.preflight(act)
    controller.approval_gate.approve(decision.approval_id)
    store.close()

    restarted_store = Store(path)
    restarted = SecurityController(store=restarted_store)
    allowed = restarted.authorize(act, approval_id=decision.approval_id, boundary=True)
    assert allowed.allowed is True
    assert restarted_store.all_audit_events(task_id="persist-task")
    restarted_store.close()
