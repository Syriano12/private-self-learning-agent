from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import pytest
import private_agent.security as security_module

from private_agent.core.execution import ExecutionContext, GenericExecutor
from private_agent.core.orchestrator import Orchestrator
from private_agent.observability import ObservabilityMonitor
from private_agent.security import ActionRequest, AuditTrail, OwnerPermissionRule, PolicyDecision, SecurityController
from private_agent.skills.lifecycle import MutationReport, MutationTestingEngine, SandboxResult, SkillLifecycleManager
from private_agent.skills.registry import SkillRegistry
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry
from private_agent.core.planner import PlanStep, TaskPlan


class CountingTool:
    name = "counting_tool"
    permission_level = "public_read"
    risk_level = "low"
    required_capabilities: list[str] = []
    declared_capabilities: list[str] = []

    def __init__(self) -> None:
        self.calls = 0

    def description(self) -> str:
        return "counting test tool"

    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    def output_schema(self) -> dict:
        return {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}

    def execute(self, inputs: dict, context: ExecutionContext) -> ToolResult:
        self.calls += 1
        return ToolResult.success({"ok": True})


def tool_registry(tool: CountingTool) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool)
    return registry


def tool_step() -> PlanStep:
    return PlanStep("integrity-step", "integrity", "counting_tool", {}, "test", [])


def trusted_execute(controller: SecurityController, action: ActionRequest, operation, **kwargs):
    binding = controller._bind_operation(action, operation, kind="test")
    return controller.execute_authorized(
        action,
        operation,
        operation_binding=binding,
        **kwargs,
    )


class StaticPlanner:
    provider = None
    max_steps = 12

    def create(self, goal: str, tools: ToolRegistry, prior_knowledge: list[dict]) -> TaskPlan:
        return TaskPlan(goal, [tool_step()], "integrity test plan")


def test_executor_rejects_caller_supplied_controller_that_is_not_trusted() -> None:
    tool = CountingTool()
    trusted = SecurityController()
    forged = SecurityController()
    result = GenericExecutor(tool_registry(tool), trusted).execute_step(
        tool_step(),
        ExecutionContext(task_id="controller-binding", security_controller=forged),
    )

    assert result.status == "blocked"
    assert result.error is not None
    assert result.error.code == "security_controller_mismatch"
    assert tool.calls == 0


def test_executor_without_trusted_controller_ignores_context_controller() -> None:
    tool = CountingTool()
    result = GenericExecutor(tool_registry(tool)).execute_step(
        tool_step(),
        ExecutionContext(task_id="untrusted-context", security_controller=SecurityController()),
    )

    assert result.status == "blocked"
    assert result.error is not None
    assert result.error.code == "security_not_configured"
    assert tool.calls == 0


def test_general_audit_record_rejects_direct_action_blocked_but_preserves_other_events() -> None:
    audit = AuditTrail()
    action = ActionRequest("direct-audit-api", "a", "tool", [], [])

    with pytest.raises(ValueError, match="action_blocked_requires_security_controller"):
        audit.record(
            "ACTION_BLOCKED",
            action,
            decision="DENY",
            policy_version=999,
            metadata={
                "authorized_policy_version": 999,
                "observed_current_policy_version": 999,
                "policy_version_source": "caller-controlled",
            },
        )
    with pytest.raises(ValueError, match="action_blocked_untrusted_writer"):
        audit._record("ACTION_BLOCKED", action, decision="DENY")
    with pytest.raises(ValueError, match="security_audit_claim_required"):
        audit._record_blocked(action, decision="DENY", capability=object())
    with pytest.raises(ValueError, match="security_audit_claim_required"):
        audit._record_blocked()
    assert getattr(audit, "_blocked_event_capability", None) is None
    assert audit.list(task_id=action.task_id) == []

    created = audit.record("TASK_CREATED", action, decision="CREATED", policy_version=1)
    assert created["event_type"] == "TASK_CREATED"


def test_public_audit_cannot_forge_execution_or_failure_claims() -> None:
    audit = AuditTrail()
    action = ActionRequest("direct-execution-audit", "a", "tool", [], [])

    for event_type, decision in (
        ("ACTION_EXECUTED", "ALLOW"),
        ("ACTION_FAILED", "FAILED"),
        ("POLICY_EVALUATED", "ALLOW"),
        ("PERMISSION_GRANTED", "ALLOW"),
        ("PERMISSION_DENIED", "DENY"),
        ("APPROVAL_REQUESTED", "PENDING"),
        ("APPROVAL_GRANTED", "APPROVED"),
        ("APPROVAL_DENIED", "DENIED"),
        ("APPROVAL_CANCELLED", "CANCELLED"),
        ("APPROVAL_EXPIRED", "EXPIRED"),
        ("APPROVAL_REVALIDATED", "ALLOW"),
        ("APPROVAL_INVALIDATED", "EXPIRED"),
        ("POLICY_CHANGED", "ALLOW"),
    ):
        with pytest.raises(ValueError, match="security_event_requires_trusted_writer"):
            audit.record(event_type, action, decision=decision, policy_version=888)
    assert audit.list(task_id=action.task_id) == []


def test_direct_security_blocked_writer_rejects_fabricated_fields_and_cross_instance_claims() -> None:
    controller = SecurityController()
    action = ActionRequest(
        "direct-blocked-writer",
        "step",
        "tool",
        ["UNKNOWN_CAPABILITY"],
        ["UNKNOWN_CAPABILITY"],
    )

    with pytest.raises(ValueError, match="security_audit_claim_required"):
        controller._record_blocked_event(
            action,
            decision="DENY",
            policy_version=999,
            metadata={"policy_version_source": "caller-controlled"},
        )

    captured: list[object] = []
    original_writer = controller.audit._write_security_claim
    controller.audit._write_security_claim = lambda claim: captured.append(claim) or {}
    decision = controller.authorize(action)
    assert decision.decision == "DENY"
    claim = captured.pop()
    controller.audit._write_security_claim = original_writer
    other_audit = AuditTrail()
    with pytest.raises(ValueError, match="security_audit_claim_untrusted"):
        other_audit._write_security_claim(claim)

    controller._record_blocked_event(claim)
    with pytest.raises(ValueError, match="security_audit_claim_untrusted"):
        controller._record_blocked_event(claim)
    assert [event for event in controller.audit.list(task_id=action.task_id) if event["event_type"] == "ACTION_BLOCKED"]


def test_security_claim_consumption_is_atomic_under_concurrent_writers() -> None:
    controller = SecurityController()
    action = ActionRequest(
        "concurrent-security-claim",
        "step",
        "tool",
        ["UNKNOWN_CAPABILITY"],
        ["UNKNOWN_CAPABILITY"],
    )
    captured: list[object] = []
    original_writer = controller.audit._write_security_claim
    controller.audit._write_security_claim = lambda claim: captured.append(claim) or {}
    assert controller.authorize(action).decision == "DENY"
    claim = captured.pop()
    controller.audit._write_security_claim = original_writer

    barrier = threading.Barrier(8)
    results: list[str] = []
    lock = threading.Lock()

    def write_claim() -> None:
        barrier.wait()
        try:
            controller.audit._write_security_claim(claim)
            result = "accepted"
        except ValueError as exc:
            result = f"rejected:{exc}"
        with lock:
            results.append(result)

    threads = [threading.Thread(target=write_claim) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert results.count("accepted") == 1
    assert results.count("rejected:security_audit_claim_untrusted") == 7
    assert len([event for event in controller.audit.list(task_id=action.task_id) if event["event_type"] == "ACTION_BLOCKED"]) == 1


def test_successful_audit_requires_a_real_execution_receipt() -> None:
    controller = SecurityController()
    tool = CountingTool()
    action = controller.action_for_tool("audit-task", "audit-step", tool, {})
    decision = controller.authorize(action, boundary=True)
    changed = controller.action_for_tool("audit-task", "audit-step", tool, {"unexpected": True})
    with pytest.raises(ValueError, match="execution_receipt_required"):
        controller.record_execution(changed, success=True, authorization=decision)
    operation = lambda: {"ok": True}
    execution = trusted_execute(controller, action, operation, authorization=decision)
    assert execution.receipt is not None
    executed = [event for event in controller.audit.list(task_id="audit-task") if event["event_type"] == "ACTION_EXECUTED"]
    assert len(executed) == 1
    with pytest.raises(ValueError, match="execution_receipt_untrusted"):
        controller.record_execution(execution.receipt)


def test_failed_execution_requires_a_real_execution_receipt() -> None:
    controller = SecurityController()
    tool = CountingTool()
    action = controller.action_for_tool("failed-audit-task", "failed-step", tool, {})
    with pytest.raises(ValueError, match="execution_receipt_required"):
        controller.record_execution(action, success=False)
    assert not controller.audit.list(task_id="failed-audit-task")
    operation = lambda: (_ for _ in ()).throw(RuntimeError("operation failed"))
    execution = trusted_execute(controller, action, operation)
    assert execution.exception is not None
    failed = [event for event in controller.audit.list(task_id="failed-audit-task") if event["event_type"] == "ACTION_FAILED"]
    assert len(failed) == 1
    assert failed[0]["decision"] == "FAILED"
    assert execution.receipt is not None
    with pytest.raises(ValueError, match="execution_receipt_untrusted"):
        controller.record_execution(execution.receipt)


def test_record_blocked_without_authorization_records_observed_only_version(tmp_path: Path) -> None:
    path = tmp_path / "blocked-observed-only.sqlite3"
    store = Store(path)
    controller = SecurityController(store=store)
    action = controller.action_for_tool("blocked-observed-task", "blocked-step", CountingTool(), {})
    store.create_owner_permission_rule(
        {
            "rule_id": "blocked-observed-version",
            "effect": "DENY",
            "scope_type": "TOOL",
            "scope_key": action.tool_or_skill,
            "conditions": {},
            "reason": "observed-only blocked audit",
        },
        actor="owner",
        reason="observed-only blocked audit",
    )
    current_version = store.latest_owner_policy_version()

    controller.record_blocked(
        action,
        reason="blocked without boundary authorization",
        metadata={
            "authorized_policy_version": 999,
            "observed_current_policy_version": 999,
            "policy_version_source": "caller-controlled",
        },
    )
    event = [event for event in controller.audit.list(task_id=action.task_id) if event["event_type"] == "ACTION_BLOCKED"][-1]
    assert event["policy_version"] == current_version
    assert event["metadata"]["authorized_policy_version"] is None
    assert event["metadata"]["observed_current_policy_version"] == current_version
    assert event["metadata"]["policy_version_source"] == "observed_current"
    store.close()

    reopened = Store(path)
    persisted = reopened.all_audit_events(task_id=action.task_id)[-1]
    assert persisted["metadata"]["authorized_policy_version"] is None
    assert persisted["metadata"]["observed_current_policy_version"] == current_version
    assert persisted["metadata"]["policy_version_source"] == "observed_current"
    reopened.close()


def test_record_blocked_uses_trusted_allow_snapshot_without_consuming_evidence() -> None:
    controller = SecurityController()
    tool = CountingTool()
    action = controller.action_for_tool("blocked-trusted-task", "blocked-step", tool, {})
    decision = controller.authorize(action, boundary=True)
    issued_version = decision.policy_version
    controller.record_blocked(action, reason="permission set blocked", authorization=decision)
    blocked = [event for event in controller.audit.list(task_id=action.task_id) if event["event_type"] == "ACTION_BLOCKED"][-1]
    assert blocked["policy_version"] == issued_version
    assert blocked["metadata"]["authorized_policy_version"] == issued_version
    assert blocked["metadata"]["observed_current_policy_version"] == issued_version
    assert blocked["metadata"]["policy_version_source"] == "trusted_authorization"
    with pytest.raises(ValueError, match="execution_receipt_required"):
        controller.record_execution(action, success=True, authorization=decision)
    operation = lambda: {"ok": True}
    execution = trusted_execute(controller, action, operation, authorization=decision)
    assert execution.receipt is not None
    assert len([event for event in controller.audit.list(task_id=action.task_id) if event["event_type"] == "ACTION_EXECUTED"]) == 1


def test_record_blocked_rejects_mutated_or_forged_allow_without_audit_event() -> None:
    controller = SecurityController()
    tool = CountingTool()
    action = controller.action_for_tool("blocked-invalid-task", "blocked-step", tool, {})
    decision = controller.authorize(action, boundary=True)
    decision.policy_version = 999

    with pytest.raises(ValueError, match="blocked_audit_authorization_mutated"):
        controller.record_blocked(action, reason="mutated allow", authorization=decision)
    assert not [event for event in controller.audit.list(task_id=action.task_id) if event["event_type"] == "ACTION_BLOCKED"]

    decision.policy_version = 1
    forged = PolicyDecision(
        decision="ALLOW",
        risk_level="LOW",
        task_id=action.task_id,
        action_id=action.action_id,
        tool_or_skill=action.tool_or_skill,
        requested_capabilities=list(action.requested_capabilities),
        policy_version=999,
        input_fingerprint=action.input_fingerprint,
        declared_capabilities=list(action.declared_capabilities),
        goal=action.goal,
        action_type=action.action_type,
        skill_status=action.skill_status,
    )
    with pytest.raises(ValueError, match="blocked_audit_authorization_untrusted"):
        controller.record_blocked(action, reason="forged allow", authorization=forged)
    assert not [event for event in controller.audit.list(task_id=action.task_id) if event["event_type"] == "ACTION_BLOCKED"]


def test_record_blocked_accepts_legitimate_deny_without_boundary_evidence(tmp_path: Path) -> None:
    store = Store(tmp_path / "blocked-deny-path.sqlite3")
    controller = SecurityController(store=store)
    tool = CountingTool()
    action = controller.action_for_tool("blocked-deny-task", "blocked-step", tool, {})
    store.create_owner_permission_rule(
        {
            "rule_id": "blocked-deny-rule",
            "effect": "DENY",
            "scope_type": "TOOL",
            "scope_key": tool.name,
            "conditions": {},
            "reason": "legitimate deny path",
        },
        actor="owner",
        reason="legitimate deny path",
    )
    decision = controller.authorize(action, boundary=True)
    assert decision.decision == "DENY"

    controller.record_blocked(action, reason="policy denied", authorization=decision)
    event = [event for event in controller.audit.list(task_id=action.task_id) if event["event_type"] == "ACTION_BLOCKED"][-1]
    assert event["metadata"]["authorized_policy_version"] is None
    assert event["metadata"]["policy_version_source"] == "observed_current"
    assert event["policy_version"] == store.latest_owner_policy_version()
    store.close()


def test_authorize_deny_and_quarantine_write_unified_observed_provenance(tmp_path: Path) -> None:
    path = tmp_path / "direct-blocked-provenance.sqlite3"
    store = Store(path)
    controller = SecurityController(store=store)
    tool = CountingTool()
    store.create_owner_permission_rule(
        OwnerPermissionRule("direct-deny", "DENY", "TOOL", tool.name, reason="direct deny"),
        actor="owner",
        reason="direct deny",
    )
    deny_action = controller.action_for_tool("direct-deny-task", "a", tool, {})
    deny = controller.authorize(deny_action, boundary=True)
    assert deny.decision == "DENY"

    quarantine_action = ActionRequest(
        task_id="direct-quarantine-task",
        action_id="a",
        tool_or_skill="skill-x",
        action_type="skill_invocation",
        skill_status="QUARANTINED",
    )
    quarantine = controller.authorize(quarantine_action, boundary=True)
    assert quarantine.decision == "QUARANTINE"

    for task_id in ("direct-deny-task", "direct-quarantine-task"):
        event = [event for event in controller.audit.list(task_id=task_id) if event["event_type"] == "ACTION_BLOCKED"][-1]
        assert event["metadata"]["authorized_policy_version"] is None
        assert event["metadata"]["observed_current_policy_version"] == event["policy_version"]
        assert event["metadata"]["policy_version_source"] == "observed_current"
    store.close()


def test_authorize_require_approval_and_observability_preserve_provenance(tmp_path: Path) -> None:
    path = tmp_path / "require-observability-provenance.sqlite3"
    store = Store(path)
    monitor = ObservabilityMonitor(store)
    controller = SecurityController(store=store, observability=monitor)
    action = ActionRequest(
        task_id="direct-require-task",
        action_id="a",
        tool_or_skill="network",
        requested_capabilities=["NETWORK_ACCESS"],
        declared_capabilities=["NETWORK_ACCESS"],
    )
    decision = controller.authorize(action, boundary=True)
    assert decision.decision == "REQUIRE_APPROVAL"

    audit_event = [event for event in controller.audit.list(task_id=action.task_id) if event["event_type"] == "ACTION_BLOCKED"][-1]
    assert audit_event["metadata"]["authorized_policy_version"] is None
    assert audit_event["metadata"]["observed_current_policy_version"] == audit_event["policy_version"]
    assert audit_event["metadata"]["policy_version_source"] == "observed_current"

    observed = [event for event in store.all_observability_events() if event["event_type"] == "ACTION_BLOCKED"][-1]
    assert observed["metadata"]["authorized_policy_version"] is None
    assert observed["metadata"]["observed_current_policy_version"] == audit_event["policy_version"]
    assert observed["metadata"]["policy_version_source"] == "observed_current"
    store.close()


def test_orchestrator_blocked_lifecycle_event_is_distinct_from_security_blocked(tmp_path: Path) -> None:
    path = tmp_path / "orchestrator-lifecycle-blocked.sqlite3"
    store = Store(path)
    monitor = ObservabilityMonitor(store)
    controller = SecurityController(store=store, observability=monitor)
    tool = CountingTool()
    agent = Orchestrator(
        store,
        tool_registry(tool),
        planner=StaticPlanner(),
        security_controller=controller,
        observability=monitor,
        approved_permissions={"private_write"},
    )

    result = agent.run("lifecycle blocked", task_id="orchestrator-lifecycle-blocked")
    assert result["final_status"] == "BLOCKED"
    assert tool.calls == 0

    security_events = [
        event for event in store.all_observability_events()
        if event["event_type"] == "ACTION_BLOCKED" and event["component"] == "security"
    ]
    lifecycle_events = [
        event for event in store.all_observability_events()
        if event["event_type"] == "ACTION_LIFECYCLE_BLOCKED"
    ]
    assert security_events
    assert lifecycle_events
    assert lifecycle_events[-1]["metadata"]["execution_status"] == "blocked"
    assert "authorized_policy_version" not in lifecycle_events[-1]["metadata"]
    assert "policy_version_source" not in lifecycle_events[-1]["metadata"]
    metrics = monitor.get_task_metrics("orchestrator-lifecycle-blocked")
    assert metrics["blocked_actions"] >= 1
    assert metrics["blocked_lifecycle_actions"] == 1
    store.close()

    reopened = Store(path)
    persisted_lifecycle = [
        event for event in reopened.all_observability_events()
        if event["event_type"] == "ACTION_LIFECYCLE_BLOCKED"
    ]
    assert persisted_lifecycle[-1]["metadata"]["execution_status"] == "blocked"
    assert "policy_version_source" not in persisted_lifecycle[-1]["metadata"]
    reopened_metrics = ObservabilityMonitor(reopened).get_task_metrics("orchestrator-lifecycle-blocked")
    assert reopened_metrics["blocked_lifecycle_actions"] == 1
    reopened.close()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("decision", "DENY"), ("policy_version", 999), ("goal", "tampered-goal")],
)
def test_decision_mutation_cannot_be_used_as_a_receipt(field: str, replacement: object) -> None:
    controller = SecurityController()
    tool = CountingTool()
    action = controller.action_for_tool("sequential-mutation-task", "sequential-mutation-step", tool, {})
    decision = controller.authorize(action, boundary=True)
    original = getattr(decision, field)
    setattr(decision, field, replacement)
    operation = lambda: {"ok": True}
    if field == "decision":
        blocked = trusted_execute(controller, action, operation, authorization=decision)
        assert blocked.receipt is None
        assert not [event for event in controller.audit.list(task_id=action.task_id) if event["event_type"] == "ACTION_EXECUTED"]
    else:
        with pytest.raises(ValueError, match="execution_authorization_mutated"):
            trusted_execute(controller, action, operation, authorization=decision)
    setattr(decision, field, original)
    execution = trusted_execute(controller, action, operation, authorization=decision)
    assert execution.receipt is not None
    assert len([event for event in controller.audit.list(task_id="sequential-mutation-task") if event["event_type"] == "ACTION_EXECUTED"]) == 1


def test_concurrent_execution_receipt_creation_accepts_one_operation_only() -> None:
    controller = SecurityController()
    tool = CountingTool()
    action = controller.action_for_tool("concurrent-audit-task", "concurrent-step", tool, {})
    decision = controller.authorize(action, boundary=True)
    operation = lambda: {"ok": True}
    start = threading.Barrier(8)
    results: list[str] = []
    results_lock = threading.Lock()
    def consume() -> None:
        start.wait(timeout=5)
        try:
            execution = trusted_execute(controller, action, operation, authorization=decision)
            outcome = "accepted" if execution.receipt is not None else "blocked"
        except Exception as exc:
            outcome = f"{type(exc).__name__}:{exc}"
        with results_lock:
            results.append(outcome)
    threads = [threading.Thread(target=consume) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert results.count("accepted") == 1
    assert len(results) == 8
    assert sum(item.startswith("ValueError:execution_authorization_in_flight") or item.startswith("ValueError:execution_authorization_untrusted") for item in results) == 7
    executed = [event for event in controller.audit.list(task_id="concurrent-audit-task") if event["event_type"] == "ACTION_EXECUTED"]
    assert len(executed) == 1


def test_concurrent_execution_receipts_remain_one_shot_across_repeated_rounds() -> None:
    for round_number in range(20):
        controller = SecurityController()
        tool = CountingTool()
        action = controller.action_for_tool(f"repeated-task-{round_number}", "repeated-step", tool, {})
        decision = controller.authorize(action, boundary=True)
        operation = lambda: {"ok": True}
        start = threading.Barrier(4)
        results: list[str] = []
        results_lock = threading.Lock()
        def consume() -> None:
            start.wait(timeout=5)
            try:
                execution = trusted_execute(controller, action, operation, authorization=decision)
                outcome = "accepted" if execution.receipt is not None else "blocked"
            except Exception as exc:
                outcome = f"{type(exc).__name__}:{exc}"
            with results_lock:
                results.append(outcome)
        threads = [threading.Thread(target=consume) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        assert results.count("accepted") == 1
        assert len([event for event in controller.audit.list(task_id=f"repeated-task-{round_number}") if event["event_type"] == "ACTION_EXECUTED"]) == 1


def test_audit_failure_consumes_receipt_and_does_not_retry_side_effect() -> None:
    controller = SecurityController()
    tool = CountingTool()
    action = controller.action_for_tool("audit-failure-task", "audit-failure-step", tool, {})
    decision = controller.authorize(action, boundary=True)
    side_effects = {"count": 0}
    def operation() -> dict[str, bool]:
        side_effects["count"] += 1
        return {"ok": True}
    original_writer = controller.audit._write_security_claim
    controller.audit._write_security_claim = lambda claim: (_ for _ in ()).throw(RuntimeError("audit sink unavailable"))
    try:
        with pytest.raises(RuntimeError, match="audit sink unavailable"):
            trusted_execute(controller, action, operation, authorization=decision)
    finally:
        controller.audit._write_security_claim = original_writer
    assert side_effects["count"] == 1


def test_concurrent_decision_mutation_cannot_change_trusted_policy_version(tmp_path: Path) -> None:
    store = Store(tmp_path / "concurrent-decision-mutation.sqlite3")
    controller = SecurityController(store=store)
    tool = CountingTool()
    action = controller.action_for_tool("decision-mutation-task", "decision-mutation-step", tool, {})
    decision = controller.authorize(action, boundary=True)
    store.create_owner_permission_rule(
        {
            "rule_id": "decision-mutation-version-bump",
            "effect": "ALLOW",
            "scope_type": "GLOBAL",
            "scope_key": "*",
            "conditions": {},
            "reason": "bump policy during decision mutation test",
        },
        actor="owner",
        reason="bump policy during decision mutation test",
    )
    current_version = store.latest_owner_policy_version()
    signature_ready = threading.Event()
    mutation_done = threading.Event()
    original_signature = controller._authorization_signature
    mutation_started = False

    def synchronized_signature(value):
        nonlocal mutation_started
        signed = original_signature(value)
        if value is decision and not mutation_started:
            mutation_started = True
            signature_ready.set()
            assert mutation_done.wait(timeout=5)
        return signed

    controller._authorization_signature = synchronized_signature

    def mutate_decision() -> None:
        assert signature_ready.wait(timeout=5)
        decision.policy_version = current_version
        mutation_done.set()

    mutator = threading.Thread(target=mutate_decision)
    mutator.start()
    operation = lambda: {"ok": True}
    with pytest.raises(ValueError, match="execution_authorization_stale"):
        trusted_execute(controller, action, operation, authorization=decision)
    mutator.join(timeout=5)

    assert not [event for event in controller.audit.list(task_id="decision-mutation-task") if event["event_type"] == "ACTION_EXECUTED"]
    store.close()


def test_forged_policy_decision_cannot_create_successful_audit() -> None:
    controller = SecurityController()
    tool = CountingTool()
    action = controller.action_for_tool("forged-task", "forged-step", tool, {})
    current = controller.policy_engine.evaluate(action)
    forged = PolicyDecision(
        decision="ALLOW",
        risk_level=current.risk_level,
        task_id=action.task_id,
        action_id=action.action_id,
        tool_or_skill=action.tool_or_skill,
        requested_capabilities=list(action.requested_capabilities),
        policy_version=current.policy_version,
        input_fingerprint=action.input_fingerprint,
        declared_capabilities=list(action.declared_capabilities),
        goal=action.goal,
        action_type=action.action_type,
        skill_status=action.skill_status,
    )
    operation = lambda: {"ok": True}
    with pytest.raises(ValueError, match="execution_authorization_untrusted"):
        trusted_execute(controller, action, operation, authorization=forged)
    assert not [event for event in controller.audit.list(task_id="forged-task") if event["event_type"] == "ACTION_EXECUTED"]


def test_stale_authorization_is_rejected_without_success_audit(tmp_path: Path) -> None:
    store = Store(tmp_path / "stale-evidence.sqlite3")
    controller = SecurityController(store=store)
    tool = CountingTool()
    action = controller.action_for_tool("stale-task", "stale-step", tool, {})
    decision = controller.authorize(action, boundary=True)
    store.create_owner_permission_rule(
        {
            "rule_id": "stale-evidence-deny",
            "effect": "DENY",
            "scope_type": "TOOL",
            "scope_key": tool.name,
            "conditions": {},
            "reason": "invalidate stale evidence",
        },
        actor="owner",
        reason="invalidate stale evidence",
    )
    operation = lambda: {"ok": True}
    with pytest.raises(ValueError, match="execution_authorization_stale"):
        trusted_execute(controller, action, operation, authorization=decision)
    assert not [event for event in controller.audit.list(task_id="stale-task") if event["event_type"] == "ACTION_EXECUTED"]
    store.close()


def test_malformed_persisted_approval_fails_closed(tmp_path: Path) -> None:
    store = Store(tmp_path / "malformed-approval.sqlite3")
    controller = SecurityController(store=store)
    tool = CountingTool()
    store.create_owner_permission_rule(
        {
            "rule_id": "malformed-approval-require",
            "effect": "REQUIRE_APPROVAL",
            "scope_type": "TOOL",
            "scope_key": tool.name,
            "conditions": {},
            "reason": "malformed approval test",
        },
        actor="owner",
        reason="malformed approval test",
    )
    action = controller.action_for_tool("malformed-task", "malformed-step", tool, {})
    pending = controller.authorize(action, boundary=True)
    assert pending.decision == "REQUIRE_APPROVAL"
    request = controller.approval_gate.get(pending.approval_id)
    assert request is not None
    payload = request.to_dict()
    payload["expires_at"] = "not-a-timestamp"
    store.save_approval_request(payload)
    controller.approval_gate._requests.clear()
    rejected = controller.authorize(action, approval_id=request.approval_id, boundary=True)
    assert rejected.allowed is False
    assert rejected.decision == "REQUIRE_APPROVAL"
    assert controller.approval_gate.get(request.approval_id).status == "EXPIRED"
    store.close()


def test_execute_active_remains_quarantined_without_external_sandbox(tmp_path: Path) -> None:
    from test_phase14a7_fail_closed import make_candidate

    store = Store(tmp_path / "active-approval-variants.sqlite3")
    controller = SecurityController(store=store)
    lifecycle = SkillLifecycleManager(
        SkillRegistry(store),
        mutation_engine=MutationTestingEngine(max_mutants=1, threshold=1.0),
        security_controller=controller,
    )
    candidate = make_candidate()
    submitted = lifecycle.submit(candidate, inputs={}, goal="runtime-goal")
    assert submitted.approval.status == "QUARANTINED"
    assert submitted.sandbox.error == "external_sandbox_required"
    lifecycle.registry.persist_candidate(candidate, status="ACTIVE")
    candidate.status = "ACTIVE"

    class SpySandbox:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, candidate, contract, inputs):
            self.calls += 1
            return SandboxResult("PASSED", output={"result": "active"})

    spy = SpySandbox()
    lifecycle.sandbox = spy
    store.create_owner_permission_rule(
        {
            "rule_id": "runtime-require-approval",
            "effect": "REQUIRE_APPROVAL",
            "scope_type": "SKILL",
            "scope_key": candidate.name,
            "conditions": {},
            "reason": "runtime approval test",
        },
        actor="owner",
        reason="runtime approval test",
    )

    missing = lifecycle.execute_active(candidate.name, {"value": 1}, task_id="runtime-task", goal="runtime-goal")
    assert missing.status == "BLOCKED"
    assert spy.calls == 0

    mismatched_action = controller.action_for_skill(
        "runtime-task", "mismatch", candidate, {"value": 1}, goal="runtime-goal"
    )
    mismatched_pending = controller.authorize(mismatched_action, boundary=True)
    controller.approval_gate.approve(mismatched_pending.approval_id, actor="owner")
    mismatched = lifecycle.execute_active(
        candidate.name,
        {"value": 2},
        task_id="runtime-task",
        action_id="mismatch",
        goal="runtime-goal",
        approval_id=mismatched_pending.approval_id,
    )
    assert mismatched.status == "BLOCKED"
    assert spy.calls == 0

    valid_action = controller.action_for_skill(
        "runtime-task", "valid", candidate, {"value": 3}, goal="runtime-goal"
    )
    valid_pending = controller.authorize(valid_action, boundary=True)
    controller.approval_gate.approve(valid_pending.approval_id, actor="owner")
    valid = lifecycle.execute_active(
        candidate.name,
        {"value": 3},
        task_id="runtime-task",
        action_id="valid",
        goal="runtime-goal",
        approval_id=valid_pending.approval_id,
    )
    assert valid.status == "QUARANTINED"
    assert valid.sandbox.error == "untrusted_sandbox_provider_not_supported"
    assert spy.calls == 0

    stale_action = controller.action_for_skill(
        "runtime-task", "stale", candidate, {"value": 4}, goal="runtime-goal"
    )
    stale_pending = controller.authorize(stale_action, boundary=True)
    controller.approval_gate.approve(stale_pending.approval_id, actor="owner")
    store.create_owner_permission_rule(
        {
            "rule_id": "runtime-version-bump",
            "effect": "ALLOW",
            "scope_type": "TOOL",
            "scope_key": "unrelated-tool",
            "conditions": {},
            "reason": "bump policy version for stale approval test",
        },
        actor="owner",
        reason="bump policy version for stale approval test",
    )
    stale = lifecycle.execute_active(
        candidate.name,
        {"value": 4},
        task_id="runtime-task",
        action_id="stale",
        goal="runtime-goal",
        approval_id=stale_pending.approval_id,
    )
    assert stale.status in {"BLOCKED", "QUARANTINED"}
    assert spy.calls == 0
    store.close()


def test_policy_change_during_tool_execution_is_detected_before_success_audit(tmp_path: Path) -> None:
    store = Store(tmp_path / "authorization-execution-window.sqlite3")

    class PolicyChangingTool(CountingTool):
        name = "policy_changing_tool"

        def execute(self, inputs: dict, context: ExecutionContext) -> ToolResult:
            self.calls += 1
            store.create_owner_permission_rule(
                {
                    "rule_id": "mid-execution-deny",
                    "effect": "DENY",
                    "scope_type": "TOOL",
                    "scope_key": self.name,
                    "conditions": {},
                    "reason": "deterministic authorization window test",
                },
                actor="owner",
                reason="deterministic authorization window test",
            )
            return ToolResult.success({"ok": True})

    tool = PolicyChangingTool()
    controller = SecurityController(store=store)
    with pytest.raises(ValueError, match="execution_authorization_stale"):
        GenericExecutor(tool_registry(tool), controller).execute_step(
            PlanStep("window-step", "window", tool.name, {}, "test", []),
            ExecutionContext(task_id="window-task", security_controller=controller),
        )
    assert tool.calls == 1
    assert not [event for event in controller.audit.list(task_id="window-task") if event["event_type"] == "ACTION_EXECUTED"]
    store.close()


def test_skill_goal_deny_blocks_execute_active_without_sandbox_call(tmp_path: Path) -> None:
    from test_phase14a7_fail_closed import make_candidate

    store = Store(tmp_path / "skill-goal-boundary.sqlite3")
    controller = SecurityController(store=store)
    lifecycle = SkillLifecycleManager(
        SkillRegistry(store),
        mutation_engine=MutationTestingEngine(max_mutants=1, threshold=1.0),
        security_controller=controller,
    )
    candidate = make_candidate()
    submitted = lifecycle.submit(candidate, inputs={}, goal="allowed-goal")
    assert submitted.approval.status == "QUARANTINED"
    assert submitted.sandbox.error == "external_sandbox_required"
    lifecycle.registry.persist_candidate(candidate, status="ACTIVE")

    class SpySandbox:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, candidate, contract, inputs):
            self.calls += 1
            return SandboxResult("PASSED", output={"result": "active"})

    spy = SpySandbox()
    lifecycle.sandbox = spy
    store.create_owner_permission_rule(
        {
            "rule_id": "deny-blocked-goal",
            "effect": "DENY",
            "scope_type": "GOAL",
            "scope_key": "blocked-goal",
            "conditions": {},
            "reason": "goal binding regression",
        },
        actor="owner",
        reason="goal binding regression",
    )

    blocked = lifecycle.execute_active("fail_closed_skill", {}, task_id="goal-task", goal="blocked-goal")
    assert blocked.status == "BLOCKED"
    assert blocked.sandbox.error is not None and "DENY" in blocked.sandbox.error
    assert spy.calls == 0

    allowed = lifecycle.execute_active("fail_closed_skill", {}, task_id="goal-task", goal="allowed-goal")
    assert allowed.status == "QUARANTINED"
    assert allowed.sandbox.error == "untrusted_sandbox_provider_not_supported"
    assert spy.calls == 0
    store.close()


def test_skill_activation_approval_binds_inputs_and_goal(tmp_path: Path, monkeypatch) -> None:
    from test_phase14a7_fail_closed import make_candidate

    store = Store(tmp_path / "skill-approval-binding.sqlite3")
    controller = SecurityController(store=store)
    lifecycle = SkillLifecycleManager(
        SkillRegistry(store),
        mutation_engine=MutationTestingEngine(max_mutants=1, threshold=1.0),
        security_controller=controller,
    )
    candidate = make_candidate()
    store.create_owner_permission_rule(
        {
            "rule_id": "approve-skill-goal",
            "effect": "REQUIRE_APPROVAL",
            "scope_type": "GOAL",
            "scope_key": "approval-goal",
            "conditions": {},
            "reason": "explicit approval binding test",
        },
        actor="owner",
        reason="explicit approval binding test",
    )
    action = controller.action_for_skill(
        "approval-task",
        "invoke:fail_closed_skill:1",
        candidate,
        {"value": 1},
        goal="approval-goal",
    )
    pending = controller.authorize(action, boundary=True)
    assert pending.approval_id
    controller.approval_gate.approve(pending.approval_id, actor="owner")

    passed_mutation = MutationReport(
        True,
        1,
        1,
        0,
        1.0,
        lifecycle.mutation_engine.threshold,
        reason="trusted_static_test_gate",
    )
    monkeypatch.setattr(
        lifecycle.mutation_engine,
        "run",
        lambda _candidate, _runner=None: passed_mutation,
    )

    rejected = lifecycle.submit(candidate, inputs={"value": 2}, goal="approval-goal", approval_id=pending.approval_id)
    assert rejected.approval.approved is False
    assert rejected.sandbox.status == "BLOCKED"
    assert rejected.sandbox.error.startswith("security_policy_blocked:")
    store.close()



def _capture_security_claims(controller: SecurityController, operation):
    captured: list[object] = []
    original_writer = controller.audit._write_security_claim
    controller.audit._write_security_claim = lambda claim: captured.append(claim) or {}
    try:
        result = operation()
    finally:
        controller.audit._write_security_claim = original_writer
    return result, captured


def test_old_claim_registration_callable_cannot_accept_dataclasses_replace_forgery() -> None:
    controller = SecurityController()
    action = ActionRequest("old-register-route", "step", "tool", [], [])
    _, claims = _capture_security_claims(controller, lambda: controller.authorize(action, boundary=True))
    legitimate = next(claim for claim in claims if claim.event_type == "PERMISSION_GRANTED")
    forged = __import__("dataclasses").replace(
        legitimate,
        claim_id="forged-registration-claim",
        issuance_nonce=object(),
    )

    register = getattr(controller.audit, "_claim_register", None)
    assert register is None or not callable(register)
    with pytest.raises((AttributeError, ValueError), match="security_audit_claim_untrusted|security_state_immutable|cannot"):
        if callable(register):
            register(forged)
        else:
            controller.audit._write_security_claim(forged)
    assert not [event for event in controller.audit.list(task_id=action.task_id) if event["event_type"] == "PERMISSION_GRANTED"]


@pytest.mark.parametrize(
    ("event_type", "decision"),
    [
        ("PERMISSION_GRANTED", "ALLOW"),
        ("ACTION_BLOCKED", "DENY"),
        ("APPROVAL_REQUESTED", "PENDING"),
        ("ACTION_EXECUTED", "ALLOW"),
        ("ACTION_FAILED", "FAILED"),
    ],
)
def test_dataclasses_replace_forgery_is_rejected_for_every_security_event_type(event_type: str, decision: str) -> None:
    controller = SecurityController()
    action = ActionRequest(f"replace-{event_type}", "step", "tool", [], [])

    def issue_real_claim():
        if event_type == "PERMISSION_GRANTED":
            return controller.authorize(action, boundary=True)
        if event_type == "ACTION_BLOCKED":
            denied = ActionRequest(action.task_id, action.action_id, action.tool_or_skill, ["UNKNOWN_CAPABILITY"], ["UNKNOWN_CAPABILITY"])
            return controller.authorize(denied, boundary=True)
        if event_type == "APPROVAL_REQUESTED":
            approval_action = ActionRequest(action.task_id, action.action_id, action.tool_or_skill, ["NETWORK_ACCESS"], ["NETWORK_ACCESS"], risk_hint="HIGH")
            return controller.authorize(approval_action, boundary=True)
        if event_type == "ACTION_EXECUTED":
            operation = lambda: {"ok": True}
            decision = controller.authorize(action, boundary=True)
            binding = controller._bind_operation(action, operation, kind="test")
            return controller.execute_authorized(action, operation, authorization=decision, operation_binding=binding)
        failing = ActionRequest(action.task_id, action.action_id, action.tool_or_skill, [], [])
        operation = lambda: (_ for _ in ()).throw(RuntimeError("failed"))
        binding = controller._bind_operation(failing, operation, kind="test")
        return controller.execute_authorized(failing, operation, operation_binding=binding)

    _, claims = _capture_security_claims(controller, issue_real_claim)
    legitimate = next(claim for claim in claims if claim.event_type == event_type)
    forged = __import__("dataclasses").replace(legitimate, claim_id=f"forged-{event_type}", issuance_nonce=object())
    register = getattr(controller.audit, "_claim_register", None)
    if callable(register):
        register(forged)
    with pytest.raises(ValueError, match="security_audit_claim_untrusted"):
        controller.audit._write_security_claim(forged)


def test_fabricated_execution_receipt_with_forged_state_is_rejected() -> None:
    controller = SecurityController()
    action = ActionRequest("fake-receipt", "step", "tool", [], [])
    decision = controller.authorize(action, boundary=True)
    fake = security_module.ExecutionReceipt(
        receipt_id="fake-receipt-id",
        authorization_id=id(decision),
        action_snapshot=security_module._action_snapshot(action),
        authorization=security_module.AuthorizationEvidenceSnapshot.from_decision(decision),
        success=True,
        metadata_snapshot=security_module._freeze_value({}),
        _state=lambda *_args, **_kwargs: True,
    )

    with pytest.raises(ValueError, match="execution_receipt_untrusted|execution_receipt_provenance_mismatch|security_audit_provenance_untrusted"):
        controller.audit._issue_security_claim(
            controller.audit._claim_issuer,
            "ACTION_EXECUTED",
            action,
            decision="ALLOW",
            reason="fake receipt",
            risk_level=decision.risk_level,
            policy_version=decision.policy_version,
            execution_evidence=fake,
            provenance=fake,
        )
    assert not [event for event in controller.audit.list(task_id=action.task_id) if event["event_type"] == "ACTION_EXECUTED"]


def test_unbound_noop_callable_does_not_produce_trusted_execution_event() -> None:
    controller = SecurityController()
    action = ActionRequest("unbound-noop", "step", "tool", [], [])

    with pytest.raises(ValueError, match="trusted_operation_binding_required"):
        controller.execute_authorized(action, lambda: {"ok": True})
    assert not [event for event in controller.audit.list(task_id=action.task_id) if event["event_type"] == "ACTION_EXECUTED"]


def test_receipt_replay_cannot_create_a_second_trusted_execution_event() -> None:
    controller = SecurityController()
    action = ActionRequest("receipt-replay", "step", "tool", [], [])
    operation_calls = 0

    def operation() -> dict[str, bool]:
        nonlocal operation_calls
        operation_calls += 1
        return {"ok": True}

    execution = trusted_execute(controller, action, operation)
    assert execution.receipt is not None
    assert operation_calls == 1
    assert len([event for event in controller.audit.list(task_id=action.task_id) if event["event_type"] == "ACTION_EXECUTED"]) == 1

    with pytest.raises(ValueError, match="execution_receipt_untrusted"):
        controller.record_execution(execution.receipt)

    assert operation_calls == 1
    assert len([event for event in controller.audit.list(task_id=action.task_id) if event["event_type"] == "ACTION_EXECUTED"]) == 1


def test_cross_action_receipt_substitution_cannot_audit_action_b() -> None:
    controller = SecurityController()
    action_a = ActionRequest("cross-action", "action-a", "tool", [], [])
    action_b = ActionRequest("cross-action", "action-b", "tool", [], [])
    execution = trusted_execute(controller, action_a, lambda: {"ok": True})
    assert execution.receipt is not None

    substituted = replace(
        execution.receipt,
        action_snapshot=security_module._action_snapshot(action_b),
    )
    with pytest.raises(ValueError, match="execution_receipt_untrusted"):
        controller.record_execution(substituted)

    assert not [event for event in controller.audit.list(task_id=action_b.task_id) if event["action_id"] == action_b.action_id]
    assert len([event for event in controller.audit.list(task_id=action_a.task_id) if event["event_type"] == "ACTION_EXECUTED"]) == 1


def test_cross_tool_receipt_substitution_cannot_audit_tool_b() -> None:
    controller = SecurityController()
    action_a = ActionRequest("cross-tool", "step", "tool-a", [], [])
    action_b = ActionRequest("cross-tool", "step", "tool-b", [], [])
    execution = trusted_execute(controller, action_a, lambda: {"ok": True})
    assert execution.receipt is not None

    substituted = replace(
        execution.receipt,
        action_snapshot=security_module._action_snapshot(action_b),
    )
    with pytest.raises(ValueError, match="execution_receipt_untrusted"):
        controller.record_execution(substituted)

    assert not [event for event in controller.audit.list(task_id=action_b.task_id) if event["tool_or_skill"] == "tool-b"]
    assert len([event for event in controller.audit.list(task_id=action_a.task_id) if event["event_type"] == "ACTION_EXECUTED"]) == 1


def test_cross_controller_receipt_is_rejected_without_audit_in_controller_b() -> None:
    controller_a = SecurityController()
    controller_b = SecurityController()
    action = ActionRequest("cross-controller", "step", "tool", [], [])
    execution = trusted_execute(controller_a, action, lambda: {"ok": True})
    assert execution.receipt is not None

    with pytest.raises(ValueError, match="execution_receipt_untrusted"):
        controller_b.record_execution(execution.receipt)

    assert controller_b.audit.list(task_id=action.task_id) == []
    assert len([event for event in controller_a.audit.list(task_id=action.task_id) if event["event_type"] == "ACTION_EXECUTED"]) == 1


def test_cross_action_operation_binding_blocks_before_operation_side_effect() -> None:
    controller = SecurityController()
    action_a = ActionRequest("binding-cross-action", "action-a", "tool", [], [])
    action_b = ActionRequest("binding-cross-action", "action-b", "tool", [], [])
    decision = controller.authorize(action_a, boundary=True)
    operation_a = lambda: {"ok": True}
    binding_a = controller._bind_operation(action_a, operation_a, kind="test")
    calls = 0

    def operation_b() -> dict[str, bool]:
        nonlocal calls
        calls += 1
        return {"ok": True}

    with pytest.raises(ValueError, match="trusted_operation_binding_required"):
        controller.execute_authorized(
            action_b,
            operation_b,
            authorization=decision,
            operation_binding=binding_a,
        )

    assert calls == 0
    assert not [event for event in controller.audit.list(task_id=action_b.task_id) if event["event_type"] == "ACTION_EXECUTED"]


def test_cross_tool_operation_binding_blocks_before_operation_side_effect() -> None:
    controller = SecurityController()
    action_a = ActionRequest("binding-cross-tool", "step", "tool-a", [], [])
    action_b = ActionRequest("binding-cross-tool", "step", "tool-b", [], [])
    decision = controller.authorize(action_a, boundary=True)
    operation_a = lambda: {"ok": True}
    binding_a = controller._bind_operation(action_a, operation_a, kind="test")
    calls = 0

    def operation_b() -> dict[str, bool]:
        nonlocal calls
        calls += 1
        return {"ok": True}

    with pytest.raises(ValueError, match="trusted_operation_binding_required"):
        controller.execute_authorized(
            action_b,
            operation_b,
            authorization=decision,
            operation_binding=binding_a,
        )

    assert calls == 0
    assert not [event for event in controller.audit.list(task_id=action_b.task_id) if event["tool_or_skill"] == "tool-b"]


def test_security_authorities_cannot_be_replaced_after_controller_construction() -> None:
    controller = SecurityController()
    original_audit = controller.audit
    original_policy_engine = controller.policy_engine
    original_approval_gate = controller.approval_gate

    for attribute, replacement in (
        ("audit", AuditTrail()),
        ("policy_engine", object()),
        ("approval_gate", object()),
    ):
        with pytest.raises(AttributeError, match="security_state_immutable"):
            setattr(controller, attribute, replacement)

    assert controller.audit is original_audit
    assert controller.policy_engine is original_policy_engine
    assert controller.approval_gate is original_approval_gate


def test_policy_engine_evaluation_without_boundary_provenance_cannot_write_permission_grant() -> None:
    controller = SecurityController()
    action = ActionRequest("evaluation-only", "step", "tool", [], [])
    evaluated = controller.policy_engine.evaluate(action)

    with pytest.raises(ValueError, match="security_audit_provenance_untrusted"):
        controller.audit._issue_security_claim(
            controller.audit._claim_issuer,
            "PERMISSION_GRANTED",
            action,
            decision="ALLOW",
            reason="evaluation only",
            risk_level=evaluated.risk_level,
            policy_version=evaluated.policy_version,
            provenance=evaluated,
        )

    assert controller.audit.list(task_id=action.task_id) == []


def test_same_id_claim_substitution_is_rejected_without_security_event() -> None:
    controller = SecurityController()
    action = ActionRequest("same-id-claim", "step", "tool", [], [])
    _, claims = _capture_security_claims(controller, lambda: controller.authorize(action, boundary=True))
    legitimate = next(claim for claim in claims if claim.event_type == "PERMISSION_GRANTED")
    other_decision = SecurityController().authorize(action, boundary=True)
    substituted = replace(legitimate, provenance=other_decision)

    with pytest.raises(ValueError, match="security_audit_claim_untrusted"):
        controller.audit._write_security_claim(substituted)

    assert not [event for event in controller.audit.list(task_id=action.task_id) if event["event_type"] == "PERMISSION_GRANTED"]


@pytest.mark.parametrize(
    "event_type",
    ["PERMISSION_GRANTED", "ACTION_BLOCKED", "APPROVAL_REQUESTED", "ACTION_EXECUTED", "ACTION_FAILED"],
)
def test_claim_from_controller_a_is_rejected_by_controller_b_for_each_sensitive_event(event_type: str) -> None:
    controller_a = SecurityController()
    controller_b = SecurityController()
    base = ActionRequest(f"issuer-cross-controller-{event_type}", "step", "tool", [], [])

    def issue_real_claim() -> None:
        if event_type == "PERMISSION_GRANTED":
            controller_a.authorize(base, boundary=True)
        elif event_type == "ACTION_BLOCKED":
            controller_a.authorize(
                ActionRequest(base.task_id, base.action_id, base.tool_or_skill, ["UNKNOWN_CAPABILITY"], ["UNKNOWN_CAPABILITY"]),
                boundary=True,
            )
        elif event_type == "APPROVAL_REQUESTED":
            controller_a.authorize(
                ActionRequest(
                    base.task_id,
                    base.action_id,
                    base.tool_or_skill,
                    ["NETWORK_ACCESS"],
                    ["NETWORK_ACCESS"],
                    risk_hint="HIGH",
                ),
                boundary=True,
            )
        elif event_type == "ACTION_EXECUTED":
            operation = lambda: {"ok": True}
            decision = controller_a.authorize(base, boundary=True)
            binding = controller_a._bind_operation(base, operation, kind="test")
            controller_a.execute_authorized(base, operation, authorization=decision, operation_binding=binding)
        else:
            operation = lambda: (_ for _ in ()).throw(RuntimeError("issuer test failure"))
            binding = controller_a._bind_operation(base, operation, kind="test")
            controller_a.execute_authorized(base, operation, operation_binding=binding)

    _, claims = _capture_security_claims(controller_a, issue_real_claim)
    claim = next(claim for claim in claims if claim.event_type == event_type)

    with pytest.raises(ValueError, match="security_audit_claim_untrusted"):
        controller_b.audit._write_security_claim(claim)

    assert controller_b.audit.list(task_id=base.task_id) == []


def test_local_issuer_rejects_raw_blocked_claim_without_provenance() -> None:
    controller = SecurityController()
    action = ActionRequest("raw-blocked-claim", "step", "tool", ["UNKNOWN_CAPABILITY"], ["UNKNOWN_CAPABILITY"])

    with pytest.raises(ValueError, match="security_audit_provenance_required"):
        controller.audit._issue_security_claim(
            controller.audit._claim_issuer,
            "ACTION_BLOCKED",
            action,
            decision="BLOCKED",
            reason="raw claim",
            risk_level="HIGH",
            policy_version=1,
            provenance=None,
        )

    assert controller.audit.list(task_id=action.task_id) == []


def test_action_executed_claim_requires_matching_execution_receipt() -> None:
    controller = SecurityController()
    action = ActionRequest("execution-proof", "step", "tool", [], [])
    decision = controller.authorize(action, boundary=True)

    with pytest.raises(ValueError, match="execution_receipt_required"):
        controller.audit._issue_security_claim(
            controller.audit._claim_issuer,
            "ACTION_EXECUTED",
            action,
            decision="ALLOW",
            reason="missing execution proof",
            risk_level=decision.risk_level,
            policy_version=decision.policy_version,
            provenance=decision,
        )

    assert not [event for event in controller.audit.list(task_id=action.task_id) if event["event_type"] == "ACTION_EXECUTED"]


def test_claim_metadata_is_snapshotted_before_caller_mutation() -> None:
    controller = SecurityController()
    action = ActionRequest("metadata-snapshot", "step", "tool", [], [])
    decision = controller.authorize(action, boundary=True)
    metadata = {"nested": {"value": "original"}}
    claim = controller.audit._issue_security_claim(
        controller.audit._claim_issuer,
        "PERMISSION_GRANTED",
        action,
        decision="ALLOW",
        reason="metadata snapshot",
        risk_level=decision.risk_level,
        policy_version=decision.policy_version,
        metadata=metadata,
        provenance=decision,
    )
    metadata["nested"]["value"] = "caller mutation"
    controller.audit._write_security_claim(claim)

    event = [event for event in controller.audit.list(task_id=action.task_id) if event["event_type"] == "PERMISSION_GRANTED"][-1]
    assert event["metadata"]["nested"]["value"] == "original"


def test_issuance_registry_injection_has_no_application_registry_to_mutate() -> None:
    controller = SecurityController()
    action = ActionRequest("registry-injection", "step", "tool", [], [])
    _, claims = _capture_security_claims(controller, lambda: controller.authorize(action, boundary=True))
    legitimate = next(claim for claim in claims if claim.event_type == "PERMISSION_GRANTED")
    claim = replace(
        legitimate,
        claim_id="forged-registry-claim",
        issuance_nonce=object(),
    )

    assert not hasattr(security_module, "_CLAIM_RECORDS")
    controller.audit._active_claims[claim.claim_id] = claim
    with pytest.raises(ValueError, match="security_audit_claim_untrusted"):
        controller.audit._write_security_claim(claim)
    assert controller.audit.list(task_id=action.task_id) == []


def test_audit_security_owner_cannot_be_rebound_after_controller_construction() -> None:
    controller = SecurityController()
    other = SecurityController()

    with pytest.raises(ValueError, match="security_owner_immutable"):
        controller.audit._bind_security_owner(other)
    with pytest.raises(AttributeError, match="security_owner_immutable"):
        controller.audit._security_owner = other

    assert controller.audit._security_owner is controller
