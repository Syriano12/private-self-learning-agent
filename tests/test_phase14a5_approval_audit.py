from __future__ import annotations

from pathlib import Path

from private_agent.core.orchestrator import Orchestrator
from private_agent.core.task_state import TaskState
from private_agent.security import ActionRequest, SecurityController
from private_agent.skills.lifecycle import SkillCandidate, SkillLifecycleManager
from private_agent.storage import Store
from private_agent.core.recovery import input_fingerprint


def high_action(*, goal: str = "publish report", action_id: str = "action-1") -> ActionRequest:
    return ActionRequest(
        task_id="task-1",
        action_id=action_id,
        tool_or_skill="publisher",
        requested_capabilities=["NETWORK_ACCESS"],
        declared_capabilities=["NETWORK_ACCESS"],
        risk_hint="HIGH",
        input_fingerprint=input_fingerprint({"value": "report"}),
        reason="publish test report",
        planned_effect="controlled test effect",
        goal=goal,
    )


def test_allow_requires_current_policy_version_risk_and_exact_goal_binding(tmp_path: Path) -> None:
    store = Store(tmp_path / "approval-binding.sqlite3")
    controller = SecurityController(store=store)
    action = high_action()
    decision = controller.preflight(action)
    request = controller.approval_gate.get(decision.approval_id)
    assert request is not None
    controller.approval_gate.approve(request.approval_id)
    current = store.latest_owner_policy_version()

    assert controller.approval_gate.allow(
        request,
        action=action,
        policy_version=current,
        risk_level=decision.risk_level,
    ) is True
    assert controller.approval_gate.allow(
        request,
        action=high_action(goal="delete report"),
        policy_version=current,
        risk_level=decision.risk_level,
    ) is False

    store.create_owner_permission_rule(
        {
            "rule_id": "invalidate-approval",
            "effect": "DENY",
            "scope_type": "GOAL",
            "scope_key": "other goal",
            "conditions": {},
            "reason": "invalidate old approval",
        },
        actor="owner",
        reason="invalidate old approval",
    )
    newer = store.latest_owner_policy_version()
    assert newer > current
    assert controller.approval_gate.allow(
        request,
        action=action,
        policy_version=current,
        risk_level=decision.risk_level,
    ) is False
    assert controller.approval_gate.allow(
        request,
        action=action,
        policy_version=newer,
        risk_level=decision.risk_level,
    ) is False
    store.close()


def test_legacy_check_cannot_become_execution_authorization(tmp_path: Path) -> None:
    store = Store(tmp_path / "legacy-approval.sqlite3")
    controller = SecurityController(store=store)
    request = controller.approval_gate.check("publish")
    assert request is not None
    controller.approval_gate.approve(request.approval_id)
    legacy_action = ActionRequest(
        task_id="legacy",
        action_id="publish",
        tool_or_skill="publish",
        requested_capabilities=["SENSITIVE_DATA_ACCESS"],
        declared_capabilities=["SENSITIVE_DATA_ACCESS"],
        risk_hint="CRITICAL",
        input_fingerprint=input_fingerprint({"action": "publish"}),
        goal="",
    )
    assert controller.approval_gate.allow(
        request,
        action=legacy_action,
        policy_version=store.latest_owner_policy_version(),
        risk_level="CRITICAL",
    ) is False
    store.close()


def test_lifecycle_and_orchestrator_audit_use_effective_owner_policy_version(tmp_path: Path) -> None:
    store = Store(tmp_path / "audit-version.sqlite3")
    controller = SecurityController(store=store)
    store.create_owner_permission_rule(
        {
            "rule_id": "audit-rule",
            "effect": "ALLOW",
            "scope_type": "TOOL",
            "scope_key": "audit-tool",
            "conditions": {},
            "reason": "audit version test",
        },
        actor="owner",
        reason="audit version test",
    )
    current = store.latest_owner_policy_version()

    lifecycle = SkillLifecycleManager.__new__(SkillLifecycleManager)
    lifecycle.security = controller
    candidate = SkillCandidate("skill-1", "audit-skill", 1, "", "test", "run")
    lifecycle._audit_skill_event(candidate, "SKILL_ACTIVATED", "test audit")

    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.security = controller
    state = TaskState.create("audit-task", "audit goal")
    orchestrator._audit_task_event(state, "TASK_RESUMED", "test task audit")

    events = controller.audit.list()
    assert events[-2]["policy_version"] == current
    assert events[-1]["policy_version"] == current

    store.disable_owner_permission_rule("audit-rule", actor="owner", reason="disable after test")
    newer = store.latest_owner_policy_version()
    orchestrator._audit_task_event(state, "RESUME_BLOCKED", "after policy change")
    assert controller.audit.list()[-1]["policy_version"] == newer
    store.close()
