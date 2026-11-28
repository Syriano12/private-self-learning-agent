from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from private_agent.core.execution import ExecutionContext, GenericExecutor, StepResult
from private_agent.core.experience import ExperienceRecord
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.planner import Planner
from private_agent.core.specialists import SpecialistProfile, SpecialistRegistry
from private_agent.core.task_state import TaskState
from private_agent.security import PolicyEngine, SecurityController, SecurityPolicy
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry


class SyntheticPathTool:
    permission_level = "public_read"
    required_capabilities = ["READ_WORKSPACE"]
    declared_capabilities = ["READ_WORKSPACE"]
    risk_level = "low"
    network_required = False
    filesystem_required = True
    planned_effect = "Read only a synthetic repository fixture"

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[dict[str, Any]] = []

    def description(self) -> str:
        return f"Synthetic local fixture for {self.name}"

    def input_schema(self) -> dict[str, Any]:
        if self.name == "inspect_repository_structure":
            return {"type": "object", "properties": {"root": {"type": "string"}}, "required": ["root"], "additionalProperties": False}
        return {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}

    def output_schema(self) -> dict[str, Any]:
        if self.name == "inspect_repository_structure":
            return {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}
        return {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"], "additionalProperties": False}

    def execute(self, inputs: dict[str, Any], context: Any) -> ToolResult:
        del context
        self.calls.append(dict(inputs))
        if self.name == "inspect_repository_structure":
            return ToolResult.success({"path": "workspace/synthetic.txt"})
        return ToolResult.success({"content": "synthetic evidence"})


class CapturingSecurityController(SecurityController):
    def __init__(self, *, store: Any | None = None, policy_engine: PolicyEngine | None = None) -> None:
        super().__init__(store=store, policy_engine=policy_engine)
        self.last_specialist_binding: dict[str, Any] | None = None
        self.last_action: Any = None

    def action_for_tool(
        self,
        task_id: str,
        action_id: str,
        tool: Any,
        inputs: dict[str, Any],
        *,
        goal: str = "",
        specialist_binding: dict[str, Any] | None = None,
    ) -> Any:
        self.last_specialist_binding = dict(specialist_binding or {})
        self.last_action = super().action_for_tool(
            task_id, action_id, tool, inputs, goal=goal, specialist_binding=specialist_binding
        )
        return self.last_action


def _setup() -> tuple[ToolRegistry, SyntheticPathTool, SyntheticPathTool, SpecialistRegistry]:
    tools = ToolRegistry()
    source = SyntheticPathTool("inspect_repository_structure")
    reader = SyntheticPathTool("read_repository_file")
    tools.register(source)
    tools.register(reader)
    specialists = SpecialistRegistry()
    specialists.register(
        SpecialistProfile(
            specialist_id="local_repository_analyst",
            version=1,
            tool_names=frozenset({reader.name}),
            granted_capabilities=frozenset({"READ_WORKSPACE"}),
        )
    )
    return tools, source, reader, specialists


def _payload() -> dict[str, Any]:
    return {
        "goal": "Read a synthetic path produced by a verified dependency",
        "rationale": "Bounded synthetic input provenance fixture",
        "steps": [
            {"id": "source", "objective": "Produce one synthetic repository path", "tool": "inspect_repository_structure", "input": {"root": "workspace"}, "reason": "Synthetic source fixture", "depends_on": []},
            {
                "id": "reader",
                "objective": "Read the verified synthetic repository path",
                "tool": "read_repository_file",
                "input": {},
                "reason": "A bounded local analyst is appropriate",
                "depends_on": ["source"],
                "input_bindings": {"path": {"source_step_id": "source", "output_key": "path"}},
                "specialist_id": "local_repository_analyst",
            },
        ],
    }


def _context(*, verified: bool = True) -> ExecutionContext:
    return ExecutionContext(
        task_id="synthetic-specialist-binding-task",
        previous_results={"source": StepResult.success("source", "inspect_repository_structure", {"path": "workspace/synthetic.txt"})},
        approved_permissions={"public_read"},
        metadata={"goal": _payload()["goal"]},
        verified_result_ids={"source"} if verified else set(),
    )


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("legacy", [False, True])
def test_unbound_current_or_legacy_assignment_cannot_cover_dependency_input(legacy: bool) -> None:
    tools, _, reader, specialists = _setup()
    payload = _payload()
    plan = Planner(specialists=specialists).validate(payload, goal=payload["goal"], tools=tools)
    step = plan.steps[1]
    unbound_assignment = specialists.create_assignment(
        "local_repository_analyst", tool=reader, objective=step.objective, inputs=step.input
    )
    if legacy:
        unbound_assignment.pop("input_bindings_digest")
    step.specialist_assignment = unbound_assignment
    executor = GenericExecutor(tools, security_controller=SecurityController(), specialist_registry=specialists)

    result = executor.execute_step(step, _context())

    assert result.status == "blocked"
    assert result.error is not None and result.error.code == "specialist_assignment_untrusted"
    assert reader.calls == []


def test_assignment_attests_to_binding_and_exact_resolved_input_before_authorization() -> None:
    tools, _, reader, specialists = _setup()
    payload = _payload()
    plan = Planner(specialists=specialists).validate(payload, goal=payload["goal"], tools=tools)
    step = plan.steps[1]
    security = CapturingSecurityController()
    executor = GenericExecutor(tools, security_controller=security, specialist_registry=specialists)

    result = executor.execute_step(step, _context())

    resolved_input = {"path": "workspace/synthetic.txt"}
    assert result.status == "success"
    assert reader.calls == [resolved_input]
    assert security.last_specialist_binding is not None
    assert security.last_specialist_binding["input_bindings_digest"] == _digest(step.input_bindings)
    assert security.last_specialist_binding["resolved_input_digest"] == _digest(resolved_input)
    assert security.last_action is not None and security.last_action.input_fingerprint


def test_legacy_assignment_remains_valid_for_an_unbound_step() -> None:
    tools, _, reader, specialists = _setup()
    goal = "Read a directly supplied synthetic repository path"
    objective = "Read the direct synthetic repository path"
    plan = Planner(specialists=specialists).validate(
        {"goal": goal, "rationale": "Legacy assignment compatibility control", "steps": [
            {"id": "reader", "objective": objective, "tool": reader.name,
             "input": {"path": "workspace/synthetic.txt"}, "reason": "Synthetic direct input",
             "depends_on": [], "specialist_id": "local_repository_analyst"}
        ]},
        goal=goal,
        tools=tools,
    )
    step = plan.steps[0]
    step.specialist_assignment.pop("input_bindings_digest")
    executor = GenericExecutor(tools, security_controller=SecurityController(), specialist_registry=specialists)

    result = executor.execute_step(step, ExecutionContext(
        task_id="synthetic-legacy-assignment-task",
        approved_permissions={"public_read"},
        metadata={"goal": goal},
    ))

    assert result.status == "success"
    assert reader.calls == [{"path": "workspace/synthetic.txt"}]


def test_unbound_specialist_approval_keeps_preflight_and_execution_fingerprints_equal() -> None:
    tools, _, reader, specialists = _setup()
    goal = "Read a directly supplied synthetic repository path"
    objective = "Read the direct synthetic repository path"
    plan = Planner(specialists=specialists).validate(
        {"goal": goal, "rationale": "Approval fingerprint compatibility", "steps": [
            {"id": "reader", "objective": objective, "tool": reader.name,
             "input": {"path": "workspace/synthetic.txt"}, "reason": "Synthetic direct input",
             "depends_on": [], "specialist_id": "local_repository_analyst"}
        ]},
        goal=goal,
        tools=tools,
    )
    step = plan.steps[0]
    execution_binding = specialists.execution_binding(
        step.specialist_assignment,
        tool=reader,
        objective=step.objective,
        inputs=step.input,
        input_bindings=step.input_bindings,
        resolved_inputs=step.input,
    )
    assert execution_binding == step.specialist_assignment

    security = SecurityController(
        policy_engine=PolicyEngine(SecurityPolicy(approval_required_risks=["LOW"]))
    )
    preflight_action = security.action_for_tool(
        "synthetic-unbound-approval-task",
        step.id,
        reader,
        step.input,
        goal=goal,
        specialist_binding=step.specialist_assignment,
    )
    requested = security.preflight(preflight_action)
    assert requested.decision == "REQUIRE_APPROVAL"
    security.approval_gate.approve(requested.approval_id, actor="synthetic-test-owner")
    execution_action = security.action_for_tool(
        "synthetic-unbound-approval-task",
        step.id,
        reader,
        step.input,
        goal=goal,
        specialist_binding=execution_binding,
    )

    assert execution_action.input_fingerprint == preflight_action.input_fingerprint
    assert security.authorize(execution_action, approval_id=requested.approval_id, boundary=True).allowed


def test_changed_binding_edge_is_rejected_before_tool_side_effect() -> None:
    tools, _, reader, specialists = _setup()
    payload = _payload()
    plan = Planner(specialists=specialists).validate(payload, goal=payload["goal"], tools=tools)
    step = plan.steps[1]
    step.input_bindings["path"]["output_key"] = "content"
    executor = GenericExecutor(tools, security_controller=SecurityController(), specialist_registry=specialists)

    result = executor.execute_step(step, _context())

    assert result.status == "blocked"
    assert result.error is not None and result.error.code == "specialist_assignment_untrusted"
    assert reader.calls == []


def test_binding_from_unverified_dependency_still_fails_closed() -> None:
    tools, _, reader, specialists = _setup()
    payload = _payload()
    plan = Planner(specialists=specialists).validate(payload, goal=payload["goal"], tools=tools)
    step = plan.steps[1]
    executor = GenericExecutor(tools, security_controller=SecurityController(), specialist_registry=specialists)

    result = executor.execute_step(step, _context(verified=False))

    assert result.status == "blocked"
    assert result.error is not None and result.error.code == "input_binding_source_unverified"
    assert reader.calls == []


def test_approved_bound_specialist_resume_reuses_exact_preflight_fingerprint(tmp_path: Any) -> None:
    tools, _, reader, specialists = _setup()
    payload = _payload()
    plan = Planner(specialists=specialists).validate(payload, goal=payload["goal"], tools=tools)
    step = plan.steps[1]
    store = Store(tmp_path / "specialist-bound-approval.sqlite3")
    security = CapturingSecurityController(
        store=store,
        policy_engine=PolicyEngine(SecurityPolicy(approval_required_risks=["LOW"]), store=store),
    )
    orchestrator = Orchestrator(
        store,
        tools,
        planner=Planner(specialists=specialists),
        security_controller=security,
        specialist_registry=specialists,
    )
    context = _context()
    state = TaskState.create(context.task_id, payload["goal"])
    state.set_plan(
        {
            "goal": plan.goal,
            "rationale": plan.rationale,
            "planning_mode": plan.planning_mode,
            "steps": [item.__dict__ for item in plan.steps],
        },
        version=1,
    )
    experience = ExperienceRecord(state.task_id, state.goal, {}, {})
    preflight_args = {
        "plan": plan,
        "prior": [],
        "learned_context": [],
        "attempts": 0,
        "step_results": [],
        "outcomes": [],
        "diagnoses": [],
        "experience": experience,
    }

    try:
        waiting = orchestrator._preflight_bound_step(state, step, context, **preflight_args)
        assert waiting is not None
        pending = state.security_decisions[step.id]
        assert pending["decision"] == "REQUIRE_APPROVAL"
        approval_id = pending["approval_id"]
        assert approval_id
        security.approval_gate.approve(approval_id, actor="synthetic-test-owner")

        resumed = orchestrator._preflight_bound_step(
            state,
            step,
            context,
            approval_ids={step.id: approval_id},
            **preflight_args,
        )
        assert resumed is None
        preflight_fingerprint = security.last_action.input_fingerprint
        assert state.security_decisions[step.id]["decision"] == "ALLOW"
        action_key = state.action_key_for_step(step)
        expected_resolved_digest = _digest({"path": "workspace/synthetic.txt"})
        assert state.action_records[action_key]["resolved_input_digest"] == expected_resolved_digest
        restored_state = TaskState.from_dict(state.to_dict())
        assert restored_state.action_key_for_step(step) == action_key
        assert restored_state.action_records[action_key]["resolved_input_digest"] == expected_resolved_digest
        assert reader.calls == []

        context.approval_ids[step.id] = approval_id
        result = orchestrator.executor.execute_step(step, context)

        assert result.status == "success"
        assert reader.calls == [{"path": "workspace/synthetic.txt"}]
        assert security.last_action is not None
        assert security.last_action.input_fingerprint == preflight_fingerprint
        assert security.last_specialist_binding is not None
        assert security.last_specialist_binding["resolved_input_digest"] == expected_resolved_digest
    finally:
        store.close()
