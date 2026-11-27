from __future__ import annotations

from typing import Any

import pytest

from private_agent.core.execution import ExecutionContext, GenericExecutor
from private_agent.core.planner import PlanStep, PlanValidationError, Planner
from private_agent.core.replanning import Replanner
from private_agent.core.specialists import SpecialistProfile, SpecialistRegistry
from private_agent.core.task_state import TaskState
from private_agent.security import SecurityController
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry


class SyntheticRepositoryTool:
    name = "read_repository_file"
    permission_level = "public_read"
    required_capabilities = ["READ_WORKSPACE"]
    declared_capabilities = ["READ_WORKSPACE"]
    risk_level = "low"
    network_required = False
    filesystem_required = True
    planned_effect = "Read a bounded synthetic source file"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def description(self) -> str:
        return "Read only a synthetic source file"

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        }

    def output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
            "additionalProperties": False,
        }

    def execute(self, inputs: dict[str, Any], context: Any) -> ToolResult:
        del context
        self.calls.append(dict(inputs))
        return ToolResult.success({"content": "synthetic evidence"}, metadata={"read_only": True})


def _setup() -> tuple[ToolRegistry, SyntheticRepositoryTool, SpecialistRegistry]:
    tools = ToolRegistry()
    tool = SyntheticRepositoryTool()
    tools.register(tool)
    specialists = SpecialistRegistry()
    specialists.register(
        SpecialistProfile(
            specialist_id="local_repository_analyst",
            version=1,
            tool_names=frozenset({tool.name}),
            granted_capabilities=frozenset({"READ_WORKSPACE"}),
        )
    )
    return tools, tool, specialists


def _payload(*, specialist_id: str = "local_repository_analyst", tool_name: str = "read_repository_file") -> dict[str, Any]:
    return {
        "goal": "Review a local synthetic repository file",
        "rationale": "The assignment is bounded to read-only local evidence",
        "steps": [
            {
                "id": "read-one",
                "objective": "Read the supplied synthetic file",
                "tool": tool_name,
                "input": {"path": "src/example.py"},
                "reason": "A repository analyst is appropriate",
                "depends_on": [],
                "specialist_id": specialist_id,
            }
        ],
    }


def test_planner_issues_server_bound_assignment_and_rejects_unknown_or_mismatched_roles() -> None:
    tools, tool, specialists = _setup()
    planner = Planner(specialists=specialists)
    plan = planner.validate(_payload(), goal="Review a local synthetic repository file", tools=tools)
    assignment = plan.steps[0].specialist_assignment

    assert assignment["specialist_id"] == "local_repository_analyst"
    assert assignment["tool_name"] == tool.name
    assert assignment["effective_capabilities"] == ["READ_WORKSPACE"]
    assert assignment["profile_version"] == 1
    assert specialists.validate_assignment(
        assignment,
        tool=tool,
        objective=plan.steps[0].objective,
        inputs=plan.steps[0].input,
    )

    other_tool = SyntheticRepositoryTool()
    other_tool.name = "different_local_reader"
    tools.register(other_tool)

    with pytest.raises(PlanValidationError, match="specialist_unknown"):
        planner.validate(
            _payload(specialist_id="invented_root_agent"),
            goal="Review a local synthetic repository file",
            tools=tools,
        )

    with pytest.raises(PlanValidationError, match="specialist_tool_not_eligible"):
        planner.validate(
            _payload(tool_name="different_local_reader"),
            goal="Review a local synthetic repository file",
            tools=tools,
        )

    invalid_tool_payload = _payload()
    invalid_tool_payload["steps"][0]["tool"] = ""
    with pytest.raises(PlanValidationError, match="step_read-one_tool_required"):
        planner.validate(
            invalid_tool_payload,
            goal="Review a local synthetic repository file",
            tools=tools,
        )


def test_registry_excludes_network_or_privileged_tools_from_local_specialists() -> None:
    tools, _, _ = _setup()

    class NetworkTool(SyntheticRepositoryTool):
        name = "search_repository_code"
        required_capabilities = ["READ_WORKSPACE", "NETWORK_ACCESS"]
        declared_capabilities = ["READ_WORKSPACE", "NETWORK_ACCESS"]
        risk_level = "high"
        network_required = True

    tools.register(NetworkTool())
    registry = SpecialistRegistry.for_local_repository_tools(tools)
    visible = registry.available_for_planning(tools)

    assert len(visible) == 1
    assert visible[0]["specialist_id"] == "local_repository_analyst"
    assert visible[0]["tool_names"] == ["read_repository_file"]

    source_less_tools = ToolRegistry()

    class SourceLessTool(SyntheticRepositoryTool):
        required_capabilities: list[str] = []
        declared_capabilities: list[str] = []

    source_less_tools.register(SourceLessTool())
    source_less_registry = SpecialistRegistry.for_local_repository_tools(source_less_tools)
    assert source_less_registry.available_for_planning(source_less_tools) == []


def test_executor_blocks_forged_or_drifted_assignment_before_tool_side_effect() -> None:
    tools, tool, specialists = _setup()
    assignment = specialists.create_assignment(
        "local_repository_analyst",
        tool=tool,
        objective="Read the supplied synthetic file",
        inputs={"path": "src/example.py"},
    )
    assignment["effective_capabilities"] = ["READ_WORKSPACE", "SYSTEM_COMMAND"]
    step = PlanStep(
        id="read-one",
        objective="Read the supplied synthetic file",
        tool=tool.name,
        input={"path": "src/example.py"},
        reason="Synthetic test",
        specialist_assignment=assignment,
    )
    executor = GenericExecutor(
        tools,
        security_controller=SecurityController(),
        specialist_registry=specialists,
    )

    result = executor.execute_step(
        step,
        ExecutionContext(
            task_id="synthetic-parent-task",
            approved_permissions={"public_read"},
            metadata={"goal": "Review a local synthetic repository file"},
        ),
    )

    assert result.status == "blocked"
    assert result.error is not None and result.error.code == "specialist_assignment_untrusted"
    assert tool.calls == []


def test_assignment_identity_survives_state_roundtrip_and_changes_action_fingerprint() -> None:
    tools, tool, specialists = _setup()
    assignment = specialists.create_assignment(
        "local_repository_analyst",
        tool=tool,
        objective="Read the supplied synthetic file",
        inputs={"path": "src/example.py"},
    )
    step = PlanStep(
        id="read-one",
        objective="Read the supplied synthetic file",
        tool=tool.name,
        input={"path": "src/example.py"},
        reason="Synthetic test",
        specialist_assignment=assignment,
    )
    state = TaskState.create("synthetic-parent-task", "Review a local synthetic repository file")
    state.set_plan(
        {
            "goal": state.goal,
            "rationale": "synthetic",
            "planning_mode": "UNKNOWN",
            "steps": [step.__dict__],
        },
        version=1,
    )
    restored = TaskState.from_dict(state.to_dict())

    assert restored.plan["steps"][0]["specialist_assignment"] == assignment
    assert restored.pending_actions == state.pending_actions

    changed = dict(assignment)
    changed["specialist_id"] = "another_specialist"
    changed_plan = {**restored.plan, "steps": [{**restored.plan["steps"][0], "specialist_assignment": changed}]}
    altered = TaskState.create("synthetic-other-task", "Review a local synthetic repository file")
    altered.set_plan(changed_plan, version=1)
    assert altered.pending_actions != restored.pending_actions


def test_replanner_reissues_assignment_from_server_profile_instead_of_reusing_old_record() -> None:
    tools, tool, specialists = _setup()
    planner = Planner(specialists=specialists)
    current_plan = planner.validate(
        _payload(),
        goal="Review a local synthetic repository file",
        tools=tools,
    )
    prior_assignment = current_plan.steps[0].specialist_assignment
    replanner = Replanner(specialists=specialists)

    recovered = replanner.replan_step(
        current_plan,
        step_id="read-one",
        tool_name=tool.name,
        inputs={"path": "src/example.py"},
        tools=tools,
        permissions={tool.name: "public_read"},
    )

    new_assignment = recovered.steps[0].specialist_assignment
    assert new_assignment["assignment_id"] != prior_assignment["assignment_id"]
    assert new_assignment["specialist_id"] == prior_assignment["specialist_id"]
    assert specialists.validate_assignment(
        new_assignment,
        tool=tool,
        objective=recovered.steps[0].objective,
        inputs=recovered.steps[0].input,
    )


def test_restored_pending_assignment_fails_closed_if_tool_contract_drifted(tmp_path: Any) -> None:
    tools, tool, specialists = _setup()
    assignment = specialists.create_assignment(
        "local_repository_analyst",
        tool=tool,
        objective="Read the supplied synthetic file",
        inputs={"path": "src/example.py"},
    )
    state = TaskState.create("synthetic-restart-task", "Review a local synthetic repository file")
    state.set_plan(
        {
            "goal": state.goal,
            "rationale": "synthetic restart case",
            "planning_mode": "UNKNOWN",
            "steps": [
                {
                    "id": "read-one",
                    "objective": "Read the supplied synthetic file",
                    "tool": tool.name,
                    "input": {"path": "src/example.py"},
                    "reason": "Synthetic restart test",
                    "depends_on": [],
                    "specialist_assignment": assignment,
                }
            ],
        },
        version=1,
    )
    database = tmp_path / "specialist-state.sqlite3"
    store = Store(database)
    store.save_task_state(state, "SPECIALIST_ASSIGNMENT_TEST")
    persisted = store.get_task_state(state.task_id)
    assert persisted is not None
    store.close()
    reopened = Store(database)
    persisted_after_restart = reopened.get_task_state(state.task_id)
    assert persisted_after_restart is not None
    restored = TaskState.from_dict(persisted_after_restart)
    restored_step = PlanStep(
        id="read-one",
        objective="Read the supplied synthetic file",
        tool=tool.name,
        input={"path": "src/example.py"},
        reason="Synthetic restart test",
        specialist_assignment=restored.plan["steps"][0]["specialist_assignment"],
    )

    tool.input_schema = lambda: {
        "type": "object",
        "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}},
        "required": ["path"],
        "additionalProperties": False,
    }
    executor = GenericExecutor(
        tools,
        security_controller=SecurityController(),
        specialist_registry=specialists,
    )
    result = executor.execute_step(
        restored_step,
        ExecutionContext(
            task_id="synthetic-restart-task",
            approved_permissions={"public_read"},
            metadata={"goal": "Review a local synthetic repository file"},
        ),
    )

    assert result.status == "blocked"
    assert result.error is not None and result.error.code == "specialist_assignment_untrusted"
    assert tool.calls == []
    reopened.close()


def test_assignment_changes_security_approval_fingerprint() -> None:
    tools, tool, specialists = _setup()
    first = specialists.create_assignment(
        "local_repository_analyst",
        tool=tool,
        objective="Read the supplied synthetic file",
        inputs={"path": "src/example.py"},
    )
    second = dict(first, assignment_id="0" * 32)
    controller = SecurityController()
    action_a = controller.action_for_tool(
        "synthetic-task", "read-one", tool, {"path": "src/example.py"},
        goal="synthetic goal", specialist_binding=first,
    )
    action_b = controller.action_for_tool(
        "synthetic-task", "read-one", tool, {"path": "src/example.py"},
        goal="synthetic goal", specialist_binding=second,
    )

    assert action_a.input_fingerprint != action_b.input_fingerprint
