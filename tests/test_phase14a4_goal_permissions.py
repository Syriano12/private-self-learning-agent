from __future__ import annotations

from pathlib import Path

from private_agent.core.execution import ExecutionContext, GenericExecutor
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.planner import PlanStep, TaskPlan
from private_agent.security import SecurityController, OwnerPermissionRule
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry


class GoalTool:
    permission_level = "public_read"
    risk_level = "low"
    required_capabilities: list[str] = []
    declared_capabilities: list[str] = []
    planned_effect = "goal permission test effect"

    def __init__(self, name: str = "goal_tool") -> None:
        self.name = name
        self.calls = 0

    def description(self) -> str:
        return "goal permission test tool"

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
        return ToolResult.success({"result": self.name})


class GoalPlanner:
    provider = None
    max_steps = 1

    def __init__(self, tool_name: str = "goal_tool") -> None:
        self.tool_name = tool_name

    def create(self, goal: str, tools: ToolRegistry, prior_knowledge: list[dict], **kwargs) -> TaskPlan:
        return TaskPlan(
            goal,
            [
                PlanStep(
                    "goal-action",
                    "goal action",
                    self.tool_name,
                    {},
                    "controlled test",
                    [],
                    success_criteria={"required_fields": ["result"]},
                )
            ],
            "goal permission test plan",
        )


def registry(tool: GoalTool) -> ToolRegistry:
    result = ToolRegistry()
    result.register(tool)
    return result


def rule(rule_id: str, effect: str, goal: str) -> OwnerPermissionRule:
    return OwnerPermissionRule(
        rule_id=rule_id,
        effect=effect,
        scope_type="GOAL",
        scope_key=goal,
        reason=f"goal test: {rule_id}",
    )


def make_agent(tmp_path: Path, tool: GoalTool, security: SecurityController) -> Orchestrator:
    store = security.store
    assert store is not None
    return Orchestrator(
        store,
        registry(tool),
        planner=GoalPlanner(tool.name),
        security_controller=security,
        max_recovery_attempts=0,
    )


def test_goal_deny_reaches_preflight_and_execution_boundary_without_tool_call(tmp_path: Path) -> None:
    store = Store(tmp_path / "goal-deny.sqlite3")
    tool = GoalTool()
    security = SecurityController(store=store)
    store.create_owner_permission_rule(
        rule("deny-target", "DENY", "delete customer data"), actor="owner", reason="protect target goal"
    )
    agent = make_agent(tmp_path, tool, security)

    result = agent.run("delete customer data", task_id="goal-deny-task")

    assert result["final_status"] == "BLOCKED"
    assert tool.calls == 0
    assert result["security_decisions"][0]["decision"] == "DENY"
    store.close()


def test_goal_allow_and_require_approval_keep_owner_precedence(tmp_path: Path) -> None:
    store = Store(tmp_path / "goal-precedence.sqlite3")
    tool = GoalTool()
    security = SecurityController(store=store)
    store.create_owner_permission_rule(
        rule("goal-review", "REQUIRE_APPROVAL", "publish report"), actor="owner", reason="review target goal"
    )
    store.create_owner_permission_rule(
        rule("goal-allow", "ALLOW", "publish report"), actor="owner", reason="allow target goal"
    )
    agent = make_agent(tmp_path, tool, security)

    result = agent.run("publish report", task_id="goal-precedence-task")

    assert result["final_status"] == "BLOCKED"
    assert result["security_decisions"][0]["decision"] == "REQUIRE_APPROVAL"
    assert tool.calls == 0
    store.close()


def test_goal_allow_applies_only_to_matching_goal_and_direct_executor_receives_goal(tmp_path: Path) -> None:
    store = Store(tmp_path / "goal-scope.sqlite3")
    tool = GoalTool()
    security = SecurityController(store=store)
    store.create_owner_permission_rule(
        rule("goal-allow", "ALLOW", "publish report"), actor="owner", reason="allow only target goal"
    )
    agent = make_agent(tmp_path, tool, security)

    matching = agent.run("publish report", task_id="goal-match-task")
    assert matching["final_status"] == "COMPLETED"
    assert tool.calls == 1

    other = agent.run("archive report", task_id="goal-other-task")
    assert other["final_status"] == "COMPLETED"
    assert tool.calls == 2

    direct = GenericExecutor(registry(tool), security).execute_step(
        PlanStep("direct-action", "direct", tool.name, {}, "controlled", []),
        ExecutionContext(
            task_id="direct-task",
            security_controller=security,
            metadata={"goal": "publish report"},
        ),
    )
    assert direct.status == "success"
    assert tool.calls == 3

    missing_goal = GenericExecutor(registry(tool), security).execute_step(
        PlanStep("missing-goal-action", "direct", tool.name, {}, "controlled", []),
        ExecutionContext(task_id="missing-goal-task", security_controller=security),
    )
    assert missing_goal.status == "blocked"
    assert tool.calls == 3
    store.close()


def test_goal_require_approval_is_preserved_on_resume_and_new_goal_deny_blocks(tmp_path: Path) -> None:
    store = Store(tmp_path / "goal-resume.sqlite3")
    tool = GoalTool()
    security = SecurityController(store=store)
    store.create_owner_permission_rule(
        rule("goal-review", "REQUIRE_APPROVAL", "publish report"), actor="owner", reason="review target goal"
    )
    agent = make_agent(tmp_path, tool, security)

    first = agent.run("publish report", task_id="goal-resume-task")
    assert first["final_status"] == "BLOCKED"
    approval_id = first["approval_requests"][0]["approval_id"]
    security.approval_gate.approve(approval_id)

    store.create_owner_permission_rule(
        rule("goal-deny-after-pause", "DENY", "publish report"), actor="owner", reason="deny changed target"
    )
    resumed = agent.resume_task("goal-resume-task", approval_ids={"goal-action": approval_id})

    assert resumed["final_status"] == "BLOCKED"
    assert tool.calls == 0
    assert resumed["security_decisions"][0]["decision"] == "DENY"
    store.close()
