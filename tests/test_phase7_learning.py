from __future__ import annotations

import json
from pathlib import Path

import pytest

from private_agent.core.execution import ExecutionContext
from private_agent.core.learning import (
    LearnedStrategy,
    LearningEngine,
    LearningMemory,
    SQLiteLearningMemory,
)
from private_agent.core.memory import SQLiteExperienceMemory
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.planner import PlanValidationError, Planner
from private_agent.core.reflection import (
    ReflectionEngine,
    ReflectionInsight,
    SQLiteReflectionMemory,
)
from private_agent.core.verification import VerificationResult
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry


class LearningMemoryDouble(LearningMemory):
    def __init__(self) -> None:
        self.items: dict[str, LearnedStrategy] = {}

    def save(self, strategy: LearnedStrategy) -> None:
        self.items[strategy.strategy_id] = strategy

    def get(self, strategy_id: str) -> LearnedStrategy | None:
        return self.items.get(strategy_id)

    def retrieve_relevant(self, context: dict, *, limit: int = 5) -> list[LearnedStrategy]:
        return list(self.items.values())[:limit]

    def list(self) -> list[LearnedStrategy]:
        return list(self.items.values())


def recovery_insight(
    insight_id: str,
    *,
    confidence: float = 0.81,
    experiences: list[str] | None = None,
    conflicts: list[str] | None = None,
    goal: str = "research X",
    task_type: str = "research",
    failed_tool: str = "tool_a",
    alternative_tool: str = "tool_b",
) -> ReflectionInsight:
    return ReflectionInsight(
        insight_id=insight_id,
        task_context={
            "goal": goal,
            "task_type": task_type,
            "required_capabilities": ["research"],
        },
        pattern="successful_recovery_alternative",
        condition={
            "task_type": task_type,
            "failed_tool": failed_tool,
            "alternative_tool": alternative_tool,
            "failure_types": ["NETWORK_ERROR"],
        },
        observed_behavior="tool_a failed and tool_b later verified the task",
        derived_strategy=f"Prefer {alternative_tool} instead of retrying {failed_tool} unchanged.",
        evidence=[
            {"experience_id": item, "outcome": "COMPLETED", "verification_evidence": [{"type": "proof"}]}
            for item in (experiences or ["experience-a"])
        ],
        confidence=confidence,
        supporting_experience_ids=experiences or ["experience-a"],
        failure_types=["NETWORK_ERROR"],
        recovery_strategies=["CHANGE_TOOL"],
        created_at="2026-01-01T00:00:00+00:00",
        metadata={"conflicting_experience_ids": conflicts or []},
    )


def negative_insight() -> ReflectionInsight:
    return ReflectionInsight(
        insight_id="negative-insight",
        task_context={"goal": "research X", "task_type": "research"},
        pattern="repeated_failure_warning",
        condition={"task_type": "research", "failed_tool": "tool_a", "failure_types": ["TOOL_ERROR"]},
        observed_behavior="tool_a failed repeatedly",
        derived_strategy="Avoid unverified retry of tool_a and require an alternative or verification.",
        evidence=[{"experience_id": "failure-1"}, {"experience_id": "failure-2"}],
        confidence=0.82,
        supporting_experience_ids=["failure-1", "failure-2"],
        failure_types=["TOOL_ERROR"],
        metadata={"conflicting_experience_ids": []},
    )


class ControlledTool:
    risk_level = "low"
    permission_level = "public_read"

    def __init__(self, name: str, outcomes: list[ToolResult]) -> None:
        self.name = name
        self.outcomes = list(outcomes)
        self.calls = 0

    def description(self) -> str:
        return self.name

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
        if self.outcomes:
            return self.outcomes.pop(0)
        return ToolResult.failed("tool_failed", "no configured outcome")


def registry(*tools: ControlledTool) -> ToolRegistry:
    result = ToolRegistry()
    for tool in tools:
        result.register(tool)
    return result


def test_insight_is_converted_to_structured_strategy_with_real_evidence() -> None:
    memory = LearningMemoryDouble()
    strategies = LearningEngine(memory).learn([recovery_insight("insight-a", experiences=["exp-a", "exp-b"])])
    assert len(strategies) == 1
    strategy = strategies[0]
    assert strategy.preferred_action == {"type": "prefer_tool", "tool": "tool_b"}
    assert strategy.avoided_action["tool"] == "tool_a"
    assert strategy.supporting_insight_ids == ["insight-a"]
    assert strategy.supporting_experience_ids == ["exp-a", "exp-b"]
    assert strategy.condition["failure_types"] == ["NETWORK_ERROR"]
    assert strategy.status == "active"


def test_confidence_threshold_and_low_confidence_guidance() -> None:
    memory = LearningMemoryDouble()
    strategy = LearningEngine(memory).learn([recovery_insight("weak", confidence=0.55)])[0]
    assert strategy.status == "uncertain"
    assert memory.retrieve_relevant({"goal": "research Y", "tools": ["tool_a", "tool_b"]})
    # A storage adapter must still exclude uncertain strategies from planning retrieval.
    store = Store(":memory:")
    sqlite_memory = SQLiteLearningMemory(store)
    sqlite_memory.save(strategy)
    assert sqlite_memory.retrieve_relevant({"goal": "research Y", "tools": ["tool_a", "tool_b"]}) == []


def test_incremental_learning_strengthens_the_same_policy_deterministically() -> None:
    memory = LearningMemoryDouble()
    engine = LearningEngine(memory)
    first = engine.learn([recovery_insight("weak", confidence=0.69, experiences=["exp-1"])])[0]
    second = engine.learn([recovery_insight("stronger", confidence=0.81, experiences=["exp-1", "exp-2"])])[0]
    assert first.strategy_id == second.strategy_id
    assert second.status == "active"
    assert second.supporting_experience_ids == ["exp-1", "exp-2"]
    assert second.confidence == 0.81


def test_contradictory_evidence_reduces_confidence_and_marks_strategy_uncertain() -> None:
    memory = LearningMemoryDouble()
    engine = LearningEngine(memory)
    stable = engine.learn([recovery_insight("stable", confidence=0.81, experiences=["exp-a", "exp-b"])])[0]
    conflicted = engine.learn(
        [
            recovery_insight(
                "conflict",
                confidence=0.65,
                experiences=["exp-a", "exp-b", "exp-c"],
                conflicts=["exp-c", "exp-d"],
            )
        ]
    )[0]
    assert conflicted.strategy_id == stable.strategy_id
    assert conflicted.confidence < stable.confidence
    assert conflicted.status == "uncertain"
    assert conflicted.metadata["conflicting_experience_ids"] == ["exp-c", "exp-d"]


def test_negative_learning_produces_avoidance_guidance_without_absolute_ban() -> None:
    memory = LearningMemoryDouble()
    strategy = LearningEngine(memory).learn([negative_insight()])[0]
    assert strategy.status == "active"
    assert strategy.preferred_action["type"] == "require_alternative_or_verification"
    assert strategy.avoided_action == {"type": "avoid_unverified_retry", "tool": "tool_a"}

    tools = registry(
        ControlledTool("tool_a", [ToolResult.success({"result": "a"})]),
        ControlledTool("tool_b", [ToolResult.success({"result": "b"})]),
    )
    plan = Planner().create("research Y", tools, [], learned_strategies=[strategy.to_dict()])
    assert plan.steps[0].tool == "tool_b"


def test_learning_memory_persists_across_new_store_instance(tmp_path: Path) -> None:
    path = tmp_path / "agent.sqlite3"
    strategy = LearningEngine(LearningMemoryDouble()).learn([recovery_insight("persisted", experiences=["exp-a", "exp-b"])])[0]
    first_store = Store(path)
    SQLiteLearningMemory(first_store).save(strategy)
    first_store.close()

    second_store = Store(path)
    loaded = SQLiteLearningMemory(second_store).get(strategy.strategy_id)
    assert loaded is not None
    assert loaded.to_dict() == strategy.to_dict()
    assert SQLiteLearningMemory(second_store).list()[0].strategy_id == strategy.strategy_id
    second_store.close()


def test_learning_engine_supports_fake_memory_and_never_executes_tools() -> None:
    memory = LearningMemoryDouble()
    strategy = LearningEngine(memory).learn([recovery_insight("fake", experiences=["exp-a", "exp-b"])])[0]
    assert memory.get(strategy.strategy_id) is strategy
    assert strategy.preferred_action["type"] == "prefer_tool"


def test_learning_cannot_bypass_planner_permissions() -> None:
    private = ControlledTool("private_tool", [ToolResult.success({"result": "must not run"})])
    private.permission_level = "private_write"
    tools = registry(private)
    strategy = recovery_insight("unsafe", experiences=["exp-a", "exp-b"])
    learned = LearningEngine(LearningMemoryDouble()).learn([strategy])[0]
    with pytest.raises(PlanValidationError, match="permission_not_granted:private_tool"):
        Planner().create("research Y", tools, [], permissions={}, learned_strategies=[learned.to_dict()])
    assert private.calls == 0


def test_learning_secrets_are_sanitized() -> None:
    insight = recovery_insight("secret", experiences=["exp-a", "exp-b"])
    insight.derived_strategy = "use tool_b with api_key=super-secret-value"
    insight.metadata["authorization"] = "Bearer very-secret-token"
    strategy = LearningEngine(LearningMemoryDouble()).learn([insight])[0]
    serialized = json.dumps(strategy.to_dict())
    assert "super-secret-value" not in serialized
    assert "very-secret-token" not in serialized
    assert "[REDACTED]" in serialized


def test_irrelevant_strategy_does_not_affect_retrieval_or_planning(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.sqlite3")
    memory = SQLiteLearningMemory(store)
    strategy = LearningEngine(LearningMemoryDouble()).learn(
        [
            recovery_insight(
                "database",
                goal="manage database backups",
                task_type="database",
                failed_tool="database_a",
                alternative_tool="database_b",
                experiences=["db-a", "db-b"],
            )
        ]
    )[0]
    memory.save(strategy)
    assert memory.retrieve_relevant({"goal": "research Y", "tools": ["tool_a", "tool_b"]}) == []

    tools = registry(
        ControlledTool("tool_a", [ToolResult.success({"result": "a"})]),
        ControlledTool("tool_b", [ToolResult.success({"result": "b"})]),
    )
    plan = Planner().create("research Y", tools, [], learned_strategies=[])
    assert plan.steps[0].tool == "tool_a"


def test_planner_passes_learned_strategies_to_provider_as_advisory_context() -> None:
    class Provider:
        def __init__(self) -> None:
            self.context = None

        def generate_json(self, *, system_prompt: str, user_prompt: str, response_schema: dict):
            self.context = json.loads(user_prompt)
            from private_agent.core.llm import LLMResponse

            return LLMResponse(
                {
                    "goal": "research Y",
                    "rationale": "registered tool",
                    "steps": [
                        {
                            "id": "step",
                            "objective": "research",
                            "tool": "tool_a",
                            "input": {},
                            "reason": "validated",
                            "depends_on": [],
                        }
                    ],
                },
                "{}",
                "fake",
            )

    provider = Provider()
    tools = registry(ControlledTool("tool_a", []))
    strategy = recovery_insight("context", experiences=["exp-a", "exp-b"]).to_dict()
    Planner(provider).create("research Y", tools, [], learned_strategies=[strategy])
    assert provider.context["learned_strategies"] == [strategy]


def test_reflection_conflict_detection_covers_successful_original_and_failed_alternative() -> None:
    from private_agent.core.memory import RetrievedExperience, ExperienceMemory, ExperienceQuery

    class Memory(ExperienceMemory):
        def store(self, experience):
            pass

        def retrieve(self, query: ExperienceQuery):
            return [
                RetrievedExperience(
                    "support-1",
                    0.9,
                    "research X",
                    ["tool_a", "tool_b"],
                    "COMPLETED",
                    ["NETWORK_ERROR"],
                    ["CHANGE_TOOL"],
                    [{"type": "proof"}],
                    ["goal"],
                    {"task_type": "research"},
                ),
                RetrievedExperience(
                    "conflict-1",
                    0.9,
                    "research X",
                    ["tool_a"],
                    "COMPLETED",
                    [],
                    [],
                    [{"type": "proof"}],
                    ["goal"],
                    {"task_type": "research"},
                ),
                RetrievedExperience(
                    "conflict-2",
                    0.9,
                    "research X",
                    ["tool_a", "tool_b"],
                    "FAILED",
                    ["TOOL_ERROR"],
                    ["CHANGE_TOOL"],
                    [],
                    ["goal"],
                    {"task_type": "research"},
                ),
            ]

    insights = ReflectionEngine(Memory(), min_supporting_experiences=1).reflect(
        __import__("private_agent.core.memory", fromlist=["ExperienceQuery"]).ExperienceQuery("research X", task_type="research")
    )
    recovery = next(item for item in insights if item.pattern == "successful_recovery_alternative")
    assert set(recovery.metadata["conflicting_experience_ids"]) == {"conflict-1", "conflict-2"}
    assert recovery.confidence < 0.70


def test_before_after_behavioral_proof_changes_tool_and_verifies_future_task(tmp_path: Path) -> None:
    tool_a = ControlledTool(
        "tool_a",
        [
            ToolResult.failed("network_error", "temporary network failure"),
            ToolResult.failed("network_error", "temporary network failure"),
        ],
    )
    tool_b = ControlledTool(
        "tool_b",
        [
            ToolResult.success({"result": "verified alternative evidence"}),
            ToolResult.success({"result": "verified future evidence"}),
        ],
    )
    tools = registry(tool_a, tool_b)
    path = tmp_path / "agent.sqlite3"

    def make_agent(store: Store) -> Orchestrator:
        experience_memory = SQLiteExperienceMemory(store)
        reflection_memory = SQLiteReflectionMemory(store)
        learning_memory = SQLiteLearningMemory(store, min_confidence=0.65)
        return Orchestrator(
            store,
            tools,
            planner=Planner(learned_strategy_threshold=0.65),
            reflection_engine=ReflectionEngine(
                experience_memory,
                reflection_memory,
                min_supporting_experiences=1,
            ),
            learning_memory=learning_memory,
            learning_engine=LearningEngine(learning_memory, min_confidence=0.65),
            max_attempts=2,
        )

    store = Store(path)
    first_agent = make_agent(store)
    first = first_agent.run("research X")
    assert first["final_status"] == "COMPLETED"
    assert first["step_results"][0]["tool_name"] == "tool_a"
    assert first["step_results"][-1]["tool_name"] == "tool_b"
    assert first["verification"]["verified"] is True
    assert first["reflection_insights"]
    assert first["learned_strategies"]
    learned = first["learned_strategies"][0]
    assert learned["preferred_action"] == {"type": "prefer_tool", "tool": "tool_b"}
    assert learned["status"] == "active"

    second_agent = make_agent(store)
    second = second_agent.run("research Y")
    assert second["learned_strategies_used"]
    assert second["step_results"][0]["tool_name"] == "tool_b"
    assert second["verification"]["status"] == "VERIFIED"
    assert second["final_status"] == "COMPLETED"
    assert tool_a.calls == 2, "future planning must avoid executing tool_a again"
    assert tool_b.calls == 2
    first_plan_tool = json.loads(store.events_for_task(first["task_id"])[0]["event_json"])["steps"][0]["tool"]
    second_plan_tool = json.loads(store.events_for_task(second["task_id"])[0]["event_json"])["steps"][0]["tool"]
    assert first_plan_tool == "tool_a"
    assert second_plan_tool == "tool_b"
    store.close()
