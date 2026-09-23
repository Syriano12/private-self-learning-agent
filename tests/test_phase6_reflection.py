from __future__ import annotations

import json
from pathlib import Path

import pytest

from private_agent.core.experience import ExperienceAction, ExperienceRecord
from private_agent.core.memory import ExperienceMemory, ExperienceQuery, RetrievedExperience, SQLiteExperienceMemory
from private_agent.core.planner import PlanValidationError, Planner
from private_agent.core.reflection import (
    ReflectionEngine,
    ReflectionInsight,
    ReflectionMemory,
    SQLiteReflectionMemory,
)
from private_agent.storage import Store
from private_agent.tools.research import ToolRegistry, WebResearchTool


def candidate(
    experience_id: str,
    goal: str = "research X",
    *,
    outcome: str = "COMPLETED",
    tools: list[str] | None = None,
    failures: list[str] | None = None,
    recoveries: list[str] | None = None,
    task_type: str = "research",
    created_at: str | None = None,
) -> RetrievedExperience:
    return RetrievedExperience(
        experience_id=experience_id,
        relevance=0.9,
        task_summary=goal,
        strategy_used=tools or ["tool_a"],
        outcome=outcome,
        failure_diagnosis=failures or [],
        recovery_strategy=recoveries or [],
        verification_evidence=[{"type": "verification", "value": "real evidence"}],
        why_relevant=["goal_overlap"],
        identity={"goal": goal, "task_type": task_type, "tools": tools or ["tool_a"]},
        created_at=created_at or "2026-01-01T00:00:00+00:00",
    )


class MemoryDouble(ExperienceMemory):
    def __init__(self, experiences: list[RetrievedExperience]):
        self.experiences = experiences
        self.stored: list[ExperienceRecord] = []
        self.queries: list[ExperienceQuery] = []

    def store(self, experience: ExperienceRecord) -> None:
        self.stored.append(experience)

    def retrieve(self, query: ExperienceQuery) -> list[RetrievedExperience]:
        self.queries.append(query)
        return list(self.experiences)


class ReflectionMemoryDouble(ReflectionMemory):
    def __init__(self):
        self.items: dict[str, ReflectionInsight] = {}

    def save(self, insight: ReflectionInsight) -> None:
        self.items[insight.insight_id] = insight

    def get(self, insight_id: str) -> ReflectionInsight | None:
        return self.items.get(insight_id)

    def list_relevant(self, context: dict, *, limit: int = 5) -> list[ReflectionInsight]:
        return list(self.items.values())[:limit]


def recovery_candidates(count: int = 2) -> list[RetrievedExperience]:
    return [
        candidate(
            f"recovery-{index}",
            tools=["tool_a", "tool_b"],
            failures=["NETWORK_ERROR"],
            recoveries=["CHANGE_TOOL"],
            created_at=f"2026-01-0{index + 1}T00:00:00+00:00",
        )
        for index in range(count)
    ]


def test_successful_recovery_generates_structured_insight() -> None:
    memory = MemoryDouble(recovery_candidates(2))
    engine = ReflectionEngine(memory, min_supporting_experiences=2)
    insights = engine.reflect(ExperienceQuery("research X", task_type="research"))
    recovery = [insight for insight in insights if insight.pattern == "successful_recovery_alternative"]
    assert len(recovery) == 1
    insight = recovery[0]
    assert insight.condition["failed_tool"] == "tool_a"
    assert insight.condition["alternative_tool"] == "tool_b"
    assert "tool_b" in insight.derived_strategy
    assert insight.supporting_experience_ids == ["recovery-0", "recovery-1"]
    assert insight.evidence[0]["experience_id"] in insight.supporting_experience_ids
    assert 0 < insight.confidence <= 0.95


def test_single_supporting_experience_does_not_create_default_recovery_pattern() -> None:
    insights = ReflectionEngine(MemoryDouble(recovery_candidates(1))).reflect(
        ExperienceQuery("research X", task_type="research")
    )
    assert not any(item.pattern == "successful_recovery_alternative" for item in insights)


def test_repeated_identical_failures_generate_failure_pattern() -> None:
    failures = [candidate(f"failure-{index}", outcome="FAILED", failures=["TOOL_ERROR"]) for index in range(3)]
    insights = ReflectionEngine(MemoryDouble(failures)).reflect(ExperienceQuery("research X", task_type="research"))
    repeated = [insight for insight in insights if insight.pattern == "repeated_failure_warning"]
    assert len(repeated) == 1
    assert repeated[0].condition["failed_tool"] == "tool_a"
    assert len(repeated[0].supporting_experience_ids) == 3
    assert repeated[0].metadata["automatic_rule"] is False


def test_alternative_success_and_evidence_ids_are_real() -> None:
    experiences = recovery_candidates(2)
    insights = ReflectionEngine(MemoryDouble(experiences), min_supporting_experiences=2).reflect(
        ExperienceQuery("research X", task_type="research")
    )
    insight = next(item for item in insights if item.pattern == "successful_recovery_alternative")
    real_ids = {item.experience_id for item in experiences}
    assert set(insight.supporting_experience_ids).issubset(real_ids)
    assert {evidence["experience_id"] for evidence in insight.evidence}.issubset(real_ids)


def test_conflicting_experiences_reduce_confidence_deterministically() -> None:
    supportive = recovery_candidates(2)
    conflicting = [candidate("conflict-1", outcome="FAILED", failures=["NETWORK_ERROR"]), candidate("conflict-2", outcome="FAILED", failures=["NETWORK_ERROR"])]
    without_conflict = next(
        item for item in ReflectionEngine(MemoryDouble(supportive), min_supporting_experiences=2).reflect(ExperienceQuery("research X", task_type="research"))
        if item.pattern == "successful_recovery_alternative"
    )
    with_conflict = next(
        item for item in ReflectionEngine(MemoryDouble(supportive + conflicting), min_supporting_experiences=2).reflect(ExperienceQuery("research X", task_type="research"))
        if item.pattern == "successful_recovery_alternative"
    )
    assert with_conflict.confidence < without_conflict.confidence
    assert set(with_conflict.metadata["conflicting_experience_ids"]) == {"conflict-1", "conflict-2"}


def test_same_input_experiences_produce_same_insight_and_confidence() -> None:
    experiences = recovery_candidates(2)
    query = ExperienceQuery("research X", task_type="research")
    first = [item.to_dict() for item in ReflectionEngine(MemoryDouble(experiences), min_supporting_experiences=2).reflect(query)]
    second = [item.to_dict() for item in ReflectionEngine(MemoryDouble(experiences), min_supporting_experiences=2).reflect(query)]
    assert first == second


def test_verification_failure_and_input_failure_patterns_are_distinct() -> None:
    verification = candidate("verification", outcome="FAILED", failures=["VERIFICATION_FAILED"])
    invalid_input = candidate("input", outcome="FAILED", failures=["INVALID_INPUT"], recoveries=["RETRY_WITH_MODIFIED_INPUT"])
    insights = ReflectionEngine(MemoryDouble([verification, invalid_input]), min_supporting_experiences=1).reflect(
        ExperienceQuery("research X", task_type="research")
    )
    assert any(item.pattern == "verification_failure_requires_evidence" for item in insights)
    assert any(item.pattern == "invalid_input_adjustment" for item in insights)


def test_irrelevant_experiences_do_not_generate_unrelated_insights() -> None:
    irrelevant = [
        candidate(f"other-{index}", goal="manage unrelated database", task_type="database", failures=["TOOL_ERROR"], outcome="FAILED")
        for index in range(3)
    ]
    insights = ReflectionEngine(MemoryDouble(irrelevant)).reflect(ExperienceQuery("research X", task_type="research"))
    assert insights == []


def test_reflection_memory_sqlite_supports_save_get_and_relevant_ordering(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.sqlite3")
    memory = ReflectionMemoryDouble()
    engine = ReflectionEngine(MemoryDouble(recovery_candidates(2)), memory, min_supporting_experiences=2)
    insights = engine.reflect(ExperienceQuery("research X", task_type="research"))
    assert insights
    insight = insights[0]
    assert memory.get(insight.insight_id) is not None

    sqlite_memory = SQLiteReflectionMemory(store)
    sqlite_memory.save(insight)
    loaded = sqlite_memory.get(insight.insight_id)
    assert loaded is not None
    assert loaded.to_dict() == insight.to_dict()
    assert sqlite_memory.list_relevant({"goal": "research X", "task_type": "research"}, limit=1)[0].insight_id == insight.insight_id


def test_reflection_does_not_execute_tools() -> None:
    class ToolLikeMemory(MemoryDouble):
        pass

    memory = ToolLikeMemory(recovery_candidates(2))
    engine = ReflectionEngine(memory, min_supporting_experiences=2)
    engine.reflect(ExperienceQuery("research X", task_type="research"))
    assert memory.queries
    assert memory.stored == []


def test_reflection_does_not_bypass_planner_permissions() -> None:
    class Provider:
        def __init__(self):
            self.context = None

        def generate_json(self, *, system_prompt: str, user_prompt: str, response_schema: dict):
            self.context = json.loads(user_prompt)
            from private_agent.core.llm import LLMResponse

            return LLMResponse(
                {
                    "goal": "research X",
                    "rationale": "unsafe test",
                    "steps": [
                        {
                            "id": "step",
                            "objective": "use reflected suggestion",
                            "tool": "private_tool",
                            "input": {},
                            "reason": "reflection is advisory",
                            "depends_on": [],
                        }
                    ],
                },
                "{}",
                "fake",
            )

    class PrivateTool:
        permission_level = "private_write"
        risk_level = "high"
        name = "private_tool"

        def description(self):
            return "private"

        def input_schema(self):
            return {"type": "object", "properties": {}, "additionalProperties": False}

        def output_schema(self):
            return {"type": "object"}

        def execute(self, inputs, context):
            raise AssertionError("reflection/planner test must not execute private tool")

    registry = ToolRegistry()
    registry.register(PrivateTool())
    provider = Provider()
    with pytest.raises(PlanValidationError, match="permission_not_granted"):
        Planner(provider).create(
            "research X",
            registry,
            [],
            permissions={},
            reflection_insights=[{"derived_strategy": "use private_tool", "confidence": 0.99}],
        )
    assert provider.context["reflection_insights"][0]["derived_strategy"] == "use private_tool"


def test_insight_secrets_are_sanitized(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.sqlite3")
    memory = SQLiteExperienceMemory(store)
    for index in range(2):
        action = ExperienceAction(
            "execute",
            "step",
            "tool_a",
            f"fp-{index}",
            "failed",
            {"secret": "api_key=do-not-store"},
            {"status": "FAILED", "evidence": [{"value": "Bearer very-secret-token"}]},
            {"failure_type": "TOOL_ERROR"},
            None,
        )
        record = ExperienceRecord(
            f"secret-{index}",
            "research api_key=do-not-store",
            {"task_type": "research", "secret": "hidden"},
            {},
            [action],
            "FAILED",
        )
        memory.store(record)
    insights = ReflectionEngine(memory).reflect(ExperienceQuery("research secret", task_type="research"))
    serialized = json.dumps([insight.to_dict() for insight in insights])
    assert "do-not-store" not in serialized
    assert "very-secret-token" not in serialized
    assert "hidden" not in serialized


def test_reflection_can_be_used_with_fake_reflection_memory() -> None:
    reflection_memory = ReflectionMemoryDouble()
    engine = ReflectionEngine(MemoryDouble(recovery_candidates(2)), reflection_memory, min_supporting_experiences=2)
    insights = engine.reflect(ExperienceQuery("research X", task_type="research"))
    assert insights
    assert set(reflection_memory.items) == {insight.insight_id for insight in insights}


def test_orchestrator_can_receive_injected_reflection_engine(tmp_path: Path) -> None:
    from private_agent.core.execution import ExecutionContext
    from private_agent.core.orchestrator import Orchestrator
    from private_agent.core.planner import PlanStep, TaskPlan
    from private_agent.storage import Store
    from private_agent.tools.contracts import ToolResult

    class Tool:
        name = "tool"
        permission_level = "public_read"
        risk_level = "low"

        def description(self):
            return "tool"

        def input_schema(self):
            return {"type": "object", "properties": {}, "additionalProperties": False}

        def output_schema(self):
            return {"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"], "additionalProperties": False}

        def execute(self, inputs, context: ExecutionContext):
            return ToolResult.success({"result": "ok"})

    class PlannerDouble:
        provider = None
        max_steps = 12

        def create(self, goal, tools, prior_knowledge, *, permissions=None, retrieved_experiences=None, reflection_insights=None):
            return TaskPlan(goal, [PlanStep("step", "run", "tool", {}, "reason", success_criteria={"required_fields": ["result"]})], "plan")

    class ReflectionDouble:
        def __init__(self):
            self.calls = []

        def reflect_for_experience(self, experience):
            self.calls.append(experience.task_id)
            return [ReflectionInsight("test-insight", {}, "test", {}, "observed", "consider", [], 0.5, [experience.task_id])]

    registry = ToolRegistry()
    registry.register(Tool())
    reflection = ReflectionDouble()
    result = Orchestrator(
        Store(tmp_path / "agent.sqlite3"),
        registry,
        planner=PlannerDouble(),
        reflection_engine=reflection,
    ).run("goal")
    assert reflection.calls == [result["task_id"]]
    assert result["reflection_insights"][0]["insight_id"] == "test-insight"
