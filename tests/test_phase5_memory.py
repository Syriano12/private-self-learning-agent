from __future__ import annotations

import json
from pathlib import Path

from private_agent.core.experience import ExperienceAction, ExperienceRecord
from private_agent.core.memory import (
    ExperienceMemory,
    ExperienceQuery,
    RetrievedExperience,
    SQLiteExperienceMemory,
)
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.planner import PlanStep, TaskPlan, Planner
from private_agent.core.replanning import ReplanRequest, Replanner
from private_agent.core.diagnosis import FailureDiagnoser
from private_agent.core.execution import ExecutionContext, StepResult
from private_agent.core.observation import Observation
from private_agent.core.verification import VerificationResult
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry


def make_experience(
    task_id: str,
    goal: str,
    *,
    outcome: str = "COMPLETED",
    tools: tuple[str, ...] = ("tool_a",),
    failure: str | None = None,
    recovery: str | None = None,
    task_type: str = "research",
) -> ExperienceRecord:
    action = ExperienceAction(
        action_type="execute",
        step_id="step",
        tool_name=tools[0],
        input_fingerprint=f"fp-{task_id}",
        execution_status="success" if outcome == "COMPLETED" else "failed",
        observation={"observed": True},
        verification={
            "status": "VERIFIED" if outcome == "COMPLETED" else "FAILED",
            "evidence": [{"type": "proof", "value": "present"}],
        },
        diagnosis={"failure_type": failure} if failure else None,
        recovery={"strategy": recovery} if recovery else None,
    )
    if len(tools) > 1:
        action2 = ExperienceAction(
            action_type="recovery_execute",
            step_id="step",
            tool_name=tools[1],
            input_fingerprint=f"fp-{task_id}-2",
            execution_status="success",
            observation={"observed": True},
            verification={"status": "VERIFIED", "evidence": [{"type": "proof", "value": "present"}]},
            recovery={"strategy": recovery} if recovery else None,
        )
        record = ExperienceRecord(
            task_id,
            goal,
            {"task_type": task_type, "required_capabilities": ["web_research"], "constraints": {"platform": "local"}},
            {"steps": [{"tool": tools[0]}]},
            [action, action2],
            outcome,
        )
    else:
        record = ExperienceRecord(
            task_id,
            goal,
            {"task_type": task_type, "required_capabilities": ["web_research"], "constraints": {"platform": "local"}},
            {"steps": [{"tool": tools[0]}]},
            [action],
            outcome,
        )
    record.build_identity()
    return record


def test_successful_similar_experience_is_retrieved(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.sqlite3")
    memory = SQLiteExperienceMemory(store)
    memory.store(make_experience("success", "research local Python agents", tools=("web_research",)))
    results = memory.retrieve(ExperienceQuery("research local Python agents", tools=["web_research"]))
    assert results
    assert results[0].experience_id == "success"
    assert results[0].outcome == "COMPLETED"
    assert results[0].relevance > 0
    assert results[0].why_relevant
    assert results[0].identity["goal"] == "research local Python agents"


def test_failed_experience_is_retrieved_as_failure_not_success(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.sqlite3")
    memory = SQLiteExperienceMemory(store)
    memory.store(make_experience("failed", "research local Python agents", outcome="FAILED", failure="TOOL_ERROR"))
    result = memory.retrieve(ExperienceQuery("research local Python agents", failure_type="TOOL_ERROR"))[0]
    assert result.outcome == "FAILED"
    assert result.failure_diagnosis == ["TOOL_ERROR"]
    assert "failure_warning" in result.why_relevant


def test_recovery_experience_exposes_alternative_strategy(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.sqlite3")
    memory = SQLiteExperienceMemory(store)
    memory.store(
        make_experience(
            "recovered",
            "research local Python agents",
            tools=("tool_a", "tool_b"),
            recovery="CHANGE_TOOL",
        )
    )
    result = memory.retrieve(ExperienceQuery("research local Python agents", recovery_strategy="CHANGE_TOOL"))[0]
    assert result.outcome == "COMPLETED"
    assert result.strategy_used == ["tool_a", "tool_b"]
    assert result.recovery_strategy == ["CHANGE_TOOL"]


def test_irrelevant_experience_does_not_out_rank_relevant_one(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.sqlite3")
    memory = SQLiteExperienceMemory(store)
    memory.store(make_experience("irrelevant", "configure unrelated database backups", tools=("database",)))
    memory.store(make_experience("relevant", "research local Python agents", tools=("web_research",)))
    results = memory.retrieve(ExperienceQuery("research local Python agents", tools=["web_research"]))
    assert results[0].experience_id == "relevant"
    assert all(result.experience_id != "irrelevant" for result in results[1:])


def test_retrieval_scoring_is_deterministic_and_not_recency_only(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.sqlite3")
    memory = SQLiteExperienceMemory(store)
    memory.store(make_experience("weak", "research local agents", tools=("other",), task_type="other"))
    memory.store(make_experience("strong", "research local Python agents", tools=("web_research",), task_type="research"))
    query = ExperienceQuery("research local Python agents", task_type="research", tools=["web_research"])
    first = [item.experience_id for item in memory.retrieve(query)]
    second = [item.experience_id for item in memory.retrieve(query)]
    assert first == second
    assert first[0] == "strong"


def test_context_limits_bound_results_characters_and_evidence(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.sqlite3")
    memory = SQLiteExperienceMemory(store, max_results=2, max_chars=1000, max_evidence_per_experience=1)
    for index in range(5):
        memory.store(make_experience(str(index), "research local Python agents", tools=("web_research",)))
    results = memory.retrieve(ExperienceQuery("research local Python agents"))
    assert len(results) <= 2
    assert all(len(item.verification_evidence) <= 1 for item in results)
    assert sum(len(str(item.to_dict())) for item in results) <= 1000


def test_secret_sanitization_applies_to_stored_and_retrieved_experience(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.sqlite3")
    memory = SQLiteExperienceMemory(store)
    record = make_experience("secret", "research api_key=super-secret-value")
    record.context["api_key"] = "another-secret"
    record.actions[0].observation["token"] = "token-secret"
    memory.store(record)
    stored = store.get_experience("secret")
    assert stored is not None
    assert "super-secret-value" not in stored["experience_json"]
    assert "another-secret" not in stored["experience_json"]
    assert "token-secret" not in stored["experience_json"]
    retrieved = memory.retrieve(ExperienceQuery("research secret"))[0].to_dict()
    serialized = json.dumps(retrieved)
    assert "super-secret-value" not in serialized
    assert "another-secret" not in serialized
    assert "token-secret" not in serialized


def test_planner_accepts_experiences_as_optional_context_without_forcing_strategy() -> None:
    class FakeProvider:
        def __init__(self):
            self.context = None

        def generate_json(self, *, system_prompt: str, user_prompt: str, response_schema: dict):
            self.context = json.loads(user_prompt)
            from private_agent.core.llm import LLMResponse

            return LLMResponse(
                {
                    "goal": "research local Python agents",
                    "rationale": "choose a safe registered tool",
                    "steps": [
                        {
                            "id": "step",
                            "objective": "research",
                            "tool": "web_research",
                            "input": {"query": "research local Python agents"},
                            "reason": "registered",
                            "depends_on": [],
                        }
                    ],
                },
                "{}",
                "fake",
            )

    provider = FakeProvider()
    registry = ToolRegistry()
    from private_agent.tools.research import WebResearchTool

    registry.register(WebResearchTool())
    experience = {"experience_id": "previous", "outcome": "FAILED", "relevance": 0.8}
    plan = Planner(provider).create("research local Python agents", registry, [], retrieved_experiences=[experience])
    assert plan.steps[0].tool == "web_research"
    assert provider.context["relevant_experiences"] == [experience]


def test_replanner_accepts_experience_context_without_executing_tools() -> None:
    class FakeProvider:
        def __init__(self):
            self.context = None

        def generate_json(self, *, system_prompt: str, user_prompt: str, response_schema: dict):
            self.context = json.loads(user_prompt)
            from private_agent.core.llm import LLMResponse

            return LLMResponse(
                {
                    "goal": "goal",
                    "rationale": "use registered alternative",
                    "steps": [
                        {
                            "id": "step",
                            "objective": "recover",
                            "tool": "b",
                            "input": {},
                            "reason": "alternative",
                            "depends_on": [],
                        }
                    ],
                },
                "{}",
                "fake",
            )

    class Tool:
        risk_level = "low"
        permission_level = "public_read"

        def __init__(self, name):
            self.name = name
            self.calls = 0

        def description(self):
            return self.name

        def input_schema(self):
            return {"type": "object", "properties": {}, "additionalProperties": False}

        def output_schema(self):
            return {"type": "object"}

        def execute(self, inputs, context):
            self.calls += 1
            return ToolResult.success({})

    registry = ToolRegistry()
    a, b = Tool("a"), Tool("b")
    registry.register(a)
    registry.register(b)
    current = TaskPlan("goal", [PlanStep("step", "recover", "a", {}, "reason")], "initial")
    diagnosis = FailureDiagnoser().diagnose(
        step_result=StepResult.failed("step", "a", "tool_exception", "unavailable"),
        observation=Observation("task", "step", "a", "failed", None, {"code": "tool_exception"}),
        verification=VerificationResult("FAILED", "failed"),
    )
    assert diagnosis is not None
    provider = FakeProvider()
    retrieved = [{"experience_id": "previous", "outcome": "COMPLETED", "relevance": 0.7}]
    plan = Replanner(provider).replan(
        ReplanRequest(
            current,
            Observation("task", "step", "a", "failed", None, {"code": "tool_exception"}),
            diagnosis,
            [],
            {},
            {"a": "public_read", "b": "public_read"},
            retrieved,
        ),
        registry,
    )
    assert plan.steps[0].tool == "b"
    assert provider.context["retrieved_experiences"] == retrieved
    assert a.calls == 0 and b.calls == 0


def test_orchestrator_retrieves_before_planning_and_stores_through_interface(tmp_path: Path) -> None:
    class MemoryDouble(ExperienceMemory):
        def __init__(self):
            self.retrieval_queries = []
            self.stored = []

        def store(self, experience: ExperienceRecord) -> None:
            self.stored.append(experience)

        def retrieve(self, query: ExperienceQuery) -> list[RetrievedExperience]:
            self.retrieval_queries.append(query)
            return [RetrievedExperience("previous", 0.8, "prior task", ["tool"], "COMPLETED", [], [], [], ["relevant"], {"task_type": "general"})]

    class Tool:
        name = "tool"
        risk_level = "low"
        permission_level = "public_read"

        def description(self):
            return "tool"

        def input_schema(self):
            return {"type": "object", "properties": {}, "additionalProperties": False}

        def output_schema(self):
            return {"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"], "additionalProperties": False}

        def execute(self, inputs, context):
            return ToolResult.success({"result": "ok"})

    class PlannerDouble:
        provider = None
        max_steps = 12

        def __init__(self):
            self.retrieved = None

        def create(self, goal, tools, prior_knowledge, *, permissions=None, retrieved_experiences=None):
            self.retrieved = retrieved_experiences
            return TaskPlan(goal, [PlanStep("step", "run", "tool", {}, "reason", success_criteria={"required_fields": ["result"]})], "plan")

    memory = MemoryDouble()
    planner = PlannerDouble()
    registry = ToolRegistry()
    registry.register(Tool())
    result = Orchestrator(Store(tmp_path / "agent.sqlite3"), registry, planner=planner, experience_memory=memory).run("goal")
    assert memory.retrieval_queries[0].goal == "goal"
    assert planner.retrieved[0]["experience_id"] == "previous"
    assert result["verification"]["verified"] is True
    assert len(memory.stored) == 1
    assert memory.stored[0].context["retrieved_experiences"][0]["experience_id"] == "previous"
