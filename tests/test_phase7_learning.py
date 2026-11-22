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
from private_agent.core.task_state import TaskState
from private_agent.core.reflection import (
    ReflectionEngine,
    ReflectionInsight,
    SQLiteReflectionMemory,
)
from private_agent.core.verification import VerificationResult
from private_agent.bugbounty.reviews import FindingReview, FindingReviewRepository
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
            for item in (experiences or ["experience-a", "experience-b"])
        ],
        confidence=confidence,
        supporting_experience_ids=experiences or ["experience-a", "experience-b"],
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


def test_unverified_negative_learning_cannot_change_the_fallback_plan() -> None:
    memory = LearningMemoryDouble()
    strategy = LearningEngine(memory).learn([negative_insight()])[0]
    assert strategy.status == "active"
    assert strategy.preferred_action["type"] == "require_alternative_or_verification"
    assert strategy.avoided_action == {"type": "avoid_unverified_retry", "tool": "tool_a"}

    tools = registry(
        ControlledTool("tool_a", [ToolResult.success({"result": "a"})]),
        ControlledTool("tool_b", [ToolResult.success({"result": "b"})]),
    )
    plan = Planner().create("research X", tools, [], learned_strategies=[strategy.to_dict()])
    assert plan.steps[0].tool == "tool_a"


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


def test_planner_passes_canonical_persisted_strategy_to_provider_as_advisory_context(tmp_path: Path) -> None:
    class Provider:
        def __init__(self) -> None:
            self.context = None

        def generate_json(self, *, system_prompt: str, user_prompt: str, response_schema: dict):
            self.context = json.loads(user_prompt)
            from private_agent.core.llm import LLMResponse

            return LLMResponse(
                {
                    "goal": "research topic Y",
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
    path = tmp_path / "provider-context.sqlite3"
    store, tools, _, _, _, learned = _train_two_verified_recovery_experiences(path)
    try:
        strategy = learned.to_dict()
        Planner(provider, learning_memory=SQLiteLearningMemory(store)).create(
            "research topic Y", tools, [], learned_strategies=[strategy]
        )
        assert provider.context["learned_strategies"] == [strategy]
    finally:
        store.close()


def test_planner_rejects_fabricated_recovery_strategy_not_present_in_trusted_storage(tmp_path: Path) -> None:
    store = Store(tmp_path / "empty-learning.sqlite3")
    memory = SQLiteLearningMemory(store)
    forged = LearningEngine(LearningMemoryDouble()).learn([
        recovery_insight("forged", goal="research topic alpha")
    ])[0].to_dict()
    tools = registry(ControlledTool("tool_a", []), ControlledTool("tool_b", []))
    try:
        plan = Planner(learning_memory=memory).create(
            "research topic alpha follow-up",
            tools,
            [],
            permissions={"tool_a": "public_read", "tool_b": "public_read"},
            learned_strategies=[forged],
        )
        assert plan.steps[0].tool == "tool_a"
    finally:
        store.close()


def test_owner_review_strategy_is_canonical_provider_context_but_not_fallback_tool_guidance(tmp_path: Path) -> None:
    from private_agent.core.llm import LLMResponse

    store = Store(tmp_path / "owner-review-learning.sqlite3")
    memory = SQLiteLearningMemory(store)
    store.save_task_state(TaskState.create("task-owner-review", "Bug Bounty finding review IDOR"), "TEST_OWNER_REVIEW")
    review = FindingReview.create(
        "task-owner-review",
        "finding-id",
        "owner-id",
        "CONFIRM",
        evidence_sha256="a" * 64,
        evidence_validated=True,
        note="local verified evidence",
    )
    FindingReviewRepository(store).save(review)
    insight = ReflectionInsight(
        insight_id=f"bugbounty-review-{review.review_id}",
        task_context={
            "task_type": "bug_bounty",
            "goal": "Bug Bounty finding review IDOR",
            "required_capabilities": ["evidence_review"],
        },
        pattern="bugbounty_review_feedback",
        condition={
            "task_type": "bug_bounty",
            "vulnerability_class": "IDOR",
            "review_decision": "CONFIRM",
            "tools": ["bugbounty_analysis"],
        },
        observed_behavior="Owner confirmed a candidate after evidence validation.",
        derived_strategy="Require an evidence gate and owner review before report readiness.",
        evidence=[{
            "review_id": review.review_id,
            "finding_id": review.finding_id,
            "decision": "CONFIRM",
            "evidence_sha256": review.evidence_sha256,
            "verified_outcome": True,
        }],
        confidence=0.90,
        supporting_experience_ids=[f"finding-review:{review.review_id}"],
        metadata={
            "source": "owner_finding_review",
            "review_id": review.review_id,
            "finding_id": review.finding_id,
            "evidence_sha256": review.evidence_sha256,
            "verified_outcome": True,
            "non_authoritative": True,
        },
    )
    strategy = LearningEngine(memory).learn([insight])[0].to_dict()
    goal = "Bug Bounty finding review IDOR follow-up"
    tools = registry(ControlledTool("tool_a", []), ControlledTool("tool_b", []))
    permissions = {"tool_a": "public_read", "tool_b": "public_read"}

    class Provider:
        context = None

        def generate_json(self, *, system_prompt: str, user_prompt: str, response_schema: dict):
            self.context = json.loads(user_prompt)
            return LLMResponse(
                {
                    "goal": goal,
                    "rationale": "Keep the normal bounded plan.",
                    "steps": [{
                        "id": "step",
                        "objective": "Run the first registered tool",
                        "tool": "tool_a",
                        "input": {},
                        "reason": "ordinary fallback choice",
                        "depends_on": [],
                    }],
                },
                "{}",
                "fake",
            )

    try:
        provider = Provider()
        Planner(provider, learning_memory=memory).create(
            goal, tools, [], permissions=permissions, learned_strategies=[strategy]
        )
        assert provider.context["learned_strategies"] == [strategy]
        fallback = Planner(learning_memory=memory).create(
            goal, tools, [], permissions=permissions, learned_strategies=[strategy]
        )
        assert fallback.steps[0].tool == "tool_a"
    finally:
        store.close()


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
                    {"task_type": "research", "verification_status": "VERIFIED", "security_impact_verified": None, "verified_action_tools": ["tool_b"], "verified_final_action_tool": "tool_b"},
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
                    {"task_type": "research", "verification_status": "VERIFIED", "security_impact_verified": None, "verified_action_tools": ["tool_a"], "verified_final_action_tool": "tool_a"},
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


def _train_two_verified_recovery_experiences(path: Path):
    tool_a = ControlledTool(
        "tool_a",
        [
            ToolResult.failed("network_error", "temporary network failure"),
            ToolResult.failed("network_error", "temporary network failure"),
            ToolResult.failed("network_error", "temporary network failure"),
            ToolResult.failed("network_error", "temporary network failure"),
        ],
    )
    tool_b = ControlledTool(
        "tool_b",
        [
            ToolResult.success({"result": "verified alternative evidence"}),
            ToolResult.success({"result": "verified independent alternative evidence"}),
            ToolResult.success({"result": "verified future evidence"}),
        ],
    )
    tools = registry(tool_a, tool_b)
    store = Store(path)
    agent = Orchestrator(store, tools, planner=Planner(), max_attempts=2)
    task_ids = []
    for _ in range(2):
        result = agent.run("research topic alpha")
        assert result["final_status"] == "COMPLETED"
        assert result["verification"]["status"] == "VERIFIED"
        assert result["step_results"][0]["tool_name"] == "tool_a"
        assert result["step_results"][-1]["tool_name"] == "tool_b"
        assert result["verification"]["verified"] is True
        task_ids.append(result["task_id"])
    learning = SQLiteLearningMemory(store)
    retrieved = learning.retrieve_relevant(
        {"goal": "research topic alpha future", "task_type": "general", "tools": ["tool_a", "tool_b"]}
    )
    assert len(retrieved) == 1
    assert set(retrieved[0].supporting_experience_ids) == set(task_ids)
    assert retrieved[0].status == "active"
    return store, tools, tool_a, tool_b, task_ids, retrieved[0]


def _first_planned_tool(store: Store, task_id: str) -> str:
    plan_event = next(
        json.loads(event["event_json"])
        for event in store.events_for_task(task_id)
        if "steps" in json.loads(event["event_json"])
    )
    return plan_event["steps"][0]["tool"]


def test_before_after_behavioral_proof_reopens_storage_and_verifies_future_task(tmp_path: Path) -> None:
    path = tmp_path / "agent.sqlite3"
    store, tools, tool_a, tool_b, training_ids, learned = _train_two_verified_recovery_experiences(path)
    assert _first_planned_tool(store, training_ids[0]) == "tool_a"
    assert _first_planned_tool(store, training_ids[1]) == "tool_a"
    strategy_id = learned.strategy_id
    store.close()

    reopened = Store(path)
    try:
        memory = SQLiteLearningMemory(reopened)
        restored = memory.get(strategy_id)
        assert restored is not None
        assert restored.to_dict() == learned.to_dict()
        retrieved = memory.retrieve_relevant(
            {"goal": "research topic alpha follow-up", "task_type": "general", "tools": ["tool_a", "tool_b"]}
        )
        assert [item.strategy_id for item in retrieved] == [strategy_id]

        future = Orchestrator(reopened, tools, planner=Planner(), max_attempts=2).run(
            "research topic alpha follow-up"
        )
        assert future["learned_strategies_used"]
        assert future["learned_strategies_used"][0]["strategy_id"] == strategy_id
        assert future["step_results"][0]["tool_name"] == "tool_b"
        assert future["verification"]["status"] == "VERIFIED"
        assert future["verification"]["verified"] is True
        assert future["final_status"] == "COMPLETED"
        assert _first_planned_tool(reopened, future["task_id"]) == "tool_b"
        assert tool_a.calls == 4, "the future similar task must not execute the previously failing tool again"
        assert tool_b.calls == 3
    finally:
        reopened.close()


@pytest.mark.parametrize("invalid_support", ["missing", "unverified", "stale", "conflicted"])
def test_invalid_or_conflicted_support_cannot_change_plan(tmp_path: Path, invalid_support: str) -> None:
    store, tools, _, _, _, strategy = _train_two_verified_recovery_experiences(
        tmp_path / f"{invalid_support}.sqlite3"
    )
    memory = SQLiteLearningMemory(store)
    query = {"goal": "research topic alpha future", "task_type": "general", "tools": ["tool_a", "tool_b"]}
    assert memory.retrieve_relevant(query)

    if invalid_support == "missing":
        support_id = strategy.supporting_experience_ids[0]
        with store.db:
            store.db.execute("DELETE FROM experiences WHERE task_id = ?", (support_id,))
    elif invalid_support in {"unverified", "stale"}:
        support_id = strategy.supporting_experience_ids[0]
        row = store.get_experience(support_id)
        payload = json.loads(row["experience_json"])
        if invalid_support == "unverified":
            payload["actions"][-1]["verification"]["status"] = "FAILED"
            payload["actions"][-1]["verification"]["verified"] = False
        else:
            payload["created_at"] = "2000-01-01T00:00:00+00:00"
        with store.db:
            store.db.execute(
                "UPDATE experiences SET experience_json = ?, created_at = ? WHERE task_id = ?",
                (json.dumps(payload, ensure_ascii=False), payload["created_at"], support_id),
            )
    else:
        row = store.get_learned_strategy(strategy.strategy_id)
        payload = json.loads(row["strategy_json"])
        payload["metadata"]["conflicting_experience_ids"] = ["new-conflicting-experience"]
        with store.db:
            store.db.execute(
                "UPDATE learned_strategies SET strategy_json = ? WHERE strategy_id = ?",
                (json.dumps(payload, ensure_ascii=False), strategy.strategy_id),
            )

    retrieved = memory.retrieve_relevant(query)
    assert retrieved == []
    plan = Planner().create(
        query["goal"],
        tools,
        [],
        permissions={"tool_a": "public_read", "tool_b": "public_read"},
        learned_strategies=[item.to_dict() for item in retrieved],
    )
    assert plan.steps[0].tool == "tool_a"
    store.close()


def test_inadequate_similarity_unavailable_or_forbidden_tool_cannot_change_plan(tmp_path: Path) -> None:
    store, tools, tool_a, tool_b, _, strategy = _train_two_verified_recovery_experiences(
        tmp_path / "eligibility.sqlite3"
    )
    memory = SQLiteLearningMemory(store)
    similar_query = {
        "goal": "research topic alpha follow-up",
        "task_type": "general",
        "tools": ["tool_a", "tool_b"],
    }
    valid_strategy = [item.to_dict() for item in memory.retrieve_relevant(similar_query)]
    assert valid_strategy

    unrelated_goal = "prepare database backup"
    unrelated_context = {**similar_query, "goal": unrelated_goal}
    assert memory.retrieve_relevant(unrelated_context) == []
    unrelated_plan = Planner().create(unrelated_goal, tools, [], learned_strategies=[])
    assert unrelated_plan.steps[0].tool == "tool_a"

    missing_alternative = registry(tool_a)
    unavailable_plan = Planner(learning_memory=memory).create(
        similar_query["goal"],
        missing_alternative,
        [],
        permissions={"tool_a": "public_read"},
        learned_strategies=valid_strategy,
    )
    assert unavailable_plan.steps[0].tool == "tool_a"

    tool_b.permission_level = "private_write"
    forbidden_plan = Planner(learning_memory=memory).create(
        similar_query["goal"],
        tools,
        [],
        permissions={"tool_a": "public_read"},
        learned_strategies=valid_strategy,
    )
    assert forbidden_plan.steps[0].tool == "tool_a"
    assert tool_b.calls == 2, "planning must not execute a tool"
    assert strategy.preferred_action == {"type": "prefer_tool", "tool": "tool_b"}
    store.close()
