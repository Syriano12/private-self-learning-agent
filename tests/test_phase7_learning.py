from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from private_agent.api import AgentAPIService, create_app
from private_agent.auth import AuthManager
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
from private_agent.bugbounty.reviews import FindingReview, FindingReviewRepository
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry
from private_agent.worker import BackgroundWorker


OWNER_PASSWORD = "phase7-learning-owner-password-123"


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
        return ToolResult.failed("tool_failed", "no configured outcome", no_side_effects=True)


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
    store, tools, _, _, training_ids, learned = _train_verified_recovery_experiences(path)
    try:
        strategy = learned.to_dict()
        Planner(provider, learning_memory=SQLiteLearningMemory(store)).create(
            "research topic Y", tools, [], learned_strategies=[strategy], task_id=training_ids[0]
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
    auth = AuthManager(store)
    owner = auth.initialize_owner(auth.owner_email, OWNER_PASSWORD)
    store.create_queued_task(
        "Bug Bounty finding review IDOR", task_id="task-owner-review", owner_user_id=owner.user_id
    )
    review = FindingReview.create(
        "task-owner-review",
        "finding-id",
        owner.user_id,
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
            goal, tools, [], permissions=permissions, learned_strategies=[strategy], task_id="task-owner-review"
        )
        assert provider.context["learned_strategies"] == [strategy]
        fallback = Planner(learning_memory=memory).create(
            goal, tools, [], permissions=permissions, learned_strategies=[strategy], task_id="task-owner-review"
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
                    recovery_transitions=[{
                        "failed_tool": "tool_a",
                        "alternative_tool": "tool_b",
                        "failure_type": "NETWORK_ERROR",
                        "recovery_strategy": "CHANGE_TOOL",
                        "step_id": "step",
                        "alternative_verified": True,
                    }],
                    failure_actions=[{"tool_name": "tool_a", "step_id": "step", "failure_type": "NETWORK_ERROR"}],
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
                    recovery_transitions=[{
                        "failed_tool": "tool_a",
                        "alternative_tool": "tool_b",
                        "failure_type": "NETWORK_ERROR",
                        "recovery_strategy": "CHANGE_TOOL",
                        "step_id": "step",
                        "alternative_verified": False,
                    }],
                    failure_actions=[{"tool_name": "tool_a", "step_id": "step", "failure_type": "NETWORK_ERROR"}],
                ),
            ]

    insights = ReflectionEngine(Memory(), min_supporting_experiences=1).reflect(
        __import__("private_agent.core.memory", fromlist=["ExperienceQuery"]).ExperienceQuery("research X", task_type="research")
    )
    recovery = next(item for item in insights if item.pattern == "successful_recovery_alternative")
    assert set(recovery.metadata["conflicting_experience_ids"]) == {"conflict-1", "conflict-2"}
    assert recovery.confidence < 0.70


def _train_verified_recovery_experiences(path: Path, *, count: int = 2):
    assert count >= 1
    tool_a = ControlledTool(
        "tool_a",
        [
            ToolResult.failed("network_error", "temporary network failure", no_side_effects=True)
            for _ in range(count * 2)
        ],
    )
    tool_b = ControlledTool(
        "tool_b",
        [
            ToolResult.success({"result": "verified alternative evidence"})
            for _ in range(count + 1)
        ],
    )
    tools = registry(tool_a, tool_b)
    store = Store(path)
    agent = Orchestrator(store, tools, planner=Planner(), max_attempts=2)
    auth = AuthManager(store)
    auth.initialize_owner(auth.owner_email, OWNER_PASSWORD)
    task_ids = []
    service = AgentAPIService(agent, token=None)
    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        login = client.post(
            "/api/v1/auth/login",
            json={"email": auth.owner_email, "password": OWNER_PASSWORD},
        )
        assert login.status_code == 200, login.text
        csrf = str(client.cookies.get("vulnora_csrf"))
        for index in range(count):
            created = client.post(
                "/api/v1/tasks",
                json={"goal": "research topic alpha"},
                headers={"X-CSRF-Token": csrf, "Idempotency-Key": f"phase7-learning-training-{index}"},
            )
            assert created.status_code == 201, created.text
            task_id = str(created.json()["task_id"])
            result = BackgroundWorker(agent, worker_id=f"phase7-learning-worker-{index}").run_once()
            assert result["status"] == "COMPLETED", result
            state = store.get_task_state(task_id)
            assert state is not None
            outcome = state["result"]
            assert outcome["final_status"] == "COMPLETED"
            assert outcome["verification"]["status"] == "VERIFIED"
            assert outcome["step_results"][0]["tool_name"] == "tool_a"
            assert outcome["step_results"][-1]["tool_name"] == "tool_b"
            assert outcome["verification"]["verified"] is True
            task_ids.append(task_id)
    learning = SQLiteLearningMemory(store)
    retrieved = learning.retrieve_relevant(
        {"goal": "research topic alpha future", "task_type": "general", "tools": ["tool_a", "tool_b"], "task_id": task_ids[0]}
    )
    if len(task_ids) < 2:
        assert retrieved == []
        return store, tools, tool_a, tool_b, task_ids, None
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


def test_single_verified_experience_below_default_support_threshold_does_not_steer(tmp_path: Path) -> None:
    store, tools, _, _, task_ids, learned = _train_verified_recovery_experiences(
        tmp_path / "one-support.sqlite3", count=1
    )
    goal = "research topic alpha follow-up"
    permissions = {"tool_a": "public_read", "tool_b": "public_read"}
    try:
        assert learned is None
        assert store.all_learned_strategies() == []
        saved = json.loads(store.get_experience(task_ids[0])["experience_json"])
        assert saved["final_outcome"] == "COMPLETED"
        assert saved["verification_status"] == "VERIFIED"
        assert saved["identity"]["verification_status"] == "VERIFIED"
        assert ReflectionEngine(SQLiteExperienceMemory(store)).min_supporting_experiences == 2

        baseline = Planner().create(goal, tools, [], permissions=permissions)
        with_memory = Planner(learning_memory=SQLiteLearningMemory(store)).create(
            goal,
            tools,
            [],
            permissions=permissions,
            learned_strategies=[],
            task_id=task_ids[0],
        )
        assert baseline.steps[0].tool == "tool_a"
        assert with_memory.steps[0].tool == baseline.steps[0].tool
    finally:
        store.close()


def test_before_after_behavioral_proof_reopens_storage_and_verifies_future_task(tmp_path: Path) -> None:
    path = tmp_path / "agent.sqlite3"
    store, tools, tool_a, tool_b, training_ids, learned = _train_verified_recovery_experiences(path)
    assert _first_planned_tool(store, training_ids[0]) == "tool_a"
    assert _first_planned_tool(store, training_ids[1]) == "tool_a"
    goal = "research topic alpha follow-up"
    permissions = {"tool_a": "public_read", "tool_b": "public_read"}
    baseline = Planner().create(goal, tools, [], permissions=permissions)
    assert baseline.steps[0].tool == "tool_a"
    assert learned.confidence == 0.81
    assert learned.confidence >= Planner().learned_strategy_threshold == 0.70
    assert len(learned.supporting_experience_ids) == 2
    strategy_id = learned.strategy_id
    store.close()

    reopened = Store(path)
    try:
        memory = SQLiteLearningMemory(reopened)
        restored = memory.get(strategy_id)
        assert restored is not None
        assert restored.to_dict() == learned.to_dict()
        assert ReflectionEngine(SQLiteExperienceMemory(reopened)).min_supporting_experiences == 2
        retrieved = memory.retrieve_relevant(
            {"goal": goal, "task_type": "general", "tools": ["tool_a", "tool_b"], "task_id": training_ids[0]}
        )
        assert [item.strategy_id for item in retrieved] == [strategy_id]
        assert set(restored.supporting_experience_ids) == set(training_ids)
        for support_id in training_ids:
            payload = json.loads(reopened.get_experience(support_id)["experience_json"])
            assert payload["final_outcome"] == "COMPLETED"
            assert payload["verification_status"] == "VERIFIED"
            assert payload["identity"]["verification_status"] == "VERIFIED"
            assert payload["actions"][-1]["verification"]["status"] == "VERIFIED"
            assert payload["actions"][-1]["verification"]["verified"] is True

        future_agent = Orchestrator(reopened, tools, planner=Planner(), max_attempts=2)
        future_service = AgentAPIService(future_agent, token=None)
        with TestClient(create_app(service=future_service), raise_server_exceptions=False) as client:
            login = client.post(
                "/api/v1/auth/login",
                json={"email": AuthManager(reopened).owner_email, "password": OWNER_PASSWORD},
            )
            assert login.status_code == 200, login.text
            csrf = str(client.cookies.get("vulnora_csrf"))
            created = client.post(
                "/api/v1/tasks",
                json={"goal": goal},
                headers={"X-CSRF-Token": csrf, "Idempotency-Key": "phase7-learning-future"},
            )
            assert created.status_code == 201, created.text
            future_task_id = str(created.json()["task_id"])
            outcome = BackgroundWorker(future_agent, worker_id="phase7-learning-future-worker").run_once()
            assert outcome["status"] == "COMPLETED", outcome
            future_state = reopened.get_task_state(future_task_id)
            assert future_state is not None
            future = future_state["result"]
        assert future["learned_strategies_used"]
        assert future["learned_strategies_used"][0]["strategy_id"] == strategy_id
        assert len(future["step_results"]) == 1
        executed_step = future["step_results"][0]
        assert executed_step["tool_name"] == "tool_b"
        assert executed_step["status"] == "success"
        assert tools.get("tool_b") is tool_b
        assert tool_b.permission_level == "public_read"
        authorization = future_state["security_decisions"][executed_step["step_id"]]
        assert authorization["action_id"] == executed_step["step_id"]
        assert authorization["task_id"] == future_task_id
        assert authorization["tool_or_skill"] == executed_step["tool_name"]
        assert authorization["decision"] == "ALLOW"
        assert future["verification"]["status"] == "VERIFIED"
        assert future["verification"]["verified"] is True
        assert future["final_status"] == "COMPLETED"
        assert _first_planned_tool(reopened, future_task_id) == "tool_b"
        assert tool_a.calls == 4, "the future similar task must not execute the previously failing tool again"
        assert tool_b.calls == 3
    finally:
        reopened.close()


def test_trusted_retrieved_strategy_only_guides_compatible_permitted_fallback(tmp_path: Path) -> None:
    store, _, _, _, training_ids, learned = _train_verified_recovery_experiences(
        tmp_path / "trusted-fallback.sqlite3"
    )
    try:
        memory = SQLiteLearningMemory(store)
        goal = "research topic alpha follow-up"
        task_id = training_ids[0]
        context = {
            "goal": goal,
            "task_type": "general",
            "tools": ["tool_a", "tool_b"],
            "task_id": task_id,
        }
        retrieved = memory.retrieve_relevant(context)
        assert [item.strategy_id for item in retrieved] == [learned.strategy_id]
        trusted_guidance = [item.to_dict() for item in retrieved]

        def create_plan(tools: ToolRegistry, permissions: dict[str, str]):
            return Planner(learning_memory=memory).create(
                goal,
                tools,
                [],
                permissions=permissions,
                learned_strategies=trusted_guidance,
                task_id=task_id,
            )

        compatible_tools = registry(ControlledTool("tool_a", []), ControlledTool("tool_b", []))
        compatible_plan = create_plan(
            compatible_tools,
            {"tool_a": "public_read", "tool_b": "public_read"},
        )
        assert compatible_plan.steps[0].tool == "tool_b"

        class ScopeRequiredControlledTool(ControlledTool):
            def input_schema(self) -> dict:
                return {
                    "type": "object",
                    "properties": {"scope": {"type": "object"}},
                    "required": ["scope"],
                    "additionalProperties": False,
                }

        incompatible_tools = registry(
            ControlledTool("tool_a", []), ScopeRequiredControlledTool("tool_b", [])
        )
        incompatible_plan = create_plan(
            incompatible_tools,
            {"tool_a": "public_read", "tool_b": "public_read"},
        )
        assert incompatible_plan.steps[0].tool == "tool_a"

        permission_limited_tools = registry(ControlledTool("tool_a", []), ControlledTool("tool_b", []))
        permission_limited_plan = create_plan(
            permission_limited_tools,
            {"tool_a": "public_read"},
        )
        assert permission_limited_plan.steps[0].tool == "tool_a"
    finally:
        store.close()


@pytest.mark.parametrize(
    "invalid_support",
    ["missing", "unverified", "mismatched", "stale", "conflicted", "malformed_failure_types", "malformed_tools"],
)
def test_invalid_or_conflicted_support_cannot_change_plan(tmp_path: Path, invalid_support: str) -> None:
    store, tools, _, _, training_ids, strategy = _train_verified_recovery_experiences(
        tmp_path / f"{invalid_support}.sqlite3"
    )
    memory = SQLiteLearningMemory(store)
    query = {"goal": "research topic alpha future", "task_type": "general", "tools": ["tool_a", "tool_b"], "task_id": training_ids[0]}
    assert memory.retrieve_relevant(query)

    if invalid_support == "missing":
        support_id = strategy.supporting_experience_ids[0]
        with store.db:
            store.db.execute("DELETE FROM experiences WHERE task_id = ?", (support_id,))
    elif invalid_support in {"unverified", "mismatched", "stale"}:
        support_id = strategy.supporting_experience_ids[0]
        row = store.get_experience(support_id)
        payload = json.loads(row["experience_json"])
        if invalid_support == "unverified":
            payload["actions"][-1]["verification"]["status"] = "FAILED"
            payload["actions"][-1]["verification"]["verified"] = False
        elif invalid_support == "mismatched":
            payload["identity"]["verification_status"] = "FAILED"
        else:
            payload["created_at"] = "2000-01-01T00:00:00+00:00"
        with store.db:
            store.db.execute(
                "UPDATE experiences SET experience_json = ?, created_at = ? WHERE task_id = ?",
                (json.dumps(payload, ensure_ascii=False), payload["created_at"], support_id),
            )
    elif invalid_support == "conflicted":
        row = store.get_learned_strategy(strategy.strategy_id)
        payload = json.loads(row["strategy_json"])
        payload["metadata"]["conflicting_experience_ids"] = ["new-conflicting-experience"]
        with store.db:
            store.db.execute(
                "UPDATE learned_strategies SET strategy_json = ? WHERE strategy_id = ?",
                (json.dumps(payload, ensure_ascii=False), strategy.strategy_id),
            )
    else:
        row = store.get_learned_strategy(strategy.strategy_id)
        payload = json.loads(row["strategy_json"])
        if invalid_support == "malformed_failure_types":
            payload["condition"]["failure_types"] = ["NETWORK_ERROR", []]
        else:
            payload["condition"]["tools"] = ["tool_a", {"name": "tool_b"}]
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
    store, tools, tool_a, tool_b, training_ids, strategy = _train_verified_recovery_experiences(
        tmp_path / "eligibility.sqlite3"
    )
    memory = SQLiteLearningMemory(store)
    similar_query = {
        "goal": "research topic alpha follow-up",
        "task_type": "general",
        "tools": ["tool_a", "tool_b"],
        "task_id": training_ids[0],
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
        task_id=training_ids[0],
    )
    assert unavailable_plan.steps[0].tool == "tool_a"

    tool_b.permission_level = "private_write"
    forbidden_plan = Planner(learning_memory=memory).create(
        similar_query["goal"],
        tools,
        [],
        permissions={"tool_a": "public_read"},
        learned_strategies=valid_strategy,
        task_id=training_ids[0],
    )
    assert forbidden_plan.steps[0].tool == "tool_a"
    assert tool_b.calls == 2, "planning must not execute a tool"
    assert strategy.preferred_action == {"type": "prefer_tool", "tool": "tool_b"}
    store.close()
