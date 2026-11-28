from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

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
from private_agent.core.observation import Observation
from private_agent.core.planner import PlanStep, PlanValidationError, Planner, TaskPlan
from private_agent.core.replanning import ReplanRequest, Replanner
from private_agent.core.recovery import RecoveryDecision, RecoveryManager
from private_agent.core.reflection import (
    ReflectionEngine,
    ReflectionInsight,
    SQLiteReflectionMemory,
)
from private_agent.core.verification import GenericVerifier, VerificationResult, VerifierRegistry
from private_agent.bugbounty.reviews import FindingReview, FindingReviewRepository
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry
from private_agent.worker import BackgroundWorker


OWNER_PASSWORD = "phase7-learning-owner-password-123"
EXPECTED_SYNTHETIC_RESULT = "verified alternative evidence"
TRUSTED_SYNTHETIC_VERIFIER_NAMES = frozenset({"independent_synthetic_fixture"})


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


class IndependentSyntheticVerifier:
    """Test-only oracle that checks the tool output against a separate local fixture."""

    name = "independent_synthetic_fixture"

    def verify(self, observation: Observation, criteria: dict[str, Any]) -> VerificationResult:
        if observation.execution_status != "success":
            return VerificationResult(
                "FAILED",
                "Synthetic execution did not succeed",
                verifier_type="untrusted-self-claim",
                details={"independent_fixture_check": True},
            )
        output = observation.observed_output
        if not isinstance(output, dict) or output.get("result") != EXPECTED_SYNTHETIC_RESULT:
            return VerificationResult(
                "FAILED",
                "Output did not match the independent local fixture",
                verifier_type="untrusted-self-claim",
                details={"independent_fixture_check": True},
            )
        digest = hashlib.sha256(EXPECTED_SYNTHETIC_RESULT.encode("utf-8")).hexdigest()
        return VerificationResult(
            "VERIFIED",
            "Independent verifier matched the expected synthetic fixture value",
            evidence=[{"type": "synthetic_fixture_digest", "sha256": digest}],
            verifier_type="untrusted-self-claim",
            details={"independent_fixture_check": True, "expected_sha256": digest},
        )


class SelfClaimingVerifier:
    name = "self_claiming_verifier"

    def verify(self, observation: Observation, criteria: dict[str, Any]) -> VerificationResult:
        return VerificationResult(
            "VERIFIED",
            "Verifier-controlled claims must not establish trust",
            evidence=[{"type": "claimed_evidence"}],
            verifier_type="independent_synthetic_fixture",
            learning_trusted=True,
        )


class ReplacementSyntheticVerifier:
    name = "independent_synthetic_fixture"

    def verify(self, observation: Observation, criteria: dict[str, Any]) -> VerificationResult:
        return VerificationResult(
            "VERIFIED",
            "Replacement verifier controlled the response",
            evidence=[{"type": "replacement_claim"}],
            details={"replacement_used": True},
        )


def trusted_synthetic_verifiers() -> VerifierRegistry:
    return VerifierRegistry(
        [IndependentSyntheticVerifier()],
        trusted_for_learning=set(TRUSTED_SYNTHETIC_VERIFIER_NAMES),
    )


def trusted_learning_memory(store: Store) -> SQLiteLearningMemory:
    return SQLiteLearningMemory(store, trusted_verifier_names=set(TRUSTED_SYNTHETIC_VERIFIER_NAMES))


def test_learning_trust_comes_from_server_allowlist_not_verifier_result_claims() -> None:
    observation = Observation(
        task_id="task-synthetic",
        step_id="step-synthetic",
        tool_name="tool_b",
        execution_status="success",
        observed_output={"result": EXPECTED_SYNTHETIC_RESULT},
    )
    generic = VerifierRegistry(
        [GenericVerifier()],
        trusted_for_learning={"generic"},
    )
    generic_result = generic.verify("generic", observation, {"required_fields": ["result"]})
    assert generic_result.status == "VERIFIED"
    assert generic_result.verifier_type == "generic"
    assert generic_result.learning_trusted is False
    assert generic.trusted_for_learning_names == frozenset()

    self_claiming = VerifierRegistry([SelfClaimingVerifier()])
    claimed_result = self_claiming.verify("self_claiming_verifier", observation)
    assert claimed_result.status == "VERIFIED"
    assert claimed_result.verifier_type == "self_claiming_verifier"
    assert claimed_result.learning_trusted is False

    allowlisted = trusted_synthetic_verifiers()
    synthetic_result = allowlisted.verify("independent_synthetic_fixture", observation)
    assert synthetic_result.status == "VERIFIED"
    assert synthetic_result.verifier_type == "independent_synthetic_fixture"
    assert synthetic_result.learning_trusted is True
    with pytest.raises(ValueError, match="verifier_name_already_registered"):
        allowlisted.register(ReplacementSyntheticVerifier())
    assert allowlisted.is_trusted_for_learning("independent_synthetic_fixture") is True

    replaced_registry = trusted_synthetic_verifiers()
    replaced_registry._verifiers["independent_synthetic_fixture"] = ReplacementSyntheticVerifier()
    replaced_result = replaced_registry.verify("independent_synthetic_fixture", observation)
    assert replaced_result.details["replacement_used"] is True
    assert replaced_result.learning_trusted is False
    assert "independent_synthetic_fixture" not in replaced_registry.trusted_for_learning_names


class CapturingPlanner(Planner):
    def __init__(self, verifier_name: str = "") -> None:
        super().__init__()
        self.experience_contexts: list[list[dict[str, Any]]] = []
        self.verifier_name = verifier_name

    def create(
        self,
        goal: str,
        tools: ToolRegistry,
        prior_knowledge: list[dict[str, Any]],
        *,
        constraints: dict[str, Any] | None = None,
        permissions: dict[str, str] | None = None,
        retrieved_experiences: list[dict[str, Any]] | None = None,
        reflection_insights: list[dict[str, Any]] | None = None,
        learned_strategies: list[dict[str, Any]] | None = None,
        clarification_answers: list[dict[str, Any]] | None = None,
        task_id: str = "",
    ) -> TaskPlan:
        self.experience_contexts.append(list(retrieved_experiences or []))
        plan = super().create(
            goal,
            tools,
            prior_knowledge,
            constraints=constraints,
            permissions=permissions,
            retrieved_experiences=retrieved_experiences,
            reflection_insights=reflection_insights,
            learned_strategies=learned_strategies,
            clarification_answers=clarification_answers,
            task_id=task_id,
        )
        if self.verifier_name:
            for step in plan.steps:
                step.verifier = self.verifier_name
        return plan


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
        Planner(provider, learning_memory=trusted_learning_memory(store)).create(
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


def _train_verified_recovery_experiences(
    path: Path,
    *,
    count: int = 2,
    verifiers: VerifierRegistry | None = None,
):
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
    verifier_registry = verifiers or trusted_synthetic_verifiers()
    trusted_names = verifier_registry.trusted_for_learning_names
    selected_verifier = next(iter(sorted(trusted_names)), "")
    agent = Orchestrator(
        store,
        tools,
        planner=CapturingPlanner(verifier_name=selected_verifier),
        verifiers=verifier_registry,
        max_attempts=2,
    )
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
    learning = trusted_learning_memory(store)
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
    verifiers = trusted_synthetic_verifiers()
    store, tools, tool_a, tool_b, training_ids, learned = _train_verified_recovery_experiences(
        path,
        verifiers=verifiers,
    )
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
        memory = trusted_learning_memory(reopened)
        restored = memory.get(strategy_id)
        assert restored is not None
        assert restored.to_dict() == learned.to_dict()
        with pytest.raises(ValueError, match="verifier_name_already_registered"):
            verifiers.register(ReplacementSyntheticVerifier())
        assert verifiers.is_trusted_for_learning("independent_synthetic_fixture") is True
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
            assert payload["actions"][-1]["verification"]["verifier_type"] == "independent_synthetic_fixture"
            assert payload["actions"][-1]["verification"]["details"]["independent_fixture_check"] is True

        future_agent = Orchestrator(
            reopened,
            tools,
            planner=CapturingPlanner(verifier_name="independent_synthetic_fixture"),
            verifiers=verifiers,
            max_attempts=2,
        )
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
        planning_experiences = future_state["context"]["retrieved_experiences"]
        assert {item["experience_id"] for item in planning_experiences} == set(training_ids)
        assert all(item["identity"]["verification_status"] == "VERIFIED" for item in planning_experiences)
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
        independent_verification = future["final_step_outcomes"][-1]["verification"]
        assert independent_verification["verifier_type"] == "independent_synthetic_fixture"
        assert independent_verification["details"]["independent_fixture_check"] is True
        assert independent_verification["evidence"][0]["type"] == "synthetic_fixture_digest"
        assert future["final_status"] == "COMPLETED"
        assert _first_planned_tool(reopened, future_task_id) == "tool_b"
        assert tool_a.calls == 4, "the future similar task must not execute the previously failing tool again"
        assert tool_b.calls == 3
    finally:
        reopened.close()


@pytest.mark.parametrize("invalid_support", ["unverified", "mismatched", "stale", "conflicted", "missing-tool"])
def test_orchestrator_does_not_forward_unrevalidated_experiences(
    tmp_path: Path,
    invalid_support: str,
) -> None:
    path = tmp_path / f"untrusted-{invalid_support}.sqlite3"
    verifiers = trusted_synthetic_verifiers()
    store, tools, tool_a, tool_b, training_ids, strategy = _train_verified_recovery_experiences(
        path,
        verifiers=verifiers,
    )
    support_id = strategy.supporting_experience_ids[0]
    if invalid_support == "conflicted":
        row = store.get_learned_strategy(strategy.strategy_id)
        payload = json.loads(row["strategy_json"])
        payload["metadata"]["conflicting_experience_ids"] = ["conflicting-support"]
        with store.db:
            store.db.execute(
                "UPDATE learned_strategies SET strategy_json = ? WHERE strategy_id = ?",
                (json.dumps(payload, ensure_ascii=False), strategy.strategy_id),
            )
    elif invalid_support != "missing-tool":
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

    execution_tools = registry(tool_a) if invalid_support == "missing-tool" else tools
    permissions = {"tool_a": "public_read"}
    if invalid_support != "missing-tool":
        permissions["tool_b"] = "public_read"
    baseline = Planner().create(
        "research topic alpha follow-up",
        execution_tools,
        [],
        permissions=permissions,
    )
    assert baseline.steps[0].tool == "tool_a"
    tool_a.outcomes = [ToolResult.success({"result": EXPECTED_SYNTHETIC_RESULT})]
    planner = CapturingPlanner(verifier_name="independent_synthetic_fixture")
    agent = Orchestrator(store, execution_tools, planner=planner, verifiers=verifiers, max_attempts=2)
    service = AgentAPIService(agent, token=None)
    try:
        with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
            login = client.post(
                "/api/v1/auth/login",
                json={"email": AuthManager(store).owner_email, "password": OWNER_PASSWORD},
            )
            assert login.status_code == 200, login.text
            csrf = str(client.cookies.get("vulnora_csrf"))
            created = client.post(
                "/api/v1/tasks",
                json={"goal": "research topic alpha follow-up"},
                headers={"X-CSRF-Token": csrf, "Idempotency-Key": f"untrusted-memory-{invalid_support}"},
            )
            assert created.status_code == 201, created.text
            task_id = str(created.json()["task_id"])
            outcome = BackgroundWorker(agent, worker_id=f"untrusted-memory-{invalid_support}").run_once()
            assert outcome["status"] == "COMPLETED", outcome
        state = store.get_task_state(task_id)
        assert state is not None
        assert planner.experience_contexts[0] == []
        assert state["context"]["retrieved_experiences"] == []
        if invalid_support != "missing-tool":
            assert state["context"]["learned_strategies_used"] == []
        assert _first_planned_tool(store, task_id) == "tool_a"
        assert tool_a.calls == 5
        assert tool_b.calls == 2
        assert set(training_ids) == set(strategy.supporting_experience_ids)
    finally:
        store.close()


def test_learned_action_execution_unknown_is_not_replayed_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "learned-execution-unknown.sqlite3"
    verifiers = trusted_synthetic_verifiers()
    store, tools, tool_a, tool_b, _training_ids, _learned = _train_verified_recovery_experiences(
        path,
        verifiers=verifiers,
    )
    store.close()

    store = Store(path)

    def interrupt_after_execution(checkpoint_type: str, state: dict[str, Any]) -> None:
        if checkpoint_type == "AFTER_EXECUTION":
            raise RuntimeError("controlled interruption after synthetic effect")

    agent = Orchestrator(
        store,
        tools,
        planner=CapturingPlanner(verifier_name="independent_synthetic_fixture"),
        verifiers=verifiers,
        checkpoint_hook=interrupt_after_execution,
        max_attempts=2,
    )
    service = AgentAPIService(agent, token=None)
    try:
        with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
            login = client.post(
                "/api/v1/auth/login",
                json={"email": AuthManager(store).owner_email, "password": OWNER_PASSWORD},
            )
            assert login.status_code == 200, login.text
            csrf = str(client.cookies.get("vulnora_csrf"))
            created = client.post(
                "/api/v1/tasks",
                json={"goal": "research topic alpha follow-up"},
                headers={"X-CSRF-Token": csrf, "Idempotency-Key": "learned-unknown-effect"},
            )
            assert created.status_code == 201, created.text
            task_id = str(created.json()["task_id"])
            result = BackgroundWorker(agent, worker_id="learned-unknown-worker").run_once()
            assert result["status"] == "EXECUTION_UNKNOWN", result

        state = store.get_task_state(task_id)
        assert state is not None and state["status"] == "EXECUTION_UNKNOWN"
        assert state["plan"]["steps"][0]["tool"] == "tool_b"
        assert tool_a.calls == 4
        assert tool_b.calls == 3, "the synthetic effect should occur exactly once before the interruption"
        assert store.get_experience(task_id) is None
    finally:
        store.close()

    reopened = Store(path)
    try:
        resumed_agent = Orchestrator(
            reopened,
            tools,
            planner=CapturingPlanner(verifier_name="independent_synthetic_fixture"),
            verifiers=verifiers,
            max_attempts=2,
        )
        idle_result = BackgroundWorker(resumed_agent, worker_id="learned-reopened-worker").run_once()
        assert idle_result == {"status": "NO_TASK", "processed": False, "task_id": ""}
        recovered_state = reopened.get_task_state(task_id)
        assert recovered_state is not None and recovered_state["status"] == "EXECUTION_UNKNOWN"
        assert tools.get("tool_b") is tool_b
        assert tool_a.calls == 4
        assert tool_b.calls == 3, "reopening and polling must not duplicate the uncertain effect"
        assert reopened.get_experience(task_id) is None
    finally:
        reopened.close()


def test_trusted_retrieved_strategy_only_guides_compatible_permitted_fallback(tmp_path: Path) -> None:
    store, _, _, _, training_ids, learned = _train_verified_recovery_experiences(
        tmp_path / "trusted-fallback.sqlite3"
    )
    try:
        memory = trusted_learning_memory(store)
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
    memory = trusted_learning_memory(store)
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


def test_generic_verifier_claim_in_sqlite_cannot_reactivate_strategy(tmp_path: Path) -> None:
    store, _, _, _, training_ids, strategy = _train_verified_recovery_experiences(
        tmp_path / "generic-provenance.sqlite3"
    )
    support_id = strategy.supporting_experience_ids[0]
    row = store.get_experience(support_id)
    payload = json.loads(row["experience_json"])
    verification = payload["actions"][-1]["verification"]
    verification["verifier_type"] = "generic"
    verification["learning_trusted"] = True
    with store.db:
        store.db.execute(
            "UPDATE experiences SET experience_json = ? WHERE task_id = ?",
            (json.dumps(payload, ensure_ascii=False), support_id),
        )

    memory = SQLiteLearningMemory(
        store,
        trusted_verifier_names={"generic", *TRUSTED_SYNTHETIC_VERIFIER_NAMES},
    )
    query = {
        "goal": "research topic alpha future",
        "task_type": "general",
        "tools": ["tool_a", "tool_b"],
        "task_id": training_ids[0],
    }
    assert memory.retrieve_relevant(query) == []
    store.close()


@pytest.mark.parametrize("invalidate_support", [False, True])
def test_recovery_replanning_only_receives_revalidated_memory(
    tmp_path: Path,
    invalidate_support: bool,
) -> None:
    path = tmp_path / f"recovery-memory-{invalidate_support}.sqlite3"
    verifiers = trusted_synthetic_verifiers()
    store, tools, tool_a, tool_b, training_ids, strategy = _train_verified_recovery_experiences(
        path,
        verifiers=verifiers,
    )
    if invalidate_support:
        support_id = strategy.supporting_experience_ids[0]
        row = store.get_experience(support_id)
        payload = json.loads(row["experience_json"])
        payload["actions"][-1]["verification"]["learning_trusted"] = False
        with store.db:
            store.db.execute(
                "UPDATE experiences SET experience_json = ? WHERE task_id = ?",
                (json.dumps(payload, ensure_ascii=False), support_id),
            )

    class FixedInitialPlanner:
        max_steps = 4

        def create(self, goal: str, available_tools: ToolRegistry, prior_knowledge: list[dict[str, Any]], **kwargs: Any):
            plan = Planner().create(
                goal,
                available_tools,
                prior_knowledge,
                permissions=kwargs.get("permissions"),
                constraints=kwargs.get("constraints"),
                task_id=kwargs.get("task_id", ""),
            )
            for step in plan.steps:
                step.verifier = "independent_synthetic_fixture"
            return plan

    class FixedChangeToolRecovery(RecoveryManager):
        def choose(
            self,
            step: PlanStep,
            diagnosis: Any,
            *,
            available_tools: list[dict[str, Any]],
            alternative_plan: PlanStep | None = None,
            allow_replan: bool = False,
        ) -> RecoveryDecision:
            return RecoveryDecision(
                "CHANGE_TOOL",
                "Use the registered local alternative after the synthetic failure",
                step.id,
                step.tool,
                replacement_tool="tool_b",
            )

    class CapturingReplanner(Replanner):
        def __init__(self) -> None:
            super().__init__()
            self.provider = object()  # Force Orchestrator through the ReplanRequest path.
            self.request: ReplanRequest | None = None

        def replan(
            self,
            request: ReplanRequest,
            available_tools: ToolRegistry,
            *,
            replacement_tool: str = "",
        ) -> TaskPlan:
            self.request = request
            return self.replan_step(
                request.current_plan,
                step_id=request.diagnosis.failed_step,
                tool_name=replacement_tool,
                inputs={},
                tools=available_tools,
                permissions=request.permissions,
            )

    tool_a.outcomes.append(ToolResult.failed("network_error", "controlled local failure", no_side_effects=True))
    tool_b.outcomes.append(ToolResult.success({"result": EXPECTED_SYNTHETIC_RESULT}))
    replanner = CapturingReplanner()
    agent = Orchestrator(
        store,
        tools,
        planner=FixedInitialPlanner(),  # type: ignore[arg-type]
        recovery_manager=FixedChangeToolRecovery(max_recovery_attempts=1),
        replanner=replanner,
        verifiers=verifiers,
        max_attempts=2,
    )
    service = AgentAPIService(agent, token=None)
    try:
        with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
            login = client.post(
                "/api/v1/auth/login",
                json={"email": AuthManager(store).owner_email, "password": OWNER_PASSWORD},
            )
            assert login.status_code == 200, login.text
            csrf = str(client.cookies.get("vulnora_csrf"))
            created = client.post(
                "/api/v1/tasks",
                json={"goal": "research topic alpha follow-up"},
                headers={"X-CSRF-Token": csrf, "Idempotency-Key": f"recovery-memory-{invalidate_support}"},
            )
            assert created.status_code == 201, created.text
            task_id = str(created.json()["task_id"])
            worker_result = BackgroundWorker(agent, worker_id=f"recovery-memory-worker-{invalidate_support}").run_once()
            assert worker_result["status"] == "COMPLETED", worker_result

        assert replanner.request is not None
        if invalidate_support:
            assert replanner.request.retrieved_experiences == []
            assert replanner.request.learned_strategies == []
        else:
            assert {item["experience_id"] for item in replanner.request.retrieved_experiences} >= set(training_ids)
            assert [item["strategy_id"] for item in replanner.request.learned_strategies] == [strategy.strategy_id]
        state = store.get_task_state(task_id)
        assert state is not None and state["status"] == "COMPLETED"
        assert tool_a.calls == 5
        assert tool_b.calls == 3
    finally:
        store.close()


def test_inadequate_similarity_unavailable_or_forbidden_tool_cannot_change_plan(tmp_path: Path) -> None:
    store, tools, tool_a, tool_b, training_ids, strategy = _train_verified_recovery_experiences(
        tmp_path / "eligibility.sqlite3"
    )
    memory = trusted_learning_memory(store)
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
