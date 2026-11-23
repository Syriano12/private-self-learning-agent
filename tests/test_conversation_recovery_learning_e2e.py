from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from private_agent.api import AgentAPIService, create_app
from private_agent.auth import AuthManager
from private_agent.core.learning import SQLiteLearningMemory
from private_agent.core.knowledge import KnowledgeAttestor, KnowledgeClaim, KnowledgeEvidence, KnowledgeSource, sha256_hex
from private_agent.core.llm import LLMProvider, LLMResponse
from private_agent.core.memory import ExperienceQuery, SQLiteExperienceMemory
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.planner import Planner
from private_agent.core.recovery import RecoveryManager
from private_agent.core.reflection import SQLiteReflectionMemory
from private_agent.observability import ObservabilityMonitor
from private_agent.security import SecurityController
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry
from private_agent.worker import BackgroundWorker

OWNER_PASSWORD = "recovery-learning-owner-password-123"
OPERATOR_PASSWORD = "recovery-learning-operator-password-123"
KNOWLEDGE_KEY = b"recovery-learning-test-attestation-key-32"
SUPPORT_GOAL = "Please research local artifact alpha and verify a recovery result"
FUTURE_GOAL = "Please research local artifact alpha for a future verified recovery"


class LocalDeterministicProvider(LLMProvider):
    """Test-only provider that proposes JSON; Planner still validates every plan."""

    def __init__(self) -> None:
        self.contexts: list[dict[str, Any]] = []
        self.initial_choices: list[str] = []

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> LLMResponse:
        del system_prompt, response_schema
        context = json.loads(user_prompt)
        self.contexts.append(context)
        goal = str(context["goal"])

        if "current_plan" in context:
            data = {
                "goal": goal,
                "rationale": "Deterministic local recovery plan after the diagnosed invalid primary-tool output",
                "steps": [
                    {
                        "id": "safe-prerequisite",
                        "objective": "run a benign local prerequisite",
                        "tool": "safe_probe",
                        "input": {"value": "local-check"},
                        "reason": "registered benign deterministic prerequisite",
                        "depends_on": [],
                        "success_criteria": {"required_fields": ["result"]},
                    },
                    {
                        "id": "primary-step",
                        "objective": "verify the local recovery result with the alternative",
                        "tool": "tool_b",
                        "input": {"value": "alpha"},
                        "reason": "registered alternative selected after INVALID_OUTPUT",
                        "depends_on": ["safe-prerequisite"],
                        "success_criteria": {"required_fields": ["result"]},
                    },
                ],
            }
            return LLMResponse(data, json.dumps(data, sort_keys=True), "local-deterministic")

        alternative = next(
            (
                item.get("preferred_action", {}).get("tool")
                for item in context.get("learned_strategies", [])
                if isinstance(item, dict)
                and item.get("status") == "active"
                and item.get("metadata", {}).get("source_patterns") == ["successful_recovery_alternative"]
                and item.get("condition", {}).get("failed_tool") == "tool_a"
                and item.get("condition", {}).get("alternative_tool") == "tool_b"
            ),
            None,
        )
        selected = "tool_b" if alternative == "tool_b" else "tool_a"
        self.initial_choices.append(selected)
        steps = []
        if selected == "tool_b":
            steps.append(
                {
                    "id": "safe-prerequisite",
                    "objective": "run a benign local prerequisite",
                    "tool": "safe_probe",
                    "input": {"value": "local-check"},
                    "reason": "registered benign deterministic prerequisite",
                    "depends_on": [],
                    "success_criteria": {"required_fields": ["result"]},
                }
            )
        steps.append(
            {
                "id": "primary-step",
                "objective": "verify the local recovery result",
                "tool": selected,
                "input": {"value": "alpha"},
                "reason": "deterministic local plan using registered tools only",
                "depends_on": ["safe-prerequisite"] if selected == "tool_b" else [],
                "success_criteria": {"required_fields": ["result"]},
            }
        )
        data = {
            "goal": goal,
            "rationale": "Deterministic local plan; learned guidance is advisory and server validation remains authoritative",
            "steps": steps,
        }
        return LLMResponse(data, json.dumps(data, sort_keys=True), "local-deterministic")


class LocalTool:
    permission_level = "public_read"
    planned_effect = "benign deterministic local acceptance fixture"
    required_capabilities: list[str] = []
    declared_capabilities: list[str] = []
    network_required = False

    def __init__(self, name: str, calls: dict[str, list[dict[str, Any]]]) -> None:
        self.name = name
        self.calls = calls[name]
        self.risk_level = "high" if name == "tool_b" else "low"

    def description(self) -> str:
        return f"Benign local deterministic fixture: {self.name}"

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        }

    def output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
            "additionalProperties": False,
        }

    def execute(self, inputs: dict[str, Any], context: Any) -> ToolResult:
        self.calls.append({"task_id": context.task_id, "inputs": dict(inputs)})
        if self.name == "tool_a":
            return ToolResult.failed("malformed_tool_result", "deterministic local invalid-output fixture")
        return ToolResult.success({"result": f"verified-local:{self.name}:{inputs['value']}:task={context.task_id}"})


def _build_components(
    database: Path,
    calls: dict[str, list[dict[str, Any]]],
    provider: LocalDeterministicProvider,
    *,
    knowledge_attestation_key: bytes | None = None,
) -> tuple[Store, AgentAPIService, Orchestrator]:
    store = Store(database, knowledge_attestation_key=knowledge_attestation_key)
    monitor = ObservabilityMonitor(store)
    security = SecurityController(store=store, observability=monitor)
    tools = ToolRegistry()
    # The registry order ensures the recovery manager selects tool_b for tool_a.
    for name in ("tool_a", "tool_b", "safe_probe"):
        tools.register(LocalTool(name, calls))
    planner = Planner(provider)
    orchestrator = Orchestrator(
        store,
        tools,
        planner=planner,
        security_controller=security,
        observability=monitor,
        recovery_manager=RecoveryManager(max_recovery_attempts=2, max_attempts_per_step=1),
    )
    return store, AgentAPIService(orchestrator, token=None), orchestrator


def _login(client: TestClient, email: str, password: str = OWNER_PASSWORD) -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert response.status_code == 200, response.text
    csrf = client.cookies.get("vulnora_csrf")
    assert csrf
    return str(csrf)


def _create_task(client: TestClient, csrf: str, goal: str, key: str) -> str:
    response = client.post(
        "/api/v1/tasks",
        json={"goal": goal},
        headers={"X-CSRF-Token": csrf, "Idempotency-Key": key},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    task_id = str(body["task_id"])
    assert body["conversation_id"] == task_id
    conversation = client.get(f"/api/v1/tasks/{task_id}/messages")
    assert conversation.status_code == 200, conversation.text
    messages = conversation.json()["messages"]
    assert len(messages) == 1
    assert messages[0]["message_kind"] == "INITIAL"
    assert messages[0]["content"] == goal
    return task_id


def _calls_for(calls: dict[str, list[dict[str, Any]]], tool: str, task_id: str) -> list[dict[str, Any]]:
    return [item for item in calls[tool] if item["task_id"] == task_id]


def _records_for_step(state: dict[str, Any], step_id: str) -> list[dict[str, Any]]:
    return [
        record
        for record in state["action_records"].values()
        if isinstance(record, dict) and record.get("step_id") == step_id
    ]


def _run_support_task(
    database: Path,
    calls: dict[str, list[dict[str, Any]]],
    *,
    idempotency_key: str,
    initialize_owner: bool,
    goal: str = SUPPORT_GOAL,
) -> tuple[Store, str, LocalDeterministicProvider]:
    provider = LocalDeterministicProvider()
    store, service, orchestrator = _build_components(database, calls, provider)
    auth = AuthManager(store)
    if initialize_owner:
        auth.initialize_owner(auth.owner_email, OWNER_PASSWORD)

    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        csrf = _login(client, auth.owner_email)
        task_id = _create_task(client, csrf, goal, idempotency_key)
        waiting = BackgroundWorker(orchestrator, worker_id=f"{idempotency_key}-first").run_once()
        assert waiting["status"] == "WAITING_APPROVAL", waiting
        assert waiting["task_id"] == task_id
        state = store.get_task_state(task_id)
        assert state is not None and state["status"] == "WAITING_APPROVAL"
        assert state["plan"]["steps"][0]["id"] == "safe-prerequisite"
        assert state["plan"]["steps"][0]["tool"] == "safe_probe"
        assert state["plan"]["steps"][1]["tool"] == "tool_b"
        approval_id = str(state["approval_ids"]["primary-step"])
        prerequisite = _records_for_step(state, "safe-prerequisite")
        assert len(prerequisite) == 1 and prerequisite[0]["status"] == "VERIFIED"
        assert len(_calls_for(calls, "safe_probe", task_id)) == 1
        assert len(_calls_for(calls, "tool_a", task_id)) == 1
        assert _calls_for(calls, "tool_b", task_id) == []
        messages = client.get(f"/api/v1/tasks/{task_id}/messages")
        assert messages.status_code == 200
        assert messages.json()["messages"][0]["status"] == "CONSUMED"

    # Reopen SQLite and rebuild service/worker; pending approval must not replay either prior action.
    store.close()
    provider = LocalDeterministicProvider()
    store, service, orchestrator = _build_components(database, calls, provider)
    reopened = store.get_task_state(task_id)
    assert reopened is not None and reopened["status"] == "WAITING_APPROVAL"
    assert len(_calls_for(calls, "safe_probe", task_id)) == 1
    assert len(_calls_for(calls, "tool_a", task_id)) == 1
    owner_email = AuthManager(store).owner_email

    worker = BackgroundWorker(orchestrator, worker_id=f"{idempotency_key}-restarted")
    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        assert _login(client, owner_email)
        assert worker.run_once()["status"] == "NO_TASK"
        assert len(_calls_for(calls, "safe_probe", task_id)) == 1
        assert len(_calls_for(calls, "tool_a", task_id)) == 1
        approval = store.get_approval_request(approval_id)
        assert approval is not None and approval["status"] == "PENDING"
        csrf = str(client.cookies.get("vulnora_csrf"))
        approved = client.post(
            f"/api/v1/approvals/{approval_id}/approve",
            json={},
            headers={"X-CSRF-Token": csrf},
        )
        assert approved.status_code == 200, approved.text
        assert approved.json()["status"] == "APPROVED"
        assert store.task_queue_record(task_id)["queue_status"] == "WAITING_APPROVAL"
        worker._reconcile_approved_waiting_tasks()
        assert store.task_queue_record(task_id)["queue_status"] == "QUEUED"
        completed = worker.run_once()
        assert completed["status"] == "COMPLETED", completed
        assert completed["task_id"] == task_id
        completed_state = store.get_task_state(task_id)
        assert completed_state is not None and completed_state["status"] == "COMPLETED"
        assert completed_state["result"]["final_status"] == "COMPLETED"
        assert completed_state["result"]["verification"]["status"] == "VERIFIED"
        assert completed_state["result"]["verification"]["verified"] is True
        assert len(_calls_for(calls, "safe_probe", task_id)) == 1
        assert len(_calls_for(calls, "tool_a", task_id)) == 1
        assert len(_calls_for(calls, "tool_b", task_id)) == 1
        state = store.get_task_state(task_id)
        assert state is not None and state["status"] == "COMPLETED"
        failed_primary = [
            record for record in _records_for_step(state, "primary-step")
            if record.get("tool_name") == "tool_a"
        ]
        verified_alternative = [
            record for record in _records_for_step(state, "primary-step")
            if record.get("tool_name") == "tool_b"
        ]
        assert len(failed_primary) == 1 and failed_primary[0]["status"] == "FAILED"
        assert len(verified_alternative) == 1 and verified_alternative[0]["status"] == "VERIFIED"
        conversation = client.get(f"/api/v1/tasks/{task_id}/messages")
        assert conversation.status_code == 200
        assert conversation.json()["messages"][0]["content"] == goal
        assert conversation.json()["messages"][0]["status"] == "CONSUMED"

    experience_row = store.get_experience(task_id)
    assert experience_row is not None
    experience = json.loads(experience_row["experience_json"])
    failed_action = next(action for action in experience["actions"] if action["tool_name"] == "tool_a")
    assert failed_action["diagnosis"]["failure_type"] == "INVALID_OUTPUT"
    assert failed_action["recovery"]["strategy"] == "CHANGE_TOOL"
    assert failed_action["recovery"]["replacement_tool"] == "tool_b"
    assert experience["final_outcome"] == "COMPLETED"
    assert experience["verification_status"] == "VERIFIED"
    return store, task_id, provider


def test_conversation_approval_restart_and_future_verified_recovery_learning(tmp_path: Path) -> None:
    database = tmp_path / "conversation-recovery-learning.sqlite3"
    calls: dict[str, list[dict[str, Any]]] = {name: [] for name in ("tool_a", "tool_b", "safe_probe")}

    store, first_id, _ = _run_support_task(
        database,
        calls,
        idempotency_key="recovery-support-one",
        initialize_owner=True,
    )
    assert SQLiteLearningMemory(store).list() == []
    store.close()

    # The second independently created task supplies the second verified support record.
    store, second_id, _ = _run_support_task(
        database,
        calls,
        idempotency_key="recovery-support-two",
        initialize_owner=False,
    )
    assert first_id != second_id
    strategies = SQLiteLearningMemory(store).list()
    assert len(strategies) == 1
    learned = strategies[0]
    assert learned.status == "active"
    assert learned.preferred_action == {"type": "prefer_tool", "tool": "tool_b"}
    assert set(learned.supporting_experience_ids) == {first_id, second_id}
    assert len(learned.supporting_experience_ids) == 2
    assert learned.metadata["source_patterns"] == ["successful_recovery_alternative"]
    assert learned.metadata["conflicting_experience_ids"] == []
    assert store.search_knowledge_claims("local artifact alpha") == []
    strategy_id = learned.strategy_id
    store.close()

    # A fresh process and SQLite connection must retrieve and revalidate the durable strategy.
    store = Store(database)
    memory = SQLiteLearningMemory(store)
    restored = memory.get(strategy_id)
    assert restored is not None and restored.to_dict() == learned.to_dict()
    future_retrieval = memory.retrieve_relevant(
        {"goal": FUTURE_GOAL, "task_type": "general", "tools": ["tool_a", "tool_b", "safe_probe"]}
    )
    assert [item.strategy_id for item in future_retrieval] == [strategy_id]
    experiences = SQLiteExperienceMemory(store).retrieve(
        ExperienceQuery(FUTURE_GOAL, task_type="general", tools=["tool_a", "tool_b", "safe_probe"])
    )
    supported = {item.experience_id: item for item in experiences}
    assert {first_id, second_id}.issubset(supported)
    for experience_id in (first_id, second_id):
        transitions = supported[experience_id].recovery_transitions
        assert any(
            transition["failed_tool"] == "tool_a"
            and transition["alternative_tool"] == "tool_b"
            and transition["failure_type"] == "INVALID_OUTPUT"
            and transition["alternative_verified"] is True
            for transition in transitions
        )

    calls_before_future = {name: len(values) for name, values in calls.items()}
    provider = LocalDeterministicProvider()
    store, service, orchestrator = _build_components(database, calls, provider)
    owner_email = AuthManager(store).owner_email
    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        csrf = _login(client, owner_email)
        future_id = _create_task(client, csrf, FUTURE_GOAL, "recovery-future-task")
        waiting = BackgroundWorker(orchestrator, worker_id="recovery-future-worker").run_once()
        assert waiting["status"] == "WAITING_APPROVAL", waiting
        assert provider.initial_choices == ["tool_b"]
        plan = store.get_task_state(future_id)["plan"]
        assert [step["tool"] for step in plan["steps"]] == ["safe_probe", "tool_b"]
        assert len(_calls_for(calls, "tool_a", future_id)) == 0
        assert len(_calls_for(calls, "safe_probe", future_id)) == 0
        approval_id = str(store.get_task_state(future_id)["approval_ids"]["primary-step"])
        approval = store.get_approval_request(approval_id)
        assert approval is not None and approval["status"] == "PENDING"
        assert approval["tool_or_skill"] == "tool_b"
        csrf = str(client.cookies.get("vulnora_csrf"))
        approved = client.post(
            f"/api/v1/approvals/{approval_id}/approve",
            json={},
            headers={"X-CSRF-Token": csrf},
        )
        assert approved.status_code == 200, approved.text
        worker = BackgroundWorker(orchestrator, worker_id="recovery-future-approved-worker")
        worker._reconcile_approved_waiting_tasks()
        result = worker.run_once()
        assert result["status"] == "COMPLETED", result
        future_state = store.get_task_state(future_id)
        assert future_state is not None and future_state["status"] == "COMPLETED"
        assert future_state["result"]["final_status"] == "COMPLETED"
        assert future_state["result"]["verification"]["status"] == "VERIFIED"
        assert future_state["result"]["verification"]["verified"] is True
        assert future_state["result"]["learned_strategies_used"][0]["strategy_id"] == strategy_id
        assert future_state["result"]["learned_strategies_used"][0]["preferred_action"]["tool"] == "tool_b"
        assert future_state["plan"]["steps"][1]["tool"] == "tool_b"
        assert len(_calls_for(calls, "tool_a", future_id)) == 0
        assert len(_calls_for(calls, "safe_probe", future_id)) == 1
        assert len(_calls_for(calls, "tool_b", future_id)) == 1
        assert len(calls["tool_a"]) == calls_before_future["tool_a"]
        assert store.search_knowledge_claims("local artifact alpha") == []
        final_conversation = client.get(f"/api/v1/tasks/{future_id}/messages")
        assert final_conversation.status_code == 200
        assert final_conversation.json()["messages"][0]["content"] == FUTURE_GOAL
        assert final_conversation.json()["messages"][0]["status"] == "CONSUMED"
    store.close()


def test_memory_retrieval_isolated_between_task_owners(tmp_path: Path) -> None:
    database = tmp_path / "owner-scoped-memory.sqlite3"
    calls: dict[str, list[dict[str, Any]]] = {name: [] for name in ("tool_a", "tool_b", "safe_probe")}
    owner_private_goal = f"{SUPPORT_GOAL} owner-a-private-target-marker"

    store, owner_task_one, _ = _run_support_task(
        database,
        calls,
        idempotency_key="owner-a-support-one",
        initialize_owner=True,
        goal=owner_private_goal,
    )
    store.close()
    store, owner_task_two, _ = _run_support_task(
        database,
        calls,
        idempotency_key="owner-a-support-two",
        initialize_owner=False,
        goal=owner_private_goal,
    )
    owner_a_id = store.task_owner_user_id(owner_task_one)
    assert owner_a_id and store.task_owner_user_id(owner_task_two) == owner_a_id
    strategies = SQLiteLearningMemory(store).list()
    assert len(strategies) == 1
    assert set(strategies[0].supporting_experience_ids) == {owner_task_one, owner_task_two}
    store.close()

    # Issue a genuinely signed, owner-A task-bound claim; the task_owners row,
    # not caller-provided claim metadata, is the isolation authority.
    store = Store(database, knowledge_attestation_key=KNOWLEDGE_KEY)
    now = datetime.now(timezone.utc).isoformat()
    review_id = "review-owner-a-private"
    finding_id = "finding-owner-a-private"
    evidence_sha256 = "a" * 64
    source_id = f"source-review-{review_id}"
    claim_id = f"claim-review-{review_id}"
    evidence_id = f"evidence-review-{review_id}"
    excerpt = "owner-a-private-evidence-marker"
    source_content = f"Owner A private review evidence: {excerpt}; private claim owner-a-private-claim-marker."
    source = KnowledgeSource(
        source_id=source_id,
        canonical_uri=f"local://bug-bounty/reviews/{review_id}",
        source_type="local",
        source_class="local_verified",
        retrieved_at=now,
        content_sha256=sha256_hex(source_content),
        revision=evidence_sha256,
        metadata={
            "task_id": owner_task_one,
            "review_id": review_id,
            "finding_id": finding_id,
            "evidence_sha256": evidence_sha256,
        },
        created_at=now,
        content=source_content,
    )
    claim = KnowledgeClaim(
        claim_id=claim_id,
        domain="cybersecurity.bugbounty",
        subject="local artifact alpha owner-a-private-target-marker",
        predicate="owner_confirmed_after_evidence_gate",
        object_text="Owner A private claim marker owner-a-private-claim-marker",
        knowledge_type="reviewed_security_finding",
        verification_status="verified",
        status="SUPPORTED",
        confidence=0.95,
        observed_at=now,
        last_verified_at=now,
        metadata={
            "review_id": review_id,
            "finding_id": finding_id,
            "evidence_sha256": evidence_sha256,
        },
        created_at=now,
        updated_at=now,
    )
    evidence = KnowledgeEvidence(
        evidence_id=evidence_id,
        claim_id=claim_id,
        source_id=source_id,
        locator="persisted owner review evidence",
        excerpt=excerpt,
        verification_status="verified",
        observed_at=now,
        metadata={"task_id": owner_task_one, "review_id": review_id, "evidence_sha256": evidence_sha256},
        created_at=now,
    )
    attestor = KnowledgeAttestor(
        KNOWLEDGE_KEY,
        issuer_authorizer=lambda _source, _claim, _evidence: owner_a_id,
    )
    attestation = attestor.issue(source, claim, [evidence], verified_at=now)
    store.save_attested_knowledge_bundle(source, claim, [evidence], attestation)
    assert [item["claim_id"] for item in store.search_knowledge_claims(FUTURE_GOAL, task_id=owner_task_one)] == [claim_id]

    auth = AuthManager(store)
    owner = next(user for user in auth.list_users() if user.role == "OWNER")
    operator_b = auth.create_user(
        "operator-b-memory@example.com",
        OPERATOR_PASSWORD,
        "OPERATOR",
        actor_user_id=owner.user_id,
    )
    store.close()

    provider = LocalDeterministicProvider()
    store, service, orchestrator = _build_components(
        database,
        calls,
        provider,
        knowledge_attestation_key=KNOWLEDGE_KEY,
    )
    owner_b_goal = "Please research local artifact alpha for a separate workspace"
    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        csrf = _login(client, operator_b.email, OPERATOR_PASSWORD)
        task_b_id = _create_task(client, csrf, owner_b_goal, "owner-b-isolation-task")
        assert store.task_owner_user_id(task_b_id) == operator_b.user_id

        # Every persisted memory family must fail closed across owners, including
        # a verified experience, its reflection/strategy support, and an attested claim.
        assert SQLiteExperienceMemory(store).retrieve(
            ExperienceQuery(owner_b_goal, task_id=task_b_id)
        ) == []
        assert SQLiteReflectionMemory(store).list_relevant(
            {"goal": owner_b_goal, "task_id": task_b_id}
        ) == []
        assert SQLiteLearningMemory(store).retrieve_relevant(
            {
                "goal": owner_b_goal,
                "task_type": "general",
                "tools": ["tool_a", "tool_b", "safe_probe"],
                "task_id": task_b_id,
            }
        ) == []
        assert store.search_knowledge_claims(owner_b_goal, task_id=task_b_id) == []

        waiting = BackgroundWorker(orchestrator, worker_id="owner-b-isolation-worker").run_once()
        assert waiting["status"] == "WAITING_APPROVAL", waiting
        assert provider.initial_choices == ["tool_a"]
        assert len(provider.contexts) == 2
        all_provider_context = json.dumps(provider.contexts, ensure_ascii=False, sort_keys=True)
        for private_value in (
            "owner-a-private-target-marker",
            "owner-a-private-claim-marker",
            "owner-a-private-evidence-marker",
            owner_task_one,
            owner_task_two,
            source_id,
            claim_id,
            evidence_id,
        ):
            assert private_value not in all_provider_context
        assert provider.contexts[0].get("learned_strategies") == []
        assert provider.contexts[1].get("retrieved_experiences") == []
        assert provider.contexts[1].get("reflection_insights") == []
        assert provider.contexts[1].get("learned_strategies") == []

        state = store.get_task_state(task_b_id)
        assert state is not None
        assert state["context"]["prior_knowledge_used"] == []
        assert state["context"]["retrieved_experiences"] == []
        assert state["context"]["reflection_insights_used"] == []
        assert state["context"]["learned_strategies_used"] == []
        assert owner_a_id != operator_b.user_id
    store.close()
