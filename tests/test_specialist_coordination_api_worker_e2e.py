from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from private_agent.api import AgentAPIService, create_app
from private_agent.core.llm import LLMProvider, LLMResponse
from private_agent.core.observation import Observation
from private_agent.core.orchestrator import Orchestrator, TaskInterrupted
from private_agent.core.planner import Planner
from private_agent.core.specialist_coordination import SpecialistResult
from private_agent.core.specialists import SpecialistProfile, SpecialistRegistry
from private_agent.core.verification import GenericVerifier, VerificationResult, VerifierRegistry
from private_agent.observability import ObservabilityMonitor
from private_agent.security import SecurityController
from private_agent.storage import Store
from private_agent.tools.contracts import ToolResult
from private_agent.tools.research import ToolRegistry
from private_agent.worker import BackgroundWorker


API_TOKEN = "specialist-e2e-local-test-token"
GOAL = "Compare synthetic fixture alpha using two bounded specialists and verify the result"
FIXTURE_SOURCE = "fixture://alpha/revision-1"
AUTHORITATIVE_SOURCE = "fixture://alpha/source-of-truth"
FIXTURE_HASH = "sha256:synthetic-alpha-revision-1"


class DeterministicSpecialistProvider(LLMProvider):
    """Local planning fixture; proves existing wiring, not model intelligence."""

    def __init__(self) -> None:
        self.contexts: list[dict[str, Any]] = []

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

        def envelope(plan: dict[str, Any], intent: str) -> dict[str, Any]:
            return {
                "goal_spec": {
                    "normalized_goal": goal,
                    "intent": intent,
                    "deliverables": ["verified synthetic fixture conclusion"],
                    "constraints": ["registered local synthetic tools only"],
                    "success_criteria": ["independent fixture verification passes"],
                    "missing_information": [],
                    "clarification_required": False,
                    "clarification_question": "",
                    "priority": 3,
                    "risk_level": "LOW",
                },
                "plan": plan,
            }

        if "current_plan" in context:
            steps = [
                {
                    "id": "source-index",
                    "objective": "Load the bounded synthetic fixture source reference",
                    "tool": "local_fixture_index",
                    "input": {"fixture": "alpha"},
                    "verifier": "generic",
                    "reason": "Establish the source revision before specialist inspection",
                    "depends_on": [],
                    "success_criteria": {"required_fields": ["source_id", "source_hash"]},
                },
                {
                    "id": "resolve-alpha",
                    "objective": "Resolve the conflicting synthetic claims against the local authoritative fixture",
                    "tool": "investigate_repository_bug",
                    "input": {},
                    "input_bindings": {
                        "source_id": {"source_step_id": "source-index", "output_key": "source_id"}
                    },
                    "verifier": "synthetic_evidence_verifier",
                    "reason": "A registered verifier-backed local resolver is required after disagreement",
                    "depends_on": ["source-index"],
                    "specialist_id": "local_repository_analyst",
                    "success_criteria": {"required_fields": ["claims"]},
                },
            ]
            result = envelope(
                {
                    "goal": goal,
                    "rationale": "Resolve the detected specialist conflict with a registered local source of truth",
                    "steps": steps,
                },
                "resolve synthetic evidence conflict",
            )
            return LLMResponse(result, json.dumps(result, sort_keys=True), "local-deterministic")

        steps = [
            {
                "id": "source-index",
                "objective": "Load the bounded synthetic fixture source reference",
                "tool": "local_fixture_index",
                "input": {"fixture": "alpha"},
                "verifier": "generic",
                "reason": "Establish the source revision before specialist inspection",
                "depends_on": [],
                "success_criteria": {"required_fields": ["source_id", "source_hash"]},
            },
            {
                "id": "analyst-inspection",
                "objective": "Assess whether synthetic fixture alpha is present",
                "tool": "read_repository_file",
                "input": {},
                "input_bindings": {
                    "source_id": {"source_step_id": "source-index", "output_key": "source_id"}
                },
                "verifier": "synthetic_evidence_verifier",
                "reason": "Use the registered local repository analyst on the verified source revision",
                "depends_on": ["source-index"],
                "specialist_id": "local_repository_analyst",
                "success_criteria": {"required_fields": ["claims"]},
            },
            {
                "id": "reader-cross-check",
                "objective": "Independently cross-check whether synthetic fixture alpha is present",
                "tool": "search_repository_code",
                "input": {},
                "input_bindings": {
                    "source_id": {"source_step_id": "source-index", "output_key": "source_id"}
                },
                "verifier": "synthetic_evidence_verifier",
                "reason": "Use a distinct registered reader with only its bounded source context",
                "depends_on": ["source-index"],
                "specialist_id": "repository_reader",
                "success_criteria": {"required_fields": ["claims"]},
            },
        ]
        result = envelope(
            {
                "goal": goal,
                "rationale": "Compare two independent, bounded specialist reports against one verified source revision",
                "steps": steps,
            },
            "compare synthetic fixture evidence",
        )
        return LLMResponse(result, json.dumps(result, sort_keys=True), "local-deterministic")


class SyntheticTool:
    permission_level = "public_read"
    planned_effect = "benign in-memory synthetic fixture operation"
    required_capabilities = ["READ_WORKSPACE"]
    declared_capabilities = ["READ_WORKSPACE"]
    risk_level = "low"
    network_required = False
    filesystem_required = False

    def __init__(self, name: str, calls: list[dict[str, Any]], behavior: str) -> None:
        self.name = name
        self.calls = calls
        self.behavior = behavior

    def description(self) -> str:
        return f"Test-only in-memory synthetic fixture tool: {self.name}"

    def input_schema(self) -> dict[str, Any]:
        if self.name == "local_fixture_index":
            properties = {"fixture": {"type": "string"}}
            required = ["fixture"]
        else:
            properties = {"source_id": {"type": "string"}}
            required = ["source_id"]
        return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}

    def output_schema(self) -> dict[str, Any]:
        if self.name == "local_fixture_index":
            return {
                "type": "object",
                "properties": {"source_id": {"type": "string"}, "source_hash": {"type": "string"}},
                "required": ["source_id", "source_hash"],
                "additionalProperties": False,
            }
        claim_schema = {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "value": {"type": "boolean"},
                "source": {"type": "string"},
            },
            "required": ["subject", "predicate", "value", "source"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {"claims": {"type": "array", "items": claim_schema}},
            "required": ["claims"],
            "additionalProperties": False,
        }

    def execute(self, inputs: dict[str, Any], context: Any) -> ToolResult:
        if self.name == "local_fixture_index":
            self.calls.append({"tool": self.name, "inputs": dict(inputs), "task_id": context.task_id})
            return ToolResult.success({"source_id": FIXTURE_SOURCE, "source_hash": FIXTURE_HASH})

        work_order = getattr(context, "work_order", None)
        assert isinstance(work_order, dict), "specialist calls must receive a server-created work order"
        self.calls.append({
            "tool": self.name,
            "inputs": dict(inputs),
            "task_id": context.task_id,
            "work_order": work_order,
        })
        if self.behavior == "analyst":
            value, source = True, FIXTURE_SOURCE
        elif self.behavior == "reader":
            value, source = False, FIXTURE_SOURCE
        else:
            value, source = True, AUTHORITATIVE_SOURCE
        return ToolResult.success({
            "claims": [{
                "subject": "fixture:alpha",
                "predicate": "exists",
                "value": value,
                "source": source,
            }]
        })


class SyntheticEvidenceVerifier:
    """Independent test oracle: checks cited local evidence; only resolver citation is authoritative."""

    name = "synthetic_evidence_verifier"

    def verify(self, observation: Observation, criteria: dict[str, Any]) -> VerificationResult:
        output = observation.observed_output
        required_fields = criteria.get("required_fields", [])
        if observation.execution_status != "success" or not isinstance(output, dict):
            return VerificationResult("FAILED", "Synthetic tool did not return a successful structured output")
        if any(field not in output for field in required_fields):
            return VerificationResult("FAILED", "Synthetic output is missing required fields")
        claims = output.get("claims")
        if not isinstance(claims, list) or len(claims) != 1 or not isinstance(claims[0], dict):
            return VerificationResult("FAILED", "Exactly one typed synthetic claim is required")
        claim = claims[0]
        if claim.get("subject") != "fixture:alpha" or claim.get("predicate") != "exists":
            return VerificationResult("FAILED", "Claim does not match the synthetic fixture contract")
        if claim.get("source") == FIXTURE_SOURCE and isinstance(claim.get("value"), bool):
            return VerificationResult(
                "VERIFIED",
                "The independent local verifier confirmed typed claim structure and a known source citation",
                evidence=[{
                    "type": "synthetic_source_citation",
                    "source": FIXTURE_SOURCE,
                    "task_id": observation.task_id,
                    "verifier": self.name,
                }],
                details={"citation_verified": True, "claim_truth_adjudicated": False},
            )
        if (
            observation.tool_name == "investigate_repository_bug"
            and claim.get("source") == AUTHORITATIVE_SOURCE
            and claim.get("value") is True
        ):
            return VerificationResult(
                "VERIFIED",
                "The independent local fixture oracle confirmed the authoritative synthetic result",
                evidence=[{
                    "type": "authoritative_synthetic_fixture",
                    "source": AUTHORITATIVE_SOURCE,
                    "task_id": observation.task_id,
                    "verifier": self.name,
                }],
                details={"citation_verified": True, "claim_truth_adjudicated": True},
            )
        return VerificationResult("FAILED", "Citation or authoritative fixture value did not match the local oracle")


def _components(
    database: Path,
    provider: DeterministicSpecialistProvider,
    calls: list[dict[str, Any]],
    *,
    checkpoint_hook=None,
    analyst_profile_version: int = 1,
):
    store = Store(database)
    monitor = ObservabilityMonitor(store)
    security = SecurityController(store=store, observability=monitor)
    tools = ToolRegistry()
    tools.register(SyntheticTool("local_fixture_index", calls, "index"))
    tools.register(SyntheticTool("read_repository_file", calls, "analyst"))
    tools.register(SyntheticTool("search_repository_code", calls, "reader"))
    tools.register(SyntheticTool("investigate_repository_bug", calls, "resolver"))
    specialists = SpecialistRegistry()
    specialists.register(SpecialistProfile(
        specialist_id="local_repository_analyst",
        version=analyst_profile_version,
        tool_names=frozenset({"read_repository_file", "investigate_repository_bug"}),
        granted_capabilities=frozenset({"READ_WORKSPACE"}),
        max_context_bytes=2048,
    ))
    specialists.register(SpecialistProfile(
        specialist_id="repository_reader",
        version=1,
        tool_names=frozenset({"search_repository_code"}),
        granted_capabilities=frozenset({"READ_WORKSPACE"}),
        max_context_bytes=2048,
    ))
    planner = Planner(provider, specialists=specialists)
    verifiers = VerifierRegistry([GenericVerifier(), SyntheticEvidenceVerifier()])
    orchestrator = Orchestrator(
        store,
        tools,
        planner=planner,
        verifiers=verifiers,
        security_controller=security,
        specialist_registry=specialists,
        observability=monitor,
        checkpoint_hook=checkpoint_hook,
        max_recovery_attempts=2,
        max_attempts_per_step=1,
    )
    return store, AgentAPIService(orchestrator, token=API_TOKEN), orchestrator


def test_http_worker_specialist_conflict_replans_and_survives_sqlite_reopen(tmp_path: Path) -> None:
    database = tmp_path / "specialist-coordination.sqlite3"
    calls: list[dict[str, Any]] = []
    provider = DeterministicSpecialistProvider()
    store, service, orchestrator = _components(database, provider, calls)
    client = TestClient(create_app(service=service), raise_server_exceptions=False)

    created = client.post(
        "/api/v1/tasks",
        json={"goal": GOAL},
        headers={"Authorization": f"Bearer {API_TOKEN}", "Idempotency-Key": "specialist-e2e-1"},
    )
    assert created.status_code == 201, created.text
    task_id = created.json()["task_id"]
    assert created.json()["conversation_id"] == task_id

    initial = client.get(f"/api/v1/tasks/{task_id}", headers={"Authorization": f"Bearer {API_TOKEN}"})
    assert initial.status_code == 200
    assert initial.json()["status"] == "QUEUED"

    worker = BackgroundWorker(orchestrator, worker_id="specialist-e2e-worker")
    worker_results = []
    for _ in range(10):
        worker_result = worker.run_once()
        worker_results.append(worker_result)
        if worker_result.get("status") == "COMPLETED":
            break
    failed_state = store.get_task_state(task_id)
    assert worker_result.get("status") == "COMPLETED", {
        "worker_results": worker_results,
        "queue_status": (store.task_queue_record(task_id) or {}).get("queue_status"),
        "saved_result": (failed_state or {}).get("result"),
        "failure_reason": (failed_state or {}).get("failure_reason"),
    }
    assert worker_result.get("task_id") == task_id

    detail = client.get(f"/api/v1/tasks/{task_id}", headers={"Authorization": f"Bearer {API_TOKEN}"})
    assert detail.status_code == 200, detail.text
    assert detail.json()["status"] == "COMPLETED"
    assert detail.json()["verification_status"] == "VERIFIED"
    state = store.get_task_state(task_id)
    assert state is not None and state["status"] == "COMPLETED"
    assert state["plan_version"] == 2
    assert state["recovery_state"]["decision"]["strategy"] == "REPLAN"
    assert state["recovery_state"]["diagnosis"]["details"]["code"] == "specialist_result_conflict"
    assert state["plan"]["steps"][-1]["id"] == "resolve-alpha"
    assert state["result"]["verification"]["status"] == "VERIFIED"
    assert state["result"]["verification"]["verified"] is True
    assert state["result"]["specialist_synthesis"]["status"] == "VERIFIED"

    # Two different profiles receive separate minimal payloads; neither sees the other role's inputs or the full task.
    role_calls = [call for call in calls if call["tool"] in {"read_repository_file", "search_repository_code", "investigate_repository_bug"}]
    assert [call["work_order"]["specialist_id"] for call in role_calls] == [
        "local_repository_analyst", "repository_reader", "local_repository_analyst"
    ]
    context_digests = set()
    source_value_digest = sha256(json.dumps(
        FIXTURE_SOURCE, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    for call in role_calls:
        work_order = call["work_order"]
        assert work_order["parent_task_id"] == task_id
        assert work_order["execution_mode"] == "same_task_internal_work_order"
        assert set(work_order) == {
            "work_order_id", "parent_task_id", "plan_version", "source_step_id", "specialist_id",
            "profile_version", "tool_name", "objective", "inputs", "dependency_provenance",
            "context_digest", "context_limit_bytes", "execution_mode",
        }
        assert work_order["context_limit_bytes"] == 2048
        assert len(work_order["context_digest"]) == 64
        context_digests.add(work_order["context_digest"])
        assert work_order["inputs"]["source_id"] == FIXTURE_SOURCE
        assert "conversation_history" not in work_order and "full_plan" not in work_order
        assert work_order["dependency_provenance"] == [{
            "target": "source_id",
            "source_step_id": "source-index",
            "output_key": "source_id",
            "value_digest": source_value_digest,
        }]
    assert len(context_digests) == 3

    # The source step and the two initial specialists run once; recovery reuses the verified source record.
    assert len([call for call in calls if call["tool"] == "local_fixture_index"]) == 1
    assert len([call for call in calls if call["tool"] == "read_repository_file"]) == 1
    assert len([call for call in calls if call["tool"] == "search_repository_code"]) == 1
    assert len([call for call in calls if call["tool"] == "investigate_repository_bug"]) == 1

    work_orders = state["context"]["specialist_work_orders"]
    results = state["context"]["specialist_results"]
    assert len(work_orders) == 3 and len(results) == 3
    typed_results = [SpecialistResult.from_dict(item) for item in results.values()]
    for result in typed_results:
        result.validate()
    assert {item.parent_task_id for item in typed_results} == {task_id}
    assert {item.verification_status for item in typed_results} == {"VERIFIED"}
    assert all(item.verification_evidence for item in typed_results)
    assert {item.plan_version for item in typed_results} == {1, 2}
    assert {item.specialist_id for item in typed_results} == {"local_repository_analyst", "repository_reader"}

    # The initial synthesis records two independently verified, conflicting claims and their citations.
    recovery_context = next(context for context in provider.contexts if "current_plan" in context)
    assert recovery_context["failure_diagnosis"]["details"]["code"] == "specialist_result_conflict"
    conflict = recovery_context["failure_diagnosis"]["details"]["conflicts"][0]
    assert conflict["subject"] == "fixture:alpha" and conflict["predicate"] == "exists"
    assert {source["specialist_id"] for source in conflict["sources"]} == {
        "local_repository_analyst", "repository_reader"
    }
    assert all(source["verification_evidence"] for source in conflict["sources"])

    # Close and reopen SQLite, then verify the API and terminal resume path do not replay any tool effect.
    store.close()
    reopened_provider = DeterministicSpecialistProvider()
    reopened_store, reopened_service, reopened_orchestrator = _components(database, reopened_provider, calls)
    reopened_state = reopened_store.get_task_state(task_id)
    assert reopened_state is not None and reopened_state["status"] == "COMPLETED"
    with TestClient(create_app(service=reopened_service), raise_server_exceptions=False) as reopened_detail:
        response = reopened_detail.get(
            f"/api/v1/tasks/{task_id}", headers={"Authorization": f"Bearer {API_TOKEN}"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "COMPLETED"
        assert response.json()["verification_status"] == "VERIFIED"
    resume_result = reopened_orchestrator.resume_task(task_id)
    assert resume_result["final_status"] == "COMPLETED"
    assert len(calls) == 4
    reopened_store.close()


def test_resume_blocks_verified_specialist_result_after_server_profile_drift(tmp_path: Path) -> None:
    database = tmp_path / "specialist-policy-drift.sqlite3"
    calls: list[dict[str, Any]] = []

    def interrupt_after_verified_specialist(checkpoint: str, state: dict[str, Any]) -> None:
        if checkpoint == "AFTER_VERIFICATION" and state.get("context", {}).get("specialist_results"):
            raise TaskInterrupted("test interruption after specialist result persistence")

    provider = DeterministicSpecialistProvider()
    store, _, orchestrator = _components(
        database, provider, calls, checkpoint_hook=interrupt_after_verified_specialist
    )
    with pytest.raises(TaskInterrupted, match="after specialist result persistence"):
        orchestrator.run(GOAL)
    task_id = store.db.execute("SELECT task_id FROM task_states").fetchone()[0]
    interrupted_state = store.get_task_state(task_id)
    assert interrupted_state is not None
    assert interrupted_state["context"]["specialist_results"]
    assert len(calls) == 2  # one synthetic source lookup and one verified specialist operation
    store.close()

    # A new server registry version invalidates the persisted assignment; resume must fail closed without re-execution.
    new_provider = DeterministicSpecialistProvider()
    reopened_store, _, reopened_orchestrator = _components(
        database, new_provider, calls, analyst_profile_version=2
    )
    blocked = reopened_orchestrator.resume_task(task_id)
    assert blocked["final_status"] == "BLOCKED"
    assert "persisted_specialist_policy_or_input_provenance_drift" in blocked["errors"]
    persisted = reopened_store.get_task_state(task_id)
    assert persisted is not None and persisted["status"] == "BLOCKED"
    assert len(calls) == 2
    reopened_store.close()
