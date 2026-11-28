from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from private_agent.core.intelligence import GeneralIntelligenceEngine, GoalSpec, IntelligenceContractError
from private_agent.core.llm import LLMProvider, LLMResponse
from private_agent.core.planner import PLAN_RESPONSE_SCHEMA, PlanValidationError, Planner
from private_agent.core.specialists import SpecialistProfile, SpecialistRegistry
from private_agent.tools.research import ToolRegistry, WebResearchTool


class StructuredFixtureProvider(LLMProvider):
    """Deterministic contract fixture; tests integration, not reasoning quality."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, *, system_prompt: str, user_prompt: str, response_schema: dict[str, Any]) -> LLMResponse:
        self.calls.append({
            "system_prompt": system_prompt,
            "context": json.loads(user_prompt),
            "response_schema": response_schema,
        })
        return LLMResponse(self.response, json.dumps(self.response, ensure_ascii=False), "fixture")


def _goal_spec(
    goal: str,
    *,
    missing: list[str] | None = None,
    clarification_required: bool = False,
    clarification_question: str = "",
) -> dict[str, Any]:
    return {
        "normalized_goal": goal,
        "intent": "bounded local analysis",
        "deliverables": ["verified result"],
        "constraints": ["registered tools only"],
        "success_criteria": ["server-side verification passes"],
        "missing_information": missing or [],
        "clarification_required": clarification_required,
        "clarification_question": clarification_question,
        "priority": 3,
        "risk_level": "LOW",
    }


def _plan(goal: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "goal": goal,
        "rationale": "A concise provider proposal to be checked by the server.",
        "steps": steps,
    }


def _research_step(step_id: str, query: str, *, priority: int = 50, depends_on: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": step_id,
        "objective": f"Research {query}",
        "tool": "web_research",
        "input": {"query": query, "limit": 3},
        "reason": "The registered read-only research tool is available.",
        "depends_on": depends_on or [],
        "priority": priority,
    }


def _research_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WebResearchTool())
    return registry


def test_goal_understanding_uses_context_and_returns_bounded_missing_information() -> None:
    goal = "Compare two fictional offline parsers and report the safer one"
    response = {
        "goal_spec": _goal_spec(
            goal,
            missing=["The fixture does not state a maximum input size"],
            clarification_required=True,
            clarification_question="What maximum input size should the comparison assume?",
        ),
        "plan": _plan(goal, [_research_step("compare", "fictional offline parsers")]),
    }
    provider = StructuredFixtureProvider(response)
    planner = Planner(provider)
    plan = planner.create(
        goal,
        _research_registry(),
        [{"concept": "synthetic prior context"}],
        constraints={"max_steps": 4, "network": "not needed"},
        permissions={"web_research": "public_read"},
        retrieved_experiences=[{"pattern": "synthetic verified comparison"}],
        reflection_insights=[{"lesson": "keep evidence separate from conclusions"}],
    )

    assert plan.goal_spec is not None
    assert plan.goal_spec.missing_information == ("The fixture does not state a maximum input size",)
    assert plan.goal_spec.clarification_required is True
    assert plan.goal_spec.clarification_question.startswith("What maximum")
    assert plan.goal_spec.analysis_mode == "PROVIDER"
    context = provider.calls[0]["context"]
    assert context["goal"] == goal
    assert context["relevant_memory"] == [{"concept": "synthetic prior context"}]
    assert context["relevant_experiences"] == [{"pattern": "synthetic verified comparison"}]
    assert context["reflection_insights"] == [{"lesson": "keep evidence separate from conclusions"}]
    assert context["permissions"] == {"web_research": "public_read"}
    assert provider.calls[0]["response_schema"] == GeneralIntelligenceEngine.response_schema(
        PLAN_RESPONSE_SCHEMA,
        expected_goal=goal,
        tool_input_schemas={
            item["name"]: item["input_schema"]
            for item in context["available_tools"]
        },
    )


def test_planner_orders_ready_steps_by_priority_without_violating_dependencies() -> None:
    goal = "Create a synthetic evidence summary"
    response = {
        "goal_spec": _goal_spec(goal),
        "plan": _plan(goal, [
            _research_step("late-independent", "late item", priority=90),
            _research_step("dependent", "dependent item", priority=1, depends_on=["root"]),
            _research_step("urgent-independent", "urgent item", priority=1),
            _research_step("root", "root item", priority=70),
        ]),
    }
    plan = Planner(StructuredFixtureProvider(response)).create(
        goal,
        _research_registry(),
        [],
        permissions={"web_research": "public_read"},
    )

    assert [step.id for step in plan.steps] == [
        "urgent-independent",
        "root",
        "dependent",
        "late-independent",
    ]
    assert plan.steps[2].depends_on == ["root"]


def test_registered_specialist_is_selected_and_bound_by_server_registry() -> None:
    goal = "Read one local synthetic repository file"
    tools = ToolRegistry()
    tools.register(_SyntheticRepositoryTool())
    specialists = SpecialistRegistry()
    specialists.register(SpecialistProfile(
        specialist_id="local_repository_analyst",
        version=1,
        tool_names=frozenset({"read_repository_file"}),
        granted_capabilities=frozenset({"READ_WORKSPACE"}),
    ))
    response = {
        "goal_spec": _goal_spec(goal),
        "plan": {
            "goal": goal,
            "rationale": "Use the listed local specialist for bounded read-only evidence.",
            "steps": [{
                "id": "read-source",
                "objective": "Read the synthetic source file",
                "tool": "read_repository_file",
                "input": {"path": "src/example.py"},
                "reason": "The server lists this specialist for the registered local reader.",
                "depends_on": [],
                "specialist_id": "local_repository_analyst",
            }],
        },
    }
    provider = StructuredFixtureProvider(response)
    plan = Planner(provider, specialists=specialists).create(
        goal,
        tools,
        [],
        permissions={"read_repository_file": "public_read"},
    )

    available = provider.calls[0]["context"]["available_specialists"]
    assert [item["specialist_id"] for item in available] == ["local_repository_analyst"]
    response_step_schema = provider.calls[0]["response_schema"]["properties"]["plan"]["properties"]["steps"]["items"]
    assert response_step_schema["properties"]["specialist_id"]["enum"] == ["local_repository_analyst"]
    assert "specialist_id" not in response_step_schema["required"]
    assignment = plan.steps[0].specialist_assignment
    assert assignment["specialist_id"] == "local_repository_analyst"
    assert assignment["effective_capabilities"] == ["READ_WORKSPACE"]


def test_unregistered_tool_and_missing_permission_remain_rejected() -> None:
    goal = "Perform a synthetic read-only lookup"
    invalid_tool = {
        "goal_spec": _goal_spec(goal),
        "plan": _plan(goal, [{
            **_research_step("bad-tool", "synthetic topic"),
            "tool": "shell",
        }]),
    }
    with pytest.raises(PlanValidationError, match="unknown_tool:shell"):
        Planner(StructuredFixtureProvider(invalid_tool)).create(goal, _research_registry(), [])

    valid_tool_unapproved = {
        "goal_spec": _goal_spec(goal),
        "plan": _plan(goal, [_research_step("not-approved", "synthetic topic")]),
    }
    with pytest.raises(PlanValidationError, match="permission_not_granted:web_research"):
        Planner(StructuredFixtureProvider(valid_tool_unapproved)).create(
            goal,
            _research_registry(),
            [],
            permissions={},
        )


def test_goal_spec_rejects_forged_fields_and_round_trips_versioned_state() -> None:
    goal = "A bounded synthetic goal"
    invalid_spec = _goal_spec(goal)
    invalid_spec["permission_level"] = "private_write"
    response = {
        "goal_spec": invalid_spec,
        "plan": _plan(goal, [_research_step("read", "synthetic topic")]),
    }
    with pytest.raises(PlanValidationError, match="goal_spec_fields_invalid"):
        Planner(StructuredFixtureProvider(response)).create(
            goal,
            _research_registry(),
            [],
            permissions={"web_research": "public_read"},
        )

    spec = GoalSpec.from_model_payload(_goal_spec(goal))
    restored = GoalSpec.from_serialized(spec.to_dict())
    assert restored == spec
    tampered = spec.to_dict()
    tampered["goal_spec_version"] = 999
    with pytest.raises(ValueError, match="persisted_goal_spec_version_invalid"):
        GoalSpec.from_serialized(tampered)


class _SyntheticRepositoryTool:
    name = "read_repository_file"
    permission_level = "public_read"
    required_capabilities = ["READ_WORKSPACE"]
    declared_capabilities = ["READ_WORKSPACE"]
    risk_level = "low"
    network_required = False
    filesystem_required = True

    def description(self) -> str:
        return "Read one synthetic local file"

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

    def execute(self, inputs: dict[str, Any], context: Any) -> Any:
        raise AssertionError("planner test must not execute tools")


class _SyntheticCalculationTool:
    permission_level = "public_read"
    required_capabilities: list[str] = []
    declared_capabilities: list[str] = []
    risk_level = "low"
    network_required = False
    filesystem_required = False

    def __init__(self, name: str, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        self.name = name
        self._inputs = inputs
        self._outputs = outputs

    def description(self) -> str:
        return f"Fictional garden calculation tool {self.name}"

    def input_schema(self) -> dict[str, Any]:
        return self._inputs

    def output_schema(self) -> dict[str, Any]:
        return self._outputs

    def execute(self, inputs: dict[str, Any], context: Any) -> Any:
        raise AssertionError("planning contract tests must never execute a tool")


def _calculation_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_SyntheticCalculationTool(
        "synthetic_bed_total",
        {
            "type": "object",
            "properties": {
                "bed_name": {"type": "string"},
                "days": {"type": "integer", "minimum": 1, "maximum": 14},
                "liters_per_day": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["bed_name", "days", "liters_per_day"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "bed_name": {"type": "string"},
                "total_liters": {"type": "integer"},
            },
            "required": ["bed_name", "total_liters"],
            "additionalProperties": False,
        },
    ))
    registry.register(_SyntheticCalculationTool(
        "synthetic_add_totals",
        {
            "type": "object",
            "properties": {"first_total": {"type": "integer"}, "second_total": {"type": "integer"}},
            "required": ["first_total", "second_total"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"combined_total": {"type": "integer"}},
            "required": ["combined_total"],
            "additionalProperties": False,
        },
    ))
    return registry


def _synthetic_calculation_plan(goal: str) -> dict[str, Any]:
    return _plan(goal, [
        {
            "id": "shams",
            "objective": "Calculate a fictional bed total",
            "tool": "synthetic_bed_total",
            "input": {"bed_name": "شمس", "days": 3, "liters_per_day": 2},
            "reason": "Known values are supplied directly from the fictional goal.",
            "depends_on": [],
        },
        {
            "id": "qamar",
            "objective": "Calculate another fictional bed total",
            "tool": "synthetic_bed_total",
            "input": {"bed_name": "قمر", "days": 3, "liters_per_day": 1},
            "reason": "Known values are supplied directly from the fictional goal.",
            "depends_on": [],
        },
        {
            "id": "sum",
            "objective": "Add the two fictional totals",
            "tool": "synthetic_add_totals",
            "input": {},
            "reason": "Totals are produced by declared prerequisite steps.",
            "depends_on": ["shams", "qamar"],
            "input_bindings": {
                "first_total": {"source_step_id": "shams", "output_key": "total_liters"},
                "second_total": {"source_step_id": "qamar", "output_key": "total_liters"},
            },
        },
    ])


def _provider_tool_input_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Encode canonical fixture inputs in the provider-neutral per-tool contract."""
    result = deepcopy(plan)
    tool_names = {step["tool"] for step in result["steps"]}
    tool_inputs: dict[str, list[dict[str, Any]]] = {name: [] for name in tool_names}
    for step in result["steps"]:
        direct = step.pop("input", {})
        bindings = step.pop("input_bindings", {})
        step.pop("permission_level", None)
        fields: dict[str, Any] = {}
        for field_name in dict.fromkeys([*direct, *bindings]):
            if field_name in direct and field_name in bindings:
                fields[field_name] = {
                    "source": "value",
                    "value": direct[field_name],
                    "binding": bindings[field_name],
                }
            elif field_name in direct:
                fields[field_name] = {"source": "value", "value": direct[field_name], "binding": None}
            else:
                fields[field_name] = {"source": "binding", "value": None, "binding": bindings[field_name]}
        tool_inputs[step["tool"]].append({"step_id": step["id"], "fields": fields})
    result["tool_inputs"] = tool_inputs
    return result


def test_provider_schema_specializes_tool_inputs_and_dependency_binding_alternatives() -> None:
    goal = "Calculate fictional garden totals using the registered synthetic tools"
    response = {
        "goal_spec": _goal_spec(goal),
        "plan": _provider_tool_input_plan(_synthetic_calculation_plan(goal)),
    }
    provider = StructuredFixtureProvider(response)

    accepted = Planner(provider).create(
        goal,
        _calculation_registry(),
        [],
        permissions={"synthetic_bed_total": "public_read", "synthetic_add_totals": "public_read"},
    )

    plan_schema = provider.calls[0]["response_schema"]["properties"]["plan"]
    step_schema = plan_schema["properties"]["steps"]["items"]
    tool_inputs_schema = plan_schema["properties"]["tool_inputs"]
    assert step_schema["properties"]["tool"]["enum"] == ["synthetic_add_totals", "synthetic_bed_total"]
    assert "input" not in step_schema["properties"]
    assert "input_bindings" not in step_schema["properties"]
    assert tool_inputs_schema["required"] == ["synthetic_add_totals", "synthetic_bed_total"]
    assert tool_inputs_schema["additionalProperties"] is False

    bed_fields = tool_inputs_schema["properties"]["synthetic_bed_total"]["items"]["properties"]["fields"]
    assert bed_fields["required"] == ["bed_name", "days", "liters_per_day"]
    assert bed_fields["additionalProperties"] is False
    assert bed_fields["properties"]["days"]["properties"]["source"]["enum"] == ["value", "binding"]
    assert bed_fields["properties"]["days"]["properties"]["value"]["minimum"] == 1
    assert bed_fields["properties"]["days"]["properties"]["value"]["type"] == ["integer", "null"]

    add_fields = tool_inputs_schema["properties"]["synthetic_add_totals"]["items"]["properties"]["fields"]
    assert add_fields["required"] == ["first_total", "second_total"]
    assert accepted.steps[0].input == {"bed_name": "شمس", "days": 3, "liters_per_day": 2}
    assert accepted.steps[-1].input == {}
    assert accepted.steps[-1].input_bindings["first_total"] == {
        "source_step_id": "shams",
        "output_key": "total_liters",
    }


def test_matching_goal_with_malformed_provider_inputs_is_rejected_without_relaxing_permissions() -> None:
    goal = "In a fictional garden, calculate two bed totals and combine them using synthetic tools"
    plan = _provider_tool_input_plan(_synthetic_calculation_plan(goal))
    for step_id in ("shams", "qamar"):
        payload = next(
            item for item in plan["tool_inputs"]["synthetic_bed_total"] if item["step_id"] == step_id
        )
        for field in ("bed_name", "days", "liters_per_day"):
            payload["fields"][field] = {
                "source": "binding",
                "value": None,
                "binding": {"source_step_id": "", "output_key": ""},
            }
    response = {"goal_spec": _goal_spec(goal), "plan": plan}
    provider = StructuredFixtureProvider(response)

    with pytest.raises(PlanValidationError, match="provider_tool_input_binding_invalid:shams.bed_name"):
        Planner(provider).create(
            goal,
            _calculation_registry(),
            [],
            permissions={"synthetic_bed_total": "public_read", "synthetic_add_totals": "public_read"},
        )

    assert provider.calls[0]["context"]["goal"] == goal


def test_provider_contract_rejects_missing_and_empty_required_values() -> None:
    goal = "Calculate fictional garden bed totals and add them using local synthetic tools"
    valid = _provider_tool_input_plan(_synthetic_calculation_plan(goal))

    missing = deepcopy(valid)
    shams = next(item for item in missing["tool_inputs"]["synthetic_bed_total"] if item["step_id"] == "shams")
    del shams["fields"]["days"]
    with pytest.raises(PlanValidationError, match="provider_tool_input_required_field_missing:shams.days"):
        Planner(StructuredFixtureProvider({"goal_spec": _goal_spec(goal), "plan": missing})).create(
            goal,
            _calculation_registry(),
            [],
            permissions={"synthetic_bed_total": "public_read", "synthetic_add_totals": "public_read"},
        )

    empty = deepcopy(valid)
    shams = next(item for item in empty["tool_inputs"]["synthetic_bed_total"] if item["step_id"] == "shams")
    shams["fields"]["bed_name"] = {"source": "value", "value": "  ", "binding": None}
    with pytest.raises(PlanValidationError, match="provider_tool_input_required_value_empty:shams.bed_name"):
        Planner(StructuredFixtureProvider({"goal_spec": _goal_spec(goal), "plan": empty})).create(
            goal,
            _calculation_registry(),
            [],
            permissions={"synthetic_bed_total": "public_read", "synthetic_add_totals": "public_read"},
        )


def test_provider_contract_rejects_tool_payload_mismatch_and_conflicting_sources() -> None:
    goal = "Calculate fictional garden bed totals and add them using local synthetic tools"
    valid = _provider_tool_input_plan(_synthetic_calculation_plan(goal))
    mismatched = deepcopy(valid)
    bed_payload = mismatched["tool_inputs"]["synthetic_bed_total"].pop(0)
    mismatched["tool_inputs"]["synthetic_add_totals"].append(bed_payload)
    with pytest.raises(PlanValidationError, match="provider_tool_input_tool_mismatch:shams"):
        Planner(StructuredFixtureProvider({"goal_spec": _goal_spec(goal), "plan": mismatched})).create(
            goal,
            _calculation_registry(),
            [],
            permissions={"synthetic_bed_total": "public_read", "synthetic_add_totals": "public_read"},
        )

    conflicting = deepcopy(valid)
    shams = next(item for item in conflicting["tool_inputs"]["synthetic_bed_total"] if item["step_id"] == "shams")
    shams["fields"]["days"]["binding"] = {"source_step_id": "other", "output_key": "days"}
    with pytest.raises(PlanValidationError, match="provider_tool_input_source_invalid:shams.days"):
        Planner(StructuredFixtureProvider({"goal_spec": _goal_spec(goal), "plan": conflicting})).create(
            goal,
            _calculation_registry(),
            [],
            permissions={"synthetic_bed_total": "public_read", "synthetic_add_totals": "public_read"},
        )


def test_provider_contract_requires_a_complete_tool_map_and_never_authors_permissions() -> None:
    goal = "Calculate fictional garden bed totals and add them using local synthetic tools"
    response_plan = _provider_tool_input_plan(_synthetic_calculation_plan(goal))
    missing_tool = deepcopy(response_plan)
    del missing_tool["tool_inputs"]["synthetic_add_totals"]
    with pytest.raises(PlanValidationError, match="provider_tool_inputs_invalid"):
        Planner(StructuredFixtureProvider({"goal_spec": _goal_spec(goal), "plan": missing_tool})).create(
            goal,
            _calculation_registry(),
            [],
            permissions={"synthetic_bed_total": "public_read", "synthetic_add_totals": "public_read"},
        )

    with pytest.raises(PlanValidationError) as captured:
        Planner(StructuredFixtureProvider({"goal_spec": _goal_spec(goal), "plan": response_plan})).create(
            goal,
            _calculation_registry(),
            [],
            permissions={},
        )
    assert "permission_not_granted:synthetic_bed_total" in captured.value.errors
    assert "permission_not_granted:synthetic_add_totals" in captured.value.errors


def test_unavailable_specialist_is_omitted_from_provider_schema_and_rejected_by_server() -> None:
    goal = "Calculate fictional garden bed totals and add them using local synthetic tools"
    response_plan = _provider_tool_input_plan(_synthetic_calculation_plan(goal))
    response_plan["steps"][0]["specialist_id"] = "none"
    provider = StructuredFixtureProvider({"goal_spec": _goal_spec(goal), "plan": response_plan})

    with pytest.raises(PlanValidationError, match="specialist_registry_unavailable:shams"):
        Planner(provider).create(
            goal,
            _calculation_registry(),
            [],
            permissions={"synthetic_bed_total": "public_read", "synthetic_add_totals": "public_read"},
        )

    step_schema = provider.calls[0]["response_schema"]["properties"]["plan"]["properties"]["steps"]["items"]
    assert "specialist_id" not in step_schema["properties"]


def test_provider_schema_field_budget_fails_closed_for_oversized_tool_input() -> None:
    input_schema = {
        "type": "object",
        "properties": {f"field_{index:04d}": {"type": "string"} for index in range(1025)},
        "required": ["value"],
        "additionalProperties": False,
    }
    input_schema["properties"]["value"] = {"type": "string"}
    tool_input_schemas = {"synthetic_tool": input_schema}

    with pytest.raises(IntelligenceContractError, match="planning_tool_input_field_limit_exceeded"):
        GeneralIntelligenceEngine.response_schema(
            PLAN_RESPONSE_SCHEMA,
            tool_input_schemas=tool_input_schemas,
        )


def test_saved_synthetic_gemini_response_is_a_regression_fixture_for_invalid_intent() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "gemini_invalid_synthetic_response.json"
    response = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert len(response["goal_spec"]["intent"]) > 128
    assert "synthetic" in response["plan"]["goal"]

    goal = (
        "في حديقة خيالية، احسب إجمالي مياه الري لثلاثة أيام لحوضَي «شمس» و«قمر» "
        "(2 لتر يوميًا للأول و1 لتر يوميًا للثاني)، ثم احسب المجموع الكلي باستخدام الأدوات الاصطناعية المدرجة. "
        "هذا اختبار تخطيط فقط؛ لا تدّعِ تنفيذ أي خطوة."
    )
    provider = StructuredFixtureProvider(response)
    with pytest.raises(PlanValidationError, match="goal_spec_intent_invalid"):
        Planner(provider).create(
            goal,
            _calculation_registry(),
            [],
            permissions={"synthetic_bed_total": "public_read", "synthetic_add_totals": "public_read"},
        )

    schema = provider.calls[0]["response_schema"]
    assert schema["properties"]["goal_spec"]["properties"]["intent"]["maxLength"] == 128
    assert schema["properties"]["plan"]["properties"]["goal"]["enum"] == [goal]
    assert "enum" not in PLAN_RESPONSE_SCHEMA["properties"]["goal"]


def test_fixture_plan_rejects_goal_mismatch_missing_inputs_and_empty_bindings() -> None:
    response = json.loads(
        (Path(__file__).parent / "fixtures" / "gemini_invalid_synthetic_response.json").read_text(encoding="utf-8")
    )
    goal = "The exact fictional garden request that the plan must preserve"
    with pytest.raises(PlanValidationError) as captured:
        Planner().validate(
            response["plan"],
            goal=goal,
            tools=_calculation_registry(),
            permissions={"synthetic_bed_total": "public_read", "synthetic_add_totals": "public_read"},
        )

    errors = captured.value.errors
    assert "goal_mismatch" in errors
    assert "missing_required_input:calculate_shams_total.bed_name" in errors
    assert "missing_required_input:calculate_shams_total.days" in errors
    assert "missing_required_input:calculate_shams_total.liters_per_day" in errors
    assert "step_calculate_shams_total_input_binding_invalid:bed_name" in errors
    assert "step_calculate_shams_total_input_binding_invalid:days" in errors


def test_valid_multistep_bindings_pass_but_empty_permissions_fail_closed() -> None:
    goal = "Calculate fictional garden bed totals and add them using local synthetic tools"
    payload = _synthetic_calculation_plan(goal)
    planner = Planner()
    registry = _calculation_registry()

    accepted = planner.validate(
        payload,
        goal=goal,
        tools=registry,
        permissions={"synthetic_bed_total": "public_read", "synthetic_add_totals": "public_read"},
    )
    assert accepted.steps[-1].input == {}
    assert accepted.steps[-1].input_bindings["first_total"] == {
        "source_step_id": "shams",
        "output_key": "total_liters",
    }

    with pytest.raises(PlanValidationError) as captured:
        planner.validate(payload, goal=goal, tools=registry, permissions={})
    assert "permission_not_granted:synthetic_bed_total" in captured.value.errors
    assert "permission_not_granted:synthetic_add_totals" in captured.value.errors


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("intent", "x" * 129, "goal_spec_intent_invalid"),
        ("normalized_goal", "x" * 4001, "goal_spec_normalized_goal_invalid"),
        ("deliverables", ["x" * 513], "goal_spec_deliverables_invalid"),
        ("success_criteria", ["ok"] * 17, "goal_spec_success_criteria_invalid"),
        ("clarification_question", "x" * 1001, "goal_spec_clarification_question_invalid"),
    ],
)
def test_goal_spec_rejects_values_outside_its_declared_schema(field: str, value: Any, error: str) -> None:
    payload = _goal_spec("Bounded synthetic goal")
    payload[field] = value
    with pytest.raises(ValueError, match=error):
        GoalSpec.from_model_payload(payload)
