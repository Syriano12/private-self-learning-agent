from __future__ import annotations

import pytest

from private_agent.api import build_orchestrator
from private_agent.core.planner import PlanValidationError, Planner
from private_agent.tools.research import ToolRegistry, WebResearchTool


class ScopeRequiredResearchTool(WebResearchTool):
    name = "scope_research"

    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"scope": {"type": "object"}},
            "required": ["scope"],
            "additionalProperties": False,
        }


class ProposedWebResearchTool(WebResearchTool):
    def planning_inputs(self, goal: str) -> dict:
        return {"query": goal, "limit": 4}


class InvalidLimitProposalTool(WebResearchTool):
    def planning_inputs(self, goal: str) -> dict:
        return {"query": goal, "limit": 0}


class MinimumLimitResearchTool(WebResearchTool):
    def input_schema(self) -> dict:
        schema = super().input_schema()
        schema["properties"]["limit"] = {"type": "integer", "minimum": 8, "maximum": 9}
        return schema


def test_default_application_registry_has_clear_outcome_when_no_fallback_inputs_fit(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    orchestrator = build_orchestrator(database_path=str(tmp_path / "agent.sqlite3"))
    try:
        assert orchestrator.planner.provider is None
        assert orchestrator.tools.available()[0]["name"] == "bugbounty_analysis"
        assert orchestrator.tools.get("bugbounty_analysis").input_schema()["required"] == ["scope"]

        with pytest.raises(PlanValidationError) as caught:
            orchestrator.planner.create(
                "Help me research practical ways to grow an online income stream",
                orchestrator.tools,
                [],
            )

        assert caught.value.errors == [
            "no_compatible_tool_for_unavailable_planner",
            "llm_provider_unavailable",
        ]
    finally:
        orchestrator.store.close()


def test_fallback_selector_rejects_preferred_tool_with_incompatible_inputs() -> None:
    tools = ToolRegistry()
    tools.register(ScopeRequiredResearchTool())
    tools.register(WebResearchTool())
    goal = "Research practical online income ideas"
    # Isolate the selector: Planner.create canonicalizes and validates stored
    # guidance before it reaches this method.
    plan = Planner()._unavailable_fallback(
        goal,
        tools,
        [],
        [{
            "status": "active",
            "confidence": 0.95,
            "strategy_id": "selector-unit-test",
            "preferred_action": {"type": "prefer_tool", "tool": "scope_research"},
        }],
        permissions={"scope_research": "public_read", "web_research": "public_read"},
    )

    assert plan.steps[0].tool == "web_research"
    assert plan.steps[0].input["query"] == goal


def test_compatible_generic_fallback_still_requires_permission() -> None:
    tools = ToolRegistry()
    tools.register(WebResearchTool())

    with pytest.raises(PlanValidationError, match="permission_not_granted:web_research"):
        Planner().create("Research a general topic", tools, [], permissions={})


def test_compatible_registered_input_proposal_still_requires_permission() -> None:
    tools = ToolRegistry()
    tools.register(ProposedWebResearchTool())

    with pytest.raises(PlanValidationError, match="permission_not_granted:web_research"):
        Planner().create("Research a general topic", tools, [], permissions={})


def test_only_compatible_tool_remains_selected_when_avoided() -> None:
    tools = ToolRegistry()
    tools.register(WebResearchTool())
    tools.register(ScopeRequiredResearchTool())
    goal = "Research a general topic"
    learned = {
        "status": "active",
        "confidence": 0.95,
        "strategy_id": "avoid-web-research",
        "avoided_action": {"type": "avoid_unchanged_retry", "tool": "web_research"},
    }

    plan = Planner()._unavailable_fallback(
        goal,
        tools,
        [],
        [learned],
        permissions={"web_research": "public_read", "scope_research": "public_read"},
    )

    assert plan.steps[0].tool == "web_research"
    assert plan.steps[0].input["query"] == goal


def test_invalid_registered_input_proposal_is_rejected_before_fallback() -> None:
    tools = ToolRegistry()
    tools.register(InvalidLimitProposalTool())
    goal = "Research a general topic"

    plan = Planner().create(
        goal,
        tools,
        [],
        permissions={"web_research": "public_read"},
    )

    assert plan.steps[0].input == {"query": goal, "limit": 6}


def test_fallback_clamps_limit_to_schema_bounds() -> None:
    tools = ToolRegistry()
    tools.register(MinimumLimitResearchTool())
    goal = "Research a general topic"

    plan = Planner().create(
        goal,
        tools,
        [],
        permissions={"web_research": "public_read"},
    )

    assert plan.steps[0].input == {"query": goal, "limit": 8}
