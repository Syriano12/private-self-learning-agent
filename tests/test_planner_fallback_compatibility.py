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


class ProposedScopeResearchTool(ScopeRequiredResearchTool):
    def planning_inputs(self, goal: str) -> dict:
        return {"scope": {"goal": goal}}


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


def test_unavailable_planner_ignores_learned_preference_for_incompatible_tool() -> None:
    tools = ToolRegistry()
    tools.register(ScopeRequiredResearchTool())
    tools.register(WebResearchTool())
    learned = {
        "status": "active",
        "confidence": 0.95,
        "strategy_id": "prefer-scope-tool",
        "preferred_action": {"type": "prefer_tool", "tool": "scope_research"},
    }

    plan = Planner().create(
        "Research practical online income ideas",
        tools,
        [],
        learned_strategies=[learned],
    )

    assert plan.steps[0].tool == "web_research"
    assert plan.steps[0].input["query"] == "Research practical online income ideas"


def test_compatible_generic_fallback_still_requires_permission() -> None:
    tools = ToolRegistry()
    tools.register(WebResearchTool())

    with pytest.raises(PlanValidationError, match="permission_not_granted:web_research"):
        Planner().create("Research a general topic", tools, [], permissions={})


def test_compatible_registered_input_proposal_still_requires_permission() -> None:
    tools = ToolRegistry()
    tools.register(ProposedScopeResearchTool())

    with pytest.raises(PlanValidationError, match="permission_not_granted:scope_research"):
        Planner().create("Investigate a scoped task", tools, [], permissions={})
