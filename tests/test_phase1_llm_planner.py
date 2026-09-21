from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from private_agent.core.llm import (
    GeminiProvider,
    LLMConfig,
    LLMHTTPError,
    LLMProviderConfigError,
    LLMRateLimitError,
    LLMStructuredOutputError,
)
from private_agent.core.planner import PLAN_RESPONSE_SCHEMA, PlanValidationError, Planner
from private_agent.tools.research import ToolRegistry, WebResearchTool


class FakeProvider:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def generate_json(self, *, system_prompt: str, user_prompt: str, response_schema: dict):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": json.loads(user_prompt),
                "response_schema": response_schema,
            }
        )
        from private_agent.core.llm import LLMResponse

        return LLMResponse(self.responses[len(self.calls) - 1], json.dumps(self.responses[len(self.calls) - 1]), "fake")


def registry() -> ToolRegistry:
    tools = ToolRegistry()
    tools.register(WebResearchTool())
    return tools


def valid_plan(query: str, *, step_id: str = "step_1") -> dict:
    return {
        "goal": query,
        "rationale": "Use the public research tool because it can gather evidence.",
        "steps": [
            {
                "id": step_id,
                "objective": "Find evidence",
                "tool": "web_research",
                "input": {"query": query, "limit": 3},
                "reason": "The tool is registered and read-only.",
                "depends_on": [],
            }
        ],
    }


def provider_with_handler(handler, *, api_key: str = "test-secret", **overrides) -> GeminiProvider:
    config = LLMConfig(api_key=api_key, backoff_base=0, backoff_max=0, **overrides)
    return GeminiProvider(config, transport=httpx.MockTransport(handler), sleep_fn=lambda _: None)


def test_gemini_provider_returns_structured_json_and_sends_prompt_and_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.params["key"] == "test-secret"
        body = json.loads(request.content)
        assert body["systemInstruction"]["parts"][0]["text"] == "system"
        assert body["contents"][0]["parts"][0]["text"] == "user"
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert body["generationConfig"]["responseSchema"] == {"type": "object"}
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}],
                "usageMetadata": {"totalTokenCount": 12},
            },
            request=request,
        )

    response = provider_with_handler(handler).generate_json(
        system_prompt="system", user_prompt="user", response_schema={"type": "object"}
    )
    assert response.data == {"ok": True}
    assert response.model == "gemini-2.0-flash"
    assert response.usage["totalTokenCount"] == 12


def test_gemini_provider_rejects_malformed_structured_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "not-json"}]}}]},
            request=request,
        )

    with pytest.raises(LLMStructuredOutputError, match="malformed JSON"):
        provider_with_handler(handler).generate_json(system_prompt="s", user_prompt="u", response_schema={})


def test_gemini_provider_retries_429_and_honors_retry_after() -> None:
    calls = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"Retry-After": "0"}, text="rate limited", request=request)

    provider = GeminiProvider(
        LLMConfig(api_key="secret", max_retries=1, backoff_base=2, backoff_max=2),
        transport=httpx.MockTransport(handler),
        sleep_fn=delays.append,
    )
    with pytest.raises(LLMRateLimitError):
        provider.generate_json(system_prompt="s", user_prompt="u", response_schema={})
    assert calls == 2
    assert delays == [0.0]


def test_gemini_provider_retries_5xx_then_succeeds() -> None:
    responses = [500, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        status = responses.pop(0)
        if status == 500:
            return httpx.Response(status, text="temporary server error", request=request)
        return httpx.Response(
            status,
            json={"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]},
            request=request,
        )

    response = provider_with_handler(handler, max_retries=1).generate_json(
        system_prompt="s", user_prompt="u", response_schema={}
    )
    assert response.data == {"ok": True}
    assert responses == []


def test_provider_errors_do_not_expose_api_key() -> None:
    secret = "super-secret-key"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=f"invalid key {secret}", request=request)

    with pytest.raises(LLMHTTPError) as caught:
        provider_with_handler(handler, api_key=secret).generate_json(
            system_prompt="s", user_prompt="u", response_schema={}
        )
    assert secret not in str(caught.value)


def test_provider_configuration_requires_key_only_when_called() -> None:
    provider = GeminiProvider(LLMConfig(api_key=""))
    with pytest.raises(LLMProviderConfigError, match="API key"):
        provider.generate_json(system_prompt="s", user_prompt="u", response_schema={})


def test_planner_passes_goal_tools_memory_constraints_and_permissions_to_provider() -> None:
    fake = FakeProvider([valid_plan("compare local agents")])
    planner = Planner(fake)
    result = planner.create(
        "compare local agents",
        registry(),
        [{"concept": "previous evidence"}],
        constraints={"max_steps": 3},
        permissions={"web_research": "public_read"},
    )
    assert result.steps[0].tool == "web_research"
    context = fake.calls[0]["user_prompt"]
    assert context["goal"] == "compare local agents"
    assert context["relevant_memory"] == [{"concept": "previous evidence"}]
    assert context["constraints"] == {"max_steps": 3}
    assert context["permissions"] == {"web_research": "public_read"}
    assert context["available_tools"][0]["input_schema"]["required"] == ["query"]
    assert fake.calls[0]["response_schema"] == PLAN_RESPONSE_SCHEMA


def test_different_goals_produce_different_plans_without_keyword_routing() -> None:
    first = valid_plan("collect evidence about local Python agents", step_id="research")
    second = valid_plan("find licensing constraints for Android deployment", step_id="licensing")
    fake = FakeProvider([first, second])
    planner = Planner(fake)
    first_plan = planner.create("collect evidence about local Python agents", registry(), [])
    second_plan = planner.create("find licensing constraints for Android deployment", registry(), [])
    assert first_plan.steps[0].id != second_plan.steps[0].id
    assert first_plan.steps[0].input["query"] != second_plan.steps[0].input["query"]
    assert [call["user_prompt"]["goal"] for call in fake.calls] == [
        "collect evidence about local Python agents",
        "find licensing constraints for Android deployment",
    ]


def test_plan_validation_rejects_unknown_tool() -> None:
    payload = valid_plan("research")
    payload["steps"][0]["tool"] = "shell"
    with pytest.raises(PlanValidationError, match="unknown_tool:shell"):
        Planner().validate(payload, goal="research", tools=registry())


def test_plan_validation_rejects_missing_required_input() -> None:
    payload = valid_plan("research")
    del payload["steps"][0]["input"]["query"]
    with pytest.raises(PlanValidationError, match="missing_required_input:step_1.query"):
        Planner().validate(payload, goal="research", tools=registry())


def test_plan_validation_rejects_duplicate_step_ids() -> None:
    payload = valid_plan("research")
    payload["steps"].append({**payload["steps"][0], "objective": "second"})
    with pytest.raises(PlanValidationError, match="duplicate_step_id:step_1"):
        Planner().validate(payload, goal="research", tools=registry())


def test_plan_validation_rejects_circular_dependency() -> None:
    payload = valid_plan("research")
    payload["steps"] = [
        {**payload["steps"][0], "id": "a", "depends_on": ["b"]},
        {**payload["steps"][0], "id": "b", "depends_on": ["a"]},
    ]
    with pytest.raises(PlanValidationError, match="circular_dependency"):
        Planner().validate(payload, goal="research", tools=registry())


def test_planner_fallback_does_not_use_keyword_routing() -> None:
    plan = Planner().create("an arbitrary goal with no research keyword", registry(), [])
    assert plan.steps[0].tool == "web_research"
    assert plan.steps[0].input["query"] == "an arbitrary goal with no research keyword"
    assert "keyword" not in plan.rationale.lower()
