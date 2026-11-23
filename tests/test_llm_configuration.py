from __future__ import annotations

import httpx
import pytest

from private_agent import api
from private_agent.core.llm import LLMProviderConfigError, build_provider


@pytest.mark.parametrize(
    ("environ", "configured", "expected_key"),
    [
        ({}, False, ""),
        ({"GEMINI_API_KEY": ""}, False, ""),
        ({"GEMINI_API_KEY": " \t\n"}, False, " \t\n"),
        ({"GEMINI_API_KEY": "synthetic-test-key"}, True, "synthetic-test-key"),
    ],
    ids=["missing", "empty", "whitespace-only", "nonempty"],
)
def test_gemini_configuration_readiness_and_generation_are_consistent(
    environ: dict[str, str], configured: bool, expected_key: str
) -> None:
    provider = build_provider(environ)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]},
            request=request,
        )

    provider.transport = httpx.MockTransport(handler)
    assert provider.is_configured is configured
    assert provider.config.api_key == expected_key

    if not configured:
        with pytest.raises(LLMProviderConfigError, match="API key"):
            provider.generate_json(system_prompt="s", user_prompt="u", response_schema={})
        assert calls == []
        return

    response = provider.generate_json(system_prompt="s", user_prompt="u", response_schema={})
    assert response.data == {"ok": True}
    assert len(calls) == 1
    assert calls[0].headers["x-goog-api-key"] == expected_key


@pytest.mark.parametrize(
    ("api_key", "selects_provider"),
    [
        (None, False),
        ("", False),
        (" \t\n", False),
        ("synthetic-test-key", True),
    ],
    ids=["missing", "empty", "whitespace-only", "nonempty"],
)
def test_api_selects_gemini_only_for_nonempty_configuration(
    tmp_path, monkeypatch: pytest.MonkeyPatch, api_key: str | None, selects_provider: bool
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    if api_key is not None:
        monkeypatch.setenv("GEMINI_API_KEY", api_key)

    provider = object()
    provider_builds: list[None] = []

    def fake_build_provider():
        provider_builds.append(None)
        return provider

    monkeypatch.setattr(api, "build_provider", fake_build_provider)
    orchestrator = api.build_orchestrator(database_path=str(tmp_path / "agent.sqlite3"))
    try:
        assert (orchestrator.planner.provider is provider) is selects_provider
        assert len(provider_builds) == int(selects_provider)
    finally:
        orchestrator.store.close()
