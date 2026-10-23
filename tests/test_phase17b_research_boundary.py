from __future__ import annotations

import httpx
import pytest

from private_agent.tools.research import WebResearchTool, _PinnedNetworkBackend, _validate_public_url


def _tool(handler, *, max_response_bytes: int = 1024) -> WebResearchTool:
    return WebResearchTool(
        timeout=2.0,
        max_response_bytes=max_response_bytes,
        transport=httpx.MockTransport(handler),
        resolver=lambda _host: ("93.184.216.34",),
    )


def test_fetch_rejects_private_literal_and_non_http_urls_without_transport_call() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, text="should not be reached")

    tool = _tool(handler)
    assert tool.fetch("http://127.0.0.1/admin")["status"] == "error"
    assert tool.fetch("ftp://example.test/file")["status"] == "error"
    assert tool.fetch("https://user:pass@example.test/")["status"] == "error"
    assert calls == []


def test_fetch_error_redacts_credentials_and_query_values() -> None:
    tool = _tool(lambda request: httpx.Response(200, text="unreachable"))
    result = tool.fetch("https://user:pass@example.test/path?token=secret-value")
    assert result["status"] == "error"
    assert "user" not in result["url"]
    assert "pass" not in result["url"]
    assert "secret-value" not in result["url"]


def test_fetch_rejects_host_resolving_to_private_address() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, text="should not be reached")

    tool = WebResearchTool(
        transport=httpx.MockTransport(handler),
        resolver=lambda _host: ("10.0.0.7",),
    )
    result = tool.fetch("https://internal.example/")
    assert result["status"] == "error"
    assert "private_address_blocked" in result["error"]
    assert calls == []


def test_fetch_rejects_redirect_to_private_address() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})

    result = _tool(handler).fetch("https://public.example/")
    assert result["status"] == "error"
    assert "private_address_blocked" in result["error"]
    assert calls == ["https://public.example/"]


def test_fetch_success_redacts_redirect_query_values() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://public.example/":
            return httpx.Response(302, headers={"location": "https://public.example/next?token=secret-value"})
        return httpx.Response(200, text="<title>Safe</title>body")

    result = _tool(handler).fetch("https://public.example/")
    assert result["status"] == 200
    assert result["url"] == "https://public.example/"
    assert "secret-value" not in str(result)


def test_search_does_not_expose_internal_fetch_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "html.duckduckgo.com"
        return httpx.Response(
            200,
            text='<div class="result"><a class="result__a" href="https://public.example/next?token=secret-value">Source</a></div>',
        )

    result = _tool(handler).search("safe query")
    assert len(result.sources) == 1
    assert set(result.sources[0]) == {"title", "url", "snippet"}
    assert result.sources[0]["url"] == "https://public.example/"
    assert "secret-value" not in str(result.sources[0])


def test_fetch_error_does_not_echo_exception_text() -> None:
    tool = WebResearchTool(resolver=lambda _host: (_ for _ in ()).throw(ValueError("secret-token-123")))
    result = tool.fetch("https://public.example/")
    assert result["status"] == "error"
    assert "secret-token-123" not in str(result)
    assert "ValueError" in result["error"]


def test_invalid_max_chars_redacts_submitted_url() -> None:
    tool = _tool(lambda request: httpx.Response(200, text="unreachable"))
    result = tool.fetch("https://user:pass@example.test/path?token=secret-value", max_chars=0)
    assert result["status"] == "error"
    assert result["url"] == "https://example.test/"
    assert "secret-value" not in str(result)


def test_search_redacts_query_and_snippet_secret_values() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            text='<div class="result"><a class="result__a" href="https://public.example/">Source</a>'
            '<div class="result__snippet">token: secret-value</div></div>',
        )

    result = _tool(handler).search("token: secret-value")
    assert result.query == "token=[REDACTED]"
    assert result.sources[0]["snippet"] == "token=[REDACTED]"
    assert "secret-value" not in str(result)


def test_dns_pinned_backend_rechecks_addresses_before_socket_connect() -> None:
    calls = {"count": 0}

    def resolver(_host: str) -> tuple[str, ...]:
        calls["count"] += 1
        return ("93.184.216.34",) if calls["count"] == 1 else ("127.0.0.1",)

    backend = _PinnedNetworkBackend(resolver)
    _validate_public_url("https://public.example/", resolver)
    with pytest.raises(OSError, match="private_address_blocked"):
        backend.connect_tcp("public.example", 443, timeout=0.1)
    assert calls["count"] == 2


def test_dns_pinned_backend_rejects_mixed_allowed_and_blocked_answers() -> None:
    backend = _PinnedNetworkBackend(lambda _host: ("93.184.216.34", "10.0.0.7"))
    with pytest.raises(OSError, match="private_address_blocked"):
        backend.connect_tcp("public.example", 443, timeout=0.1)


def test_public_ipv6_literal_is_validated_without_downgrade() -> None:
    assert _validate_public_url("https://[2001:4860:4860::8888]/", lambda _host: ()) == "https://[2001:4860:4860::8888]/"


def test_fetch_enforces_bounded_response_before_html_parse() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, text="A" * 2048)

    result = _tool(handler, max_response_bytes=128).fetch("https://public.example/")
    assert result["status"] == "error"
    assert "response_too_large" in result["error"]


def test_fetch_returns_bounded_public_html_from_mock_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://public.example/"
        return httpx.Response(200, text="<html><title>Safe</title><body>bounded evidence token: secret-value</body></html>")

    result = _tool(handler).fetch("https://public.example/")
    assert result["status"] == 200
    assert result["url"] == "https://public.example/"
    assert result["title"] == "Safe"
    assert "bounded evidence" in result["text"]
    assert "secret-value" not in result["text"]
