from __future__ import annotations

import ipaddress
import re
import socket
import ssl
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, quote, urljoin, urlparse, urlunsplit

import httpx
import httpcore
from bs4 import BeautifulSoup

from private_agent.tools.contracts import LegacyToolAdapter, ToolResult, contract_errors


MAX_SEARCH_LIMIT = 10
MAX_REDIRECTS = 3
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_FETCH_CHARS = 200_000
_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata.google.internal",
        "metadata",
        "instance-data.ec2.internal",
    }
)
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_SECRET_VALUE_PATTERN = re.compile(
    r"(?i)\b(api[_ -]?key|token|secret|password|authorization|bearer|credential)\b\s*[:=]\s*([^\s,;]+)"
)


@dataclass
class ResearchResult:
    query: str
    sources: list[dict[str, Any]]
    errors: list[str]


@dataclass(frozen=True)
class _BoundedResponse:
    url: str
    status_code: int
    headers: Mapping[str, str]
    body: bytes


class _PinnedStream(httpx.SyncByteStream):
    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def __iter__(self):
        yield from self._stream

    def close(self) -> None:
        self._stream.close()


class _PinnedNetworkBackend(httpcore.NetworkBackend):
    """Resolve again immediately before connect, never passing a hostname to DNS."""

    def __init__(self, resolver: Callable[[str], tuple[str, ...]]) -> None:
        self._resolver = resolver
        self._backend = httpcore.SyncBackend()

    def connect_tcp(self, host: str, port: int, timeout: float | None = None, local_address: str | None = None, socket_options=None):
        addresses = tuple(self._resolver(host))
        if not addresses:
            raise OSError("dns_resolution_failed")
        if any(_blocked_address(address) for address in addresses):
            raise OSError("private_address_blocked")
        last_error: BaseException | None = None
        for address in addresses:
            try:
                return self._backend.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except Exception as exc:
                last_error = exc
        raise OSError(str(last_error) if last_error is not None else "public_connection_failed")

    def connect_unix_socket(self, path: str, timeout: float | None = None, socket_options=None):
        raise OSError("unix_socket_not_allowed")


class _PinnedHTTPTransport(httpx.BaseTransport):
    def __init__(self, resolver: Callable[[str], tuple[str, ...]]) -> None:
        self._pool = httpcore.ConnectionPool(
            ssl_context=ssl.create_default_context(),
            max_connections=1,
            max_keepalive_connections=0,
            keepalive_expiry=0,
            http1=True,
            http2=False,
            retries=0,
            network_backend=_PinnedNetworkBackend(resolver),
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        core_request = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        response = self._pool.handle_request(core_request)
        return httpx.Response(
            status_code=response.status,
            headers=response.headers,
            stream=_PinnedStream(response.stream),
            extensions=response.extensions,
        )

    def close(self) -> None:
        self._pool.close()


def _default_resolver(host: str) -> tuple[str, ...]:
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        except OSError:
            return ()
        addresses = {str(info[4][0]) for info in infos if info and info[4]}
        return tuple(sorted(addresses))
    return (str(literal),)


def _blocked_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return True
    return bool(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def _validate_public_url(url: str, resolver: Callable[[str], tuple[str, ...]]) -> str:
    if not isinstance(url, str) or not url or len(url) > 2048:
        raise ValueError("url_invalid")
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("scheme_not_allowed")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("url_userinfo_forbidden")
    try:
        hostname = parsed.hostname
        parsed.port
    except ValueError as exc:
        raise ValueError("url_port_invalid") from exc
    if not hostname:
        raise ValueError("url_host_missing")
    hostname = hostname.rstrip(".").lower()
    if hostname in _BLOCKED_HOSTNAMES:
        raise ValueError("private_address_blocked")
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        addresses = tuple(resolver(hostname))
    else:
        addresses = (str(literal),)
    if not addresses:
        raise ValueError("dns_resolution_failed")
    if any(_blocked_address(address) for address in addresses):
        raise ValueError("private_address_blocked")
    return url


def _safe_error(exc: BaseException, *, prefix: str) -> str:
    """Expose an error code, never the original URL or response body."""
    detail = str(exc).split(":", 1)[0].strip()
    known_codes = {
        "url_invalid",
        "scheme_not_allowed",
        "url_userinfo_forbidden",
        "url_port_invalid",
        "url_host_missing",
        "private_address_blocked",
        "dns_resolution_failed",
        "redirect_location_missing",
        "redirect_limit_exceeded",
        "response_too_large",
        "max_chars_invalid",
        "httpx.HTTPStatusError",
    }
    if detail not in known_codes:
        detail = type(exc).__name__
    return f"{prefix}:{detail}"


def _redacted_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    try:
        parsed = urlparse(value)
        hostname = parsed.hostname or ""
        port = parsed.port
        netloc = hostname if port is None else f"{hostname}:{port}"
        # Public result payloads expose only the origin. The complete URL is
        # retained only in the in-process request path and never in reports.
        return urlunsplit((parsed.scheme.lower(), netloc, "/", "", ""))[:2048]
    except (TypeError, ValueError):
        return ""


def _redact_text(value: Any, *, max_length: int = 4096) -> str:
    text = value if isinstance(value, str) else str(value)
    text = _SECRET_VALUE_PATTERN.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    return text[:max_length]


class WebResearchTool:
    name = "web_research"
    risk_level = "high"
    permission_level = "public_read"
    required_capabilities = ["NETWORK_ACCESS"]
    declared_capabilities = ["NETWORK_ACCESS"]
    network_required = True
    planned_effect = "Read public web sources through external HTTP requests"

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_SEARCH_LIMIT},
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def description(self) -> str:
        return "Search public web sources and return candidate URLs and snippets."

    def output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "sources": {"type": "array"},
                "errors": {"type": "array"},
            },
            "required": ["query", "sources", "errors"],
            "additionalProperties": False,
        }

    def __init__(
        self,
        timeout: float = 15.0,
        *,
        max_response_bytes: int = 1_000_000,
        max_redirects: int = MAX_REDIRECTS,
        transport: httpx.BaseTransport | None = None,
        resolver: Callable[[str], tuple[str, ...]] | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("research_timeout_invalid")
        if max_response_bytes < 1 or max_response_bytes > MAX_RESPONSE_BYTES:
            raise ValueError("research_response_limit_invalid")
        if max_redirects < 0 or max_redirects > MAX_REDIRECTS:
            raise ValueError("research_redirect_limit_invalid")
        self.timeout = float(timeout)
        self.max_response_bytes = int(max_response_bytes)
        self.max_redirects = int(max_redirects)
        self._transport = transport
        self._resolver = resolver or _default_resolver

    def _request_bounded(self, url: str) -> _BoundedResponse:
        current = url
        transport = self._transport or _PinnedHTTPTransport(self._resolver)
        with httpx.Client(
            timeout=self.timeout,
            follow_redirects=False,
            trust_env=False,
            headers={"User-Agent": "PrivateAgent/0.1 research"},
            transport=transport,
        ) as client:
            for redirect_count in range(self.max_redirects + 1):
                _validate_public_url(current, self._resolver)
                with client.stream("GET", current) as response:
                    if response.status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("redirect_location_missing")
                        if redirect_count >= self.max_redirects:
                            raise ValueError("redirect_limit_exceeded")
                        current = urljoin(current, location)
                        continue
                    response.raise_for_status()
                    body = bytearray()
                    for chunk in response.iter_bytes(chunk_size=min(64 * 1024, self.max_response_bytes)):
                        body.extend(chunk)
                        if len(body) > self.max_response_bytes:
                            raise ValueError("response_too_large")
                    return _BoundedResponse(
                        url=str(response.url),
                        status_code=int(response.status_code),
                        headers=dict(response.headers),
                        body=bytes(body),
                    )
        raise ValueError("redirect_limit_exceeded")

    @staticmethod
    def _decode_body(response: _BoundedResponse) -> str:
        content_type = str(response.headers.get("content-type", ""))
        match = re.search(r"charset=([A-Za-z0-9._-]+)", content_type, flags=re.IGNORECASE)
        encoding = match.group(1) if match else "utf-8"
        try:
            return response.body.decode(encoding, errors="replace")
        except LookupError:
            return response.body.decode("utf-8", errors="replace")

    def _search_internal(self, query: str, limit: int = 5) -> ResearchResult:
        errors: list[str] = []
        sources: list[dict[str, Any]] = []
        if not isinstance(query, str) or not query.strip():
            return ResearchResult(query=_redact_text(query), sources=[], errors=["query_required"])
        if type(limit) is not int or limit < 1 or limit > MAX_SEARCH_LIMIT:
            return ResearchResult(query=_redact_text(query), sources=[], errors=["limit_invalid"])
        url = "https://html.duckduckgo.com/html/?q=" + quote(query)
        try:
            response = self._request_bounded(url)
            soup = BeautifulSoup(self._decode_body(response), "html.parser")
            for result in soup.select(".result")[:limit]:
                link = result.select_one(".result__a")
                snippet = result.select_one(".result__snippet")
                if link and link.get("href"):
                    sources.append(
                        {
                            "title": _redact_text(link.get_text(" ", strip=True)),
                            "url": _redacted_url(link["href"]),
                            "_fetch_url": link["href"],
                            "snippet": _redact_text(snippet.get_text(" ", strip=True) if snippet else ""),
                        }
                    )
        except Exception as exc:
            errors.append(_safe_error(exc, prefix="search_failed"))
        return ResearchResult(query=_redact_text(query), sources=sources, errors=errors)

    def search(self, query: str, limit: int = 5) -> ResearchResult:
        result = self._search_internal(query, limit=limit)
        return ResearchResult(
            query=result.query,
            sources=[{key: value for key, value in source.items() if key != "_fetch_url"} for source in result.sources],
            errors=list(result.errors),
        )

    def fetch(self, url: str, max_chars: int = 8000) -> dict[str, Any]:
        if not isinstance(max_chars, int) or max_chars < 1 or max_chars > MAX_FETCH_CHARS:
            return {"url": _redacted_url(url), "status": "error", "error": "fetch_failed:max_chars_invalid"}
        original_url = url
        if isinstance(url, str) and url.startswith("//"):
            url = "https:" + url
        parsed = urlparse(url)
        if parsed.path.startswith("/l/"):
            target = parse_qs(parsed.query).get("uddg", [None])[0]
            if target:
                url = target
        try:
            response = self._request_bounded(url)
            soup = BeautifulSoup(self._decode_body(response), "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
            return {
                "url": _redacted_url(response.url),
                "title": _redact_text(soup.title.get_text(strip=True) if soup.title else ""),
                "text": _redact_text(text, max_length=max_chars),
                "status": response.status_code,
            }
        except Exception as exc:
            # Keep the submitted URL for correlation, but do not echo an
            # exception that could contain credentials or response content.
            safe_url = _redacted_url(original_url)
            return {"url": safe_url, "status": "error", "error": _safe_error(exc, prefix="fetch_failed")}

    def execute(self, inputs: dict[str, Any], context: Any) -> ToolResult:
        del context
        query = inputs["query"]
        limit = inputs.get("limit", 5)
        if type(limit) is not int or limit < 1 or limit > MAX_SEARCH_LIMIT:
            return ToolResult.failed("invalid_input", "Research limit is invalid", output={"query": _redact_text(query), "sources": [], "errors": ["limit_invalid"]})
        found = self._search_internal(query, limit=limit)
        errors = list(found.errors)
        if not found.sources:
            return ToolResult.failed(
                "no_sources",
                "Web research returned no sources",
                output={"query": _redact_text(query), "sources": [], "errors": errors},
            )

        fetched: list[dict[str, Any]] = []
        for source in found.sources[:limit]:
            url = source.get("_fetch_url") or source.get("url")
            if not url:
                errors.append("source_missing_url")
                continue
            public_source = {key: value for key, value in source.items() if key != "_fetch_url"}
            fetched.append({**public_source, **self.fetch(url)})
        return ToolResult.success(
            {"query": _redact_text(query), "sources": fetched, "errors": errors},
            metadata={"source_count": len(fetched)},
        )


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, tool: Any) -> None:
        candidate = tool
        if contract_errors(candidate):
            if callable(getattr(tool, "search", None)) and callable(getattr(tool, "fetch", None)):
                candidate = LegacyToolAdapter(tool)
            else:
                raise ValueError(f"Tool does not satisfy contract: {contract_errors(tool)}")
        remaining = contract_errors(candidate)
        if remaining:
            raise ValueError(f"Tool does not satisfy contract: {remaining}")
        self._tools[candidate.name] = candidate

    def available(self) -> list[dict[str, str]]:
        return [{"name": name, "risk_level": getattr(tool, "risk_level", "unknown"), "permission_level": getattr(tool, "permission_level", "unknown")} for name, tool in self._tools.items()]

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                **metadata,
                "description": getattr(tool, "description", lambda: "")(),
                "input_schema": getattr(tool, "input_schema", lambda: {})(),
                "output_schema": getattr(tool, "output_schema", lambda: {})(),
            }
            for name, tool in self._tools.items()
            for metadata in [{
                "name": name,
                "risk_level": getattr(tool, "risk_level", "unknown"),
                "permission_level": getattr(tool, "permission_level", "unknown"),
            }]
        ]

    def get(self, name: str) -> Any:
        return self._tools[name]
