"""Bounded passive retrieval from explicitly allowlisted public sources.

This module is the only research-stage path that may perform public HTTP GETs.
It accepts no credentials, follows only the safety boundary provided by
``WebResearchTool`` (HTTPS, DNS/IP checks, redirect validation, response and
time limits), and never turns a public page into authorization.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping
from urllib.parse import urlparse, urlunsplit

import httpx

from private_agent.tools.research import WebResearchTool

MAX_PASSIVE_SOURCES = 4
MAX_PASSIVE_URL_LENGTH = 2048
MAX_PASSIVE_EXCERPT_CHARS = 8_000
MAX_PASSIVE_CACHE_ENTRIES = 16
MAX_PASSIVE_CACHE_TTL_SECONDS = 300.0

_ALLOWED_HOSTS = {
    "hackerone_security": frozenset({"hackerone.com", "www.hackerone.com"}),
    "hackerone_program": frozenset({"hackerone.com", "www.hackerone.com"}),
    "nvd_cve": frozenset({"nvd.nist.gov"}),
}
_HACKERONE_PROGRAM_PATH = re.compile(r"^/[a-z0-9][a-z0-9-]{1,80}$")
_NVD_CVE_PATH = re.compile(r"^/vuln/detail/CVE-[0-9]{4}-[0-9]{4,}$")
_SECRET_TEXT = re.compile(
    r"(?i)(\b(?:authorization|cookie|api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|passwd|secret|private[_-]?key)\b\s*[:=]\s*)(['\"]?)[^\s,'\"}]+\2"
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}")
_PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z ]+-----[\s\S]*?-----END [A-Z ]+-----")


def _public_url(source_type: str, value: Any) -> str:
    if source_type not in _ALLOWED_HOSTS:
        raise ValueError("passive_source_type_invalid")
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_PASSIVE_URL_LENGTH:
        raise ValueError("passive_source_url_invalid")
    parsed = urlparse(value.strip())
    if parsed.scheme.lower() != "https":
        raise ValueError("passive_source_https_required")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("passive_source_userinfo_forbidden")
    if parsed.fragment or parsed.query:
        raise ValueError("passive_source_query_forbidden")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("passive_source_port_invalid") from exc
    if port not in (None, 443):
        raise ValueError("passive_source_port_invalid")
    host = (parsed.hostname or "").rstrip(".").lower()
    if host not in _ALLOWED_HOSTS[source_type]:
        raise ValueError("passive_source_host_not_allowlisted")
    path = parsed.path.rstrip("/") or "/"
    if source_type == "hackerone_security" and path != "/security":
        raise ValueError("passive_source_path_not_allowlisted")
    if source_type == "hackerone_program" and not _HACKERONE_PROGRAM_PATH.fullmatch(path):
        raise ValueError("passive_source_path_not_allowlisted")
    if source_type == "nvd_cve" and not _NVD_CVE_PATH.fullmatch(path):
        raise ValueError("passive_source_path_not_allowlisted")
    return urlunsplit(("https", host, path, "", ""))


def normalize_passive_source_requests(value: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """Validate the explicit public-source contract without performing I/O."""
    payload = dict(value or {})
    raw_sources = payload.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ValueError("passive_sources_list_required")
    if len(raw_sources) > MAX_PASSIVE_SOURCES:
        raise ValueError("passive_sources_too_many")
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_sources:
        if not isinstance(raw, Mapping) or set(raw) != {"source_type", "url"}:
            raise ValueError("passive_source_fields_invalid")
        source_type = raw.get("source_type")
        if not isinstance(source_type, str) or source_type not in _ALLOWED_HOSTS:
            raise ValueError("passive_source_type_invalid")
        url = _public_url(source_type, raw.get("url"))
        key = (source_type, url)
        if key in seen:
            raise ValueError("passive_source_duplicate")
        seen.add(key)
        normalized.append({"source_type": source_type, "url": url})
    return normalized


def _redact_text(value: Any, *, limit: int = MAX_PASSIVE_EXCERPT_CHARS) -> str:
    text = value if isinstance(value, str) else str(value or "")
    text = _PRIVATE_KEY.sub("[REDACTED_PRIVATE_KEY]", text)
    text = _SECRET_TEXT.sub(r"\1\2[REDACTED]\2", text)
    text = _BEARER.sub("Bearer [REDACTED]", text)
    return text[:limit]


def _safe_error(value: Any) -> str:
    text = value if isinstance(value, str) else type(value).__name__
    code = text.split(":", 1)[0].strip()
    return code[:128] or "source_fetch_failed"


class PassiveResearchAdapter:
    """Fetch only normalized public sources, sequentially and with a small cache."""

    def __init__(
        self,
        *,
        timeout: float = 15.0,
        max_response_bytes: int = 1_000_000,
        fetcher: Callable[..., dict[str, Any]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        transport: httpx.BaseTransport | None = None,
        resolver: Callable[[str], tuple[str, ...]] | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("passive_timeout_invalid")
        self._timeout = timeout
        self._max_response_bytes = max_response_bytes
        self._fetcher = fetcher
        self._transport = transport
        self._resolver = resolver
        self._clock = clock
        self._cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}

    def _fetch(self, source_type: str, url: str, *, max_chars: int) -> dict[str, Any]:
        if self._fetcher is not None:
            return self._fetcher(url, max_chars=max_chars)
        tool = WebResearchTool(
            timeout=self._timeout,
            max_response_bytes=self._max_response_bytes,
            max_redirects=3,
            transport=self._transport,
            resolver=self._resolver,
            url_validator=lambda candidate: _public_url(source_type, candidate),
        )
        return tool.fetch(url, max_chars=max_chars)

    def _cached(self, key: tuple[str, str]) -> dict[str, Any] | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if self._clock() - stored_at > MAX_PASSIVE_CACHE_TTL_SECONDS:
            self._cache.pop(key, None)
            return None
        return dict(value)

    def _store(self, key: tuple[str, str], value: dict[str, Any]) -> None:
        if len(self._cache) >= MAX_PASSIVE_CACHE_ENTRIES:
            oldest = min(self._cache, key=lambda item: self._cache[item][0])
            self._cache.pop(oldest, None)
        self._cache[key] = (self._clock(), dict(value))

    @staticmethod
    def _fit(output: dict[str, Any], max_output_bytes: int) -> dict[str, Any]:
        def size() -> int:
            return len(json.dumps(output, ensure_ascii=False, sort_keys=True).encode("utf-8"))

        if size() <= max_output_bytes:
            return output
        output["limitations"] = sorted(set([*output["limitations"], "stage_output_limit_reached"]))
        for retrieval in output["retrievals"]:
            retrieval["text_excerpt"] = ""
        while size() > max_output_bytes and output["retrievals"]:
            output["retrievals"].pop()
        if size() > max_output_bytes:
            return {
                "kind": "PASSIVE_RESEARCH",
                "network_accessed": output["network_accessed"],
                "source_requests": output["source_requests"],
                "network_requests": output["network_requests"],
                "cache_hits": output["cache_hits"],
                "retrievals": [],
                "errors": ["stage_output_limit_reached"],
                "limitations": ["stage_output_limit_reached"],
            }
        return output

    def collect(
        self,
        source_requests: list[Mapping[str, Any]],
        *,
        max_items: int,
        max_output_bytes: int,
    ) -> dict[str, Any]:
        normalized = normalize_passive_source_requests({"sources": source_requests})
        bounded = normalized[:max_items]
        retrievals: list[dict[str, Any]] = []
        errors: list[str] = []
        network_requests = 0
        cache_hits = 0
        for request in bounded:
            source_type = request["source_type"]
            url = request["url"]
            key = (source_type, url)
            record = self._cached(key)
            if record is not None:
                cache_hits += 1
                record["provenance"] = {**record["provenance"], "cache_hit": True}
                retrievals.append(record)
                continue
            network_requests += 1
            retrieved_at = datetime.now(timezone.utc).isoformat()
            try:
                fetched = self._fetch(source_type, url, max_chars=min(MAX_PASSIVE_EXCERPT_CHARS, max_output_bytes))
            except Exception as exc:
                fetched = {"status": "error", "error": type(exc).__name__}
            status = fetched.get("status") if isinstance(fetched, Mapping) else "error"
            text = _redact_text(fetched.get("text", "") if isinstance(fetched, Mapping) else "")
            if not isinstance(status, int) or status < 100 or status > 599:
                error = _safe_error(fetched.get("error", "source_fetch_failed") if isinstance(fetched, Mapping) else "source_fetch_failed")
                errors.append(f"{source_type}:{error}")
                record = {
                    "source_type": source_type,
                    "source_url": url,
                    "retrieved_at": retrieved_at,
                    "status_code": None,
                    "content_sha256": "",
                    "content_length": 0,
                    "text_excerpt": "",
                    "error": error,
                    "provenance": {"adapter": source_type, "transport": "https", "authorization_basis": "public_source_only", "cache_hit": False},
                }
            else:
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                record = {
                    "source_type": source_type,
                    "source_url": url,
                    "retrieved_at": retrieved_at,
                    "status_code": int(status),
                    "content_sha256": digest,
                    "content_length": len(text.encode("utf-8")),
                    "text_excerpt": text,
                    "provenance": {"adapter": source_type, "transport": "https", "authorization_basis": "public_source_only", "cache_hit": False},
                }
            self._store(key, record)
            retrievals.append(dict(record))
        return self._fit(
            {
                "kind": "PASSIVE_RESEARCH",
                "network_accessed": bool(network_requests),
                "source_requests": len(bounded),
                "network_requests": network_requests,
                "cache_hits": cache_hits,
                "retrievals": retrievals,
                "errors": errors[:max_items],
                "limitations": ["public_sources_only", "public_source_is_not_authorization", "active_testing_not_performed"],
            },
            max_output_bytes,
        )


__all__ = ["MAX_PASSIVE_SOURCES", "PassiveResearchAdapter", "normalize_passive_source_requests"]
