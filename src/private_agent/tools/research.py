from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

import httpx
from bs4 import BeautifulSoup


@dataclass
class ResearchResult:
    query: str
    sources: list[dict[str, Any]]
    errors: list[str]


class WebResearchTool:
    name = "web_research"
    risk_level = "low"
    permission_level = "public_read"

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    def search(self, query: str, limit: int = 5) -> ResearchResult:
        errors: list[str] = []
        sources: list[dict[str, Any]] = []
        url = "https://html.duckduckgo.com/html/?q=" + quote(query)
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True, headers={"User-Agent": "PrivateAgent/0.1 research"}) as client:
                response = client.get(url)
                response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for result in soup.select(".result")[:limit]:
                link = result.select_one(".result__a")
                snippet = result.select_one(".result__snippet")
                if link and link.get("href"):
                    sources.append({"title": link.get_text(" ", strip=True), "url": link["href"], "snippet": snippet.get_text(" ", strip=True) if snippet else ""})
        except Exception as exc:
            errors.append(f"search_failed: {type(exc).__name__}: {exc}")
        return ResearchResult(query=query, sources=sources, errors=errors)

    def fetch(self, url: str, max_chars: int = 8000) -> dict[str, Any]:
        if url.startswith("//"):
            url = "https:" + url
        parsed = urlparse(url)
        if parsed.path.startswith("/l/"):
            target = parse_qs(parsed.query).get("uddg", [None])[0]
            if target:
                url = target
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True, headers={"User-Agent": "PrivateAgent/0.1 research"}) as client:
                response = client.get(url)
                response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]): tag.decompose()
            text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
            return {"url": str(response.url), "title": soup.title.get_text(strip=True) if soup.title else "", "text": text[:max_chars], "status": response.status_code}
        except Exception as exc:
            return {"url": url, "status": "error", "error": f"fetch_failed: {type(exc).__name__}: {exc}"}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, tool: Any) -> None:
        self._tools[tool.name] = tool

    def available(self) -> list[dict[str, str]]:
        return [{"name": name, "risk_level": getattr(tool, "risk_level", "unknown"), "permission_level": getattr(tool, "permission_level", "unknown")} for name, tool in self._tools.items()]

    def get(self, name: str) -> Any:
        return self._tools[name]
