from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Mapping

import httpx


class LLMError(RuntimeError):
    """Base class for provider-neutral LLM failures."""


class LLMProviderConfigError(LLMError):
    """Raised when a provider cannot be used with the supplied configuration."""


class LLMTransportError(LLMError):
    """Raised when a request cannot reach the provider."""


class LLMHTTPError(LLMError):
    """Raised for a non-success provider response after retry handling."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMRateLimitError(LLMHTTPError):
    """Raised when the provider still returns HTTP 429 after retries."""


class LLMResponseError(LLMError):
    """Raised when the provider response has an unexpected shape."""


class LLMStructuredOutputError(LLMResponseError):
    """Raised when structured JSON output cannot be decoded."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "gemini"
    model: str = "gemini-2.0-flash"
    api_key: str = ""
    timeout: float = 30.0
    max_retries: int = 2
    backoff_base: float = 0.5
    backoff_max: float = 8.0

    def validate(self) -> None:
        if not self.provider.strip():
            raise LLMProviderConfigError("LLM provider is required")
        if not self.model.strip():
            raise LLMProviderConfigError("LLM model is required")
        if self.timeout <= 0:
            raise LLMProviderConfigError("LLM timeout must be positive")
        if self.max_retries < 0:
            raise LLMProviderConfigError("LLM max_retries cannot be negative")
        if self.backoff_base < 0 or self.backoff_max < 0:
            raise LLMProviderConfigError("LLM backoff values cannot be negative")


@dataclass(frozen=True)
class LLMResponse:
    data: Any
    raw_text: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Provider-neutral interface used by the planner and future components."""

    @abstractmethod
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> LLMResponse:
        raise NotImplementedError


def _redact(text: str, secrets: tuple[str, ...]) -> str:
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


class GeminiProvider(LLMProvider):
    """Gemini REST provider with structured JSON output and safe retries."""

    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(
        self,
        config: LLMConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.transport = transport
        self.sleep_fn = sleep_fn
        self.config.validate()

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> LLMResponse:
        if not self.config.api_key:
            raise LLMProviderConfigError("Gemini API key is required")

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": response_schema,
            },
        }
        url = self.endpoint.format(model=self.config.model)
        secrets = (self.config.api_key,)
        last_status: int | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                with httpx.Client(timeout=self.config.timeout, transport=self.transport) as client:
                    response = client.post(url, params={"key": self.config.api_key}, json=payload)
            except httpx.HTTPError as exc:
                safe = _redact(str(exc), secrets)
                if attempt < self.config.max_retries:
                    self.sleep_fn(self._backoff(attempt))
                    continue
                raise LLMTransportError(f"Gemini request failed: {safe}") from exc

            last_status = response.status_code
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < self.config.max_retries:
                    delay = _retry_after_seconds(response.headers.get("Retry-After"))
                    self.sleep_fn(delay if delay is not None else self._backoff(attempt))
                    continue
                body = _redact(response.text[:500], secrets)
                if response.status_code == 429:
                    raise LLMRateLimitError(
                        f"Gemini rate limit persisted after retries: {body}",
                        status_code=response.status_code,
                    )
                raise LLMHTTPError(
                    f"Gemini server error after retries ({response.status_code}): {body}",
                    status_code=response.status_code,
                )

            if response.status_code >= 400:
                body = _redact(response.text[:500], secrets)
                raise LLMHTTPError(
                    f"Gemini request rejected ({response.status_code}): {body}",
                    status_code=response.status_code,
                )

            try:
                response_payload = response.json()
            except ValueError as exc:
                raise LLMResponseError("Gemini returned a non-JSON envelope") from exc

            try:
                raw_text = response_payload["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError) as exc:
                raise LLMResponseError("Gemini response did not contain candidate text") from exc

            try:
                data = json.loads(raw_text)
            except (TypeError, json.JSONDecodeError) as exc:
                safe = _redact(str(raw_text)[:500], secrets)
                raise LLMStructuredOutputError(f"Gemini returned malformed JSON: {safe}") from exc

            usage = response_payload.get("usageMetadata", {})
            return LLMResponse(data=data, raw_text=raw_text, model=self.config.model, usage=usage)

        raise LLMHTTPError("Gemini request failed without a response", status_code=last_status)

    def _backoff(self, attempt: int) -> float:
        return min(self.config.backoff_max, self.config.backoff_base * (2**attempt))


def build_provider(environ: Mapping[str, str] | None = None) -> LLMProvider:
    """Build the configured provider without exposing credentials or requiring them at import time."""

    values = os.environ if environ is None else environ
    provider_name = values.get("AGENT_LLM_PROVIDER", "gemini").strip().lower()
    if provider_name != "gemini":
        raise LLMProviderConfigError(f"Unsupported LLM provider: {provider_name}")

    def as_float(name: str, default: str) -> float:
        try:
            return float(values.get(name, default))
        except ValueError as exc:
            raise LLMProviderConfigError(f"Invalid numeric setting: {name}") from exc

    def as_int(name: str, default: str) -> int:
        try:
            return int(values.get(name, default))
        except ValueError as exc:
            raise LLMProviderConfigError(f"Invalid integer setting: {name}") from exc

    config = LLMConfig(
        provider=provider_name,
        model=values.get("GEMINI_MODEL", "gemini-2.0-flash"),
        api_key=values.get("GEMINI_API_KEY", ""),
        timeout=as_float("AGENT_LLM_TIMEOUT", "30"),
        max_retries=as_int("AGENT_LLM_MAX_RETRIES", "2"),
        backoff_base=as_float("AGENT_LLM_BACKOFF_BASE", "0.5"),
        backoff_max=as_float("AGENT_LLM_BACKOFF_MAX", "8"),
    )
    return GeminiProvider(config)
