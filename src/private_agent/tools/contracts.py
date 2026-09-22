from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


TOOL_STATUSES = {"success", "failed", "blocked"}


@dataclass(frozen=True)
class ToolError:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


@dataclass
class ToolResult:
    status: str
    output: Any = None
    error: ToolError | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, output: Any, *, metadata: dict[str, Any] | None = None) -> "ToolResult":
        return cls("success", output=output, metadata=metadata or {})

    @classmethod
    def failed(
        cls,
        code: str,
        message: str,
        *,
        output: Any = None,
        details: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            "failed",
            output=output,
            error=ToolError(code, message, details or {}),
            metadata=metadata or {},
        )

    @classmethod
    def blocked(
        cls,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            "blocked",
            error=ToolError(code, message, details or {}),
            metadata=metadata or {},
        )


class ToolContract(Protocol):
    name: str
    risk_level: str
    permission_level: str

    def description(self) -> str:
        ...

    def input_schema(self) -> dict[str, Any]:
        ...

    def output_schema(self) -> dict[str, Any]:
        ...

    def execute(self, inputs: dict[str, Any], context: Any) -> ToolResult:
        ...


def validate_schema_value(value: Any, schema: dict[str, Any], *, path: str = "value") -> list[str]:
    """Return deterministic validation errors for the supported JSON-schema subset."""

    if not isinstance(schema, dict):
        return [f"invalid_schema:{path}"]
    errors: list[str] = []
    expected = schema.get("type")
    type_matches = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }
    if expected in type_matches and not type_matches[expected]:
        return [f"invalid_type:{path}:{expected}"]

    if isinstance(value, dict):
        required = schema.get("required", [])
        for name in required:
            if name not in value:
                errors.append(f"missing_required_input:{path}.{name}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    errors.append(f"unknown_input:{path}.{name}")
        for name, property_schema in properties.items():
            if name in value:
                errors.extend(validate_schema_value(value[name], property_schema, path=f"{path}.{name}"))

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            errors.extend(validate_schema_value(item, schema["items"], path=f"{path}[{index}]"))

    if isinstance(value, int) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            errors.append(f"below_minimum:{path}:{minimum}")
        if maximum is not None and value > maximum:
            errors.append(f"above_maximum:{path}:{maximum}")
    return errors


class LegacyToolAdapter:
    """Adapt the pre-Phase-2 search/fetch shape without coupling the executor to it."""

    def __init__(self, legacy_tool: Any) -> None:
        self._legacy_tool = legacy_tool
        self.name = legacy_tool.name
        self.risk_level = getattr(legacy_tool, "risk_level", "unknown")
        self.permission_level = getattr(legacy_tool, "permission_level", "unknown")

    def description(self) -> str:
        return f"Legacy adapter for {self.name}"

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        }

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

    def execute(self, inputs: dict[str, Any], context: Any) -> ToolResult:
        query = inputs["query"]
        limit = inputs.get("limit", 5)
        found = self._legacy_tool.search(query, limit=limit)
        errors = list(getattr(found, "errors", []) or [])
        sources = list(getattr(found, "sources", []) or [])
        if not sources:
            return ToolResult.failed(
                "no_sources",
                "Legacy tool returned no sources",
                output={"query": query, "sources": [], "errors": errors},
            )
        fetched: list[dict[str, Any]] = []
        for source in sources[:limit]:
            url = source.get("url") if isinstance(source, dict) else None
            if not url:
                errors.append("source_missing_url")
                continue
            page = self._legacy_tool.fetch(url)
            fetched.append({**source, **page})
        output = {"query": query, "sources": fetched, "errors": errors}
        return ToolResult.success(output, metadata={"compatibility_adapter": True})


REQUIRED_CONTRACT_METHODS = ("description", "input_schema", "output_schema", "execute")


def contract_errors(tool: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(getattr(tool, "name", None), str) or not tool.name.strip():
        errors.append("name_required")
    for method in REQUIRED_CONTRACT_METHODS:
        if not callable(getattr(tool, method, None)):
            errors.append(f"missing_contract_method:{method}")
    if not isinstance(getattr(tool, "risk_level", None), str):
        errors.append("risk_level_required")
    if not isinstance(getattr(tool, "permission_level", None), str):
        errors.append("permission_level_required")
    return errors
