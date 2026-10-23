from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from private_agent.bugbounty.workflow import BugBountyAgent, markdown_report
from private_agent.tools.contracts import ToolResult


class BugBountyAnalysisTool:
    """Run the supported local Bug Bounty workflow as a normal authorized tool.

    The workflow is deliberately deterministic and local-only. It accepts
    caller-supplied scope and artifacts, but the Bug Bounty package remains the
    authority for scope validation, evidence bounds, and finding promotion.
    """

    name = "bugbounty_analysis"
    permission_level = "public_read"
    risk_level = "LOW"
    required_capabilities: list[str] = []
    declared_capabilities: list[str] = []
    network_required = False
    filesystem_required = False
    planned_effect = "Analyze explicitly supplied local Bug Bounty fixtures or artifacts"

    def __init__(self, store: Any) -> None:
        self.store = store

    def description(self) -> str:
        return "Bounded local Bug Bounty analysis; no external target access"

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "scope": {"type": "object"},
                "fixture": {"type": "string"},
                "urls": {"type": "array", "items": {"type": "string"}},
                "openapi": {"type": "object"},
                "source_documents": {"type": "object"},
            },
            "required": ["scope"],
            "additionalProperties": False,
        }

    def output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "run": {"type": "object"},
                "report": {"type": "string"},
            },
            "required": ["run", "report"],
            "additionalProperties": False,
        }

    def execute(self, inputs: dict[str, Any], context: Any) -> ToolResult:
        del context
        try:
            scope = inputs.get("scope")
            if not isinstance(scope, Mapping):
                raise ValueError("scope_object_required")
            scope = dict(scope)
            if "authorization_reference" not in scope and isinstance(scope.get("scope_reference"), str):
                scope["authorization_reference"] = scope.pop("scope_reference")
            urls = inputs.get("urls", [])
            if not isinstance(urls, list) or any(not isinstance(item, str) for item in urls):
                raise ValueError("urls_invalid")
            source_documents = inputs.get("source_documents", {})
            if not isinstance(source_documents, Mapping) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in source_documents.items()
            ):
                raise ValueError("source_documents_invalid")
            openapi = inputs.get("openapi")
            if openapi is not None and not isinstance(openapi, Mapping):
                raise ValueError("openapi_invalid")
            run = BugBountyAgent(store=self.store).run(
                scope,
                fixture_variant=inputs.get("fixture") or None,
                urls=tuple(urls),
                openapi=dict(openapi) if isinstance(openapi, Mapping) else None,
                source_documents=dict(source_documents),
                mode="deterministic",
                persist=True,
            )
            return ToolResult.success({"run": run.to_dict(public=True), "report": markdown_report(run)})
        except (TypeError, ValueError) as exc:
            return ToolResult.failed(
                "bugbounty_invalid_input",
                "Bug Bounty input was rejected by the bounded local workflow",
                details={"error": str(exc)[:160]},
            )
