from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from private_agent.bugbounty.local_evidence import normalize_execution_stage_refs
from private_agent.bugbounty.research import ResearchStageRepository
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
                "research_stage_refs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "stage_id": {"type": "string"},
                            "stage_type": {"type": "string"},
                            "result_sha256": {"type": "string"},
                        },
                        "required": ["stage_id", "stage_type", "result_sha256"],
                        "additionalProperties": False,
                    },
                },
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
            provided_stage_refs = normalize_execution_stage_refs(inputs.get("research_stage_refs", []))
        except (TypeError, ValueError) as exc:
            return ToolResult.failed(
                "bugbounty_invalid_input",
                "Bug Bounty input was rejected by the bounded local workflow",
                details={"error": str(exc)[:160]},
                no_side_effects=True,
            )
        metadata = getattr(context, "metadata", None)
        metadata = metadata if isinstance(metadata, Mapping) else {}
        provenance = metadata.get("program_analysis_provenance")
        verified_stage_refs: list[dict[str, str]] = []
        if provenance is None:
            if provided_stage_refs:
                return ToolResult.blocked(
                    "bugbounty_research_lineage_unattested",
                    "Research-stage references require matching server handoff provenance",
                    no_side_effects=True,
                )
        else:
            if not isinstance(provenance, Mapping):
                return ToolResult.blocked(
                    "bugbounty_research_lineage_invalid",
                    "Stored program-analysis provenance is malformed",
                    no_side_effects=True,
                )
            try:
                expected_stage_refs = normalize_execution_stage_refs(provenance.get("research_stage_refs", []))
                analysis_id = provenance.get("analysis_id")
                owner_user_id = provenance.get("scope_attested_by_user_id")
                if (
                    provenance.get("scope_attestation") is not True
                    or provenance.get("execution_mode") != "deterministic_local"
                    or not isinstance(analysis_id, str)
                    or not analysis_id
                    or not isinstance(owner_user_id, str)
                    or not owner_user_id
                    or provided_stage_refs != expected_stage_refs
                ):
                    raise ValueError("bugbounty_research_lineage_binding_mismatch")
                verified_stage_refs = ResearchStageRepository(self.store).verify_execution_sources(
                    analysis_id,
                    owner_user_id=owner_user_id,
                    source_stage_refs=expected_stage_refs,
                )
            except (KeyError, TypeError, ValueError) as exc:
                return ToolResult.blocked(
                    "bugbounty_research_lineage_invalid",
                    "Selected research-stage lineage no longer matches the completed owner-bound results",
                    details={"error": str(exc)[:160]},
                    no_side_effects=True,
                )
        # The deterministic run persists evidence; if it raises after beginning,
        # let the executor preserve EXECUTION_UNKNOWN instead of claiming no effect.
        run = BugBountyAgent(store=self.store).run(
            scope,
            fixture_variant=inputs.get("fixture") or None,
            urls=tuple(urls),
            openapi=dict(openapi) if isinstance(openapi, Mapping) else None,
            source_documents=dict(source_documents),
            research_stage_refs=verified_stage_refs,
            mode="deterministic",
            persist=True,
        )
        return ToolResult.success({"run": run.to_dict(public=True), "report": markdown_report(run)})
