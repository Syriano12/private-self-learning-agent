"""Safe local execution for research-stage control-plane work.

This executor deliberately has no scanner, subprocess, shell, or arbitrary URL
capability. Local evidence is summarized deterministically. Public passive research
is available only through the explicitly allowlisted adapter; active target access,
scanners, and external verification remain unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .local_evidence import build_local_hypotheses, build_local_inventory, build_local_passive_observations, build_local_report, build_local_verification
from .passive_research import PassiveResearchAdapter
from .research import ResearchStage, ResearchStageStatus, ResearchStageType


@dataclass(frozen=True)
class StageExecutionOutcome:
    status: str
    result: dict[str, Any]
    partial_result: dict[str, Any]
    error_code: str = ""
    error_message: str = ""


class ResearchStageExecutor:
    """Execute only deterministic, local stage work."""

    def __init__(self, *, passive_adapter: PassiveResearchAdapter | None = None) -> None:
        self.passive_adapter = passive_adapter or PassiveResearchAdapter()

    def execute(self, stage: ResearchStage, analysis: Mapping[str, Any] | None) -> StageExecutionOutcome:
        if stage.stage_type == ResearchStageType.PROGRAM_ANALYSIS.value:
            if analysis is None:
                return StageExecutionOutcome(
                    ResearchStageStatus.FAILED.value,
                    {},
                    {},
                    "program_analysis_missing",
                    "The persisted program analysis is missing",
                )
            status = str(analysis.get("status", "REVIEW_REQUIRED"))
            scope = analysis.get("scope_proposal")
            if not isinstance(scope, Mapping):
                return StageExecutionOutcome(
                    ResearchStageStatus.PARTIAL.value,
                    {},
                    {"analysis_id": str(analysis.get("analysis_id", "")), "scope_status": status},
                    "scope_review_required",
                    "No executable scope proposal was persisted",
                )
            return StageExecutionOutcome(
                ResearchStageStatus.COMPLETED.value,
                {
                    "analysis_id": str(analysis.get("analysis_id", "")),
                    "scope_status": status,
                    "source_sha256": str((analysis.get("source") or {}).get("content_sha256", "")),
                    "scope_digest": str(scope.get("scope_digest", "")),
                    "review_only": True,
                },
                {},
            )

        if stage.stage_type == ResearchStageType.INVENTORY.value:
            if not stage.input_payload.get("artifacts"):
                return StageExecutionOutcome(
                    ResearchStageStatus.UNAVAILABLE.value,
                    {},
                    {},
                    "external_runtime_unavailable",
                    "Inventory requires bounded local artifacts supplied by the caller",
                )
            return StageExecutionOutcome(
                ResearchStageStatus.COMPLETED.value,
                build_local_inventory(
                    stage.input_payload,
                    max_items=stage.budget.max_items,
                    max_output_bytes=stage.budget.max_output_bytes,
                ),
                {},
            )
        if stage.stage_type == ResearchStageType.PASSIVE_RESEARCH.value:
            if stage.input_payload.get("sources"):
                result = self.passive_adapter.collect(
                    stage.input_payload["sources"],
                    max_items=stage.budget.max_items,
                    max_output_bytes=stage.budget.max_output_bytes,
                )
                successful = sum(1 for item in result["retrievals"] if isinstance(item.get("status_code"), int))
                if successful == len(result["retrievals"]) and successful > 0 and not result["errors"]:
                    return StageExecutionOutcome(ResearchStageStatus.COMPLETED.value, result, {})
                if successful == 0:
                    return StageExecutionOutcome(
                        ResearchStageStatus.UNAVAILABLE.value,
                        result,
                        {},
                        "passive_sources_unavailable",
                        "No allowlisted public passive source was retrievable in this environment",
                    )
                return StageExecutionOutcome(
                    ResearchStageStatus.PARTIAL.value,
                    result,
                    {},
                    "passive_sources_partial" if successful else "passive_sources_unavailable",
                    "One or more public passive sources could not be retrieved completely",
                )
            if not stage.input_payload.get("artifacts"):
                return StageExecutionOutcome(
                    ResearchStageStatus.UNAVAILABLE.value,
                    {},
                    {},
                    "external_runtime_unavailable",
                    "Passive research requires bounded local artifacts when network access is unavailable",
                )
            return StageExecutionOutcome(
                ResearchStageStatus.COMPLETED.value,
                build_local_passive_observations(
                    {"artifacts": stage.input_payload.get("artifacts", [])},
                    max_items=stage.budget.max_items,
                    max_output_bytes=stage.budget.max_output_bytes,
                ),
                {},
            )
        if stage.stage_type == ResearchStageType.HYPOTHESIS_GENERATION.value:
            if not stage.input_payload.get("artifacts"):
                return StageExecutionOutcome(
                    ResearchStageStatus.UNAVAILABLE.value,
                    {},
                    {},
                    "external_runtime_unavailable",
                    "Hypothesis generation requires bounded local evidence when external research is unavailable",
                )
            return StageExecutionOutcome(
                ResearchStageStatus.COMPLETED.value,
                build_local_hypotheses(
                    stage.input_payload,
                    max_items=stage.budget.max_items,
                    max_output_bytes=stage.budget.max_output_bytes,
                ),
                {},
            )
        if stage.stage_type == ResearchStageType.VERIFICATION.value:
            if not stage.input_payload.get("artifacts"):
                return StageExecutionOutcome(
                    ResearchStageStatus.UNAVAILABLE.value,
                    {},
                    {},
                    "external_runtime_unavailable",
                    "Verification requires explicit local fixtures; external active testing is disabled",
                )
            result = build_local_verification(
                stage.input_payload,
                max_items=stage.budget.max_items,
                max_output_bytes=stage.budget.max_output_bytes,
            )
            if not result["cases"]:
                return StageExecutionOutcome(
                    ResearchStageStatus.PARTIAL.value,
                    result,
                    {},
                    "local_fixture_required",
                    "No bounded local verification fixture cases were supplied",
                )
            return StageExecutionOutcome(ResearchStageStatus.COMPLETED.value, result, {},)
        if stage.stage_type == ResearchStageType.REPORT_PREPARATION.value:
            if not stage.input_payload.get("artifacts"):
                return StageExecutionOutcome(
                    ResearchStageStatus.UNAVAILABLE.value,
                    {},
                    {},
                    "external_runtime_unavailable",
                    "Report preparation requires local stage-result artifacts; external collection is disabled",
                )
            result = build_local_report(
                stage.input_payload,
                max_items=stage.budget.max_items,
                max_output_bytes=stage.budget.max_output_bytes,
            )
            if int(result.get("recognized_stage_results", 0)) < 1:
                return StageExecutionOutcome(
                    ResearchStageStatus.PARTIAL.value,
                    result,
                    {},
                    "local_stage_results_required",
                    "No recognized local inventory, hypothesis, passive, or fixture result was supplied",
                )
            return StageExecutionOutcome(ResearchStageStatus.COMPLETED.value, result, {},)

        return StageExecutionOutcome(
            ResearchStageStatus.FAILED.value,
            {},
            {},
            "stage_type_invalid",
            "Research stage type is not executable",
        )
