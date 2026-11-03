"""Snapshot-bound materialization helpers for the provider-neutral AI contract."""
from __future__ import annotations
from typing import Any

from private_agent.tools.python_ai_contract import (
    _bind_bundle_identity,
    build_ai_analysis_contract,
    build_ai_analysis_request,
    compact_analysis_evidence_registry,
    expand_analysis_evidence_registry,
    validate_ai_hypothesis_response as _validate_response,
)
from private_agent.tools.repository_snapshot import RepositorySnapshot


def materialize_ai_analysis_request(
    analysis_data: dict[str, Any], snapshot_value: RepositorySnapshot | dict[str, Any],
) -> dict[str, Any]:
    snapshot = snapshot_value if isinstance(snapshot_value, RepositorySnapshot) else RepositorySnapshot.from_dict(snapshot_value)
    required = {"python_ast", "call_graph", "data_flow", "analysis_evidence_registry", "ai_analysis_contract"}
    if not isinstance(analysis_data, dict) or not required.issubset(analysis_data):
        raise ValueError("ai_analysis_bundle_incomplete")
    expected_contract = build_ai_analysis_contract(
        snapshot, analysis_data["python_ast"], analysis_data["call_graph"],
        analysis_data["data_flow"], analysis_data["analysis_evidence_registry"],
    )
    if analysis_data["ai_analysis_contract"] != expected_contract:
        raise ValueError("ai_analysis_bundle_digest_mismatch")
    full_registry = expand_analysis_evidence_registry(
        analysis_data["analysis_evidence_registry"], analysis_data["python_ast"], snapshot,
    )
    request = build_ai_analysis_request(
        snapshot, analysis_data["python_ast"], analysis_data["call_graph"],
        analysis_data["data_flow"], full_registry,
    )
    return _bind_bundle_identity(request, expected_contract)


def validate_ai_hypothesis_response(
    response: Any, request: dict[str, Any], analysis_data: dict[str, Any],
    snapshot_value: RepositorySnapshot | dict[str, Any], inspector: Any,
) -> dict[str, Any]:
    snapshot = snapshot_value if isinstance(snapshot_value, RepositorySnapshot) else RepositorySnapshot.from_dict(snapshot_value)
    expected_request = materialize_ai_analysis_request(analysis_data, snapshot)
    if request != expected_request:
        raise ValueError("ai_analysis_request_mismatch")
    full_registry = expand_analysis_evidence_registry(
        analysis_data["analysis_evidence_registry"], analysis_data["python_ast"], snapshot,
    )
    return _validate_response(response, request, full_registry, snapshot, inspector)
