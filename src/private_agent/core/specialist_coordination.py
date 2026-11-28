from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from private_agent.core.specialists import SpecialistRegistry


MAX_SPECIALIST_EVIDENCE_BYTES = 8192


@dataclass(frozen=True)
class SpecialistToolExecutionContext:
    """Non-authoritative, minimized context exposed to a registered specialist tool."""

    task_id: str
    work_order: dict[str, Any]

    @property
    def metadata(self) -> dict[str, Any]:
        return {"specialist_work_order": _json_copy(self.work_order)}


class SpecialistCoordinationError(ValueError):
    """Raised when an internal specialist work order or result is not trustworthy."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise SpecialistCoordinationError("specialist_context_not_json_safe") from exc


@dataclass(frozen=True)
class SpecialistWorkOrder:
    """A server-created, same-task assignment passed to one registered local tool."""

    work_order_id: str
    parent_task_id: str
    plan_version: int
    source_step_id: str
    specialist_id: str
    profile_version: int
    tool_name: str
    assignment_digest: str
    input_digest: str
    input_bindings_digest: str
    capability_digest: str
    tool_contract_digest: str
    output_schema_digest: str
    context_digest: str
    context_limit_bytes: int
    context: dict[str, Any]

    def tool_payload(self) -> dict[str, Any]:
        return {
            "work_order_id": self.work_order_id,
            "parent_task_id": self.parent_task_id,
            "plan_version": self.plan_version,
            "source_step_id": self.source_step_id,
            "specialist_id": self.specialist_id,
            "profile_version": self.profile_version,
            "tool_name": self.tool_name,
            "objective": self.context["objective"],
            "inputs": _json_copy(self.context["inputs"]),
            "dependency_provenance": _json_copy(self.context["dependency_provenance"]),
            "context_digest": self.context_digest,
            "context_limit_bytes": self.context_limit_bytes,
            "execution_mode": "same_task_internal_work_order",
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "work_order_id": self.work_order_id,
            "parent_task_id": self.parent_task_id,
            "plan_version": self.plan_version,
            "source_step_id": self.source_step_id,
            "specialist_id": self.specialist_id,
            "profile_version": self.profile_version,
            "tool_name": self.tool_name,
            "assignment_digest": self.assignment_digest,
            "input_digest": self.input_digest,
            "input_bindings_digest": self.input_bindings_digest,
            "capability_digest": self.capability_digest,
            "tool_contract_digest": self.tool_contract_digest,
            "output_schema_digest": self.output_schema_digest,
            "context_digest": self.context_digest,
            "context_limit_bytes": self.context_limit_bytes,
            "execution_mode": "same_task_internal_work_order",
        }


@dataclass(frozen=True)
class SpecialistResult:
    """Typed result provenance; output is accepted only after an independent verifier passes."""

    schema_version: int
    work_order_id: str
    parent_task_id: str
    plan_version: int
    source_step_id: str
    specialist_id: str
    profile_version: int
    tool_name: str
    assignment_digest: str
    input_digest: str
    input_bindings_digest: str
    capability_digest: str
    tool_contract_digest: str
    output_schema_digest: str
    output_digest: str
    execution_status: str
    verification_status: str
    verified: bool
    verification_digest: str
    verification_evidence: list[dict[str, Any]]
    status: str

    @classmethod
    def from_execution(
        cls,
        provenance: Any,
        *,
        step_result: Any,
        verification: Any,
    ) -> "SpecialistResult":
        if not isinstance(provenance, dict) or not isinstance(verification, dict):
            raise SpecialistCoordinationError("specialist_result_provenance_missing")
        expected = {
            "work_order_id", "parent_task_id", "plan_version", "source_step_id",
            "specialist_id", "profile_version", "tool_name", "assignment_digest",
            "input_digest", "input_bindings_digest", "capability_digest",
            "tool_contract_digest", "output_schema_digest", "context_digest",
            "context_limit_bytes", "execution_mode",
        }
        if set(provenance) != expected or provenance.get("execution_mode") != "same_task_internal_work_order":
            raise SpecialistCoordinationError("specialist_result_provenance_invalid")
        if (
            getattr(step_result, "step_id", None) != provenance["source_step_id"]
            or getattr(step_result, "tool_name", None) != provenance["tool_name"]
        ):
            raise SpecialistCoordinationError("specialist_result_source_mismatch")
        execution_status = getattr(step_result, "status", None)
        verification_status = verification.get("status")
        verified = verification.get("verified") is True
        if execution_status not in {"success", "failed", "blocked"}:
            raise SpecialistCoordinationError("specialist_result_execution_status_invalid")
        if verification_status not in {"VERIFIED", "FAILED", "INSUFFICIENT", "BLOCKED"}:
            raise SpecialistCoordinationError("specialist_result_verification_status_invalid")
        if verified != (verification_status == "VERIFIED"):
            raise SpecialistCoordinationError("specialist_result_verification_boolean_mismatch")
        if execution_status == "success" and verified:
            status = "VERIFIED"
        elif execution_status == "blocked" or verification_status == "BLOCKED":
            status = "BLOCKED"
        elif execution_status == "success":
            status = "UNTRUSTED"
        else:
            status = "FAILED"
        output_digest = _digest(getattr(step_result, "output", None)) if execution_status == "success" else ""
        evidence = verification.get("evidence", [])
        if not isinstance(evidence, list) or any(not isinstance(item, dict) for item in evidence):
            raise SpecialistCoordinationError("specialist_result_evidence_invalid")
        evidence_copy = _json_copy(evidence)
        if len(_canonical_json(evidence_copy).encode("utf-8")) > MAX_SPECIALIST_EVIDENCE_BYTES:
            raise SpecialistCoordinationError("specialist_result_evidence_too_large")
        value = cls(
            schema_version=1,
            work_order_id=str(provenance["work_order_id"]),
            parent_task_id=str(provenance["parent_task_id"]),
            plan_version=provenance["plan_version"],
            source_step_id=str(provenance["source_step_id"]),
            specialist_id=str(provenance["specialist_id"]),
            profile_version=provenance["profile_version"],
            tool_name=str(provenance["tool_name"]),
            assignment_digest=str(provenance["assignment_digest"]),
            input_digest=str(provenance["input_digest"]),
            input_bindings_digest=str(provenance["input_bindings_digest"]),
            capability_digest=str(provenance["capability_digest"]),
            tool_contract_digest=str(provenance["tool_contract_digest"]),
            output_schema_digest=str(provenance["output_schema_digest"]),
            output_digest=output_digest,
            execution_status=execution_status,
            verification_status=verification_status,
            verified=verified,
            verification_digest=_digest(verification),
            verification_evidence=evidence_copy,
            status=status,
        )
        value.validate()
        return value

    @classmethod
    def from_dict(cls, payload: Any) -> "SpecialistResult":
        fields = {
            "schema_version", "work_order_id", "parent_task_id", "plan_version", "source_step_id",
            "specialist_id", "profile_version", "tool_name", "assignment_digest", "input_digest",
            "input_bindings_digest", "capability_digest", "tool_contract_digest", "output_schema_digest",
            "output_digest", "execution_status", "verification_status", "verified", "verification_digest",
            "verification_evidence", "status",
        }
        if not isinstance(payload, dict) or set(payload) != fields:
            raise SpecialistCoordinationError("specialist_result_fields_invalid")
        try:
            value = cls(**payload)
        except (TypeError, ValueError) as exc:
            raise SpecialistCoordinationError("specialist_result_fields_invalid") from exc
        value.validate()
        return value

    def validate(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise SpecialistCoordinationError("specialist_result_schema_version_invalid")
        if (
            not isinstance(self.work_order_id, str) or re.fullmatch(r"[a-f0-9]{32}", self.work_order_id) is None
            or not isinstance(self.parent_task_id, str) or not self.parent_task_id
            or not isinstance(self.source_step_id, str) or not self.source_step_id
            or not isinstance(self.specialist_id, str) or not self.specialist_id
            or not isinstance(self.tool_name, str) or not self.tool_name
        ):
            raise SpecialistCoordinationError("specialist_result_identity_invalid")
        if (
            isinstance(self.plan_version, bool) or not isinstance(self.plan_version, int) or self.plan_version < 1
            or isinstance(self.profile_version, bool) or not isinstance(self.profile_version, int) or self.profile_version < 1
        ):
            raise SpecialistCoordinationError("specialist_result_version_invalid")
        for name in (
            "assignment_digest", "input_digest", "input_bindings_digest", "capability_digest",
            "tool_contract_digest", "output_schema_digest", "verification_digest",
        ):
            if not isinstance(getattr(self, name), str) or re.fullmatch(r"[a-f0-9]{64}", getattr(self, name)) is None:
                raise SpecialistCoordinationError(f"specialist_result_{name}_invalid")
        if self.output_digest and re.fullmatch(r"[a-f0-9]{64}", self.output_digest) is None:
            raise SpecialistCoordinationError("specialist_result_output_digest_invalid")
        if self.execution_status not in {"success", "failed", "blocked"}:
            raise SpecialistCoordinationError("specialist_result_execution_status_invalid")
        if self.verification_status not in {"VERIFIED", "FAILED", "INSUFFICIENT", "BLOCKED"}:
            raise SpecialistCoordinationError("specialist_result_verification_status_invalid")
        if not isinstance(self.verified, bool) or self.verified != (self.verification_status == "VERIFIED"):
            raise SpecialistCoordinationError("specialist_result_verified_invalid")
        expected_status = (
            "VERIFIED" if self.execution_status == "success" and self.verified
            else "BLOCKED" if self.execution_status == "blocked" or self.verification_status == "BLOCKED"
            else "UNTRUSTED" if self.execution_status == "success"
            else "FAILED"
        )
        if self.status != expected_status:
            raise SpecialistCoordinationError("specialist_result_trust_status_invalid")
        if not isinstance(self.verification_evidence, list) or any(not isinstance(item, dict) for item in self.verification_evidence):
            raise SpecialistCoordinationError("specialist_result_evidence_invalid")
        if self.verified and not self.verification_evidence:
            raise SpecialistCoordinationError("specialist_verified_result_evidence_required")
        if len(_canonical_json(self.verification_evidence).encode("utf-8")) > MAX_SPECIALIST_EVIDENCE_BYTES:
            raise SpecialistCoordinationError("specialist_result_evidence_too_large")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "work_order_id": self.work_order_id,
            "parent_task_id": self.parent_task_id,
            "plan_version": self.plan_version,
            "source_step_id": self.source_step_id,
            "specialist_id": self.specialist_id,
            "profile_version": self.profile_version,
            "tool_name": self.tool_name,
            "assignment_digest": self.assignment_digest,
            "input_digest": self.input_digest,
            "input_bindings_digest": self.input_bindings_digest,
            "capability_digest": self.capability_digest,
            "tool_contract_digest": self.tool_contract_digest,
            "output_schema_digest": self.output_schema_digest,
            "output_digest": self.output_digest,
            "execution_status": self.execution_status,
            "verification_status": self.verification_status,
            "verified": self.verified,
            "verification_digest": self.verification_digest,
            "verification_evidence": _json_copy(self.verification_evidence),
            "status": self.status,
        }


class SpecialistCoordinator:
    """Create bounded same-task work orders; dispatch still uses GenericExecutor and server authorization."""

    def __init__(self, registry: SpecialistRegistry) -> None:
        self.registry = registry

    def prepare_work_order(
        self,
        step: Any,
        *,
        task_id: str,
        plan_version: int,
        tool: Any,
        resolved_inputs: dict[str, Any],
        execution_context: Any,
    ) -> SpecialistWorkOrder:
        assignment = getattr(step, "specialist_assignment", None)
        if not isinstance(assignment, dict) or not assignment:
            raise SpecialistCoordinationError("specialist_assignment_missing")
        if not isinstance(task_id, str) or not task_id or task_id != getattr(execution_context, "task_id", None):
            raise SpecialistCoordinationError("specialist_task_binding_mismatch")
        if isinstance(plan_version, bool) or not isinstance(plan_version, int) or plan_version < 1:
            raise SpecialistCoordinationError("specialist_plan_version_invalid")
        if callable(getattr(tool, "_execute_trusted", None)):
            raise SpecialistCoordinationError("specialist_tool_boundary_not_isolated")
        if not self.registry.validate_assignment(
            assignment,
            tool=tool,
            objective=step.objective,
            inputs=step.input,
            input_bindings=step.input_bindings,
            task_id=task_id,
        ):
            raise SpecialistCoordinationError("specialist_assignment_untrusted")
        profile = self.registry.get_profile(assignment["specialist_id"])
        if profile is None or profile.version != assignment.get("profile_version"):
            raise SpecialistCoordinationError("specialist_profile_unavailable")
        if not isinstance(resolved_inputs, dict):
            raise SpecialistCoordinationError("specialist_resolved_inputs_invalid")

        dependency_provenance: list[dict[str, str]] = []
        for target, binding in sorted(step.input_bindings.items()):
            source_step_id = binding["source_step_id"]
            output_key = binding["output_key"]
            source_result = execution_context.previous_results.get(source_step_id)
            if (
                source_step_id not in execution_context.verified_result_ids
                or source_result is None
                or getattr(source_result, "step_id", None) != source_step_id
                or getattr(source_result, "status", None) != "success"
                or getattr(source_result, "error", None) is not None
                or not isinstance(getattr(source_result, "output", None), dict)
                or output_key not in source_result.output
                or target not in resolved_inputs
                or resolved_inputs[target] != source_result.output[output_key]
            ):
                raise SpecialistCoordinationError("specialist_dependency_provenance_invalid")
            dependency_provenance.append({
                "target": target,
                "source_step_id": source_step_id,
                "output_key": output_key,
                "value_digest": _digest(resolved_inputs[target]),
            })

        context = {
            "specialist_id": profile.specialist_id,
            "profile_version": profile.version,
            "objective": step.objective,
            "tool_name": step.tool,
            "inputs": _json_copy(resolved_inputs),
            "dependency_provenance": dependency_provenance,
        }
        serialized = _canonical_json(context)
        context_bytes = len(serialized.encode("utf-8"))
        if context_bytes > profile.max_context_bytes:
            raise SpecialistCoordinationError("specialist_context_limit_exceeded")
        assignment_digest = _digest(assignment)
        input_digest = _digest(resolved_inputs)
        bindings_digest = _digest(step.input_bindings)
        context_digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        work_order_id = _digest({
            "task_id": task_id,
            "plan_version": plan_version,
            "step_id": step.id,
            "assignment_digest": assignment_digest,
            "context_digest": context_digest,
        })[:32]
        return SpecialistWorkOrder(
            work_order_id=work_order_id,
            parent_task_id=task_id,
            plan_version=plan_version,
            source_step_id=step.id,
            specialist_id=profile.specialist_id,
            profile_version=profile.version,
            tool_name=step.tool,
            assignment_digest=assignment_digest,
            input_digest=input_digest,
            input_bindings_digest=bindings_digest,
            capability_digest=assignment["capability_digest"],
            tool_contract_digest=assignment["tool_contract_digest"],
            output_schema_digest=assignment["output_schema_digest"],
            context_digest=context_digest,
            context_limit_bytes=profile.max_context_bytes,
            context=context,
        )

    @staticmethod
    def result_from_execution(
        provenance: Any,
        *,
        step_result: Any,
        verification: Any,
        expected_task_id: str,
        expected_plan_version: int,
        expected_step_id: str,
        expected_assignment: dict[str, Any],
    ) -> SpecialistResult:
        result = SpecialistResult.from_execution(provenance, step_result=step_result, verification=verification)
        if (
            result.parent_task_id != expected_task_id
            or result.plan_version != expected_plan_version
            or result.source_step_id != expected_step_id
            or result.assignment_digest != _digest(expected_assignment)
            or result.tool_name != getattr(step_result, "tool_name", None)
            or result.output_digest != (_digest(step_result.output) if getattr(step_result, "status", None) == "success" else "")
        ):
            raise SpecialistCoordinationError("specialist_result_provenance_mismatch")
        return result

    @staticmethod
    def record_result(state: Any, result: SpecialistResult) -> None:
        result.validate()
        if result.parent_task_id != getattr(state, "task_id", None):
            raise SpecialistCoordinationError("specialist_result_task_mismatch")
        work_orders = state.context.get("specialist_work_orders", {})
        provenance = work_orders.get(result.work_order_id) if isinstance(work_orders, dict) else None
        if not isinstance(provenance, dict) or not SpecialistCoordinator._result_matches_provenance(result, provenance):
            raise SpecialistCoordinationError("specialist_result_work_order_mismatch")
        records = state.context.setdefault("specialist_results", {})
        if not isinstance(records, dict):
            raise SpecialistCoordinationError("specialist_result_store_invalid")
        payload = result.to_dict()
        existing = records.get(result.work_order_id)
        if existing is not None and existing != payload:
            raise SpecialistCoordinationError("specialist_result_conflict")
        records[result.work_order_id] = payload
        state.touch()

    @staticmethod
    def _validate_work_order(provenance: Any, *, task_id: str) -> dict[str, Any]:
        expected = {
            "work_order_id", "parent_task_id", "plan_version", "source_step_id", "specialist_id",
            "profile_version", "tool_name", "assignment_digest", "input_digest", "input_bindings_digest",
            "capability_digest", "tool_contract_digest", "output_schema_digest", "context_digest",
            "context_limit_bytes", "execution_mode",
        }
        if not isinstance(provenance, dict) or set(provenance) != expected:
            raise SpecialistCoordinationError("specialist_work_order_provenance_invalid")
        if (
            not isinstance(provenance["work_order_id"], str)
            or re.fullmatch(r"[a-f0-9]{32}", provenance["work_order_id"]) is None
            or provenance["parent_task_id"] != task_id
            or not isinstance(provenance["source_step_id"], str)
            or not provenance["source_step_id"]
            or not isinstance(provenance["specialist_id"], str)
            or not provenance["specialist_id"]
            or not isinstance(provenance["tool_name"], str)
            or not provenance["tool_name"]
            or provenance["execution_mode"] != "same_task_internal_work_order"
        ):
            raise SpecialistCoordinationError("specialist_work_order_identity_invalid")
        if (
            isinstance(provenance["plan_version"], bool)
            or not isinstance(provenance["plan_version"], int)
            or provenance["plan_version"] < 1
            or isinstance(provenance["profile_version"], bool)
            or not isinstance(provenance["profile_version"], int)
            or provenance["profile_version"] < 1
            or isinstance(provenance["context_limit_bytes"], bool)
            or not isinstance(provenance["context_limit_bytes"], int)
            or not 512 <= provenance["context_limit_bytes"] <= 65536
        ):
            raise SpecialistCoordinationError("specialist_work_order_version_invalid")
        for name in (
            "assignment_digest", "input_digest", "input_bindings_digest", "capability_digest",
            "tool_contract_digest", "output_schema_digest", "context_digest",
        ):
            if not isinstance(provenance[name], str) or re.fullmatch(r"[a-f0-9]{64}", provenance[name]) is None:
                raise SpecialistCoordinationError(f"specialist_work_order_{name}_invalid")
        expected_id = _digest({
            "task_id": task_id,
            "plan_version": provenance["plan_version"],
            "step_id": provenance["source_step_id"],
            "assignment_digest": provenance["assignment_digest"],
            "context_digest": provenance["context_digest"],
        })[:32]
        if provenance["work_order_id"] != expected_id:
            raise SpecialistCoordinationError("specialist_work_order_id_mismatch")
        return _json_copy(provenance)

    @staticmethod
    def _result_matches_provenance(result: SpecialistResult, provenance: dict[str, Any]) -> bool:
        return all(
            getattr(result, field) == provenance.get(field)
            for field in (
                "work_order_id", "parent_task_id", "plan_version", "source_step_id", "specialist_id",
                "profile_version", "tool_name", "assignment_digest", "input_digest",
                "input_bindings_digest", "capability_digest", "tool_contract_digest", "output_schema_digest",
            )
        )

    @staticmethod
    def result_matches_verified_record(
        result: SpecialistResult,
        provenance: dict[str, Any],
        *,
        step_result: Any,
        verification: Any,
    ) -> bool:
        if not isinstance(verification, dict) or not isinstance(getattr(step_result, "metadata", None), dict):
            return False
        return (
            SpecialistCoordinator._result_matches_provenance(result, provenance)
            and step_result.metadata.get("specialist_work_order_provenance") == provenance
            and getattr(step_result, "step_id", None) == result.source_step_id
            and getattr(step_result, "tool_name", None) == result.tool_name
            and getattr(step_result, "status", None) == result.execution_status
            and result.output_digest == (_digest(step_result.output) if result.execution_status == "success" else "")
            and result.verification_status == verification.get("status")
            and result.verified is (verification.get("verified") is True)
            and result.verification_digest == _digest(verification)
        )

    @staticmethod
    def record_work_order(state: Any, provenance: Any) -> None:
        normalized = SpecialistCoordinator._validate_work_order(provenance, task_id=getattr(state, "task_id", ""))
        work_orders = state.context.setdefault("specialist_work_orders", {})
        if not isinstance(work_orders, dict):
            raise SpecialistCoordinationError("specialist_work_order_store_invalid")
        work_order_id = normalized["work_order_id"]
        existing = work_orders.get(work_order_id)
        if existing is not None and existing != normalized:
            raise SpecialistCoordinationError("specialist_work_order_conflict")
        work_orders[work_order_id] = normalized
        state.touch()

    @staticmethod
    def validate_work_order_for_step(
        provenance: Any,
        *,
        task_id: str,
        plan_version: int,
        step: Any,
    ) -> dict[str, Any]:
        normalized = SpecialistCoordinator._validate_work_order(provenance, task_id=task_id)
        assignment = getattr(step, "specialist_assignment", None)
        if (
            normalized["plan_version"] != plan_version
            or normalized["source_step_id"] != getattr(step, "id", None)
            or normalized["tool_name"] != getattr(step, "tool", None)
            or not isinstance(assignment, dict)
            or normalized["assignment_digest"] != _digest(assignment)
            or normalized["input_bindings_digest"] != _digest(getattr(step, "input_bindings", {}))
            or assignment.get("specialist_id") != normalized["specialist_id"]
            or assignment.get("profile_version") != normalized["profile_version"]
            or assignment.get("capability_digest") != normalized["capability_digest"]
            or assignment.get("tool_contract_digest") != normalized["tool_contract_digest"]
            or assignment.get("output_schema_digest") != normalized["output_schema_digest"]
        ):
            raise SpecialistCoordinationError("specialist_work_order_step_mismatch")
        return normalized

    @staticmethod
    def validate_state_results(payload: Any, *, task_id: str, work_orders: Any = None) -> None:
        if work_orders is None:
            work_orders = {}
        if not isinstance(work_orders, dict) or len(work_orders) > 256:
            raise SpecialistCoordinationError("specialist_work_order_store_invalid")
        normalized_work_orders: dict[str, dict[str, Any]] = {}
        for work_order_id, raw_provenance in work_orders.items():
            provenance = SpecialistCoordinator._validate_work_order(raw_provenance, task_id=task_id)
            if work_order_id != provenance["work_order_id"]:
                raise SpecialistCoordinationError("specialist_work_order_key_mismatch")
            normalized_work_orders[work_order_id] = provenance
        if payload is None:
            payload = {}
        if not isinstance(payload, dict) or len(payload) > 256:
            raise SpecialistCoordinationError("specialist_result_store_invalid")
        for work_order_id, raw in payload.items():
            result = SpecialistResult.from_dict(raw)
            provenance = normalized_work_orders.get(work_order_id)
            if (
                work_order_id != result.work_order_id
                or result.parent_task_id != task_id
                or provenance is None
                or not SpecialistCoordinator._result_matches_provenance(result, provenance)
            ):
                raise SpecialistCoordinationError("specialist_result_provenance_mismatch")

    @staticmethod
    def synthesize_verified(
        records: list[tuple[SpecialistResult, dict[str, Any]]],
        *,
        task_id: str,
    ) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        claim_sources: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for result, output in records:
            result.validate()
            if (
                result.parent_task_id != task_id
                or result.status != "VERIFIED"
                or result.verified is not True
                or not isinstance(output, dict)
                or result.output_digest != _digest(output)
            ):
                raise SpecialistCoordinationError("specialist_synthesis_input_untrusted")
            claims = output.get("claims", [])
            if not isinstance(claims, list):
                raise SpecialistCoordinationError("specialist_synthesis_claims_invalid")
            normalized_claims: list[dict[str, Any]] = []
            for claim in claims:
                if (
                    not isinstance(claim, dict)
                    or set(claim) - {"subject", "predicate", "value", "source"}
                    or not isinstance(claim.get("subject"), str)
                    or not claim["subject"].strip()
                    or not isinstance(claim.get("predicate"), str)
                    or not claim["predicate"].strip()
                    or "value" not in claim
                    or ("source" in claim and (not isinstance(claim["source"], str) or not claim["source"].strip()))
                ):
                    raise SpecialistCoordinationError("specialist_synthesis_claim_invalid")
                normalized = _json_copy(claim)
                normalized_claims.append(normalized)
                key = (normalized["subject"], normalized["predicate"])
                claim_sources.setdefault(key, []).append({
                    "work_order_id": result.work_order_id,
                    "source_step_id": result.source_step_id,
                    "specialist_id": result.specialist_id,
                    "value": normalized["value"],
                    "verification_evidence": _json_copy(result.verification_evidence),
                })
            items.append({
                "work_order_id": result.work_order_id,
                "source_step_id": result.source_step_id,
                "specialist_id": result.specialist_id,
                "tool_name": result.tool_name,
                "output_digest": result.output_digest,
                "claims": normalized_claims,
                "verification_evidence": _json_copy(result.verification_evidence),
            })
        conflicts = []
        for (subject, predicate), sources in sorted(claim_sources.items()):
            values = {_canonical_json(source["value"]) for source in sources}
            if len(values) > 1:
                conflicts.append({"subject": subject, "predicate": predicate, "sources": sources})
        return {
            "task_id": task_id,
            "status": "CONFLICT" if conflicts else "VERIFIED" if items else "EMPTY",
            "items": items,
            "conflicts": conflicts,
        }
