from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any


_ASSIGNMENT_FIELDS = frozenset(
    {
        "assignment_id",
        "specialist_id",
        "profile_version",
        "tool_name",
        "objective_digest",
        "input_digest",
        "effective_capabilities",
        "capability_digest",
        "tool_contract_digest",
        "output_schema_digest",
    }
)
_SAFE_CAPABILITIES = frozenset({"READ_WORKSPACE"})
_SAFE_LOCAL_TOOLS = frozenset(
    {
        "inspect_repository_structure",
        "list_repository_files",
        "read_repository_file",
        "search_repository_code",
        "investigate_repository_bug",
    }
)


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _capabilities(tool: Any, field: str) -> frozenset[str]:
    raw = getattr(tool, field, [])
    if not isinstance(raw, (list, tuple, set, frozenset)) or any(
        not isinstance(item, str) or not item for item in raw
    ):
        raise ValueError("specialist_tool_capabilities_invalid")
    return frozenset(raw)


def _tool_contract(tool: Any) -> dict[str, Any]:
    name = getattr(tool, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("specialist_tool_name_invalid")
    required = _capabilities(tool, "required_capabilities")
    declared_value = getattr(tool, "declared_capabilities", None)
    declared = required if declared_value is None else _capabilities(tool, "declared_capabilities")
    if not required.issubset(declared):
        raise ValueError("specialist_tool_capability_undeclared")
    if (
        name not in _SAFE_LOCAL_TOOLS
        or getattr(tool, "permission_level", "") != "public_read"
        or getattr(tool, "risk_level", "") != "low"
        or getattr(tool, "network_required", False) is not False
        or "READ_WORKSPACE" not in required
        or "READ_WORKSPACE" not in declared
        or not required.issubset(_SAFE_CAPABILITIES)
        or not declared.issubset(_SAFE_CAPABILITIES)
    ):
        raise ValueError("specialist_tool_not_eligible")
    try:
        input_schema = tool.input_schema()
        output_schema = tool.output_schema()
    except Exception as exc:
        raise ValueError("specialist_tool_schema_unavailable") from exc
    if not isinstance(input_schema, dict) or not isinstance(output_schema, dict):
        raise ValueError("specialist_tool_schema_invalid")
    filesystem_required = getattr(tool, "filesystem_required", False)
    if not isinstance(filesystem_required, bool):
        raise ValueError("specialist_tool_filesystem_policy_invalid")
    return {
        "tool_name": name,
        "required_capabilities": sorted(required),
        "declared_capabilities": sorted(declared),
        "permission_level": "public_read",
        "risk_level": "low",
        "network_required": False,
        "filesystem_required": filesystem_required,
        "input_schema_digest": _digest(input_schema),
        "output_schema_digest": _digest(output_schema),
    }


@dataclass(frozen=True)
class SpecialistProfile:
    """A server-owned, versioned boundary for an in-process specialist assignment."""

    specialist_id: str
    version: int
    tool_names: frozenset[str]
    granted_capabilities: frozenset[str]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", self.specialist_id):
            raise ValueError("specialist_id_invalid")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("specialist_profile_version_invalid")
        if not self.tool_names or any(not isinstance(name, str) or not name for name in self.tool_names):
            raise ValueError("specialist_tool_allowlist_invalid")
        if not self.granted_capabilities.issubset(_SAFE_CAPABILITIES):
            raise ValueError("specialist_capabilities_not_allowed")

    @property
    def capability_digest(self) -> str:
        return _digest(sorted(self.granted_capabilities))

    def public_dict(self) -> dict[str, Any]:
        return {
            "specialist_id": self.specialist_id,
            "profile_version": self.version,
            "tool_names": sorted(self.tool_names),
            "granted_capabilities": sorted(self.granted_capabilities),
            "capability_digest": self.capability_digest,
            "execution_mode": "same_task_tool_boundary",
        }


class SpecialistRegistry:
    """Server-owned registry that issues and revalidates non-elevating assignments.

    This first implementation deliberately supports only bounded, local, read-only
    repository tools. It does not create a child task/process, grant credentials,
    enable network access, or bypass the task's normal security/approval boundary.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, SpecialistProfile] = {}

    def register(self, profile: SpecialistProfile) -> None:
        if not isinstance(profile, SpecialistProfile):
            raise TypeError("specialist_profile_required")
        if profile.specialist_id in self._profiles:
            raise ValueError("specialist_profile_already_registered")
        self._profiles[profile.specialist_id] = profile

    def available_for_planning(self, tools: Any) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for profile in self._profiles.values():
            allowed = []
            for name in sorted(profile.tool_names):
                try:
                    tool = tools.get(name)
                    contract = _tool_contract(tool)
                except Exception:
                    continue
                if set(contract["required_capabilities"]).issubset(profile.granted_capabilities) and set(
                    contract["declared_capabilities"]
                ).issubset(profile.granted_capabilities):
                    allowed.append(name)
            if allowed:
                item = profile.public_dict()
                item["tool_names"] = allowed
                result.append(item)
        return result

    def create_assignment(
        self,
        specialist_id: Any,
        *,
        tool: Any,
        objective: str,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(specialist_id, str) or not specialist_id:
            raise ValueError("specialist_id_required")
        profile = self._profiles.get(specialist_id)
        if profile is None:
            raise ValueError("specialist_unknown")
        contract = _tool_contract(tool)
        if contract["tool_name"] not in profile.tool_names:
            raise ValueError("specialist_tool_not_allowed")
        required = set(contract["required_capabilities"])
        declared = set(contract["declared_capabilities"])
        if not required.issubset(profile.granted_capabilities) or not declared.issubset(profile.granted_capabilities):
            raise ValueError("specialist_capability_exceeds_grant")
        if not isinstance(objective, str) or not objective.strip() or not isinstance(inputs, dict):
            raise ValueError("specialist_assignment_context_invalid")
        return {
            "assignment_id": uuid.uuid4().hex,
            "specialist_id": profile.specialist_id,
            "profile_version": profile.version,
            "tool_name": contract["tool_name"],
            "objective_digest": _digest(objective),
            "input_digest": _digest(inputs),
            "effective_capabilities": contract["declared_capabilities"],
            "capability_digest": _digest(contract["declared_capabilities"]),
            "tool_contract_digest": _digest(
                {key: value for key, value in contract.items() if key != "output_schema_digest"}
            ),
            "output_schema_digest": contract["output_schema_digest"],
        }

    def validate_assignment(
        self,
        assignment: Any,
        *,
        tool: Any,
        objective: str,
        inputs: dict[str, Any],
    ) -> bool:
        if not isinstance(assignment, dict) or set(assignment) != _ASSIGNMENT_FIELDS:
            return False
        if any(
            not isinstance(value, str) or not value
            for key, value in assignment.items()
            if key not in {"profile_version", "effective_capabilities"}
        ):
            return False
        effective = assignment.get("effective_capabilities")
        if not isinstance(effective, list) or any(not isinstance(item, str) or not item for item in effective):
            return False
        if effective != sorted(set(effective)):
            return False
        version = assignment.get("profile_version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            return False
        if re.fullmatch(r"[a-f0-9]{32}", assignment["assignment_id"]) is None:
            return False
        try:
            expected = self.create_assignment(
                assignment["specialist_id"], tool=tool, objective=objective, inputs=inputs
            )
        except Exception:
            return False
        # The server-issued ID is stable across persistence/restart; recompute all
        # policy and context bindings while preserving that original ID.
        expected["assignment_id"] = assignment["assignment_id"]
        return assignment == expected

    @classmethod
    def for_local_repository_tools(cls, tools: Any) -> "SpecialistRegistry":
        eligible: set[str] = set()
        for name in sorted(_SAFE_LOCAL_TOOLS):
            try:
                tool = tools.get(name)
                contract = _tool_contract(tool)
            except Exception:
                continue
            if set(contract["required_capabilities"]).issubset(_SAFE_CAPABILITIES) and set(
                contract["declared_capabilities"]
            ).issubset(_SAFE_CAPABILITIES):
                eligible.add(name)
        registry = cls()
        if eligible:
            registry.register(
                SpecialistProfile(
                    specialist_id="local_repository_analyst",
                    version=1,
                    tool_names=frozenset(eligible),
                    granted_capabilities=frozenset(_SAFE_CAPABILITIES),
                )
            )
        return registry
