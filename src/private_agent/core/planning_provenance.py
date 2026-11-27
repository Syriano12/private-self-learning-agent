from __future__ import annotations

from typing import Any

UNKNOWN = "UNKNOWN"
PROVIDER_VALIDATED = "PROVIDER_VALIDATED"
DETERMINISTIC_FALLBACK = "DETERMINISTIC_FALLBACK"
DETERMINISTIC_ROUTE = "DETERMINISTIC_ROUTE"
DETERMINISTIC_REPLAN = "DETERMINISTIC_REPLAN"

PLANNING_MODES = frozenset(
    {
        UNKNOWN,
        PROVIDER_VALIDATED,
        DETERMINISTIC_FALLBACK,
        DETERMINISTIC_ROUTE,
        DETERMINISTIC_REPLAN,
    }
)


def normalize_planning_mode(value: Any) -> str:
    """Return a recognized server mode, failing closed for absent/legacy values."""
    return value if isinstance(value, str) and value in PLANNING_MODES else UNKNOWN


def normalize_plan_version(value: Any) -> int:
    """Return a non-negative plan version, using zero when provenance is absent."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def planning_mode_from_plan(plan: Any) -> str:
    """Read provenance only from a plan already persisted by the server."""
    return normalize_planning_mode(plan.get("planning_mode") if isinstance(plan, dict) else None)


def plan_version_from_state(state: Any) -> int:
    """Return a valid persisted plan version, or zero when none is available."""
    if not isinstance(state, dict):
        return 0
    return normalize_plan_version(state.get("plan_version", 0))
