"""Bounded, review-only program analysis for authorized security work.

Local analysis never fetches a program URL. An explicit public-policy opt-in is
restricted to the allowlisted passive adapter and still never treats public
text as authorization. The module canonicalizes source identity, hashes
bounded policy evidence, and produces proposals that must be reviewed and
submitted through the existing scope and approval boundaries.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse, urlunparse

from .models import ScopeDefinition, safe_json, stable_digest, utc_now, public_reference
from .passive_research import PassiveResearchAdapter, normalize_passive_source_requests

MAX_PROGRAM_URL_LENGTH = 2048
MAX_POLICY_BYTES = 128_000
MAX_POLICY_LINES = 2_000
MAX_ASSETS = 100
MAX_PUBLIC_POLICY_HOSTS = 100
_PUBLIC_POLICY_HOST_RE = re.compile(
    r"(?i)(?<![@a-z0-9-])(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}(?![a-z0-9-])"
)


class ProgramAnalysisStatus(str):
    READY_FOR_SCOPE_REVIEW = "READY_FOR_SCOPE_REVIEW"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


def _public_policy_observations(policy_text: str) -> dict[str, Any]:
    """Derive bounded, explicitly untrusted observations from public text."""
    hosts = sorted({match.lower().rstrip(".") for match in _PUBLIC_POLICY_HOST_RE.findall(policy_text)})
    marker_patterns = {
        "in_scope": r"\b(?:in[ -]scope|scope|eligible)\b",
        "out_of_scope": r"\b(?:out[ -]of[ -]scope|excluded|not[ -]in[ -]scope)\b",
        "methods_or_testing": r"\b(?:method|testing|rate[ -]limit|prohibited|restricted)\b",
    }
    markers = sorted(name for name, pattern in marker_patterns.items() if re.search(pattern, policy_text, flags=re.IGNORECASE))
    return {
        "kind": "PUBLIC_POLICY_OBSERVATIONS",
        "candidate_hosts": hosts[:MAX_PUBLIC_POLICY_HOSTS],
        "scope_markers": markers,
        "authorization_derived": False,
        "scope_proposal_source": "untrusted_public_text",
    }


@dataclass(frozen=True)
class PublicPolicyEvidence:
    source_type: str
    canonical_url: str
    content: str
    content_sha256: str
    content_length: int
    status_code: int
    retrieved_at: str
    observations: dict[str, Any]


class PublicPolicyRetriever:
    """Fetch only the explicitly supported public policy page shape.

    This is a provenance helper, not an authorization provider. The adapter is
    intentionally restricted to the existing ``hackerone_program`` allowlist;
    authenticated APIs and arbitrary program URLs are not accepted.
    """

    def __init__(self, adapter: PassiveResearchAdapter | None = None) -> None:
        self.adapter = adapter or PassiveResearchAdapter()

    def retrieve(self, program_url: str) -> PublicPolicyEvidence:
        normalized = normalize_passive_source_requests(
            {"sources": [{"source_type": "hackerone_program", "url": program_url}]}
        )
        result = self.adapter.collect(normalized, max_items=1, max_output_bytes=32_000)
        retrievals = result.get("retrievals") if isinstance(result, Mapping) else None
        if not isinstance(retrievals, list) or not retrievals or not isinstance(retrievals[0], Mapping):
            raise ValueError("public_policy_unavailable")
        record = retrievals[0]
        status_code = record.get("status_code")
        content = record.get("text_excerpt")
        if type(status_code) is not int or not 200 <= status_code < 300:
            raise ValueError("public_policy_unavailable")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("public_policy_empty")
        digest = record.get("content_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("public_policy_digest_missing")
        retrieved_at = record.get("retrieved_at")
        return PublicPolicyEvidence(
            source_type="hackerone_program",
            canonical_url=str(record.get("source_url", normalized[0]["url"])),
            content=content,
            content_sha256=digest,
            content_length=len(content.encode("utf-8")),
            status_code=status_code,
            retrieved_at=str(retrieved_at or utc_now()),
            observations={
                **_public_policy_observations(content),
                "retrieval": {
                    "status_code": status_code,
                    "content_sha256": digest,
                    "content_length": len(content.encode("utf-8")),
                    "retrieved_at": str(retrieved_at or utc_now()),
                    "adapter": "hackerone_program",
                    "authorization_basis": "public_source_only",
                },
            },
        )


def _bounded_text(value: Any, *, limit: int, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name}_must_be_text")
    value = value.strip()
    if not value:
        raise ValueError(f"{name}_required")
    if len(value) > limit:
        raise ValueError(f"{name}_too_large")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
        raise ValueError(f"{name}_contains_control_character")
    return value


def canonical_program_url(value: str) -> str:
    value = _bounded_text(value, limit=MAX_PROGRAM_URL_LENGTH, name="program_url")
    parsed = urlparse(value)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("program_url_scheme_not_allowed")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("program_url_userinfo_not_allowed")
    if parsed.hostname is None or len(parsed.hostname) > 253:
        raise ValueError("program_url_host_invalid")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("program_url_port_invalid") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ValueError("program_url_port_invalid")
    host = parsed.hostname.lower().rstrip(".")
    netloc = host
    if port is not None and not ((parsed.scheme.lower() == "https" and port == 443) or (parsed.scheme.lower() == "http" and port == 80)):
        netloc = f"{host}:{port}"
    path = parsed.path or "/"
    if "\\" in path or any(segment in {".", ".."} for segment in path.split("/")):
        raise ValueError("program_url_path_invalid")
    return urlunparse((parsed.scheme.lower(), netloc, path.rstrip("/") or "/", "", parsed.query, ""))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def scope_proposal_digest(proposal: Mapping[str, Any] | None) -> str:
    """Digest the exact persisted, public-safe proposal representation."""
    if not isinstance(proposal, Mapping):
        return ""
    canonical = json.dumps(safe_json(dict(proposal)), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_iso(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("policy_expiry_invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("policy_expiry_must_be_timezone_aware")
    return parsed.astimezone(timezone.utc).isoformat()


def _parse_scope_json(policy_text: str) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        payload = json.loads(policy_text)
    except json.JSONDecodeError:
        return None, ["policy_is_not_json_scope"]
    if not isinstance(payload, Mapping):
        return None, ["policy_json_object_required"]
    scope_payload = payload.get("scope", payload)
    if not isinstance(scope_payload, Mapping):
        return None, ["policy_scope_object_required"]
    return dict(scope_payload), []


def _parse_line_policy(policy_text: str) -> tuple[dict[str, Any] | None, list[str]]:
    values: dict[str, list[str]] = {}
    for raw_line in policy_text.splitlines()[:MAX_POLICY_LINES]:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
        values.setdefault(normalized, []).append(value.strip())
    limitations: list[str] = []
    authorization = (values.get("authorization_reference") or values.get("authorization_id") or [""])[0]
    program_id = (values.get("program_id") or values.get("program") or [""])[0]
    expires = (values.get("expires_at") or values.get("expiry") or [""])[0]
    scopes = values.get("in_scope") or values.get("scope") or []
    excludes = values.get("out_of_scope") or values.get("exclude") or values.get("excluded_host") or []
    identities = values.get("permitted_identities") or values.get("identity") or []
    resources = values.get("permitted_resources") or values.get("resource") or []
    methods = tuple(sorted({item.strip().upper() for raw in (values.get("permitted_methods") or ["GET"]) for item in raw.split(",") if item.strip()}))
    assets: list[dict[str, Any]] = []
    for index, raw in enumerate(scopes[:MAX_ASSETS]):
        parts = raw.split()
        if not parts:
            continue
        host = parts[0].lower()
        path = parts[1] if len(parts) > 1 else "/"
        asset_methods = tuple(sorted({item.upper() for item in (parts[2].split(",") if len(parts) > 2 else methods)}))
        assets.append({"asset_id": f"program-asset-{index + 1}", "host": host, "path_prefix": path, "methods": list(asset_methods), "active_allowed": False})
    if not authorization:
        limitations.append("authorization_reference_missing")
    if not program_id:
        limitations.append("program_id_missing")
    if not expires:
        limitations.append("policy_expiry_missing")
    if not assets:
        limitations.append("in_scope_assets_missing")
    if not identities:
        limitations.append("permitted_identities_missing")
    if not resources:
        limitations.append("permitted_resources_missing")
    if limitations:
        return None, limitations
    scope = {
        "program_id": program_id,
        "authorization_reference": authorization,
        "issued_at": (values.get("issued_at") or [utc_now()])[0],
        "expires_at": _parse_iso(expires),
        "assets": assets,
        "excluded_hosts": [item.lower() for item in excludes[:MAX_ASSETS]],
        "permitted_methods": list(methods),
        "prohibited_methods": ["DELETE", "PATCH", "PUT", "POST"],
        "permitted_identities": identities[:100],
        "permitted_resources": resources[:100],
        "active_testing": (values.get("active_testing", ["false"])[0].lower() == "true"),
    }
    return scope, []


@dataclass(frozen=True)
class ProgramSource:
    canonical_url: str
    source_type: str
    retrieved_at: str
    content_sha256: str
    content_length: int
    revision: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_url": self.canonical_url,
            "source_type": self.source_type,
            "retrieved_at": self.retrieved_at,
            "content_sha256": self.content_sha256,
            "content_length": self.content_length,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class ProgramAnalysis:
    analysis_id: str
    program_url: str
    source: ProgramSource
    status: str
    scope_proposal: dict[str, Any] | None
    exclusions: tuple[str, ...]
    ambiguities: tuple[str, ...]
    limitations: tuple[str, ...]
    created_at: str
    observations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, public: bool = True) -> dict[str, Any]:
        scope = safe_json(self.scope_proposal) if self.scope_proposal is not None else None
        if scope is not None and "authorization_reference" in scope:
            scope["authorization_reference"] = public_reference(scope["authorization_reference"])
        return {
            "analysis_id": self.analysis_id,
            "program_url": self.program_url,
            "source": self.source.to_dict(),
            "status": self.status,
            "scope_proposal": scope,
            "proposal_digest": scope_proposal_digest(scope),
            "exclusions": list(self.exclusions),
            "ambiguities": list(self.ambiguities),
            "limitations": list(self.limitations),
            "created_at": self.created_at,
            "observations": safe_json(self.observations),
        }


def analyze_program(
    program_url: str,
    policy_text: str,
    *,
    source_type: str = "user_provided_policy",
    now: datetime | None = None,
    fetched_automatically: bool = False,
    observations: Mapping[str, Any] | None = None,
) -> ProgramAnalysis:
    canonical_url = canonical_program_url(program_url)
    policy = _bounded_text(policy_text, limit=MAX_POLICY_BYTES, name="policy_text") if policy_text else ""
    retrieved_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    source = ProgramSource(
        canonical_url=canonical_url,
        source_type=_bounded_text(source_type, limit=64, name="source_type"),
        retrieved_at=retrieved_at,
        content_sha256=_sha256_text(policy) if policy else "",
        content_length=len(policy.encode("utf-8")),
    )
    if not policy:
        return ProgramAnalysis(
            "", canonical_url, source, ProgramAnalysisStatus.REVIEW_REQUIRED, None, (),
            ("policy_evidence_required",), ("program_url_not_fetched_automatically",), retrieved_at,
            safe_json(observations or {}),
        )
    scope_payload, json_limits = _parse_scope_json(policy)
    if scope_payload is None:
        line_scope, line_limits = _parse_line_policy(policy)
        scope_payload = line_scope
        limitations = list(line_limits if line_scope is not None else [*json_limits, *line_limits])
    else:
        limitations = list(json_limits)
    ambiguities: list[str] = []
    exclusions: tuple[str, ...] = ()
    proposal: dict[str, Any] | None = None
    derived_observations = safe_json(observations or {})
    if source_type == "hackerone_program":
        derived_observations = {
            **_public_policy_observations(policy),
            **(derived_observations if isinstance(derived_observations, Mapping) else {}),
        }
    if scope_payload is not None:
        try:
            scope = ScopeDefinition.from_dict(scope_payload, now=now)
            proposal = scope.to_dict(public=False)
            exclusions = tuple(scope.excluded_hosts)
        except (TypeError, ValueError) as exc:
            limitations.append(f"scope_validation_failed:{str(exc)[:128]}")
    if proposal is None:
        ambiguities.extend(limitations or ["scope_not_compiled"])
    status = ProgramAnalysisStatus.READY_FOR_SCOPE_REVIEW if proposal is not None and not ambiguities else ProgramAnalysisStatus.REVIEW_REQUIRED
    if fetched_automatically:
        limitations.extend(("public_policy_excerpt_bounded", "public_policy_not_authorization"))
    elif "program_url_not_fetched_automatically" not in limitations:
        limitations.append("program_url_not_fetched_automatically")
    return ProgramAnalysis(
        "", canonical_url, source, status, proposal, exclusions,
        tuple(sorted(set(ambiguities))), tuple(sorted(set(limitations))), retrieved_at,
        derived_observations,
    )


class ProgramAnalysisRepository:
    """Owner-scoped persistence for review proposals; never stores raw policy text."""

    def __init__(self, store: Any) -> None:
        if store is None or not hasattr(store, "db") or not hasattr(store, "_db_lock"):
            raise ValueError("program_analysis_store_required")
        self.store = store
        with store._db_lock, store.db:
            store.db.execute(
                """CREATE TABLE IF NOT EXISTS bugbounty_program_analyses (
                    analysis_id TEXT PRIMARY KEY,
                    owner_user_id TEXT NOT NULL,
                    analysis_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )
            store.db.execute("CREATE INDEX IF NOT EXISTS idx_program_analysis_owner ON bugbounty_program_analyses(owner_user_id, created_at)")

    @staticmethod
    def owner_key(owner_user_id: str | None) -> str:
        return owner_user_id or "legacy-owner"

    def save(self, analysis: ProgramAnalysis, *, owner_user_id: str | None) -> ProgramAnalysis:
        owner = self.owner_key(owner_user_id)
        analysis_id = "PROGRAM-" + stable_digest([owner, analysis.program_url, analysis.source.content_sha256])[:16].upper()
        stored = replace(analysis, analysis_id=analysis_id)
        payload = stored.to_dict(public=True)
        with self.store._db_lock, self.store.db:
            self.store.db.execute(
                "INSERT OR REPLACE INTO bugbounty_program_analyses(analysis_id, owner_user_id, analysis_json, created_at) VALUES (?,?,?,?)",
                (analysis_id, owner, json.dumps(payload, ensure_ascii=False, sort_keys=True), stored.created_at),
            )
        return stored

    def get(self, analysis_id: str, *, owner_user_id: str | None, is_owner: bool = False) -> dict[str, Any] | None:
        owner = self.owner_key(owner_user_id)
        with self.store._db_lock:
            if is_owner:
                row = self.store.db.execute("SELECT analysis_json FROM bugbounty_program_analyses WHERE analysis_id = ?", (analysis_id,)).fetchone()
            else:
                row = self.store.db.execute("SELECT analysis_json FROM bugbounty_program_analyses WHERE analysis_id = ? AND owner_user_id = ?", (analysis_id, owner)).fetchone()
        if row is None:
            return None
        payload = json.loads(row["analysis_json"])
        return payload if isinstance(payload, dict) else None

    def list(self, *, owner_user_id: str | None, is_owner: bool = False, limit: int = 50) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            raise ValueError("program_analysis_limit_invalid")
        owner = self.owner_key(owner_user_id)
        with self.store._db_lock:
            if is_owner:
                rows = self.store.db.execute("SELECT analysis_json FROM bugbounty_program_analyses ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            else:
                rows = self.store.db.execute("SELECT analysis_json FROM bugbounty_program_analyses WHERE owner_user_id = ? ORDER BY created_at DESC LIMIT ?", (owner, limit)).fetchall()
        return [json.loads(row["analysis_json"]) for row in rows if isinstance(json.loads(row["analysis_json"]), dict)]
