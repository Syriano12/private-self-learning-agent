from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import private_agent.storage as storage_module
from private_agent.core.knowledge import (
    KnowledgeAttestation,
    KnowledgeAttestor,
    KnowledgeClaim,
    KnowledgeConflict,
    KnowledgeEvidence,
    KnowledgeSource,
    KnowledgeValidationError,
    sha256_hex,
)
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.planner import Planner
from private_agent.storage import KnowledgeItem, Store, UnsupportedSchemaVersion
from private_agent.tools.research import ToolRegistry


KEY = b"0123456789abcdef0123456789abcdef"
WRONG_KEY = b"fedcba9876543210fedcba9876543210"
REVIEWER_ID = "test.knowledge-reviewer"
SOURCE_TEXT = "Public source text: verified planning claims."


def stamp(delta: timedelta = timedelta(0)) -> str:
    return (datetime.now(timezone.utc) + delta).isoformat()


def make_source(
    source_id: str = "source-1",
    content: str = SOURCE_TEXT,
    *,
    source_class: str = "primary",
) -> KnowledgeSource:
    return KnowledgeSource(
        source_id=source_id,
        canonical_uri="https://example.com/docs/agent",
        source_type="web",
        source_class=source_class,
        retrieved_at=stamp(timedelta(seconds=-1)),
        content_sha256=sha256_hex(content),
        revision="rev-1",
        status="ACTIVE",
        metadata={"title": "Agent documentation"},
        content=content,
    )


def make_claim(claim_id: str = "claim-1", *, subject: str = "planning trust") -> KnowledgeClaim:
    return KnowledgeClaim(
        claim_id=claim_id,
        domain="agent",
        subject=subject,
        predicate="requires",
        object_text="verified evidence",
        knowledge_type="fact",
        verification_status="verified",
        status="SUPPORTED",
        confidence=0.9,
        observed_at=stamp(timedelta(seconds=-1)),
        last_verified_at=stamp(timedelta(seconds=-1)),
    )


def make_evidence(
    claim_id: str = "claim-1",
    source_id: str = "source-1",
    *,
    excerpt: str = "verified planning claims.",
    evidence_id: str = "evidence-1",
) -> KnowledgeEvidence:
    return KnowledgeEvidence(
        evidence_id=evidence_id,
        claim_id=claim_id,
        source_id=source_id,
        locator="https://example.com/docs/agent#claims",
        excerpt=excerpt,
        verification_status="verified",
        observed_at=stamp(timedelta(seconds=-1)),
    )


def save_attested(
    store: Store,
    *,
    verified_at: str | None = None,
    subject: str = "planning trust",
) -> tuple[KnowledgeSource, KnowledgeClaim, KnowledgeEvidence, KnowledgeAttestation]:
    src = make_source()
    cl = make_claim(subject=subject)
    ev = make_evidence(claim_id=cl.claim_id, source_id=src.source_id)
    attestor = KnowledgeAttestor(
        KEY, issuer_authorizer=lambda source, claim, evidence: REVIEWER_ID
    )
    attestation = attestor.issue(
        src,
        cl,
        [ev],
        verified_at=verified_at,
    )
    store.save_attested_knowledge_bundle(src, cl, [ev], attestation)
    return src, cl, ev, attestation


class FakeResearch:
    name = "web_research"
    risk_level = "low"
    permission_level = "public_read"

    def search(self, query: str, limit: int = 5):
        from private_agent.tools.research import ResearchResult

        return ResearchResult(
            query,
            [
                {"title": "Source A", "url": "https://example.com/a", "snippet": "evidence"},
                {"title": "Source B", "url": "https://example.com/b", "snippet": "evidence"},
            ],
            [],
        )

    def fetch(self, url: str, max_chars: int = 8000):
        return {"url": url, "title": "Source", "text": SOURCE_TEXT, "status": 200}


class RecordingPlanner(Planner):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[list[dict], list[dict]]] = []

    def create(
        self,
        goal,
        tools,
        prior_knowledge,
        *,
        constraints=None,
        permissions=None,
        retrieved_experiences=None,
        reflection_insights=None,
        learned_strategies=None,
    ):
        experiences = list(retrieved_experiences or [])
        self.calls.append((list(prior_knowledge), experiences))
        return super().create(
            goal,
            tools,
            prior_knowledge,
            constraints=constraints,
            permissions=permissions,
            retrieved_experiences=experiences,
            reflection_insights=reflection_insights,
            learned_strategies=learned_strategies,
        )


def test_schema_v12_migrates_to_attestation_schema_v13(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "schema-v12.sqlite3"
    monkeypatch.setattr(storage_module, "CURRENT_SCHEMA_VERSION", 12)
    v12 = Store(path)
    v12.close()

    monkeypatch.setattr(storage_module, "CURRENT_SCHEMA_VERSION", 13)
    migrated = Store(path)
    versions = [
        int(row["schema_version"])
        for row in migrated.db.execute("SELECT schema_version FROM task_state_schema ORDER BY schema_version")
    ]
    assert versions[-1] == 13
    assert migrated.db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='knowledge_attestations'"
    ).fetchone()
    assert migrated.db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_knowledge_attestations_claim'"
    ).fetchone()
    migrated.close()


def test_schema_v13_missing_attestation_index_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "schema-v13-missing-index.sqlite3"
    store = Store(path)
    store.db.execute("DROP INDEX idx_knowledge_attestations_claim")
    store.db.commit()
    store.close()
    with pytest.raises(UnsupportedSchemaVersion, match="inconsistent_schema_version:13"):
        Store(path)


def test_signed_content_bound_bundle_is_active_and_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "attested.sqlite3"
    store = Store(path, knowledge_attestation_key=KEY)
    src, cl, _, _ = save_attested(store)
    active = store.search_knowledge_claims("planning trust")
    assert len(active) == 1
    assert active[0]["claim_id"] == cl.claim_id
    assert active[0]["source_count"] == 1
    assert active[0]["is_attested"] is True
    assert active[0]["is_active"] is True
    assert active[0]["status"] == "SUPPORTED"
    assert active[0]["verification_status"] == "verified"

    stored_claim = store.get_knowledge_claim(cl.claim_id)
    assert stored_claim["status"] == "PROPOSED"
    assert stored_claim["verification_status"] == "unverified"
    assert store.get_knowledge_source(src.source_id)["provenance_attested"] is False
    store.close()

    reopened = Store(path, knowledge_attestation_key=KEY)
    assert reopened.search_knowledge_claims("planning trust")[0]["is_active"] is True
    reopened.close()


def test_forged_status_without_attestation_and_missing_key_fail_closed(tmp_path: Path) -> None:
    src = make_source()
    cl = make_claim()
    ev = make_evidence()
    store = Store(tmp_path / "forged.sqlite3", knowledge_attestation_key=KEY)
    saved = store.save_knowledge_bundle(src, cl, [ev])
    assert saved["claim"]["status"] == "PROPOSED"
    assert store.search_knowledge_claims("planning trust") == []

    without_key = Store(tmp_path / "without-key.sqlite3")
    with pytest.raises(KnowledgeValidationError, match="knowledge_attestation_key_missing"):
        attestation = KnowledgeAttestor(
            KEY, issuer_authorizer=lambda source, claim, evidence: REVIEWER_ID
        ).issue(src, cl, [ev])
        without_key.save_attested_knowledge_bundle(src, cl, [ev], attestation)
    assert without_key.search_knowledge_claims("planning trust") == []
    store.close()
    without_key.close()


def test_signature_tampering_and_key_mismatch_do_not_activate_knowledge(tmp_path: Path) -> None:
    path = tmp_path / "key-mismatch.sqlite3"
    store = Store(path, knowledge_attestation_key=KEY)
    _, cl, _, attestation = save_attested(store)
    tampered = KnowledgeAttestation.from_dict(
        {**attestation.to_dict(), "verified_by": "forged.reviewer"}
    )
    with pytest.raises(KnowledgeValidationError, match="knowledge_attestation_invalid"):
        store.save_attested_knowledge_bundle(
            make_source(), cl, [make_evidence()], tampered
        )
    store.close()

    wrong_key_store = Store(path, knowledge_attestation_key=WRONG_KEY)
    assert wrong_key_store.search_knowledge_claims("planning trust") == []
    wrong_key_store.close()


def test_evidence_must_be_bound_to_source_content(tmp_path: Path) -> None:
    src = make_source()
    cl = make_claim()
    ev = make_evidence(excerpt="This text is not present in the source.")
    with pytest.raises(KnowledgeValidationError, match="attestation_evidence_not_in_source"):
        KnowledgeAttestor(
            KEY, issuer_authorizer=lambda source, claim, evidence: REVIEWER_ID
        ).issue(src, cl, [ev])


def test_attestor_requires_explicit_authorized_issuer() -> None:
    src = make_source()
    cl = make_claim()
    ev = make_evidence()
    with pytest.raises(KnowledgeValidationError, match="attestation_issuer_unauthorized"):
        KnowledgeAttestor(KEY).issue(src, cl, [ev])

    restricted = KnowledgeAttestor(
        KEY, issuer_authorizer=lambda source, claim, evidence: None
    )
    with pytest.raises(KnowledgeValidationError, match="attestation_issuer_unauthorized"):
        restricted.issue(src, cl, [ev])

    authorized = KnowledgeAttestor(
        KEY, issuer_authorizer=lambda source, claim, evidence: REVIEWER_ID
    )
    attestation = authorized.issue(src, cl, [ev])
    assert authorized.verify(attestation)


def test_attestor_rejects_sources_explicitly_marked_unverified() -> None:
    src = make_source(source_class="unverified")
    cl = make_claim()
    ev = make_evidence()
    attestor = KnowledgeAttestor(
        KEY, issuer_authorizer=lambda source, claim, evidence: REVIEWER_ID
    )
    with pytest.raises(KnowledgeValidationError, match="attestation_source_unverified"):
        attestor.issue(src, cl, [ev])


def test_stale_attestation_and_persisted_tampering_fail_closed(tmp_path: Path) -> None:
    stale_store = Store(tmp_path / "stale.sqlite3", knowledge_attestation_key=KEY)
    save_attested(stale_store, verified_at=stamp(timedelta(days=-31)))
    assert stale_store.search_knowledge_claims("planning trust") == []
    stale = stale_store.search_knowledge_claims("planning trust", include_unverified=True)
    assert stale and stale[0]["is_attested"] is True
    assert stale[0]["is_fresh"] is False
    stale_store.close()

    tampered_store = Store(tmp_path / "tampered.sqlite3", knowledge_attestation_key=KEY)
    _, cl, _, _ = save_attested(tampered_store)
    tampered_store.db.execute(
        "UPDATE knowledge_claims SET confidence = 0.1 WHERE claim_id = ?", (cl.claim_id,)
    )
    tampered_store.db.commit()
    assert tampered_store.search_knowledge_claims("planning trust") == []
    tampered_store.close()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("status", "QUARANTINED"),
        ("verification_status", "verified"),
        ("last_verified_at", stamp(timedelta(seconds=-1))),
    ],
)
def test_persisted_claim_trust_state_tampering_fails_closed(
    tmp_path: Path, column: str, value: str
) -> None:
    store = Store(tmp_path / f"tampered-{column}.sqlite3", knowledge_attestation_key=KEY)
    _, cl, _, _ = save_attested(store)
    store.db.execute(
        f"UPDATE knowledge_claims SET {column} = ? WHERE claim_id = ?",
        (value, cl.claim_id),
    )
    store.db.commit()
    assert store.search_knowledge_claims("planning trust") == []
    store.close()


@pytest.mark.parametrize("tampering", ["delete", "resolve"])
def test_signed_conflict_suppression_survives_mutable_conflict_tampering(
    tmp_path: Path, tampering: str
) -> None:
    path = tmp_path / f"conflict-suppression-{tampering}.sqlite3"
    store = Store(path, knowledge_attestation_key=KEY)
    src, cl, ev, _ = save_attested(store)
    other_claim = store.save_knowledge_claim(
        make_claim("claim-other", subject="conflicting planning trust")
    )
    other_evidence = store.save_knowledge_evidence(
        make_evidence(
            claim_id=other_claim.claim_id,
            evidence_id="evidence-other",
        )
    )
    conflict = KnowledgeConflict(
        conflict_id=f"conflict-{tampering}",
        claim_id=cl.claim_id,
        conflicting_claim_id=other_claim.claim_id,
        conflict_type="direct_negation",
        evidence_ids=[ev.evidence_id, other_evidence.evidence_id],
    )
    store.save_knowledge_conflict(conflict)
    newly_issued = KnowledgeAttestor(
        KEY, issuer_authorizer=lambda source, claim, evidence: REVIEWER_ID
    ).issue(src, cl, [ev])
    saved_during_conflict = store.save_attested_knowledge_bundle(
        src, cl, [ev], newly_issued
    )
    assert conflict.conflict_id in saved_during_conflict["attestation"][
        "conflict_suppression_ids"
    ]
    before_tampering = store.search_knowledge_claims(
        "planning trust", include_unverified=True
    )[0]
    assert conflict.conflict_id in before_tampering["conflict_suppression_ids"]
    assert before_tampering["is_active"] is False

    if tampering == "delete":
        store.db.execute(
            "DELETE FROM knowledge_conflicts WHERE conflict_id = ?",
            (conflict.conflict_id,),
        )
    else:
        store.db.execute(
            "UPDATE knowledge_conflicts SET status = 'RESOLVED' WHERE conflict_id = ?",
            (conflict.conflict_id,),
        )
    store.db.commit()
    store.close()

    reopened = Store(path, knowledge_attestation_key=KEY)
    result = reopened.search_knowledge_claims(
        "planning trust", include_unverified=True
    )[0]
    assert result["conflict_count"] == 0
    assert conflict.conflict_id in result["conflict_suppression_ids"]
    assert result["is_active"] is False
    assert reopened.search_knowledge_claims("planning trust") == []
    reopened.close()


def test_version_1_attestation_verifies_and_conflict_suppression_upgrades_to_v2(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "legacy-v1-attestation.sqlite3", knowledge_attestation_key=KEY)
    _, _, _, current = save_attested(store)
    store.close()

    legacy_payload = current.to_dict()
    legacy_payload["attestation_version"] = 1
    legacy_payload.pop("conflict_suppression_ids")
    legacy_payload.pop("signature")
    legacy_payload["signature"] = hmac.new(
        KEY,
        json.dumps(
            legacy_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    legacy = KnowledgeAttestation.from_dict(legacy_payload)
    attestor = KnowledgeAttestor(KEY)

    assert legacy.attestation_version == 1
    assert attestor.verify(legacy)
    upgraded = attestor.suppress_for_conflict(legacy, "legacy-conflict")
    assert upgraded.attestation_version == 2
    assert upgraded.conflict_suppression_ids == ["legacy-conflict"]
    assert attestor.verify(upgraded)


def test_open_conflict_requires_attestation_key_when_signed_rows_exist(tmp_path: Path) -> None:
    path = tmp_path / "conflict-without-key.sqlite3"
    store = Store(path, knowledge_attestation_key=KEY)
    _, cl, ev, _ = save_attested(store)
    other_claim = store.save_knowledge_claim(
        make_claim("claim-other", subject="conflicting planning trust")
    )
    other_evidence = store.save_knowledge_evidence(
        make_evidence(claim_id=other_claim.claim_id, evidence_id="evidence-other")
    )
    conflict = KnowledgeConflict(
        conflict_id="conflict-without-key",
        claim_id=cl.claim_id,
        conflicting_claim_id=other_claim.claim_id,
        conflict_type="direct_negation",
        evidence_ids=[ev.evidence_id, other_evidence.evidence_id],
    )
    store.close()

    without_key = Store(path)
    with pytest.raises(KnowledgeValidationError, match="knowledge_attestation_key_missing"):
        without_key.save_knowledge_conflict(conflict)
    assert without_key.list_knowledge_conflicts(cl.claim_id) == []
    without_key.close()

    reopened = Store(path, knowledge_attestation_key=KEY)
    assert reopened.search_knowledge_claims("planning trust")[0]["is_active"] is True
    reopened.close()


def test_conflict_suppression_applies_even_if_persisted_claim_row_is_tampered(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "conflict-tampered-claim.sqlite3", knowledge_attestation_key=KEY)
    _, cl, ev, _ = save_attested(store)
    other_claim = store.save_knowledge_claim(
        make_claim("claim-other", subject="conflicting planning trust")
    )
    other_evidence = store.save_knowledge_evidence(
        make_evidence(claim_id=other_claim.claim_id, evidence_id="evidence-other")
    )
    store.db.execute(
        "UPDATE knowledge_claims SET confidence = 0.1 WHERE claim_id = ?", (cl.claim_id,)
    )
    store.db.commit()
    conflict = KnowledgeConflict(
        conflict_id="conflict-tampered-claim",
        claim_id=cl.claim_id,
        conflicting_claim_id=other_claim.claim_id,
        conflict_type="direct_negation",
        evidence_ids=[ev.evidence_id, other_evidence.evidence_id],
    )
    store.save_knowledge_conflict(conflict)
    store.db.execute(
        "UPDATE knowledge_claims SET confidence = ? WHERE claim_id = ?",
        (cl.confidence, cl.claim_id),
    )
    store.db.execute(
        "DELETE FROM knowledge_conflicts WHERE conflict_id = ?", (conflict.conflict_id,)
    )
    store.db.commit()

    result = store.search_knowledge_claims("planning trust", include_unverified=True)[0]
    assert conflict.conflict_id in result["conflict_suppression_ids"]
    assert result["is_active"] is False
    assert store.search_knowledge_claims("planning trust") == []
    store.close()


def test_planner_receives_only_attested_typed_memory_not_legacy_rows(tmp_path: Path) -> None:
    store = Store(tmp_path / "planner-memory.sqlite3", knowledge_attestation_key=KEY)
    _, cl, _, _ = save_attested(store, subject="portable research planning")
    store.add_knowledge(
        KnowledgeItem(
            id="legacy-planner-note",
            domain="agent",
            concept="portable research planning",
            knowledge_type="fact",
            content="Legacy caller-provided memory must not reach planning.",
            source="https://example.com/legacy",
            evidence="unverified",
        )
    )
    registry = ToolRegistry()
    registry.register(FakeResearch())
    planner = RecordingPlanner()
    agent = Orchestrator(store, registry, planner=planner)

    result = agent.run("Compare portable research planning approaches")
    assert len(planner.calls[-1][0]) == 1
    assert planner.calls[-1][0][0]["claim_id"] == cl.claim_id
    assert planner.calls[-1][0][0]["is_attested"] is True
    assert "Legacy caller-provided memory" not in str(planner.calls[-1][0])
    assert result["prior_knowledge_used"][0]["is_active"] is True
    store.close()


def test_malformed_attestation_row_claim_version_fails_closed(tmp_path: Path) -> None:
    store = Store(tmp_path / "malformed-attestation-version.sqlite3", knowledge_attestation_key=KEY)
    save_attested(store)
    store.db.execute(
        "UPDATE knowledge_attestations SET claim_version = ?",
        ("not-an-integer",),
    )
    store.db.commit()
    assert store.search_knowledge_claims("planning trust") == []
    store.close()
