from __future__ import annotations

from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import private_agent.storage as storage_module
from private_agent.core.knowledge import (
    KnowledgeClaim,
    KnowledgeConflict,
    KnowledgeEvidence,
    KnowledgeSource,
    KnowledgeValidationError,
    sha256_hex,
)
from private_agent.storage import KnowledgeItem, Store, UnsupportedSchemaVersion


def stamp(delta: timedelta = timedelta(0)) -> str:
    return (datetime.now(timezone.utc) + delta).isoformat()


def source(
    source_id: str = "source-1",
    *,
    uri: str = "https://example.com/docs/agent",
    revision: str = "rev-1",
    content: str = "source content",
    retrieved_at: str | None = None,
) -> KnowledgeSource:
    return KnowledgeSource(
        source_id=source_id,
        canonical_uri=uri,
        source_type="web",
        source_class="primary",
        retrieved_at=retrieved_at or stamp(),
        published_at=stamp(timedelta(days=-1)),
        content_sha256=sha256_hex(content),
        revision=revision,
        status="ACTIVE",
        metadata={"title": "Example documentation"},
        content=content,
    )


def claim(
    claim_id: str = "claim-1",
    *,
    subject: str = "agent",
    object_text: str = "uses bounded evidence",
    verification_status: str = "verified",
    status: str = "SUPPORTED",
    last_verified_at: str | None = None,
    valid_until: str = "",
) -> KnowledgeClaim:
    return KnowledgeClaim(
        claim_id=claim_id,
        domain="agent",
        subject=subject,
        predicate="behavior",
        object_text=object_text,
        knowledge_type="fact",
        verification_status=verification_status,
        status=status,
        confidence=0.86,
        observed_at=stamp(),
        last_verified_at=last_verified_at or stamp(),
        valid_until=valid_until,
        metadata={"source": "test"},
    )


def evidence(
    evidence_id: str,
    claim_id: str,
    source_id: str,
    *,
    excerpt: str = "The agent uses bounded evidence.",
    task_id: str = "",
    step_id: str = "",
) -> KnowledgeEvidence:
    return KnowledgeEvidence(
        evidence_id=evidence_id,
        claim_id=claim_id,
        source_id=source_id,
        task_id=task_id,
        step_id=step_id,
        locator="https://example.com/docs/agent#evidence",
        excerpt=excerpt,
        verification_status="verified",
        observed_at=stamp(),
    )


def test_schema_v4_reopens_and_migrates_to_v5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "schema-v4.sqlite3"
    monkeypatch.setattr(storage_module, "CURRENT_SCHEMA_VERSION", 4)
    legacy = Store(path)
    assert [row["schema_version"] for row in legacy.db.execute("SELECT schema_version FROM task_state_schema ORDER BY schema_version")] == [1, 2, 3, 4]
    legacy.close()

    monkeypatch.setattr(storage_module, "CURRENT_SCHEMA_VERSION", 5)
    migrated = Store(path)
    versions = [row["schema_version"] for row in migrated.db.execute("SELECT schema_version FROM task_state_schema ORDER BY schema_version")]
    assert versions == [1, 2, 3, 4, 5]
    for object_type, name in (
        ("table", "knowledge_sources"),
        ("table", "knowledge_claims"),
        ("table", "knowledge_evidence"),
        ("table", "knowledge_conflicts"),
        ("index", "idx_knowledge_sources_identity"),
        ("index", "idx_knowledge_claims_fingerprint"),
        ("index", "idx_knowledge_claims_freshness"),
        ("index", "idx_knowledge_evidence_claim"),
        ("index", "idx_knowledge_conflicts_claim"),
    ):
        assert migrated.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?", (object_type, name)
        ).fetchone()
    migrated.close()

    reopened = Store(path)
    assert reopened.db.execute("SELECT current_version FROM owner_policy_state WHERE singleton = 1").fetchone()
    assert reopened.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='knowledge_claims'").fetchone()
    reopened.close()


def test_schema_v5_missing_object_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "inconsistent-v5.sqlite3"
    store = Store(path)
    store.db.execute("DROP TABLE knowledge_evidence")
    store.db.commit()
    store.close()
    with pytest.raises(UnsupportedSchemaVersion, match="inconsistent_schema_version:5"):
        Store(path)


def test_schema_v5_missing_required_index_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "inconsistent-v5-index.sqlite3"
    store = Store(path)
    store.db.execute("DROP INDEX idx_knowledge_claims_freshness")
    store.db.commit()
    store.close()
    with pytest.raises(UnsupportedSchemaVersion, match="inconsistent_schema_version:5"):
        Store(path)


def test_source_identity_is_canonical_and_deduplicated_but_revision_changes_are_distinct(tmp_path: Path) -> None:
    store = Store(tmp_path / "sources.sqlite3")
    first = store.save_knowledge_source(source("source-a", uri="HTTPS://Example.com/docs/agent#fragment"))
    duplicate = store.save_knowledge_source(source("source-b", uri="https://example.com/docs/agent"))
    changed_revision = store.save_knowledge_source(source("source-c", revision="rev-2"))

    assert first.source_id == duplicate.source_id == "source-a"
    assert changed_revision.source_id == "source-c"
    assert store.db.execute("SELECT COUNT(*) AS count FROM knowledge_sources").fetchone()["count"] == 2
    assert store.get_knowledge_source("source-a")["canonical_uri"] == "https://example.com/docs/agent"


def test_claim_fingerprint_deduplicates_same_claim_without_overwriting_it(tmp_path: Path) -> None:
    store = Store(tmp_path / "claims.sqlite3")
    first = store.save_knowledge_claim(claim("claim-a"))
    duplicate = store.save_knowledge_claim(claim("claim-b"))

    assert first.claim_id == duplicate.claim_id == "claim-a"
    assert store.db.execute("SELECT COUNT(*) AS count FROM knowledge_claims").fetchone()["count"] == 1
    stored = store.get_knowledge_claim("claim-a")
    assert stored["object_text"] == "uses bounded evidence"


def test_claim_refresh_requires_monotonic_version_and_updates_freshness(tmp_path: Path) -> None:
    store = Store(tmp_path / "claim-refresh.sqlite3")
    initial = claim("refreshable", subject="refreshable agent", last_verified_at=stamp(timedelta(days=-2)), status="SUPPORTED")
    store.save_knowledge_bundle(source(), initial, [evidence("refreshable-evidence", initial.claim_id, "source-1")])
    assert store.search_knowledge_claims("refreshable", max_age_seconds=60) == []

    refreshed = claim(
        "refreshable",
        subject="refreshable agent",
        last_verified_at=stamp(),
        status="SUPPORTED",
    )
    refreshed = KnowledgeClaim(**{**refreshed.to_dict(), "version": 2, "claim_fingerprint": ""})
    saved = store.refresh_knowledge_claim(
        refreshed,
        [evidence("refreshable-evidence-v2", refreshed.claim_id, "source-1")],
    )
    assert saved.version == 2
    assert saved.verification_status == "unverified"
    assert saved.status == "PROPOSED"
    assert store.search_knowledge_claims("refreshable", max_age_seconds=60) == []
    with pytest.raises(KnowledgeValidationError, match="claim_version_not_advanced"):
        store.refresh_knowledge_claim(refreshed)


def test_conflicting_claims_and_evidence_are_preserved_independently(tmp_path: Path) -> None:
    store = Store(tmp_path / "conflicts.sqlite3")
    store.save_knowledge_source(source())
    store.save_knowledge_source(source("source-2", uri="https://official.example/agent", revision="rev-9"))
    first = store.save_knowledge_claim(claim("claim-a"))
    second = store.save_knowledge_claim(claim("claim-b", object_text="executes arbitrary host code"))
    store.save_knowledge_evidence(evidence("evidence-a", first.claim_id, "source-1"))
    store.save_knowledge_evidence(evidence("evidence-b", second.claim_id, "source-2", excerpt="The agent executes arbitrary host code."))

    conflict = KnowledgeConflict(
        conflict_id="conflict-1",
        claim_id=first.claim_id,
        conflicting_claim_id=second.claim_id,
        conflict_type="direct_negation",
        status="OPEN",
        detected_at=stamp(),
        evidence_ids=["evidence-a", "evidence-b"],
    )
    saved = store.save_knowledge_conflict(conflict)
    duplicate = store.save_knowledge_conflict(
        KnowledgeConflict(
            conflict_id="conflict-2",
            claim_id=second.claim_id,
            conflicting_claim_id=first.claim_id,
            conflict_type="direct_negation",
            status="OPEN",
            detected_at=stamp(),
            evidence_ids=["evidence-b", "evidence-a"],
        )
    )

    assert saved.conflict_id == duplicate.conflict_id == "conflict-1"
    assert store.db.execute("SELECT COUNT(*) AS count FROM knowledge_claims").fetchone()["count"] == 2
    assert store.db.execute("SELECT COUNT(*) AS count FROM knowledge_evidence").fetchone()["count"] == 2
    assert store.db.execute("SELECT COUNT(*) AS count FROM knowledge_conflicts").fetchone()["count"] == 1
    assert store.list_knowledge_conflicts("claim-a")[0]["conflicting_claim_id"] == "claim-b"


def test_claim_retrieval_excludes_stale_unverified_and_conflicted_claims_by_default(tmp_path: Path) -> None:
    store = Store(tmp_path / "freshness.sqlite3")
    fresh = claim("fresh", subject="fresh agent")
    fresh = store.save_knowledge_bundle(source(), fresh, [evidence("fresh-evidence", fresh.claim_id, "source-1")])["claim"]
    fresh_id = str(fresh["claim_id"])
    expired = store.save_knowledge_claim(
        claim("expired", subject="expired agent", valid_until=stamp(timedelta(days=-1)))
    )
    unverified = store.save_knowledge_claim(
        claim("unverified", subject="unverified agent", verification_status="unverified", status="PROPOSED", last_verified_at="")
    )
    conflicting = store.save_knowledge_claim(claim("conflicted", subject="conflicted agent"))
    conflicting_other = store.save_knowledge_claim(
        claim("conflicted-other", subject="other conflicted agent", object_text="different")
    )
    store.save_knowledge_evidence(evidence("conflict-evidence-a", conflicting.claim_id, "source-1"))
    store.save_knowledge_evidence(evidence("conflict-evidence-b", conflicting_other.claim_id, "source-1"))
    store.save_knowledge_conflict(
        KnowledgeConflict(
            conflict_id="open-conflict",
            claim_id=conflicting.claim_id,
            conflicting_claim_id="conflicted-other",
            conflict_type="direct_negation",
            detected_at=stamp(),
            evidence_ids=["conflict-evidence-a", "conflict-evidence-b"],
        )
    )

    results = store.search_knowledge_claims("agent")
    # Caller-supplied source/claim/evidence fields are advisory until a
    # trusted ingestion boundary exists; no claim is active here.
    assert results == []
    visible_with_flags = store.search_knowledge_claims(
        "agent",
        include_unverified=True,
        as_of=stamp(timedelta(days=2)),
        max_age_seconds=1,
    )
    assert visible_with_flags
    assert all(item["is_active"] is False for item in visible_with_flags)
    assert expired.claim_id not in [item["claim_id"] for item in results]
    assert unverified.claim_id not in [item["claim_id"] for item in results]


def test_bundle_does_not_project_unattested_typed_claim_to_legacy_items(tmp_path: Path) -> None:
    path = tmp_path / "projection.sqlite3"
    store = Store(path)
    src = source()
    cl = claim("projected", subject="projection agent")
    ev = evidence("projection-evidence", cl.claim_id, src.source_id)
    saved = store.save_knowledge_bundle(src, cl, [ev])
    assert saved["claim"]["verification_status"] == "unverified"
    assert store.search_knowledge("projection agent") == []
    store.close()

    reopened = Store(path)
    assert reopened.get_knowledge_source(src.source_id) is not None
    assert reopened.get_knowledge_claim(cl.claim_id) is not None
    assert reopened.list_knowledge_evidence(cl.claim_id)[0]["evidence_id"] == ev.evidence_id
    assert reopened.search_knowledge("projection agent") == []
    reopened.close()


def test_bundle_rebinds_evidence_to_canonical_deduplicated_ids(tmp_path: Path) -> None:
    store = Store(tmp_path / "bundle-dedup.sqlite3")
    first_source = source("source-first")
    first_claim = claim("claim-first", subject="dedup agent")
    store.save_knowledge_bundle(first_source, first_claim, [evidence("evidence-first", first_claim.claim_id, first_source.source_id)])

    duplicate_source = source("source-second")
    duplicate_claim = claim("claim-second", subject="dedup agent")
    saved = store.save_knowledge_bundle(
        duplicate_source,
        duplicate_claim,
        [evidence("evidence-second", duplicate_claim.claim_id, duplicate_source.source_id)],
    )

    assert saved["source"]["source_id"] == first_source.source_id
    assert saved["claim"]["claim_id"] == first_claim.claim_id
    assert saved["evidence"][0]["evidence_id"] == "evidence-first"
    assert store.db.execute("SELECT COUNT(*) AS count FROM knowledge_evidence").fetchone()["count"] == 1


def test_active_retrieval_requires_verified_evidence_from_active_source(tmp_path: Path) -> None:
    store = Store(tmp_path / "evidence-status.sqlite3")
    stale_source = KnowledgeSource(
        source_id="stale-source",
        canonical_uri="https://stale.example/docs",
        source_type="web",
        source_class="primary",
        retrieved_at=stamp(),
        content_sha256=sha256_hex("stale"),
        revision="stale-1",
        status="STALE",
    )
    stale_claim = claim("stale-claim", subject="stale source agent")
    store.save_knowledge_source(stale_source)
    store.save_knowledge_claim(stale_claim)
    store.save_knowledge_evidence(evidence("stale-evidence", stale_claim.claim_id, stale_source.source_id))

    unverified_claim = claim("unverified-evidence-claim", subject="unverified evidence agent")
    store.save_knowledge_source(source())
    store.save_knowledge_claim(unverified_claim)
    unverified_item = evidence("unverified-evidence", unverified_claim.claim_id, "source-1")
    store.save_knowledge_evidence(
        KnowledgeEvidence(**{**unverified_item.to_dict(), "verification_status": "unverified", "evidence_fingerprint": ""})
    )

    assert store.search_knowledge_claims("agent") == []


def test_worker_lease_rejects_cross_task_evidence_provenance(tmp_path: Path) -> None:
    store = Store(tmp_path / "lease-provenance.sqlite3")
    store.save_knowledge_source(source())
    stored_claim = store.save_knowledge_claim(claim("lease-claim"))
    bound = evidence("bound", stored_claim.claim_id, "source-1", task_id="task-a", step_id="step-a")
    cross_task = evidence("cross-task", stored_claim.claim_id, "source-1", task_id="task-b")
    with pytest.raises(KnowledgeValidationError, match="knowledge_task_lease_required"):
        store.save_knowledge_evidence(bound)
    with store.lease_context("task-a", "lease-a", "worker-a"):
        store.save_knowledge_evidence(bound)
        with pytest.raises(KnowledgeValidationError, match="knowledge_task_provenance_mismatch"):
            store.save_knowledge_evidence(cross_task)
    assert [item["evidence_id"] for item in store.list_knowledge_evidence(stored_claim.claim_id)] == ["bound"]


def test_legacy_knowledge_projection_is_bounded_and_sanitized(tmp_path: Path) -> None:
    store = Store(tmp_path / "legacy-safety.sqlite3")
    token = "access_token=" + "legacy-secret-value-12345"
    store.add_knowledge(
        KnowledgeItem(
            id="legacy-safe",
            domain="test",
            concept="legacy",
            knowledge_type="fact",
            content="x" * 10000,
            source="https://example.com/docs?" + token,
            evidence="Bearer " + "legacy-bearer-value-12345",
        )
    )
    row = store.db.execute("SELECT * FROM knowledge_items WHERE id = 'legacy-safe'").fetchone()
    assert len(row["content"]) == 8192
    assert "legacy-secret-value-12345" not in row["source"]
    assert "legacy-bearer-value-12345" not in row["evidence"]


def test_malformed_provenance_and_hashes_are_rejected(tmp_path: Path) -> None:
    store = Store(tmp_path / "invalid.sqlite3")
    with pytest.raises(KnowledgeValidationError, match="source_hash_invalid"):
        store.save_knowledge_source(
            KnowledgeSource(
                source_id="bad-source",
                canonical_uri="https://example.com",
                source_type="web",
                source_class="primary",
                retrieved_at=stamp(),
                content_sha256="not-a-sha256",
            )
        )
    with pytest.raises(KnowledgeValidationError, match="source_uri_credentials_forbidden"):
        store.save_knowledge_source(
            KnowledgeSource(
                source_id="credential-source",
                canonical_uri="https://user:password@example.com/docs",
                source_type="web",
                source_class="primary",
                retrieved_at=stamp(),
                content_sha256="a" * 64,
            )
        )
    with pytest.raises(KnowledgeValidationError, match="source_uri_sensitive_query"):
        store.save_knowledge_source(
            KnowledgeSource(
                source_id="query-credential-source",
                canonical_uri="https://example.com/docs?access_token=" + "hidden-query-value-12345",
                source_type="web",
                source_class="primary",
                retrieved_at=stamp(),
                content_sha256="b" * 64,
            )
        )
    with pytest.raises(KnowledgeValidationError, match="timestamp_timezone_required"):
        store.save_knowledge_source(
            KnowledgeSource(
                source_id="naive-source",
                canonical_uri="https://example.com/docs",
                source_type="web",
                source_class="primary",
                retrieved_at="2026-01-01T00:00:00",
                content_sha256="a" * 64,
            )
        )
    valid = claim("valid")
    with pytest.raises(KnowledgeValidationError, match="claim_fingerprint_mismatch"):
        store.save_knowledge_claim(KnowledgeClaim(**{**valid.to_dict(), "claim_fingerprint": "b" * 64}))
    with pytest.raises(KnowledgeValidationError, match="confidence_invalid"):
        store.save_knowledge_claim(KnowledgeClaim(**{**valid.to_dict(), "claim_id": "nan", "confidence": float("nan"), "claim_fingerprint": ""}))
    store.save_knowledge_source(source())
    store.save_knowledge_claim(valid)
    store.save_knowledge_claim(claim("other", object_text="other claim"))
    with pytest.raises(KnowledgeValidationError, match="evidence_digest_mismatch"):
        store.save_knowledge_evidence(
            KnowledgeEvidence(
                evidence_id="bad-evidence",
                claim_id=valid.claim_id,
                source_id="source-1",
                locator="line:1",
                excerpt="bounded evidence",
                excerpt_digest="c" * 64,
                observed_at=stamp(),
            )
        )
    with pytest.raises(KnowledgeValidationError, match="conflict_self_reference"):
        store.save_knowledge_conflict(
            KnowledgeConflict(
                conflict_id="self-conflict",
                claim_id=valid.claim_id,
                conflicting_claim_id=valid.claim_id,
                conflict_type="direct_negation",
                detected_at=stamp(),
            )
        )
    with pytest.raises(KnowledgeValidationError, match="conflict_evidence_invalid"):
        store.save_knowledge_conflict(
            KnowledgeConflict(
                conflict_id="missing-evidence-conflict",
                claim_id=valid.claim_id,
                conflicting_claim_id="other",
                conflict_type="direct_negation",
                detected_at=stamp(),
                evidence_ids=["missing-evidence"],
            )
        )


def test_source_content_binding_and_future_timestamps_fail_closed() -> None:
    with pytest.raises(KnowledgeValidationError, match="source_content_hash_mismatch"):
        KnowledgeSource(
            source_id="mismatch-source",
            canonical_uri="https://example.com/mismatch",
            source_type="web",
            source_class="primary",
            retrieved_at=stamp(),
            content_sha256=sha256_hex("actual content"),
            content="different content",
        )
    with pytest.raises(KnowledgeValidationError, match="retrieved_at_future"):
        KnowledgeSource(
            source_id="future-source",
            canonical_uri="https://example.com/future",
            source_type="web",
            source_class="primary",
            retrieved_at=stamp(timedelta(days=1)),
            content_sha256=sha256_hex("future content"),
            content="future content",
        )


def test_direct_verified_claim_and_legacy_flags_are_downgraded(tmp_path: Path) -> None:
    store = Store(tmp_path / "knowledge-trust.sqlite3")
    saved = store.save_knowledge_claim(claim("caller-asserted"))
    assert saved.verification_status == "unverified"
    assert saved.status == "PROPOSED"
    assert not saved.last_verified_at

    store.add_knowledge(
        KnowledgeItem(
            id="legacy-asserted",
            domain="test",
            concept="legacy",
            knowledge_type="fact",
            content="caller asserted verified",
            source="https://example.com/legacy",
            evidence="caller evidence",
            verification_status="verified",
            last_verified_at=stamp(),
        )
    )
    row = store.db.execute(
        "SELECT verification_status, source_reliability, last_verified_at FROM knowledge_items WHERE id = 'legacy-asserted'"
    ).fetchone()
    assert row["verification_status"] == "needs_review"
    assert row["source_reliability"] == 0.0
    assert row["last_verified_at"] == ""


def test_open_conflict_requires_verified_bound_evidence(tmp_path: Path) -> None:
    store = Store(tmp_path / "open-conflict-evidence.sqlite3")
    first = store.save_knowledge_claim(claim("open-a"))
    second = store.save_knowledge_claim(claim("open-b", object_text="different"))
    with pytest.raises(KnowledgeValidationError, match="open_conflict_evidence_required"):
        store.save_knowledge_conflict(
            KnowledgeConflict(
                conflict_id="open-without-evidence",
                claim_id=first.claim_id,
                conflicting_claim_id=second.claim_id,
                conflict_type="direct_negation",
                detected_at=stamp(),
            )
        )


def test_evidence_is_bounded_and_sanitized_before_persistence(tmp_path: Path) -> None:
    store = Store(tmp_path / "sanitized.sqlite3")
    store.save_knowledge_source(source())
    stored_claim = store.save_knowledge_claim(claim("sanitized"))
    token = "Bearer " + "secret-token-value-12345"
    api_key = "api_key=" + "hidden-value-12345"
    item = evidence(
        "safe-evidence",
        stored_claim.claim_id,
        "source-1",
        excerpt=f"Public line. {token}; {api_key}",
    )
    store.save_knowledge_evidence(item)
    row = store.list_knowledge_evidence(stored_claim.claim_id)[0]
    assert "secret-token-value-12345" not in row["excerpt"]
    assert "hidden-value-12345" not in row["excerpt"]
    assert "[REDACTED]" in row["excerpt"]
    assert row["excerpt_sha256"] == sha256_hex(row["excerpt"])

    with pytest.raises(KnowledgeValidationError, match="evidence_excerpt_too_large"):
        store.save_knowledge_evidence(
            evidence("too-large", stored_claim.claim_id, "source-1", excerpt="x" * 2049)
        )


def test_claim_locator_and_conflict_note_are_sanitized_before_persistence(tmp_path: Path) -> None:
    store = Store(tmp_path / "claim-sanitized.sqlite3")
    store.save_knowledge_source(source())
    token = "Bearer " + "claim-secret-value-12345"
    api_key = "api_key=" + "claim-hidden-value-12345"
    first = claim("sensitive", subject=f"agent {token}", object_text=f"uses {api_key}")
    second = claim("sensitive-other", subject="other agent", object_text="different")
    store.save_knowledge_claim(first)
    store.save_knowledge_claim(second)
    store.save_knowledge_evidence(
        evidence(
            "sensitive-evidence",
            first.claim_id,
            "source-1",
            excerpt="safe excerpt",
        )
    )
    store.save_knowledge_evidence(evidence("sensitive-evidence-other", second.claim_id, "source-1"))
    conflict = KnowledgeConflict(
        conflict_id="sensitive-conflict",
        claim_id=first.claim_id,
        conflicting_claim_id=second.claim_id,
        conflict_type="direct_negation",
        detected_at=stamp(),
        evidence_ids=["sensitive-evidence", "sensitive-evidence-other"],
        resolution_note=f"review {token}; {api_key}",
    )
    store.save_knowledge_conflict(conflict)
    saved_claim = store.get_knowledge_claim(first.claim_id)
    saved_conflict = store.list_knowledge_conflicts(first.claim_id)[0]
    assert "claim-secret-value-12345" not in saved_claim["subject"]
    assert "claim-hidden-value-12345" not in saved_claim["object_text"]
    assert "claim-secret-value-12345" not in saved_conflict["resolution_note"]
    assert "claim-hidden-value-12345" not in saved_conflict["resolution_note"]


def test_source_and_claim_contracts_are_data_only_and_do_not_touch_security_or_tools(tmp_path: Path) -> None:
    store = Store(tmp_path / "advisory.sqlite3")
    saved = store.save_knowledge_bundle(source(), claim("advisory"), [evidence("advisory-evidence", "advisory", "source-1")])
    assert saved["claim"]["claim_id"] == "advisory"
    assert store.db.execute("SELECT COUNT(*) AS count FROM audit_events").fetchone()["count"] == 0
    assert store.db.execute("SELECT COUNT(*) AS count FROM approval_requests").fetchone()["count"] == 0


def test_unattested_bundle_preserves_existing_legacy_row_without_overwrite(tmp_path: Path) -> None:
    store = Store(tmp_path / "projection-rollback.sqlite3")
    store.add_knowledge(
        KnowledgeItem(
            id="projection-conflict",
            domain="agent",
            concept="agent",
            knowledge_type="fact",
            content="legacy content",
            source="https://example.com/docs/agent",
            evidence="legacy evidence",
        )
    )
    current_source = source()
    current_claim = claim("projection-conflict", object_text="new content")
    current_evidence = evidence("projection-evidence", current_claim.claim_id, current_source.source_id)

    store.save_knowledge_bundle(current_source, current_claim, [current_evidence])

    assert store.get_knowledge_source(current_source.source_id) is not None
    assert store.get_knowledge_claim(current_claim.claim_id) is not None
    assert store.list_knowledge_evidence(current_claim.claim_id)
    legacy = store.db.execute(
        "SELECT content FROM knowledge_items WHERE id = 'projection-conflict'"
    ).fetchone()
    assert legacy["content"] == "legacy content"


def test_knowledge_writes_are_serialized_for_one_store_connection(tmp_path: Path) -> None:
    store = Store(tmp_path / "knowledge-thread-safety.sqlite3")

    def save_duplicate(index: int) -> KnowledgeSource:
        return store.save_knowledge_source(source(source_id=f"source-{index}"))

    with ThreadPoolExecutor(max_workers=8) as executor:
        saved = list(executor.map(save_duplicate, range(8)))

    assert len(saved) == 8
    assert {item.identity_key for item in saved} == {source().identity_key}
    assert store.db.execute("SELECT COUNT(*) AS count FROM knowledge_sources").fetchone()["count"] == 1
