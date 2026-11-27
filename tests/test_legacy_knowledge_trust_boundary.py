from __future__ import annotations

from datetime import datetime, timezone

from private_agent.core.llm import LLMResponse
from private_agent.core.planner import Planner
from private_agent.storage import KnowledgeItem, Store
from private_agent.tools.research import ToolRegistry, WebResearchTool


def _insert_historical_knowledge(
    store: Store,
    *,
    item_id: str,
    verification_status: str,
    source_reliability: float,
    evidence: str = "synthetic legacy evidence",
    last_verified_at: str = "",
) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    store.db.execute(
        """INSERT INTO knowledge_items (
               id, domain, concept, knowledge_type, content, source, evidence,
               confidence, verification_status, source_reliability, version,
               usage_count, success_rate, created_at, updated_at, last_verified_at,
               related_skills, contradictions, dependencies
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            item_id,
            "test",
            "legacy planning marker",
            "fact",
            f"synthetic {item_id} content",
            "synthetic historical source",
            evidence,
            0.9,
            verification_status,
            source_reliability,
            1,
            0,
            0.0,
            created_at,
            created_at,
            last_verified_at,
            "[]",
            "[]",
            "[]",
        ),
    )


class CapturingProvider:
    def __init__(self) -> None:
        self.context = None

    def generate_json(self, *, system_prompt: str, user_prompt: str, response_schema: dict):
        import json

        self.context = json.loads(user_prompt)
        return LLMResponse(
            {
                "goal": "legacy planning marker",
                "rationale": "use a registered tool",
                "steps": [
                    {
                        "id": "step",
                        "objective": "research",
                        "tool": "web_research",
                        "input": {"query": "legacy planning marker"},
                        "reason": "registered",
                        "depends_on": [],
                    }
                ],
            },
            "{}",
            "fake",
        )


def test_legacy_knowledge_fails_closed_at_planner_boundary(tmp_path) -> None:
    store = Store(tmp_path / "legacy-trust.sqlite3")
    store.add_knowledge(
        KnowledgeItem(
            id="caller-asserted",
            domain="test",
            concept="legacy planning marker",
            knowledge_type="fact",
            content="synthetic caller assertion",
            source="synthetic source",
            evidence="synthetic evidence",
            confidence=1.0,
            verification_status="verified",
            source_reliability=1.0,
            last_verified_at=datetime.now(timezone.utc).isoformat(),
        )
    )
    written_by_current_api = store.db.execute(
        "SELECT verification_status, source_reliability, last_verified_at "
        "FROM knowledge_items WHERE id = 'caller-asserted'"
    ).fetchone()
    assert written_by_current_api["verification_status"] == "needs_review"
    assert written_by_current_api["source_reliability"] == 0.0
    assert written_by_current_api["last_verified_at"] == ""

    # This legacy row reproduces the review challenge: an old verified label
    # with positive reliability, but no verification timestamp or evidence.
    _insert_historical_knowledge(
        store,
        item_id="historical-verified-empty-evidence",
        verification_status="verified",
        source_reliability=0.9,
        evidence="",
    )
    # Even apparently complete legacy fields are not a trusted attestation:
    # the old schema cannot bind them to typed, verified provenance.
    _insert_historical_knowledge(
        store,
        item_id="historical-verified-populated",
        verification_status="verified",
        source_reliability=0.9,
        evidence="synthetic excerpt that appears relevant",
        last_verified_at=datetime.now(timezone.utc).isoformat(),
    )
    _insert_historical_knowledge(
        store,
        item_id="historical-needs-review",
        verification_status="needs_review",
        source_reliability=0.9,
    )
    store.db.commit()

    before = [
        tuple(row)
        for row in store.db.execute(
            "SELECT id, verification_status, evidence, last_verified_at "
            "FROM knowledge_items ORDER BY id"
        ).fetchall()
    ]
    relevant = store.search_knowledge("legacy planning marker")
    assert relevant == []
    after = [
        tuple(row)
        for row in store.db.execute(
            "SELECT id, verification_status, evidence, last_verified_at "
            "FROM knowledge_items ORDER BY id"
        ).fetchall()
    ]
    assert after == before, "retrieval must not mutate or discard persisted knowledge"
    assert len(after) == 4

    provider = CapturingProvider()
    tools = ToolRegistry()
    tools.register(WebResearchTool())
    plan = Planner(provider).create("legacy planning marker", tools, relevant)

    assert plan.steps[0].tool == "web_research"
    assert provider.context["relevant_memory"] == []
    store.close()
