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
            "synthetic evidence",
            0.9,
            verification_status,
            source_reliability,
            1,
            0,
            0.0,
            created_at,
            created_at,
            "",
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

        import_context = json.loads(user_prompt)
        self.context = import_context
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


def test_only_verified_legacy_knowledge_reaches_planner_context(tmp_path) -> None:
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

    # These rows model existing database entries from earlier storage versions.
    _insert_historical_knowledge(
        store,
        item_id="historical-needs-review",
        verification_status="needs_review",
        source_reliability=0.9,
    )
    _insert_historical_knowledge(
        store,
        item_id="historical-unverified",
        verification_status="unverified",
        source_reliability=0.9,
    )
    _insert_historical_knowledge(
        store,
        item_id="historical-verified-zero-reliability",
        verification_status="verified",
        source_reliability=0.0,
    )
    _insert_historical_knowledge(
        store,
        item_id="historical-verified",
        verification_status="verified",
        source_reliability=0.6,
    )
    store.db.commit()

    relevant = store.search_knowledge("legacy planning marker")
    assert [item["id"] for item in relevant] == ["historical-verified"]

    provider = CapturingProvider()
    tools = ToolRegistry()
    tools.register(WebResearchTool())
    plan = Planner(provider).create("legacy planning marker", tools, relevant)

    assert plan.steps[0].tool == "web_research"
    assert provider.context["relevant_memory"] == relevant
    assert provider.context["relevant_memory"][0]["verification_status"] == "verified"
    store.close()
