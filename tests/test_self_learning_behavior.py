from __future__ import annotations

from pathlib import Path

from private_agent.core.orchestrator import Orchestrator
from private_agent.storage import KnowledgeItem, Store
from private_agent.tools.research import ToolRegistry


class FakeResearch:
    name = "web_research"
    risk_level = "low"
    permission_level = "public_read"

    def search(self, query: str, limit: int = 5):
        from private_agent.tools.research import ResearchResult
        return ResearchResult(query, [{"title": "Source A", "url": "https://example.com/a", "snippet": "evidence"}, {"title": "Source B", "url": "https://example.com/b", "snippet": "evidence"}], [])

    def fetch(self, url: str, max_chars: int = 8000):
        return {"url": url, "title": "Source", "text": "معرفة وكيل بحث محلي قابل للنقل portable Python research agent knowledge", "status": 200}


def test_SELF_LEARNING_BEHAVIOR_TEST(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.sqlite3")
    registry = ToolRegistry()
    registry.register(FakeResearch())
    agent = Orchestrator(store, registry)

    first = agent.run("ابحث عن بنية وكيل بحث محلي قابل للنقل")
    assert first["verification"]["verified"] is True
    assert store.search_knowledge("وكيل بحث محلي")

    second = agent.run("قارن خيارات بناء وكيل بحث محلي قابل للنقل")
    assert second["prior_knowledge_used"], "later task must retrieve prior knowledge"
    assert "استُرجعت" in agent.planner.create(second["goal"], registry, second["prior_knowledge_used"]).rationale
    assert second["verification"]["verified"] is True
