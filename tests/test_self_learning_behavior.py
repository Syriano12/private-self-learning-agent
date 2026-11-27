from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from private_agent.api import AgentAPIService, create_app
from private_agent.auth import AuthManager
from private_agent.core.orchestrator import Orchestrator
from private_agent.core.planner import Planner
from private_agent.storage import KnowledgeItem, Store
from private_agent.tools.research import ToolRegistry
from private_agent.worker import BackgroundWorker


OWNER_PASSWORD = "self-learning-owner-password-123"


class FakeResearch:
    name = "web_research"
    risk_level = "low"
    permission_level = "public_read"

    def search(self, query: str, limit: int = 5):
        from private_agent.tools.research import ResearchResult
        return ResearchResult(query, [{"title": "Source A", "url": "https://example.com/a", "snippet": "evidence"}, {"title": "Source B", "url": "https://example.com/b", "snippet": "evidence"}], [])

    def fetch(self, url: str, max_chars: int = 8000):
        return {"url": url, "title": "Source", "text": "معرفة وكيل بحث محلي قابل للنقل portable Python research agent knowledge", "status": 200}


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


def test_unreviewed_legacy_knowledge_is_not_reused_by_future_tasks(tmp_path: Path) -> None:
    store = Store(tmp_path / "agent.sqlite3")
    registry = ToolRegistry()
    registry.register(FakeResearch())
    planner = RecordingPlanner()
    agent = Orchestrator(store, registry, planner=planner)
    auth = AuthManager(store)
    auth.initialize_owner(auth.owner_email, OWNER_PASSWORD)
    service = AgentAPIService(agent, token=None)

    with TestClient(create_app(service=service), raise_server_exceptions=False) as client:
        login = client.post(
            "/api/v1/auth/login",
            json={"email": auth.owner_email, "password": OWNER_PASSWORD},
        )
        assert login.status_code == 200, login.text
        csrf = str(client.cookies.get("vulnora_csrf"))

        def run_owned_task(goal: str, key: str, worker_id: str) -> dict:
            created = client.post(
                "/api/v1/tasks",
                json={"goal": goal},
                headers={"X-CSRF-Token": csrf, "Idempotency-Key": key},
            )
            assert created.status_code == 201, created.text
            task_id = str(created.json()["task_id"])
            worker_result = BackgroundWorker(agent, worker_id=worker_id).run_once()
            assert worker_result["status"] == "COMPLETED", worker_result
            state = store.get_task_state(task_id)
            assert state is not None
            return state["result"]

        first = run_owned_task(
            "ابحث عن بنية وكيل بحث محلي قابل للنقل",
            "self-learning-owner-first",
            "self-learning-owner-worker-first",
        )
        assert first["verification"]["verified"] is True
        assert store.search_knowledge("وكيل بحث محلي") == []
        assert store.db.execute(
            "SELECT COUNT(*) AS count FROM knowledge_items WHERE verification_status = 'needs_review'"
        ).fetchone()["count"] > 0

        second = run_owned_task(
            "قارن خيارات بناء وكيل بحث محلي قابل للنقل",
            "self-learning-owner-second",
            "self-learning-owner-worker-second",
        )
        assert second["prior_knowledge_used"] == [], "legacy knowledge must not enter planning context"
        assert planner.calls[-1][0] == []
        assert planner.calls[-1][1], "the separate experience-memory path must still retrieve prior work"
        assert second["verification"]["verified"] is True
