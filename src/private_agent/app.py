from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from private_agent.core.orchestrator import Orchestrator
from private_agent.storage import Store
from private_agent.tools.research import ToolRegistry, WebResearchTool


def build_orchestrator() -> Orchestrator:
    store = Store(os.getenv("AGENT_DB_PATH", "data/agent.sqlite3"))
    registry = ToolRegistry()
    registry.register(WebResearchTool(timeout=float(os.getenv("AGENT_HTTP_TIMEOUT", "15"))))
    return Orchestrator(store, registry, max_attempts=int(os.getenv("AGENT_MAX_ATTEMPTS", "2")))


class RunRequest(BaseModel):
    goal: str


app = FastAPI(title="Private Self-Learning Agent", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "private-agent"}


@app.get("/tools")
def tools() -> list[dict[str, str]]:
    return build_orchestrator().tools.available()


@app.post("/tasks/run")
def run_task(request: RunRequest) -> dict[str, Any]:
    if not request.goal.strip():
        return {"status": "rejected", "reason": "goal_required"}
    return build_orchestrator().run(request.goal)
