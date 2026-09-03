from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class KnowledgeItem:
    id: str
    domain: str
    concept: str
    knowledge_type: str
    content: str
    source: str
    evidence: str
    confidence: float = 0.5
    verification_status: str = "unverified"
    source_reliability: float = 0.5
    version: int = 1
    usage_count: int = 0
    success_rate: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    last_verified_at: str = ""
    related_skills: str = "[]"
    contradictions: str = "[]"
    dependencies: str = "[]"


class Store:
    def __init__(self, path: str | Path = "data/agent.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.init_schema()

    def init_schema(self) -> None:
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
          id TEXT PRIMARY KEY, goal TEXT NOT NULL, status TEXT NOT NULL,
          plan_json TEXT NOT NULL, result_json TEXT NOT NULL, attempts INTEGER NOT NULL,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS episodes (
          id TEXT PRIMARY KEY, task_id TEXT NOT NULL, goal TEXT NOT NULL,
          outcome TEXT NOT NULL, lessons_json TEXT NOT NULL, observations_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS knowledge_items (
          id TEXT PRIMARY KEY, domain TEXT NOT NULL, concept TEXT NOT NULL,
          knowledge_type TEXT NOT NULL, content TEXT NOT NULL, source TEXT NOT NULL,
          evidence TEXT NOT NULL, confidence REAL NOT NULL, verification_status TEXT NOT NULL,
          source_reliability REAL NOT NULL, version INTEGER NOT NULL, usage_count INTEGER NOT NULL,
          success_rate REAL NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
          last_verified_at TEXT NOT NULL, related_skills TEXT NOT NULL,
          contradictions TEXT NOT NULL, dependencies TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS skills (
          name TEXT NOT NULL, version INTEGER NOT NULL, description TEXT NOT NULL,
          implementation TEXT NOT NULL, confidence REAL NOT NULL, success_rate REAL NOT NULL,
          enabled INTEGER NOT NULL, source_knowledge TEXT NOT NULL, tests_json TEXT NOT NULL,
          updated_at TEXT NOT NULL, PRIMARY KEY(name, version)
        );
        """)
        self.db.commit()

    def save_task(self, task_id: str, goal: str, status: str, plan: Any, result: Any, attempts: int) -> None:
        stamp = now()
        self.db.execute("INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?,?,?)",
            (task_id, goal, status, json.dumps(plan), json.dumps(result), attempts, stamp, stamp))
        self.db.commit()

    def save_episode(self, episode_id: str, task_id: str, goal: str, outcome: str, lessons: Any, observations: Any) -> None:
        self.db.execute("INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?)",
            (episode_id, task_id, goal, outcome, json.dumps(lessons), json.dumps(observations), now()))
        self.db.commit()

    def add_knowledge(self, item: KnowledgeItem) -> None:
        if not item.created_at: item.created_at = now()
        item.updated_at = now()
        self.db.execute("INSERT OR REPLACE INTO knowledge_items VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", tuple(asdict(item).values()))
        self.db.commit()

    def search_knowledge(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        terms = [t.lower() for t in query.split() if len(t) > 2]
        rows = self.db.execute("SELECT * FROM knowledge_items ORDER BY confidence DESC, updated_at DESC").fetchall()
        scored = []
        for row in rows:
            text = f"{row['concept']} {row['content']} {row['domain']}".lower()
            score = sum(term in text for term in terms)
            if score: scored.append((score, dict(row)))
        return [item for _, item in sorted(scored, key=lambda x: (x[0], x[1]['confidence']), reverse=True)[:limit]]

    def recent_episodes(self, limit: int = 10) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute("SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()]

    def close(self) -> None:
        self.db.close()
