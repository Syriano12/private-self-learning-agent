from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from private_agent.storage import Store, now


@dataclass
class Skill:
    name: str
    version: int
    description: str
    implementation: str
    confidence: float
    success_rate: float
    enabled: bool
    source_knowledge: list[str]
    tests: list[str]


class SkillRegistry:
    def __init__(self, store: Store) -> None:
        self.store = store

    def create(self, name: str, description: str, implementation: str, source_knowledge: list[str] | None = None) -> Skill:
        skill = Skill(name, 1, description, implementation, 0.5, 0.0, False, source_knowledge or [], [])
        self.store.db.execute("INSERT OR REPLACE INTO skills VALUES (?,?,?,?,?,?,?,?,?,?)", (skill.name, skill.version, skill.description, skill.implementation, skill.confidence, skill.success_rate, int(skill.enabled), json.dumps(skill.source_knowledge), json.dumps(skill.tests), now()))
        self.store.db.commit()
        return skill

    def enable(self, name: str, version: int = 1) -> None:
        self.store.db.execute("UPDATE skills SET enabled=1 WHERE name=? AND version=?", (name, version))
        self.store.db.commit()

    def disable(self, name: str, version: int = 1) -> None:
        self.store.db.execute("UPDATE skills SET enabled=0 WHERE name=? AND version=?", (name, version))
        self.store.db.commit()

    def list_enabled(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.store.db.execute("SELECT * FROM skills WHERE enabled=1 ORDER BY name, version").fetchall()]
