from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from private_agent.storage import KnowledgeItem, Store, now
from private_agent.tools.research import ToolRegistry


@dataclass
class PlanStep:
    id: str
    objective: str
    tool: str
    status: str = "pending"
    output: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskPlan:
    goal: str
    steps: list[PlanStep]
    rationale: str


class Planner:
    """General planner: derives steps from goal signals and available tools, not example-specific workflow."""
    def create(self, goal: str, tools: ToolRegistry, prior_knowledge: list[dict[str, Any]]) -> TaskPlan:
        lower = goal.lower()
        steps: list[PlanStep] = []
        if any(word in lower for word in ("ابحث", "research", "قارن", "compare", "مصادر", "sources")):
            steps.append(PlanStep(str(uuid.uuid4()), "اكتشاف مصادر متعددة للهدف", "web_research"))
            steps.append(PlanStep(str(uuid.uuid4()), "جلب محتوى المصادر المرشحة واستخراج الأدلة", "web_research"))
            steps.append(PlanStep(str(uuid.uuid4()), "التحقق من التغطية والتعارضات وتلخيص النتيجة", "knowledge_verify"))
        else:
            steps.append(PlanStep(str(uuid.uuid4()), "تفكيك الهدف إلى معلومات أو أفعال قابلة للتحقق", "knowledge_verify"))
        if prior_knowledge:
            rationale = f"استُرجعت {len(prior_knowledge)} عناصر معرفة سابقة وستؤثر في ترتيب التنفيذ."
        else:
            rationale = "لا توجد معرفة سابقة مطابقة؛ ستبدأ الخطة بالاستكشاف والتحقق."
        return TaskPlan(goal, steps, rationale)


class Verifier:
    def verify_research(self, result: dict[str, Any]) -> dict[str, Any]:
        sources = result.get("sources", [])
        fetched = [s for s in sources if s.get("text") and s.get("status") == 200]
        errors = result.get("errors", [])
        verified = len(fetched) >= 2 and not (len(fetched) == 0 and errors)
        return {"verified": verified, "source_count": len(sources), "fetched_count": len(fetched), "errors": errors, "criteria": {"multiple_sources": len(fetched) >= 2, "evidence_present": bool(fetched)}}


class Orchestrator:
    def __init__(self, store: Store, tools: ToolRegistry, max_attempts: int = 2) -> None:
        self.store, self.tools, self.max_attempts = store, tools, max_attempts
        self.planner, self.verifier = Planner(), Verifier()

    def run(self, goal: str) -> dict[str, Any]:
        task_id = str(uuid.uuid4())
        prior = self.store.search_knowledge(goal)
        plan = self.planner.create(goal, self.tools, prior)
        observations: list[dict[str, Any]] = []
        errors: list[str] = []
        attempts = 0
        sources: list[dict[str, Any]] = []
        while attempts < self.max_attempts and not sources:
            attempts += 1
            try:
                query = goal
                found = self.tools.get("web_research").search(query, limit=6)
                if found.errors: errors.extend(found.errors)
                sources = found.sources
                observations.append({"attempt": attempts, "action": "search", "source_count": len(sources), "errors": found.errors})
                if not sources:
                    plan.steps[0].status = "failed"
                    plan.steps[0].output = {"errors": found.errors}
                    continue
                fetched = []
                for source in sources[:5]:
                    page = self.tools.get("web_research").fetch(source["url"])
                    fetched.append({**source, **page})
                sources = fetched
                plan.steps[0].status = "completed"
                plan.steps[1].status = "completed"
                break
            except Exception as exc:
                errors.append(f"attempt_{attempts}: {type(exc).__name__}: {exc}")
                observations.append({"attempt": attempts, "exception": errors[-1]})
        result = {"task_id": task_id, "goal": goal, "sources": sources, "errors": errors, "prior_knowledge_used": prior, "attempts": attempts}
        verification = self.verifier.verify_research(result)
        result["verification"] = verification
        plan.steps[-1].status = "completed" if verification["verified"] else "failed"
        status = "completed" if verification["verified"] else "needs_research"
        self.store.save_task(task_id, goal, status, {"rationale": plan.rationale, "steps": [step.__dict__ for step in plan.steps]}, result, attempts)
        lessons = ["تم استخدام معرفة سابقة" if prior else "لا توجد معرفة سابقة مطابقة", f"التحقق: {verification['verified']}"]
        self.store.save_episode(str(uuid.uuid4()), task_id, goal, status, lessons, observations)
        for source in sources:
            if source.get("text"):
                item = KnowledgeItem(id=str(uuid.uuid4()), domain="web-research", concept=source.get("title", goal), knowledge_type="evidence", content=source["text"][:3000], source=source.get("url", ""), evidence=source.get("snippet", ""), confidence=0.7 if verification["verified"] else 0.4, verification_status="verified" if verification["verified"] else "needs_review", source_reliability=0.6)
                self.store.add_knowledge(item)
        return result
