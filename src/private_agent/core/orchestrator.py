from __future__ import annotations

import uuid
from typing import Any

from private_agent.core.execution import ExecutionContext, GenericExecutor, StepResult
from private_agent.core.planner import Planner, TaskPlan
from private_agent.storage import KnowledgeItem, Store
from private_agent.tools.research import ToolRegistry


class Verifier:
    def verify_research(self, result: dict[str, Any]) -> dict[str, Any]:
        sources = result.get("sources", [])
        fetched = [s for s in sources if s.get("text") and s.get("status") == 200]
        errors = result.get("errors", [])
        verified = len(fetched) >= 2 and not (len(fetched) == 0 and errors)
        return {
            "verified": verified,
            "source_count": len(sources),
            "fetched_count": len(fetched),
            "errors": errors,
            "criteria": {
                "multiple_sources": len(fetched) >= 2,
                "evidence_present": bool(fetched),
            },
        }


class Orchestrator:
    def __init__(
        self,
        store: Store,
        tools: ToolRegistry,
        max_attempts: int = 2,
        planner: Planner | None = None,
        executor: GenericExecutor | None = None,
    ) -> None:
        self.store, self.tools, self.max_attempts = store, tools, max_attempts
        self.planner = planner or Planner()
        self.executor = executor or GenericExecutor(tools)
        self.verifier = Verifier()

    def run(self, goal: str) -> dict[str, Any]:
        task_id = str(uuid.uuid4())
        prior = self.store.search_knowledge(goal)
        plan = self.planner.create(goal, self.tools, prior)
        observations: list[dict[str, Any]] = []
        errors: list[str] = []
        attempts = 0
        sources: list[dict[str, Any]] = []
        step_results: list[StepResult] = []

        while attempts < self.max_attempts and not sources:
            attempts += 1
            execution_results = self.executor.execute_plan(
                plan,
                ExecutionContext(task_id=task_id, metadata={"attempt": attempts}),
            )
            step_results = execution_results
            observations.extend(
                {
                    "attempt": attempts,
                    "action": "execute_step",
                    **step_result.to_dict(),
                }
                for step_result in execution_results
            )
            for step_result in execution_results:
                if step_result.error:
                    errors.append(f"{step_result.error.code}: {step_result.error.message}")
                if step_result.status == "success" and isinstance(step_result.output, dict):
                    candidate_sources = step_result.output.get("sources")
                    if isinstance(candidate_sources, list):
                        sources = candidate_sources
            for step, step_result in zip(plan.steps, execution_results):
                step.status = step_result.status
                if isinstance(step_result.output, dict):
                    step.output = step_result.output
                elif step_result.error:
                    step.output = {"error": step_result.error.to_dict()}
            if sources:
                break

        result = {
            "task_id": task_id,
            "goal": goal,
            "sources": sources,
            "errors": errors,
            "prior_knowledge_used": prior,
            "attempts": attempts,
            "step_results": [step_result.to_dict() for step_result in step_results],
        }
        verification = self.verifier.verify_research(result)
        result["verification"] = verification
        if plan.steps:
            plan.steps[-1].status = "completed" if verification["verified"] else plan.steps[-1].status
        status = "completed" if verification["verified"] else "needs_research"
        self.store.save_task(
            task_id,
            goal,
            status,
            {
                "rationale": plan.rationale,
                "steps": [step.__dict__ for step in plan.steps],
            },
            result,
            attempts,
        )
        lessons = [
            "تم استخدام معرفة سابقة" if prior else "لا توجد معرفة سابقة مطابقة",
            f"التحقق: {verification['verified']}",
        ]
        self.store.save_episode(str(uuid.uuid4()), task_id, goal, status, lessons, observations)
        for source in sources:
            if source.get("text"):
                item = KnowledgeItem(
                    id=str(uuid.uuid4()),
                    domain="web-research",
                    concept=source.get("title", goal),
                    knowledge_type="evidence",
                    content=source["text"][:3000],
                    source=source.get("url", ""),
                    evidence=source.get("snippet", ""),
                    confidence=0.7 if verification["verified"] else 0.4,
                    verification_status="verified" if verification["verified"] else "needs_review",
                    source_reliability=0.6,
                )
                self.store.add_knowledge(item)
        return result
