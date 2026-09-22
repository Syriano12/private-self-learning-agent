from __future__ import annotations

import uuid
from typing import Any

from private_agent.core.execution import ExecutionContext, GenericExecutor, StepResult
from private_agent.core.observation import Observation, StepOutcome
from private_agent.core.planner import Planner
from private_agent.core.verification import VerificationResult, VerifierRegistry
from private_agent.storage import KnowledgeItem, Store
from private_agent.tools.research import ToolRegistry


class Orchestrator:
    def __init__(
        self,
        store: Store,
        tools: ToolRegistry,
        max_attempts: int = 2,
        planner: Planner | None = None,
        executor: GenericExecutor | None = None,
        verifiers: VerifierRegistry | None = None,
    ) -> None:
        self.store, self.tools, self.max_attempts = store, tools, max_attempts
        self.planner = planner or Planner()
        self.executor = executor or GenericExecutor(tools)
        self.verifiers = verifiers or VerifierRegistry()

    def run(self, goal: str) -> dict[str, Any]:
        task_id = str(uuid.uuid4())
        prior = self.store.search_knowledge(goal)
        plan = self.planner.create(goal, self.tools, prior)
        self.store.save_event(task_id, "plan_created", {"goal": goal, "steps": [step.__dict__ for step in plan.steps]})
        for step in plan.steps:
            self.store.save_event(task_id, "step_started", {"tool_name": step.tool}, step_id=step.id)

        # Phase 3 deliberately executes once. Verification failure is reported, not retried or replanned.
        execution_results = self.executor.execute_plan(
            plan,
            ExecutionContext(task_id=task_id, metadata={"attempt": 1}),
        )
        outcomes: list[StepOutcome] = []
        observations: list[dict[str, Any]] = []
        errors: list[str] = []

        for step, step_result in zip(plan.steps, execution_results):
            self.store.save_event(task_id, "tool_executed", step_result.to_dict(), step_id=step.id)
            observation = Observation.from_step_result(task_id, step_result)
            self.store.save_observation(observation)
            self.store.save_event(task_id, "observation_created", observation.to_dict(), step_id=step.id)
            self.store.save_event(
                task_id,
                "verification_started",
                {"verifier": self._verifier_for_step(step.tool, step.verifier)},
                step_id=step.id,
            )

            verifier_name = self._verifier_for_step(step.tool, step.verifier)
            criteria = self._criteria_for_step(step.tool, step.success_criteria)
            verification = self.verifiers.verify(verifier_name, observation, criteria)
            self.store.save_verification(task_id, step.id, step.tool, verification)
            self.store.save_event(task_id, "verification_completed", verification.to_dict(), step_id=step.id)
            outcome = StepOutcome(
                task_id=task_id,
                step_id=step.id,
                tool_name=step.tool,
                execution_status=step_result.status,
                verification_status=verification.status,
                observation=observation,
                verification=verification.to_dict(),
            )
            outcomes.append(outcome)
            self.store.save_event(task_id, "step_completed", outcome.to_dict(), step_id=step.id)
            observations.append(outcome.to_dict())

            step.status = self._step_status(step_result, verification)
            step.output = step_result.output if isinstance(step_result.output, dict) else (
                {"error": step_result.error.to_dict()} if step_result.error else {}
            )
            if step_result.error:
                errors.append(f"{step_result.error.code}: {step_result.error.message}")
            if verification.status in {"FAILED", "BLOCKED"}:
                errors.append(f"verification_{verification.status.lower()}: {verification.reason}")

        verification_payload = self._task_verification(outcomes)
        sources = self._collect_sources(outcomes)
        result = {
            "task_id": task_id,
            "goal": goal,
            "sources": sources,
            "errors": errors,
            "prior_knowledge_used": prior,
            "attempts": 1,
            "step_results": [step_result.to_dict() for step_result in execution_results],
            "observations": observations,
            "step_outcomes": [outcome.to_dict() for outcome in outcomes],
            "verification": verification_payload,
        }
        status = "completed" if verification_payload["verified"] else "needs_research"
        self.store.save_task(
            task_id,
            goal,
            status,
            {"rationale": plan.rationale, "steps": [step.__dict__ for step in plan.steps]},
            result,
            1,
        )
        lessons = [
            "تم استخدام معرفة سابقة" if prior else "لا توجد معرفة سابقة مطابقة",
            f"التحقق: {verification_payload['verified']}",
        ]
        self.store.save_episode(str(uuid.uuid4()), task_id, goal, status, lessons, observations)
        self._persist_knowledge(sources, goal, verification_payload)
        return result

    def _verifier_for_step(self, tool_name: str, requested: str) -> str:
        if requested:
            return requested
        return tool_name if tool_name in self.verifiers.available() else "generic"

    @staticmethod
    def _criteria_for_step(tool_name: str, criteria: dict[str, Any]) -> dict[str, Any]:
        if criteria:
            return criteria
        if tool_name == "web_research":
            return {"minimum_sources": 2, "require_non_empty_content": True}
        return {}

    @staticmethod
    def _step_status(step_result: StepResult, verification: VerificationResult) -> str:
        if step_result.status == "blocked":
            return "blocked"
        if step_result.status != "success":
            return "failed"
        return "completed" if verification.status == "VERIFIED" else verification.status.lower()

    @staticmethod
    def _task_verification(outcomes: list[StepOutcome]) -> dict[str, Any]:
        if not outcomes:
            return {
                "status": "INSUFFICIENT",
                "verified": False,
                "reason": "Plan produced no step outcomes",
                "evidence": [],
                "verifier_type": "orchestrator",
                "details": {"code": "no_step_outcomes"},
                "criteria": {},
                "errors": [],
            }
        statuses = [outcome.verification_status for outcome in outcomes]
        if "BLOCKED" in statuses:
            status = "BLOCKED"
        elif "FAILED" in statuses:
            status = "FAILED"
        elif "INSUFFICIENT" in statuses:
            status = "INSUFFICIENT"
        else:
            status = "VERIFIED"
        return {
            "status": status,
            "verified": status == "VERIFIED",
            "reason": f"Step verification statuses: {', '.join(statuses)}",
            "evidence": [{"step_id": outcome.step_id, "status": outcome.verification_status} for outcome in outcomes],
            "verifier_type": "orchestrator",
            "details": {"step_count": len(outcomes), "step_statuses": statuses},
            "criteria": {},
            "errors": [],
        }

    @staticmethod
    def _collect_sources(outcomes: list[StepOutcome]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for outcome in outcomes:
            output = outcome.observation.observed_output
            if isinstance(output, dict) and isinstance(output.get("sources"), list):
                sources.extend(output["sources"])
        return sources

    def _persist_knowledge(self, sources: list[dict[str, Any]], goal: str, verification: dict[str, Any]) -> None:
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
