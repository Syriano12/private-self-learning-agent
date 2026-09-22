from __future__ import annotations

import uuid
import inspect
from typing import Any

from private_agent.core.diagnosis import FailureDiagnoser, FailureDiagnosis
from private_agent.core.execution import ExecutionContext, GenericExecutor, StepResult
from private_agent.core.experience import ExperienceAction, ExperienceRecord
from private_agent.core.memory import ExperienceMemory, ExperienceQuery, SQLiteExperienceMemory
from private_agent.core.observation import Observation, StepOutcome
from private_agent.core.planner import PlanStep, Planner, TaskPlan
from private_agent.core.recovery import RecoveryDecision, RecoveryManager
from private_agent.core.replanning import ReplanError, ReplanRequest, Replanner
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
        diagnoser: FailureDiagnoser | None = None,
        recovery_manager: RecoveryManager | None = None,
        replanner: Replanner | None = None,
        max_recovery_attempts: int | None = None,
        max_attempts_per_step: int = 2,
        approved_permissions: set[str] | None = None,
        experience_memory: ExperienceMemory | None = None,
    ) -> None:
        self.store, self.tools, self.max_attempts = store, tools, max_attempts
        self.approved_permissions = approved_permissions or {"public_read"}
        self.planner = planner or Planner()
        self.executor = executor or GenericExecutor(tools)
        self.verifiers = verifiers or VerifierRegistry()
        self.diagnoser = diagnoser or FailureDiagnoser()
        self.recovery = recovery_manager or RecoveryManager(
            max_recovery_attempts=max_attempts if max_recovery_attempts is None else max_recovery_attempts,
            max_attempts_per_step=max_attempts_per_step,
        )
        self.replanner = replanner or Replanner(getattr(self.planner, "provider", None))
        self.experience_memory = experience_memory or SQLiteExperienceMemory(store)

    def run(self, goal: str) -> dict[str, Any]:
        task_id = str(uuid.uuid4())
        self.recovery.attempts.clear()
        prior = self.store.search_knowledge(goal)
        retrieved = self.experience_memory.retrieve(ExperienceQuery(goal=goal))
        retrieved_payload = [item.to_dict() for item in retrieved]
        initial_plan = self._create_plan(goal, prior, retrieved_payload)
        current_plan = initial_plan
        experience = ExperienceRecord(
            task_id=task_id,
            goal=goal,
            context={"prior_knowledge_used": prior, "retrieved_experiences": retrieved_payload},
            initial_plan=self._plan_payload(initial_plan),
        )
        all_step_results: list[StepResult] = []
        all_outcomes: list[StepOutcome] = []
        latest_outcomes: dict[str, StepOutcome] = {}
        errors: list[str] = []
        diagnoses: list[dict[str, Any]] = []
        rounds = 0
        terminal_status = "FAILED"

        self.store.save_event(task_id, "plan_created", {"goal": goal, "steps": [step.__dict__ for step in current_plan.steps]})
        max_rounds = 1 + self.recovery.max_recovery_attempts
        while rounds < max_rounds:
            rounds += 1
            if rounds > 1:
                self.store.save_event(
                    task_id,
                    "recovery_round_started",
                    {"round": rounds, "plan": self._plan_payload(current_plan)},
                )
            for step in current_plan.steps:
                self.store.save_event(task_id, "step_started", {"tool_name": step.tool, "round": rounds}, step_id=step.id)

            execution_results = self.executor.execute_plan(
                current_plan,
                ExecutionContext(
                    task_id=task_id,
                    approved_permissions=set(self.approved_permissions),
                    metadata={"attempt": rounds},
                ),
            )
            all_step_results.extend(execution_results)
            recovery_requested = False
            latest_outcomes = {}

            for step, step_result in zip(current_plan.steps, execution_results):
                outcome, diagnosis, action = self._observe_verify_diagnose(
                    task_id,
                    step,
                    step_result,
                    rounds,
                    experience,
                )
                all_outcomes.append(outcome)
                latest_outcomes[step.id] = outcome
                if diagnosis:
                    diagnoses.append(diagnosis.to_dict())
                if step_result.error:
                    errors.append(f"{step_result.error.code}: {step_result.error.message}")
                if outcome.verification_status in {"FAILED", "BLOCKED"}:
                    errors.append(f"verification_{outcome.verification_status.lower()}: {outcome.verification['reason']}")

                if outcome.verification_status == "VERIFIED" and outcome.execution_status == "success":
                    step.status = "completed"
                    continue

                if diagnosis is None:
                    diagnosis = FailureDiagnosis(
                        "UNKNOWN_FAILURE",
                        "No diagnosis was produced for an unsuccessful step",
                        failed_step=step.id,
                        tool_name=step.tool,
                        recoverable=False,
                        suggested_recovery="ABORT",
                    )
                    diagnoses.append(diagnosis.to_dict())
                decision = self.recovery.choose(
                    step,
                    diagnosis,
                    available_tools=self.tools.describe(),
                    allow_replan=self.replanner.provider is not None,
                )
                attempt = self.recovery.record(decision, diagnosis, tool_name=step.tool, inputs=step.input)
                self.store.save_event(task_id, "failure_diagnosed", diagnosis.to_dict(), step_id=step.id)
                self.store.save_event(task_id, "recovery_selected", decision.to_dict(), step_id=step.id)
                if action is not None:
                    action.diagnosis = diagnosis.to_dict()
                    action.recovery = decision.to_dict()

                if decision.strategy == "ABORT":
                    self.recovery.update_last(outcome=decision.terminal_status or "FAILED", verification_status=outcome.verification_status)
                    terminal_status = decision.terminal_status or "FAILED"
                    errors.append(decision.reason)
                    recovery_requested = False
                    break

                try:
                    current_plan = self._apply_decision(current_plan, step, decision, diagnosis, experience)
                except ReplanError as exc:
                    self.recovery.update_last(outcome="BLOCKED", verification_status=outcome.verification_status)
                    terminal_status = "BLOCKED"
                    errors.append(str(exc))
                    self.store.save_event(task_id, "replan_failed", {"error": str(exc)}, step_id=step.id)
                    recovery_requested = False
                    break

                attempt.outcome = "replanned"
                recovery_requested = True
                terminal_status = "FAILED"
                break

            if recovery_requested:
                continue
            if latest_outcomes and all(outcome.verification_status == "VERIFIED" for outcome in latest_outcomes.values()):
                terminal_status = "COMPLETED"
            break

        final_outcomes = list(latest_outcomes.values())
        verification_payload = self._task_verification(final_outcomes)
        if verification_payload["verified"]:
            terminal_status = "COMPLETED"
        elif terminal_status == "COMPLETED":
            terminal_status = "FAILED"

        experience.final_outcome = terminal_status
        result = {
            "task_id": task_id,
            "goal": goal,
            "sources": self._collect_sources(final_outcomes),
            "errors": errors,
            "prior_knowledge_used": prior,
            "attempts": rounds,
            "step_results": [step_result.to_dict() for step_result in all_step_results],
            "observations": [outcome.observation.to_dict() for outcome in all_outcomes],
            "step_outcomes": [outcome.to_dict() for outcome in all_outcomes],
            "final_step_outcomes": [outcome.to_dict() for outcome in final_outcomes],
            "verification": verification_payload,
            "diagnoses": diagnoses,
            "recovery_attempts": [attempt.to_dict() for attempt in self.recovery.attempts],
            "experience": experience.to_dict(),
            "final_status": terminal_status,
        }
        storage_status = {
            "COMPLETED": "completed",
            "BLOCKED": "blocked",
            "ABORTED": "aborted",
        }.get(terminal_status, "failed")
        self.store.save_task(
            task_id,
            goal,
            storage_status,
            {"rationale": current_plan.rationale, "steps": [step.__dict__ for step in current_plan.steps]},
            result,
            rounds,
        )
        lessons = [
            "تم استخدام معرفة سابقة" if prior else "لا توجد معرفة سابقة مطابقة",
            f"النتيجة النهائية: {terminal_status}",
            f"عدد محاولات الاسترداد: {len(self.recovery.attempts)}",
        ]
        self.store.save_episode(str(uuid.uuid4()), task_id, goal, storage_status, lessons, result["experience"]["actions"])
        self._persist_knowledge(result["sources"], goal, verification_payload)
        self.experience_memory.store(experience)
        return result

    def _create_plan(
        self,
        goal: str,
        prior: list[dict[str, Any]],
        retrieved: list[dict[str, Any]],
    ) -> TaskPlan:
        parameters = inspect.signature(self.planner.create).parameters
        kwargs: dict[str, Any] = {}
        if "permissions" in parameters:
            kwargs["permissions"] = self._permissions()
        if "retrieved_experiences" in parameters:
            kwargs["retrieved_experiences"] = retrieved
        return self.planner.create(goal, self.tools, prior, **kwargs)

    def _observe_verify_diagnose(
        self,
        task_id: str,
        step: PlanStep,
        step_result: StepResult,
        round_number: int,
        experience: ExperienceRecord,
    ) -> tuple[StepOutcome, FailureDiagnosis | None, ExperienceAction | None]:
        self.store.save_event(task_id, "tool_executed", {"round": round_number, **step_result.to_dict()}, step_id=step.id)
        observation = Observation.from_step_result(task_id, step_result)
        self.store.save_observation(observation)
        self.store.save_event(task_id, "observation_created", observation.to_dict(), step_id=step.id)
        verifier_name = self._verifier_for_step(step.tool, step.verifier)
        criteria = self._criteria_for_step(step.tool, step.success_criteria)
        self.store.save_event(task_id, "verification_started", {"verifier": verifier_name, "criteria": criteria}, step_id=step.id)
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
        self.store.save_event(task_id, "step_completed", outcome.to_dict(), step_id=step.id)
        action = ExperienceAction(
            action_type="execute" if round_number == 1 else "recovery_execute",
            step_id=step.id,
            tool_name=step.tool,
            input_fingerprint=self._input_fingerprint(step.input),
            execution_status=step_result.status,
            observation=observation.to_dict(),
            verification=verification.to_dict(),
        )
        experience.add_action(action)
        diagnosis = self.diagnoser.diagnose(step_result=step_result, observation=observation, verification=verification)
        return outcome, diagnosis, action

    def _apply_decision(
        self,
        current_plan: TaskPlan,
        step: PlanStep,
        decision: RecoveryDecision,
        diagnosis: FailureDiagnosis,
        experience: ExperienceRecord,
    ) -> TaskPlan:
        permissions = self._permissions()
        recovery_retrieved = [
            item.to_dict()
            for item in self.experience_memory.retrieve(
                ExperienceQuery(
                    goal=current_plan.goal,
                    tools=[step.tool],
                    failure_type=diagnosis.failure_type,
                    recovery_strategy=decision.strategy,
                )
            )
        ]
        experience.context.setdefault("recovery_retrieved_experiences", []).extend(recovery_retrieved)
        failure_observation = Observation(**experience.actions[-1].observation)
        replan_request = ReplanRequest(
            current_plan=current_plan,
            failure_observation=failure_observation,
            diagnosis=diagnosis,
            executed_history=[action.to_dict() for action in experience.actions],
            constraints={"max_steps": self.planner.max_steps},
            permissions=permissions,
            retrieved_experiences=recovery_retrieved,
        )
        if self.replanner.provider is not None:
            next_plan = self.replanner.replan(replan_request, self.tools, replacement_tool=decision.replacement_tool)
        elif decision.strategy == "CHANGE_TOOL":
            next_plan = self.replanner.replan_step(
                current_plan,
                step_id=step.id,
                tool_name=decision.replacement_tool,
                inputs=step.input,
                tools=self.tools,
                permissions=permissions,
            )
        elif decision.strategy in {"RETRY", "CHANGE_PARAMETERS", "RETRY_WITH_MODIFIED_INPUT"}:
            inputs = decision.modified_input or step.input
            next_plan = self.replanner.replan_step(
                current_plan,
                step_id=step.id,
                tool_name=step.tool,
                inputs=inputs,
                tools=self.tools,
                permissions=permissions,
            )
        elif decision.strategy == "REPLAN":
            next_plan = self.replanner.replan(replan_request, self.tools, replacement_tool=decision.replacement_tool)
        else:
            raise ReplanError(f"unsupported_recovery_strategy:{decision.strategy}")
        self.store.save_event(
            experience.task_id,
            "plan_replanned",
            {"strategy": decision.strategy, "plan": self._plan_payload(next_plan)},
            step_id=step.id,
        )
        return next_plan

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

    def _permissions(self) -> dict[str, str]:
        return {
            metadata["name"]: metadata["permission_level"]
            for metadata in self.tools.available()
            if metadata["permission_level"] in self.approved_permissions
        }

    @staticmethod
    def _input_fingerprint(inputs: dict[str, Any]) -> str:
        from private_agent.core.recovery import input_fingerprint

        return input_fingerprint(inputs)

    @staticmethod
    def _plan_payload(plan: TaskPlan) -> dict[str, Any]:
        return {
            "goal": plan.goal,
            "rationale": plan.rationale,
            "steps": [step.__dict__ for step in plan.steps],
        }

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
