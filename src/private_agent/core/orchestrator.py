from __future__ import annotations

import uuid
import inspect
from typing import Any

from private_agent.core.diagnosis import FailureDiagnoser, FailureDiagnosis
from private_agent.core.execution import ExecutionContext, GenericExecutor, StepResult
from private_agent.core.experience import ExperienceAction, ExperienceRecord
from private_agent.core.learning import LearningEngine, LearningMemory, SQLiteLearningMemory
from private_agent.core.memory import ExperienceMemory, ExperienceQuery, SQLiteExperienceMemory
from private_agent.core.observation import Observation, StepOutcome
from private_agent.core.planner import PlanStep, Planner, TaskPlan
from private_agent.core.recovery import RecoveryDecision, RecoveryManager
from private_agent.core.replanning import ReplanError, ReplanRequest, Replanner
from private_agent.core.reflection import ReflectionEngine, ReflectionMemory, SQLiteReflectionMemory
from private_agent.core.task_state import (
    StateIntegrityError,
    TaskState,
    TERMINAL_STATUSES,
    action_fingerprint,
)
from private_agent.core.verification import VerificationResult, VerifierRegistry
from private_agent.storage import KnowledgeItem, Store
from private_agent.security import PolicyDecision, SecurityController
from private_agent.tools.research import ToolRegistry


class TaskInterrupted(RuntimeError):
    """Internal deterministic interruption seam used to prove checkpoint resume behavior."""


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
        reflection_memory: ReflectionMemory | None = None,
        reflection_engine: ReflectionEngine | None = None,
        learning_memory: LearningMemory | None = None,
        learning_engine: LearningEngine | None = None,
        security_controller: SecurityController | None = None,
        checkpoint_hook: Any | None = None,
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
        self.reflection_memory = reflection_memory or SQLiteReflectionMemory(store)
        self.reflection_engine = reflection_engine or ReflectionEngine(self.experience_memory, self.reflection_memory)
        self.learning_memory = learning_memory or SQLiteLearningMemory(store)
        self.learning_engine = learning_engine or LearningEngine(self.learning_memory)
        self.security = security_controller
        self.checkpoint_hook = checkpoint_hook

    def run(
        self,
        goal: str,
        *,
        task_id: str | None = None,
        approval_ids: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if task_id:
            existing_state = self.store.get_task_state(task_id)
            if existing_state is not None:
                return self.resume_task(task_id, approval_ids=approval_ids)
        task_id = task_id or str(uuid.uuid4())
        approval_ids = approval_ids or {}
        self.recovery.attempts.clear()
        state = TaskState.create(task_id, goal)
        state.context["attempts"] = 0
        self._checkpoint(state, "TASK_CREATED", phase="TASK_CREATED", successful=True)
        state.checkpoint("TASK_PLANNING", status="PLANNING", phase="PLANNING", successful=True)
        self._save_checkpoint(state, "TASK_PLANNING")
        prior = self.store.search_knowledge(goal)
        retrieved = self.experience_memory.retrieve(ExperienceQuery(goal=goal))
        retrieved_payload = [item.to_dict() for item in retrieved]
        reflection_context = [
            insight.to_dict()
            for insight in self.reflection_memory.list_relevant({"goal": goal}, limit=5)
        ]
        learned_context = [
            strategy.to_dict()
            for strategy in self.learning_memory.retrieve_relevant(
                {
                    "goal": goal,
                    "task_type": "general",
                    "tools": [metadata["name"] for metadata in self.tools.available()],
                },
                limit=5,
            )
        ]
        initial_plan = self._create_plan(goal, prior, retrieved_payload, reflection_context, learned_context)
        current_plan = initial_plan
        state.context.update(
            {
                "prior_knowledge_used": prior,
                "retrieved_experiences": retrieved_payload,
                "reflection_insights_used": reflection_context,
                "learned_strategies_used": learned_context,
            }
        )
        state.set_plan(self._plan_payload(initial_plan), version=1)
        state.checkpoint("PLAN_CREATED", status="READY", phase="PLAN_CREATED", successful=True)
        self._save_checkpoint(state, "PLAN_CREATED")
        security_decisions: list[PolicyDecision] = []
        if self.security is not None:
            security_decisions = self.security.preflight_plan(
                task_id,
                initial_plan,
                self.tools,
                approval_ids=approval_ids,
            )
            state.security_decisions = {
                decision.action_id: decision.to_dict() for decision in security_decisions
            }
            for decision in security_decisions:
                if decision.approval_id:
                    state.approval_ids[decision.action_id] = decision.approval_id
            self.store.save_event(
                task_id,
                "security_preflight",
                {"decisions": [decision.to_dict() for decision in security_decisions]},
            )
            blocked = [decision for decision in security_decisions if not decision.allowed]
            if blocked:
                waiting = any(decision.decision == "REQUIRE_APPROVAL" for decision in blocked)
                state.recovery_state = {
                    "reason": "approval_required" if waiting else "policy_blocked",
                    "decisions": [decision.to_dict() for decision in blocked],
                }
                state.result = {"final_status": "BLOCKED"}
                state.checkpoint(
                    "BEFORE_WAITING_APPROVAL" if waiting else "RESUME_BLOCKED",
                    status="WAITING_APPROVAL" if waiting else "BLOCKED",
                    phase="WAITING_APPROVAL" if waiting else "SECURITY_BLOCKED",
                    successful=False,
                )
                self._save_checkpoint(state, "BEFORE_WAITING_APPROVAL" if waiting else "RESUME_BLOCKED")
                return self._security_blocked_result(
                    task_id,
                    goal,
                    initial_plan,
                    prior,
                    learned_context,
                    blocked,
                    state=state,
                )
        experience = ExperienceRecord(
            task_id=task_id,
            goal=goal,
            context={
                "prior_knowledge_used": prior,
                "retrieved_experiences": retrieved_payload,
                "reflection_insights_used": reflection_context,
                "learned_strategies_used": learned_context,
            },
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
            recovery_requested = False
            latest_outcomes = {}
            pending_steps = [
                step for step in current_plan.steps
                if state.action_key_for_step(step) not in set(state.completed_actions)
            ]
            if not pending_steps:
                terminal_status = "COMPLETED"
                break
            execution_context = ExecutionContext(
                task_id=task_id,
                approved_permissions=set(self.approved_permissions),
                metadata={"attempt": rounds},
                security_controller=self.security,
                approval_ids={**state.approval_ids, **approval_ids},
            )
            for step in pending_steps:
                self.store.save_event(task_id, "step_started", {"tool_name": step.tool, "round": rounds}, step_id=step.id)
                key = state.action_key_for_step(step)
                state.mark_action(
                    key,
                    status="EXECUTING",
                    step_id=step.id,
                    tool_name=step.tool,
                    input_fingerprint=action_fingerprint(
                        plan_version=state.plan_version,
                        step_id=step.id,
                        tool=step.tool,
                        inputs=step.input,
                    ),
                )
                state.context["attempts"] = rounds
                state.checkpoint("BEFORE_EXECUTION", status="EXECUTING", phase="BEFORE_EXECUTION", step_id=step.id)
                self._save_checkpoint(state, "BEFORE_EXECUTION")
                step_result = self._execute_one(step, execution_context)
                all_step_results.append(step_result)
                execution_context.previous_results[step.id] = step_result
                state.mark_action(
                    key,
                    status="OBSERVING",
                    step_id=step.id,
                    tool_name=step.tool,
                    input_fingerprint=action_fingerprint(
                        plan_version=state.plan_version,
                        step_id=step.id,
                        tool=step.tool,
                        inputs=step.input,
                    ),
                    execution_status=step_result.status,
                    output=step_result.to_dict(),
                )
                state.checkpoint("AFTER_EXECUTION", status="OBSERVING", phase="AFTER_EXECUTION", step_id=step.id)
                self._save_checkpoint(state, "AFTER_EXECUTION")
                outcome, diagnosis, action = self._observe_verify_diagnose(
                    task_id,
                    step,
                    step_result,
                    rounds,
                    experience,
                )
                all_outcomes.append(outcome)
                latest_outcomes[step.id] = outcome
                state.observations[key] = outcome.observation.to_dict()
                state.checkpoint("AFTER_OBSERVATION", status="VERIFYING", phase="AFTER_OBSERVATION", step_id=step.id)
                self._save_checkpoint(state, "AFTER_OBSERVATION")
                state.verification_results[key] = outcome.verification
                if diagnosis:
                    diagnoses.append(diagnosis.to_dict())
                if step_result.error:
                    errors.append(f"{step_result.error.code}: {step_result.error.message}")
                if outcome.verification_status in {"FAILED", "BLOCKED"}:
                    errors.append(f"verification_{outcome.verification_status.lower()}: {outcome.verification['reason']}")

                if outcome.verification_status == "VERIFIED" and outcome.execution_status == "success":
                    step.status = "completed"
                    state.mark_action(
                        key,
                        status="VERIFIED",
                        step_id=step.id,
                        tool_name=step.tool,
                        input_fingerprint=action_fingerprint(
                            plan_version=state.plan_version,
                            step_id=step.id,
                            tool=step.tool,
                            inputs=step.input,
                        ),
                        observation=outcome.observation.to_dict(),
                        verification=outcome.verification,
                    )
                    state.checkpoint("AFTER_VERIFICATION", status="READY", phase="AFTER_VERIFICATION", step_id=step.id, successful=True)
                    self._save_checkpoint(state, "AFTER_VERIFICATION")
                    continue

                state.mark_action(
                    key,
                    status="FAILED",
                    step_id=step.id,
                    tool_name=step.tool,
                    input_fingerprint=action_fingerprint(
                        plan_version=state.plan_version,
                        step_id=step.id,
                        tool=step.tool,
                        inputs=step.input,
                    ),
                    observation=outcome.observation.to_dict(),
                    verification=outcome.verification,
                )
                state.checkpoint("AFTER_VERIFICATION", status="READY", phase="AFTER_VERIFICATION", step_id=step.id)
                self._save_checkpoint(state, "AFTER_VERIFICATION")

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

                state.set_plan(self._plan_payload(current_plan), version=state.plan_version + 1)
                state.recovery_state = {
                    "decision": decision.to_dict(),
                    "diagnosis": diagnosis.to_dict(),
                    "round": rounds,
                }
                state.checkpoint("AFTER_RECOVERY", status="READY", phase="AFTER_RECOVERY", step_id=step.id, successful=True)
                self._save_checkpoint(state, "AFTER_RECOVERY")
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
            "learned_strategies_used": learned_context,
            "security_decisions": [decision.to_dict() for decision in security_decisions],
            "experience": experience.to_dict(),
            "final_status": terminal_status,
        }
        state.result = result
        state.context["attempts"] = rounds
        if terminal_status == "COMPLETED":
            state.checkpoint("TASK_COMPLETED", status="COMPLETED", phase="TASK_COMPLETED", successful=True)
            self._save_checkpoint(state, "TASK_COMPLETED")
        else:
            state.checkpoint("TASK_FAILED", status="FAILED", phase="TASK_FAILED", successful=False)
            self._save_checkpoint(state, "TASK_FAILED")
        self.experience_memory.store(experience)
        reflection_insights = self.reflection_engine.reflect_for_experience(experience)
        result["reflection_insights"] = [insight.to_dict() for insight in reflection_insights]
        learned_strategies = self.learning_engine.learn(reflection_insights)
        result["learned_strategies"] = [strategy.to_dict() for strategy in learned_strategies]
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
        return result

    def _security_blocked_result(
        self,
        task_id: str,
        goal: str,
        plan: TaskPlan,
        prior: list[dict[str, Any]],
        learned_context: list[dict[str, Any]],
        decisions: list[PolicyDecision],
        *,
        state: TaskState | None = None,
    ) -> dict[str, Any]:
        approval_requests = []
        if self.security is not None:
            for decision in decisions:
                if decision.approval_id:
                    request = self.security.approval_gate.get(decision.approval_id)
                    if request is not None:
                        approval_requests.append(request.to_dict())
        experience = ExperienceRecord(
            task_id=task_id,
            goal=goal,
            context={
                "prior_knowledge_used": prior,
                "learned_strategies_used": learned_context,
                "security_decisions": [decision.to_dict() for decision in decisions],
            },
            initial_plan=self._plan_payload(plan),
            final_outcome="BLOCKED",
        )
        self.experience_memory.store(experience)
        result = {
            "task_id": task_id,
            "goal": goal,
            "sources": [],
            "errors": ["security_policy_blocked"],
            "prior_knowledge_used": prior,
            "attempts": 0,
            "step_results": [],
            "observations": [],
            "step_outcomes": [],
            "final_step_outcomes": [],
            "verification": self._task_verification([]),
            "diagnoses": [],
            "recovery_attempts": [],
            "learned_strategies_used": learned_context,
            "security_decisions": [decision.to_dict() for decision in decisions],
            "approval_requests": approval_requests,
            "experience": experience.to_dict(),
            "final_status": "BLOCKED",
        }
        self.store.save_event(
            task_id,
            "security_action_blocked",
            {"decisions": result["security_decisions"], "approval_requests": approval_requests},
        )
        if state is not None:
            state.result = result
            state.context["attempts"] = 0
            state.checkpoint(
                "SECURITY_BLOCKED",
                status=state.status,
                phase=state.current_phase,
                successful=False,
            )
            self._save_checkpoint(state, "SECURITY_BLOCKED")
        self.store.save_task(task_id, goal, "blocked", self._plan_payload(plan), result, 0)
        return result

    def resume_task(
        self,
        task_id: str,
        *,
        approval_ids: dict[str, str] | None = None,
        unknown_resolutions: dict[str, dict[str, Any] | str] | None = None,
    ) -> dict[str, Any]:
        """Load, validate, re-authorize and continue one durable task."""
        approval_ids = approval_ids or {}
        unknown_resolutions = unknown_resolutions or {}
        try:
            payload = self.store.get_task_state(task_id)
            if payload is None:
                return self._resume_failure(task_id, "task_state_not_found")
            state = TaskState.from_dict(payload)
        except (StateIntegrityError, ValueError, TypeError, KeyError) as exc:
            return self._resume_failure(task_id, f"invalid_persisted_state:{type(exc).__name__}:{exc}")

        if state.status in TERMINAL_STATUSES:
            self._audit_task_event(state, "TASK_RESUMED", "Terminal task state reloaded without re-execution")
            return state.result or self._resume_failure(task_id, "terminal_state_missing_result")

        self.recovery.attempts.clear()
        self._audit_task_event(state, "TASK_RESUMED", "Task state loaded after process restart")
        state.context["resume_count"] = int(state.context.get("resume_count", 0)) + 1
        state.approval_ids.update(approval_ids)

        unresolved = [
            (key, record)
            for key, record in state.action_records.items()
            if record.get("status") == "EXECUTING"
        ]
        if state.status in {"EXECUTING", "EXECUTION_UNKNOWN"} and not unresolved:
            return self._resume_blocked(state, "ambiguous_execution_state_missing_action_record")
        if state.status in {"EXECUTING", "EXECUTION_UNKNOWN"} or unresolved:
            for key, record in unresolved:
                state.transition("EXECUTION_UNKNOWN", phase="EXECUTION_UNKNOWN", step_id=record.get("step_id", ""))
                state.recovery_state = {
                    "reason": "execution_interrupted_before_verified_result",
                    "action_key": key,
                    "step_id": record.get("step_id", ""),
                }
                state.touch()
                self._save_checkpoint(state, "EXECUTION_UNKNOWN")
                self._audit_task_event(state, "EXECUTION_UNKNOWN", "External side effect may have occurred")
                resolution = unknown_resolutions.get(key) or unknown_resolutions.get(record.get("step_id", ""))
                if resolution is None:
                    return self._resume_blocked(state, "ambiguous_execution_state_requires_observation")
                if not self._resolve_unknown(state, key, record, resolution):
                    return self._resume_blocked(state, "unknown_execution_resolution_not_safe")
            if state.status == "EXECUTION_UNKNOWN":
                state.transition("READY", phase="UNKNOWN_RESOLVED")
                self._save_checkpoint(state, "UNKNOWN_RESOLVED")

        if state.status == "OBSERVING":
            state.transition("READY", phase="RESUME_OBSERVING")
            self._save_checkpoint(state, "RESUME_OBSERVING")
        elif state.status == "VERIFYING":
            state.transition("READY", phase="RESUME_VERIFYING")
            self._save_checkpoint(state, "RESUME_VERIFYING")

        try:
            plan = self._plan_from_payload(state.plan)
        except (StateIntegrityError, KeyError, TypeError, ValueError) as exc:
            return self._resume_blocked(state, f"persisted_plan_invalid:{type(exc).__name__}:{exc}")
        if self.security is not None:
            try:
                security_ok, decisions = self._revalidate_resume_security(state, plan)
            except (KeyError, StateIntegrityError, ValueError) as exc:
                return self._resume_blocked(state, f"resume_security_revalidation_error:{type(exc).__name__}:{exc}")
            if not security_ok:
                return self._resume_blocked(state, "resume_security_revalidation_blocked", decisions=decisions)

        return self._resume_execute_state(state, plan)

    def _revalidate_resume_security(
        self,
        state: TaskState,
        plan: TaskPlan,
    ) -> tuple[bool, list[PolicyDecision]]:
        completed = set(state.completed_actions)
        pending_steps = [step for step in plan.steps if state.action_key_for_step(step) not in completed]
        if not pending_steps:
            state.checkpoint("APPROVAL_REVALIDATED", status="READY", phase="APPROVAL_REVALIDATED", successful=True)
            self._save_checkpoint(state, "APPROVAL_REVALIDATED")
            return True, []
        pending_plan = TaskPlan(plan.goal, pending_steps, plan.rationale)
        decisions = self.security.preflight_plan(
            state.task_id,
            pending_plan,
            self.tools,
            approval_ids=state.approval_ids,
        )
        state.security_decisions = {decision.action_id: decision.to_dict() for decision in decisions}
        for decision in decisions:
            if decision.approval_id:
                state.approval_ids[decision.action_id] = decision.approval_id
        blocked = [decision for decision in decisions if not decision.allowed]
        if blocked:
            waiting = any(decision.decision == "REQUIRE_APPROVAL" for decision in blocked)
            state.recovery_state = {
                "reason": "approval_required" if waiting else "policy_blocked",
                "decisions": [decision.to_dict() for decision in blocked],
            }
            state.checkpoint(
                "BEFORE_WAITING_APPROVAL" if waiting else "RESUME_BLOCKED",
                status="WAITING_APPROVAL" if waiting else "BLOCKED",
                phase="WAITING_APPROVAL" if waiting else "RESUME_SECURITY_BLOCKED",
            )
            self._save_checkpoint(state, "BEFORE_WAITING_APPROVAL" if waiting else "RESUME_BLOCKED")
            return False, decisions
        state.checkpoint("APPROVAL_REVALIDATED", status="READY", phase="APPROVAL_REVALIDATED", successful=True)
        self._save_checkpoint(state, "APPROVAL_REVALIDATED")
        return True, decisions

    def _resolve_unknown(
        self,
        state: TaskState,
        key: str,
        record: dict[str, Any],
        resolution: dict[str, Any] | str,
    ) -> bool:
        if isinstance(resolution, str):
            resolution = {"decision": resolution}
        decision = str(resolution.get("decision", resolution.get("status", ""))).lower()
        if decision == "retry":
            if resolution.get("observed_not_executed") is not True:
                return False
            record["status"] = "NOT_STARTED"
            record["resolution"] = "explicit_observation_not_executed"
            state.pending_actions = list(dict.fromkeys(state.pending_actions + [key]))
            state.recovery_state = {"action_key": key, "resolution": "retry"}
            return True
        if decision == "verified":
            observation = resolution.get("observation")
            verification = resolution.get("verification")
            if not isinstance(observation, dict) or not isinstance(verification, dict):
                return False
            if verification.get("status") != "VERIFIED" or not verification.get("evidence"):
                return False
            state.observations[key] = observation
            state.verification_results[key] = verification
            state.mark_action(
                key,
                status="VERIFIED",
                step_id=str(record.get("step_id", "")),
                tool_name=str(record.get("tool_name", "")),
                input_fingerprint=str(record.get("input_fingerprint", "")),
                observation=observation,
                verification=verification,
                resolution="explicit_external_observation",
            )
            return True
        return False

    def _resume_execute_state(self, state: TaskState, current_plan: TaskPlan) -> dict[str, Any]:
        prior = list(state.context.get("prior_knowledge_used", []))
        learned_context = list(state.context.get("learned_strategies_used", []))
        retrieved_payload = list(state.context.get("retrieved_experiences", []))
        reflection_context = list(state.context.get("reflection_insights_used", []))
        experience = self._experience_from_state(state, current_plan)
        all_step_results = [
            self._step_result_from_dict(record.get("output"))
            for record in state.action_records.values()
            if isinstance(record.get("output"), dict) and self._step_result_from_dict(record.get("output")) is not None
        ]
        all_step_results = [item for item in all_step_results if item is not None]
        all_outcomes: list[StepOutcome] = []
        latest_outcomes: dict[str, StepOutcome] = {}
        for key, observation_payload in state.observations.items():
            verification = state.verification_results.get(key)
            if not isinstance(verification, dict):
                continue
            try:
                observation = Observation(**observation_payload)
                outcome = StepOutcome(
                    state.task_id,
                    observation.step_id,
                    observation.tool_name,
                    observation.execution_status,
                    str(verification.get("status", "FAILED")),
                    observation,
                    verification,
                )
            except (TypeError, KeyError):
                continue
            all_outcomes.append(outcome)
            latest_outcomes[outcome.step_id] = outcome
        errors = list(state.result.get("errors", []))
        diagnoses = list(state.result.get("diagnoses", []))
        security_decisions = list(state.security_decisions.values())
        rounds = max(0, int(state.context.get("attempts", 0)))
        terminal_status = "FAILED"
        max_rounds = max(rounds + 1, 1 + self.recovery.max_recovery_attempts)
        while rounds < max_rounds:
            rounds += 1
            recovery_requested = False
            latest_outcomes = {
                outcome.step_id: outcome for outcome in all_outcomes
                if outcome.verification_status == "VERIFIED"
            }
            pending_steps = [
                step for step in current_plan.steps
                if state.action_key_for_step(step) not in set(state.completed_actions)
            ]
            if not pending_steps:
                terminal_status = "COMPLETED"
                break
            context = ExecutionContext(
                task_id=state.task_id,
                approved_permissions=set(self.approved_permissions),
                metadata={"attempt": rounds, "goal": state.goal},
                security_controller=self.security,
                approval_ids=state.approval_ids,
            )
            for step in pending_steps:
                key = state.action_key_for_step(step)
                record = state.action_records.get(key, {})
                if record.get("status") == "OBSERVING" and isinstance(record.get("output"), dict):
                    step_result = self._step_result_from_dict(record["output"])
                    if step_result is None:
                        return self._resume_blocked(state, "persisted_execution_result_corrupted")
                else:
                    state.mark_action(
                        key,
                        status="EXECUTING",
                        step_id=step.id,
                        tool_name=step.tool,
                        input_fingerprint=action_fingerprint(
                            plan_version=state.plan_version,
                            step_id=step.id,
                            tool=step.tool,
                            inputs=step.input,
                        ),
                    )
                    state.context["attempts"] = rounds
                    state.checkpoint("BEFORE_EXECUTION", status="EXECUTING", phase="BEFORE_EXECUTION", step_id=step.id)
                    self._save_checkpoint(state, "BEFORE_EXECUTION")
                    step_result = self._execute_one(step, context)
                    state.mark_action(
                        key,
                        status="OBSERVING",
                        step_id=step.id,
                        tool_name=step.tool,
                        input_fingerprint=action_fingerprint(
                            plan_version=state.plan_version,
                            step_id=step.id,
                            tool=step.tool,
                            inputs=step.input,
                        ),
                        execution_status=step_result.status,
                        output=step_result.to_dict(),
                    )
                    self._save_checkpoint_after(state, "AFTER_EXECUTION", step.id)
                all_step_results.append(step_result)
                context.previous_results[step.id] = step_result
                outcome, diagnosis, action = self._observe_verify_diagnose(
                    state.task_id, step, step_result, rounds, experience
                )
                all_outcomes.append(outcome)
                latest_outcomes[step.id] = outcome
                state.observations[key] = outcome.observation.to_dict()
                self._save_checkpoint_after(state, "AFTER_OBSERVATION", step.id, status="VERIFYING")
                state.verification_results[key] = outcome.verification
                if diagnosis:
                    diagnoses.append(diagnosis.to_dict())
                if step_result.error:
                    errors.append(f"{step_result.error.code}: {step_result.error.message}")
                if outcome.verification_status in {"FAILED", "BLOCKED"}:
                    errors.append(f"verification_{outcome.verification_status.lower()}: {outcome.verification['reason']}")
                if outcome.verification_status == "VERIFIED" and outcome.execution_status == "success":
                    state.mark_action(
                        key,
                        status="VERIFIED",
                        step_id=step.id,
                        tool_name=step.tool,
                        input_fingerprint=action_fingerprint(
                            plan_version=state.plan_version,
                            step_id=step.id,
                            tool=step.tool,
                            inputs=step.input,
                        ),
                        observation=outcome.observation.to_dict(),
                        verification=outcome.verification,
                    )
                    state.checkpoint("AFTER_VERIFICATION", status="READY", phase="AFTER_VERIFICATION", step_id=step.id, successful=True)
                    self._save_checkpoint(state, "AFTER_VERIFICATION")
                    continue
                state.mark_action(
                    key,
                    status="FAILED",
                    step_id=step.id,
                    tool_name=step.tool,
                    input_fingerprint=action_fingerprint(
                        plan_version=state.plan_version,
                        step_id=step.id,
                        tool=step.tool,
                        inputs=step.input,
                    ),
                    observation=outcome.observation.to_dict(),
                    verification=outcome.verification,
                )
                self._save_checkpoint_after(state, "AFTER_VERIFICATION", step.id)
                if diagnosis is None:
                    diagnosis = FailureDiagnosis(
                        "UNKNOWN_FAILURE", "No diagnosis was produced for an unsuccessful step",
                        failed_step=step.id, tool_name=step.tool, recoverable=False, suggested_recovery="ABORT",
                    )
                decision = self.recovery.choose(
                    step, diagnosis, available_tools=self.tools.describe(), allow_replan=self.replanner.provider is not None
                )
                attempt = self.recovery.record(decision, diagnosis, tool_name=step.tool, inputs=step.input)
                if decision.strategy == "ABORT":
                    self.recovery.update_last(outcome=decision.terminal_status or "FAILED", verification_status=outcome.verification_status)
                    terminal_status = decision.terminal_status or "FAILED"
                    errors.append(decision.reason)
                    break
                try:
                    current_plan = self._apply_decision(current_plan, step, decision, diagnosis, experience)
                except ReplanError as exc:
                    terminal_status = "BLOCKED"
                    errors.append(str(exc))
                    break
                state.set_plan(self._plan_payload(current_plan), version=state.plan_version + 1)
                state.recovery_state = {"decision": decision.to_dict(), "diagnosis": diagnosis.to_dict(), "round": rounds}
                state.checkpoint("AFTER_RECOVERY", status="READY", phase="AFTER_RECOVERY", step_id=step.id, successful=True)
                self._save_checkpoint(state, "AFTER_RECOVERY")
                attempt.outcome = "replanned"
                recovery_requested = True
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
            "task_id": state.task_id,
            "goal": state.goal,
            "sources": self._collect_sources(final_outcomes),
            "errors": errors,
            "prior_knowledge_used": prior,
            "attempts": rounds,
            "step_results": [item.to_dict() for item in all_step_results],
            "observations": [item.observation.to_dict() for item in all_outcomes],
            "step_outcomes": [item.to_dict() for item in all_outcomes],
            "final_step_outcomes": [item.to_dict() for item in final_outcomes],
            "verification": verification_payload,
            "diagnoses": diagnoses,
            "recovery_attempts": [attempt.to_dict() for attempt in self.recovery.attempts],
            "learned_strategies_used": learned_context,
            "security_decisions": security_decisions,
            "experience": experience.to_dict(),
            "final_status": terminal_status,
        }
        state.result = result
        state.context["attempts"] = rounds
        if terminal_status == "COMPLETED":
            state.checkpoint("TASK_COMPLETED", status="COMPLETED", phase="TASK_COMPLETED", successful=True)
            self._save_checkpoint(state, "TASK_COMPLETED")
        else:
            state.checkpoint("TASK_FAILED", status="FAILED", phase="TASK_FAILED")
            self._save_checkpoint(state, "TASK_FAILED")
        self.experience_memory.store(experience)
        reflection_insights = self.reflection_engine.reflect_for_experience(experience)
        result["reflection_insights"] = [insight.to_dict() for insight in reflection_insights]
        learned_strategies = self.learning_engine.learn(reflection_insights)
        result["learned_strategies"] = [strategy.to_dict() for strategy in learned_strategies]
        state.result = result
        state.touch()
        self._save_checkpoint(state, "TASK_COMPLETED" if terminal_status == "COMPLETED" else "TASK_FAILED")
        self.store.save_task(state.task_id, state.goal, terminal_status.lower(), self._plan_payload(current_plan), result, rounds)
        return result

    def _resume_blocked(
        self,
        state: TaskState,
        reason: str,
        *,
        decisions: list[PolicyDecision] | None = None,
    ) -> dict[str, Any]:
        waiting = state.status == "WAITING_APPROVAL"
        unresolved_unknown = "ambiguous_execution_state" in reason
        if not waiting and not unresolved_unknown:
            state.transition("BLOCKED", phase="RESUME_BLOCKED")
        state.recovery_state = {**state.recovery_state, "reason": reason}
        result = dict(state.result or {})
        result.update(
            {
                "task_id": state.task_id,
                "goal": state.goal,
                "errors": list(result.get("errors", [])) + [reason],
                "security_decisions": [decision.to_dict() for decision in decisions] if decisions else list(state.security_decisions.values()),
                "final_status": "BLOCKED",
            }
        )
        if self.security is not None:
            result["approval_requests"] = [
                request.to_dict()
                for request in (
                    self.security.approval_gate.get(approval_id)
                    for approval_id in state.approval_ids.values()
                )
                if request is not None
            ]
        state.result = result
        state.checkpoint(
            "RESUME_BLOCKED",
            status=state.status,
            phase="WAITING_APPROVAL" if waiting else "EXECUTION_UNKNOWN" if unresolved_unknown else "RESUME_BLOCKED",
        )
        self._save_checkpoint(state, "RESUME_BLOCKED")
        self._audit_task_event(state, "RESUME_BLOCKED", reason)
        return result

    def _resume_failure(self, task_id: str, reason: str) -> dict[str, Any]:
        self.store.save_event(task_id, "resume_blocked", {"reason": reason})
        return {"task_id": task_id, "errors": [reason], "final_status": "BLOCKED", "verification": {"verified": False, "status": "BLOCKED"}}

    def _plan_from_payload(self, payload: dict[str, Any]) -> TaskPlan:
        if not isinstance(payload, dict) or not isinstance(payload.get("steps"), list):
            raise StateIntegrityError("persisted_plan_missing")
        allowed = {"id", "objective", "tool", "input", "reason", "depends_on", "verifier", "success_criteria", "status", "output"}
        steps = []
        for raw in payload["steps"]:
            if not isinstance(raw, dict) or not set(raw).issubset(allowed):
                raise StateIntegrityError("persisted_plan_step_invalid")
            steps.append(
                PlanStep(
                    id=raw["id"], objective=raw["objective"], tool=raw["tool"], input=dict(raw.get("input", {})),
                    reason=raw.get("reason", ""), depends_on=list(raw.get("depends_on", [])), verifier=raw.get("verifier", ""),
                    success_criteria=dict(raw.get("success_criteria", {})), status=raw.get("status", "pending"), output=dict(raw.get("output", {})),
                )
            )
        if not steps:
            raise StateIntegrityError("persisted_plan_empty")
        return TaskPlan(str(payload.get("goal", "")), steps, str(payload.get("rationale", "")))

    def _experience_from_state(self, state: TaskState, plan: TaskPlan) -> ExperienceRecord:
        payload = state.result.get("experience", {})
        actions = [ExperienceAction(**item) for item in payload.get("actions", []) if isinstance(item, dict)]
        return ExperienceRecord(
            task_id=state.task_id,
            goal=state.goal,
            context=dict(payload.get("context", state.context)),
            initial_plan=dict(payload.get("initial_plan", self._plan_payload(plan))),
            actions=actions,
            final_outcome=str(payload.get("final_outcome", "")),
            identity=dict(payload.get("identity", {})),
            metadata=dict(payload.get("metadata", {})),
            created_at=str(payload.get("created_at", "")) or None or ExperienceRecord.__dataclass_fields__["created_at"].default_factory(),
        )

    @staticmethod
    def _step_result_from_dict(payload: Any) -> StepResult | None:
        if not isinstance(payload, dict) or not payload.get("step_id") or not payload.get("tool_name"):
            return None
        error_payload = payload.get("error")
        error = None
        if isinstance(error_payload, dict):
            from private_agent.tools.contracts import ToolError

            error = ToolError(error_payload.get("code", "unknown"), error_payload.get("message", ""), error_payload.get("details", {}))
        return StepResult(payload["step_id"], payload["tool_name"], payload.get("status", "failed"), payload.get("output"), error, dict(payload.get("metadata", {})))

    def _save_checkpoint_after(self, state: TaskState, checkpoint_type: str, step_id: str, *, status: str = "OBSERVING") -> None:
        state.checkpoint(checkpoint_type, status=status, phase=checkpoint_type, step_id=step_id)
        self._save_checkpoint(state, checkpoint_type)

    def _checkpoint(
        self,
        state: TaskState,
        checkpoint_type: str,
        *,
        status: str | None = None,
        phase: str | None = None,
        step_id: str | None = None,
        successful: bool = False,
    ) -> None:
        state.checkpoint(checkpoint_type, status=status, phase=phase, step_id=step_id, successful=successful)
        self._save_checkpoint(state, checkpoint_type)

    def _save_checkpoint(self, state: TaskState, checkpoint_type: str) -> None:
        state.validate()
        self.store.save_task_state(state, checkpoint_type)
        self._audit_task_event(state, "CHECKPOINT_SAVED", checkpoint_type)
        special_events = {
            "TASK_CREATED": "TASK_CREATED",
            "TASK_COMPLETED": "TASK_COMPLETED",
            "TASK_FAILED": "TASK_FAILED",
            "EXECUTION_UNKNOWN": "EXECUTION_UNKNOWN",
            "AFTER_RECOVERY": "RECOVERY_REQUIRED",
        }
        special_event = special_events.get(checkpoint_type)
        if special_event:
            self._audit_task_event(state, special_event, checkpoint_type)
        if self.checkpoint_hook is not None:
            self.checkpoint_hook(checkpoint_type, state.to_dict())

    def _audit_task_event(self, state: TaskState, event_type: str, reason: str) -> None:
        if self.security is None:
            return
        from private_agent.security import ActionRequest

        action = ActionRequest(
            task_id=state.task_id,
            action_id=f"task-state:{event_type}:{state.current_step}",
            tool_or_skill="task-state",
            declared_capabilities=[],
            reason=reason,
            planned_effect="Persist or resume task state",
            input_fingerprint=state.state_fingerprint,
            action_type="task_state",
        )
        self.security.audit.record(
            event_type,
            action,
            decision=state.status,
            reason=reason,
            policy_version=self.security.policy_engine.policy.policy_version,
        )

    def _create_plan(
        self,
        goal: str,
        prior: list[dict[str, Any]],
        retrieved: list[dict[str, Any]],
        reflection_insights: list[dict[str, Any]],
        learned_strategies: list[dict[str, Any]],
    ) -> TaskPlan:
        parameters = inspect.signature(self.planner.create).parameters
        kwargs: dict[str, Any] = {}
        if "permissions" in parameters:
            kwargs["permissions"] = self._permissions()
        if "retrieved_experiences" in parameters:
            kwargs["retrieved_experiences"] = retrieved
        if "reflection_insights" in parameters:
            kwargs["reflection_insights"] = reflection_insights
        if "learned_strategies" in parameters:
            kwargs["learned_strategies"] = learned_strategies
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

    def _execute_one(self, step: PlanStep, context: ExecutionContext) -> StepResult:
        if callable(getattr(self.executor, "execute_step", None)):
            return self.executor.execute_step(step, context)
        results = self.executor.execute_plan(
            TaskPlan(context.metadata.get("goal", ""), [step], "checkpointed single-step execution"),
            context,
        )
        if not results:
            return StepResult.failed(step.id, step.tool, "empty_executor_result", "Executor returned no step result")
        return results[0]

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
        recovery_reflections = [
            insight.to_dict()
            for insight in self.reflection_memory.list_relevant(
                {
                    "goal": current_plan.goal,
                    "task_type": "general",
                    "tools": [step.tool],
                    "failure_types": [diagnosis.failure_type],
                    "recovery_strategies": [decision.strategy],
                },
                limit=5,
            )
        ]
        recovery_learning = [
            strategy.to_dict()
            for strategy in self.learning_memory.retrieve_relevant(
                {
                    "goal": current_plan.goal,
                    "task_type": "general",
                    "tools": [step.tool],
                    "failure_types": [diagnosis.failure_type],
                },
                limit=5,
            )
        ]
        experience.context.setdefault("recovery_retrieved_experiences", []).extend(recovery_retrieved)
        experience.context.setdefault("recovery_reflection_insights", []).extend(recovery_reflections)
        experience.context.setdefault("recovery_learned_strategies", []).extend(recovery_learning)
        failure_observation = Observation(**experience.actions[-1].observation)
        replan_request = ReplanRequest(
            current_plan=current_plan,
            failure_observation=failure_observation,
            diagnosis=diagnosis,
            executed_history=[action.to_dict() for action in experience.actions],
            constraints={"max_steps": self.planner.max_steps},
            permissions=permissions,
            retrieved_experiences=recovery_retrieved,
            reflection_insights=recovery_reflections,
            learned_strategies=recovery_learning,
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
