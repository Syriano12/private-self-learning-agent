from __future__ import annotations

import uuid
import inspect
import json
import re
from time import monotonic
from typing import Any

from private_agent.core.diagnosis import FailureDiagnoser, FailureDiagnosis
from private_agent.core.execution import ExecutionContext, GenericExecutor, StepResult
from private_agent.core.experience import ExperienceAction, ExperienceRecord
from private_agent.core.learning import LearningEngine, LearningMemory, SQLiteLearningMemory
from private_agent.core.memory import ExperienceMemory, ExperienceQuery, SQLiteExperienceMemory
from private_agent.core.observation import Observation, StepOutcome
from private_agent.core.planner import CapabilityUnavailableError, PlanStep, Planner, PlanValidationError, TaskPlan
from private_agent.core.recovery import RecoveryDecision, RecoveryManager
from private_agent.core.replanning import ReplanError, ReplanRequest, Replanner
from private_agent.core.reflection import ReflectionEngine, ReflectionMemory, SQLiteReflectionMemory
from private_agent.core.task_state import (
    MAX_TASK_GOAL_LENGTH,
    StateIntegrityError,
    TaskState,
    TaskStopReasonCode,
    TERMINAL_STATUSES,
    action_fingerprint,
)
from private_agent.core.verification import VerificationResult, VerifierRegistry
from private_agent.observability import ObservabilityMonitor
from private_agent.storage import KnowledgeItem, Store
from private_agent.security import PolicyDecision, SecurityController
from private_agent.tools.research import ToolRegistry

MAX_CONVERSATION_REPLANS = 32


class TaskInterrupted(RuntimeError):
    """Internal deterministic interruption seam used to prove checkpoint resume behavior."""


_SAFE_TOOL_REFERENCE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


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
        observability: ObservabilityMonitor | None = None,
    ) -> None:
        self.store, self.tools, self.max_attempts = store, tools, max_attempts
        self.approved_permissions = approved_permissions or {"public_read"}
        self.planner = planner or Planner()
        self.security = security_controller or SecurityController(store=store)
        if executor is not None:
            raise TypeError("unsupported_executor: production execution owns GenericExecutor")
        self.executor = executor or GenericExecutor(
            tools,
            security_controller=self.security,
            lease_store=store,
        )
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
        self.checkpoint_hook = checkpoint_hook
        self.observability = observability or ObservabilityMonitor(store)
        self.security.set_observability(self.observability)

    def _emit_observability(
        self,
        event_type: str,
        *,
        task_id: str = "",
        action_id: str = "",
        step_id: str = "",
        severity: str = "INFO",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.observability.emit(
            event_type,
            task_id=task_id,
            action_id=action_id,
            step_id=step_id,
            severity=severity,
            component="orchestrator",
            metadata=metadata or {},
        )

    @staticmethod
    def _action_id_for_step(state: TaskState, step_id: str) -> str | None:
        return next(
            (
                action_id
                for action_id, record in state.action_records.items()
                if record.get("step_id") == step_id
            ),
            None,
        )

    @staticmethod
    def _set_policy_stop_reason(state: TaskState, decisions: list[PolicyDecision]) -> None:
        blocked = [decision for decision in decisions if not decision.allowed]
        if not blocked:
            state.set_stop_reason(None)
            return
        pending = next((decision for decision in blocked if decision.decision == "REQUIRE_APPROVAL"), None)
        selected = pending or blocked[0]
        action_record = state.action_records.get(selected.action_id, {})
        step_id = action_record.get("step_id") if isinstance(action_record, dict) else None
        state.set_stop_reason(
            TaskStopReasonCode.APPROVAL_REQUIRED if pending else TaskStopReasonCode.POLICY_BLOCKED,
            step_id=step_id,
            action_id=selected.action_id,
        )

    def _verified_outcomes_for_plan(self, state: TaskState, plan: TaskPlan) -> dict[str, StepOutcome]:
        outcomes: dict[str, StepOutcome] = {}
        for step in plan.steps:
            key = state.action_key_for_step(step)
            record = state.action_records.get(key)
            observation_payload = state.observations.get(key)
            verification = state.verification_results.get(key)
            if (
                key not in state.completed_actions
                or not isinstance(record, dict)
                or record.get("status") != "VERIFIED"
                or record.get("step_id") != step.id
                or not isinstance(observation_payload, dict)
                or not isinstance(verification, dict)
            ):
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
            outcomes[step.id] = outcome
        return outcomes

    def _block_conversation_replan(self, state: TaskState, reason: str) -> dict[str, Any]:
        result = {
            "task_id": state.task_id,
            "goal": state.goal,
            "errors": [reason],
            "verification": {"verified": False, "status": "BLOCKED"},
            "final_status": "BLOCKED",
        }
        state.result = result
        state.checkpoint(
            "CONVERSATION_REPLAN_BLOCKED",
            status="BLOCKED",
            phase="CONVERSATION_REPLAN_BLOCKED",
            successful=False,
        )
        self._save_checkpoint(state, "CONVERSATION_REPLAN_BLOCKED")
        self.store.save_event(state.task_id, "conversation_replan_blocked", {"reason": reason})
        return result

    def _maybe_replan_from_followups(
        self,
        state: TaskState,
        current_plan: TaskPlan,
        *,
        prior: list[dict[str, Any]],
        retrieved: list[dict[str, Any]],
        reflection_context: list[dict[str, Any]],
        learned_context: list[dict[str, Any]],
    ) -> tuple[TaskPlan, list[PolicyDecision], dict[str, Any] | None, bool]:
        """Consume pending follow-ups only after a persisted verified-action checkpoint."""
        if not self.store.has_active_lease_context(state.task_id):
            return current_plan, list(state.security_decisions.values()), None, False
        has_verified_current_step = any(
            isinstance(record, dict)
            and record.get("step_id") == state.current_step
            and record.get("status") == "VERIFIED"
            and key in state.completed_actions
            for key, record in state.action_records.items()
        )
        if state.status != "READY" or not state.current_step or not has_verified_current_step:
            return current_plan, list(state.security_decisions.values()), None, False
        try:
            pending = self.store.pending_task_messages_for_worker(state.task_id)
        except ValueError as exc:
            # A stale or non-verification checkpoint is a reason to leave messages pending,
            # never to infer that an action completed.
            if str(exc) in {
                "conversation_safe_checkpoint_required",
                "conversation_verified_action_required",
            }:
                return current_plan, list(state.security_decisions.values()), None, False
            raise
        if not pending:
            return current_plan, list(state.security_decisions.values()), None, False
        if state.context.get("task_kind") == "bug_bounty":
            blocked = self._block_conversation_replan(state, "conversation_task_kind_unsupported")
            return current_plan, list(state.security_decisions.values()), blocked, False

        messages = self.store.task_messages(state.task_id)
        followups = [item for item in messages if item.get("message_kind") == "FOLLOW_UP"]
        effective_goal = state.goal
        if followups:
            effective_goal += "\n\nUser follow-up messages:\n" + "\n".join(
                f"- {str(item.get('content', '')).strip()}" for item in followups if str(item.get("content", "")).strip()
            )
        try:
            proposed = self._create_plan(effective_goal, prior, retrieved, reflection_context, learned_context)
            if not proposed.steps:
                raise ValueError("conversation_plan_empty")
            previous_completed: list[tuple[PlanStep, str, dict[str, Any]]] = []
            for step in current_plan.steps:
                old_key = state.action_key_for_step(step)
                if old_key not in state.completed_actions:
                    continue
                record = state.action_records.get(old_key)
                observation = state.observations.get(old_key)
                verification = state.verification_results.get(old_key)
                if (
                    not isinstance(record, dict)
                    or record.get("status") != "VERIFIED"
                    or record.get("action_key") != old_key
                    or record.get("input_fingerprint") != old_key
                    or record.get("step_id") != step.id
                    or record.get("tool_name") != step.tool
                    or not isinstance(observation, dict)
                    or record.get("observation") != observation
                    or not isinstance(verification, dict)
                    or verification.get("status") != "VERIFIED"
                    or verification.get("verified") is not True
                    or record.get("verification") != verification
                ):
                    raise ValueError("conversation_completed_action_untrusted")
                previous_completed.append((step, old_key, record))

            preserved_by_id = {step.id: step for step, _, _ in previous_completed}
            candidate_to_id: dict[str, str] = {}
            exact_matches: set[str] = set()

            def equivalent(left: PlanStep, right: PlanStep) -> bool:
                return all(
                    getattr(left, name) == getattr(right, name)
                    for name in ("id", "objective", "tool", "input", "reason", "depends_on", "verifier", "success_criteria")
                )

            for index, step in enumerate(proposed.steps, start=1):
                existing = preserved_by_id.get(step.id)
                if existing is not None and equivalent(existing, step):
                    candidate_to_id[step.id] = existing.id
                    exact_matches.add(step.id)
                else:
                    candidate_to_id[step.id] = f"conversation-{state.plan_version + 1}-{index}"

            new_steps: list[PlanStep] = []
            for step in proposed.steps:
                if step.id in exact_matches:
                    continue
                new_id = candidate_to_id[step.id]
                dependencies = [candidate_to_id.get(dep, dep) for dep in step.depends_on]
                new_steps.append(
                    PlanStep(
                        id=new_id,
                        objective=step.objective,
                        tool=step.tool,
                        input=dict(step.input),
                        reason=step.reason,
                        depends_on=dependencies,
                        verifier=step.verifier,
                        success_criteria=dict(step.success_criteria),
                        status="pending",
                        output={},
                    )
                )
            merged_steps = [step for step, _, _ in previous_completed] + new_steps
            all_ids = [step.id for step in merged_steps]
            if not merged_steps or len(merged_steps) > 64 or len(set(all_ids)) != len(all_ids):
                raise ValueError("conversation_plan_shape_invalid")
            id_set = set(all_ids)
            if any(not set(step.depends_on).issubset(id_set) for step in merged_steps):
                raise ValueError("conversation_plan_dependency_invalid")
            merged_plan = TaskPlan(
                goal=effective_goal,
                steps=merged_steps,
                rationale=f"{proposed.rationale}; previously verified actions retained",
            )
        except Exception as exc:
            blocked = self._block_conversation_replan(state, "conversation_replan_failed")
            self.store.save_event(
                state.task_id,
                "conversation_replan_error",
                {"error_category": type(exc).__name__},
            )
            return current_plan, list(state.security_decisions.values()), blocked, False

        old_completed_keys = {old_key for _, old_key, _ in previous_completed}
        old_observations = {key: state.observations.get(key) for _, key, _ in previous_completed}
        old_verifications = {key: state.verification_results.get(key) for _, key, _ in previous_completed}
        old_records = {key: dict(record) for _, key, record in previous_completed}
        completed_before = list(state.completed_actions)
        state.set_plan(self._plan_payload(merged_plan), version=state.plan_version + 1)
        new_completed_keys: list[str] = []
        for step, old_key, record in previous_completed:
            new_key = state.action_key_for_step(step)
            remapped = dict(old_records[old_key])
            remapped["action_key"] = new_key
            remapped["input_fingerprint"] = new_key
            state.action_records[new_key] = remapped
            if old_observations[old_key] is not None:
                state.observations[new_key] = old_observations[old_key]
            if old_verifications[old_key] is not None:
                state.verification_results[new_key] = old_verifications[old_key]
            if old_key != new_key:
                state.action_records.pop(old_key, None)
                state.observations.pop(old_key, None)
                state.verification_results.pop(old_key, None)
            new_completed_keys.append(new_key)
        state.completed_actions = [key for key in completed_before if key not in old_completed_keys] + new_completed_keys
        state.pending_actions = [
            state.action_key_for_step(step)
            for step in merged_plan.steps
            if state.action_key_for_step(step) not in set(state.completed_actions)
        ]
        message_ids = [str(item["message_id"]) for item in pending]
        state.context["conversation_replans"] = list(state.context.get("conversation_replans", [])) + [
            {"message_ids": message_ids, "plan_version": state.plan_version}
        ]
        state.checkpoint(
            "CONVERSATION_REPLANNED",
            status="READY",
            phase="CONVERSATION_REPLANNED",
            successful=True,
        )
        self._save_checkpoint(state, "CONVERSATION_REPLANNED", consume_message_ids=message_ids)
        self.store.save_event(
            state.task_id,
            "conversation_plan_updated",
            {"message_ids": message_ids, "plan_version": state.plan_version, "step_count": len(merged_plan.steps)},
        )
        if self.security is not None:
            try:
                security_ok, decisions = self._revalidate_resume_security(state, merged_plan)
            except (KeyError, StateIntegrityError, ValueError):
                blocked = self._resume_blocked(state, "conversation_security_revalidation_error")
                return merged_plan, list(state.security_decisions.values()), blocked, True
            if not security_ok:
                blocked = self._resume_blocked(
                    state,
                    "conversation_security_revalidation_blocked",
                    decisions=decisions,
                )
                return merged_plan, decisions, blocked, True
        else:
            decisions = []
        return merged_plan, decisions, None, True

    def run(
        self,
        goal: str,
        *,
        task_id: str | None = None,
        approval_ids: dict[str, str] | None = None,
        task_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if task_id:
            existing_state = self.store.get_task_state(task_id)
            if existing_state is not None and existing_state.get("status") != "QUEUED":
                return self.resume_task(task_id, approval_ids=approval_ids)
        if not isinstance(goal, str):
            raise ValueError("goal_must_be_string")
        goal = goal.strip()
        if not goal:
            raise ValueError("goal_required")
        if len(goal) > MAX_TASK_GOAL_LENGTH:
            raise ValueError("goal_too_long")
        task_id = task_id or str(uuid.uuid4())
        approval_ids = approval_ids or {}
        self.recovery.attempts.clear()
        state = TaskState.create(task_id, goal)
        if task_context is not None:
            if not isinstance(task_context, dict):
                raise ValueError("task_context_must_be_object")
            state.context.update(task_context)
        state.context["attempts"] = 0
        self._checkpoint(state, "TASK_CREATED", phase="TASK_CREATED", successful=True)
        state.checkpoint("TASK_PLANNING", status="PLANNING", phase="PLANNING", successful=True)
        self._save_checkpoint(state, "TASK_PLANNING")
        # Legacy knowledge rows are advisory and have no source-bound trust
        # record. Only signed, fresh typed claims may enter planning context.
        prior = self.store.search_knowledge_claims(goal)
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
                    "task_type": "bug_bounty" if state.context.get("task_kind") == "bug_bounty" else "general",
                    "tools": [metadata["name"] for metadata in self.tools.available()],
                },
                limit=5,
            )
        ]
        self._emit_observability(
            "MEMORY_RETRIEVED",
            task_id=task_id,
            metadata={
                "knowledge_count": len(prior),
                "experience_count": len(retrieved_payload),
                "reflection_count": len(reflection_context),
                "learned_strategy_count": len(learned_context),
            },
        )
        planning_started = monotonic()
        try:
            if state.context.get("task_kind") == "bug_bounty":
                request = state.context.get("bugbounty_request")
                if not isinstance(request, dict):
                    raise ValueError("bugbounty_request_missing")
                scanner = request.get("scanner", "deterministic")
                if scanner == "deterministic":
                    tool_name = "bugbounty_analysis"
                    step_input = {
                        key: request.get(key)
                        for key in ("scope", "fixture", "urls", "openapi", "source_documents")
                        if key in request
                    }
                    objective = "Run bounded local Bug Bounty analysis"
                    reason = "Explicit API request routed to the registered local-only analysis contract"
                    required_fields = ["run", "report"]
                elif scanner in {"httpx", "nuclei"}:
                    tool_name = "httpx_recon" if scanner == "httpx" else "nuclei_scan"
                    step_input = {
                        "scope": request.get("scope"),
                        "target": request.get("target"),
                        "identity": request.get("identity"),
                        "resource_id": request.get("resource_id"),
                    }
                    objective = f"Run bounded {scanner} reconnaissance within the explicit authorized scope"
                    reason = f"Explicit API request routed to the registered {scanner} adapter; results remain observations"
                    required_fields = ["scan"]
                else:
                    raise ValueError("scanner_type_unsupported")
                try:
                    self.tools.get(tool_name)
                except KeyError as exc:
                    raise CapabilityUnavailableError([tool_name]) from exc
                initial_plan = TaskPlan(
                    goal=goal,
                    steps=[PlanStep(
                        id=f"bugbounty-{task_id}",
                        objective=objective,
                        tool=tool_name,
                        input=step_input,
                        reason=reason,
                        depends_on=[],
                        success_criteria={"required_fields": required_fields},
                    )],
                    rationale="Specialized Bug Bounty route; normal plan validation and execution authorization remain authoritative",
                )
            else:
                initial_plan = self._create_plan(goal, prior, retrieved_payload, reflection_context, learned_context)
        except CapabilityUnavailableError as exc:
            registered_names = [str(metadata["name"]) for metadata in self.tools.available()]
            registered_set = set(registered_names)
            if exc.reason == "no_registered_tools":
                if registered_names:
                    raise PlanValidationError(["capability_registry_changed_during_planning"]) from exc
                unavailable_names: list[str] = []
            else:
                unavailable_names = [name for name in exc.unavailable_tools if name not in registered_set]
                if not unavailable_names:
                    raise PlanValidationError(["capability_registry_changed_during_planning"]) from exc
            return self._capability_gap_result(
                state,
                unavailable_tool_names=unavailable_names,
                registered_tool_names=registered_names,
            )
        except Exception as exc:
            self._emit_observability(
                "PLAN_REJECTED",
                task_id=task_id,
                severity="ERROR",
                metadata={"error_category": type(exc).__name__},
            )
            raise
        self._emit_observability(
            "PLAN_CREATED",
            task_id=task_id,
            metadata={
                "step_count": len(initial_plan.steps),
                "duration_ms": round((monotonic() - planning_started) * 1000, 3),
            },
        )
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
                self._set_policy_stop_reason(state, blocked)
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
        verified_previous_results: dict[str, StepResult] = {}

        self.store.save_event(task_id, "plan_created", {"goal": goal, "steps": [step.__dict__ for step in current_plan.steps]})
        max_rounds = 1 + self.recovery.max_recovery_attempts + MAX_CONVERSATION_REPLANS
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
                step for step in self._ordered_plan_steps(current_plan)
                if state.action_key_for_step(step) not in set(state.completed_actions)
            ]
            if not pending_steps:
                terminal_status = "COMPLETED"
                break
            execution_context = ExecutionContext(
                task_id=task_id,
                previous_results={},
                approved_permissions=set(self.approved_permissions),
                metadata={"attempt": rounds, "goal": current_plan.goal},
                security_controller=self.security,
                approval_ids={**state.approval_ids, **approval_ids},
                observability=self.observability,
            )
            for step in pending_steps:
                self.store.save_event(task_id, "step_started", {"tool_name": step.tool, "round": rounds}, step_id=step.id)
                key = state.action_key_for_step(step)
                self._emit_observability(
                    "ACTION_PROPOSED",
                    task_id=task_id,
                    action_id=step.id,
                    step_id=step.id,
                    metadata={"tool_name": step.tool, "round": rounds},
                )
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
                self._emit_observability(
                    "ACTION_STARTED",
                    task_id=task_id,
                    action_id=step.id,
                    step_id=step.id,
                    metadata={"tool_name": step.tool, "round": rounds},
                )
                step_context = self._isolated_execution_context(execution_context, verified_previous_results)
                step_result = self._execute_one(step, step_context)
                all_step_results.append(step_result)
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

                self._emit_observability(
                    "ACTION_COMPLETED" if step_result.status == "success" else "ACTION_LIFECYCLE_BLOCKED" if step_result.status == "blocked" else "ACTION_FAILED",
                    task_id=task_id,
                    action_id=step.id,
                    step_id=step.id,
                    severity="INFO" if step_result.status == "success" else "ERROR",
                    metadata={
                        "execution_status": step_result.status,
                        "verification_status": outcome.verification_status,
                        "error_category": step_result.error.code if step_result.error else "",
                        "duration_ms": step_result.metadata.get("duration_ms", 0),
                    },
                )

                if outcome.verification_status == "VERIFIED" and outcome.execution_status == "success":
                    step.status = "completed"
                    verified_previous_results[step.id] = step_result
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
                    current_plan, updated_decisions, conversation_blocked, conversation_updated = self._maybe_replan_from_followups(
                        state,
                        current_plan,
                        prior=prior,
                        retrieved=retrieved_payload,
                        reflection_context=reflection_context,
                        learned_context=learned_context,
                    )
                    if conversation_blocked is not None:
                        return conversation_blocked
                    if conversation_updated:
                        security_decisions = updated_decisions
                        latest_outcomes = self._verified_outcomes_for_plan(state, current_plan)
                        recovery_requested = True
                        break
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
                recovery_started = monotonic()
                self._emit_observability(
                    "RECOVERY_STARTED",
                    task_id=task_id,
                    action_id=step.id,
                    step_id=step.id,
                    metadata={"failure_type": diagnosis.failure_type},
                )
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
                    self._emit_observability(
                        "RECOVERY_COMPLETED",
                        task_id=task_id,
                        action_id=step.id,
                        step_id=step.id,
                        severity="ERROR",
                        metadata={"strategy": decision.strategy, "duration_ms": round((monotonic() - recovery_started) * 1000, 3)},
                    )
                    recovery_requested = False
                    break

                self._emit_observability(
                    "REPLAN_STARTED",
                    task_id=task_id,
                    action_id=step.id,
                    step_id=step.id,
                    metadata={"strategy": decision.strategy},
                )
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
                verified_previous_results.clear()
                self._emit_observability(
                    "PLAN_UPDATED",
                    task_id=task_id,
                    metadata={"plan_version": state.plan_version, "step_count": len(current_plan.steps)},
                )
                self._emit_observability(
                    "REPLAN_COMPLETED",
                    task_id=task_id,
                    action_id=step.id,
                    step_id=step.id,
                    metadata={"strategy": decision.strategy, "duration_ms": round((monotonic() - recovery_started) * 1000, 3)},
                )
                self._emit_observability(
                    "RECOVERY_COMPLETED",
                    task_id=task_id,
                    action_id=step.id,
                    step_id=step.id,
                    metadata={"strategy": decision.strategy, "duration_ms": round((monotonic() - recovery_started) * 1000, 3)},
                )
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
        if self._task_verification_allows_completion(verification_payload):
            terminal_status = "COMPLETED"
        elif terminal_status == "COMPLETED":
            terminal_status = "FAILED"

        experience.final_outcome = terminal_status
        experience.verification_status = str(verification_payload["status"])
        experience.verification_scope = str(verification_payload.get("verification_scope", ""))
        impact_verified = verification_payload.get("security_impact_verified")
        experience.security_impact_verified = impact_verified if isinstance(impact_verified, bool) else None
        experience.identity = {}
        experience.build_identity()
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
            state.set_stop_reason(
                TaskStopReasonCode.TASK_FAILED,
                step_id=state.current_step or None,
                action_id=self._action_id_for_step(state, state.current_step) if state.current_step else None,
            )
            state.checkpoint("TASK_FAILED", status="FAILED", phase="TASK_FAILED", successful=False)
            self._save_checkpoint(state, "TASK_FAILED")
        self.experience_memory.store(experience)
        self._emit_observability(
            "REFLECTION_STARTED",
            task_id=task_id,
            metadata={"action_count": len(experience.actions)},
        )
        reflection_insights = self.reflection_engine.reflect_for_experience(experience)
        result["reflection_insights"] = [insight.to_dict() for insight in reflection_insights]
        self._emit_observability(
            "REFLECTION_COMPLETED",
            task_id=task_id,
            metadata={"insight_count": len(reflection_insights)},
        )
        self._emit_observability(
            "LEARNING_STARTED",
            task_id=task_id,
            metadata={"insight_count": len(reflection_insights)},
        )
        learned_strategies = self.learning_engine.learn(reflection_insights)
        result["learned_strategies"] = [strategy.to_dict() for strategy in learned_strategies]
        self._emit_observability(
            "LEARNING_COMPLETED",
            task_id=task_id,
            metadata={"strategy_count": len(learned_strategies)},
        )
        if not learned_strategies:
            self._emit_observability(
                "LEARNING_REJECTED",
                task_id=task_id,
                severity="NOTICE",
                metadata={"reason": "no_eligible_strategy", "insight_count": len(reflection_insights)},
            )
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

    def _capability_gap_result(
        self,
        state: TaskState,
        *,
        unavailable_tool_names: list[str],
        registered_tool_names: list[str],
    ) -> dict[str, Any]:
        requested_tools = sorted({
            name for name in unavailable_tool_names
            if isinstance(name, str) and _SAFE_TOOL_REFERENCE.fullmatch(name)
        })[:20]
        registered_tools = sorted({
            name for name in registered_tool_names
            if isinstance(name, str) and _SAFE_TOOL_REFERENCE.fullmatch(name)
        })[:50]
        code = "UNREGISTERED_TOOL_REQUEST" if unavailable_tool_names else "NO_REGISTERED_TOOLS"
        summary = (
            "The planner requested a tool that is not registered; no tool was executed or installed."
            if unavailable_tool_names
            else "No tools are registered for planning; no tool was executed or installed."
        )
        gap = {
            "code": code,
            "source": "planner_registry_validation",
            "summary": summary,
            "requested_tools": requested_tools,
            "unavailable_tool_count": len(unavailable_tool_names),
            "registered_tools": registered_tools,
            "execution_attempted": False,
            "acquisition_attempted": False,
            "permissions_granted": False,
        }
        state.recovery_state = {"reason": "capability_unavailable", "capability_gap": gap}
        result = {
            "task_id": state.task_id,
            "goal": state.goal,
            "sources": [],
            "errors": ["capability_unavailable"],
            "attempts": 0,
            "step_results": [],
            "observations": [],
            "step_outcomes": [],
            "final_step_outcomes": [],
            "verification": self._task_verification([]),
            "diagnoses": [],
            "recovery_attempts": [],
            "security_decisions": [],
            "capability_gap": gap,
            "final_status": "BLOCKED",
        }
        state.result = result
        state.context["attempts"] = 0
        state.checkpoint(
            "CAPABILITY_GAP_DETECTED",
            status="BLOCKED",
            phase="CAPABILITY_GAP",
            successful=False,
        )
        self._save_checkpoint(state, "CAPABILITY_GAP_DETECTED")
        self.store.save_event(state.task_id, "capability_gap_detected", gap)
        self._emit_observability(
            "TASK_BLOCKED",
            task_id=state.task_id,
            severity="ERROR",
            metadata={
                "reason": "capability_unavailable",
                "gap_code": code,
                "unavailable_tool_count": len(unavailable_tool_names),
                "execution_attempted": False,
                "acquisition_attempted": False,
            },
        )
        self.store.save_task(
            state.task_id,
            state.goal,
            "blocked",
            {"goal": state.goal, "rationale": "Planning stopped at the registered-tool boundary", "steps": []},
            result,
            0,
        )
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
        self._emit_observability(
            "TASK_BLOCKED",
            task_id=task_id,
            severity="ERROR",
            metadata={
                "reason": "security_policy_blocked",
                "blocked_action_count": len(decisions),
            },
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
            if self.security is None:
                self._emit_observability(
                    "TASK_RESUMED",
                    task_id=state.task_id,
                    metadata={"terminal_state": True},
                )
            return state.result or self._resume_failure(task_id, "terminal_state_missing_result")

        def valid_persisted_action_record(key: str, record: Any) -> bool:
            if not isinstance(key, str) or not isinstance(record, dict):
                return False
            status = record.get("status")
            if status not in ("NOT_STARTED", "EXECUTING", "OBSERVING", "VERIFIED", "FAILED"):
                return False
            if record.get("action_key") != key:
                return False
            for field in ("step_id", "tool_name", "input_fingerprint", "updated_at"):
                if not isinstance(record.get(field), str):
                    return False
            if status == "NOT_STARTED":
                return (
                    key in state.pending_actions
                    and key not in state.completed_actions
                    and key not in state.failed_actions
                    and record.get("tool_name") == ""
                    and record.get("input_fingerprint") == ""
                    and record.get("execution_status") in (None, "")
                    and record.get("output") is None
                    and record.get("observation") is None
                    and record.get("verification") is None
                )
            if not all(record.get(field) for field in ("step_id", "tool_name", "input_fingerprint")):
                return False
            if record.get("input_fingerprint") != key:
                return False
            if status == "EXECUTING":
                return (
                    key in state.pending_actions
                    and key not in state.completed_actions
                    and record.get("execution_status") in (None, "")
                    and record.get("output") is None
                    and record.get("observation") is None
                    and record.get("verification") is None
                    and key not in state.observations
                    and key not in state.verification_results
                )

            execution_status = record.get("execution_status")
            if execution_status not in ("success", "failed", "blocked"):
                return False
            try:
                step_result = self._step_result_from_dict(record.get("output"))
            except (KeyError, TypeError, ValueError, AttributeError):
                return False
            if (
                step_result is None
                or step_result.step_id != record.get("step_id")
                or step_result.tool_name != record.get("tool_name")
                or step_result.status != execution_status
            ):
                return False
            if status == "OBSERVING":
                observation = state.observations.get(key)
                if (
                    key not in state.pending_actions
                    or key in state.completed_actions
                    or record.get("observation") is not None
                    or record.get("verification") is not None
                    or key in state.verification_results
                ):
                    return False
                if observation is not None and (
                    not isinstance(observation, dict)
                    or observation.get("task_id") != state.task_id
                    or observation.get("step_id") != record.get("step_id")
                    or observation.get("tool_name") != record.get("tool_name")
                    or observation.get("execution_status") != execution_status
                    or observation.get("observed_output") != step_result.output
                ):
                    return False
                return True

            if status == "VERIFIED":
                if key not in state.completed_actions:
                    return False
            elif key not in state.failed_actions or key in state.completed_actions:
                return False
            observation = record.get("observation")
            verification = record.get("verification")
            if (
                not isinstance(observation, dict)
                or observation != state.observations.get(key)
                or observation.get("task_id") != state.task_id
                or observation.get("step_id") != record.get("step_id")
                or observation.get("tool_name") != record.get("tool_name")
                or observation.get("execution_status") != execution_status
                or observation.get("observed_output") != step_result.output
                or not isinstance(verification, dict)
                or verification != state.verification_results.get(key)
                or verification.get("status") not in ("VERIFIED", "FAILED", "INSUFFICIENT", "BLOCKED")
                or verification.get("verified") is not (verification.get("status") == "VERIFIED")
            ):
                return False
            if status == "VERIFIED":
                return (
                    execution_status == "success"
                    and step_result.error is None
                    and observation.get("observed_error") is None
                    and verification.get("status") == "VERIFIED"
                )
            return True

        invalid_action_record_keys = [
            key for key, record in state.action_records.items()
            if not valid_persisted_action_record(key, record)
        ]
        if invalid_action_record_keys:
            for key in invalid_action_record_keys:
                state.action_records[key] = {
                    "action_key": key,
                    "status": "FAILED",
                    "execution_status": "failed",
                    "step_id": "",
                    "tool_name": "",
                    "input_fingerprint": "",
                    "output": None,
                    "observation": None,
                    "verification": None,
                    "approval_id": "",
                    "updated_at": state.updated_at,
                }
                state.completed_actions = [item for item in state.completed_actions if item != key]
                state.pending_actions = [item for item in state.pending_actions if item != key]
                if key not in state.failed_actions:
                    state.failed_actions.append(key)
            state.recovery_state = {
                **state.recovery_state,
                "invalid_action_record_keys": invalid_action_record_keys,
            }
            state.touch()
            if state.status == "EXECUTING":
                state.transition("EXECUTION_UNKNOWN", phase="EXECUTION_UNKNOWN", step_id=state.current_step)
            elif state.status == "WAITING_APPROVAL":
                state.transition("BLOCKED", phase="RESUME_BLOCKED")
            reason = (
                "ambiguous_execution_state_invalid_action_record"
                if state.status == "EXECUTION_UNKNOWN"
                else "invalid_persisted_action_record"
            )
            return self._resume_blocked(state, reason)

        self.recovery.attempts.clear()
        self._audit_task_event(state, "TASK_RESUMED", "Task state loaded after process restart")
        if self.security is None:
            self._emit_observability("TASK_RESUMED", task_id=state.task_id, metadata={"terminal_state": False})
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
                if self.security is None:
                    self._emit_observability(
                        "EXECUTION_UNKNOWN",
                        task_id=state.task_id,
                        action_id=record.get("step_id", ""),
                        step_id=record.get("step_id", ""),
                        severity="ERROR",
                        metadata={"reason": "external_side_effect_may_have_occurred"},
                    )
                resolution = unknown_resolutions.get(key) or unknown_resolutions.get(record.get("step_id", ""))
                if resolution is None:
                    return self._resume_blocked(state, "ambiguous_execution_state_requires_observation")
                if not self._resolve_unknown(state, key, record, resolution):
                    return self._resume_blocked(state, "unknown_execution_resolution_not_safe")
            if state.status == "EXECUTION_UNKNOWN":
                state.transition("READY", phase="UNKNOWN_RESOLVED")
                self._save_checkpoint(state, "UNKNOWN_RESOLVED")

        if state.status == "OBSERVING":
            state.checkpoint("RESUME_OBSERVING", phase="RESUME_OBSERVING")
            self._save_checkpoint(state, "RESUME_OBSERVING")
        elif state.status == "VERIFYING":
            state.checkpoint("RESUME_VERIFYING", phase="RESUME_VERIFYING")
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
            self._set_policy_stop_reason(state, blocked)
            self._save_checkpoint(state, "BEFORE_WAITING_APPROVAL" if waiting else "RESUME_BLOCKED")
            return False, decisions
        resumed_lifecycle_status = state.status if state.status in {"OBSERVING", "VERIFYING"} else "READY"
        state.checkpoint(
            "APPROVAL_REVALIDATED",
            status=resumed_lifecycle_status,
            phase="APPROVAL_REVALIDATED",
            successful=True,
        )
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
        binding = resolution.get("binding") if isinstance(resolution, dict) else None
        expected_binding = {
            "task_id": state.task_id,
            "action_key": key,
            "step_id": str(record.get("step_id", "")),
            "tool_name": str(record.get("tool_name", "")),
            "input_fingerprint": str(record.get("input_fingerprint", "")),
        }
        if not isinstance(binding, dict) or any(
            binding.get(name) != value for name, value in expected_binding.items()
        ):
            return False
        decision = str(resolution.get("decision", resolution.get("status", ""))).lower()
        if decision == "retry":
            # A caller-controlled ``observed_not_executed`` flag is not proof
            # that a side effect did not happen.  Until a trusted execution
            # boundary can issue a task/action-bound attestation, retrying an
            # EXECUTION_UNKNOWN action remains fail-closed.
            return False
        if decision == "verified":
            # No trusted attestation channel exists for externally supplied
            # observations yet.  Binding alone is integrity, not authenticity;
            # never promote caller-provided evidence to VERIFIED/COMPLETED.
            return False
        return False

    def _resume_execute_state(self, state: TaskState, current_plan: TaskPlan) -> dict[str, Any]:
        completed_actions = set(state.completed_actions)
        verified_previous_results: dict[str, StepResult] = {}
        for step in current_plan.steps:
            key = state.action_key_for_step(step)
            record = state.action_records.get(key)
            record_observation = record.get("observation") if isinstance(record, dict) else None
            state_observation = state.observations.get(key)
            if key not in completed_actions:
                if isinstance(record, dict) and record.get("status") == "VERIFIED":
                    return self._resume_blocked(state, "persisted_action_completion_conflict")
                continue
            record_verification = record.get("verification") if isinstance(record, dict) else None
            state_verification = state.verification_results.get(key)
            try:
                step_result = self._step_result_from_dict(record.get("output")) if isinstance(record, dict) else None
            except (KeyError, TypeError, ValueError):
                step_result = None
            if (
                not isinstance(record, dict)
                or record.get("action_key") != key
                or record.get("status") != "VERIFIED"
                or record.get("step_id") != step.id
                or record.get("tool_name") != step.tool
                or record.get("input_fingerprint") != key
                or record.get("execution_status") != "success"
                or not isinstance(record_verification, dict)
                or record_verification.get("status") != "VERIFIED"
                or record_verification.get("verified") is not True
                or state_verification != record_verification
                or step_result is None
                or step_result.step_id != step.id
                or step_result.tool_name != step.tool
                or step_result.status != "success"
                or step_result.error is not None
                or not isinstance(record_observation, dict)
                or record_observation != state_observation
                or record_observation.get("task_id") != state.task_id
                or record_observation.get("step_id") != step.id
                or record_observation.get("tool_name") != step.tool
                or record_observation.get("execution_status") != "success"
                or record_observation.get("observed_error") is not None
                or record_observation.get("observed_output") != step_result.output
            ):
                return self._resume_blocked(state, "persisted_verified_action_untrusted")
            verified_previous_results[step.id] = step_result

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
        max_rounds = max(rounds + 1, 1 + self.recovery.max_recovery_attempts) + MAX_CONVERSATION_REPLANS
        if self.store.has_active_lease_context(state.task_id):
            current_plan, updated_decisions, conversation_blocked, conversation_updated = self._maybe_replan_from_followups(
                state,
                current_plan,
                prior=prior,
                retrieved=retrieved_payload,
                reflection_context=reflection_context,
                learned_context=learned_context,
            )
            if conversation_blocked is not None:
                return conversation_blocked
            if conversation_updated:
                security_decisions = [item.to_dict() for item in updated_decisions]
                verified_previous_results = {}
                for step in current_plan.steps:
                    step_result = self._verified_result_from_action_record(state, step)
                    if step_result is not None:
                        verified_previous_results[step.id] = step_result
                latest = self._verified_outcomes_for_plan(state, current_plan)
                outcome_by_step = {item.step_id: item for item in all_outcomes}
                outcome_by_step.update(latest)
                all_outcomes = list(outcome_by_step.values())
        while rounds < max_rounds:
            rounds += 1
            recovery_requested = False
            latest_outcomes = {
                outcome.step_id: outcome for outcome in all_outcomes
                if outcome.verification_status == "VERIFIED"
            }
            pending_steps = [
                step for step in self._ordered_plan_steps(current_plan)
                if state.action_key_for_step(step) not in set(state.completed_actions)
            ]
            if not pending_steps:
                terminal_status = "COMPLETED"
                break
            context = ExecutionContext(
                task_id=state.task_id,
                previous_results={},
                approved_permissions=set(self.approved_permissions),
                metadata={"attempt": rounds, "goal": current_plan.goal},
                security_controller=self.security,
                approval_ids=state.approval_ids,
                observability=self.observability,
            )
            for step in pending_steps:
                key = state.action_key_for_step(step)
                record = state.action_records.get(key, {})
                self._emit_observability(
                    "ACTION_PROPOSED",
                    task_id=state.task_id,
                    action_id=step.id,
                    step_id=step.id,
                    metadata={"tool_name": step.tool, "round": rounds, "resumed": True},
                )
                if record.get("status") == "OBSERVING" and isinstance(record.get("output"), dict):
                    step_result = self._step_result_from_dict(record["output"])
                    if step_result is None:
                        return self._resume_blocked(state, "persisted_execution_result_corrupted")
                else:
                    state.observations.pop(key, None)
                    state.verification_results.pop(key, None)
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
                        execution_status="",
                        output=None,
                        observation=None,
                        verification=None,
                    )
                    state.context["attempts"] = rounds
                    state.checkpoint("BEFORE_EXECUTION", status="EXECUTING", phase="BEFORE_EXECUTION", step_id=step.id)
                    self._save_checkpoint(state, "BEFORE_EXECUTION")
                    self._emit_observability(
                        "ACTION_STARTED",
                        task_id=state.task_id,
                        action_id=step.id,
                        step_id=step.id,
                        metadata={"tool_name": step.tool, "round": rounds, "resumed": True},
                    )
                    step_context = self._isolated_execution_context(context, verified_previous_results)
                    step_result = self._execute_one(step, step_context)
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
                self._emit_observability(
                    "ACTION_COMPLETED" if step_result.status == "success" else "ACTION_LIFECYCLE_BLOCKED" if step_result.status == "blocked" else "ACTION_FAILED",
                    task_id=state.task_id,
                    action_id=step.id,
                    step_id=step.id,
                    severity="INFO" if step_result.status == "success" else "ERROR",
                    metadata={
                        "execution_status": step_result.status,
                        "verification_status": outcome.verification_status,
                        "error_category": step_result.error.code if step_result.error else "",
                        "duration_ms": step_result.metadata.get("duration_ms", 0),
                        "resumed": True,
                    },
                )
                if outcome.verification_status == "VERIFIED" and outcome.execution_status == "success":
                    verified_previous_results[step.id] = step_result
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
                    current_plan, updated_decisions, conversation_blocked, conversation_updated = self._maybe_replan_from_followups(
                        state,
                        current_plan,
                        prior=prior,
                        retrieved=retrieved_payload,
                        reflection_context=reflection_context,
                        learned_context=learned_context,
                    )
                    if conversation_blocked is not None:
                        return conversation_blocked
                    if conversation_updated:
                        security_decisions = [item.to_dict() for item in updated_decisions]
                        verified_previous_results = {}
                        for current_step in current_plan.steps:
                            previous_result = self._verified_result_from_action_record(state, current_step)
                            if previous_result is not None:
                                verified_previous_results[current_step.id] = previous_result
                        latest = self._verified_outcomes_for_plan(state, current_plan)
                        outcome_by_step = {item.step_id: item for item in all_outcomes}
                        outcome_by_step.update(latest)
                        all_outcomes = list(outcome_by_step.values())
                        recovery_requested = True
                        break
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
                self._save_checkpoint_after(state, "AFTER_VERIFICATION", step.id, status="FAILED")
                if diagnosis is None:
                    diagnosis = FailureDiagnosis(
                        "UNKNOWN_FAILURE", "No diagnosis was produced for an unsuccessful step",
                        failed_step=step.id, tool_name=step.tool, recoverable=False, suggested_recovery="ABORT",
                    )
                recovery_started = monotonic()
                self._emit_observability(
                    "RECOVERY_STARTED",
                    task_id=state.task_id,
                    action_id=step.id,
                    step_id=step.id,
                    metadata={"failure_type": diagnosis.failure_type, "resumed": True},
                )
                decision = self.recovery.choose(
                    step, diagnosis, available_tools=self.tools.describe(), allow_replan=self.replanner.provider is not None
                )
                attempt = self.recovery.record(decision, diagnosis, tool_name=step.tool, inputs=step.input)
                if decision.strategy == "ABORT":
                    self.recovery.update_last(outcome=decision.terminal_status or "FAILED", verification_status=outcome.verification_status)
                    terminal_status = decision.terminal_status or "FAILED"
                    errors.append(decision.reason)
                    self._emit_observability(
                        "RECOVERY_COMPLETED",
                        task_id=state.task_id,
                        action_id=step.id,
                        step_id=step.id,
                        severity="ERROR",
                        metadata={"strategy": decision.strategy, "duration_ms": round((monotonic() - recovery_started) * 1000, 3), "resumed": True},
                    )
                    break
                self._emit_observability(
                    "REPLAN_STARTED",
                    task_id=state.task_id,
                    action_id=step.id,
                    step_id=step.id,
                    metadata={"strategy": decision.strategy, "resumed": True},
                )
                try:
                    current_plan = self._apply_decision(current_plan, step, decision, diagnosis, experience)
                except ReplanError as exc:
                    terminal_status = "BLOCKED"
                    errors.append(str(exc))
                    break
                state.set_plan(self._plan_payload(current_plan), version=state.plan_version + 1)
                verified_previous_results.clear()
                self._emit_observability(
                    "PLAN_UPDATED",
                    task_id=state.task_id,
                    metadata={"plan_version": state.plan_version, "step_count": len(current_plan.steps), "resumed": True},
                )
                self._emit_observability(
                    "REPLAN_COMPLETED",
                    task_id=state.task_id,
                    action_id=step.id,
                    step_id=step.id,
                    metadata={"strategy": decision.strategy, "duration_ms": round((monotonic() - recovery_started) * 1000, 3), "resumed": True},
                )
                self._emit_observability(
                    "RECOVERY_COMPLETED",
                    task_id=state.task_id,
                    action_id=step.id,
                    step_id=step.id,
                    metadata={"strategy": decision.strategy, "duration_ms": round((monotonic() - recovery_started) * 1000, 3), "resumed": True},
                )
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
        if self._task_verification_allows_completion(verification_payload):
            terminal_status = "COMPLETED"
        elif terminal_status == "COMPLETED":
            terminal_status = "FAILED"
        experience.final_outcome = terminal_status
        experience.verification_status = str(verification_payload["status"])
        experience.verification_scope = str(verification_payload.get("verification_scope", ""))
        impact_verified = verification_payload.get("security_impact_verified")
        experience.security_impact_verified = impact_verified if isinstance(impact_verified, bool) else None
        experience.identity = {}
        experience.build_identity()
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
            state.set_stop_reason(
                TaskStopReasonCode.TASK_FAILED,
                step_id=state.current_step or None,
                action_id=self._action_id_for_step(state, state.current_step) if state.current_step else None,
            )
            state.checkpoint("TASK_FAILED", status="FAILED", phase="TASK_FAILED")
            self._save_checkpoint(state, "TASK_FAILED")
        self.experience_memory.store(experience)
        self._emit_observability(
            "REFLECTION_STARTED",
            task_id=state.task_id,
            metadata={"action_count": len(experience.actions), "resumed": True},
        )
        reflection_insights = self.reflection_engine.reflect_for_experience(experience)
        result["reflection_insights"] = [insight.to_dict() for insight in reflection_insights]
        self._emit_observability(
            "REFLECTION_COMPLETED",
            task_id=state.task_id,
            metadata={"insight_count": len(reflection_insights), "resumed": True},
        )
        self._emit_observability(
            "LEARNING_STARTED",
            task_id=state.task_id,
            metadata={"insight_count": len(reflection_insights), "resumed": True},
        )
        learned_strategies = self.learning_engine.learn(reflection_insights)
        result["learned_strategies"] = [strategy.to_dict() for strategy in learned_strategies]
        self._emit_observability(
            "LEARNING_COMPLETED",
            task_id=state.task_id,
            metadata={"strategy_count": len(learned_strategies), "resumed": True},
        )
        if not learned_strategies:
            self._emit_observability(
                "LEARNING_REJECTED",
                task_id=state.task_id,
                severity="NOTICE",
                metadata={"reason": "no_eligible_strategy", "resumed": True},
            )
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
        execution_unknown = state.status == "EXECUTION_UNKNOWN"
        waiting = state.status == "WAITING_APPROVAL"
        unresolved_unknown = "ambiguous_execution_state" in reason
        if not waiting and not unresolved_unknown:
            state.transition("BLOCKED", phase="RESUME_BLOCKED")
        state.recovery_state = {**state.recovery_state, "reason": reason}
        if execution_unknown:
            state.set_stop_reason(
                TaskStopReasonCode.EXECUTION_UNKNOWN,
                step_id=state.current_step or None,
                action_id=self._action_id_for_step(state, state.current_step) if state.current_step else None,
            )
        elif decisions:
            self._set_policy_stop_reason(state, decisions)
        elif waiting:
            # Keep the state reason absent when resume failed for a different,
            # unclassified cause; do not claim approval is still the blocker.
            state.set_stop_reason(None)
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
        if self.security is None:
            self._emit_observability(
                "RESUME_BLOCKED",
                task_id=state.task_id,
                step_id=state.current_step,
                severity="ERROR",
                metadata={"reason": reason},
            )
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
            verification_status=str(payload.get("verification_status", "")),
            verification_scope=str(payload.get("verification_scope", "")),
            security_impact_verified=payload.get("security_impact_verified") if isinstance(payload.get("security_impact_verified"), bool) else None,
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

    def _save_checkpoint(
        self,
        state: TaskState,
        checkpoint_type: str,
        *,
        consume_message_ids: list[str] | tuple[str, ...] = (),
    ) -> None:
        state.validate()
        self.store.save_task_state(state, checkpoint_type, consume_message_ids=consume_message_ids)
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
        self._audit_task_event(state, "CHECKPOINT_SAVED", checkpoint_type)
        if self.security is None:
            if special_event:
                self._emit_observability(
                    special_event,
                    task_id=state.task_id,
                    step_id=state.current_step,
                    severity="ERROR" if special_event == "TASK_FAILED" else "INFO",
                    metadata={"checkpoint_type": checkpoint_type},
                )
            self._emit_observability(
                "CHECKPOINT_SAVED",
                task_id=state.task_id,
                step_id=state.current_step,
                metadata={
                    "checkpoint_type": checkpoint_type,
                    "state_version": state.state_version,
                    "state_fingerprint": state.state_fingerprint,
                },
            )
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
        decision = self.security.policy_engine.evaluate(action)
        self.security.audit.record(
            event_type,
            action,
            decision=state.status,
            reason=reason,
            risk_level=decision.risk_level,
            policy_version=decision.policy_version,
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
        observation_started = monotonic()
        self._emit_observability(
            "OBSERVATION_STARTED",
            task_id=task_id,
            action_id=step.id,
            step_id=step.id,
            metadata={"tool_name": step.tool},
        )
        observation = Observation.from_step_result(task_id, step_result)
        self.store.save_observation(observation)
        self.store.save_event(task_id, "observation_created", observation.to_dict(), step_id=step.id)
        self._emit_observability(
            "OBSERVATION_COMPLETED",
            task_id=task_id,
            action_id=step.id,
            step_id=step.id,
            metadata={
                "tool_name": step.tool,
                "execution_status": step_result.status,
                "duration_ms": round((monotonic() - observation_started) * 1000, 3),
            },
        )
        verifier_name = self._verifier_for_step(step.tool, step.verifier)
        criteria = self._criteria_for_step(step.tool, step.success_criteria)
        self.store.save_event(task_id, "verification_started", {"verifier": verifier_name, "criteria": criteria}, step_id=step.id)
        verification_started = monotonic()
        self._emit_observability(
            "VERIFICATION_STARTED",
            task_id=task_id,
            action_id=step.id,
            step_id=step.id,
            metadata={"verifier": verifier_name},
        )
        verification = self.verifiers.verify(verifier_name, observation, criteria)
        self.store.save_verification(task_id, step.id, step.tool, verification)
        self.store.save_event(task_id, "verification_completed", verification.to_dict(), step_id=step.id)
        self._emit_observability(
            "VERIFICATION_PASSED" if verification.status == "VERIFIED" else "VERIFICATION_FAILED",
            task_id=task_id,
            action_id=step.id,
            step_id=step.id,
            severity="INFO" if verification.status == "VERIFIED" else "ERROR",
            metadata={
                "status": verification.status,
                "evidence_count": len(verification.evidence),
                "duration_ms": round((monotonic() - verification_started) * 1000, 3),
            },
        )
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

    @staticmethod
    def _ordered_plan_steps(plan: TaskPlan) -> list[PlanStep]:
        ordered, error = GenericExecutor._dependency_order(plan.steps)
        if error is not None:
            raise StateIntegrityError(f"invalid_plan_dependencies:{error}")
        return ordered

    def _verified_result_from_action_record(self, state: TaskState, step: PlanStep) -> StepResult | None:
        key = state.action_key_for_step(step)
        record = state.action_records.get(key)
        verification = record.get("verification") if isinstance(record, dict) else None
        record_observation = record.get("observation") if isinstance(record, dict) else None
        if (
            key not in state.completed_actions
            or not isinstance(record, dict)
            or record.get("action_key") != key
            or record.get("status") != "VERIFIED"
            or record.get("step_id") != step.id
            or record.get("tool_name") != step.tool
            or record.get("input_fingerprint") != key
            or record.get("execution_status") != "success"
            or not isinstance(verification, dict)
            or verification.get("status") != "VERIFIED"
            or verification.get("verified") is not True
            or state.verification_results.get(key) != verification
            or not isinstance(record_observation, dict)
            or record_observation != state.observations.get(key)
            or record_observation.get("task_id") != state.task_id
            or record_observation.get("step_id") != step.id
            or record_observation.get("tool_name") != step.tool
            or record_observation.get("execution_status") != "success"
            or record_observation.get("observed_error") is not None
        ):
            return None
        result = self._step_result_from_dict(record.get("output"))
        if (
            result is None
            or result.step_id != step.id
            or result.tool_name != step.tool
            or result.status != "success"
            or result.error is not None
            or record_observation.get("observed_output") != result.output
        ):
            return None
        return result

    def _isolated_execution_context(
        self,
        base: ExecutionContext,
        verified_results: dict[str, StepResult],
    ) -> ExecutionContext:
        previous_results: dict[str, StepResult] = {}
        for step_id, result in verified_results.items():
            if (
                not isinstance(result, StepResult)
                or result.step_id != step_id
                or result.status != "success"
                or result.error is not None
            ):
                raise StateIntegrityError("verified_dependency_result_identity_mismatch")
            try:
                payload = json.loads(json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False))
            except (TypeError, ValueError) as exc:
                raise StateIntegrityError("verified_dependency_result_not_json_safe") from exc
            copied = self._step_result_from_dict(payload)
            if (
                copied is None
                or copied.step_id != step_id
                or copied.tool_name != result.tool_name
                or copied.status != "success"
                or copied.error is not None
            ):
                raise StateIntegrityError("verified_dependency_result_copy_invalid")
            previous_results[step_id] = copied
        try:
            limits = json.loads(json.dumps(base.limits, ensure_ascii=False, allow_nan=False))
            metadata = json.loads(json.dumps(base.metadata, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise StateIntegrityError("execution_context_not_json_safe") from exc
        return ExecutionContext(
            task_id=base.task_id,
            previous_results=previous_results,
            approved_permissions=set(base.approved_permissions),
            limits=limits,
            metadata=metadata,
            security_controller=self.security,
            approval_ids=dict(base.approval_ids),
            observability=base.observability,
        )

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
    def _task_verification_allows_completion(payload: dict[str, Any]) -> bool:
        if payload.get("verified") is True:
            return True
        details = payload.get("details", {})
        step_statuses = details.get("step_statuses", []) if isinstance(details, dict) else []
        return (
            payload.get("status") == "INSUFFICIENT"
            and payload.get("verification_scope") == "STATIC_ONLY"
            and payload.get("security_impact_verified") is False
            and isinstance(details, dict)
            and details.get("code") == "security_impact_not_verified"
            and details.get("verification_scope") == "STATIC_ONLY"
            and details.get("security_impact_verified") is False
            and isinstance(step_statuses, list)
            and bool(step_statuses)
            and len(step_statuses) == details.get("step_count")
            and all(status == "VERIFIED" for status in step_statuses)
        )

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
        limited_scope_steps: list[StepOutcome] = []
        limited_scopes: list[str] = []
        for outcome in outcomes:
            verification = outcome.verification if isinstance(outcome.verification, dict) else {}
            details = verification.get("details", {})
            if isinstance(details, dict) and details.get("security_impact_verified") is False:
                scope = details.get("verification_scope")
                if isinstance(scope, str) and scope:
                    limited_scope_steps.append(outcome)
                    limited_scopes.append(scope)
        limited_scope_completion = status == "VERIFIED" and bool(limited_scope_steps)
        if limited_scope_completion:
            status = "INSUFFICIENT"
        scope_label = (
            "STATIC_ONLY"
            if limited_scope_steps and all(scope.startswith("static_") for scope in limited_scopes)
            else "LIMITED"
            if limited_scope_steps
            else "TASK_CRITERIA"
        )
        details_payload: dict[str, Any] = {"step_count": len(outcomes), "step_statuses": statuses}
        if limited_scope_steps:
            details_payload.update(
                {
                    "verification_scope": scope_label,
                    "security_impact_verified": False,
                    "limited_step_ids": [outcome.step_id for outcome in limited_scope_steps],
                }
            )
        if limited_scope_completion:
            details_payload["code"] = "security_impact_not_verified"
        reason = (
            "Step checks passed within a limited static-only scope; security impact was not verified"
            if limited_scope_completion
            else f"Step verification statuses: {', '.join(statuses)}"
        )
        payload = {
            "status": status,
            "verified": status == "VERIFIED",
            "reason": reason,
            "evidence": [{"step_id": outcome.step_id, "status": outcome.verification_status} for outcome in outcomes],
            "verifier_type": "orchestrator",
            "verification_scope": scope_label,
            "details": details_payload,
            "criteria": {},
            "errors": [],
        }
        if limited_scope_steps:
            payload["security_impact_verified"] = False
        return payload

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
