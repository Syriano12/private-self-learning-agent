from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from private_agent.core.execution import ExecutionContext, GenericExecutor
from private_agent.core.planner import PlanStep, TaskPlan
from private_agent.core.task_state import TaskState
from private_agent.security import SecurityController
from private_agent.storage import Store
import private_agent.tools.patching as patching
from private_agent.tools.patching import compute_local_revision
from private_agent.tools.research import ToolRegistry
from private_agent.tools.repository import RepositoryInspector, register_repository_tools


@pytest.fixture
def patch_repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src" / "demo").mkdir(parents=True)
    (root / "src" / "demo" / "service.py").write_text(
        "from pathlib import Path\n"
        "\n"
        "def validate_token(access_token):\n"
        "    return bool(access_token)\n"
        "\n"
        "def login(request):\n"
        "    Path('EXECUTED_BY_PATCH_TOOL').write_text('must never appear')\n"
        "    return validate_token(token=request.token)\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("A bounded patch fixture.\n", encoding="utf-8")
    return root


def make_registry(root: Path, workspace_root: Path):
    inspector = RepositoryInspector(local_repositories={"example/demo": root})
    registry = ToolRegistry()
    register_repository_tools(registry, inspector, patch_workspace_root=workspace_root)
    return registry, inspector


def test_workspace_root_cannot_overlap_configured_source_tree(patch_repository: Path, tmp_path: Path):
    inspector = RepositoryInspector(local_repositories={"example/demo": patch_repository})
    with pytest.raises(ValueError, match="patch_workspace_root_must_be_separate"):
        register_repository_tools(ToolRegistry(), inspector, patch_workspace_root=patch_repository)


def patch_text() -> str:
    return (
        "--- a/src/demo/service.py\n"
        "+++ b/src/demo/service.py\n"
        "@@ -6,3 +6,3 @@\n"
        " def login(request):\n"
        "     Path('EXECUTED_BY_PATCH_TOOL').write_text('must never appear')\n"
        "-    return validate_token(token=request.token)\n"
        "+    return validate_token(request.token)\n"
    )


def prepare_inputs(root: Path, **overrides):
    base_revision = overrides.pop("base_revision", None) or compute_local_revision(root)
    values = {
        "repository": "example/demo",
        "base_revision": base_revision,
        "bug_report": "login fails because validate_token receives the wrong keyword argument",
        "permitted_files": ["src/demo/service.py"],
        "patch": patch_text(),
    }
    values.update(overrides)
    return values


def step(step_id: str, tool: str, inputs: dict) -> PlanStep:
    return PlanStep(step_id, f"Run {tool}", tool, inputs, "The owner supplied an explicit bounded patch operation")


def run_tool(executor: GenericExecutor, step_id: str, tool: str, inputs: dict, **context):
    return executor.execute_step(step(step_id, tool, inputs), ExecutionContext(**context))


def test_prepare_validates_patch_and_preserves_source(patch_repository: Path, tmp_path: Path):
    registry, _ = make_registry(patch_repository, tmp_path / "agent-workspaces")
    before = hashlib.sha256((patch_repository / "src/demo/service.py").read_bytes()).hexdigest()
    result = run_tool(GenericExecutor(registry, SecurityController()), "prepare", "prepare_repository_patch", prepare_inputs(patch_repository))

    assert result.status == "success"
    assert result.output["status"] == "PREPARED"
    assert result.output["validation"] == {"structural": "passed", "target_code_executed": False, "tests_run": []}
    assert result.output["changed_files"] == ["src/demo/service.py"]
    assert hashlib.sha256((patch_repository / "src/demo/service.py").read_bytes()).hexdigest() == before
    assert not (patch_repository / "EXECUTED_BY_PATCH_TOOL").exists()


def test_prepare_links_patch_to_current_phase15b_evidence(patch_repository: Path, tmp_path: Path):
    registry, _ = make_registry(patch_repository, tmp_path / "agent-workspaces")
    evidence_path = patch_repository / "src/demo/service.py"
    evidence = {
        "path": "src/demo/service.py",
        "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "line_start": 9,
        "line_end": 9,
    }
    diagnosis = {
        "repository": "example/demo",
        "revision": "",
        "root_cause": {"status": "ESTABLISHED", "kind": "static_call_signature_mismatch"},
        "evidence": [evidence],
    }
    result = run_tool(
        GenericExecutor(registry, SecurityController()),
        "prepare",
        "prepare_repository_patch",
        prepare_inputs(patch_repository, diagnosis=diagnosis),
    )

    assert result.status == "success"
    assert result.output["diagnosis"]["available"] is True
    assert result.output["diagnosis"]["evidence_refs"][0]["sha256"] == evidence["sha256"]


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"permitted_files": ["README.md"]}, "patch_scope_exceeded"),
        ({"permitted_files": ["node_modules/service.py"]}, "excluded_path"),
        ({"base_revision": "local-tree:" + "0" * 64}, "base_revision_mismatch"),
        ({"patch": patch_text().replace("src/demo/service.py", "../outside.py")}, "invalid_path"),
        ({"patch": patch_text().replace("request.token", "'ghp_" + "12345678901234567890" + "'")}, "patch_contains_sensitive_literal"),
    ],
)
def test_prepare_rejects_unsafe_scope_revision_and_sensitive_patch(patch_repository: Path, tmp_path: Path, overrides, expected):
    registry, _ = make_registry(patch_repository, tmp_path / "agent-workspaces")
    result = run_tool(
        GenericExecutor(registry, SecurityController()),
        "prepare",
        "prepare_repository_patch",
        prepare_inputs(patch_repository, **overrides),
    )
    assert result.status in {"blocked", "failed"}
    assert result.error.code == expected
    assert not list((tmp_path / "agent-workspaces").glob("*/workspace"))


def test_prepare_rejects_stale_diagnosis_evidence(patch_repository: Path, tmp_path: Path):
    registry, _ = make_registry(patch_repository, tmp_path / "agent-workspaces")
    diagnosis = {
        "repository": "example/demo",
        "evidence": [{"path": "src/demo/service.py", "sha256": "0" * 64}],
    }
    result = run_tool(
        GenericExecutor(registry, SecurityController()),
        "prepare",
        "prepare_repository_patch",
        prepare_inputs(patch_repository, diagnosis=diagnosis),
    )
    assert result.status == "blocked"
    assert result.error.code == "diagnosis_evidence_stale"


def test_prepare_rejects_ambiguous_hunk_and_symlink_repository(patch_repository: Path, tmp_path: Path):
    registry, _ = make_registry(patch_repository, tmp_path / "agent-workspaces")
    malformed = patch_text().replace("@@ -6,3 +6,3 @@", "@@ -60,3 +60,3 @@")
    bad = run_tool(
        GenericExecutor(registry, SecurityController()),
        "prepare",
        "prepare_repository_patch",
        prepare_inputs(patch_repository, patch=malformed),
    )
    assert bad.status == "failed"
    assert bad.error.code in {"patch_context_ambiguous", "patch_context_mismatch"}

    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    revision_before_symlink = compute_local_revision(patch_repository)
    (patch_repository / "src/demo/link.py").symlink_to(outside)
    symlink_result = run_tool(
        GenericExecutor(registry, SecurityController()),
        "prepare",
        "prepare_repository_patch",
        prepare_inputs(
            patch_repository,
            base_revision=revision_before_symlink,
            permitted_files=["src/demo/link.py"],
            patch=patch_text().replace("src/demo/service.py", "src/demo/link.py"),
        ),
    )
    assert symlink_result.status in {"blocked", "failed"}
    assert symlink_result.error.code in {"symlink_in_repository", "symlink_path"}


def test_prepare_supports_a_structurally_different_json_fixture(tmp_path: Path):
    root = tmp_path / "config-repo"
    root.mkdir()
    config = root / "settings.json"
    config.write_text('{"enabled": false, "mode": "safe"}\n', encoding="utf-8")
    inspector = RepositoryInspector(local_repositories={"example/config": root})
    registry = ToolRegistry()
    register_repository_tools(registry, inspector, patch_workspace_root=tmp_path / "agent-workspaces")
    revision = compute_local_revision(root)
    diff = (
        "--- a/settings.json\n"
        "+++ b/settings.json\n"
        "@@ -1,1 +1,1 @@\n"
        '-{"enabled": false, "mode": "safe"}\n'
        '+{"enabled": true, "mode": "safe"}\n'
    )
    result = run_tool(
        GenericExecutor(registry, SecurityController()),
        "prepare",
        "prepare_repository_patch",
        {
            "repository": "example/config",
            "base_revision": revision,
            "bug_report": "The feature flag is disabled in the reviewed configuration",
            "permitted_files": ["settings.json"],
            "patch": diff,
        },
    )
    assert result.status == "success"
    assert result.output["validation"]["structural"] == "passed"
    assert config.read_text(encoding="utf-8") == '{"enabled": false, "mode": "safe"}\n'


def test_prepare_enforces_patch_size_limit_without_creating_a_session(patch_repository: Path, tmp_path: Path):
    registry, _ = make_registry(patch_repository, tmp_path / "agent-workspaces")
    huge = "--- a/src/demo/service.py\n+++ b/src/demo/service.py\n" + ("+x\n" * 30000)
    result = run_tool(
        GenericExecutor(registry, SecurityController()),
        "prepare",
        "prepare_repository_patch",
        prepare_inputs(patch_repository, patch=huge),
    )
    assert result.status == "failed"
    assert result.error.code == "patch_size_exceeded"
    assert not list((tmp_path / "agent-workspaces").glob("*/metadata.json"))


def test_prepare_bounds_permitted_files_and_diagnosis_size(patch_repository: Path, tmp_path: Path):
    registry, _ = make_registry(patch_repository, tmp_path / "agent-workspaces")
    too_many_files = run_tool(
        GenericExecutor(registry, SecurityController()),
        "prepare",
        "prepare_repository_patch",
        prepare_inputs(patch_repository, permitted_files=[f"src/demo/f{index}.py" for index in range(9)]),
    )
    assert too_many_files.status == "blocked"
    assert too_many_files.error.code == "file_count_exceeded"

    oversized_diagnosis = run_tool(
        GenericExecutor(registry, SecurityController()),
        "prepare",
        "prepare_repository_patch",
        prepare_inputs(patch_repository, diagnosis={"root_cause": {"details": "x" * 70000}}),
    )
    assert oversized_diagnosis.status == "blocked"
    assert oversized_diagnosis.error.code == "diagnosis_size_exceeded"
    assert not list((tmp_path / "agent-workspaces").glob("*/metadata.json"))


def test_apply_requires_bound_owner_approval_and_writes_only_isolated_workspace(patch_repository: Path, tmp_path: Path):
    store = Store(str(tmp_path / "patch.sqlite3"))
    registry, _ = make_registry(patch_repository, tmp_path / "agent-workspaces")
    controller = SecurityController(store=store)
    executor = GenericExecutor(registry, controller)
    prepared = run_tool(executor, "prepare", "prepare_repository_patch", prepare_inputs(patch_repository))
    patch_id = prepared.output["patch_id"]

    first = run_tool(
        executor,
        "apply",
        "apply_repository_patch",
        {"patch_id": patch_id},
        approved_permissions={"public_read", "isolated_write"},
        task_id="task-apply",
        metadata={"goal": "Apply the approved isolated patch"},
    )
    assert first.status == "blocked"
    assert first.error.code == "permission_not_approved"
    pending = store.all_approval_requests()
    assert len(pending) == 1
    approval_id = pending[0]["approval_id"]
    controller.approval_gate.approve(approval_id, actor="owner")

    applied = run_tool(
        executor,
        "apply",
        "apply_repository_patch",
        {"patch_id": patch_id},
        approved_permissions={"public_read", "isolated_write"},
        task_id="task-apply",
        metadata={"goal": "Apply the approved isolated patch"},
        approval_ids={"apply": approval_id},
    )
    assert applied.status == "success"
    assert applied.output["status"] == "APPLIED"
    assert applied.output["workspace"] == f"{patch_id}/workspace"
    assert (tmp_path / "agent-workspaces" / patch_id / "workspace/src/demo/service.py").read_text(encoding="utf-8").endswith("return validate_token(request.token)\n")
    assert (patch_repository / "src/demo/service.py").read_text(encoding="utf-8").endswith("return validate_token(token=request.token)\n")
    assert not (patch_repository / "EXECUTED_BY_PATCH_TOOL").exists()
    store.close()


def test_direct_mutating_patch_tool_calls_fail_closed_without_workspace_side_effects(patch_repository: Path, tmp_path: Path):
    store = Store(str(tmp_path / "direct-call.sqlite3"))
    workspace_root = tmp_path / "agent-workspaces"
    registry, _ = make_registry(patch_repository, workspace_root)
    controller = SecurityController(store=store)
    executor = GenericExecutor(registry, controller)
    prepared = run_tool(executor, "prepare", "prepare_repository_patch", prepare_inputs(patch_repository))
    patch_id = prepared.output["patch_id"]
    direct_context = ExecutionContext(
        task_id="direct-call",
        approved_permissions={"public_read", "isolated_write"},
        metadata={"goal": "Apply the approved isolated patch"},
        security_controller=controller,
    )

    direct_apply = registry.get("apply_repository_patch").execute({"patch_id": patch_id}, direct_context)

    assert direct_apply.status == "blocked"
    assert direct_apply.error is not None
    assert direct_apply.error.code == "security_execution_required"
    assert not list(workspace_root.glob("*/workspace"))

    first = run_tool(
        executor,
        "apply",
        "apply_repository_patch",
        {"patch_id": patch_id},
        approved_permissions={"public_read", "isolated_write"},
        task_id="direct-call",
        metadata={"goal": "Apply the approved isolated patch"},
    )
    approval_id = store.all_approval_requests()[0]["approval_id"]
    controller.approval_gate.approve(approval_id, actor="owner")
    applied = run_tool(
        executor,
        "apply",
        "apply_repository_patch",
        {"patch_id": patch_id},
        approved_permissions={"public_read", "isolated_write"},
        task_id="direct-call",
        metadata={"goal": "Apply the approved isolated patch"},
        approval_ids={"apply": approval_id},
    )
    assert first.status == "blocked"
    assert applied.status == "success"
    workspace_file = workspace_root / patch_id / "workspace/src/demo/service.py"
    before_direct_rollback = workspace_file.read_bytes()

    direct_rollback = registry.get("rollback_repository_patch").execute(
        {"patch_id": patch_id},
        ExecutionContext(
            task_id="direct-call",
            approved_permissions={"public_read", "isolated_write"},
            metadata={"goal": "Rollback the approved patch"},
            security_controller=controller,
        ),
    )

    assert direct_rollback.status == "blocked"
    assert direct_rollback.error is not None
    assert direct_rollback.error.code == "security_execution_required"
    assert workspace_file.read_bytes() == before_direct_rollback
    assert "token=request.token" not in workspace_file.read_text(encoding="utf-8")
    assert "token=request.token" in (patch_repository / "src/demo/service.py").read_text(encoding="utf-8")
    store.close()


def test_owner_deny_blocks_apply_before_isolated_workspace_write(patch_repository: Path, tmp_path: Path):
    store = Store(str(tmp_path / "deny.sqlite3"))
    registry, _ = make_registry(patch_repository, tmp_path / "agent-workspaces")
    store.create_owner_permission_rule(
        {"rule_id": "deny-patch", "effect": "DENY", "scope_type": "TOOL", "scope_key": "apply_repository_patch", "conditions": {}},
        actor="owner",
        reason="Disable patch application for this test",
    )
    controller = SecurityController(store=store)
    executor = GenericExecutor(registry, controller)
    prepared = run_tool(executor, "prepare", "prepare_repository_patch", prepare_inputs(patch_repository))
    result = run_tool(
        executor,
        "apply",
        "apply_repository_patch",
        {"patch_id": prepared.output["patch_id"]},
        approved_permissions={"public_read", "isolated_write"},
        task_id="deny-task",
        metadata={"goal": "Apply patch"},
    )
    assert result.status == "blocked"
    assert result.error.code == "security_policy_blocked"
    assert not list((tmp_path / "agent-workspaces").glob("*/workspace"))
    store.close()


def test_rollback_refuses_concurrent_workspace_change_and_never_touches_source(patch_repository: Path, tmp_path: Path):
    store = Store(str(tmp_path / "rollback.sqlite3"))
    registry, _ = make_registry(patch_repository, tmp_path / "agent-workspaces")
    controller = SecurityController(store=store)
    executor = GenericExecutor(registry, controller)
    prepared = run_tool(executor, "prepare", "prepare_repository_patch", prepare_inputs(patch_repository))
    patch_id = prepared.output["patch_id"]
    first = run_tool(executor, "apply", "apply_repository_patch", {"patch_id": patch_id}, approved_permissions={"public_read", "isolated_write"}, task_id="rollback-task", metadata={"goal": "Apply patch"})
    approval = store.all_approval_requests()[0]["approval_id"]
    controller.approval_gate.approve(approval, actor="owner")
    applied = run_tool(executor, "apply", "apply_repository_patch", {"patch_id": patch_id}, approved_permissions={"public_read", "isolated_write"}, task_id="rollback-task", metadata={"goal": "Apply patch"}, approval_ids={"apply": approval})
    assert applied.status == "success"

    workspace_file = tmp_path / "agent-workspaces" / patch_id / "workspace/src/demo/service.py"
    workspace_file.write_text(workspace_file.read_text(encoding="utf-8") + "# concurrent review edit\n", encoding="utf-8")
    (tmp_path / "agent-workspaces" / patch_id / "workspace/extra.txt").write_text("unexpected", encoding="utf-8")
    rollback_first = run_tool(executor, "rollback", "rollback_repository_patch", {"patch_id": patch_id}, approved_permissions={"public_read", "isolated_write"}, task_id="rollback-task", metadata={"goal": "Rollback patch"})
    assert rollback_first.status == "blocked"
    rollback_approval = store.all_approval_requests()[-1]["approval_id"]
    controller.approval_gate.approve(rollback_approval, actor="owner")
    rollback = run_tool(executor, "rollback", "rollback_repository_patch", {"patch_id": patch_id}, approved_permissions={"public_read", "isolated_write"}, task_id="rollback-task", metadata={"goal": "Rollback patch"}, approval_ids={"rollback": rollback_approval})
    assert rollback.status == "blocked"
    assert rollback.error.code == "rollback_conflict"
    assert "token=request.token" in (patch_repository / "src/demo/service.py").read_text(encoding="utf-8")
    store.close()


def test_rollback_restores_an_existing_empty_file(patch_repository: Path, tmp_path: Path):
    empty_file = patch_repository / "empty.py"
    empty_file.write_bytes(b"")
    diff = (
        "--- a/empty.py\n"
        "+++ b/empty.py\n"
        "@@ -0,0 +1,1 @@\n"
        "+# temporary review content\n"
    )
    registry, _ = make_registry(patch_repository, tmp_path / "agent-workspaces")
    store = Store(str(tmp_path / "empty.sqlite3"))
    controller = SecurityController(store=store)
    executor = GenericExecutor(registry, controller)
    prepared = run_tool(
        executor,
        "prepare",
        "prepare_repository_patch",
        prepare_inputs(patch_repository, permitted_files=["empty.py"], patch=diff),
    )
    patch_id = prepared.output["patch_id"]
    first = run_tool(
        executor,
        "apply",
        "apply_repository_patch",
        {"patch_id": patch_id},
        approved_permissions={"public_read", "isolated_write"},
        task_id="empty-file-task",
        metadata={"goal": "Apply patch"},
    )
    approval_id = store.all_approval_requests()[0]["approval_id"]
    controller.approval_gate.approve(approval_id, actor="owner")
    applied = run_tool(
        executor,
        "apply",
        "apply_repository_patch",
        {"patch_id": patch_id},
        approved_permissions={"public_read", "isolated_write"},
        task_id="empty-file-task",
        metadata={"goal": "Apply patch"},
        approval_ids={"apply": approval_id},
    )
    assert first.status == "blocked"
    assert applied.status == "success"
    rollback_first = run_tool(
        executor,
        "rollback",
        "rollback_repository_patch",
        {"patch_id": patch_id},
        approved_permissions={"public_read", "isolated_write"},
        task_id="empty-file-task",
        metadata={"goal": "Rollback patch"},
    )
    rollback_approval = store.all_approval_requests()[-1]["approval_id"]
    controller.approval_gate.approve(rollback_approval, actor="owner")
    rolled_back = run_tool(
        executor,
        "rollback",
        "rollback_repository_patch",
        {"patch_id": patch_id},
        approved_permissions={"public_read", "isolated_write"},
        task_id="empty-file-task",
        metadata={"goal": "Rollback patch"},
        approval_ids={"rollback": rollback_approval},
    )
    assert rollback_first.status == "blocked"
    assert rolled_back.status == "success"
    assert (tmp_path / "agent-workspaces" / patch_id / "workspace/empty.py").read_bytes() == b""
    assert empty_file.read_bytes() == b""
    store.close()


def test_rollback_refuses_interleaved_write_before_compare_and_swap(patch_repository: Path, tmp_path: Path, monkeypatch):
    registry, _ = make_registry(patch_repository, tmp_path / "agent-workspaces")
    store = Store(str(tmp_path / "rollback-race.sqlite3"))
    controller = SecurityController(store=store)
    executor = GenericExecutor(registry, controller)
    prepared = run_tool(executor, "prepare", "prepare_repository_patch", prepare_inputs(patch_repository))
    patch_id = prepared.output["patch_id"]
    first = run_tool(
        executor,
        "apply",
        "apply_repository_patch",
        {"patch_id": patch_id},
        approved_permissions={"public_read", "isolated_write"},
        task_id="rollback-race-task",
        metadata={"goal": "Apply patch"},
    )
    approval_id = store.all_approval_requests()[0]["approval_id"]
    controller.approval_gate.approve(approval_id, actor="owner")
    applied = run_tool(
        executor,
        "apply",
        "apply_repository_patch",
        {"patch_id": patch_id},
        approved_permissions={"public_read", "isolated_write"},
        task_id="rollback-race-task",
        metadata={"goal": "Apply patch"},
        approval_ids={"apply": approval_id},
    )
    assert first.status == "blocked"
    assert applied.status == "success"
    workspace_file = tmp_path / "agent-workspaces" / patch_id / "workspace/src/demo/service.py"
    original_safe_write = patching._safe_write
    injected = False

    def interleave(root, path, data, *, overwrite, expected_sha256=None):
        nonlocal injected
        if expected_sha256 is not None and not injected:
            (root / path).write_text("# interleaved review write\n", encoding="utf-8")
            injected = True
        return original_safe_write(root, path, data, overwrite=overwrite, expected_sha256=expected_sha256)

    monkeypatch.setattr(patching, "_safe_write", interleave)
    rollback_first = run_tool(
        executor,
        "rollback",
        "rollback_repository_patch",
        {"patch_id": patch_id},
        approved_permissions={"public_read", "isolated_write"},
        task_id="rollback-race-task",
        metadata={"goal": "Rollback patch"},
    )
    rollback_approval = store.all_approval_requests()[-1]["approval_id"]
    controller.approval_gate.approve(rollback_approval, actor="owner")
    rollback = run_tool(
        executor,
        "rollback",
        "rollback_repository_patch",
        {"patch_id": patch_id},
        approved_permissions={"public_read", "isolated_write"},
        task_id="rollback-race-task",
        metadata={"goal": "Rollback patch"},
        approval_ids={"rollback": rollback_approval},
    )
    assert rollback_first.status == "blocked"
    assert rollback.status == "blocked"
    assert rollback.error.code == "rollback_conflict"
    assert workspace_file.read_text(encoding="utf-8") == "# interleaved review write\n"
    assert "token=request.token" in (patch_repository / "src/demo/service.py").read_text(encoding="utf-8")
    store.close()


def test_orchestrator_persists_patch_metadata_and_never_executes_target_code(patch_repository: Path, tmp_path: Path):
    from private_agent.core.orchestrator import Orchestrator

    registry, _ = make_registry(patch_repository, tmp_path / "agent-workspaces")
    store = Store(str(tmp_path / "orchestrator.sqlite3"))
    controller = SecurityController(store=store)
    inputs = prepare_inputs(patch_repository)

    class StaticPlanner:
        max_steps = 12

        def create(self, goal, tools, prior_knowledge, **kwargs):
            del tools, prior_knowledge, kwargs
            planned = step("prepare", "prepare_repository_patch", inputs)
            planned.success_criteria = {"required_fields": ["status", "patch_id", "validation"]}
            return TaskPlan(goal, [planned], "Explicit patch preparation")

    agent = Orchestrator(
        store,
        registry,
        planner=StaticPlanner(),
        security_controller=controller,
        approved_permissions={"public_read", "isolated_write"},
    )
    result = agent.run("Prepare a minimal patch for the diagnosed bug")
    assert result["final_status"] == "COMPLETED"
    task_state = TaskState.from_dict(store.get_task_state(result["task_id"]))
    output = task_state.result["step_results"][0]["output"]
    assert output["status"] == "PREPARED"
    assert output["validation"]["target_code_executed"] is False
    assert not (patch_repository / "EXECUTED_BY_PATCH_TOOL").exists()
    store.close()


def test_remote_repository_patch_is_rejected_without_local_write_authority(tmp_path: Path):
    local_fixture = tmp_path / "local-fixture"
    local_fixture.mkdir()
    inspector = RepositoryInspector(local_repositories={"example/demo": local_fixture})
    registry = ToolRegistry()
    register_repository_tools(registry, inspector, patch_workspace_root=tmp_path / "agent-workspaces")
    result = run_tool(
        GenericExecutor(registry, SecurityController()),
        "prepare",
        "prepare_repository_patch",
        {
            "repository": "other/remote",
            "base_revision": "local-tree:" + "0" * 64,
            "bug_report": "remote bug",
            "permitted_files": ["src/demo/service.py"],
            "patch": patch_text(),
        },
    )
    assert result.status == "blocked"
    assert result.error.code == "modification_source_unsupported"
