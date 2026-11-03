from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from private_agent.core.execution import ExecutionContext, GenericExecutor
from private_agent.core.planner import PlanStep
from private_agent.security import SecurityController
from private_agent.tools.repository import RepositoryInspector, RepositoryLimits, register_repository_tools
from private_agent.tools.research import ToolRegistry


@pytest.fixture
def python_repository(tmp_path: Path) -> Path:
    root = tmp_path / "python-repository"
    (root / "src" / "demo").mkdir(parents=True)
    (root / "src" / "demo" / "service.py").write_text(
        "# café module\n"
        "import os as operating_system\n"
        "from pathlib import Path as PathAlias\n"
        "mapper = lambda target: render(target)\n"
        "\n"
        "@route\n"
        "def greet(café: str, /, count: int = 1, *items: object, label: str = 'x', **options: object) -> str:\n"
        "    return render(café, count)\n"
        "\n"
        "class Service(Base):\n"
        "    @property\n"
        "    def value(self) -> str:\n"
        "        return self._value\n"
        "\n"
        "    def run(self, item: int) -> str:\n"
        "        return helper(\n"
        "            item\n"
        "        )\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("Python AST fixture.\n", encoding="utf-8")
    return root


def _inspect_structure(root: Path, *, limits: RepositoryLimits | None = None):
    inspector = RepositoryInspector(local_repositories={"example/demo": root}, limits=limits)
    result = inspector.inspect("inspect_repository_structure", {"repository": "example/demo"})
    return inspector, result


def _all_ast_records(ast_output: dict):
    for file_record in ast_output["files"]:
        for collection in ("symbols", "imports", "parameters", "decorators", "annotations", "calls", "diagnostics"):
            yield from file_record.get(collection, [])


def test_repository_structure_extracts_python_symbols_and_binds_each_record_to_snapshot(python_repository):
    inspector, result = _inspect_structure(python_repository)
    ast_output = result["data"]["python_ast"]
    snapshot = result["snapshot"]

    assert ast_output["parser_id"] == "python.ast"
    assert ast_output["parser_version"] == f"{sys.version_info.major}.{sys.version_info.minor}"
    module = next(record for record in _all_ast_records(ast_output) if record["kind"] == "module")
    greet = next(record for record in _all_ast_records(ast_output) if record.get("name") == "greet")
    service = next(record for record in _all_ast_records(ast_output) if record.get("name") == "Service")
    method = next(record for record in _all_ast_records(ast_output) if record.get("name") == "run")
    lambda_symbol = next(record for record in _all_ast_records(ast_output) if record.get("kind") == "lambda")
    assert module["parent_id"] is None
    assert greet["parent_id"] == module["symbol_id"]
    assert service["parent_id"] == module["symbol_id"]
    assert method["parent_id"] == service["symbol_id"]
    lambda_parameter = next(item for item in ast_output["files"][0]["parameters"] if item["name"] == "target")
    assert lambda_parameter["parent_id"] == lambda_symbol["symbol_id"]
    assert any(item["target"] == "render" and item["parent_id"] == lambda_symbol["symbol_id"] for item in ast_output["files"][0]["calls"])
    assert greet["return_annotation"] == "str"

    imports = ast_output["files"][0]["imports"]
    assert any(item["module"] == "os" and item["aliases"] == [{"name": "os", "asname": "operating_system"}] for item in imports)
    assert any(item["module"] == "pathlib" and item["level"] == 0 for item in imports)
    parameters = ast_output["files"][0]["parameters"]
    unicode_parameter = next(item for item in parameters if item["name"] == "café")
    assert unicode_parameter["annotation"] == "str"
    assert unicode_parameter["parent_id"] == greet["symbol_id"]
    assert unicode_parameter["column_start"] == len("def greet(") + 1
    assert any(item["name"] == "route" and item["parent_id"] == greet["symbol_id"] for item in ast_output["files"][0]["decorators"])
    assert any(item["target"] == "render" and item["parent_id"] == greet["symbol_id"] for item in ast_output["files"][0]["calls"])
    multiline_call = next(item for item in ast_output["files"][0]["calls"] if item["target"] == "helper")
    assert multiline_call["line_start"] < multiline_call["line_end"]
    assert multiline_call["column_start"] > multiline_call["column_end"]

    records = list(_all_ast_records(ast_output))
    assert records
    for record in records:
        reference = inspector.validate_evidence_ref(record["evidence_ref"], snapshot)
        assert reference.path == record["path"]
        assert reference.line_start == record["line_start"]
        assert reference.line_end == record["line_end"]
        assert reference.parser_id == "python.ast"
        assert reference.parser_version == ast_output["parser_version"]


def test_python_ast_syntax_error_is_reported_without_symbols_or_execution(tmp_path):
    root = tmp_path / "invalid-python"
    root.mkdir()
    marker = root / "executed.marker"
    (root / "broken.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\ndef broken(:\n    pass\n",
        encoding="utf-8",
    )
    inspector, result = _inspect_structure(root)
    file_result = next(item for item in result["data"]["python_ast"]["files"] if item["file_path"] == "broken.py")

    assert file_result["status"] == "syntax_error"
    assert file_result["symbols"] == []
    assert file_result["diagnostics"][0]["line_start"] == 3
    assert inspector.validate_evidence_ref(file_result["diagnostics"][0]["evidence_ref"], result["snapshot"])
    assert not marker.exists()


def test_large_invalid_utf8_redacted_and_empty_files_are_explicitly_skipped(tmp_path, monkeypatch):
    root = tmp_path / "bounded-python"
    root.mkdir()
    (root / "large.py").write_text("x = '" + "a" * 300 + "'\n", encoding="utf-8")
    (root / "invalid.py").write_bytes(b"def invalid():\n\xff\n")
    (root / "empty.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("PHASE2_TEST_API_TOKEN", "phase2-secret-fixture-value")
    (root / "redacted.py").write_text(
        "def use_token():\n    token = 'phase2-secret-fixture-value'\n    return token\n",
        encoding="utf-8",
    )
    inspector, result = _inspect_structure(root, limits=RepositoryLimits(max_file_bytes=128))
    ast_output = result["data"]["python_ast"]
    by_path = {item["file_path"]: item for item in ast_output["files"]}

    assert by_path["large.py"]["status"] == "unavailable"
    assert by_path["large.py"]["reason"] == "file_too_large"
    assert by_path["invalid.py"]["status"] == "unavailable"
    assert by_path["invalid.py"]["reason"] == "binary_file"
    assert by_path["redacted.py"]["status"] == "redacted_source_not_parsed"
    assert by_path["redacted.py"]["symbols"] == []
    assert by_path["empty.py"]["status"] == "empty_source"
    assert "phase2-secret-fixture-value" not in str(result)
    assert all(inspector.validate_evidence_ref(record["evidence_ref"], result["snapshot"]) for record in _all_ast_records(ast_output))


def test_ast_snapshot_rejects_stale_evidence_and_invalidates_mid_analysis_change(python_repository, monkeypatch):
    from private_agent.tools.github_access import RepositoryAccessError
    from private_agent.tools.local_repository import LocalRepository

    inspector, result = _inspect_structure(python_repository)
    ast_output = result["data"]["python_ast"]
    function = next(record for record in _all_ast_records(ast_output) if record.get("name") == "greet")
    old_ref = function["evidence_ref"]
    service = python_repository / "src/demo/service.py"
    service.write_text(service.read_text(encoding="utf-8") + "\n# changed after snapshot\n", encoding="utf-8")
    with pytest.raises(RepositoryAccessError) as stale:
        inspector.validate_evidence_ref(old_ref, result["snapshot"])
    assert stale.value.code == "source_changed_during_analysis"

    root = python_repository
    source_path = root / "src/demo/service.py"
    original_read = LocalRepository.read
    changed = False

    def change_during_read(view, path, *, max_bytes=None):
        nonlocal changed
        payload = original_read(view, path, max_bytes=max_bytes)
        if path == "src/demo/service.py" and not changed:
            changed = True
            source_path.write_text(source_path.read_text(encoding="utf-8") + "\n# race\n", encoding="utf-8")
        return payload

    monkeypatch.setattr(LocalRepository, "read", change_during_read)
    with pytest.raises(RepositoryAccessError) as during:
        inspector.inspect("inspect_repository_structure", {"repository": "example/demo"})
    assert during.value.code == "source_changed_during_analysis"


def test_ast_never_follows_symlink_or_accepts_a_traversal_path(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("class OutsideSecret:\n    pass\n", encoding="utf-8")
    (root / "escape.py").symlink_to(outside)
    inspector = RepositoryInspector(local_repositories={"example/demo": root})

    result = inspector.inspect("inspect_repository_structure", {"repository": "example/demo"})
    names = {record.get("name") for record in _all_ast_records(result["data"]["python_ast"])}
    assert "OutsideSecret" not in names
    assert {"path": "escape.py", "reason": "symlink"} in result["snapshot"]["excluded_files"]
    with pytest.raises(Exception):
        inspector.inspect("read_repository_file", {"repository": "example/demo", "path": "../outside.py"})


def test_ast_record_budget_is_bounded_and_marked_truncated(tmp_path):
    root = tmp_path / "many-python-symbols"
    root.mkdir()
    source = "\n".join(f"def function_{index}(value):\n    return call_{index}(value)" for index in range(120)) + "\n"
    (root / "many.py").write_text(source, encoding="utf-8")
    limits = RepositoryLimits(max_response_bytes=131072)
    _, result = _inspect_structure(root, limits=limits)
    ast_output = result["data"]["python_ast"]

    assert ast_output["truncated"] is True
    assert ast_output["record_count"] <= 40
    assert result["snapshot"]["analysis_truncated"] is True


def test_bug_investigator_includes_ast_evidence_without_promoting_it_to_a_finding(python_repository):
    inspector = RepositoryInspector(local_repositories={"example/demo": python_repository})
    registry = ToolRegistry()
    register_repository_tools(registry, inspector)
    executor = GenericExecutor(registry, SecurityController())
    step = PlanStep(
        "phase2-investigate",
        "Read-only investigation using repository evidence",
        "investigate_repository_bug",
        {"repository": "example/demo", "bug_report": "An intermittent background failure is reported; no trace is available."},
        "Read-only diagnostic request",
        success_criteria={"required_fields": ["root_cause", "evidence", "python_ast"]},
    )
    result = executor.execute_step(step, ExecutionContext(metadata={"goal": "Investigate a bug in example/demo"}))

    assert result.status == "success"
    report = result.output
    assert report["root_cause"]["status"] == "UNRESOLVED"
    assert report["python_ast"]["files"]
    assert any(record["kind"] == "class" and record["name"] == "Service" for record in report["python_ast"]["files"][0]["symbols"])
    assert all(
        inspector.validate_evidence_ref(record["evidence_ref"], report["snapshot"])
        for record in _all_ast_records(report["python_ast"])
    )
    assert not [finding for finding in report["findings"] if finding.get("kind") == "ast_symbol"]
