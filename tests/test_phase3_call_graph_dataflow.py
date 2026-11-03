from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from private_agent.tools.repository import RepositoryInspector


@pytest.fixture
def flow_repository(tmp_path: Path) -> Path:
    root = tmp_path / "flow-repository"
    package = root / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "service.py").write_text(
        "from .helpers import sanitize, execute\n\n"
        "def handle(request):\n"
        "    raw = request.value\n"
        "    safe = sanitize(raw)\n"
        "    return execute(safe)\n",
        encoding="utf-8",
    )
    (package / "helpers.py").write_text(
        "def sanitize(value):\n"
        "    return value\n\n"
        "def execute(statement):\n"
        "    return statement\n\n"
        "def never_called(value):\n"
        "    return value\n",
        encoding="utf-8",
    )
    return root


def inspect_flow(root: Path) -> tuple[RepositoryInspector, dict]:
    inspector = RepositoryInspector(local_repositories={"example/flow": root})
    result = inspector.inspect("inspect_repository_structure", {"repository": "example/flow"})
    return inspector, result


def complete_evidence_registry(report: dict, snapshot: dict) -> dict:
    from private_agent.tools.repository_snapshot import RepositorySnapshot
    from private_agent.tools.python_ai_contract import expand_analysis_evidence_registry

    return expand_analysis_evidence_registry(
        report["analysis_evidence_registry"], report["python_ast"], RepositorySnapshot.from_dict(snapshot),
    )


def test_bounded_multi_file_analysis_stays_inside_response_limit(tmp_path: Path):
    root = tmp_path / "bounded-multi-file"
    root.mkdir()
    for module in range(6):
        source = "".join(
            f"def function_{module}_{index}():\n    return helper_{module}_{index}()\n"
            for index in range(13)
        )
        (root / f"module_{module}.py").write_text(source, encoding="utf-8")
    inspector = RepositoryInspector(local_repositories={"example/multi-file": root})

    result = inspector.inspect("inspect_repository_structure", {"repository": "example/multi-file"})
    report = result["data"]
    encoded = json.dumps(result, ensure_ascii=False).encode("utf-8")

    assert len(encoded) <= inspector.limits.max_response_bytes
    assert len(report["python_ast"]["files"]) == 6
    assert report["python_ast"]["truncated"] is True
    assert report["call_graph"]["complete"] is False
    assert report["data_flow"]["complete"] is False


def test_structure_resolves_imported_calls_and_binds_edges_to_snapshot(flow_repository: Path):
    inspector, result = inspect_flow(flow_repository)
    report = result["data"]

    assert "call_graph" in report
    graph = report["call_graph"]
    modules = {item["module_name"]: item for item in report["python_ast"]["files"]}
    service_symbols = {item["name"]: item for item in modules["pkg.service"]["symbols"]}
    helper_symbols = {
        item["name"]: item
        for item in modules["pkg.helpers"]["symbols"]
        if item["kind"] in {"function", "async_function", "method", "async_method"}
    }
    handle = service_symbols["handle"]
    sanitize = helper_symbols["sanitize"]
    edge = next(
        item for item in graph["edges"]
        if item["source_symbol_id"] == handle["symbol_id"]
        and item["target_symbol_id"] == sanitize["symbol_id"]
    )
    assert edge["resolution"] == "POTENTIAL_UNIQUE"
    assert edge["runtime_target_confirmed"] is False
    never_called = helper_symbols["never_called"]
    assert not any(
        item.get("source_symbol_id") == handle["symbol_id"]
        and item.get("target_symbol_id") == never_called["symbol_id"]
        for item in graph["edges"]
    )

    full_registry = complete_evidence_registry(report, result["snapshot"])
    registry = {item["evidence_id"]: item["evidence_ref"] for item in full_registry["references"]}
    assert edge["evidence_ids"]
    assert all(inspector.validate_evidence_ref(registry[ref_id], result["snapshot"]) for ref_id in edge["evidence_ids"])
    for item in [*graph["nodes"], *graph["edges"]]:
        assert item["evidence_ids"]
        assert set(item["evidence_ids"]).issubset(registry)


def test_structure_tracks_parameter_argument_and_return_flow_across_functions(flow_repository: Path):
    inspector, result = inspect_flow(flow_repository)
    report = result["data"]

    assert "data_flow" in report
    data_flow = report["data_flow"]
    nodes = {item["node_id"]: item for item in data_flow["nodes"]}
    parameters = {
        item["symbol_id"]: item
        for file in report["python_ast"]["files"]
        for item in file["parameters"]
    }
    helper_functions = {
        item["name"]: item
        for file in report["python_ast"]["files"]
        for item in file["symbols"]
        if item["kind"] in {"function", "async_function", "method", "async_method"}
    }
    sanitize_parameter = next(
        item for item in parameters.values()
        if item["parent_id"] == helper_functions["sanitize"]["symbol_id"]
    )
    assert any(
        edge["kind"] == "argument_to_parameter"
        and edge["target_node_id"] == sanitize_parameter["symbol_id"]
        and edge["certainty"] == "POTENTIAL_INTERPROCEDURAL"
        for edge in data_flow["edges"]
    )
    assert any(
        edge["kind"] == "callee_return_to_call_result"
        and edge["certainty"] == "POTENTIAL_INTERPROCEDURAL"
        for edge in data_flow["edges"]
    )
    assert any(
        node.get("role_candidate") == "SOURCE_CANDIDATE" and node.get("name") == "request"
        for node in nodes.values()
    )
    assert any(
        node.get("role_candidate") == "SINK_CANDIDATE"
        and node.get("kind") == "call_argument"
        and node.get("sink_target_candidate") == "execute"
        for node in nodes.values()
    )
    full_registry = complete_evidence_registry(report, result["snapshot"])
    evidence = {item["evidence_id"]: item["evidence_ref"] for item in full_registry["references"]}
    for item in [*data_flow["nodes"], *data_flow["edges"]]:
        assert item["evidence_ids"]
        assert all(inspector.validate_evidence_ref(evidence[key], result["snapshot"]) for key in item["evidence_ids"])


def test_sink_candidate_is_attached_to_argument_with_a_source_to_sink_path(flow_repository: Path):
    _, result = inspect_flow(flow_repository)
    flow = result["data"]["data_flow"]
    nodes = {item["node_id"]: item for item in flow["nodes"]}
    sources = [item["node_id"] for item in flow["nodes"] if item.get("role_candidate") == "SOURCE_CANDIDATE" and item.get("name") == "request"]
    sinks = [item["node_id"] for item in flow["nodes"] if item.get("role_candidate") == "SINK_CANDIDATE" and item.get("kind") == "call_argument" and item.get("sink_target_candidate") == "execute"]

    assert sources
    assert sinks
    assert not any(item.get("role_candidate") == "SINK_CANDIDATE" and item.get("kind") == "call_result" for item in flow["nodes"])
    adjacency: dict[str, list[str]] = {}
    for edge in flow["edges"]:
        adjacency.setdefault(edge["source_node_id"], []).append(edge["target_node_id"])
    reachable = set(sources)
    pending = list(sources)
    while pending:
        current = pending.pop()
        for target in adjacency.get(current, []):
            if target not in reachable:
                reachable.add(target)
                pending.append(target)
    assert any(sink in reachable for sink in sinks), nodes


def test_dense_flow_node_limit_stops_before_evidence_registry_exhaustion(tmp_path: Path):
    root = tmp_path / "dense-flow"
    root.mkdir()
    arguments = ", ".join(["request"] * 20)
    body = "def handle(request):\n" + "".join(f"    consume({arguments})\n" for _ in range(26))
    (root / "dense.py").write_text(body, encoding="utf-8")
    inspector = RepositoryInspector(local_repositories={"example/dense-flow": root})
    result = inspector.inspect("inspect_repository_structure", {"repository": "example/dense-flow"})

    flow = result["data"]["data_flow"]
    assert len(flow["nodes"]) <= 64
    assert flow["truncated"] is True
    assert flow["complete"] is False
    assert len(result["data"]["analysis_evidence_registry"]["references"]) <= 160


def test_structure_exposes_versioned_provider_neutral_ai_input_without_source_snippets(flow_repository: Path):
    from private_agent.tools.python_ai_contract import materialize_ai_analysis_request

    _, result = inspect_flow(flow_repository)
    report = result["data"]

    assert "ai_analysis_contract" in report
    contract = report["ai_analysis_contract"]
    assert contract["contract_id"] == "vulnora.static_analysis.ai_bundle_ref"
    assert contract["provider_independent"] is True
    assert contract["source_code_included"] is False
    request = materialize_ai_analysis_request(report, result["snapshot"])
    assert request["schema_version"] == 1
    assert request["provider_independent"] is True
    assert request["source_code_included"] is False
    assert request["repository"]["snapshot_id"] == result["snapshot"]["snapshot_id"]
    assert request["call_graph"]["edges"]
    assert request["data_flow"]["edges"]
    assert request["evidence_catalog"]
    assert "def handle" not in str(request)
    assert "content" not in request
    assert "excerpt" not in str(request)


def test_valid_ai_hypothesis_remains_unverified_without_semantic_support(flow_repository: Path):
    from private_agent.tools.python_analysis_bundle import materialize_ai_analysis_request, validate_ai_hypothesis_response

    inspector, result = inspect_flow(flow_repository)
    report = result["data"]
    request = materialize_ai_analysis_request(report, result["snapshot"])
    evidence_id = complete_evidence_registry(report, result["snapshot"])["references"][0]["evidence_id"]
    response = {
        "schema_version": 1,
        "hypotheses": [{
            "hypothesis_id": "h-possible",
            "statement": "The call path may reach a sensitive operation; no exploitability was established.",
            "confidence": 0.99,
            "supporting_evidence_ids": [evidence_id],
            "uncertainties": ["No runtime validation was performed."],
            "missing_evidence": ["A validated source-to-sink security condition."],
        }],
    }
    accepted = validate_ai_hypothesis_response(
        response, request, report, result["snapshot"], inspector,
    )
    hypothesis = accepted["hypotheses"][0]
    assert accepted["trusted"] is False
    assert accepted["authority"] == "ADVISORY_ONLY"
    assert hypothesis["status"] == "UNVERIFIED"
    assert hypothesis["evidence_state"] == "CITATIONS_PRESENT_NOT_SEMANTICALLY_VERIFIED"
    assert hypothesis["finding_promotion_allowed"] is False


def test_ai_response_cannot_include_approval_execution_or_permission_fields(flow_repository: Path):
    from private_agent.tools.python_analysis_bundle import materialize_ai_analysis_request, validate_ai_hypothesis_response

    inspector, result = inspect_flow(flow_repository)
    report = result["data"]
    request = materialize_ai_analysis_request(report, result["snapshot"])
    response = {"schema_version": 1, "hypotheses": [], "approved": True}
    with pytest.raises(ValueError, match="forbidden_authority_field"):
        validate_ai_hypothesis_response(
            response, request, report, result["snapshot"], inspector,
        )


def test_low_level_ai_validator_rejects_forged_request_authority_claims(flow_repository: Path):
    from private_agent.tools.python_analysis_bundle import materialize_ai_analysis_request
    from private_agent.tools.python_ai_contract import validate_ai_hypothesis_response as validate_low_level

    inspector, result = inspect_flow(flow_repository)
    report = result["data"]
    request = materialize_ai_analysis_request(report, result["snapshot"])
    request["execution_authority"] = "shell"
    with pytest.raises(ValueError, match="ai_analysis_authority_claims_invalid"):
        validate_low_level(
            {"schema_version": 1, "hypotheses": []}, request,
            complete_evidence_registry(report, result["snapshot"]), result["snapshot"], inspector,
        )


def test_low_level_ai_validator_rejects_request_content_changed_after_binding(flow_repository: Path):
    from private_agent.tools.python_analysis_bundle import materialize_ai_analysis_request
    from private_agent.tools.python_ai_contract import validate_ai_hypothesis_response as validate_low_level

    inspector, result = inspect_flow(flow_repository)
    report = result["data"]
    request = materialize_ai_analysis_request(report, result["snapshot"])
    request["repository_summary"]["symbol_count"] += 1
    with pytest.raises(ValueError, match="ai_analysis_request_digest_mismatch"):
        validate_low_level(
            {"schema_version": 1, "hypotheses": []}, request,
            complete_evidence_registry(report, result["snapshot"]), result["snapshot"], inspector,
        )


def test_low_level_ai_validator_rejects_forged_compact_evidence_even_with_new_request_id(flow_repository: Path):
    from private_agent.tools.python_analysis_bundle import materialize_ai_analysis_request
    from private_agent.tools.python_ai_contract import validate_ai_hypothesis_response as validate_low_level

    inspector, result = inspect_flow(flow_repository)
    report = result["data"]
    request = materialize_ai_analysis_request(report, result["snapshot"])
    request["evidence_catalog"][0]["file_path"] = "forged/path.py"
    content = dict(request)
    content.pop("request_id")
    request["request_id"] = "ai-request:" + hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:32]

    with pytest.raises(ValueError, match="ai_analysis_evidence_catalog_mismatch"):
        validate_low_level(
            {"schema_version": 1, "hypotheses": []}, request,
            complete_evidence_registry(report, result["snapshot"]), result["snapshot"], inspector,
        )


@pytest.mark.parametrize("shape", ["deep", "large"])
def test_low_level_ai_validator_rejects_unbounded_provider_responses(flow_repository: Path, shape: str):
    from private_agent.tools.python_ai_contract import validate_ai_hypothesis_response as validate_low_level

    inspector, result = inspect_flow(flow_repository)
    report = result["data"]
    from private_agent.tools.python_analysis_bundle import materialize_ai_analysis_request

    request = materialize_ai_analysis_request(report, result["snapshot"])
    if shape == "deep":
        response: dict[str, object] = {"schema_version": 1, "hypotheses": []}
        current = response
        for _ in range(80):
            child: dict[str, object] = {}
            current["nested"] = child
            current = child
    else:
        response = {"schema_version": 1, "hypotheses": [], "padding": "x" * 300000}
    with pytest.raises(ValueError, match="ai_hypothesis_response_(depth_limit|limit)"):
        validate_low_level(
            response, request, complete_evidence_registry(report, result["snapshot"]),
            result["snapshot"], inspector,
        )


@pytest.mark.parametrize("field", ["tool_calls", "permissions", "validated", "finding_status", "authorization"])
def test_ai_response_cannot_smuggle_authority_in_hypothesis_fields(flow_repository: Path, field: str):
    from private_agent.tools.python_analysis_bundle import materialize_ai_analysis_request, validate_ai_hypothesis_response

    inspector, result = inspect_flow(flow_repository)
    report = result["data"]
    request = materialize_ai_analysis_request(report, result["snapshot"])
    response = {
        "schema_version": 1,
        "hypotheses": [{
            "hypothesis_id": "h-1",
            "statement": "A syntactically valid claim remains advisory.",
            "confidence": 0.7,
            "supporting_evidence_ids": [],
            "uncertainties": [],
            "missing_evidence": [],
            field: True,
        }],
    }
    with pytest.raises(ValueError, match="forbidden_authority_field"):
        validate_ai_hypothesis_response(response, request, report, result["snapshot"], inspector)


def test_ai_response_rejects_snapshot_evidence_after_source_changes(flow_repository: Path):
    from private_agent.tools.github_access import RepositoryAccessError
    from private_agent.tools.python_analysis_bundle import materialize_ai_analysis_request, validate_ai_hypothesis_response

    inspector, result = inspect_flow(flow_repository)
    report = result["data"]
    request = materialize_ai_analysis_request(report, result["snapshot"])
    evidence_id = complete_evidence_registry(report, result["snapshot"])["references"][0]["evidence_id"]
    (flow_repository / "pkg" / "helpers.py").write_text(
        (flow_repository / "pkg" / "helpers.py").read_text(encoding="utf-8") + "# changed after analysis\n",
        encoding="utf-8",
    )
    response = {
        "schema_version": 1,
        "hypotheses": [{
            "hypothesis_id": "stale",
            "statement": "A stale reference must not be accepted.",
            "confidence": 0.5,
            "supporting_evidence_ids": [evidence_id],
            "uncertainties": [],
            "missing_evidence": [],
        }],
    }
    with pytest.raises(RepositoryAccessError) as error:
        validate_ai_hypothesis_response(
            response, request, report, result["snapshot"], inspector,
        )
    assert error.value.code == "source_changed_during_analysis"


def test_ai_response_rejects_unissued_evidence_id(flow_repository: Path):
    from private_agent.tools.python_analysis_bundle import materialize_ai_analysis_request, validate_ai_hypothesis_response

    inspector, result = inspect_flow(flow_repository)
    report = result["data"]
    request = materialize_ai_analysis_request(report, result["snapshot"])
    response = {
        "schema_version": 1,
        "hypotheses": [{
            "hypothesis_id": "fabricated-citation",
            "statement": "This citation was not issued for this snapshot.",
            "confidence": 1.0,
            "supporting_evidence_ids": ["evidence:" + ("0" * 64)],
            "uncertainties": [],
            "missing_evidence": [],
        }],
    }
    with pytest.raises(ValueError, match="ai_hypothesis_evidence_reference_invalid"):
        validate_ai_hypothesis_response(response, request, report, result["snapshot"], inspector)


def test_ai_bundle_digest_binds_phase2_ast_base_evidence(flow_repository: Path):
    from copy import deepcopy
    from private_agent.tools.python_analysis_bundle import materialize_ai_analysis_request

    _, result = inspect_flow(flow_repository)
    report = deepcopy(result["data"])
    reference = next(
        symbol["evidence_ref"]
        for file in report["python_ast"]["files"]
        for symbol in file.get("symbols", [])
        if isinstance(symbol.get("evidence_ref"), dict)
    )
    reference["rule_version"] = "tampered-but-schema-valid"
    with pytest.raises(ValueError, match="analysis_evidence_base_mismatch"):
        materialize_ai_analysis_request(report, result["snapshot"])


def test_ai_request_rejects_graph_references_outside_issued_catalog(flow_repository: Path):
    from copy import deepcopy
    from private_agent.tools.python_ai_contract import build_ai_analysis_contract, materialize_ai_analysis_request
    from private_agent.tools.repository_snapshot import RepositorySnapshot

    _, result = inspect_flow(flow_repository)
    report = deepcopy(result["data"])
    report["call_graph"]["nodes"][0]["evidence_ids"] = ["ev:" + ("0" * 32)]
    snapshot = RepositorySnapshot.from_dict(result["snapshot"])
    report["ai_analysis_contract"] = build_ai_analysis_contract(
        snapshot, report["python_ast"], report["call_graph"], report["data_flow"],
        report["analysis_evidence_registry"],
    )
    with pytest.raises(ValueError, match="ai_analysis_evidence_reference_invalid"):
        materialize_ai_analysis_request(report, result["snapshot"])


def test_dynamic_call_stays_unresolved_and_is_never_runtime_confirmed(tmp_path: Path):
    root = tmp_path / "dynamic-repository"
    root.mkdir()
    (root / "dynamic.py").write_text(
        "def invoke(obj):\n"
        "    return getattr(obj, 'run')()\n",
        encoding="utf-8",
    )
    inspector = RepositoryInspector(local_repositories={"example/dynamic": root})
    result = inspector.inspect("inspect_repository_structure", {"repository": "example/dynamic"})
    graph = result["data"]["call_graph"]
    dynamic = [item for item in graph["edges"] if item.get("resolution") == "UNRESOLVED_DYNAMIC"]
    assert dynamic
    assert all(item["runtime_target_confirmed"] is False for item in dynamic)
    assert all(item["target_symbol_id"] is None for item in dynamic)


def test_excessive_ast_depth_is_bounded_and_marks_dataflow_incomplete(tmp_path: Path):
    root = tmp_path / "deep-expression"
    root.mkdir()
    expression = " + ".join("value" for _ in range(180))
    (root / "deep.py").write_text(f"def deep(value):\n    return {expression}\n", encoding="utf-8")
    inspector = RepositoryInspector(local_repositories={"example/deep": root})

    data = inspector.inspect("inspect_repository_structure", {"repository": "example/deep"})["data"]

    assert data["data_flow"]["complete"] is False
    assert data["data_flow"]["truncated"] is True
    assert any(item["kind"] == "ast_depth_limit" for item in data["data_flow"]["unresolved"])


def test_static_analysis_never_executes_repository_module_body(tmp_path: Path):
    root = tmp_path / "non-executed-repository"
    root.mkdir()
    marker = tmp_path / "must-not-exist"
    escaped = str(marker).replace("\\", "\\\\").replace("'", "\\'")
    (root / "bomb.py").write_text(
        f"from pathlib import Path\nPath('{escaped}').write_text('executed')\nraise RuntimeError('must not run')\n"
        "def helper(value):\n    return value\n",
        encoding="utf-8",
    )
    inspector = RepositoryInspector(local_repositories={"example/non-executed": root})
    result = inspector.inspect("inspect_repository_structure", {"repository": "example/non-executed"})

    assert result["data"]["python_ast"]["files"][0]["status"] == "parsed"
    assert not marker.exists()


def test_method_and_external_calls_are_labeled_as_candidates_not_runtime_facts(tmp_path: Path):
    root = tmp_path / "method-repository"
    root.mkdir()
    (root / "service.py").write_text(
        "import json\n"
        "class Service:\n"
        "    def start(self, value):\n"
        "        return self.finish(value)\n"
        "    def finish(self, value):\n"
        "        return json.dumps(value)\n"
        "def dispatch(service, value):\n"
        "    return service.start(value)\n",
        encoding="utf-8",
    )
    inspector = RepositoryInspector(local_repositories={"example/method": root})
    result = inspector.inspect("inspect_repository_structure", {"repository": "example/method"})
    report = result["data"]
    graph = report["call_graph"]
    symbols = {item["symbol_id"]: item for file in report["python_ast"]["files"] for item in file["symbols"]}
    finish = next(item for item in symbols.values() if item["qualname"].endswith("Service.finish"))
    internal = [item for item in graph["edges"] if item.get("target_symbol_id") == finish["symbol_id"]]
    assert any(item["kind"] == "calls_candidate" and item["resolution"] == "POTENTIAL_UNIQUE" for item in internal)
    assert all(item["runtime_target_confirmed"] is False for item in internal)
    assert any(
        item["kind"] == "calls_declared_import_candidate"
        and item["resolution"] == "DECLARED_IMPORT_NOT_IN_ANALYZED_TREE"
        and item["certainty"] == "POTENTIAL_EXTERNAL_OR_UNANALYZED"
        for item in graph["edges"]
    )
    assert any(item.get("resolution") == "UNRESOLVED_STATIC" and item["target_symbol_id"] is None for item in graph["edges"])


def test_import_resolution_handles_src_layout_package_init_and_dotted_imports(tmp_path: Path):
    root = tmp_path / "src-layout"
    package = root / "src" / "demo"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from .helpers import helper as init_helper\n"
        "def entry(value):\n"
        "    return init_helper(value)\n",
        encoding="utf-8",
    )
    (package / "helpers.py").write_text(
        "def helper(value):\n"
        "    return value\n",
        encoding="utf-8",
    )
    (root / "consumer.py").write_text(
        "import demo.helpers\n"
        "def direct(value):\n"
        "    return demo.helpers.helper(value)\n",
        encoding="utf-8",
    )
    inspector = RepositoryInspector(local_repositories={"example/src-layout": root})
    result = inspector.inspect("inspect_repository_structure", {"repository": "example/src-layout"})
    report = result["data"]
    symbols = {item["symbol_id"]: item for file in report["python_ast"]["files"] for item in file["symbols"]}
    helper = next(
        item for item in symbols.values()
        if item["name"] == "helper" and item["path"] == "src/demo/helpers.py"
    )
    callers = {"entry", "direct"}
    resolved_callers = {
        symbols[edge["source_symbol_id"]]["name"]
        for edge in report["call_graph"]["edges"]
        if edge.get("kind") == "calls_candidate" and edge.get("target_symbol_id") == helper["symbol_id"]
    }
    assert callers.issubset(resolved_callers)


def test_syntax_error_file_makes_dataflow_and_ai_request_explicitly_incomplete(tmp_path: Path):
    from private_agent.tools.python_analysis_bundle import materialize_ai_analysis_request

    root = tmp_path / "syntax-error-repository"
    root.mkdir()
    (root / "broken.py").write_text("def broken(:\n    return input()\n", encoding="utf-8")
    inspector = RepositoryInspector(local_repositories={"example/syntax-error": root})
    result = inspector.inspect("inspect_repository_structure", {"repository": "example/syntax-error"})
    report = result["data"]
    assert report["python_ast"]["files"][0]["status"] == "syntax_error"
    assert report["call_graph"]["complete"] is False
    assert report["data_flow"]["complete"] is False
    request = materialize_ai_analysis_request(report, result["snapshot"])
    assert request["repository_summary"]["coverage_complete"] is False
    assert request["call_graph"]["complete"] is False
    assert request["data_flow"]["complete"] is False


def test_large_python_file_is_reported_as_incomplete_not_parsed_past_limits(tmp_path: Path):
    root = tmp_path / "large-repository"
    root.mkdir()
    (root / "large.py").write_text("def sample():\n    return 1\n" + ("# padding\n" * 9000), encoding="utf-8")
    inspector = RepositoryInspector(local_repositories={"example/large": root})
    result = inspector.inspect("inspect_repository_structure", {"repository": "example/large"})
    report = result["data"]
    large_record = next(item for item in report["python_ast"]["files"] if item["file_path"] == "large.py")
    assert large_record["status"] != "parsed"
    assert large_record["truncated"] is True
    assert report["call_graph"]["complete"] is False
    assert report["data_flow"]["complete"] is False
