from __future__ import annotations

import base64

import httpx
import pytest

from private_agent.tools.github_access import (
    GitHubAccess,
    RepositoryAccessError,
    parse_repository,
    validate_repository_path,
)




def response(request: httpx.Request, status: int = 200, *, json=None, content: bytes | None = None, headers=None):
    return httpx.Response(status, json=json, content=content, headers=headers, request=request)


def public_routes(content: bytes = b"hello"):
    encoded = base64.b64encode(content).decode()
    return {
        "/repos/acme/demo": {"private": False, "default_branch": "main"},
        "/repos/acme/demo/commits/main": {"sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
        "/repos/acme/demo/git/trees/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": {
            "truncated": False,
            "tree": [{"path": "README.md", "type": "blob", "size": len(content), "sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}],
        },
        "/repos/acme/demo/git/blobs/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb": {"encoding": "base64", "content": encoded},
    }


def test_parse_repository_accepts_canonical_forms_and_rejects_ambiguous_identifiers() -> None:
    assert parse_repository("acme/demo") == "acme/demo"
    assert parse_repository("https://github.com/acme/demo") == "acme/demo"
    assert parse_repository("https://github.com/acme/demo.git") == "acme/demo"
    for value in (
        "http://github.com/acme/demo",
        "https://evil.example/acme/demo",
        "https://user:pass@github.com/acme/demo",
        "https://github.com/acme/demo?secret=1",
        "https://github.com/acme/demo#fragment",
        "https://github.com/acme/%64emo",
        "acme/../demo",
        "acme\\demo",
        "acme/demo.git",
    ):
        with pytest.raises(RepositoryAccessError) as raised:
            parse_repository(value)
        assert raised.value.code == "invalid_repository"
        assert "secret" not in str(raised.value)


def test_validate_repository_path_is_relative_and_forbids_controls_and_git_metadata() -> None:
    assert validate_repository_path("src/main.py") == "src/main.py"
    for value in ("/src/main.py", "../main.py", "src/../main.py", "src\\main.py", "src/%2e/main.py", ".git/config", "src/.git/config", "src\x00main.py"):
        with pytest.raises(RepositoryAccessError) as raised:
            validate_repository_path(value)
        assert raised.value.code == "invalid_path"


def test_environment_is_explicit_and_repr_does_not_expose_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_READ_TOKEN", "super-secret")
    monkeypatch.setenv("GITHUB_READ_REPOSITORIES", " acme/private, https://github.com/acme/second.git ")
    monkeypatch.setenv("GH_TOKEN", "ambient-secret")
    access = GitHubAccess.from_environment()
    assert access.authorized_private_repositories == frozenset({"acme/private", "acme/second"})
    assert "super-secret" not in repr(access)
    assert "ambient-secret" not in repr(access)


def test_public_repository_is_anonymous_pinned_and_read_from_indexed_blob() -> None:
    calls: list[httpx.Request] = []
    routes = public_routes()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        route = request.url.path
        if route.endswith("/git/trees/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"):
            assert request.url.params.get("recursive") == "1"
        payload = routes.get(route)
        assert payload is not None
        return response(request, json=payload)

    view = GitHubAccess(transport=httpx.MockTransport(handler)).open("https://github.com/acme/demo.git", ref="main")
    assert view.metadata() == {
        "repository": "acme/demo",
        "ref": "main",
        "commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "private": False,
        "source": "github_api",
    }
    assert view.tree(limit=200) == {
        "files": [{"path": "README.md", "type": "file", "size": 5, "sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}],
        "truncated": False,
    }
    assert view.read("README.md") == b"hello"
    assert all("authorization" not in request.headers for request in calls)
    assert sum(request.url.path.endswith("/git/trees/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa") for request in calls) == 1


def test_allowlisted_private_repository_uses_token_only_after_anonymous_metadata() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/repos/acme/private":
            return response(request, json={"private": True, "default_branch": "main"})
        if request.url.path.endswith("/commits/main"):
            assert request.headers.get("authorization") == "Bearer read-secret"
            return response(request, json={"sha": "cccccccccccccccccccccccccccccccccccccccc"})
        if request.url.path.endswith("/git/trees/cccccccccccccccccccccccccccccccccccccccc"):
            assert request.headers.get("authorization") == "Bearer read-secret"
            return response(request, json={"truncated": False, "tree": []})
        raise AssertionError(request.url)

    view = GitHubAccess(
        token="read-secret",
        authorized_private_repositories=("acme/private",),
        transport=httpx.MockTransport(handler),
    ).open("acme/private")
    assert view.metadata()["private"] is True
    assert "authorization" not in seen[0].headers
    view.tree()


def test_nonallowlisted_repository_never_receives_token_and_missing_is_safe() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return response(request, status=404, json={"message": "private repository secret"})

    with pytest.raises(RepositoryAccessError) as raised:
        GitHubAccess(token="read-secret", transport=httpx.MockTransport(handler)).open("acme/private")
    assert raised.value.code == "repository_unavailable"
    assert len(seen) == 1
    assert "authorization" not in seen[0].headers
    assert "private repository secret" not in str(raised.value)


def test_private_metadata_is_denied_when_not_allowlisted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return response(request, json={"private": True, "default_branch": "main"})

    with pytest.raises(RepositoryAccessError) as raised:
        GitHubAccess(token="read-secret", transport=httpx.MockTransport(handler)).open("acme/private")
    assert raised.value.code == "private_repository_not_authorized"


def test_ref_is_percent_encoded_and_commit_sha_pins_tree() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(str(request.url))
        if request.url.path == "/repos/acme/demo":
            return response(request, json={"private": False, "default_branch": "main"})
        if request.url.raw_path.split(b"?")[0] == b"/repos/acme/demo/commits/feature%2Fone":
            return response(request, json={"sha": "dddddddddddddddddddddddddddddddddddddddd"})
        if request.url.path == "/repos/acme/demo/git/trees/dddddddddddddddddddddddddddddddddddddddd":
            return response(request, json={"truncated": False, "tree": []})
        raise AssertionError(request.url)

    view = GitHubAccess(transport=httpx.MockTransport(handler)).open("acme/demo", ref="feature/one")
    assert view.metadata()["commit"] == "dddddddddddddddddddddddddddddddddddddddd"
    view.tree()
    assert any("/commits/feature%2Fone" in item for item in paths)


@pytest.mark.parametrize(
    ("status", "headers", "code"),
    [(403, {"x-ratelimit-remaining": "0"}, "rate_limited"), (429, {}, "rate_limited"), (404, {}, "repository_unavailable")],
)
def test_safe_status_errors(status: int, headers: dict[str, str], code: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return response(request, status=status, headers=headers, json={"message": "do not expose me"})

    with pytest.raises(RepositoryAccessError) as raised:
        GitHubAccess(transport=httpx.MockTransport(handler)).open("acme/demo")
    assert raised.value.code == code
    assert "do not expose me" not in str(raised.value)


def test_timeout_malformed_base64_and_response_size_are_safe() -> None:
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("secret timeout detail", request=request)

    with pytest.raises(RepositoryAccessError) as raised:
        GitHubAccess(transport=httpx.MockTransport(timeout_handler)).open("acme/demo")
    assert raised.value.code == "timeout"
    assert "secret timeout detail" not in str(raised.value)

    def malformed_handler(request: httpx.Request) -> httpx.Response:
        return response(request, content=b"not-json")

    with pytest.raises(RepositoryAccessError) as raised:
        GitHubAccess(transport=httpx.MockTransport(malformed_handler)).open("acme/demo")
    assert raised.value.code == "api_error"

    def oversized_handler(request: httpx.Request) -> httpx.Response:
        return response(request, content=b"x" * 11)

    with pytest.raises(RepositoryAccessError) as raised:
        GitHubAccess(max_api_bytes=10, transport=httpx.MockTransport(oversized_handler)).open("acme/demo")
    assert raised.value.code == "response_too_large"


def test_malformed_base64_and_oversized_indexed_file_fail_without_returning_bytes() -> None:
    def malformed_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/acme/demo":
            return response(request, json={"private": False, "default_branch": "main"})
        if request.url.path.endswith("/commits/main"):
            return response(request, json={"sha": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"})
        if request.url.path.endswith("/git/trees/eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"):
            return response(request, json={"truncated": False, "tree": [{"path": "x", "type": "blob", "size": 1, "sha": "ffffffffffffffffffffffffffffffffffffffff"}]})
        return response(request, json={"encoding": "base64", "content": "%%%"})

    view = GitHubAccess(transport=httpx.MockTransport(malformed_handler)).open("acme/demo")
    with pytest.raises(RepositoryAccessError) as raised:
        view.read("x")
    assert raised.value.code == "api_error"

    def oversized_file_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/acme/demo":
            return response(request, json={"private": False, "default_branch": "main"})
        if request.url.path.endswith("/commits/main"):
            return response(request, json={"sha": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"})
        return response(request, json={"truncated": False, "tree": [{"path": "x", "type": "blob", "size": 100, "sha": "ffffffffffffffffffffffffffffffffffffffff"}]})

    view = GitHubAccess(max_file_bytes=10, transport=httpx.MockTransport(oversized_file_handler)).open("acme/demo")
    with pytest.raises(RepositoryAccessError) as raised:
        view.read("x")
    assert raised.value.code == "file_too_large"


def test_redirect_is_not_followed_and_symlink_or_submodule_is_rejected() -> None:
    requests: list[httpx.Request] = []

    def redirect_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response(request, status=302, headers={"location": "https://evil.example/steal"})

    with pytest.raises(RepositoryAccessError) as raised:
        GitHubAccess(transport=httpx.MockTransport(redirect_handler)).open("acme/demo")
    assert raised.value.code == "api_error"
    assert len(requests) == 1

    def unsafe_tree_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/acme/demo":
            return response(request, json={"private": False, "default_branch": "main"})
        if request.url.path.endswith("/commits/main"):
            return response(request, json={"sha": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"})
        return response(request, json={"truncated": False, "tree": [{"path": "link", "type": "symlink", "mode": "120000", "size": 1, "sha": "1111111111111111111111111111111111111111"}]})

    view = GitHubAccess(transport=httpx.MockTransport(unsafe_tree_handler)).open("acme/demo")
    with pytest.raises(RepositoryAccessError) as raised:
        view.tree()
    assert raised.value.code == "api_error"


def test_redact_removes_token_without_exposing_it_as_a_property() -> None:
    access = GitHubAccess(token="super-secret")
    assert access.redact("Authorization: Bearer super-secret") == "Authorization: Bearer [REDACTED]"
    assert "token" not in vars(access)


def test_public_allowlisted_repository_stays_anonymous():
    seen = []
    routes = public_routes()
    def handler(request):
        seen.append(request)
        return response(request, json=routes[request.url.path])
    access = GitHubAccess(token="read-secret", authorized_private_repositories=("acme/demo",),
                          transport=httpx.MockTransport(handler))
    view = access.open("acme/demo")
    view.tree()
    view.read("README.md")
    assert all("authorization" not in request.headers for request in seen)
