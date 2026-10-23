from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from private_agent.api import create_app
from private_agent.auth import AuthError, AuthManager, hash_password, verify_password
from private_agent.storage import Store
from tests.test_phase12_api import APITestTool, make_service

OWNER_EMAIL = "ashbm1995@gmail.com"
OWNER_PASSWORD = "owner-password-for-tests-123"
OPERATOR_PASSWORD = "operator-password-for-tests-123"


def _client(service) -> TestClient:
    return TestClient(create_app(service=service), raise_server_exceptions=False)


def _login(client: TestClient, email: str, password: str) -> str:
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    csrf = client.cookies.get("vulnora_csrf")
    assert csrf
    return str(csrf)


def test_scrypt_password_hash_is_verifiable_and_not_plaintext() -> None:
    encoded = hash_password(OWNER_PASSWORD)
    assert encoded.startswith("scrypt$")
    assert OWNER_PASSWORD not in encoded
    assert verify_password(OWNER_PASSWORD, encoded)
    assert not verify_password("wrong-password-123", encoded)


def test_owner_initialization_is_one_time_and_email_bound(tmp_path: Path) -> None:
    store = Store(tmp_path / "auth.sqlite3")
    auth = AuthManager(store)
    with pytest.raises(AuthError) as invalid_email:
        auth.initialize_owner("other@example.com", OWNER_PASSWORD)
    assert invalid_email.value.code == "owner_email_not_allowed"
    created = auth.initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    assert created.role == "OWNER"
    assert created.email == OWNER_EMAIL
    with pytest.raises(AuthError) as duplicate_owner:
        auth.initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    assert duplicate_owner.value.code == "owner_already_initialized"
    with pytest.raises(AuthError) as owner_role:
        auth.create_user("second-owner@example.com", OWNER_PASSWORD, "OWNER")
    assert owner_role.value.code == "role_not_allowed"
    row = store.db.execute("SELECT password_hash FROM auth_users WHERE email = ?", (OWNER_EMAIL,)).fetchone()
    assert row is not None
    assert OWNER_PASSWORD not in str(row["password_hash"])
    store.close()


def test_schema_v7_reopens_with_auth_and_task_ownership_tables(tmp_path: Path) -> None:
    path = tmp_path / "schema-v7.sqlite3"
    store = Store(path)
    versions = [row["schema_version"] for row in store.db.execute("SELECT schema_version FROM task_state_schema ORDER BY schema_version")]
    assert versions == [1, 2, 3, 4, 5, 6, 7]
    assert store.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='auth_users'").fetchone()
    store.close()
    reopened = Store(path)
    assert reopened.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='auth_sessions'").fetchone()
    assert reopened.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='auth_events'").fetchone()
    assert reopened.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='task_owners'").fetchone()
    reopened.close()


def test_session_requires_csrf_and_logout_revokes_cookie(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token=None)
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    client = _client(service)
    assert client.get("/api/v1/tasks").status_code == 401
    csrf = _login(client, OWNER_EMAIL, OWNER_PASSWORD)
    assert client.get("/api/v1/auth/session").json()["user"]["role"] == "OWNER"
    missing_csrf = client.post("/api/v1/tasks", json={"goal": "must fail"})
    assert missing_csrf.status_code == 403
    created = client.post("/api/v1/tasks", json={"goal": "owner task"}, headers={"X-CSRF-Token": csrf})
    assert created.status_code == 201, created.text
    logout = client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": csrf})
    assert logout.status_code == 200
    assert client.get("/api/v1/tasks").status_code == 401
    store.close()


def test_production_session_cookie_flags_keep_session_httponly_and_csrf_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_COOKIE_SECURE", "1")
    monkeypatch.setenv("AGENT_COOKIE_SAMESITE", "lax")
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token=None)
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    response = _client(service).post("/api/v1/auth/login", json={"email": OWNER_EMAIL, "password": OWNER_PASSWORD})
    assert response.status_code == 200
    cookie_headers = response.headers.get_list("set-cookie")
    session_cookie = next(item for item in cookie_headers if item.startswith("vulnora_session="))
    csrf_cookie = next(item for item in cookie_headers if item.startswith("vulnora_csrf="))
    assert "HttpOnly" in session_cookie and "Secure" in session_cookie and "SameSite=lax" in session_cookie
    assert "HttpOnly" not in csrf_cookie and "Secure" in csrf_cookie and "SameSite=lax" in csrf_cookie
    store.close()


def test_rbac_prevents_viewer_mutation_and_owner_only_management(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token=None)
    auth = AuthManager(store)
    auth.initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    owner_client = _client(service)
    owner_csrf = _login(owner_client, OWNER_EMAIL, OWNER_PASSWORD)
    created = owner_client.post(
        "/api/v1/users",
        json={"email": "viewer@example.com", "password": OPERATOR_PASSWORD, "role": "VIEWER"},
        headers={"X-CSRF-Token": owner_csrf},
    )
    assert created.status_code == 201, created.text
    viewer_client = _client(service)
    viewer_csrf = _login(viewer_client, "viewer@example.com", OPERATOR_PASSWORD)
    assert viewer_client.get("/api/v1/metrics").status_code == 403
    assert viewer_client.get("/api/v1/users").status_code == 403
    assert viewer_client.post(
        "/api/v1/tasks",
        json={"goal": "viewer must not create"},
        headers={"X-CSRF-Token": viewer_csrf},
    ).status_code == 403
    user_id = created.json()["user_id"]
    assert viewer_client.patch(
        f"/api/v1/users/{user_id}/role",
        json={"role": "OPERATOR"},
        headers={"X-CSRF-Token": viewer_csrf},
    ).status_code == 403
    assert owner_client.post(
        "/api/v1/users/{}/disable".format(user_id),
        headers={"X-CSRF-Token": owner_csrf},
    ).status_code == 200
    assert owner_client.patch(
        f"/api/v1/users/{user_id}/role",
        json={"role": "OWNER"},
        headers={"X-CSRF-Token": owner_csrf},
    ).status_code == 422
    store.close()


def test_repeated_failures_lock_account_without_logging_password_or_session(tmp_path: Path) -> None:
    store = Store(tmp_path / "lockout.sqlite3")
    auth = AuthManager(store, max_failed_attempts=3, lockout_seconds=30)
    auth.initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    for _ in range(3):
        with pytest.raises(AuthError) as failed_login:
            auth.login(OWNER_EMAIL, "wrong-password-123")
        assert failed_login.value.code == "invalid_credentials"
    with pytest.raises(AuthError) as locked_login:
        auth.login(OWNER_EMAIL, OWNER_PASSWORD)
    assert locked_login.value.code == "invalid_credentials"
    row = store.db.execute("SELECT failed_login_count, locked_until FROM auth_users WHERE email = ?", (OWNER_EMAIL,)).fetchone()
    assert int(row["failed_login_count"]) >= 3
    assert row["locked_until"]
    locked_count = int(row["failed_login_count"])
    locked_until = str(row["locked_until"])
    events = [dict(item) for item in store.db.execute("SELECT * FROM auth_events").fetchall()]
    serialized = json.dumps(events)
    assert "wrong-password-123" not in serialized
    assert OWNER_PASSWORD not in serialized
    assert "session_token" not in serialized
    after_locked_attempt = store.db.execute("SELECT failed_login_count, locked_until FROM auth_users WHERE email = ?", (OWNER_EMAIL,)).fetchone()
    assert int(after_locked_attempt["failed_login_count"]) == locked_count
    assert str(after_locked_attempt["locked_until"]) == locked_until
    store.close()


def test_login_rate_limit_blocks_rotating_unknown_emails_before_password_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_AUTH_MAX_LOGIN_ATTEMPTS", "3")
    monkeypatch.setenv("AGENT_AUTH_RATE_WINDOW_SECONDS", "60")
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token=None)
    client = _client(service)
    for index in range(3):
        failed = client.post("/api/v1/auth/login", json={"email": f"unknown-{index}@example.com", "password": "wrong-password-123"})
        assert failed.status_code == 401
    limited = client.post("/api/v1/auth/login", json={"email": "another-unknown@example.com", "password": "wrong-password-123"})
    assert limited.status_code == 429
    assert limited.json()["error_code"] == "login_rate_limited"
    events = [dict(item) for item in store.db.execute("SELECT * FROM auth_events").fetchall()]
    serialized = json.dumps(events)
    assert "wrong-password-123" not in serialized
    assert "testclient" not in serialized
    store.close()


def test_login_rate_limit_uses_normalized_proxy_client_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_AUTH_MAX_LOGIN_ATTEMPTS", "3")
    monkeypatch.setenv("AGENT_AUTH_RATE_WINDOW_SECONDS", "60")
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token=None)
    client = _client(service)
    headers = {"X-Real-IP": "198.51.100.10"}
    for index in range(3):
        failed = client.post(
            "/api/v1/auth/login",
            json={"email": f"proxy-unknown-{index}@example.com", "password": "wrong-password-123"},
            headers=headers,
        )
        assert failed.status_code == 401
    limited = client.post(
        "/api/v1/auth/login",
        json={"email": "proxy-unknown-fourth@example.com", "password": "wrong-password-123"},
        headers=headers,
    )
    assert limited.status_code == 429
    different_client = client.post(
        "/api/v1/auth/login",
        json={"email": "proxy-unknown-other@example.com", "password": "wrong-password-123"},
        headers={"X-Real-IP": "198.51.100.11"},
    )
    assert different_client.status_code == 401
    store.close()


def test_operator_task_ownership_prevents_cross_user_idor_and_owner_can_audit(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token=None)
    auth = AuthManager(store)
    auth.initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    owner_client = _client(service)
    owner_csrf = _login(owner_client, OWNER_EMAIL, OWNER_PASSWORD)
    first = owner_client.post(
        "/api/v1/users",
        json={"email": "operator-one@example.com", "password": OPERATOR_PASSWORD, "role": "OPERATOR"},
        headers={"X-CSRF-Token": owner_csrf},
    )
    second = owner_client.post(
        "/api/v1/users",
        json={"email": "operator-two@example.com", "password": "operator-two-password-123", "role": "OPERATOR"},
        headers={"X-CSRF-Token": owner_csrf},
    )
    assert first.status_code == second.status_code == 201
    first_client = _client(service)
    first_csrf = _login(first_client, "operator-one@example.com", OPERATOR_PASSWORD)
    created = first_client.post("/api/v1/tasks", json={"goal": "operator one private task"}, headers={"X-CSRF-Token": first_csrf, "Idempotency-Key": "operator-one-private-task"})
    assert created.status_code == 201, created.text
    task_id = created.json()["task_id"]
    assert store.task_owner_user_id(task_id) == first.json()["user_id"]
    second_client = _client(service)
    _login(second_client, "operator-two@example.com", "operator-two-password-123")
    assert second_client.get("/api/v1/tasks").json()["items"] == []
    for suffix in ("", "/timeline", "/metrics"):
        response = second_client.get(f"/api/v1/tasks/{task_id}{suffix}")
        assert response.status_code == 404, response.text
    for path in (
        f"/api/v1/bug-bounty/tasks/{task_id}",
        f"/api/v1/bug-bounty/tasks/{task_id}/evidence",
        f"/api/v1/failures?task_id={task_id}",
        f"/api/v1/security-events?task_id={task_id}",
        f"/api/v1/tool-events?task_id={task_id}",
    ):
        assert second_client.get(path).status_code == 404
    denied_resume = second_client.post(
        f"/api/v1/tasks/{task_id}/resume",
        json={},
        headers={"X-CSRF-Token": second_client.cookies.get("vulnora_csrf")},
    )
    assert denied_resume.status_code == 404
    collision = second_client.post(
        "/api/v1/tasks",
        json={"goal": "operator one private task"},
        headers={"X-CSRF-Token": second_client.cookies.get("vulnora_csrf"), "Idempotency-Key": "operator-one-private-task"},
    )
    assert collision.status_code == 409
    assert collision.json()["error_code"] == "idempotency_conflict"
    assert owner_client.get(f"/api/v1/tasks/{task_id}").status_code == 200
    store.close()


def test_sessions_are_stored_as_hashes_and_expire_without_password_recovery(tmp_path: Path) -> None:
    store = Store(tmp_path / "session-hash.sqlite3")
    auth = AuthManager(store)
    auth.initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    login = auth.login(OWNER_EMAIL, OWNER_PASSWORD)
    assert login.session_token
    assert login.csrf_token
    assert login.session_token not in str(store.db.execute("SELECT * FROM auth_sessions").fetchall())
    session_hash = store.db.execute("SELECT session_hash, csrf_hash FROM auth_sessions").fetchone()
    assert session_hash["session_hash"] != login.session_token
    assert session_hash["csrf_hash"] != login.csrf_token
    store.db.execute("UPDATE auth_sessions SET expires_at = '1970-01-01T00:00:00+00:00'")
    store.db.commit()
    with pytest.raises(AuthError) as expired_session:
        auth.authenticate_session(login.session_token)
    assert expired_session.value.code == "session_expired"
    store.close()


def test_malformed_auth_lock_timestamp_fails_closed_without_internal_error(tmp_path: Path) -> None:
    store = Store(tmp_path / "malformed-auth.sqlite3")
    auth = AuthManager(store)
    auth.initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    store.db.execute("UPDATE auth_users SET locked_until = 'not-a-timestamp'")
    store.db.commit()
    with pytest.raises(AuthError) as malformed:
        auth.login(OWNER_EMAIL, OWNER_PASSWORD)
    assert malformed.value.code == "auth_state_invalid"
    store.close()


def test_legacy_bearer_remains_owner_compatibility_only(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token="legacy-owner-token-123456")
    client = _client(service)
    response = client.get("/api/v1/metrics", headers={"Authorization": "Bearer legacy-owner-token-123456"})
    assert response.status_code == 200
    assert client.post("/api/v1/tasks", json={"goal": "legacy compatibility"}, headers={"Authorization": "Bearer legacy-owner-token-123456"}).status_code == 201
    store.close()


def test_known_bearer_placeholder_is_not_accepted_by_api(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token="replace-with-a-long-random-owner-token")
    response = _client(service).get(
        "/api/v1/metrics",
        headers={"Authorization": "Bearer replace-with-a-long-random-owner-token"},
    )
    assert response.status_code == 503
    assert response.json()["error_code"] == "api_auth_not_configured"
    store.close()


def test_required_owner_auth_disables_legacy_bearer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_REQUIRE_OWNER_AUTH", "true")
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token="legacy-owner-token-123456")
    client = _client(service)
    response = client.get("/api/v1/metrics", headers={"Authorization": "Bearer legacy-owner-token-123456"})
    assert response.status_code == 401
    assert response.json()["error_code"] == "legacy_bearer_disabled"
    store.close()


def test_initialized_owner_disables_legacy_bearer_even_without_env_flag(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token="legacy-owner-token-123456")
    AuthManager(store).initialize_owner(OWNER_EMAIL, OWNER_PASSWORD)
    response = _client(service).get("/api/v1/metrics", headers={"Authorization": "Bearer legacy-owner-token-123456"})
    assert response.status_code == 401
    assert response.json()["error_code"] == "legacy_bearer_disabled"
    store.close()


def test_cors_rejects_external_http_origin_for_cookie_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_ALLOWED_ORIGINS", "http://frontend.example")
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token=None)
    with pytest.raises(ValueError, match="cors_origin_insecure"):
        create_app(service=service)
    store.close()


def test_invalid_multiple_owner_state_fails_closed_and_never_falls_back_to_bearer(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path, APITestTool("safe"), token="legacy-owner-token-123456")
    stamp = "2026-01-01T00:00:00+00:00"
    password_hash = hash_password("temporary-audit-password-123")
    for index, email in enumerate(("owner-one@example.com", "owner-two@example.com")):
        store.db.execute(
            """INSERT INTO auth_users
               (user_id, email, role, password_hash, status, failed_login_count,
                locked_until, last_login_at, created_at, updated_at)
               VALUES (?, ?, 'OWNER', ?, 'ACTIVE', 0, '', '', ?, ?)""",
            (f"user:owner-{index}", email, password_hash, stamp, stamp),
        )
    store.db.commit()
    assert service.auth_manager.owner_status() == "invalid"
    assert service.owner_auth_enforced()
    ready, checks = service.readiness()
    assert not ready
    assert checks["owner_authentication"] == "invalid"
    client = _client(service)
    response = client.get("/api/v1/metrics", headers={"Authorization": "Bearer legacy-owner-token-123456"})
    assert response.status_code == 401
    assert response.json()["error_code"] == "legacy_bearer_disabled"
    store.close()
