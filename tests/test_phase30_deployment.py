from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from private_agent.api import APIError, TokenAuthenticator
from private_agent.storage import Store
from scripts.sqlite_backup import backup_database, restore_database


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = BACKEND_ROOT / "deploy" / "production"


def test_production_service_contract_uses_real_loopback_worker_command() -> None:
    unit = (PRODUCTION_ROOT / "vulnora-backend.service").read_text(encoding="utf-8")

    assert "private_agent.cli serve --with-worker" in unit
    assert "--host 127.0.0.1" in unit
    assert "EnvironmentFile=/etc/vulnora/backend.env" in unit
    assert "Restart=on-failure" in unit
    assert "ReadWritePaths=/var/lib/vulnora" in unit


def test_production_runbook_bootstraps_owner_as_service_user() -> None:
    runbook = (PRODUCTION_ROOT / "README.md").read_text(encoding="utf-8")
    assert "sudo -u vulnora -- /opt/vulnora/backend/.venv/bin/python -m private_agent.cli auth init-owner" in runbook
    assert "--database /var/lib/vulnora/agent.sqlite3" in runbook


def test_reverse_proxy_contract_keeps_api_same_origin_and_backend_private() -> None:
    proxy = (PRODUCTION_ROOT / "nginx-vulnora.conf").read_text(encoding="utf-8")

    assert "location /api/" in proxy
    assert "proxy_pass http://127.0.0.1:8123" in proxy
    assert "try_files $uri $uri/ /index.html" in proxy
    assert "listen 443 ssl" in proxy
    assert "server_name vulnora.example.invalid" in proxy
    assert "return 308 https://vulnora.example.invalid$request_uri" in proxy
    assert "if ($host != vulnora.example.invalid) { return 444; }" in proxy
    assert "Content-Security-Policy" in proxy
    assert "frame-ancestors 'none'" in proxy
    assert "X-CSRF-Token" in proxy


def test_production_environment_contract_uses_persistent_sqlite_and_no_secrets() -> None:
    env = (PRODUCTION_ROOT / "backend.env.example").read_text(encoding="utf-8")

    assert "AGENT_DB_PATH=/var/lib/vulnora/agent.sqlite3" in env
    assert "AGENT_REQUIRE_OWNER_AUTH=1" in env
    assert "AGENT_OWNER_EMAIL=ashbm1995@gmail.com" in env
    assert "AGENT_COOKIE_SECURE=1" in env
    assert "AGENT_API_TOKEN=" not in env
    assert "GEMINI_API_KEY=CHANGE_ME" not in env
    assert "# GEMINI_API_KEY=" in env
    assert "VITE_" not in env
    assert "BEGIN " not in env


def test_production_auth_rejects_placeholder_and_weak_tokens() -> None:
    for token in ("CHANGE_ME_USE_A_LONG_RANDOM_OWNER_TOKEN", "replace-with-a-long-random-owner-token", "short"):
        authenticator = TokenAuthenticator(token=token)
        assert authenticator.configured is False
        with pytest.raises(APIError) as error:
            authenticator.authenticate(f"Bearer {token}")
        assert error.value.error_code == "api_auth_not_configured"

    assert TokenAuthenticator(token="phase30-owner-token-7d9e3b").configured is True


def test_sqlite_backup_restore_preserves_real_agent_task_state(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite3"
    backup = tmp_path / "backup.sqlite3"
    restored = tmp_path / "restored.sqlite3"

    store = Store(source)
    created = store.create_queued_task("backup restore acceptance task")
    task_id = str(created["task_id"])
    store.close()

    backup_database(source, backup)
    restore_database(backup, restored)

    reopened = Store(restored)
    state = reopened.get_task_state(task_id)
    reopened.close()

    assert state is not None
    assert state["task_id"] == task_id
    assert state["status"] == "QUEUED"


def test_sqlite_backup_rejects_missing_or_corrupt_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        backup_database(tmp_path / "missing.sqlite3", tmp_path / "backup.sqlite3")

    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_bytes(b"not sqlite")
    with pytest.raises(sqlite3.DatabaseError):
        backup_database(corrupt, tmp_path / "backup-corrupt.sqlite3")


def test_restore_uses_integrity_check_before_publishing_destination(tmp_path: Path) -> None:
    invalid_backup = tmp_path / "invalid.sqlite3"
    invalid_backup.write_bytes(b"not sqlite")
    destination = tmp_path / "restored.sqlite3"

    with pytest.raises(sqlite3.DatabaseError):
        restore_database(invalid_backup, destination)

    assert not destination.exists()


def test_sqlite_backup_rejects_symlink_source(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite3"
    source.write_bytes(b"not sqlite")
    symlink = tmp_path / "source-link.sqlite3"
    symlink.symlink_to(source)

    with pytest.raises(ValueError, match="symlink"):
        backup_database(symlink, tmp_path / "backup.sqlite3")


def test_sqlite_backup_rejects_symlink_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite3"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE t (value INTEGER)")
        connection.commit()
    destination = tmp_path / "destination-link.sqlite3"
    destination.symlink_to(tmp_path / "other.sqlite3")

    with pytest.raises(ValueError, match="symlink"):
        backup_database(source, destination)

    assert destination.is_symlink()
