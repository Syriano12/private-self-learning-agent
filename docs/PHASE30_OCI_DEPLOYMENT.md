# Phase 30 — OCI deployment package and recovery contract

## Scope

Phase 30 adds a reviewable deployment package for the existing Vulnora frontend, Agent API, controlled worker, and SQLite persistence. It does not provision OCI, install system packages, create systemd services, expose a public listener, or activate `httpx`/Nuclei.

## Delivered contract

- `deploy/production/vulnora-backend.service` runs the real `private_agent.cli serve --with-worker` command on loopback, with a persistent database directory and conservative systemd hardening.
- `deploy/production/nginx-vulnora.conf` keeps the API private behind HTTPS same-origin routing and serves the built Vulnora SPA.
- `deploy/production/backend.env.example` documents runtime values without real credentials.
- `scripts/sqlite_backup.py` creates and restores consistent SQLite snapshots atomically, checking source and destination integrity before publication.
- `deploy/production/README.md` describes installation, readiness, backup, restore, rollback, and acceptance gates.

## Security boundaries

The unit is not a scanner sandbox. It runs the API and worker only. Phase 29's public scanner adapter quarantine remains in force; a configuration value, executable path, or caller-supplied flag cannot issue `RuntimeActivationProof` or cause host scanner execution. A target host must separately satisfy the Phase 28 preflight and scanner-specific OCI acceptance suite before any future runtime work.

HTTPS is terminated by the reverse proxy. The frontend uses `/api/v1` as its same-origin API base and never receives a password, session token, or `AGENT_API_TOKEN` through a build-time `VITE_*` value. The backend remains bound to `127.0.0.1`; Nginx preserves the HttpOnly session cookie and forwards `X-CSRF-Token` for state-changing API requests. Legacy Authorization forwarding is not the owner-auth path and is disabled after owner initialization.

## Recovery contract

SQLite backup uses the online backup API, a temporary file in the destination directory, `PRAGMA integrity_check`, restrictive permissions, and atomic `os.replace`. Restore must be performed with the service stopped, after preserving the current database. Database restoration cannot prove whether an external side effect occurred, so it does not authorize retrying `EXECUTION_UNKNOWN` tasks.

## What remains unverified

This package has not been installed on OCI. The current Sandbox does not prove rootless OCI isolation, cgroup enforcement, seccomp behavior, network denial, process cleanup, image provenance, or scanner-specific acceptance. It is therefore a deployment package, not a production-readiness claim.
