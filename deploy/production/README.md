# Vulnora production deployment package

This directory is a deployment package, not an executed deployment. It targets a dedicated Ubuntu 24.04 host (x86_64 or ARM64) with Python 3.11+, a supported virtual environment, Nginx, and systemd. It does not install packages, create OCI resources, or enable scanner execution in the current environment.

## Intended layout

```text
/opt/vulnora/backend/       # private-self-learning-agent checkout or release
/opt/vulnora/backend/.venv/ # isolated Python environment with the backend installed
/opt/vulnora/frontend/      # Vulnora checkout with a built dist/ directory
/etc/vulnora/backend.env   # mode 0600, outside Git
/var/lib/vulnora/           # mode 0700, owned by vulnora:vulnora
```

The backend listens only on `127.0.0.1:8123`. Nginx terminates HTTPS and serves the frontend from the same origin, proxying `/api/`, `/health`, and `/ready` to the backend. Build the frontend with `VITE_AGENT_API_BASE_URL=/api/v1`; never place authentication credentials in a `VITE_*` variable.

## Administrator-controlled setup

The following is a runbook for an authorized administrator on the target host. Review the host's change, firewall, certificate, backup, and rollback procedures before applying it.

1. Create a dedicated `vulnora` service account and the directories shown above. The account must not be root. Give it ownership only of `/var/lib/vulnora` and the application release directories it must read.
2. Create a Python 3.11+ virtual environment under `/opt/vulnora/backend/.venv`, install the pinned backend dependencies from the repository's packaging metadata, and install the backend package. Do not copy `.env` files or credentials into Git.
3. Copy `backend.env.example` to `/etc/vulnora/backend.env`, preserve `AGENT_REQUIRE_OWNER_AUTH=1`, set mode `0600`, and keep `AGENT_ALLOWED_ORIGINS` empty for the same-origin Nginx setup unless a separately reviewed origin is required. Do **not** configure `AGENT_API_TOKEN` in this mode: legacy Bearer authentication is deliberately disabled.
4. Create `/var/lib/vulnora`, set mode `0700`, and set `AGENT_DB_PATH=/var/lib/vulnora/agent.sqlite3`. Do not store the live database in the checkout or an ephemeral directory.
5. Before starting the service, run the following command from a protected administrator terminal. It prompts twice for the new password; do not place that password in an environment variable, command history, file, or chat. The only accepted initial email is `ashbm1995@gmail.com`, and the operation refuses to run again once any account exists.

   ```bash
   cd /opt/vulnora/backend
   sudo -u vulnora -- /opt/vulnora/backend/.venv/bin/python -m private_agent.cli auth init-owner \
     --database /var/lib/vulnora/agent.sqlite3
   ```

   The command creates the single `OWNER` account only. There is no public registration endpoint and no HTTP endpoint for owner initialization. After HTTPS is available, sign in through Vulnora; the server issues an HttpOnly, Secure, SameSite=Lax session cookie and a separate CSRF cookie. All state-changing API calls require the matching `X-CSRF-Token` header.
6. Install `vulnora-backend.service` as `/etc/systemd/system/vulnora-backend.service`, review its hardening directives, then run `systemctl daemon-reload` and `systemctl enable --now vulnora-backend.service`. A missing or malformed environment file must prevent a successful startup.
7. Build the frontend in a controlled release directory with `VITE_AGENT_API_BASE_URL=/api/v1 pnpm build`. Install `nginx-vulnora.conf` after replacing the example hostname and certificate paths. Validate the Nginx configuration, reload it, and verify HTTPS certificate renewal separately.
8. Validate from the public HTTPS origin: `/health` is live, `/ready` is ready only while the worker is running, unauthenticated `/api/v1/*` is rejected, login works only for the initialized owner, a CSRF-less state change is rejected, and an authenticated task can be created and observed to a terminal state. Do not expose port 8123 directly.

## Owner authentication and roles

The initial `OWNER` can create only `OPERATOR` and `VIEWER` accounts through the authenticated API/Users control. It cannot create a second owner, reassign the initial owner, or disable it. `OWNER` can review all task records and approve or deny bound security actions. `OPERATOR` can create and resume only tasks it owns; `VIEWER` is read-only. Non-owner task requests are owner-scoped server-side, including task state, timeline, metrics, Bug Bounty evidence, and approvals; a cross-user task identifier receives the same `404` response as an unknown identifier.

Role checks in the frontend are present only for usability. The backend independently checks authenticated role, CSRF token, and task ownership. Login failures are recorded without passwords or session tokens; repeated failures lock the affected account for the configured interval. In addition, `AGENT_AUTH_MAX_LOGIN_ATTEMPTS` limits expensive password verification per remote-keyed time window and `AGENT_AUTH_MAX_CONCURRENT_KDFS` bounds concurrent scrypt work. Session tokens and CSRF tokens are stored only as hashes in SQLite.

## Backup and restore

The backup helper uses SQLite's online backup API, performs an integrity check, writes a `0600` temporary file in the destination directory, and atomically publishes it. A live database can be backed up without stopping the service:

```bash
cd /opt/vulnora/backend
umask 077
python3 scripts/sqlite_backup.py backup \
  --source /var/lib/vulnora/agent.sqlite3 \
  --destination /var/backups/vulnora/agent-$(date -u +%Y%m%dT%H%M%SZ).sqlite3
```

For restore, stop the service first so no process is using the destination, preserve the current database, restore to a temporary or planned path, run the application's readiness and task-history checks, then start the service:

```bash
sudo systemctl stop vulnora-backend.service
sudo cp --preserve=mode,ownership /var/lib/vulnora/agent.sqlite3 /var/backups/vulnora/pre-restore.sqlite3
sudo -u vulnora python3 scripts/sqlite_backup.py restore \
  --source /var/backups/vulnora/known-good.sqlite3 \
  --destination /var/lib/vulnora/agent.sqlite3
sudo systemctl start vulnora-backend.service
```

The helper refuses a missing or corrupt source and never publishes a destination that fails `PRAGMA integrity_check`. It cannot decide whether a task's external side effect occurred; after an interruption, the existing lease fencing and `EXECUTION_UNKNOWN` rules remain authoritative and must not be bypassed by restore.

Use at least two independent backup destinations and periodically test restore on a copy. Backups contain task history and may contain sensitive metadata; protect them like the database and never commit them.

## Rollback and restart

Keep the previous application release and its dependency lock state until the new release passes health, readiness, authenticated task lifecycle, approval, and restart checks. To roll back application code, stop the service, switch the release directory through the administrator's deployment mechanism, start the service, and verify the schema/readiness contract. Do not roll back by deleting the database or by resetting Git in place.

A process restart must not blindly rerun an uncertain task. The worker claims tasks with leases, fences stale writes, and preserves `EXECUTION_UNKNOWN` conservatively. An operator must inspect the task and use only the supported, bound resume path after the required evidence and approval checks.

## Scanner boundary

Phase 29 intentionally removed host execution from public `httpx` and Nuclei adapters. This deployment package does not enable them. `RuntimeActivationProof` remains unavailable until a dedicated OCI rootless runtime passes the preflight and scanner-specific acceptance suite described in `docs/PHASE28_PORTABLE_RUNTIME_KIT.md`. Podman or Docker presence alone is not sufficient, and no host fallback is allowed.

## Acceptance commands before handoff

Run from the backend checkout without external targets:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python3 -m compileall -q src tests scripts
python3 -m private_agent.cli runtime-preflight --json
```

The first two commands validate the application package. The preflight must be `PASS` on the target host before a separate, administrator-approved runtime acceptance process can be considered. It is expected to be blocked in the current Manus Sandbox.
