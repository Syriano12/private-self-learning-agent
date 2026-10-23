from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="private-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a general goal")
    run.add_argument("goal")
    serve = sub.add_parser("serve", help="start FastAPI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--with-worker",
        action="store_true",
        help="start the durable background worker with the API lifecycle",
    )
    serve.add_argument("--worker-id", default=None)
    bounty = sub.add_parser("bug-bounty", help="run bounded authorized Bug Bounty research")
    bounty.add_argument("--scope", required=True, help="JSON scope definition path")
    bounty.add_argument(
        "--fixture",
        choices=("secure", "idor_horizontal", "idor_vertical", "idor_nested", "false_positive", "inconclusive"),
        default=None,
        help="deterministic in-memory local fixture; no external target is contacted",
    )
    bounty.add_argument("--source-file", action="append", default=[], help="authorized local source artifact")
    bounty.add_argument("--openapi", help="authorized local OpenAPI JSON artifact")
    bounty.add_argument("--url", action="append", default=[], help="supplied URL for passive parsing only")
    bounty.add_argument("--database", default="data/agent.sqlite3", help="local SQLite path for bounded run metadata")
    bounty.add_argument("--report", help="write a Markdown report to this local path")
    bounty.add_argument("--no-persist", action="store_true", help="do not persist run metadata")
    preflight = sub.add_parser("runtime-preflight", help="read-only host checks for the portable scanner runtime kit")
    preflight.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    auth = sub.add_parser("auth", help="manage local owner authentication")
    auth_sub = auth.add_subparsers(dest="auth_command", required=True)
    init_owner = auth_sub.add_parser("init-owner", help="create the configured owner once using interactive password input")
    init_owner.add_argument("--database", default=os.getenv("AGENT_DB_PATH", "data/agent.sqlite3"))
    init_owner.add_argument("--email", default=os.getenv("AGENT_OWNER_EMAIL", "ashbm1995@gmail.com"))
    args = parser.parse_args()
    if args.command == "runtime-preflight":
        from private_agent.skills.runtime_preflight import HostPreflightStatus, run_host_preflight

        report = run_host_preflight()
        payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
        print(payload)
        if report.verdict is HostPreflightStatus.FAIL:
            raise SystemExit(1)
        if report.verdict is HostPreflightStatus.INCONCLUSIVE:
            raise SystemExit(2)
        return
    if args.command == "auth":
        if args.auth_command != "init-owner":
            raise SystemExit("unsupported_auth_command")
        from private_agent.auth import AuthError, AuthManager
        from private_agent.storage import Store

        first = getpass.getpass("New Vulnora owner password: ")
        confirmation = getpass.getpass("Confirm Vulnora owner password: ")
        if first != confirmation:
            raise SystemExit("password_confirmation_mismatch")
        store = Store(args.database)
        try:
            try:
                user = AuthManager(store).initialize_owner(args.email, first)
            except AuthError as exc:
                raise SystemExit(exc.code) from exc
        finally:
            store.close()
        print(json.dumps({"status": "initialized", "user_id": user.user_id, "email": user.email, "role": user.role}))
        return
    if args.command == "serve":
        import uvicorn

        if not args.with_worker:
            from private_agent.api import create_app

            uvicorn.run(create_app(require_worker=True), host=args.host, port=args.port)
            return

        from private_agent.api import build_api_service, create_app
        from private_agent.worker import BackgroundWorker

        service = build_api_service()
        worker = BackgroundWorker(
            service.orchestrator,
            worker_id=args.worker_id or os.getenv("AGENT_WORKER_ID"),
            lease_ttl_seconds=int(os.getenv("AGENT_WORKER_LEASE_TTL", "60")),
            heartbeat_interval_seconds=float(os.getenv("AGENT_WORKER_HEARTBEAT", ""))
            if os.getenv("AGENT_WORKER_HEARTBEAT")
            else None,
            max_execution_age_seconds=float(os.getenv("AGENT_WORKER_MAX_AGE", "300")),
            poll_interval_seconds=float(os.getenv("AGENT_WORKER_POLL_INTERVAL", "0.1")),
            idle_poll_interval_seconds=float(os.getenv("AGENT_WORKER_IDLE_INTERVAL", "1.0")),
            error_backoff_seconds=float(os.getenv("AGENT_WORKER_ERROR_BACKOFF", "1.0")),
        )
        uvicorn.run(create_app(service=service, worker=worker), host=args.host, port=args.port)
        return
    if args.command == "bug-bounty":
        from private_agent.bugbounty import BugBountyAgent, markdown_report
        from private_agent.storage import Store

        scope_path = Path(args.scope)
        scope_payload = json.loads(scope_path.read_text(encoding="utf-8"))
        source_documents: dict[str, str] = {}
        for source_file in args.source_file:
            path = Path(source_file)
            if path.stat().st_size > 512_000:
                raise SystemExit(f"source_file_too_large:{path}")
            source_documents[str(path)] = path.read_text(encoding="utf-8")
        openapi = None
        if args.openapi:
            path = Path(args.openapi)
            if path.stat().st_size > 512_000:
                raise SystemExit(f"openapi_too_large:{path}")
            openapi = json.loads(path.read_text(encoding="utf-8"))
        store = None if args.no_persist else Store(args.database)
        run_result = BugBountyAgent(store=store).run(
            scope_payload,
            fixture_variant=args.fixture,
            urls=tuple(args.url),
            openapi=openapi,
            source_documents=source_documents,
            mode="deterministic",
            persist=not args.no_persist,
        )
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(markdown_report(run_result), encoding="utf-8")
        print(json.dumps(run_result.to_dict(), ensure_ascii=False, indent=2))
        return
    from private_agent.app import build_orchestrator
    result = build_orchestrator().run(args.goal)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
