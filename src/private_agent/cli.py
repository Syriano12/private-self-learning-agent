from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(prog="private-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a general goal")
    run.add_argument("goal")
    serve = sub.add_parser("serve", help="start FastAPI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        uvicorn.run("private_agent.app:app", host=args.host, port=args.port)
        return
    from private_agent.app import build_orchestrator
    result = build_orchestrator().run(args.goal)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
