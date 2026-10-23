from __future__ import annotations

import argparse
import os
import sqlite3
import tempfile
from pathlib import Path


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label}_symlink_not_allowed:{path}")
    if not path.exists():
        raise FileNotFoundError(f"{label}_missing:{path}")
    if not path.is_file():
        raise ValueError(f"{label}_not_regular_file:{path}")


def _integrity_check(connection: sqlite3.Connection) -> None:
    row = connection.execute("PRAGMA integrity_check").fetchone()
    if row != ("ok",):
        detail = "unknown" if not row else str(row[0])
        raise sqlite3.DatabaseError(f"sqlite_integrity_check_failed:{detail}")


def _snapshot(source: Path, destination: Path) -> Path:
    source = source.expanduser()
    destination = destination.expanduser()
    _require_regular_file(source, "sqlite_source")
    if destination.is_symlink():
        raise ValueError(f"sqlite_destination_symlink_not_allowed:{destination}")
    if source.resolve() == destination.resolve():
        raise ValueError("sqlite_source_destination_must_differ")
    destination.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        source_uri = f"file:{source.resolve()}?mode=ro"
        with sqlite3.connect(source_uri, uri=True) as source_db:
            _integrity_check(source_db)
            with sqlite3.connect(temporary) as destination_db:
                source_db.backup(destination_db)
                _integrity_check(destination_db)
                destination_db.commit()
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
        return destination
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def backup_database(source: Path, destination: Path) -> Path:
    """Create a consistent SQLite snapshot and publish it atomically."""
    return _snapshot(Path(source), Path(destination))


def restore_database(backup: Path, destination: Path) -> Path:
    """Restore a validated SQLite snapshot atomically.

    The caller must stop the service before replacing its active database.
    """
    return _snapshot(Path(backup), Path(destination))


def main() -> int:
    parser = argparse.ArgumentParser(description="Atomic SQLite backup/restore for Vulnora")
    subcommands = parser.add_subparsers(dest="operation", required=True)
    for operation in ("backup", "restore"):
        command = subcommands.add_parser(operation)
        command.add_argument("--source", required=True, type=Path)
        command.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    if args.operation == "backup":
        backup_database(args.source, args.destination)
    else:
        restore_database(args.source, args.destination)
    print(args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
