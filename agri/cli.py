"""Repository and stored source-inspection commands; no live ingestion."""

import argparse
import json
from pathlib import Path

from agri.source_registry import inspect_sources
from agri.synthetic_data import generate_synthetic, validate_data
from agri.workspace import PROJECT_ROOT, setup, status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AgriMini v0 — Phases 1–3")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("setup", "status", "inspect-sources", "generate-synthetic", "validate-data"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, default=PROJECT_ROOT)
        if name == "inspect-sources":
            command.add_argument("--source", help="Inspect one stored source ID")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect-sources":
            report = inspect_sources(args.root, args.source)
        elif args.command == "generate-synthetic":
            report = generate_synthetic(args.root)
        elif args.command == "validate-data":
            report = validate_data(args.root)
        else:
            report = setup(args.root) if args.command == "setup" else status(args.root)
    except (OSError, ValueError) as exc:
        phase = 2 if args.command in ("setup", "inspect-sources") else 3
        print(json.dumps({"phase": phase, "status": "RED", "error": str(exc)}))
        return 1
    print(json.dumps(report, indent=2))
    return 1 if report["status"] == "RED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
