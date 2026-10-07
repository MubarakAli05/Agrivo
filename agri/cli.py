"""Offline workspace, tokenizer, and model-check commands; no live ingestion."""

import argparse
import json
from pathlib import Path

from agri.model_check import check_model, model_report_path, validate_model
from agri.source_registry import inspect_sources
from agri.synthetic_data import generate_synthetic, validate_data
from agri.workspace import PROJECT_ROOT, setup, status
from tokenizer.train_tokenizer import train_tokenizer, validate_tokenizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AgriMini v0 — Phases 1–5")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("setup", "status", "inspect-sources", "generate-synthetic", "validate-data",
                 "train-tokenizer", "validate-tokenizer", "check-model", "validate-model"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, default=PROJECT_ROOT)
        if name == "inspect-sources":
            command.add_argument("--source", help="Inspect one stored source ID")
        if name in ("train-tokenizer", "check-model"):
            command.add_argument("--dry-run", action="store_true", help="Run a tiny preflight without saving artifacts")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect-sources":
            report = inspect_sources(args.root, args.source)
        elif args.command == "generate-synthetic":
            report = generate_synthetic(args.root)
        elif args.command == "validate-data":
            report = validate_data(args.root)
        elif args.command == "train-tokenizer":
            report = train_tokenizer(args.root, args.dry_run)
        elif args.command == "validate-tokenizer":
            report = validate_tokenizer(args.root)
        elif args.command == "check-model":
            report = check_model(args.root, args.dry_run)
        elif args.command == "validate-model":
            report = validate_model(args.root)
        else:
            report = setup(args.root) if args.command == "setup" else status(args.root)
    except (OSError, ValueError) as exc:
        phase = 4 if args.command in ("train-tokenizer", "validate-tokenizer", "status") else (
            2 if args.command in ("setup", "inspect-sources") else 3)
        if args.command in ("check-model", "validate-model") or (
                args.command in ("setup", "status") and model_report_path(args.root).exists()):
            phase = 5
        print(json.dumps({"phase": phase, "status": "RED", "error": str(exc)}))
        return 1
    print(json.dumps(report, indent=2))
    return 1 if report["status"] == "RED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
