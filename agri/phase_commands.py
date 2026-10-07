"""Lazy command registration for optional later-phase components."""

import argparse
from importlib import import_module
from pathlib import Path
from typing import Any


COMMANDS: dict[str, tuple[int, str, tuple[str, ...]]] = {
    "build-index": (8, "retrieval.index.build_index", ()),
    "validate-index": (8, "retrieval.index.validate_index", ()),
    "search": (8, "retrieval.index.search", ("question", "limit", "split")),
}

OPTIONS: dict[str, dict[str, Any]] = {
    "dry_run": {"action": "store_true", "help": "Run a bounded preflight without publishing artifacts"},
    "prompt": {"required": True, "help": "Text prompt for the synthetic-only model"},
    "question": {"required": True, "help": "Question to retrieve or answer with explicit evidence"},
    "max_new_tokens": {"type": int, "default": 32},
    "limit": {"type": int, "default": 3},
    "split": {"choices": ("train", "validation", "test"), "default": "train"},
    "source_id": {"required": True, "help": "Registered source ID"},
    "snapshot": {"type": Path, "required": True},
    "purpose": {"choices": ("research_training", "commercial_training", "redistribution"),
                "default": "research_training"},
    "port": {"type": int, "default": 8765},
}


def register_commands(commands: Any, root: Path) -> None:
    for name, (_, _, options) in COMMANDS.items():
        parser = commands.add_parser(name)
        parser.add_argument("--root", type=Path, default=root)
        for option in options:
            flag = "--source" if option == "source_id" else "--" + option.replace("_", "-")
            parser.add_argument(flag, dest=option, **OPTIONS[option])


def execute(args: argparse.Namespace) -> dict[str, Any] | None:
    _, target, options = COMMANDS[args.command]
    module, function = target.rsplit(".", 1)
    handler = getattr(import_module(module), function)
    return handler(args.root, **{name: getattr(args, name) for name in options})
