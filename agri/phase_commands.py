"""Lazy command registration for optional later-phase components."""

import argparse
from importlib import import_module
from pathlib import Path
from typing import Any


COMMANDS: dict[str, tuple[int, str, tuple[str, ...]]] = {
    "training-plan": (6, "training.trainer.training_plan", ()),
    "train-model": (6, "training.trainer.train_model", ("dry_run",)),
    "validate-training": (6, "training.trainer.validate_training", ()),
    "generate": (7, "models.transformer.inference.generate", ("prompt", "max_new_tokens")),
    "build-index": (8, "retrieval.index.build_index", ()),
    "validate-index": (8, "retrieval.index.validate_index", ()),
    "search": (8, "retrieval.index.search", ("question", "limit", "split")),
    "answer": (9, "retrieval.qa.answer_query", ("question", "max_new_tokens")),
    "inspect-adapter": (10, "sources.ingestion.inspect_adapter", ("source_id",)),
    "adapter-schema": (10, "agri.phase_commands.show_adapter_schema", ("source_id",)),
    "ingest-snapshot": (10, "sources.ingestion.ingest_snapshot", ("source_id", "snapshot", "purpose")),
    "vision-plan": (14, "models.vision_baseline.workflow.vision_plan", ()),
    "check-vision": (14, "models.vision_baseline.workflow.check_vision", ("dry_run",)),
    "validate-vision": (14, "models.vision_baseline.workflow.validate_vision", ()),
    "serve": (15, "app.api.server.serve", ("port",)),
}

ADAPTER_PHASES = {"plantvillage": 10, "plantdoc": 10, "soilgrids": 11, "isric": 11,
                  "data_gov": 12, "ssurgo": 13}

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


def command_phase(args: argparse.Namespace) -> int:
    if args.command in {"inspect-adapter", "adapter-schema", "ingest-snapshot"}:
        return ADAPTER_PHASES.get(args.source_id, 10)
    return COMMANDS[args.command][0]


def show_adapter_schema(root: Path, source_id: str) -> dict[str, Any]:
    from sources.ingestion import adapter_schema
    schema = adapter_schema(source_id)
    return {"phase": schema["phase"], "status": "GREEN", "schema": schema,
            "scope": "Format documentation only; no ingestion approval granted"}


def execute(args: argparse.Namespace) -> dict[str, Any] | None:
    _, target, options = COMMANDS[args.command]
    module, function = target.rsplit(".", 1)
    handler = getattr(import_module(module), function)
    return handler(args.root, **{name: getattr(args, name) for name in options})
