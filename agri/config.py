"""Versioned configuration for the offline-first proof of working."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "project": "AgriMini-v0",
    "seed": 42,
    "offline": True,
    "device": "cpu",
    "model": {
        "name": "AgriTransformer-v0",
        "vocab_size": 2048,
        "layers": 4,
        "hidden_size": 256,
        "attention_heads": 4,
        "feed_forward_size": 1024,
        "context_length": 256,
        "dropout": 0.1,
        "pretrained": False,
    },
    "training": {
        "dry_run": True,
        "allow_large_runs": False,
        "max_steps": 20,
        "epochs": 1,
        "batch_size": 2,
        "checkpoint_every_steps": 10,
        "learning_rate": 0.0003,
    },
}


def default_config() -> dict[str, Any]:
    return deepcopy(DEFAULT_CONFIG)


def validate_config(config: Any) -> dict[str, Any]:
    def check_keys(value: Any, expected: dict[str, Any], label: str) -> None:
        if not isinstance(value, dict) or set(value) != set(expected):
            raise ValueError(f"{label}: expected keys {sorted(expected)}")

    def integer(value: Any, label: str, minimum: int = 1) -> None:
        if type(value) is not int or value < minimum:
            raise ValueError(f"{label}: expected integer >= {minimum}")

    check_keys(config, DEFAULT_CONFIG, "config")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported configuration schema_version")
    if config["project"] != "AgriMini-v0":
        raise ValueError("project must be AgriMini-v0")
    integer(config["seed"], "seed", 0)
    if type(config["offline"]) is not bool:
        raise ValueError("offline must be a boolean")
    if config["device"] not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be cpu, cuda, or mps")

    model = config["model"]
    check_keys(model, DEFAULT_CONFIG["model"], "model")
    if model["name"] != "AgriTransformer-v0" or model["pretrained"] is not False:
        raise ValueError("Only randomly initialized AgriTransformer-v0 is supported")
    for key in ("vocab_size", "layers", "hidden_size", "attention_heads",
                "feed_forward_size", "context_length"):
        integer(model[key], f"model.{key}")
    if model["hidden_size"] % model["attention_heads"]:
        raise ValueError("hidden_size must be divisible by attention_heads")
    dropout = model["dropout"]
    if type(dropout) not in (float, int) or not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")

    training = config["training"]
    check_keys(training, DEFAULT_CONFIG["training"], "training")
    for key in ("dry_run", "allow_large_runs"):
        if type(training[key]) is not bool:
            raise ValueError(f"training.{key} must be a boolean")
    for key in ("max_steps", "epochs", "batch_size", "checkpoint_every_steps"):
        integer(training[key], f"training.{key}")
    rate = training["learning_rate"]
    if type(rate) not in (float, int) or not 0 < rate <= 1:
        raise ValueError("learning_rate must be in (0, 1]")
    if training["checkpoint_every_steps"] > training["max_steps"]:
        raise ValueError("checkpoint interval must not exceed max_steps")
    return deepcopy(config)


def load_config(path: Path) -> dict[str, Any]:
    return validate_config(json.loads(path.read_text(encoding="utf-8")))
